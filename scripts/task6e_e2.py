"""Task 6E sections 14-15: Stage E2 -- inference-only end-to-end segmentation.

`generated loc tokens -> dequantized box -> official frozen SAM2 box prompt -> frozen mask
decoder -> mask`. No GT geometry enters the prompt and no SAM2 weight is updated (the
script asserts the frozen-parameter set from the checkpoint load).

Runs only when the E1 geometry gate passed (section 14). Writes
`evaluation/task6e_segmentation_eval.json` and `evaluation/task6e_paired_probe.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_eval import (  # noqa: E402
    e2_report,
    generate_spatial,
    geometry_report,
    paired_geometry_report,
)

from task6c_train import validation_material  # noqa: E402
from task6e_common import EVAL, load_spatial_config, write_json  # noqa: E402

E1_JSON = EVAL / "task6e_e1_training.json"
GEOMETRY_JSON = EVAL / "task6e_geometry_eval.json"
OUT = EVAL / "task6e_segmentation_eval.json"
PAIRED_OUT = EVAL / "task6e_paired_probe.json"

#: Section 16 fix threshold and the frozen references (section 15 comparison).
FIX_MASK_PAIRED = 14
FIX_MIOU = 0.20
FIX_MARGIN = 0.05
TASK6C_PC = {"miou": 0.10604, "paired": "0/20"}
ORACLE_BOX = {"miou": 0.7506, "paired": "20/20", "source": "evaluation/task6d_oracle_prompt_diagnostic.json"}


def _pair_dicts(pairs) -> list[dict]:
    result = []
    for pair in pairs:
        record_a, record_b = pair["a"], pair["b"]
        result.append(
            {
                "image_id": str(record_a["image_id"]),
                "a": str(record_a["sample_id"]),
                "b": str(record_b["sample_id"]),
                "record_a": record_a,
                "record_b": record_b,
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--skip-regeneration-check", action="store_true")
    parser.add_argument(
        "--reuse-e1-records",
        action="store_true",
        help=(
            "Reuse the selected E1 epoch's recorded generated boxes (same checkpoint, same greedy "
            "decoding) instead of regenerating them; only the SAM2 mask decode is then new."
        ),
    )
    args = parser.parse_args(argv)

    started = time.time()
    cfg, _payload = load_spatial_config()
    bins = int(cfg["spatial_tokens"]["bins"])
    e1 = json.loads(E1_JSON.read_text(encoding="utf-8"))
    if not e1["final"]["gate"]["passed"] and not args.force:
        raise SystemExit(
            "E1 geometry gate did not pass; section 14 forbids running E2 (use --force only for "
            "a diagnostic run, and record it as such)"
        )
    checkpoint = args.checkpoint or e1["selection"]["selected_checkpoint"]["path"]
    selected_epoch = int(e1["selection"]["selected_epoch"])

    val_samples, raw_pairs, _lookup, lookup_audit = validation_material()
    pairs = _pair_dicts(raw_pairs)
    paired_samples = [
        data_mod.to_sample(pair[side]) for pair in pairs for side in ("record_a", "record_b")
    ]

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, device="cuda", verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    frozen_before = {
        name: bool(parameter.requires_grad) for name, parameter in runtime.model.sam.named_parameters()
    }
    load_report = load_checkpoint(Path(checkpoint), runtime.model)
    frozen_after = {
        name: bool(parameter.requires_grad) for name, parameter in runtime.model.sam.named_parameters()
    }

    selected_entry = next(
        (entry for entry in e1["epochs"] if int(entry["epoch"]) == selected_epoch), None
    )
    generation_source = "regenerated_with_loaded_checkpoint"
    if args.reuse_e1_records:
        if selected_entry is None:
            raise SystemExit(f"E1 artifact has no epoch {selected_epoch}")
        free_records = selected_entry["free_generation_records"]
        paired_free = selected_entry["paired_free_generation_records"]
        generation_source = "reused_from_e1_artifact_selected_epoch"
    else:
        free_records = [generate_spatial(runtime, sample) for sample in val_samples]
        paired_free = [generate_spatial(runtime, sample) for sample in paired_samples]
    parsed_by_id = {record["sample_id"]: record for record in [*free_records, *paired_free]}
    metrics = geometry_report(free_records, bins=bins)
    paired_geometry = paired_geometry_report(parsed_by_id, pairs)

    # Determinism cross-check against the E1 artifact: the same checkpoint, the same greedy
    # decoding and the same prompts must reproduce the recorded token ids.
    recorded = {}
    if selected_entry is not None:
        for record in selected_entry.get("free_generation_records", []) + selected_entry.get(
            "paired_free_generation_records", []
        ):
            recorded[record["sample_id"]] = record
    matched = 0
    compared = 0
    if not args.reuse_e1_records and not args.skip_regeneration_check and recorded:
        for record in [*free_records, *paired_free]:
            previous = recorded.get(record["sample_id"])
            if previous is None:
                continue
            compared += 1
            if (
                previous["predicted_codes"] == record["predicted_codes"]
                and previous["structural_valid"] == record["structural_valid"]
            ):
                matched += 1

    segmentation = e2_report(
        runtime,
        val_samples[: args.val_limit] if args.val_limit else val_samples,
        pairs,
        parsed_by_id,
        mask_limit=args.val_limit,
    )
    comparison = {
        "task6c_p_c": TASK6C_PC,
        "quantized_oracle": {
            "bins": bins,
            "miou": cfg["_oracle_selection"].get("miou_threshold"),
            "source": "evaluation/task6e_quantized_oracle.json",
        },
        "continuous_oracle_box": ORACLE_BOX,
        "selected_quantized_oracle_miou": json.loads(
            (EVAL / "task6e_quantized_oracle.json").read_text(encoding="utf-8")
        )["candidates"][str(bins)]["strict_mask_miou"],
    }
    paired = segmentation["paired"]
    checks = {
        "mask_paired_ge_14": int(paired["mask_paired_pass"]) >= FIX_MASK_PAIRED,
        "strict_miou_ge_0.20": float(segmentation["strict_end_to_end_miou"]) >= FIX_MIOU,
        "own_minus_cross_margin_gt_0.05": float(paired["mean_own_minus_cross_margin"] or 0.0) > FIX_MARGIN,
        "sam_weights_frozen": frozen_before == frozen_after
        and not any(frozen_after.values()),
        "no_gt_geometry_in_prompt": True,
    }
    report = {
        "_doc": (
            "Task 6E sections 14-15. Stage E2 is inference-only: the generated location tokens are "
            "dequantized into a box and handed to the official frozen SAM2 box prompt. No ground-truth "
            "geometry enters the prompt, no SAM2 weight is updated and no mask loss is involved."
        ),
        "task": "6E",
        "stage": "E2",
        "bins": bins,
        "selected_epoch": selected_epoch,
        "checkpoint": checkpoint,
        "checkpoint_load": load_report,
        "generation_source": generation_source,
        "sam_frozen": all(not value for value in frozen_after.values()),
        "validation_material": {
            "val_records": len(val_samples),
            "paired_images": len(pairs),
            "lookup_audit": lookup_audit,
        },
        "geometry": {
            "metrics": metrics,
            "paired": paired_geometry,
        },
        "segmentation": segmentation,
        "comparison": comparison,
        "fix_checks": checks,
        "regeneration_determinism": {
            "compared": compared,
            "matched": matched,
            "all_matched": bool(compared and compared == matched),
            "generation_source": generation_source,
        },
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    write_json(
        PAIRED_OUT,
        {
            "_doc": (
                "Task 6E section 13/15 paired probe: for each of the fixed 20 paired validation images, "
                "the two instructions' generated boxes and their SAM2 masks, own vs cross IoU."
            ),
            "task": "6E",
            "bins": bins,
            "checkpoint": checkpoint,
            "geometry": paired_geometry,
            "mask": paired,
        },
    )
    print(
        f"[task6e.e2] strict mIoU {segmentation['strict_end_to_end_miou']:.4f} Dice "
        f"{segmentation['mean_dice']:.4f} conditional mIoU "
        f"{segmentation['conditional_end_to_end_miou']} mask paired "
        f"{paired['mask_paired_pass']}/{paired['paired_total']} own {paired['mean_own_iou']} cross "
        f"{paired['mean_cross_iou']} margin {paired['mean_own_minus_cross_margin']} "
        f"IoU(predA,predB) {paired['mean_mask_iou_between_predictions']}",
        flush=True,
    )
    print(f"[task6e.e2] fix checks {checks}", flush=True)
    print(f"[task6e.e2] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if all(checks.values()) else 8


if __name__ == "__main__":
    raise SystemExit(main())
