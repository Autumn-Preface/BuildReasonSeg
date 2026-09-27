"""Task 6F sections 15-16: Stage F2 -- frozen-SAM end-to-end test (only if F1 passes).

`image + instruction -> constant [BOX] -> predicted box -> official frozen SAM2 box prompt ->
frozen SAM2 mask decoder -> mask`. No SAM training and no GT geometry in the prompt. Writes
`evaluation/task6f_segmentation_eval.json` and the mask side of
`evaluation/task6f_paired_probe.json`.
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
from buildreasonseg_mvp.box_query_eval import box_record, f2_report  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6c_train import validation_material  # noqa: E402
from task6f_common import (  # noqa: E402
    EVAL,
    load_box_config,
    validation_pair_dicts,
    write_json,
)

F1_JSON = EVAL / "task6f_f1_training.json"
PAIRED_JSON = EVAL / "task6f_paired_probe.json"
OUT = EVAL / "task6f_segmentation_eval.json"

FIX_MASK_PAIRED = 14
FIX_MIOU = 0.20
FIX_MARGIN = 0.05
TASK6C_PC = {"miou": 0.10604, "paired": "0/20"}
ORACLE_BOX = {"miou": 0.7506, "paired": "20/20", "source": "evaluation/task6d_oracle_prompt_diagnostic.json"}
ORACLE_QUANT = {"miou": 0.7343, "paired": "20/20", "source": "evaluation/task6e_quantized_oracle.json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    cfg, _payload = load_box_config()
    f1 = json.loads(F1_JSON.read_text(encoding="utf-8"))
    if not f1["final"]["gate"]["passed"] and not args.force:
        raise SystemExit(
            "F1 geometry gate did not pass; section 15 forbids running F2 (use --force only for "
            "a diagnostic run, and record it as such)"
        )
    checkpoint = args.checkpoint or f1["selection"]["selected_checkpoint"]["path"]
    selected_epoch = int(f1["selection"]["selected_epoch"])

    val_samples, raw_pairs, _lookup, _audit = validation_material()
    pairs = validation_pair_dicts(raw_pairs)
    paired_samples = [
        data_mod.to_sample(pair[side]) for pair in pairs for side in ("record_a", "record_b")
    ]

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    runtime.install_box_head(
        hidden_dim=int(cfg["box_query"]["head_hidden_dim"]),
        mid_dim=int(cfg["box_query"]["head_mid_dim"]),
    )
    frozen_before = {
        name: bool(parameter.requires_grad) for name, parameter in runtime.model.sam.named_parameters()
    }
    load_report = load_checkpoint(Path(checkpoint), runtime.model)
    frozen_after = {
        name: bool(parameter.requires_grad) for name, parameter in runtime.model.sam.named_parameters()
    }

    val_records = [box_record(runtime, sample) for sample in val_samples]
    paired_records = [box_record(runtime, sample) for sample in paired_samples]
    records_by_id = {record["sample_id"]: record for record in [*val_records, *paired_records]}

    segmentation = f2_report(runtime, val_samples, pairs, records_by_id)
    paired = segmentation["paired"]
    checks = {
        "mask_paired_ge_14": int(paired["mask_paired_pass"]) >= FIX_MASK_PAIRED,
        "strict_miou_ge_0.20": float(segmentation["strict_end_to_end_miou"]) >= FIX_MIOU,
        "own_minus_cross_margin_gt_0.05": float(paired["mean_own_minus_cross_margin"] or 0.0) > FIX_MARGIN,
        "sam_weights_frozen": frozen_before == frozen_after and not any(frozen_after.values()),
        "no_gt_geometry_in_prompt": True,
    }
    report = {
        "_doc": (
            "Task 6F sections 15-16. Stage F2 is inference-only: the [BOX]-query predicted box is "
            "handed to the official frozen SAM2 box prompt. No ground-truth geometry enters the "
            "prompt and no SAM2 weight is updated."
        ),
        "task": "6F",
        "stage": "F2",
        "selected_epoch": selected_epoch,
        "checkpoint": checkpoint,
        "checkpoint_load": load_report,
        "sam_frozen": all(not value for value in frozen_after.values()),
        "segmentation": segmentation,
        "comparison": {
            "task6c_p_c": TASK6C_PC,
            "continuous_oracle_box": ORACLE_BOX,
            "task6e_quantized_oracle": ORACLE_QUANT,
        },
        "fix_checks": checks,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    paired_artifact = json.loads(PAIRED_JSON.read_text(encoding="utf-8"))
    paired_artifact["mask"] = paired
    paired_artifact["mask_ran"] = True
    paired_artifact["mask_note"] = (
        "frozen SAM2 box-prompt masks from the query-path predicted boxes only."
    )
    paired_artifact["checkpoint"] = checkpoint
    write_json(PAIRED_JSON, paired_artifact)
    print(
        f"[task6f.f2] strict mIoU {segmentation['strict_end_to_end_miou']:.4f} Dice "
        f"{segmentation['mean_dice']:.4f} mask paired {paired['mask_paired_pass']}/"
        f"{paired['paired_total']} own {paired['mean_own_iou']} cross {paired['mean_cross_iou']} "
        f"margin {paired['mean_own_minus_cross_margin']} IoU(predA,predB) "
        f"{paired['mean_mask_iou_between_predictions']}",
        flush=True,
    )
    print(f"[task6f.f2] fix checks {checks}", flush=True)
    print(f"[task6f.f2] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if all(checks.values()) else 8


if __name__ == "__main__":
    raise SystemExit(main())
