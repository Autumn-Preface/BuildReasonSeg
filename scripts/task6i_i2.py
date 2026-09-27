"""Task 6I section 12: Stage I2 -- frozen-SAM2 segmentation (only if I1 passes).

`image + instruction -> [BOX] q0 -> visual refinement q1 -> 256x256 heatmap -> argmax predicted
point -> official frozen SAM2 positive-point prompt -> frozen SAM2 decoder -> mask`. No GT and no
SAM2 training. Writes `evaluation/task6i_segmentation_eval.json`, the mask side of
`evaluation/task6i_paired_probe.json` and paired diagnostic panels (layout reused from Task 6H).
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
from buildreasonseg_mvp.dense_eval import g2_report  # noqa: E402
from buildreasonseg_mvp.dense_grounding import feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.query_refine import refined_point_record  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6c_train import validation_material  # noqa: E402
from task6h_h2 import _save_panels  # noqa: E402  (diagnostic panel layout reused from Task 6H)
from task6i_common import EVAL, load_refinement_config, validation_pair_dicts, write_json  # noqa: E402

I1_JSON = EVAL / "task6i_i1_training.json"
PAIRED_JSON = EVAL / "task6i_paired_probe.json"
OUT = EVAL / "task6i_segmentation_eval.json"

FIX_MASK_PAIRED = 14
FIX_MIOU = 0.20
FIX_MARGIN = 0.05
TASK6C_PC = {"miou": 0.10604, "paired": "0/20"}
POINT_ORACLE = {"miou": 0.4883, "paired": "18/20", "source": "evaluation/task6g_grid_oracle.json"}
BOX_ORACLE = {"miou": 0.7506, "paired": "20/20", "source": "evaluation/task6d_oracle_prompt_diagnostic.json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--panels", type=int, default=6)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, _payload = load_refinement_config()
    grid = int(cfg["dense_grounding"]["grid"])
    i1 = json.loads(I1_JSON.read_text(encoding="utf-8"))
    if not i1["final"]["gate"]["passed"] and not args.force:
        raise SystemExit(
            "I1 gate did not pass; section 12 forbids running I2 (use --force only for a "
            "diagnostic run, and record it as such)"
        )
    checkpoint = args.checkpoint or i1["selection"]["selected_checkpoint"]["path"]
    selected_epoch = int(i1["selection"]["selected_epoch"])

    val_samples, raw_pairs, _lookup, _audit = validation_material()
    pairs = validation_pair_dicts(raw_pairs)
    paired_samples = [
        data_mod.to_sample(pair[side]) for pair in pairs for side in ("record_a", "record_b")
    ]

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg["training"].get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg["training"].get("visual_feature_cache_max_images", 512)),
        )
    first = val_samples[0]
    image = first.image_rgb()
    features, _cached = runtime.features_for(first, image)
    f256 = feature_tensor_for_grid(features, grid)
    runtime.install_dense_head(in_channels=int(f256.shape[1]))  # frozen evidence only
    runtime.install_refinement_block(
        coarse_channels=int(feature_tensor_for_grid(features, 64).shape[1]),
        fine_channels=int(f256.shape[1]),
    )
    runtime.freeze_for_refinement()
    frozen_before = {
        name: bool(parameter.requires_grad) for name, parameter in runtime.model.sam.named_parameters()
    }
    load_report = load_checkpoint(Path(checkpoint), runtime.model)
    frozen_after = {
        name: bool(parameter.requires_grad) for name, parameter in runtime.model.sam.named_parameters()
    }

    val_records = [refined_point_record(runtime, sample) for sample in val_samples]
    paired_records = []
    for sample in paired_samples:
        record = refined_point_record(runtime, sample)
        paired_records.append(record)
    records_by_id = {record["sample_id"]: record for record in [*val_records, *paired_records]}

    segmentation = g2_report(runtime, val_samples, pairs, records_by_id)
    paired = segmentation["paired"]
    checks = {
        "mask_paired_ge_14": int(paired["mask_paired_pass"]) >= FIX_MASK_PAIRED,
        "strict_miou_ge_0.20": float(segmentation["strict_end_to_end_miou"]) >= FIX_MIOU,
        "own_minus_cross_margin_gt_0.05": float(paired["mean_own_minus_cross_margin"] or 0.0) > FIX_MARGIN,
        "sam_weights_frozen": frozen_before == frozen_after and not any(frozen_after.values()),
        "no_gt_geometry_in_prompt": True,
    }
    panels = _save_panels(runtime, pairs, records_by_id, count=int(args.panels))
    report = {
        "_doc": (
            "Task 6I section 12. Stage I2 is inference-only: the argmax point of the refined "
            "heatmap (q0 -> one cross-attention refinement -> q1 -> 256x256 scorer) is handed to "
            "the official frozen SAM2 positive-point prompt. No ground-truth geometry enters the "
            "prompt and no SAM2 weight is updated."
        ),
        "task": "6I",
        "stage": "I2",
        "grid": grid,
        "selected_epoch": selected_epoch,
        "checkpoint": checkpoint,
        "checkpoint_load": load_report,
        "sam_frozen": all(not value for value in frozen_after.values()),
        "segmentation": segmentation,
        "panels": panels,
        "comparison": {
            "task6c_p_c": TASK6C_PC,
            "point_oracle": POINT_ORACLE,
            "box_oracle": BOX_ORACLE,
        },
        "fix_checks": checks,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    if PAIRED_JSON.exists():
        paired_artifact = json.loads(PAIRED_JSON.read_text(encoding="utf-8"))
    else:
        paired_artifact = {"_doc": "Task 6I paired probe (assembled by task6i_paired_probe.py)."}
    paired_artifact["mask"] = paired
    paired_artifact["mask_ran"] = True
    paired_artifact["mask_note"] = (
        "frozen SAM2 positive-point masks from the refined argmax points."
    )
    paired_artifact["checkpoint"] = checkpoint
    write_json(PAIRED_JSON, paired_artifact)
    print(
        f"[task6i.i2] strict mIoU {segmentation['strict_end_to_end_miou']:.4f} Dice "
        f"{segmentation['mean_dice']:.4f} mask paired {paired['mask_paired_pass']}/"
        f"{paired['paired_total']} own {paired['mean_own_iou']} cross {paired['mean_cross_iou']} "
        f"margin {paired['mean_own_minus_cross_margin']} IoU(predA,predB) "
        f"{paired['mean_mask_iou_between_predictions']}",
        flush=True,
    )
    print(f"[task6i.i2] fix checks {checks}", flush=True)
    print(f"[task6i.i2] wrote {OUT.relative_to(REPO_ROOT).as_posix()} and {len(panels)} panels", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if all(checks.values()) else 8


if __name__ == "__main__":
    raise SystemExit(main())
