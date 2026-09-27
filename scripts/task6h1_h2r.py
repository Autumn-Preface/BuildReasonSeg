"""Task 6H.1 sections 20-21: Stage H2-R -- frozen-SAM2 segmentation (only if H1-R passes).

`image + instruction -> [BOX] -> dense heatmap logits -> argmax point -> official frozen SAM2
positive-point prompt -> frozen SAM2 mask decoder -> mask`. No GT, no SAM2 training. Writes
`evaluation/task6h1_segmentation_eval.json`, the mask side of `evaluation/task6h1_paired_probe.json`
and paired diagnostic panels (layout reused from the Task 6H stage).
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
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_objective import point_record  # noqa: E402

from task6c_train import validation_material  # noqa: E402
from task6h1_common import EVAL, load_point_config, validation_pair_dicts, write_json  # noqa: E402
from task6h_h2 import _save_panels  # noqa: E402  (diagnostic panel layout reused from Task 6H)

H1R_JSON = EVAL / "task6h1_h1r_training.json"
PAIRED_JSON = EVAL / "task6h1_paired_probe.json"
OUT = EVAL / "task6h1_segmentation_eval.json"

FIX_MASK_PAIRED = 14
FIX_MIOU = 0.20
FIX_MARGIN = 0.05
TASK6C_PC = {"miou": 0.10604, "paired": "0/20"}
POINT_ORACLE = {"miou": 0.4876, "paired": "18/20", "source": "evaluation/task6d_oracle_prompt_diagnostic.json"}
BOX_ORACLE = {"miou": 0.7506, "paired": "20/20", "source": "evaluation/task6d_oracle_prompt_diagnostic.json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--panels", type=int, default=6)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, _payload = load_point_config()
    grid = int(cfg["dense_grounding"]["grid"])
    h1r = json.loads(H1R_JSON.read_text(encoding="utf-8"))
    if not h1r["final"]["gate"]["passed"] and not args.force:
        raise SystemExit(
            "H1-R gate did not pass; section 20 forbids running H2-R (use --force only for a "
            "diagnostic run, and record it as such)"
        )
    checkpoint = args.checkpoint or h1r["selection"]["selected_checkpoint"]["path"]
    selected_epoch = int(h1r["selection"]["selected_epoch"])

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
    runtime.install_dense_head(in_channels=int(feature_tensor_for_grid(features, grid).shape[1]))
    frozen_before = {
        name: bool(parameter.requires_grad) for name, parameter in runtime.model.sam.named_parameters()
    }
    load_report = load_checkpoint(Path(checkpoint), runtime.model)
    frozen_after = {
        name: bool(parameter.requires_grad) for name, parameter in runtime.model.sam.named_parameters()
    }

    val_records = [point_record(runtime, sample) for sample in val_samples]
    paired_records = []
    for sample in paired_samples:
        record = point_record(runtime, sample)
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
            "Task 6H.1 sections 20-21. Stage H2-R is inference-only: the argmax point of the "
            "point-supervised heatmap is handed to the official frozen SAM2 positive-point prompt. "
            "No ground-truth geometry enters the prompt and no SAM2 weight is updated."
        ),
        "task": "6H.1",
        "stage": "H2-R",
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
    paired_artifact = json.loads(PAIRED_JSON.read_text(encoding="utf-8"))
    paired_artifact["mask"] = paired
    paired_artifact["mask_ran"] = True
    paired_artifact["mask_note"] = (
        "frozen SAM2 positive-point masks from the point-supervised argmax points."
    )
    paired_artifact["checkpoint"] = checkpoint
    write_json(PAIRED_JSON, paired_artifact)
    print(
        f"[task6h1.h2r] strict mIoU {segmentation['strict_end_to_end_miou']:.4f} Dice "
        f"{segmentation['mean_dice']:.4f} mask paired {paired['mask_paired_pass']}/"
        f"{paired['paired_total']} own {paired['mean_own_iou']} cross {paired['mean_cross_iou']} "
        f"margin {paired['mean_own_minus_cross_margin']} IoU(predA,predB) "
        f"{paired['mean_mask_iou_between_predictions']}",
        flush=True,
    )
    print(f"[task6h1.h2r] fix checks {checks}", flush=True)
    print(f"[task6h1.h2r] wrote {OUT.relative_to(REPO_ROOT).as_posix()} and {len(panels)} panels", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if all(checks.values()) else 8


if __name__ == "__main__":
    raise SystemExit(main())
