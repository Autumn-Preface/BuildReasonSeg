"""Task 6G sections 2-3: grid-snapped POINT oracle and spatial-level selection.

For every validation target, the frozen Task 6D deterministic interior point is snapped to the
nearest cell centre of the 64x64, 128x128 and 256x256 grids and fed through the official frozen
SAM2 positive-point prompt path, on the same 120 validation records and 20 paired validation
images. The selected grid is the **smallest** one that keeps paired mask >= 18/20 and strict mIoU
>= 0.4576 (continuous point oracle 0.4876 minus 0.03); otherwise
`DENSE_GRID_POINT_PATH_INADEQUATE` and the task stops before training.

Writes `evaluation/task6g_grid_oracle.json`.
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
from buildreasonseg_mvp.dense_grounding import (  # noqa: E402
    GRID_CANDIDATES,
    feature_level_report,
    snap_point_to_cell,
    snapped_point,
)
from buildreasonseg_mvp.grounding import decode_mask_from_geometry, distance_transform_point  # noqa: E402
from buildreasonseg_mvp.metrics import dice_from_logits, mask_iou_from_logits  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_seed  # noqa: E402

from task6c6_common import EVAL, write_json  # noqa: E402
from task6c_train import validation_material  # noqa: E402
from task6g_common import CONFIG, validation_pair_dicts  # noqa: E402

OUT = EVAL / "task6g_grid_oracle.json"
CONTINUOUS_POINT_MIOU = 0.4876
CONTINUOUS_POINT_PAIRED = 18
MIOU_TOLERANCE = 0.03


def _iou(logits: torch.Tensor, mask) -> float:
    target = torch.as_tensor(mask).float()
    return float(mask_iou_from_logits(logits, target, tuple(target.shape), threshold=0.0))


def _dice(logits: torch.Tensor, mask) -> float:
    target = torch.as_tensor(mask).float()
    return float(dice_from_logits(logits, target, tuple(target.shape), threshold=0.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", default="64,128,256")
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    # The oracle does not consume the selection it produces, so it loads the config directly
    # (load_dense_config would require this artifact to exist first).
    cfg = load_config(CONFIG)
    cfg["dense_grounding"]["grid"] = 64  # placeholder; the dense head is not installed here
    grids = [int(value) for value in args.grids.split(",")]

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    val_samples, raw_pairs, _lookup, _audit = validation_material()
    if args.limit_val:
        val_samples = val_samples[: int(args.limit_val)]
    pairs = validation_pair_dicts(raw_pairs)
    if args.limit_pairs:
        pairs = pairs[: int(args.limit_pairs)]

    probe_image = val_samples[0].image_rgb()
    probe_features, _cached = runtime.features_for(val_samples[0], probe_image)
    level_report = feature_level_report(probe_features)

    report = {
        "_doc": (
            "Task 6G sections 2-3. Grid-snapped point oracle: the frozen Task 6D interior point is "
            "snapped to each candidate grid's nearest cell centre and fed to the official frozen "
            "SAM2 positive-point prompt on the fixed 120 validation records and 20 paired "
            "validation images. GT points are used only here (oracle)."
        ),
        "task": "6G",
        "continuous_point_oracle_reference": {
            "miou": CONTINUOUS_POINT_MIOU,
            "paired": f"{CONTINUOUS_POINT_PAIRED}/20",
            "source": "evaluation/task6d_oracle_prompt_diagnostic.json",
        },
        "feature_levels": level_report,
        "candidates": {},
    }

    for grid in grids:
        records = []
        for sample in val_samples:
            image = sample.image_rgb()
            features, _cached = runtime.features_for(sample, image)
            mask = sample.target_mask()
            point = distance_transform_point(mask)
            cell = snap_point_to_cell(point, grid)
            snapped = snapped_point(point, grid)
            displacement_normalized = (
                abs(snapped[0] - point[0]) + abs(snapped[1] - point[1])
            ) / 2.0
            tensor = torch.as_tensor(snapped, dtype=torch.float32, device=runtime.device).reshape(1, -1)
            result = decode_mask_from_geometry(runtime.model.sam, features, tensor, "point")
            records.append(
                {
                    "sample_id": str(sample.sample_id),
                    "image_id": str(sample.image_id),
                    "level": int(sample.level),
                    "gt_point": list(point),
                    "snapped_point": list(snapped),
                    "snapped_cell": list(cell),
                    "displacement_normalized": displacement_normalized,
                    "displacement_512px": displacement_normalized * 512.0,
                    "iou": _iou(result.low_res_logits, mask),
                    "dice": _dice(result.low_res_logits, mask),
                }
            )
            del result, features

        paired = []
        for pair in pairs:
            sample_a = data_mod.to_sample(pair["record_a"])
            sample_b = data_mod.to_sample(pair["record_b"])
            image = sample_a.image_rgb()
            features, _cached = runtime.features_for(sample_a, image)
            target_a, target_b = sample_a.target_mask(), sample_b.target_mask()
            point_a = snapped_point(distance_transform_point(target_a), grid)
            point_b = snapped_point(distance_transform_point(target_b), grid)
            result_a = decode_mask_from_geometry(
                runtime.model.sam, features,
                torch.as_tensor(point_a, dtype=torch.float32, device=runtime.device).reshape(1, -1),
                "point",
            )
            result_b = decode_mask_from_geometry(
                runtime.model.sam, features,
                torch.as_tensor(point_b, dtype=torch.float32, device=runtime.device).reshape(1, -1),
                "point",
            )
            own_a = _iou(result_a.low_res_logits, target_a)
            own_b = _iou(result_b.low_res_logits, target_b)
            cross_a = _iou(result_a.low_res_logits, target_b)
            cross_b = _iou(result_b.low_res_logits, target_a)
            paired.append(
                {
                    "image_id": str(pair["image_id"]),
                    "own_a": own_a,
                    "own_b": own_b,
                    "cross_a": cross_a,
                    "cross_b": cross_b,
                    "paired_pass": bool(own_a > cross_a and own_b > cross_b),
                }
            )
            del result_a, result_b, features

        report["candidates"][str(grid)] = {
            "grid": grid,
            "strict_miou": float(sum(record["iou"] for record in records) / len(records)),
            "dice": float(sum(record["dice"] for record in records) / len(records)),
            "mean_displacement_normalized": float(
                sum(record["displacement_normalized"] for record in records) / len(records)
            ),
            "mean_displacement_512px": float(
                sum(record["displacement_512px"] for record in records) / len(records)
            ),
            "paired_pass": sum(1 for item in paired if item["paired_pass"]),
            "paired_total": len(paired),
            "paired_records": paired,
            "val_records": records,
        }
        print(
            f"[task6g:oracle] grid {grid}: mIoU "
            f"{report['candidates'][str(grid)]['strict_miou']:.4f} dice "
            f"{report['candidates'][str(grid)]['dice']:.4f} paired "
            f"{report['candidates'][str(grid)]['paired_pass']}/{len(paired)} disp "
            f"{report['candidates'][str(grid)]['mean_displacement_512px']:.2f}px",
            flush=True,
        )

    threshold = CONTINUOUS_POINT_MIOU - MIOU_TOLERANCE
    qualifying = [
        grid
        for grid in grids
        if report["candidates"][str(grid)]["paired_pass"] >= CONTINUOUS_POINT_PAIRED
        and report["candidates"][str(grid)]["strict_miou"] >= threshold
    ]
    selected = min(qualifying) if qualifying else None
    report["selection"] = {
        "rule": (
            "smallest grid whose snapped-point oracle keeps paired >= 18/20 and strict mIoU >= "
            f"{threshold:.4f} (continuous point oracle {CONTINUOUS_POINT_MIOU} minus {MIOU_TOLERANCE})"
        ),
        "miou_threshold": threshold,
        "qualifying_grids": qualifying,
        "selected_grid": selected,
        "verdict": f"PROCEED_WITH_GRID_{selected}" if selected else "DENSE_GRID_POINT_PATH_INADEQUATE",
    }
    report["seconds"] = round(time.time() - started, 2)
    write_json(OUT, report)
    print(f"[task6g:oracle] selection: {json.dumps(report['selection'], ensure_ascii=False)}", flush=True)
    print(f"[task6g:oracle] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if selected else 4


if __name__ == "__main__":
    raise SystemExit(main())
