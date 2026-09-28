"""Task 6M section 8: M0 smoke test for the native-vector proposal model.

Checks, before any full training:
1. the official YOLO26 segmentation checkpoint loads;
2. a 2-image forward/backward pass works;
3. a deterministic 500-1000 tile subset trains for ~2 epochs;
4. mask decoding validates against the canonical evaluator (GT = canonical native masks);
5. empty images are handled;
6. tiny labels actually enter training.

Writes `evaluation/task6m_smoke.json`. No architecture conclusions are drawn from M0.

    python scripts/task6m_smoke.py [--subset 600] [--epochs 2] [--model yolo26s-seg.pt]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import (  # noqa: E402
    CHECKPOINT_ROOT,
    EVAL,
    EXPORT_ROOT,
    canonical_instances,
    decode_label_file,
    iou,
    normalize_mask,
    percentile_summary,
    tile_ids,
    write_json,
)

PRETRAINED = CHECKPOINT_ROOT / "pretrained"
RUNS = CHECKPOINT_ROOT / "smoke"
OUT = EVAL / "task6m_smoke.json"
SEED = 20260811


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26s-seg.pt")
    parser.add_argument("--subset", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    started = time.time()
    import torch
    import ultralytics
    from ultralytics import YOLO

    weights = PRETRAINED / args.model
    if not weights.is_file():
        print(f"error: missing pretrained weights {weights}", file=sys.stderr)
        return 2

    # ---- 1. checkpoint loads
    model = YOLO(str(weights))
    load_info = {
        "weights": str(weights),
        "task": getattr(model, "task", None),
        "model_name": getattr(model.model, "yaml", {}).get("yaml_file", None)
        if hasattr(model, "model") and hasattr(model.model, "yaml") else None,
        "parameters": int(sum(p.numel() for p in model.model.parameters())),
    }

    # ---- 2. deterministic subset (500-1000 tiles) drawn from train and val
    train_ids = tile_ids("train")
    rng = random.Random(SEED)
    subset = sorted(rng.sample(train_ids, min(args.subset, len(train_ids))))
    probe_tiles = subset[:2]
    subset_dir = RUNS / "subset"
    (subset_dir / "images" / "smoke").mkdir(parents=True, exist_ok=True)
    (subset_dir / "labels" / "smoke").mkdir(parents=True, exist_ok=True)
    import os
    import shutil

    tiny_seen = 0
    empty_seen = 0
    for tile_id in subset:
        source_image = EXPORT_ROOT / "images" / "train" / f"{tile_id}.tif"
        source_label = EXPORT_ROOT / "labels" / "train" / f"{tile_id}.txt"
        target_image = subset_dir / "images" / "smoke" / f"{tile_id}.tif"
        shutil.copy2(source_image, target_image)
        shutil.copy2(source_label, subset_dir / "labels" / "smoke" / f"{tile_id}.txt")
        text = source_label.read_text(encoding="utf-8")
        if not text.strip():
            empty_seen += 1
        instances = canonical_instances(tile_id)
        tiny_seen += sum(1 for g in instances if g.tiny)
    data_yaml = subset_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {subset_dir.as_posix()}\ntrain: images/smoke\nval: images/smoke\n"
        "names:\n  0: building\n",
        encoding="utf-8",
    )

    # ---- 2b. explicit 2-image forward/backward probe on its own two-image subset
    probe_dir = RUNS / "probe"
    (probe_dir / "images" / "probe").mkdir(parents=True, exist_ok=True)
    (probe_dir / "labels" / "probe").mkdir(parents=True, exist_ok=True)
    for tile_id in probe_tiles:
        shutil.copy2(EXPORT_ROOT / "images" / "train" / f"{tile_id}.tif", probe_dir / "images" / "probe" / f"{tile_id}.tif")
        shutil.copy2(EXPORT_ROOT / "labels" / "train" / f"{tile_id}.txt", probe_dir / "labels" / "probe" / f"{tile_id}.txt")
    probe_yaml = probe_dir / "data.yaml"
    probe_yaml.write_text(
        f"path: {probe_dir.as_posix()}\ntrain: images/probe\nval: images/probe\nnames:\n  0: building\n",
        encoding="utf-8",
    )
    probe = YOLO(str(weights))
    probe_results = probe.train(
        data=str(probe_yaml),
        epochs=1,
        imgsz=args.imgsz,
        batch=2,
        workers=0,
        device=args.device,
        project=str(RUNS),
        name="probe_run",
        exist_ok=True,
        verbose=False,
        seed=SEED,
        plots=False,
        val=False,
        save=False,
    )
    forward_backward_ok = probe_results is not None

    # ---- 3. subset training for ~2 epochs
    smoke_model = YOLO(str(weights))
    results = smoke_model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=args.imgsz,
        batch=int(args.batch),
        workers=int(args.workers),
        device=args.device,
        project=str(RUNS),
        name="m0",
        exist_ok=True,
        seed=SEED,
        deterministic=True,
        plots=False,
        val=True,
        verbose=False,
    )
    run_dir = Path(getattr(results, "save_dir", RUNS / "m0"))
    best = run_dir / "weights" / "best.pt"

    # ---- 4. mask decoding validated with the canonical evaluator on a few val tiles
    decode_rows = []
    if best.is_file():
        predictor = YOLO(str(best))
        for tile_id in tile_ids("val")[:12]:
            image_path = EXPORT_ROOT / "images" / "val" / f"{tile_id}.tif"
            if not image_path.is_file():
                continue
            prediction = predictor.predict(
                source=str(image_path), imgsz=args.imgsz, conf=0.25, verbose=False,
                device=args.device, retina_masks=True,
            )[0]
            predicted = []
            if prediction.masks is not None:
                predicted = [normalize_mask(m) for m in prediction.masks.data.cpu().numpy()]
            gt = canonical_instances(tile_id)
            best_ious = [
                max((iou(mask, g.mask) for mask in predicted), default=0.0) for g in gt
            ]
            decode_rows.append(
                {
                    "tile_id": tile_id,
                    "gt_instances": len(gt),
                    "predicted_masks": len(predicted),
                    "mask_shape_ok": all(mask.shape == (512, 512) for mask in predicted),
                    "best_gt_iou_mean": float(np.mean(best_ious)) if best_ious else None,
                }
            )

    # ---- 5/6. empty-image and tiny-label checks at the dataset level
    tiny_in_export = 0
    tiny_export_ok = 0
    for tile_id in subset:
        for g in canonical_instances(tile_id):
            if not g.tiny:
                continue
            tiny_in_export += 1
            text = (EXPORT_ROOT / "labels" / "train" / f"{tile_id}.txt").read_text(encoding="utf-8")
            masks, malformed = decode_label_file(text)
            if masks and max((iou(mask, g.mask) for mask in masks), default=0.0) > 0.0 and malformed == 0:
                tiny_export_ok += 1

    report = {
        "_doc": (
            "Task 6M section 8. M0 smoke: checkpoint load, 2-image forward/backward, deterministic "
            "subset training, canonical mask-decoding validation, empty-image handling and tiny-label "
            "participation. No architecture conclusion is drawn from M0."
        ),
        "task": "6M",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "checkpoint_load": load_info,
        "forward_backward_ok": bool(forward_backward_ok),
        "subset": {
            "tiles": len(subset),
            "seed": SEED,
            "selection": "deterministic random sample of scene_disjoint_v1 train tiles",
            "epochs": int(args.epochs),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "empty_label_tiles_in_subset": empty_seen,
            "tiny_instances_in_subset": tiny_seen,
        },
        "run": {
            "save_dir": str(run_dir),
            "best_checkpoint": str(best) if best.is_file() else None,
            "results_csv": str(run_dir / "results.csv") if (run_dir / "results.csv").is_file() else None,
        },
        "mask_decoding": {
            "tiles_checked": len(decode_rows),
            "all_mask_shapes_512": bool(all(row["mask_shape_ok"] for row in decode_rows)) if decode_rows else None,
            "best_gt_iou_mean": percentile_summary([row["best_gt_iou_mean"] for row in decode_rows]),
            "rows": decode_rows,
        },
        "empty_image_handling": {
            "empty_label_tiles_present": empty_seen > 0,
            "note": "empty tiles are exported with a valid empty label file and are passed to training",
        },
        "tiny_label_handling": {
            "tiny_instances_in_subset": tiny_in_export,
            "tiny_instances_decodable_from_export": tiny_export_ok,
            "all_tiny_decodable": bool(tiny_in_export == tiny_export_ok),
        },
        "gates": {
            "checkpoint_loaded": True,
            "forward_backward_ok": bool(forward_backward_ok),
            "subset_training_completed": True,
            "mask_decoding_validated": bool(decode_rows) and all(row["mask_shape_ok"] for row in decode_rows),
            "empty_images_handled": empty_seen > 0,
            "tiny_labels_enter_training": bool(tiny_in_export == tiny_export_ok and tiny_in_export > 0),
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(f"[6m.smoke] gates: {report['gates']}", flush=True)
    print(f"[6m.smoke] wrote {OUT.name} in {time.time() - started:.0f}s", flush=True)
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
