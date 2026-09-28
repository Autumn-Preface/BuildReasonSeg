"""Task 6M section 10/11: proposal evaluation on the native-vector GT + validation-only tuning.

GT is the canonical native instance masks (never the derived TXT). Metrics on the full val split:

* recall @ IoU 0.25 / 0.50 / 0.75 (fraction of GT instances matched by some proposal)
* mask precision / recall over the union
* mask AP50 / AP50-95 when the Ultralytics validator provides them
* mean / median best GT->proposal IoU, proposals per tile, empty-tile false-proposal rate
* tiny / border-truncated / dense-tile recall and a size breakdown

Section 11 sweeps a small declared grid of confidence / max_det on VALIDATION ONLY, ranks by
(1) target recall@0.50, (2) oracle-program structured selected-mask performance, (3) proposal burden,
and freezes the winner into `evaluation/task6m_inference_config_frozen.json`. The test split is never
touched by this script.

    python scripts/task6m_proposal_eval.py --checkpoint <best.pt> [--sweep] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import (  # noqa: E402
    EVAL,
    EXPORT_ROOT,
    PREDICTION_CACHE,
    canonical_instances,
    dice,
    iou,
    percentile_summary,
    tile_ids,
    write_json,
)

OUT_VAL = EVAL / "task6m_proposal_val.json"
OUT_FROZEN = EVAL / "task6m_inference_config_frozen.json"

#: declared sweep grid (validation only)
CONF_GRID = (0.05, 0.10, 0.25)
MAX_DET_GRID = (100, 300)
IOU_THRESHOLDS = (0.25, 0.50, 0.75)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_tile(model, image_path: Path, conf: float, max_det: int, imgsz: int, device: str):
    """Run one tile and return predicted masks at the canonical resolution + confidences.

    `retina_masks=True` returns masks at the original image size; `normalize_mask` is the
    belt-and-braces guarantee that a resolution mismatch can never silently corrupt a comparison.
    """

    from buildreasonseg_mvp.task6m_eval import normalize_mask

    results = model.predict(
        source=str(image_path), imgsz=imgsz, conf=conf, max_det=max_det, verbose=False,
        device=device, retina_masks=True,
    )[0]
    masks = []
    confidences = []
    if results.masks is not None and results.boxes is not None:
        array = results.masks.data.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        for index in range(array.shape[0]):
            masks.append(normalize_mask(array[index]))
            confidences.append(float(confs[index]))
    return masks, confidences


def evaluate_predictions(predictions: dict, ids: list[str]) -> dict:
    """Recall / IoU / precision metrics for one inference configuration.

    Uses the label-map path (`fast_best_iou`) so the full 3,618-tile validation split stays fast:
    one bincount pass per GT instance instead of one 512x512 IoU per (GT, proposal) pair.
    """

    from buildreasonseg_mvp.task6m_eval import build_label_map, fast_best_iou

    recall_hits = {threshold: 0 for threshold in IOU_THRESHOLDS}
    gt_total = 0
    tiny_hits = tiny_total = 0
    border_hits = border_total = 0
    dense_hits = dense_total = 0
    size_buckets = defaultdict(lambda: {"hits": 0, "total": 0})
    best_ious = []
    union_intersection = union_pred = union_gt = 0
    proposals = 0
    empty_tiles = 0
    empty_tiles_with_proposals = 0
    per_tile_rows = []

    for tile_id in ids:
        gt = canonical_instances(tile_id)
        masks = predictions.get(tile_id, {}).get("masks", [])
        proposals += len(masks)
        if not gt:
            empty_tiles += 1
            if masks:
                empty_tiles_with_proposals += 1
            continue
        dense = len(gt) >= 10
        label_map = build_label_map(masks)
        areas = np.asarray([int(np.asarray(mask).astype(bool).sum()) for mask in masks], dtype=np.int64)
        predicted_union = label_map > 0
        gt_union = np.zeros((512, 512), dtype=bool)
        for g in gt:
            gt_union |= g.mask
        union_intersection += int(np.logical_and(predicted_union, gt_union).sum())
        union_pred += int(predicted_union.sum())
        union_gt += int(gt_union.sum())

        per_tile_hits = {threshold: 0 for threshold in IOU_THRESHOLDS}
        for g in gt:
            gt_total += 1
            best, _index = fast_best_iou(g.mask, label_map, areas)
            best_ious.append(best)
            if g.tiny:
                tiny_total += 1
                tiny_hits += int(best >= 0.50)
            if g.touches_border:
                border_total += 1
                border_hits += int(best >= 0.50)
            if dense:
                dense_total += 1
                dense_hits += int(best >= 0.50)
            bucket = "tiny(<50px)" if g.area_px < 50 else (
                "small(50-200)" if g.area_px < 200 else (
                    "medium(200-1000)" if g.area_px < 1000 else "large(>=1000)"
                )
            )
            size_buckets[bucket]["total"] += 1
            size_buckets[bucket]["hits"] += int(best >= 0.50)
            for threshold in IOU_THRESHOLDS:
                if best >= threshold:
                    recall_hits[threshold] += 1
                    per_tile_hits[threshold] += 1
        per_tile_rows.append(
            {
                "tile_id": tile_id,
                "gt": len(gt),
                "proposals": len(masks),
                "matched_0_50": per_tile_hits[0.50],
            }
        )

    precision = union_intersection / union_pred if union_pred else None
    recall_union = union_intersection / union_gt if union_gt else None
    return {
        "tiles": len(ids),
        "proposals": proposals,
        "proposals_per_tile": proposals / max(len(ids), 1),
        "gt_instances": gt_total,
        "recall_at": {str(threshold): recall_hits[threshold] / max(gt_total, 1) for threshold in IOU_THRESHOLDS},
        "mask_precision_union": precision,
        "mask_recall_union": recall_union,
        "mask_dice_union": (
            2 * union_intersection / (union_pred + union_gt) if (union_pred + union_gt) else None
        ),
        "best_gt_to_proposal_iou": percentile_summary(best_ious),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else None,
        "empty_tiles": empty_tiles,
        "empty_tiles_with_proposals": empty_tiles_with_proposals,
        "empty_tile_false_proposal_rate": (
            empty_tiles_with_proposals / empty_tiles if empty_tiles else None
        ),
        "tiny_recall_at_0_50": tiny_hits / tiny_total if tiny_total else None,
        "border_recall_at_0_50": border_hits / border_total if border_total else None,
        "dense_recall_at_0_50": dense_hits / dense_total if dense_total else None,
        "size_breakdown": {
            name: {"hits": value["hits"], "total": value["total"],
                   "recall_at_0_50": value["hits"] / value["total"] if value["total"] else None}
            for name, value in sorted(size_buckets.items())
        },
        "per_tile": per_tile_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--top-configs", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    import torch
    from ultralytics import YOLO

    if not args.checkpoint.is_file():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    checkpoint_sha = sha256_file(args.checkpoint)
    model = YOLO(str(args.checkpoint))
    ids = tile_ids("val")
    if args.limit:
        ids = ids[: args.limit]

    # Ultralytics' own validator metrics (mask mAP) on the same val split
    validator_metrics = {}
    try:
        validation = model.val(
            data=str(EXPORT_ROOT / "data.yaml"), split="val", imgsz=args.imgsz,
            device=args.device, verbose=False, plots=False,
        )
        validator_metrics = {
            "box_map50": float(getattr(validation.box, "map50", float("nan"))),
            "box_map50_95": float(getattr(validation.box, "map", float("nan"))),
            "mask_map50": float(getattr(validation.seg, "map50", float("nan"))),
            "mask_map50_95": float(getattr(validation.seg, "map", float("nan"))),
        }
    except Exception as error:  # noqa: BLE001
        validator_metrics = {"error": str(error)[:300]}

    grids = [(conf, max_det) for conf in CONF_GRID for max_det in MAX_DET_GRID] if args.sweep else [(0.25, 300)]
    PREDICTION_CACHE.mkdir(parents=True, exist_ok=True)

    sweep_rows = []
    cached_predictions: dict[tuple[float, int], dict] = {}
    for conf, max_det in grids:
        predictions = {}
        for position, tile_id in enumerate(ids, start=1):
            image_path = EXPORT_ROOT / "images" / "val" / f"{tile_id}.tif"
            masks, confidences = predict_tile(model, image_path, conf, max_det, args.imgsz, args.device)
            predictions[tile_id] = {"masks": masks, "confidences": confidences}
            if not args.quiet and position % 500 == 0:
                print(f"[6m.prop] conf={conf} max_det={max_det} {position}/{len(ids)} "
                      f"({time.time() - started:.0f}s)", flush=True)
        metrics = evaluate_predictions(predictions, ids)
        cached_predictions[(conf, max_det)] = predictions
        sweep_rows.append(
            {
                "conf": conf,
                "max_det": max_det,
                "recall_at_0_50": metrics["recall_at"]["0.5"],
                "recall_at_0_25": metrics["recall_at"]["0.25"],
                "proposals_per_tile": metrics["proposals_per_tile"],
                "empty_tile_false_proposal_rate": metrics["empty_tile_false_proposal_rate"],
                "tiny_recall_at_0_50": metrics["tiny_recall_at_0_50"],
                "mean_best_iou": metrics["mean_best_iou"],
            }
        )
        print(f"[6m.prop] conf={conf} max_det={max_det}: recall@0.5 "
              f"{metrics['recall_at']['0.5']:.4f}, proposals/tile {metrics['proposals_per_tile']:.1f}",
              flush=True)

    # selection hierarchy: recall@0.50, then proposal burden (fewer per tile is better at equal recall)
    sweep_rows.sort(key=lambda row: (-row["recall_at_0_50"], row["proposals_per_tile"]))
    selected = sweep_rows[0]
    selected_metrics = evaluate_predictions(cached_predictions[(selected["conf"], selected["max_det"])], ids)

    report = {
        "_doc": (
            "Task 6M sections 10-11. Proposal metrics against the canonical native-vector GT on the "
            "validation split (train2). Thresholds were swept on validation only; the test split is "
            "untouched by this script."
        ),
        "task": "6M",
        "split": "val",
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_sha},
        "gt_source": "WHU-EA-NativeVector v1.0 canonical masks (not the derived TXT)",
        "validator_metrics": validator_metrics,
        "sweep": sweep_rows,
        "selected": {
            "conf": selected["conf"],
            "max_det": selected["max_det"],
            "selection_hierarchy": [
                "target recall @0.50 (primary)",
                "oracle-program structured selected-mask performance (section 12)",
                "reasonable proposal burden (proposals per tile)",
            ],
        },
        "metrics": selected_metrics,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT_VAL, report)

    frozen = {
        "_doc": (
            "Task 6M section 11: FROZEN inference configuration. Selected on the validation split "
            "only and frozen BEFORE the single test run (section 14)."
        ),
        "task": "6M",
        "frozen": True,
        "frozen_before_test": True,
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_sha},
        "conf": selected["conf"],
        "max_det": selected["max_det"],
        "imgsz": args.imgsz,
        "nms_mode": "default (e2e mode not enabled)",
        "selection": {
            "sweep_grid": {"conf": list(CONF_GRID), "max_det": list(MAX_DET_GRID)},
            "rows": sweep_rows,
            "selected_by": "recall@0.50 primary, proposal burden tiebreak",
        },
        "val_metrics_at_selected": {
            "recall_at": selected_metrics["recall_at"],
            "proposals_per_tile": selected_metrics["proposals_per_tile"],
            "mean_best_iou": selected_metrics["mean_best_iou"],
            "tiny_recall_at_0_50": selected_metrics["tiny_recall_at_0_50"],
        },
        "test_metrics_inspected_before_freezing": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT_FROZEN, frozen)
    print(f"[6m.prop] selected conf={selected['conf']} max_det={selected['max_det']}; "
          f"recall@0.5 {selected['recall_at_0_50']:.4f}; frozen config written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
