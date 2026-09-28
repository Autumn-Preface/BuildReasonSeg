"""Task 6K single-pass read-only evidence scan over all 4038 aligned tiles.

Computes, per tile, everything the Part B/C/D/G artifacts need:

* source raster inventory (shape, dtype, unique values, foreground fraction, empty flag);
* raw 8-connected (primary) and 4-connected (sensitivity) semantic components;
* the historical contour pipeline reproduced in memory, with the loss of each stage recorded
  separately (``<50`` filter, ``RETR_EXTERNAL`` hole filling, ``approxPolyDP``);
* the actual YOLO polygon labels parsed and rasterized, compared with the emulator;
* the current BuildReasonSeg component representation;
* a clearly labelled merge-risk heuristic on the raw semantic components.

Writes the per-tile cache to gitignored ``artifacts/task6k/tile_scan.jsonl`` plus a run summary.
Reads nothing outside the read-only sources; writes nothing outside BuildReasonSeg.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_source_audit import (  # noqa: E402
    CONVERTED_CANDIDATES,
    MIN_APPROX_VERTICES,
    THRESHOLD,
    audit_contours,
    binarize,
    boundary_displacement,
    dice,
    external_contour_target,
    hole_analysis,
    iou,
    parse_yolo_polygons,
    path_provenance,
    polygon_vertex_clip_stats,
    post_filter_target,
    precision_recall,
    raw_components,
    rasterize_polygons,
    read_semantic_label,
    simplified_polygon_target,
)

from task6k_common import aligned_tiles  # noqa: E402

CACHE_DIR = REPO_ROOT / "artifacts" / "task6k"
CACHE = CACHE_DIR / "tile_scan.jsonl"
SUMMARY = CACHE_DIR / "tile_scan_summary.json"


def merge_risk_of_component(mask: np.ndarray) -> dict:
    """Clearly-labelled heuristic merge risk of one semantic component (not instance truth).

    Applied only to components with raster area >= 150 px (the frozen tiny-component
    threshold); smaller specks are recorded as ``low`` without a distance transform, which is
    both cheaper and honest: a <150 px speck cannot plausibly be several merged buildings.
    """

    import cv2

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    n_pieces_1, _ = cv2.connectedComponents(eroded, connectivity=8)
    pieces_1 = max(0, int(n_pieces_1) - 1)

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if float(distance.max()) >= 2.0:
        peak_candidates = (distance >= 0.7 * float(distance.max())) & (distance >= 2.0)
        dilated = cv2.dilate(peak_candidates.astype(np.uint8), kernel, iterations=1)
        n_peaks, _ = cv2.connectedComponents(dilated, connectivity=8)
        peaks = max(0, int(n_peaks) - 1)
    else:
        peaks = 1 if mask.any() else 0

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = float(sum(cv2.contourArea(c) for c in contours))
    hull_area = 0.0
    if contours:
        points = np.vstack(contours)
        hull = cv2.convexHull(points)
        hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 1.0

    if pieces_1 >= 2 or peaks >= 3:
        risk = "high"
    elif peaks == 2 or solidity < 0.75 or pieces_1 == 0:
        risk = "medium"
    else:
        risk = "low"
    return {
        "erosion1_pieces": pieces_1,
        "dt_peaks": peaks,
        "solidity": round(solidity, 4),
        "risk": risk,
    }


MIN_MERGE_RISK_AREA = 150


def scan_tile(tile) -> dict:
    import cv2

    cv2_imread_failed = cv2.imread(str(tile.source_label), cv2.IMREAD_GRAYSCALE) is None
    mask = read_semantic_label(tile.source_label)
    unique_values = [int(v) for v in np.unique(mask)]
    binary = binarize(mask)
    foreground = binary > 0

    labels8, records8 = raw_components(binary, connectivity=8)
    _labels4, records4 = raw_components(binary, connectivity=4)

    audit = audit_contours(binary)
    post_filter = post_filter_target(binary)
    simplified = simplified_polygon_target(binary)
    external = external_contour_target(binary)
    topology = hole_analysis(binary)

    polygons, parse_info = parse_yolo_polygons(tile.yolo_label)
    clip_stats = polygon_vertex_clip_stats(polygons)
    yolo_mask = rasterize_polygons(polygons, tile.width, tile.height)

    precision, recall = precision_recall(simplified, yolo_mask)
    boundary = boundary_displacement(simplified, yolo_mask)
    approx_boundary = boundary_displacement(post_filter, simplified)

    current_components = tile.metadata["components"] if tile.metadata else []

    risks = []
    for record in records8:
        if int(record.area_px) >= MIN_MERGE_RISK_AREA:
            risks.append(merge_risk_of_component(labels8 == record.component_id))
        else:
            risks.append(
                {"erosion1_pieces": 0, "dt_peaks": 0, "solidity": 1.0, "risk": "low",
                 "skipped_below_area": MIN_MERGE_RISK_AREA}
            )
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    high_ids = []
    for record, risk in zip(records8, risks):
        risk_counts[risk["risk"]] += 1
        if risk["risk"] == "high":
            high_ids.append(int(record.component_id))

    return {
        "split": tile.split,
        "source_split": tile.source_split,
        "stem": tile.stem,
        "cv2_imread_failed": bool(cv2_imread_failed),
        "source": {
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
            "dtype": str(mask.dtype),
            "unique_values": unique_values,
            "foreground_pixels": int(foreground.sum()),
            "foreground_fraction": float(foreground.mean()),
            "empty": bool(foreground.sum() == 0),
            "threshold": int(THRESHOLD),
        },
        "raw_components_8": {
            "count": len(records8),
            "areas": [int(record.area_px) for record in records8],
            "bbox_wh": [[int(record.width_px), int(record.height_px)] for record in records8],
            "centroids": [[round(float(record.centroid_px[0]), 4), round(float(record.centroid_px[1]), 4)] for record in records8],
            "border_touch": [bool(record.touches_image_border) for record in records8],
        },
        "raw_components_4": {"count": len(records4)},
        "contours": {
            "before_filter": audit.n_contours_before,
            "kept": audit.n_contours_kept,
            "removed": audit.n_contours_removed,
            "removed_contour_areas": [round(v, 6) for v in audit.removed_contour_areas],
            "removed_raster_pixels": list(audit.removed_raster_pixels),
            "kept_raster_pixels": list(audit.kept_raster_pixels),
            "approx_vertex_counts": list(audit.approx_vertex_counts),
            "dropped_below_three_vertices": int(audit.dropped_below_three_vertices),
        },
        "topology": {
            **topology,
            "external_pixels": int(external.sum()),
            "external_minus_foreground": int((external & ~foreground).sum()),
        },
        "yolo": {
            "n_polygons": len(polygons),
            "malformed_lines": parse_info["malformed_lines"],
            "wrong_class_lines": parse_info["wrong_class_lines"],
            "vertices_total": clip_stats["vertices"],
            "out_of_range_vertices": clip_stats["out_of_range_vertices"],
            "raster_pixels": int(yolo_mask.sum()),
        },
        "current": {
            "n_components": len(current_components),
            "raster_pixels": int(sum(int(entry["area_px"]) for entry in current_components)),
            "zero_area_components": int(sum(1 for entry in current_components if int(entry["area_px"]) == 0)),
        },
        "compare": {
            "iou_postfilter_vs_simplified": iou(post_filter, simplified),
            "iou_simplified_vs_yolo": iou(simplified, yolo_mask),
            "iou_postfilter_vs_yolo": iou(post_filter, yolo_mask),
            "iou_raw_vs_postfilter": iou(foreground, post_filter),
            "dice_simplified_vs_yolo": dice(simplified, yolo_mask),
            "precision_simplified_vs_yolo": precision,
            "recall_simplified_vs_yolo": recall,
            "area_bias_simplified_vs_yolo": float(
                (simplified.sum() - yolo_mask.sum()) / max(int(yolo_mask.sum()), 1)
            ),
            "area_bias_postfilter_vs_simplified": float(
                (post_filter.sum() - simplified.sum()) / max(int(simplified.sum()), 1)
            ),
            "boundary_mean_px": boundary["mean_boundary_distance_px"],
            "boundary_p95_px": boundary["p95_boundary_distance_px"],
            "approx_boundary_mean_px": approx_boundary["mean_boundary_distance_px"],
            "approx_boundary_p95_px": approx_boundary["p95_boundary_distance_px"],
            "postfilter_pixels": int(post_filter.sum()),
            "simplified_pixels": int(simplified.sum()),
        },
        "merge_risk": {
            "counts": risk_counts,
            "high_component_ids": high_ids,
            "per_component": [
                {
                    "component_id": int(record.component_id),
                    "area_px": int(record.area_px),
                    "border": bool(record.touches_image_border),
                    **risk,
                }
                for record, risk in zip(records8, risks)
            ],
        },
    }


def main() -> int:
    started = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tiles = aligned_tiles()
    print(f"[task6k.scan] {len(tiles)} aligned tiles", flush=True)
    with CACHE.open("w", encoding="utf-8") as handle:
        for index, tile in enumerate(tiles, start=1):
            record = scan_tile(tile)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 500 == 0:
                print(f"[task6k.scan] {index}/{len(tiles)} ({time.time() - started:.0f}s)", flush=True)
    summary = {
        "tiles": len(tiles),
        "cache": str(CACHE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "seconds": round(time.time() - started, 2),
        "path_provenance": path_provenance(),
        "converted_candidates": [str(path) for path in CONVERTED_CANDIDATES],
        "min_approx_vertices": MIN_APPROX_VERTICES,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[task6k.scan] wrote {CACHE.name} + summary in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
