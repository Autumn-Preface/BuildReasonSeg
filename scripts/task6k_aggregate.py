"""Task 6K Parts B/C/G: raw semantic component statistics, conversion loss, merge-risk heuristic.

Reads the per-tile scan cache (``artifacts/task6k/tile_scan.jsonl``) and the current component
maps, and writes:

* `evaluation/task6k_raw_component_stats.json`            (section 5)
* `evaluation/task6k_conversion_loss.json`                (section 6)
* `evaluation/task6k_merge_risk.json`                     (section 12)

The merge-risk analysis is a clearly labelled HEURISTIC: a binary semantic mask cannot reveal
physical instance identity, so no true instance recovery is claimed.
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
    MIN_CONTOUR_AREA,
    count_distribution,
    percentile_summary,
    write_json,
)

from task6k_common import CURRENT_ROOT, EVAL, aligned_tiles  # noqa: E402

CACHE = REPO_ROOT / "artifacts" / "task6k" / "tile_scan.jsonl"
MIN_MERGE_RISK_AREA = 150


def load_scan() -> list[dict]:
    with CACHE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values):
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def raw_component_stats(records: list[dict]) -> dict:
    counts8 = [record["raw_components_8"]["count"] for record in records]
    counts4 = [record["raw_components_4"]["count"] for record in records]
    areas = [area for record in records for area in record["raw_components_8"]["areas"]]
    widths = [wh[0] for record in records for wh in record["raw_components_8"]["bbox_wh"]]
    heights = [wh[1] for record in records for wh in record["raw_components_8"]["bbox_wh"]]
    border_flags = [flag for record in records for flag in record["raw_components_8"]["border_touch"]]
    foreground = [record["source"]["foreground_pixels"] for record in records]
    fractions = [record["source"]["foreground_fraction"] for record in records]
    current = [record["current"]["n_components"] for record in records]
    tiny = [area for area in areas if area < MIN_CONTOUR_AREA]
    empty_tiles = sum(1 for record in records if record["source"]["empty"])

    by_split = {}
    for split in ("train", "val", "test"):
        subset = [record for record in records if record["split"] == split]
        by_split[split] = {
            "tiles": len(subset),
            "raw_components_8": sum(record["raw_components_8"]["count"] for record in subset),
            "raw_components_4": sum(record["raw_components_4"]["count"] for record in subset),
            "current_components": sum(record["current"]["n_components"] for record in subset),
            "components_per_tile_mean": _mean([record["raw_components_8"]["count"] for record in subset]),
        }

    return {
        "_doc": (
            "Task 6K section 5. Raw 8-connected semantic components of the original WHU raster "
            "labels after the historical threshold 127 (4-connectivity is reported as a "
            "sensitivity diagnostic). TERMINOLOGY: these are semantic connected components, NOT "
            "verified physical-building instances; the source raster is binary, so mutually "
            "touching buildings are already merged here."
        ),
        "task": "6K",
        "scope": {
            "representation": "original WHU raster semantic labels (threshold 127)",
            "tiles": len(records),
            "splits": "current BuildReasonSeg train+val+test (2508/627/903)",
            "primary_connectivity": 8,
            "sensitivity_connectivity": 4,
        },
        "totals": {
            "raw_components_8": sum(counts8),
            "raw_components_4": sum(counts4),
            "current_components": sum(current),
            "empty_tiles": int(empty_tiles),
            "components_8_minus_current": sum(counts8) - sum(current),
        },
        "components_per_tile_8": percentile_summary(counts8),
        "components_per_tile_4": percentile_summary(counts4),
        "current_components_per_tile": percentile_summary(current),
        "area_px_8": percentile_summary(areas),
        "bbox_width_px_8": percentile_summary(widths),
        "bbox_height_px_8": percentile_summary(heights),
        "border_touch": {
            "components": int(sum(border_flags)),
            "rate": float(sum(border_flags) / max(len(border_flags), 1)),
        },
        "foreground_pixels_per_tile": percentile_summary(foreground),
        "foreground_fraction_per_tile": percentile_summary(fractions),
        "tiles_by_component_count_8": count_distribution(counts8, [1, 2, 5, 10, 20]),
        "small_components_below_area_50": {
            "count": len(tiny),
            "rate_of_raw_components": float(len(tiny) / max(len(areas), 1)),
            "pixels": int(sum(tiny)),
            "share_of_foreground_pixels": float(
                sum(tiny) / max(sum(foreground), 1)
            ),
            "note": (
                "These are the raw semantic components whose RASTER area is below the historical "
                "cv2.contourArea < 50 filter; the filter itself uses contourArea, so the removed "
                "set is measured directly in task6k_conversion_loss.json."
            ),
        },
        "by_split": by_split,
        "current_representation_reference": {
            "total_components": 36926,
            "source": "datasets/whu/component_manifest.json",
        },
    }


def conversion_loss(records: list[dict]) -> dict:
    total_foreground = sum(record["source"]["foreground_pixels"] for record in records)
    before = sum(record["contours"]["before_filter"] for record in records)
    removed = sum(record["contours"]["removed"] for record in records)
    kept = sum(record["contours"]["kept"] for record in records)
    removed_pixels = sum(sum(record["contours"]["removed_raster_pixels"]) for record in records)
    dropped_lt3 = sum(record["contours"]["dropped_below_three_vertices"] for record in records)

    removed_areas = [area for record in records for area in record["contours"]["removed_contour_areas"]]
    removed_raster = [px for record in records for px in record["contours"]["removed_raster_pixels"]]
    affected = [record for record in records if record["contours"]["removed"] > 0]
    removed_per_affected = [record["contours"]["removed"] for record in affected]

    hole_pixels = sum(record["topology"]["hole_pixels"] for record in records)
    components_with_holes = sum(record["topology"]["components_with_holes"] for record in records)
    tiles_with_holes = sum(1 for record in records if record["topology"]["hole_pixels"] > 0)

    approx_iou = [record["compare"]["iou_postfilter_vs_simplified"] for record in records]
    approx_boundary_mean = [record["compare"]["approx_boundary_mean_px"] for record in records]
    approx_boundary_p95 = [record["compare"]["approx_boundary_p95_px"] for record in records]
    approx_area_bias = [record["compare"]["area_bias_postfilter_vs_simplified"] for record in records]
    raw_vs_postfilter_iou = [record["compare"]["iou_raw_vs_postfilter"] for record in records]
    vertex_counts = [count for record in records for count in record["contours"]["approx_vertex_counts"]]

    return {
        "_doc": (
            "Task 6K section 6. The historical semantic -> YOLO polygon pipeline is reproduced IN "
            "MEMORY for every aligned tile and each stage's loss is measured separately: the "
            "`cv2.contourArea < 50` filter, the `RETR_EXTERNAL` topology loss (hole filling), and "
            "the `approxPolyDP(0.001 * arcLength)` simplification. No dataset was modified."
        ),
        "task": "6K",
        "pipeline_parameters": {
            "threshold": 127,
            "contour_mode": "RETR_EXTERNAL",
            "contour_approx": "CHAIN_APPROX_SIMPLE",
            "min_contour_area": MIN_CONTOUR_AREA,
            "epsilon_ratio": 0.001,
            "min_approx_vertices": 3,
            "extra_detail_vs_supplied_snippet": (
                "The historical script (mask_to_yolo.py lines 111-127) also (a) skips polygons "
                "with `len(approx) < 3` and (b) clips coordinates into [0, 1]. Both are reproduced."
            ),
        },
        "scope": {"tiles": len(records), "total_foreground_pixels": int(total_foreground)},
        "stage_1_area_filter": {
            "external_contours_before_filter": int(before),
            "removed": int(removed),
            "removed_rate": float(removed / max(before, 1)),
            "removed_area_below_three_vertices": int(dropped_lt3),
            "kept": int(kept),
            "removed_raster_pixels": int(removed_pixels),
            "removed_raster_pixel_rate_of_foreground": float(removed_pixels / max(total_foreground, 1)),
            "tiles_affected": len(affected),
            "tiles_affected_rate": float(len(affected) / max(len(records), 1)),
            "removed_per_affected_tile": percentile_summary(removed_per_affected),
            "removed_contour_area_cv2": percentile_summary(removed_areas),
            "removed_raster_area_px": percentile_summary(removed_raster),
            "note": (
                "The filter uses cv2.contourArea; raster pixel area of the same contours is "
                "reported alongside because the two differ (contourArea is the polygon area of the "
                "simplified contour chain)."
            ),
        },
        "stage_2_retr_external_topology": {
            "components_with_holes": int(components_with_holes),
            "rate_of_raw_components": float(
                components_with_holes / max(sum(record["raw_components_8"]["count"] for record in records), 1)
            ),
            "tiles_with_holes": int(tiles_with_holes),
            "hole_pixels": int(hole_pixels),
            "hole_pixel_rate_of_foreground": float(hole_pixels / max(total_foreground, 1)),
            "reconstruction_difference_pixels": int(
                sum(record["topology"]["external_minus_foreground"] for record in records)
            ),
            "interpretation": (
                "External contours fill interior holes, so these pixels are ADDED to the mask "
                "(courtyards / light wells). Measured difference between the raw semantic mask and "
                "the external-contour rasterization."
            ),
        },
        "stage_3_polygon_approximation": {
            "postfilter_vs_simplified_iou": percentile_summary(approx_iou),
            "postfilter_vs_simplified_area_bias": percentile_summary(approx_area_bias),
            "postfilter_vs_simplified_boundary_mean_px": percentile_summary(approx_boundary_mean),
            "postfilter_vs_simplified_boundary_p95_px": percentile_summary(approx_boundary_p95),
            "approx_vertices_per_polygon": percentile_summary(vertex_counts),
            "tiles_with_iou_below_0_99": sum(1 for v in approx_iou if v < 0.99),
            "tiles_with_iou_below_0_95": sum(1 for v in approx_iou if v < 0.95),
        },
        "cumulative_loss_raw_vs_postfilter": {
            "iou_raw_vs_postfilter": percentile_summary(raw_vs_postfilter_iou),
            "tiles_with_iou_below_0_99": sum(1 for v in raw_vs_postfilter_iou if v < 0.99),
            "note": (
                "Raw semantic mask vs the post-filter external-contour target: this combines the "
                "`<50` deletion (loses foreground) with the hole filling (adds foreground)."
            ),
        },
        "decomposition_summary": {
            "stage_1_filter_removed_pixels": int(removed_pixels),
            "stage_1_filter_removed_rate": float(removed_pixels / max(total_foreground, 1)),
            "stage_2_hole_pixels_added": int(hole_pixels),
            "stage_2_hole_rate": float(hole_pixels / max(total_foreground, 1)),
            "stage_3_mean_iou": _mean(approx_iou),
            "stage_3_mean_boundary_px": _mean(approx_boundary_mean),
        },
    }


def _component_map_names():
    names = []
    for split in ("train", "val", "test"):
        directory = CURRENT_ROOT / "components" / split
        if directory.is_dir():
            for path in sorted(directory.glob("*.png")):
                names.append((split, path))
    return names


def current_component_merge_risk(limit: int | None = None) -> dict:
    """The same heuristic applied to the CURRENT pseudo-instance components (component maps)."""

    kernel = np.ones((3, 3), np.uint8)
    counts = {"low": 0, "medium": 0, "high": 0}
    examples = []
    total = 0
    for index, (split, path) in enumerate(_component_map_names()):
        if limit is not None and index >= limit:
            break
        labels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if labels is None:
            continue
        for component_id in np.unique(labels):
            if int(component_id) == 0:
                continue
            mask = labels == component_id
            area = int(mask.sum())
            if area < MIN_MERGE_RISK_AREA:
                counts["low"] += 1
                total += 1
                continue
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            n_pieces, _ = cv2.connectedComponents(eroded, connectivity=8)
            pieces = max(0, int(n_pieces) - 1)
            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            if float(distance.max()) >= 2.0:
                peaks_mask = (distance >= 0.7 * float(distance.max())) & (distance >= 2.0)
                dilated = cv2.dilate(peaks_mask.astype(np.uint8), kernel, iterations=1)
                n_peaks, _ = cv2.connectedComponents(dilated, connectivity=8)
                peaks = max(0, int(n_peaks) - 1)
            else:
                peaks = 1
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_poly = float(sum(cv2.contourArea(c) for c in contours))
            hull_area = float(cv2.contourArea(cv2.convexHull(np.vstack(contours)))) if contours else 0.0
            solidity = (area_poly / hull_area) if hull_area > 0 else 1.0
            if pieces >= 2 or peaks >= 3:
                risk = "high"
            elif peaks == 2 or solidity < 0.75 or pieces == 0:
                risk = "medium"
            else:
                risk = "low"
            counts[risk] += 1
            total += 1
            if risk == "high" and len(examples) < 10:
                examples.append({"split": split, "stem": path.stem, "component_id": int(component_id), "area_px": area})
    return {"counts": counts, "total": total, "high_examples": examples}


def merge_risk(records: list[dict]) -> dict:
    counts = {"low": 0, "medium": 0, "high": 0}
    high_examples = []
    per_split = {}
    for record in records:
        for entry in record["merge_risk"]["per_component"]:
            counts[entry["risk"]] += 1
            if entry["risk"] == "high" and len(high_examples) < 15:
                high_examples.append(
                    {
                        "split": record["split"],
                        "stem": record["stem"],
                        "component_id": entry["component_id"],
                        "area_px": entry["area_px"],
                        "border": entry["border"],
                        "erosion1_pieces": entry["erosion1_pieces"],
                        "dt_peaks": entry["dt_peaks"],
                        "solidity": entry["solidity"],
                    }
                )
        bucket = per_split.setdefault(record["split"], {"low": 0, "medium": 0, "high": 0, "tiles": 0})
        bucket["tiles"] += 1
        for key in ("low", "medium", "high"):
            bucket[key] += record["merge_risk"]["counts"][key]

    dense = [record for record in records if record["raw_components_8"]["count"] >= 10]
    sparse = [record for record in records if record["raw_components_8"]["count"] < 10]

    def risk_rate(subset):
        high = sum(record["merge_risk"]["counts"]["high"] for record in subset)
        total = sum(sum(record["merge_risk"]["counts"].values()) for record in subset)
        return float(high / max(total, 1)), total

    dense_rate, dense_total = risk_rate(dense)
    sparse_rate, sparse_total = risk_rate(sparse)
    total = sum(counts.values())
    return {
        "_doc": (
            "Task 6K section 12. HEURISTIC merge-risk analysis of the raw semantic components. A "
            "binary semantic mask cannot reveal physical instance identity: this measures the "
            "sensitivity of a connected component to 1-px erosion, the number of distance-transform "
            "peaks and the solidity (area / convex-hull area). It does NOT recover true instances "
            "and must not be read as merge ground truth."
        ),
        "task": "6K",
        "heuristic_definition": {
            "high": "1-px erosion splits into >= 2 pieces, OR >= 3 distance-transform peaks",
            "medium": "exactly 2 distance-transform peaks, OR solidity < 0.75, OR the component vanishes under 1-px erosion (everywhere <= 2 px thick)",
            "low": "none of the above",
            "applied_to": "raw 8-connected semantic components with raster area >= 150 px (the frozen tiny threshold)",
            "revision_note": (
                "A first version of this heuristic classified `1-px erosion yields exactly 1 piece` "
                "as medium, which fired for 92.7% of components (the base rate) and carried no "
                "information; that clause was removed so the classes are discriminative."
            ),
        },
        "raw_components": {
            "total": total,
            "counts": counts,
            "rates": {key: float(value / max(total, 1)) for key, value in counts.items()},
            "high_examples": high_examples,
            "by_split": per_split,
        },
        "current_components": current_component_merge_risk(),
        "dense_tile_concentration": {
            "definition": "dense tile = raw 8-connected component count >= 10",
            "dense_tiles": len(dense),
            "dense_components": int(dense_total),
            "dense_high_rate": dense_rate,
            "sparse_tiles": len(sparse),
            "sparse_components": int(sparse_total),
            "sparse_high_rate": sparse_rate,
            "risk_concentrates_in_dense_tiles": bool(dense_rate > sparse_rate),
        },
    }


def main() -> int:
    started = time.time()
    records = load_scan()
    print(f"[task6k.aggregate] {len(records)} tile records", flush=True)
    write_json(EVAL / "task6k_raw_component_stats.json", raw_component_stats(records))
    write_json(EVAL / "task6k_conversion_loss.json", conversion_loss(records))
    write_json(EVAL / "task6k_merge_risk.json", merge_risk(records))
    print(f"[task6k.aggregate] wrote raw_component_stats + conversion_loss + merge_risk in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
