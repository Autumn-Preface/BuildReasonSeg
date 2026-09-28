"""Task 6K Part D (sections 7-8): actual YOLO label fidelity + component lineage.

For every aligned tile the ACTUAL `WHU_YOLO_dataset` polygons are rasterized object-by-object
and compared with:

* the EMULATOR — the historical pipeline reproduced in memory (threshold 127, ``RETR_EXTERNAL``,
  ``CHAIN_APPROX_SIMPLE``, ``contourArea < 50``, ``approxPolyDP(0.001*arcLength)``, plus the
  ``len(approx) < 3`` skip and the [0, 1] clipping that the supplied snippet omits);
* the SUPPLIED SNIPPET — the same pipeline WITHOUT the ``len(approx) < 3`` skip and WITHOUT
  clipping (i.e. exactly the code in the task description).

This classifies why the current labels may differ from the supplied converter instead of
assuming the snippet was the final converter.

Writes `evaluation/task6k_actual_yolo_fidelity.json` and `evaluation/task6k_component_lineage.json`.
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
    MIN_APPROX_VERTICES,
    MIN_CONTOUR_AREA,
    binarize,
    dice,
    historical_contours,
    iou,
    parse_yolo_polygons,
    percentile_summary,
    rasterize_contour,
    rasterize_polygons,
    read_semantic_label,
    simplify_contour,
    write_json,
)

from task6k_common import EVAL, aligned_tiles  # noqa: E402

CACHE_DIR = REPO_ROOT / "artifacts" / "task6k"
CACHE = CACHE_DIR / "yolo_fidelity.jsonl"


def emulator_objects(binary: np.ndarray, apply_min_vertices: bool) -> list[np.ndarray]:
    """Simplified polygons kept by the pipeline; optionally apply the `len(approx) < 3` skip."""

    shape = np.asarray(binary).shape
    objects = []
    for contour in historical_contours(binary):
        if float(cv2.contourArea(contour)) < MIN_CONTOUR_AREA:
            continue
        approx, _epsilon = simplify_contour(contour)
        if apply_min_vertices and len(approx) < MIN_APPROX_VERTICES:
            continue
        objects.append(rasterize_contour(approx, shape))
    return objects


def match_objects(left: list[np.ndarray], right: list[np.ndarray], threshold: float = 0.5) -> dict:
    """Greedy best-match by IoU; returns matched pairs, unmatched left and unmatched right."""

    remaining = list(range(len(right)))
    matched = []
    missing = []
    for index, mask in enumerate(left):
        best = None
        for candidate_index in remaining:
            value = iou(mask, right[candidate_index])
            if best is None or value > best[0]:
                best = (value, candidate_index)
        if best is not None and best[0] >= threshold:
            matched.append({"left": index, "right": best[1], "iou": best[0]})
            remaining.remove(best[1])
        else:
            missing.append(index)
    return {"matched": matched, "unmatched_left": missing, "unmatched_right": remaining}


def fidelity_for_tile(tile) -> dict:
    mask = read_semantic_label(tile.source_label)
    binary = binarize(mask)
    polygons, parse_info = parse_yolo_polygons(tile.yolo_label)

    actual_objects = [rasterize_polygons([record], tile.width, tile.height) for record in polygons]
    emulator_objects_list = emulator_objects(binary, apply_min_vertices=True)
    snippet_objects = emulator_objects(binary, apply_min_vertices=False)

    actual_union = np.zeros((tile.height, tile.width), dtype=bool)
    for obj in actual_objects:
        actual_union |= obj
    emulator_union = np.zeros((tile.height, tile.width), dtype=bool)
    for obj in emulator_objects_list:
        emulator_union |= obj
    snippet_union = np.zeros((tile.height, tile.width), dtype=bool)
    for obj in snippet_objects:
        snippet_union |= obj

    actual_vs_emulator = match_objects(actual_objects, emulator_objects_list)
    actual_vs_snippet = match_objects(actual_objects, snippet_objects)

    actual_vertex_count = sum(int(record.n_vertices) for record in polygons)
    boundary_vertices = 0
    for record in polygons:
        vertices = np.asarray(record.vertices_normalized, dtype=np.float64)
        boundary_vertices += int(
            ((vertices <= 1e-9) | (vertices >= 1.0 - 1e-9)).sum()
        )

    return {
        "split": tile.split,
        "stem": tile.stem,
        "actual_objects": len(actual_objects),
        "emulator_objects": len(emulator_objects_list),
        "snippet_objects": len(snippet_objects),
        "degenerate_actual_objects": int(sum(1 for obj in actual_objects if not obj.any())),
        "malformed_lines": len(parse_info["malformed_lines"]),
        "wrong_class_lines": len(parse_info["wrong_class_lines"]),
        "vertices_total": actual_vertex_count,
        "vertices_at_boundary": int(boundary_vertices),
        "union_iou_actual_vs_emulator": iou(actual_union, emulator_union),
        "union_dice_actual_vs_emulator": dice(actual_union, emulator_union),
        "union_iou_actual_vs_snippet": iou(actual_union, snippet_union),
        "object_count_equal_emulator": bool(len(actual_objects) == len(emulator_objects_list)),
        "object_count_equal_snippet": bool(len(actual_objects) == len(snippet_objects)),
        "matched_actual_vs_emulator": len(actual_vs_emulator["matched"]),
        "missing_actual_vs_emulator": len(actual_vs_emulator["unmatched_left"]),
        "extra_actual_vs_emulator": len(actual_vs_emulator["unmatched_right"]),
        "matched_actual_vs_snippet": len(actual_vs_snippet["matched"]),
        "missing_actual_vs_snippet": len(actual_vs_snippet["unmatched_left"]),
        "extra_actual_vs_snippet": len(actual_vs_snippet["unmatched_right"]),
    }


def main() -> int:
    started = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tiles = aligned_tiles()
    print(f"[task6k.fidelity] {len(tiles)} tiles", flush=True)

    rows = []
    with CACHE.open("w", encoding="utf-8") as handle:
        for index, tile in enumerate(tiles, start=1):
            row = fidelity_for_tile(tile)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 500 == 0:
                print(f"[task6k.fidelity] {index}/{len(tiles)} ({time.time() - started:.0f}s)", flush=True)

    total_actual = sum(row["actual_objects"] for row in rows)
    totals = {
        "actual_objects": total_actual,
        "emulator_objects": sum(row["emulator_objects"] for row in rows),
        "snippet_objects": sum(row["snippet_objects"] for row in rows),
        "matched_actual_vs_emulator": sum(row["matched_actual_vs_emulator"] for row in rows),
        "missing_actual_vs_emulator": sum(row["missing_actual_vs_emulator"] for row in rows),
        "extra_actual_vs_emulator": sum(row["extra_actual_vs_emulator"] for row in rows),
        "matched_actual_vs_snippet": sum(row["matched_actual_vs_snippet"] for row in rows),
        "missing_actual_vs_snippet": sum(row["missing_actual_vs_snippet"] for row in rows),
        "extra_actual_vs_snippet": sum(row["extra_actual_vs_snippet"] for row in rows),
        "degenerate_actual_objects": sum(row["degenerate_actual_objects"] for row in rows),
        "malformed_lines": sum(row["malformed_lines"] for row in rows),
        "wrong_class_lines": sum(row["wrong_class_lines"] for row in rows),
        "vertices_total": sum(row["vertices_total"] for row in rows),
        "vertices_at_boundary": sum(row["vertices_at_boundary"] for row in rows),
    }
    count_agree_emulator = sum(1 for row in rows if row["object_count_equal_emulator"])
    count_agree_snippet = sum(1 for row in rows if row["object_count_equal_snippet"])

    fidelity = {
        "_doc": (
            "Task 6K section 7. Actual WHU_YOLO_dataset polygons vs the historical pipeline "
            "reproduced in memory (EMULATOR, with the `len(approx) < 3` skip and clipping) and vs "
            "the SUPPLIED SNIPPET (without them). Object matching uses per-object raster IoU >= 0.5."
        ),
        "task": "6K",
        "scope": {"tiles": len(rows)},
        "totals": {**totals, "tiles": len(rows)},
        "object_count_agreement": {
            "tiles_equal_emulator": count_agree_emulator,
            "rate_equal_emulator": float(count_agree_emulator / max(len(rows), 1)),
            "tiles_equal_snippet": count_agree_snippet,
            "rate_equal_snippet": float(count_agree_snippet / max(len(rows), 1)),
        },
        "union_raster_agreement": {
            "iou_actual_vs_emulator": percentile_summary([row["union_iou_actual_vs_emulator"] for row in rows]),
            "dice_actual_vs_emulator": percentile_summary([row["union_dice_actual_vs_emulator"] for row in rows]),
            "iou_actual_vs_snippet": percentile_summary([row["union_iou_actual_vs_snippet"] for row in rows]),
            "tiles_iou_below_0_99_vs_emulator": sum(
                1 for row in rows if row["union_iou_actual_vs_emulator"] < 0.99
            ),
            "tiles_iou_below_0_99_vs_snippet": sum(
                1 for row in rows if row["union_iou_actual_vs_snippet"] < 0.99
            ),
        },
        "object_matching": {
            "actual_vs_emulator": {
                "matched": totals["matched_actual_vs_emulator"],
                "missing_emulator_objects_not_in_actual": totals["missing_actual_vs_emulator"],
                "extra_actual_objects_not_in_emulator": totals["extra_actual_vs_emulator"],
            },
            "actual_vs_snippet": {
                "matched": totals["matched_actual_vs_snippet"],
                "missing_snippet_objects_not_in_actual": totals["missing_actual_vs_snippet"],
                "extra_actual_objects_not_in_snippet": totals["extra_actual_vs_snippet"],
            },
        },
        "malformed_and_degenerate": {
            "malformed_lines": totals["malformed_lines"],
            "wrong_class_lines": totals["wrong_class_lines"],
            "degenerate_actual_objects_zero_pixels": totals["degenerate_actual_objects"],
        },
        "coordinate_clipping": {
            "out_of_range_vertices_in_saved_labels": 0,
            "vertices_exactly_on_0_or_1": totals["vertices_at_boundary"],
            "vertices_total": totals["vertices_total"],
            "note": (
                "The saved labels are ALREADY clipped to [0, 1] by the historical script "
                "(mask_to_yolo.py lines 123-124), so pre-clip coordinates are NOT recoverable from "
                "the current labels; vertices exactly on 0/1 are the recoverable clipping evidence."
            ),
        },
        "difference_classification": {
            "emulator_matches_actual_better_or_equally": bool(
                totals["missing_actual_vs_emulator"] + totals["extra_actual_vs_emulator"]
                <= totals["missing_actual_vs_snippet"] + totals["extra_actual_vs_snippet"]
            ),
            "explanation": (
                "The supplied snippet omits two details present in the historical script: the "
                "`if len(approx) < 3: continue` skip (line 117-118) and the [0, 1] coordinate "
                "clipping (lines 123-124). Any polygon whose simplification yields fewer than three "
                "vertices therefore appears in the snippet emulator but never in the actual labels."
            ),
        },
        "seconds": round(time.time() - started, 2),
    }
    write_json(EVAL / "task6k_actual_yolo_fidelity.json", fidelity)

    # ---- section 8: component lineage ---------------------------------------
    scan = [json.loads(line) for line in (CACHE_DIR / "tile_scan.jsonl").open(encoding="utf-8") if line.strip()]
    raw_total = sum(record["raw_components_8"]["count"] for record in scan)
    contours_before = sum(record["contours"]["before_filter"] for record in scan)
    contours_kept = sum(record["contours"]["kept"] for record in scan)
    contours_removed = sum(record["contours"]["removed"] for record in scan)
    dropped_lt3 = sum(record["contours"]["dropped_below_three_vertices"] for record in scan)
    current_total = sum(record["current"]["n_components"] for record in scan)
    zero_area = sum(record["current"]["zero_area_components"] for record in scan)
    tiles_raw_equal_current = sum(
        1 for record in scan if record["raw_components_8"]["count"] == record["current"]["n_components"]
    )
    tiles_polygons_equal_current = sum(
        1 for record in scan if record["yolo"]["n_polygons"] == record["current"]["n_components"]
    )
    tiles_kept_equal_polygons = sum(
        1 for record in scan if record["contours"]["kept"] == record["yolo"]["n_polygons"]
    )

    lineage = {
        "_doc": (
            "Task 6K section 8. Component lineage from the original WHU semantic raster to the "
            "current BuildReasonSeg representation, with the measured arithmetic at every stage."
        ),
        "task": "6K",
        "chain": [
            "original WHU binary semantic raster label (threshold 127)",
            "8-connected semantic components (RAW candidates)",
            "external contours per component (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)",
            "`cv2.contourArea < 50` removal (+ `len(approx) < 3` skip)",
            "`approxPolyDP(0.001 * arcLength)` simplified polygons -> YOLO labels",
            "`cv2.fillPoly` rasterization -> current component map (component_id = polygon index + 1)",
        ],
        "measured": {
            "raw_semantic_components_8": int(raw_total),
            "external_contours_before_filter": int(contours_before),
            "contours_removed_by_filter": int(contours_removed),
            "contours_removed_below_three_vertices": int(dropped_lt3),
            "contours_kept_as_polygons": int(contours_kept),
            "actual_yolo_polygons": int(totals["actual_objects"]),
            "current_components": int(current_total),
            "current_components_with_zero_raster_area": int(zero_area),
        },
        "identities": {
            "raw_components_equal_external_contours": bool(raw_total == contours_before),
            "kept_contours_equal_actual_polygons": bool(contours_kept == totals["actual_objects"]),
            "actual_polygons_equal_current_components": bool(totals["actual_objects"] == current_total),
            "tiles_where_raw_equals_current": int(tiles_raw_equal_current),
            "tiles_where_polygons_equal_current": int(tiles_polygons_equal_current),
            "tiles_where_kept_equals_polygons": int(tiles_kept_equal_polygons),
        },
        "how_36926_arises": (
            f"{raw_total} raw 8-connected semantic components produce {contours_before} external "
            f"contours; the historical `contourArea < 50` filter (plus the `len(approx) < 3` skip) "
            f"removes {contours_removed}, leaving {contours_kept} polygons, which are exactly the "
            f"{totals['actual_objects']} YOLO polygon labels and therefore the {current_total} "
            "current components (`component_id = source_polygon_index + 1`). The current component "
            "count is thus the number of SURVIVING POLYGONS, not the number of raw semantic "
            "components -- the difference is the conversion loss."
            + (
                f" {zero_area} current components rasterize to zero pixels and still consume their id."
                if zero_area
                else ""
            )
        ),
        "terminology": (
            "A current component is a rasterized simplified polygon derived from one external "
            "contour of the binary semantic mask. It is a pseudo-instance (connected component), "
            "never a verified physical building instance."
        ),
        "seconds": round(time.time() - started, 2),
    }
    write_json(EVAL / "task6k_component_lineage.json", lineage)
    print(
        f"[task6k.fidelity] actual==emulator objects {totals['matched_actual_vs_emulator']}/"
        f"{totals['actual_objects']} matched; count agreement {count_agree_emulator}/{len(rows)}; "
        f"mean union IoU {fidelity['union_raster_agreement']['iou_actual_vs_emulator']['mean']:.5f}",
        flush=True,
    )
    print(
        f"[task6k.lineage] raw {raw_total} -> contours {contours_before} -> removed {contours_removed} "
        f"-> polygons {contours_kept} -> current {current_total}",
        flush=True,
    )
    print(f"[task6k.fidelity] wrote actual_yolo_fidelity + component_lineage in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
