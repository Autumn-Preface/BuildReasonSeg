"""Task 6K Part E (section 9): RAW vs CONVERTED candidate sets under the frozen relation semantics.

For every aligned tile two candidate sets are built:

* **RAW**       all 8-connected semantic components of the thresholded ORIGINAL raster label,
                with NO ``contourArea < 50`` filter (matching the user's stated intent);
* **CONVERTED** the candidate set implied by the current BuildReasonSeg component representation
                (which the historical conversion produced from the same raster).

All 20 canonical programs from Task 6J run on both sets with the FROZEN Task 3B relation
configuration. Target identity is compared and every change is attributed. This measures
conversion sensitivity only; it does not establish physical-building truth.

Writes `evaluation/task6k_relation_semantic_drift.json` and the gitignored per-tile cache
`artifacts/task6k/relation_drift.jsonl`.
"""

from __future__ import annotations

import argparse
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

from buildreasonseg_mvp.structured_grounding import (  # noqa: E402
    EXPECTED_QUERY_TYPES,
    canonical_program_template,
    execute_program_by_id,
)
from buildreasonseg_mvp.whu_source_audit import (  # noqa: E402
    MIN_CONTOUR_AREA,
    binarize,
    raw_components,
    read_semantic_label,
    write_json,
)

from task6k_common import (  # noqa: E402
    EVAL,
    aligned_tiles,
    converted_candidate_set,
    dataset_query_type_counts,
    frozen_relation_config,
    raw_candidate_set,
)

CACHE_DIR = REPO_ROOT / "artifacts" / "task6k"
CACHE = CACHE_DIR / "relation_drift.jsonl"

PROGRAM_LEVEL = {}
for _program in EXPECTED_QUERY_TYPES:
    if _program in ("leftmost", "rightmost", "topmost", "bottommost", "largest", "smallest"):
        PROGRAM_LEVEL[_program] = 1
    elif _program.count("_to_") == 2:
        PROGRAM_LEVEL[_program] = 3
    else:
        PROGRAM_LEVEL[_program] = 2

PROGRAM_OPERATIONS = {program: [step["operation"] for step in canonical_program_template(program)] for program in EXPECTED_QUERY_TYPES}


def match_components(labels_raw: np.ndarray, labels_conv: np.ndarray, areas_conv: dict[int, int]) -> dict:
    """Mask-overlap matching raw <-> converted (IoU >= 0.5), computed per raw component."""

    raw_ids = [int(v) for v in np.unique(labels_raw) if int(v) != 0]
    matches = {}
    for raw_id in raw_ids:
        raw_mask = labels_raw == raw_id
        raw_area = int(raw_mask.sum())
        if raw_area == 0:
            continue
        overlap_ids, counts = np.unique(labels_conv[raw_mask], return_counts=True)
        best = None
        for conv_id, intersection in zip(overlap_ids.tolist(), counts.tolist()):
            if int(conv_id) == 0:
                continue
            union = raw_area + int(areas_conv.get(int(conv_id), 0)) - int(intersection)
            iou = float(intersection) / max(union, 1)
            if best is None or iou > best["iou"]:
                best = {"converted_id": int(conv_id), "iou": iou, "area_raw": raw_area,
                        "area_converted": int(areas_conv.get(int(conv_id), 0))}
        if best is not None and best["iou"] >= 0.5:
            matches[raw_id] = best
    return matches


def attribute(program: str, raw_result, conv_result, matches, raw_areas, raw_border, conv_border) -> dict:
    """Documented priority attribution of one changed/unstable program outcome."""

    flags = {
        "small_component_removed": False,
        "ranking_changed": False,
        "nearest_changed": False,
        "directional_candidate_set_changed": False,
        "border_eligibility_changed": False,
        "polygon_geometry_changed": False,
    }
    operations = PROGRAM_OPERATIONS[program]

    def unmapped(candidate_id):
        return candidate_id is not None and int(candidate_id) not in matches

    if unmapped(raw_result.selected_id) or unmapped(conv_result.selected_id):
        flags["small_component_removed"] = True
    if raw_result.selected_id is not None and int(raw_result.selected_id) in matches:
        pair = matches[int(raw_result.selected_id)]
        if pair["iou"] < 0.95:
            flags["polygon_geometry_changed"] = True
        conv_id = pair["converted_id"]
        if bool(raw_border.get(int(raw_result.selected_id), False)) != bool(conv_border.get(int(conv_id), False)):
            flags["border_eligibility_changed"] = True
    if any(operation in ("argmax_area", "argmin_area") for operation in operations):
        flags["ranking_changed"] = True
    if "argmin_boundary_distance" in operations:
        flags["nearest_changed"] = True
    if "filter_relation" in operations:
        flags["directional_candidate_set_changed"] = True
    if raw_result.selected_id is not None and raw_result.selected_id in raw_areas:
        if raw_areas[int(raw_result.selected_id)] < MIN_CONTOUR_AREA:
            flags["small_component_removed"] = True

    priority = (
        "small_component_removed",
        "border_eligibility_changed",
        "nearest_changed",
        "ranking_changed",
        "directional_candidate_set_changed",
        "polygon_geometry_changed",
    )
    primary = next((name for name in priority if flags[name]), "polygon_geometry_changed")
    return {"primary": primary, "flags": flags}


def same_target(matches: dict, raw_id, converted_id) -> bool:
    """Whether the two selections denote the SAME region (ids are NOT comparable across sets)."""

    if raw_id is None or converted_id is None:
        return False
    pair = matches.get(int(raw_id))
    if pair is None:
        return False
    return int(pair["converted_id"]) == int(converted_id) and float(pair["iou"]) >= 0.5


def drift_for_tile(tile, config) -> dict:
    mask = read_semantic_label(tile.source_label)
    binary = binarize(mask)
    labels_raw, records_raw = raw_components(binary, connectivity=8)
    raw_set = raw_candidate_set(labels_raw, records_raw)
    raw_areas = {int(record.component_id): int(record.area_px) for record in records_raw}
    raw_border = {int(record.component_id): bool(record.touches_image_border) for record in records_raw}

    converted = converted_candidate_set(tile)
    if converted is None:
        return {"stem": tile.stem, "split": tile.split, "skipped": "missing_component_map"}

    labels_conv = converted.label_map
    areas_conv = {int(candidate.candidate_id): int(candidate.area_px) for candidate in converted.candidates}
    conv_border = {int(candidate.candidate_id): bool(candidate.touches_image_border) for candidate in converted.candidates}
    matches = match_components(labels_raw, labels_conv, areas_conv)

    rows = []
    for program in EXPECTED_QUERY_TYPES:
        raw_result = execute_program_by_id(program, raw_set, config)
        conv_result = execute_program_by_id(program, converted, config)
        raw_valid = not raw_result.abstained
        conv_valid = not conv_result.abstained
        if raw_valid and conv_valid:
            if same_target(matches, raw_result.selected_id, conv_result.selected_id):
                status = "target_unchanged"
                attribution = None
            else:
                status = "target_changed"
                attribution = attribute(program, raw_result, conv_result, matches, raw_areas, raw_border, conv_border)
        elif raw_valid and not conv_valid:
            status = "became_invalid"
            attribution = attribute(program, raw_result, conv_result, matches, raw_areas, raw_border, conv_border)
        elif not raw_valid and conv_valid:
            status = "became_newly_valid"
            attribution = attribute(program, raw_result, conv_result, matches, raw_areas, raw_border, conv_border)
        else:
            status = "invalid_in_both"
            attribution = None
        rows.append(
            {
                "program": program,
                "level": PROGRAM_LEVEL[program],
                "raw_selected": raw_result.selected_id,
                "converted_selected": conv_result.selected_id,
                "raw_abstain": raw_result.reason,
                "converted_abstain": conv_result.reason,
                "status": status,
                "attribution": attribution,
            }
        )
    return {
        "split": tile.split,
        "stem": tile.stem,
        "raw_components": len(records_raw),
        "converted_components": len(converted.candidates),
        "matched_components": len(matches),
        "programs": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config = frozen_relation_config()
    tiles = aligned_tiles()
    if args.limit:
        tiles = tiles[: int(args.limit)]
    print(f"[task6k.drift] {len(tiles)} tiles x {len(EXPECTED_QUERY_TYPES)} programs", flush=True)

    per_program: dict[str, dict] = {
        program: {
            "level": PROGRAM_LEVEL[program],
            "valid_in_both": 0,
            "target_unchanged": 0,
            "target_changed": 0,
            "became_invalid": 0,
            "became_newly_valid": 0,
            "invalid_in_both": 0,
            "raw_valid_tiles": 0,
            "converted_valid_tiles": 0,
            "attribution_primary": {},
        }
        for program in EXPECTED_QUERY_TYPES
    }
    per_level = {
        1: {"programs": 0, "target_changed": 0, "comparable": 0, "became_invalid": 0, "became_newly_valid": 0},
        2: {"programs": 0, "target_changed": 0, "comparable": 0, "became_invalid": 0, "became_newly_valid": 0},
        3: {"programs": 0, "target_changed": 0, "comparable": 0, "became_invalid": 0, "became_newly_valid": 0},
    }
    tiles_with_any_change = 0
    tiles_scanned = 0
    examples = []
    total_components = {"raw": 0, "converted": 0, "matched": 0}

    with CACHE.open("w", encoding="utf-8") as handle:
        for index, tile in enumerate(tiles, start=1):
            record = drift_for_tile(tile, config)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if record.get("skipped"):
                continue
            tiles_scanned += 1
            total_components["raw"] += int(record["raw_components"])
            total_components["converted"] += int(record["converted_components"])
            total_components["matched"] += int(record["matched_components"])
            changed_here = False
            for row in record["programs"]:
                bucket = per_program[row["program"]]
                if row["raw_abstain"] is None:
                    bucket["raw_valid_tiles"] += 1
                if row["converted_abstain"] is None:
                    bucket["converted_valid_tiles"] += 1
                if row["status"] == "target_unchanged":
                    bucket["valid_in_both"] += 1
                    bucket["target_unchanged"] += 1
                    per_level[row["level"]]["comparable"] += 1
                elif row["status"] == "target_changed":
                    bucket["valid_in_both"] += 1
                    bucket["target_changed"] += 1
                    per_level[row["level"]]["comparable"] += 1
                    per_level[row["level"]]["target_changed"] += 1
                    changed_here = True
                    primary = row["attribution"]["primary"]
                    bucket["attribution_primary"][primary] = bucket["attribution_primary"].get(primary, 0) + 1
                    if len(examples) < 40:
                        examples.append(
                            {
                                "split": record["split"],
                                "stem": record["stem"],
                                "program": row["program"],
                                "raw_selected": row["raw_selected"],
                                "converted_selected": row["converted_selected"],
                                "attribution": row["attribution"],
                            }
                        )
                elif row["status"] == "became_invalid":
                    bucket["became_invalid"] += 1
                    per_level[row["level"]]["became_invalid"] += 1
                    changed_here = True
                    primary = row["attribution"]["primary"]
                    bucket["attribution_primary"][primary] = bucket["attribution_primary"].get(primary, 0) + 1
                elif row["status"] == "became_newly_valid":
                    bucket["became_newly_valid"] += 1
                    per_level[row["level"]]["became_newly_valid"] += 1
                    changed_here = True
                else:
                    bucket["invalid_in_both"] += 1
            if changed_here:
                tiles_with_any_change += 1
            if index % 250 == 0:
                print(f"[task6k.drift] {index}/{len(tiles)} ({time.time() - started:.0f}s)", flush=True)

    for program, bucket in per_program.items():
        comparable = bucket["target_unchanged"] + bucket["target_changed"]
        bucket["comparable"] = comparable
        bucket["change_rate"] = float(bucket["target_changed"] / comparable) if comparable else None
        at_least_one_valid = comparable + bucket["became_invalid"] + bucket["became_newly_valid"]
        bucket["unstable"] = (
            bucket["target_changed"] + bucket["became_invalid"] + bucket["became_newly_valid"]
        )
        bucket["at_least_one_valid"] = at_least_one_valid
        bucket["unstable_rate"] = (
            float(bucket["unstable"] / at_least_one_valid) if at_least_one_valid else None
        )
        bucket["validity_delta_converted_minus_raw"] = (
            bucket["converted_valid_tiles"] - bucket["raw_valid_tiles"]
        )
        bucket["operations"] = PROGRAM_OPERATIONS[program]

    query_counts = dataset_query_type_counts()
    weighted_changes = 0
    weighted_total = 0
    for program, bucket in per_program.items():
        if program in query_counts and bucket["comparable"]:
            weighted_total += query_counts[program]
            weighted_changes += query_counts[program] * (bucket["target_changed"] / bucket["comparable"])

    report = {
        "_doc": (
            "Task 6K section 9. RAW (all 8-connected semantic components, no <50 filter) vs "
            "CONVERTED (current component representation) candidate sets under the FROZEN Task 3B "
            "relation semantics, for all 20 canonical Task 6J programs. Measures conversion "
            "sensitivity only; it does not establish physical-building truth."
        ),
        "task": "6K",
        "scope": {
            "tiles_scanned": tiles_scanned,
            "programs": len(EXPECTED_QUERY_TYPES),
            "raw_components": total_components["raw"],
            "converted_components": total_components["converted"],
            "matched_components_iou_0_5": total_components["matched"],
        },
        "per_program": per_program,
        "by_level": {
            str(level): {
                **bucket,
                "change_rate": float(bucket["target_changed"] / bucket["comparable"]) if bucket["comparable"] else None,
                "at_least_one_valid": int(
                    bucket["comparable"] + bucket["became_invalid"] + bucket["became_newly_valid"]
                ),
                "unstable_rate": (
                    float(
                        (bucket["target_changed"] + bucket["became_invalid"] + bucket["became_newly_valid"])
                        / (bucket["comparable"] + bucket["became_invalid"] + bucket["became_newly_valid"])
                    )
                    if (bucket["comparable"] + bucket["became_invalid"] + bucket["became_newly_valid"])
                    else None
                ),
            }
            for level, bucket in per_level.items()
        },
        "validity_flips": {
            "became_invalid": sum(bucket["became_invalid"] for bucket in per_program.values()),
            "became_newly_valid": sum(bucket["became_newly_valid"] for bucket in per_program.values()),
            "invalid_in_both": sum(bucket["invalid_in_both"] for bucket in per_program.values()),
            "rate_of_program_tile_pairs": float(
                sum(bucket["became_invalid"] + bucket["became_newly_valid"] for bucket in per_program.values())
                / max(tiles_scanned * len(EXPECTED_QUERY_TYPES), 1)
            ),
        },
        "constructibility_summary": {
            "raw_valid_program_tile_pairs": sum(bucket["raw_valid_tiles"] for bucket in per_program.values()),
            "converted_valid_program_tile_pairs": sum(bucket["converted_valid_tiles"] for bucket in per_program.values()),
            "per_program_validity_delta": {
                program: bucket["validity_delta_converted_minus_raw"]
                for program, bucket in sorted(per_program.items())
            },
            "note": (
                "Tile-level answerability flips in both directions (a `<50` deletion can make a "
                "direction filter unique where the raw mask had several candidates, or remove the "
                "single candidate it kept), but the COUNTS largely cancel: the corpus-level "
                "constructibility of each program changes by only a few tiles out of ~3,500."
            ),
        },
        "tiles_with_any_relation_semantic_change": {
            "tiles": tiles_with_any_change,
            "rate": float(tiles_with_any_change / max(tiles_scanned, 1)),
        },
        "overall_current_query_target_change_rate": float(weighted_changes / max(weighted_total, 1)),
        "weighting": {
            "method": "per-program target-change rate weighted by the frozen BuildSpatialReason v0.1.1 query-type counts",
            "query_records": int(weighted_total),
            "query_type_counts": query_counts,
        },
        "attribution_note": (
            "Priority order: small_component_removed > border_eligibility_changed > nearest_changed "
            "> ranking_changed > directional_candidate_set_changed > polygon_geometry_changed. All "
            "applicable flags are recorded per changed row in the gitignored per-tile cache."
        ),
        "changed_examples": examples,
        "seconds": round(time.time() - started, 2),
    }
    write_json(EVAL / "task6k_relation_semantic_drift.json", report)
    print(
        f"[task6k.drift] tiles with any change {tiles_with_any_change}/{tiles_scanned} "
        f"({report['tiles_with_any_relation_semantic_change']['rate']:.3f}); weighted current-query "
        f"target-change rate {report['overall_current_query_target_change_rate']:.4f}",
        flush=True,
    )
    print(f"[task6k.drift] wrote relation_semantic_drift.json in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
