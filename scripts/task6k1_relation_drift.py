"""Task 6K.1 Part F (section 10): VECTOR vs PSEUDO relation-answer drift.

The same 20 canonical programs from Task 6J are executed on two candidate sets per corpus tile:

* **VECTOR** ---the native manually delineated building instances (true instance identity);
* **PSEUDO** ---the current BuildReasonSeg component representation (rasterized historical polygons,
  i.e. connected components of the semantic raster).

Targets are compared through mask-overlap matching (ids are not comparable across sets), every
change is attributed, and the result is weighted by the frozen BuildSpatialReason v0.1.1 query mix.

Writes `evaluation/task6k1_vector_vs_pseudo_relation_drift.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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

from task6k1_common import (  # noqa: E402
    EVAL,
    candidate_set_from_instance_view,
    load_instance_view,
    match_components_by_overlap,
    write_json,
)

OUT = EVAL / "task6k1_vector_vs_pseudo_relation_drift.json"
CACHE = REPO_ROOT / "artifacts" / "task6k1" / "relation_drift_vector_pseudo.jsonl"

PROGRAM_LEVEL = {}
for _program in EXPECTED_QUERY_TYPES:
    if _program in ("leftmost", "rightmost", "topmost", "bottommost", "largest", "smallest"):
        PROGRAM_LEVEL[_program] = 1
    elif _program.count("_to_") == 2:
        PROGRAM_LEVEL[_program] = 3
    else:
        PROGRAM_LEVEL[_program] = 2
PROGRAM_OPERATIONS = {p: [s["operation"] for s in canonical_program_template(p)] for p in EXPECTED_QUERY_TYPES}


def attribute(program: str, vector_result, pseudo_result, matches, context: dict) -> dict:
    """Section 10 attribution classes for one changed relation answer."""

    flags = {
        "one_vector_to_multiple_pseudo": bool(context.get("spans_multiple_pseudo")),
        "below_50_removal": False,
        "border_clipping": False,
        "multiple_vector_merged_into_one_pseudo": bool(context.get("pseudo_selection_merges_vectors")),
        "eligibility_ranking_consequences": False,
        "geometry_approximation": False,
        "vector_target_missing_in_pseudo": False,
    }
    operations = PROGRAM_OPERATIONS[program]
    if vector_result.selected_id is not None:
        pair = matches.get(int(vector_result.selected_id))
        if pair is None:
            flags["vector_target_missing_in_pseudo"] = True
            if int(context.get("vector_target_area_px") or 0) < 50:
                flags["below_50_removal"] = True
            if context.get("vector_target_border") or context.get("vector_target_clipped"):
                flags["border_clipping"] = True
        elif pair["iou"] < 0.95:
            flags["geometry_approximation"] = True
    if any(operation in ("argmax_area", "argmin_area") for operation in operations):
        flags["eligibility_ranking_consequences"] = True
    if "argmin_boundary_distance" in operations:
        flags["eligibility_ranking_consequences"] = True
    if "filter_relation" in operations:
        flags["eligibility_ranking_consequences"] = True
    priority = (
        "one_vector_to_multiple_pseudo",
        "below_50_removal",
        "border_clipping",
        "multiple_vector_merged_into_one_pseudo",
        "eligibility_ranking_consequences",
        "geometry_approximation",
        "vector_target_missing_in_pseudo",
    )
    primary = next((name for name in priority if flags[name]), "vector_target_missing_in_pseudo")
    return {"primary": primary, "flags": flags}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    from task6k_common import aligned_tiles, converted_candidate_set, dataset_query_type_counts, frozen_relation_config

    config = frozen_relation_config()
    query_counts = dataset_query_type_counts()
    tiles = aligned_tiles()
    if args.limit:
        tiles = tiles[: int(args.limit)]
    print(f"[task6k1.drift] {len(tiles)} tiles x {len(EXPECTED_QUERY_TYPES)} programs", flush=True)

    per_program = {
        program: {
            "level": PROGRAM_LEVEL[program],
            "target_unchanged": 0, "target_changed": 0,
            "became_invalid": 0, "became_newly_valid": 0, "invalid_in_both": 0,
            "vector_valid_tiles": 0, "pseudo_valid_tiles": 0,
            "attribution_primary": {}, "operations": PROGRAM_OPERATIONS[program],
        }
        for program in EXPECTED_QUERY_TYPES
    }
    per_level = {level: {"comparable": 0, "target_changed": 0, "became_invalid": 0, "became_newly_valid": 0}
                 for level in (1, 2, 3)}
    tiles_with_any_change = 0
    tiles_scanned = 0
    totals = {"vector": 0, "pseudo": 0, "matched": 0, "vector_missing": 0, "merged_pseudo": 0}
    examples = []
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    with CACHE.open("w", encoding="utf-8") as handle:
        for position, tile in enumerate(tiles, start=1):
            view = load_instance_view(tile.stem)
            pseudo = converted_candidate_set(tile)
            if view is None or pseudo is None:
                continue
            vector_set = candidate_set_from_instance_view(view, source="vector")
            vector_labels = vector_set.label_map
            pseudo_labels = pseudo.label_map
            pseudo_areas = {int(c.candidate_id): int(c.area_px) for c in pseudo.candidates}
            matches = match_components_by_overlap(vector_labels, pseudo_labels, pseudo_areas, threshold=0.5)

            # does any vector instance span >= 2 pseudo instances (25 % of its own pixels each)?
            spans = set()
            for local_id in [int(v) for v in np.unique(vector_labels) if int(v) != 0]:
                mask = vector_labels == local_id
                area = int(mask.sum())
                if area == 0:
                    continue
                overlap_ids, counts = np.unique(pseudo_labels[mask], return_counts=True)
                big = [int(i) for i, c in zip(overlap_ids.tolist(), counts.tolist())
                       if int(i) != 0 and c / area >= 0.25]
                if len(big) >= 2:
                    spans.add(local_id)

            rows = []
            changed_here = False
            pseudo_areas_full = {int(c.candidate_id): int(c.area_px) for c in pseudo.candidates}
            for program in EXPECTED_QUERY_TYPES:
                vector_result = execute_program_by_id(program, vector_set, config)
                pseudo_result = execute_program_by_id(program, pseudo, config)

                def build_context(selected_vector_id):
                    """Per-answer evidence for the section 10 attribution classes."""

                    context = {
                        "spans_multiple_pseudo": False,
                        "pseudo_selection_merges_vectors": False,
                        "vector_target_area_px": None,
                        "vector_target_border": False,
                        "vector_target_clipped": False,
                    }
                    if selected_vector_id is None:
                        return context
                    local_id = int(selected_vector_id)
                    mask = vector_labels == local_id
                    area = int(mask.sum())
                    context["vector_target_area_px"] = area
                    context["spans_multiple_pseudo"] = local_id in spans
                    context["vector_target_border"] = bool(view["touches_border"][local_id - 1])
                    context["vector_target_clipped"] = bool(view["clipped"][local_id - 1])
                    # does the pseudo answer merge several native buildings, one of them the target?
                    if pseudo_result.selected_id is not None:
                        pseudo_mask = pseudo_labels == int(pseudo_result.selected_id)
                        contained = 0
                        target_inside = False
                        for other_id in [int(v) for v in np.unique(vector_labels) if int(v) != 0]:
                            other_mask = vector_labels == other_id
                            other_area = int(other_mask.sum())
                            if other_area == 0:
                                continue
                            overlap = int(np.logical_and(other_mask, pseudo_mask).sum())
                            if overlap / other_area >= 0.80:
                                contained += 1
                                if other_id == local_id:
                                    target_inside = True
                        context["pseudo_selection_merges_vectors"] = bool(contained >= 2 and target_inside)
                    return context

                vector_valid = not vector_result.abstained
                pseudo_valid = not pseudo_result.abstained
                bucket = per_program[program]
                if vector_valid:
                    bucket["vector_valid_tiles"] += 1
                if pseudo_valid:
                    bucket["pseudo_valid_tiles"] += 1
                if vector_valid and pseudo_valid:
                    pair = matches.get(int(vector_result.selected_id)) if vector_result.selected_id is not None else None
                    same = bool(pair is not None and int(pair["right_id"]) == int(pseudo_result.selected_id))
                    if same:
                        status = "target_unchanged"
                        attribution = None
                    else:
                        status = "target_changed"
                        attribution = attribute(program, vector_result, pseudo_result, matches,
                                                build_context(vector_result.selected_id))
                elif vector_valid and not pseudo_valid:
                    status = "became_invalid"
                    attribution = attribute(program, vector_result, pseudo_result, matches,
                                            build_context(vector_result.selected_id))
                elif not vector_valid and pseudo_valid:
                    status = "became_newly_valid"
                    attribution = attribute(program, vector_result, pseudo_result, matches,
                                            build_context(None))
                else:
                    status = "invalid_in_both"
                    attribution = None
                rows.append({
                    "program": program,
                    "vector_selected": vector_result.selected_id,
                    "pseudo_selected": pseudo_result.selected_id,
                    "status": status,
                    "attribution": attribution,
                })
                if status == "target_unchanged":
                    bucket["target_unchanged"] += 1
                    per_level[PROGRAM_LEVEL[program]]["comparable"] += 1
                elif status == "target_changed":
                    bucket["target_changed"] += 1
                    per_level[PROGRAM_LEVEL[program]]["comparable"] += 1
                    per_level[PROGRAM_LEVEL[program]]["target_changed"] += 1
                    changed_here = True
                    primary = attribution["primary"]
                    bucket["attribution_primary"][primary] = bucket["attribution_primary"].get(primary, 0) + 1
                    if len(examples) < 40:
                        examples.append({
                            "stem": tile.stem, "program": program,
                            "vector_selected": vector_result.selected_id,
                            "pseudo_selected": pseudo_result.selected_id,
                            "attribution": attribution,
                        })
                elif status == "became_invalid":
                    bucket["became_invalid"] += 1
                    per_level[PROGRAM_LEVEL[program]]["became_invalid"] += 1
                    changed_here = True
                elif status == "became_newly_valid":
                    bucket["became_newly_valid"] += 1
                    per_level[PROGRAM_LEVEL[program]]["became_newly_valid"] += 1
                    changed_here = True
            totals["vector"] += len(vector_set.candidates)
            totals["pseudo"] += len(pseudo.candidates)
            totals["matched"] += len(matches)
            totals["vector_missing"] += len([v for v in np.unique(vector_labels) if int(v) != 0]) - len(matches)
            tiles_scanned += 1
            if changed_here:
                tiles_with_any_change += 1
            handle.write(json.dumps({"stem": tile.stem, "programs": rows}, ensure_ascii=False, sort_keys=True) + "\n")
            if position % 500 == 0:
                print(f"[task6k1.drift] {position}/{len(tiles)} ({time.time() - started:.0f}s)", flush=True)

    weighted_changes = 0
    weighted_total = 0
    for program, bucket in per_program.items():
        comparable = bucket["target_unchanged"] + bucket["target_changed"]
        bucket["comparable"] = comparable
        bucket["change_rate"] = float(bucket["target_changed"] / comparable) if comparable else None
        at_least_one = comparable + bucket["became_invalid"] + bucket["became_newly_valid"]
        bucket["unstable"] = bucket["target_changed"] + bucket["became_invalid"] + bucket["became_newly_valid"]
        bucket["unstable_rate"] = float(bucket["unstable"] / at_least_one) if at_least_one else None
        if program in query_counts and comparable:
            weighted_total += query_counts[program]
            weighted_changes += query_counts[program] * (bucket["target_changed"] / comparable)

    report = {
        "_doc": (
            "Task 6K.1 section 10. VECTOR (native manually delineated instances) vs PSEUDO (current "
            "component representation) relation-answer drift for all 20 canonical programs under the "
            "frozen Task 3B semantics. Targets are compared through mask-overlap matching because ids "
            "are not comparable across candidate sets."
        ),
        "task": "6K.1",
        "scope": {
            "tiles_scanned": tiles_scanned,
            "programs": len(EXPECTED_QUERY_TYPES),
            "vector_instances": totals["vector"],
            "pseudo_instances": totals["pseudo"],
            "matched_vector_instances_iou_0_5": totals["matched"],
            "vector_instances_without_pseudo_match": totals["vector_missing"],
            "vector_instances_without_pseudo_match_rate": totals["vector_missing"] / max(totals["vector"], 1),
            "merge_metric_note": (
                "The merge rate (several native buildings inside one pseudo-instance) is measured by "
                "CONTAINMENT in evaluation/task6k1_vector_vs_pseudo_matching.json; the 1:1 matching used "
                "here can never express a merge, so no merge number is reported in this artifact."
            ),
        },
        "per_program": per_program,
        "by_level": {
            str(level): {
                **bucket,
                "change_rate": float(bucket["target_changed"] / bucket["comparable"]) if bucket["comparable"] else None,
                "at_least_one_valid": bucket["comparable"] + bucket["became_invalid"] + bucket["became_newly_valid"],
                "unstable_rate": (
                    float((bucket["target_changed"] + bucket["became_invalid"] + bucket["became_newly_valid"])
                          / (bucket["comparable"] + bucket["became_invalid"] + bucket["became_newly_valid"]))
                    if (bucket["comparable"] + bucket["became_invalid"] + bucket["became_newly_valid"]) else None
                ),
            }
            for level, bucket in per_level.items()
        },
        "tiles_with_any_relation_semantic_change": {
            "tiles": tiles_with_any_change,
            "rate": float(tiles_with_any_change / max(tiles_scanned, 1)),
        },
        "overall_current_query_target_change_rate": float(weighted_changes / max(weighted_total, 1)),
        "weighting": {
            "method": "per-program target-change rate weighted by the frozen BuildSpatialReason v0.1.1 query-type counts",
            "query_records": int(weighted_total),
        },
        "changed_examples": examples,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6k1.drift] tiles with any change {tiles_with_any_change}/{tiles_scanned} "
        f"({report['tiles_with_any_relation_semantic_change']['rate']:.3f}); weighted current-query "
        f"target-change rate {report['overall_current_query_target_change_rate']:.4f}",
        flush=True,
    )
    for level, bucket in sorted(report["by_level"].items()):
        print(f"[task6k1.drift]   level {level}: comparable {bucket['comparable']} changed "
              f"{bucket['target_changed']} rate {bucket['change_rate']} unstable {bucket['unstable_rate']}", flush=True)
    print(f"[task6k1.drift] wrote {OUT.name} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
