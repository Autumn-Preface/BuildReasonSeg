"""Task 6K Part K (section 16): reusable dataset-audit schema v1.

A machine-readable schema for auditing any candidate source dataset with the same questions that
were asked of WHU, so a future candidate (SpaceNet 2, WHU-Mix Vector, ...) can be scored against
identical fields. Fields are filled from THIS audit where the evidence exists; anything not
verified locally is marked "UNKNOWN" exactly as required -- no external dataset facts are
invented.

Writes `evaluation/dataset_audit_schema_v1.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_source_audit import write_json  # noqa: E402

from task6k_common import EVAL  # noqa: E402

WHU_FIELDS = {
    "annotation_type": "semantic_raster_binary",
    "true_stable_instance_id_available": False,
    "instance_identity_note": (
        "The source label is a binary semantic mask. Instances in the current representation are "
        "connected components (pseudo-instances) produced by RETR_EXTERNAL, so no true physical "
        "instance id exists anywhere in the chain."
    ),
    "geometry_source_of_truth": "rasterized_polygon_component_map",
    "image_count": 17388,
    "image_count_with_buildings": 4038,
    "instance_count": 36926,
    "instances_per_tile": {"min": 1, "median": 7, "mean": 9.14, "p90": 19, "max": 52},
    "area_distribution_px": {"min": 50, "median": 1176, "p90": 3072, "max": 131054},
    "tiny_target_rate": "MEASURED_IN_task6k_raw_component_stats",
    "border_truncation_rate": "MEASURED_IN_task6k_raw_component_stats",
    "multi_instance_tile_rate": "MEASURED_IN_task6k_raw_component_stats",
    "candidate_density": "MEASURED_IN_task6k_raw_component_stats",
    "l1_l2_l3_constructible_query_rate": "MEASURED_IN_task6k_relation_semantic_drift",
    "nearest_constructibility": "MEASURED_IN_task6k_relation_semantic_drift",
    "directional_relation_constructibility": "MEASURED_IN_task6k_relation_semantic_drift",
    "relation_ambiguity_rate": "MEASURED_IN_task6k_relation_semantic_drift",
    "geographic_diversity_metadata": {
        "available": False,
        "detail": (
            "No scene/geographic grouping metadata is present in the labels or the stripped local "
            "copy; the shapefile subtree exists locally but was not joined to tiles in this audit."
        ),
    },
    "local_storage_footprint": "MEASURED_IN_task6k_source_inventory",
    "license": "UNKNOWN",
    "source_resolution": "UNKNOWN",
}


def field_spec() -> dict:
    return {
        "annotation_type": {
            "type": "enum",
            "allowed": ["semantic", "raster_instance", "vector_polygon", "unknown"],
            "question": "Is ground truth a semantic mask, a raster instance map, or vector polygons?",
        },
        "true_stable_instance_id_available": {
            "type": "boolean",
            "question": "Does the source provide a stable per-instance identity (not derived)?",
        },
        "image_count": {"type": "integer", "question": "Number of source tiles/images."},
        "instance_count": {
            "type": "integer",
            "question": "Number of instances (or derived pseudo-instances) in the corpus.",
        },
        "instances_per_tile": {
            "type": "distribution",
            "question": "min/median/mean/p90/max instances per tile.",
        },
        "area_distribution": {
            "type": "distribution",
            "question": "Instance area distribution (state the unit).",
        },
        "tiny_target_rate": {
            "type": "rate",
            "question": "Share of instances below a declared tiny threshold (state the threshold).",
        },
        "border_truncation_rate": {
            "type": "rate",
            "question": "Share of instances touching the tile border.",
        },
        "multi_instance_tile_rate": {
            "type": "rate",
            "question": "Share of tiles with >= 2 instances (relation queries need >= 2).",
        },
        "candidate_density": {
            "type": "distribution",
            "question": "Candidates per tile, i.e. the size of the selection problem.",
        },
        "l1_l2_l3_constructible_query_rate": {
            "type": "rate",
            "question": "Share of tiles where L1/L2/L3 canonical programs are constructible.",
        },
        "nearest_constructibility": {
            "type": "rate",
            "question": "Share of tiles where the frozen `nearest` relation yields a target.",
        },
        "directional_relation_constructibility": {
            "type": "rate",
            "question": "Share of tiles where a directional filter yields exactly one candidate.",
        },
        "relation_ambiguity_rate": {
            "type": "rate",
            "question": "Share of relation evaluations rejected as ambiguous by the frozen thresholds.",
        },
        "geographic_diversity_metadata": {
            "type": "object",
            "question": "Is scene/city/region metadata available for a geographic split?",
        },
        "local_storage_footprint": {
            "type": "object",
            "question": "Local bytes per split/subtree, if known.",
        },
        "license": {
            "type": "string",
            "question": "License status; UNKNOWN unless externally verified.",
        },
        "source_resolution": {
            "type": "string",
            "question": "Native ground sample distance/resolution; UNKNOWN unless verified.",
        },
    }


def main() -> int:
    schema = {
        "_doc": (
            "Task 6K section 16. Reusable dataset-audit schema v1: the same questions that were "
            "asked of WHU, so a future candidate dataset can be scored on identical fields. Fields "
            "whose value is not verified locally are marked UNKNOWN; no external dataset facts are "
            "fabricated."
        ),
        "schema_version": "v1",
        "purpose": (
            "Decide whether a candidate source dataset is suitable as the PRIMARY corpus for "
            "instance-level spatial-reasoning annotations, and how it must be split."
        ),
        "fields": field_spec(),
        "instances": {
            "whu_satellite_ii_east_asia": {
                "audited_in": "Task 6K",
                "audit_artifacts": [
                    "evaluation/task6k_source_inventory.json",
                    "evaluation/task6k_raw_component_stats.json",
                    "evaluation/task6k_conversion_loss.json",
                    "evaluation/task6k_actual_yolo_fidelity.json",
                    "evaluation/task6k_component_lineage.json",
                    "evaluation/task6k_relation_semantic_drift.json",
                    "evaluation/task6k_split_audit.json",
                    "evaluation/task6k_merge_risk.json",
                    "evaluation/task6k_task6j_cross_analysis.json",
                    "evaluation/task6k_dataset_decision.json",
                ],
                **WHU_FIELDS,
            }
        },
        "candidate_template": {
            "_comment": (
                "Copy this template for each future candidate (SpaceNet 2, WHU-Mix Vector, ...). "
                "Leave license/source_resolution as UNKNOWN until externally verified."
            ),
            "annotation_type": "UNKNOWN",
            "true_stable_instance_id_available": None,
            "image_count": None,
            "instance_count": None,
            "instances_per_tile": "UNKNOWN",
            "area_distribution": "UNKNOWN",
            "tiny_target_rate": "UNKNOWN",
            "border_truncation_rate": "UNKNOWN",
            "multi_instance_tile_rate": "UNKNOWN",
            "candidate_density": "UNKNOWN",
            "l1_l2_l3_constructible_query_rate": "UNKNOWN",
            "nearest_constructibility": "UNKNOWN",
            "directional_relation_constructibility": "UNKNOWN",
            "relation_ambiguity_rate": "UNKNOWN",
            "geographic_diversity_metadata": "UNKNOWN",
            "local_storage_footprint": "UNKNOWN",
            "license": "UNKNOWN",
            "source_resolution": "UNKNOWN",
        },
        "decision_gates": {
            "replace_primary_dataset_when": [
                "material conversion-induced relation drift (weighted current-query target-change rate above the recorded threshold)",
                "substantial small-object deletion",
                "high heuristic merge risk",
                "lack of true instance identity becomes binding for the intended claims",
                "the Task 6J mismatch is tied to pseudo-instance semantics rather than a replaceable proposal model",
                "split structure unsuitable for the intended paper claims",
            ],
            "keep_as_primary_when": [
                "conversion loss is small",
                "relation targets are very stable",
                "pseudo-instance identity is sufficiently reliable",
                "the Task 6J failure is mainly a replaceable proposal-model problem",
            ],
            "insufficient_evidence_when": ["source and converted data cannot be aligned reliably"],
        },
    }
    write_json(EVAL / "dataset_audit_schema_v1.json", schema)
    print("[task6k.schema] wrote dataset_audit_schema_v1.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
