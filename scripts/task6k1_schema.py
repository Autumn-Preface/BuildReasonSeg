"""Task 6K.1 Part J (section 14): native-vector instance of the reusable dataset-audit schema.

Fills the same schema v1 that Task 6K defined for the pseudo-instance representation, so the two can
be compared field by field.

Writes `evaluation/dataset_audit_schema_whu_native_vector.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_vector_audit import DBF_PATH, PRJ_PATH, SHP_PATH, SHX_PATH  # noqa: E402

from task6k1_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "dataset_audit_schema_whu_native_vector.json"


def load(name: str) -> dict:
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> int:
    structure = load("task6k1_vector_structure.json")
    georef = load("task6k1_whole_image_georef.json")
    alignment = load("task6k1_vector_raster_alignment.json")
    instances = load("task6k1_vector_instance_stats.json")
    comparison = load("task6k1_vector_vs_pseudo_matching.json")
    drift = load("task6k1_vector_vs_pseudo_relation_drift.json")
    split = load("task6k1_split_geographic_audit.json")
    verdict = load("task6k1_dataset_decision.json")
    schema = load("dataset_audit_schema_v1.json")

    geometry = structure.get("geometry", {})
    total_instances = instances.get("corpus", {}).get("total_instances")
    per_tile = instances.get("corpus", {}).get("instances_per_tile", {})
    extents = split.get("extents", {})
    scripted = comparison.get("per_tile", [])

    report = {
        "_doc": (
            "Task 6K.1 section 14. `dataset_audit_schema_v1` instance for the NATIVE WHU East-Asia "
            "vector map (true instance identity), for direct comparison with Task 6K's pseudo-instance "
            "instance of the same schema."
        ),
        "schema_version": schema.get("schema_version", "1.0"),
        "dataset_name": "WHU Satellite Dataset II (East Asia) ---native vector building map",
        "instance": {
            "annotation_type": "manually delineated polygon building footprint map (ESRI shapefile)",
            "true_instance_identity_available": True,
            "identity_key": "shp record order (1-based)",
            "identity_key_note": (
                "the DBF attribute table is degenerate (every field constant over all 34085 records), "
                "so OBJECTID cannot be used as an identity key"
            ),
            "image_count": 3,
            "image_pixels": sum(
                data.get("image_metadata", {}).get("pixels", 0) for data in georef.get("rasters", {}).values()
            ),
            "instance_count": geometry.get("feature_count"),
            "instances_per_tile": {
                "mean": per_tile.get("mean"),
                "median": per_tile.get("median"),
                "p90": per_tile.get("p90"),
                "max": per_tile.get("max"),
            },
            "corpus_clipped_instances": total_instances,
            "distinct_features_touched_in_corpus": instances.get("corpus", {}).get("distinct_features_touched"),
            "area_distribution_map_units": geometry.get("area_map_units"),
            "area_units_note": (
                "areas are in WGS_1984_World_Mercator map units (metres, inflated by 1/cos(lat)^2 "
                "relative to ground area); no conversion to ground m^2 is attempted"
            ),
            "tiny_target_rate": {
                "definition": "< 50 map-unit^2 features",
                "value": None,
                "note": "not used as a gate here; Task 6K measured the pseudo-side equivalent",
            },
            "border_truncation_rate": instances.get("corpus", {}).get("border_truncation_rate"),
            "multipart_features": geometry.get("parts_per_feature", {}).get("2"),
            "features_with_holes": geometry.get("features_with_holes"),
            "vector_raster_agreement": {
                "mean_iou": alignment.get("vector_vs_raster_label", {})
                .get("all_positive_tiles", {})
                .get("mean_iou"),
                "tiles_ge_0_90": alignment.get("vector_vs_raster_label", {})
                .get("all_positive_tiles", {})
                .get("tiles_iou_ge_0_90"),
                "tiles_lt_0_50": alignment.get("vector_vs_raster_label", {})
                .get("all_positive_tiles", {})
                .get("tiles_iou_lt_0_50"),
                "best_offset_zero_everywhere": alignment.get("offset_search", {}).get("sampled_tiles", 0)
                == alignment.get("offset_search", {}).get("zero_offset_is_best", -1),
            },
            "candidate_density": {
                "instances_per_pixel": (
                    total_instances / max(instances.get("corpus", {}).get("tiles", 1), 1) / (512 * 512)
                    if total_instances
                    else None
                ),
                "pseudo_instances_same_corpus": comparison.get("counts", {}).get("pseudo_instances"),
                "raster_semantic_components_same_corpus": comparison.get("counts", {}).get(
                    "raster_semantic_components"
                ),
            },
            "relation_constructibility": {
                "level_1_valid_tiles": drift.get("per_program", {}).get("leftmost", {}).get("vector_valid_tiles"),
                "level_2_valid_tiles_example": drift.get("per_program", {}).get("largest_to_left_of", {}).get(
                    "vector_valid_tiles"
                ),
                "level_3_valid_tiles_example": drift.get("per_program", {}).get(
                    "largest_to_left_of_to_nearest", {}
                ).get("vector_valid_tiles"),
                "note": "per-program valid-tile counts for the native vector candidate set",
            },
            "merge_split_structure": {
                "component_merge_rate_primary": comparison.get(
                    "many_vector_to_one_semantic_component", {}
                ).get("components_containing_multiple_features_rate"),
                "merge_rate_pseudo_instances_containing_multiple_buildings": comparison.get(
                    "many_vector_to_one_pseudo_instance", {}
                ).get("rate"),
                "share_of_buildings_inside_merged_pseudo": comparison.get(
                    "many_vector_to_one_pseudo_instance", {}
                ).get("features_inside_multi_feature_pseudo_rate"),
                "split_rate_buildings_spanning_multiple_pseudo": comparison.get(
                    "one_vector_to_many_components", {}
                ).get("rate"),
                "vector_instances_missing_from_pseudo": drift.get("scope", {}).get(
                    "vector_instances_without_pseudo_match_rate"
                ),
            },
            "geographic_diversity": {
                "reference_crs": structure.get("projection", {}).get("projection_name"),
                "approx_lon_lat_bbox_all": "125.246856, 39.605662 to 125.751973, 39.670870",
                "train_extent_km": extents.get("train", {}).get("map_extent_km"),
                "test_extent_km": extents.get("test", {}).get("map_extent_km"),
                "split_is_geographic": False,
                "val_tiles_adjacent_to_train_rate": split.get("spatial_correlation_risk", {}).get(
                    "val_tiles_directly_adjacent_to_train_tiles_rate"
                ),
                "test_tiles_adjacent_to_train_or_val_rate": split.get("spatial_correlation_risk", {}).get(
                    "test_tiles_directly_adjacent_to_train_or_val_rate"
                ),
            },
            "local_storage": {
                "shp_bytes": SHP_PATH.stat().st_size,
                "shx_bytes": SHX_PATH.stat().st_size,
                "dbf_bytes": DBF_PATH.stat().st_size,
                "prj_bytes": PRJ_PATH.stat().st_size,
            },
            "license": "UNKNOWN",
            "source_resolution": "UNKNOWN",
            "provenance": {
                "shp": str(SHP_PATH),
                "reader": "standard-library ESRI Shapefile reader (no shapefile package installed)",
                "external_counts_ignored": ["29085", "34085-as-published"],
            },
            "verdict": verdict.get("verdict"),
        },
        "comparison_with_pseudo_instance_schema_instance": {
            "pseudo_instance_schema_artifact": "evaluation/dataset_audit_schema_v1.json",
            "true_instance_identity": {
                "native_vector": True,
                "pseudo_instances": schema.get("instance", {}).get("true_instance_identity_available"),
            },
            "instance_count": {
                "native_vector": geometry.get("feature_count"),
                "pseudo_instances": schema.get("instance", {}).get("instance_count"),
            },
            "border_truncation_rate": {
                "native_vector": instances.get("corpus", {}).get("border_truncation_rate"),
                "pseudo_instances": (
                    scripted[0].get("border_rate") if scripted and isinstance(scripted[0], dict) else None
                ),
            },
            "caveat": "counts differ by view: 34085 native features, 36926 pseudo-instances, 38309 raw semantic components",
        },
        "per_tile_source": "evaluation/task6k1_vector_instance_stats.json",
    }
    write_json(OUT, report)
    print(f"[task6k1.schema] wrote {OUT.name} (verdict {report['instance']['verdict']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
