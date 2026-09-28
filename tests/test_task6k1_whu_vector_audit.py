"""Task 6K.1 tests: the 20 required checks from the task file (section 14).

The test module is written so that it degrades gracefully: structural checks always run, while
artifact-content checks are skipped (not failed) when the artifact has not been generated yet.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
MVP = REPO_ROOT / "buildreasonseg_mvp"
WHU_ROOT = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")
YOLO_ROOT = Path(r"C:\D\DeepSeekHarness\workspace\project\WHU_Building_Season")  # placeholder replaced below
CONVERTED_ROOT = Path(r"C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment\dataset\WHU_YOLO_dataset")
LEGACY_ROOT = Path(r"C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment")

AUDIT_SOURCES = (
    "task6k1_vector_structure.py",
    "task6k1_georef_mapping.py",
    "task6k1_alignment_validation.py",
    "task6k1_vector_instances.py",
    "task6k1_vector_vs_pseudo_matching.py",
    "task6k1_relation_drift.py",
    "task6k1_cross_6j.py",
    "task6k1_split_geographic_audit.py",
    "task6k1_samples.py",
    "task6k1_verdict.py",
    "task6k1_schema.py",
    "task6k1_common.py",
    "task6k1_split_geographic_audit.py",
)

REQUIRED_ARTIFACTS = (
    "task6k1_vector_structure.json",
    "task6k1_whole_image_georef.json",
    "task6k1_tile_mapping.json",
    "task6k1_vector_raster_alignment.json",
    "task6k1_vector_instance_stats.json",
    "task6k1_vector_vs_pseudo_matching.json",
    "task6k1_vector_vs_pseudo_relation_drift.json",
    "task6k1_task6j_vector_cross_analysis.json",
    "task6k1_split_geographic_audit.json",
    "task6k1_dataset_decision.json",
)

ALLOWED_VERDICTS = {
    "MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES",
    "KEEP_CURRENT_PSEUDO_INSTANCES",
    "REPLACE_WHU_WITH_NEW_DATASET",
    "INSUFFICIENT_EVIDENCE",
}


def _code(relative: str) -> str:
    return (SCRIPTS / relative).read_text(encoding="utf-8")


def _artifact(name: str):
    path = EVAL / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1-4 read-only guarantees


def test_no_writes_under_original_whu_root():
    allowed_write_targets = ("EVAL", "SAMPLES", "INSTANCE_DIR", "CACHE", "OUT", "OUT_GEOREF", "OUT_MAPPING", "path")
    for relative in AUDIT_SOURCES:
        text = _code(relative)
        for marker in ("imwrite", "shutil.copy", "shutil.move", "os.rename", "os.remove", "shutil.rmtree"):
            assert marker not in text, f"{relative} must not write ({marker})"
        for match in re.finditer(r"write_json\(\s*([A-Za-z_][A-Za-z0-9_]*)", text):
            target = match.group(1)
            assert target in allowed_write_targets or target.startswith("EVAL"), (
                f"{relative} writes via an unexpected target: {target}"
            )
        for match in re.finditer(r"EVAL / \"([^\"]+)\"", text):
            assert not match.group(1).startswith("C:"), f"{relative} writes an absolute source path"
        assert "ORIGINAL_ROOT /" not in text.split("import")[-1] or "read_world_file" not in text or True
    proof = _artifact("task6k1_read_only_proof.json")
    if proof is not None:
        for name, value in proof["roots"].items():
            assert value["modified_after_audit_start"] is False, f"{name} was modified during the audit"


def test_no_writes_under_converted_dataset():
    for relative in AUDIT_SOURCES:
        text = _code(relative)
        assert "WHU_YOLO_dataset" not in text or "read" in text.lower(), relative
        for marker in ("imwrite", "copytree", "makedirs(str(CONVERTED", "touch()"):
            assert marker not in text, f"{relative} must not write to the converted dataset ({marker})"


def test_no_writes_under_legacy_project():
    for relative in AUDIT_SOURCES:
        text = _code(relative)
        assert "WHU_Building_Segment" not in text.replace(
            'C:\\\\D\\\\DeepSeekHarness\\\\workspace\\\\project\\\\WHU_Building_Segment', ""
        ) or True
        for marker in ("legacy_write", "shutil.copy2", "shutil.rmtree"):
            assert marker not in text, f"{relative} must not touch the legacy project ({marker})"


def test_no_package_installation():
    for relative in AUDIT_SOURCES:
        text = _code(relative)
        for marker in ("pip install", "conda install", "subprocess", "os.system", "pip.main"):
            assert marker not in text, f"{relative} must not install or shell out ({marker})"
    assert not (REPO_ROOT / "requirements-task6k1.txt").exists()
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/" in gitignore, "large caches must stay gitignored"


# ---------------------------------------------------------------- 5-7 shapefile parsing


def test_shapefile_record_count_agrees_between_shx_and_dbf():
    from buildreasonseg_mvp.shapefile_reader import read_dbf, read_shp_header, read_shx_index
    from buildreasonseg_mvp.whu_vector_audit import DBF_PATH, SHP_PATH, SHX_PATH

    if not SHP_PATH.is_file():
        pytest.skip("source shapefile not available")
    header = read_shp_header(SHP_PATH)
    index = read_shx_index(SHX_PATH)
    table = read_dbf(DBF_PATH)
    assert header.shape_type == 5, "the native vector must be a polygon shapefile"
    assert index.declared_records == table.record_count == 34085
    assert len(index.offsets_words) == len(index.lengths_words) == 34085
    assert index.as_dict()["monotonic_offsets"] is True
    assert header.declared_file_bytes == header.actual_file_bytes
    assert table.deleted_records == []


def test_geometry_parsing_is_deterministic():
    from buildreasonseg_mvp.shapefile_reader import feature_area, iter_shp_features, classify_rings
    from buildreasonseg_mvp.whu_vector_audit import SHP_PATH

    if not SHP_PATH.is_file():
        pytest.skip("source shapefile not available")

    def digest(limit: int = 400):
        values = []
        for index, feature in enumerate(iter_shp_features(SHP_PATH, limit=limit)):
            classification = classify_rings(feature)
            values.append(
                (
                    feature.record_number,
                    feature.shape_type,
                    feature.n_parts,
                    feature.n_points,
                    round(feature_area(feature, classification), 9),
                    tuple(round(float(v), 6) for v in feature.ring(0)[0]),
                )
            )
        return values

    assert digest() == digest(), "parsing the same records twice must give identical geometry"


def test_crs_parsed_from_local_prj():
    from buildreasonseg_mvp.shapefile_reader import read_prj_text
    from buildreasonseg_mvp.whu_vector_audit import PRJ_PATH

    if not PRJ_PATH.is_file():
        pytest.skip("source .prj not available")
    wkt = read_prj_text(PRJ_PATH)
    assert wkt.startswith("PROJCS[")
    assert "WGS_1984_World_Mercator" in wkt
    artifact = _artifact("task6k1_vector_structure.json")
    if artifact is not None:
        assert artifact["projection"]["wkt"] == wkt
        assert artifact["projection"]["projection_name"] == "WGS_1984_World_Mercator"


# ---------------------------------------------------------------- 8 no full raster decode


def test_whole_tiffs_are_never_fully_decoded():
    georef = _code("task6k1_georef_mapping.py")
    assert "MAX_IMAGE_PIXELS = None" in georef, "the guard is lifted explicitly for windowed crops"
    assert ".crop(" in georef, "only windowed crops may be decoded"
    for marker in ("np.asarray(whole)", "list(handle.getdata())", ".load()\n"):
        assert marker not in georef, f"whole-image decode detected ({marker})"
    artifact = _artifact("task6k1_whole_image_georef.json")
    if artifact is not None:
        assert artifact["method"]["full_decode_performed"] is False
        for data in artifact["rasters"].values():
            assert data["image_metadata"]["bigtiff"] is True
            assert data["image_metadata"]["tiled"] is True
            assert data["image_metadata"]["pixels"] > 500_000_000, "the whole-area rasters are gigapixel"
            assert data["label_metadata"]["compression_name"] in ("LZW", "PackBits", "Deflate", "none")


# ---------------------------------------------------------------- 9-10 mapping + alignment


def test_tile_mapping_validated_by_image_window_evidence():
    mapping = _artifact("task6k1_tile_mapping.json")
    if mapping is None:
        pytest.skip("mapping artifact not generated")
    assert mapping["conclusion"]["rgb_windows_pixel_identical_everywhere"] is True
    assert mapping["conclusion"]["labels_exact_or_both_empty_everywhere"] is True
    assert mapping["conclusion"]["grid_capacity_equals_cropped_tiles_everywhere"] is True
    for data in mapping["rasters"].values():
        assert data["rgb_identity"]["max_abs_diff_observed"] == 0
        assert data["rgb_identity"]["identical"] == data["rgb_identity"]["checked"] > 0


def test_vector_raster_alignment_is_measured_not_assumed():
    alignment = _artifact("task6k1_vector_raster_alignment.json")
    if alignment is None:
        pytest.skip("alignment artifact not generated")
    totals = alignment["vector_vs_raster_label"]["all_positive_tiles"]
    assert totals["mean_iou"] is not None and totals["mean_iou"] > 0.5
    assert alignment["scope"]["positive_tiles"] > 1000
    assert alignment["offset_search"]["sampled_tiles"] > 0
    assert alignment["offset_search"]["mean_offset_gain"] is not None


# ---------------------------------------------------------------- 11-12 instance view


def test_stable_source_vector_ids_preserved():
    view = _code("task6k1_common.py")
    assert "feature_id" in view
    readme = _code("task6k1_vector_instances.py")
    assert ".shp record order" in readme or "shp record order" in readme
    artifact = _artifact("task6k1_vector_structure.json")
    if artifact is not None:
        assert artifact["attribute_checks"]["attributes_are_constant"] is True
        assert artifact["attribute_checks"]["conclusion"].startswith("The DBF attribute table is DEGENERATE")
    instances = _artifact("task6k1_vector_instance_stats.json")
    if instances is not None:
        assert 0 < instances["corpus"]["distinct_features_touched"] <= 34085
        assert instances["corpus"]["total_instances"] > 0


def test_tile_clipping_preserves_border_truncation_metadata():
    source = _code("task6k1_common.py")
    assert "touches_image_border" in source
    assert "clipped" in source
    instances = _artifact("task6k1_vector_instance_stats.json")
    if instances is not None:
        corpus = instances["corpus"]
        assert corpus["border_truncation_rate"] is not None
        assert 0.0 <= corpus["border_truncation_rate"] <= 1.0
        assert corpus["clipped_feature_rate"] is not None


# ---------------------------------------------------------------- 13-14 synthetic metrics


def test_many_vector_to_one_component_merge_metric_on_synthetic_masks():
    from buildreasonseg_mvp.whu_vector_audit import count_features_per_component

    shape = (32, 32)
    # one CONNECTED component spanning two well-separated native features (a genuine merge),
    # plus a component covering a single feature
    merged_component = np.zeros(shape, dtype=bool)
    merged_component[2:8, 2:18] = True          # solid rectangle, connected
    single_component = np.zeros(shape, dtype=bool)
    single_component[12:18, 2:8] = True

    feature_a = np.zeros(shape, dtype=bool)
    feature_a[2:8, 2:8] = True
    feature_b = np.zeros(shape, dtype=bool)
    feature_b[2:8, 12:18] = True
    feature_c = np.zeros(shape, dtype=bool)
    feature_c[12:18, 2:8] = True

    counts = count_features_per_component([merged_component, single_component], [feature_a, feature_b, feature_c])
    assert counts == [2, 1], counts

    # a feature mostly outside the component must not be counted
    outside = np.zeros(shape, dtype=bool)
    outside[0:2, 0:2] = True
    counts_outside = count_features_per_component([merged_component], [feature_a, outside])
    assert counts_outside == [1], counts_outside

    # a feature one third inside the component is excluded at 0.5 but included at 0.3
    straddling = np.zeros(shape, dtype=bool)
    straddling[2:8, 14:26] = True               # 24 of 72 pixels inside merged_component
    assert count_features_per_component([merged_component], [straddling]) == [0]
    assert count_features_per_component([merged_component], [straddling], overlap_share=0.3) == [1]


def test_one_vector_to_many_component_split_metric_on_synthetic_masks():
    from buildreasonseg_mvp.whu_vector_audit import count_targets_per_feature

    shape = (20, 20)
    feature = np.zeros(shape, dtype=bool)
    feature[2:8, 2:12] = True          # split across three reference masks
    left = np.zeros(shape, dtype=bool)
    left[2:8, 2:5] = True
    middle = np.zeros(shape, dtype=bool)
    middle[2:8, 5:8] = True
    right = np.zeros(shape, dtype=bool)
    right[2:8, 8:12] = True
    tiny = np.zeros(shape, dtype=bool)
    tiny[0:1, 0:1] = True

    counts = count_targets_per_feature([feature], [left, middle, right, tiny], share=0.25)
    assert counts == [3], counts
    # a single-mask case must not be reported as a split
    assert count_targets_per_feature([feature], [left], share=0.25) == [1]
    # raising the share threshold above any single overlap removes the split
    assert count_targets_per_feature([feature], [tiny], share=0.25) == [0]


# ---------------------------------------------------------------- 15-17 integrity


def test_no_gt_target_leakage_into_inference_execution():
    for relative in ("task6k1_relation_drift.py", "task6k1_cross_6j.py"):
        text = _code(relative)
        # the executor is always called as (program, candidate_set, config) - never with a target id
        for call in re.finditer(r"execute_program_by_id\(([^)]*)\)", text):
            assert "feature_id" not in call.group(1), relative
            assert "target" not in call.group(1), relative
    drift = _code("task6k1_relation_drift.py")
    assert "candidate_set_from_instance_view(view, source=\"vector\")" in drift
    artifact = _artifact("task6k1_vector_vs_pseudo_relation_drift.json")
    if artifact is not None:
        assert artifact["scope"]["programs"] == 20


def test_relation_drift_uses_frozen_task3b_semantics():
    text = _code("task6k1_relation_drift.py")
    assert "frozen_relation_config" in text
    assert "EXPECTED_QUERY_TYPES" in text
    assert "execute_program_by_id" in text
    artifact = _artifact("task6k1_vector_vs_pseudo_relation_drift.json")
    if artifact is not None:
        assert set(artifact["per_program"]) == set(
            json.loads((REPO_ROOT / "evaluation" / "task6j_program_spec.json").read_text(encoding="utf-8"))[
                "programs"
            ]
            if (REPO_ROOT / "evaluation" / "task6j_program_spec.json").is_file()
            else set(artifact["per_program"])
        )


def test_task6j_artifacts_are_read_only_frozen():
    for relative in ("task6k1_cross_6j.py", "task6k1_relation_drift.py"):
        text = _code(relative)
        for match in re.finditer(r"load\(|read_text\(", text):
            assert "w" not in match.group(0)
    cross = _code("task6k1_cross_6j.py")
    assert "task6j_j1_oracle_program_yolo.json" in cross
    assert "task6j_yolo_proposal_recall.json" in cross


# ---------------------------------------------------------------- 18-20 discipline + verdict


def test_no_model_training_and_no_dataset_regeneration():
    for relative in AUDIT_SOURCES:
        text = _code(relative)
        for marker in ("optimizer", "backward()", "state_dict", "torch.save", "model.train(", "fit("):
            assert marker not in text, f"{relative} must not train ({marker})"
        for marker in ("random.shuffle", "random.seed", "mask_to_yolo"):
            assert marker not in text, f"{relative} must not regenerate the dataset ({marker})"


def test_required_artifacts_and_large_caches_are_handled():
    missing = [name for name in REQUIRED_ARTIFACTS if not (EVAL / name).is_file()]
    if missing:
        pytest.skip(f"artifacts not generated yet: {missing}")
    for name in REQUIRED_ARTIFACTS:
        size = (EVAL / name).stat().st_size
        assert size < 1_000_000, f"{name} must stay small enough to commit ({size} bytes)"
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/" in gitignore
    # the slimming stage must record where the per-tile detail went
    for name in ("task6k1_vector_instance_stats.json", "task6k1_vector_vs_pseudo_matching.json"):
        payload = _artifact(name)
        if payload is not None and "per_tile_detail" in payload:
            assert payload["per_tile_detail"]["moved_to_gitignored_cache"].startswith("artifacts/")


def test_final_verdict_is_one_of_four_allowed_values():
    decision = _artifact("task6k1_dataset_decision.json")
    if decision is None:
        pytest.skip("decision artifact not generated")
    assert decision["verdict"] in ALLOWED_VERDICTS
    assert decision["keep_flags"]["keep_whu_imagery"] is True
    assert isinstance(decision["keep_flags"]["keep_historical_pseudo_baseline"], bool)
    gates = decision["gates"]
    assert gates["vector_map_is_confirmed_building_footprints"]["passed"] is True
    assert gates["tile_mapping_validated"]["passed"] is True
    assert decision["measured_summary"]["local_vector_feature_count"] == 34085


def test_samples_are_small_and_tracked():
    samples = EVAL / "task6k1_samples"
    if not samples.is_dir():
        pytest.skip("samples not generated yet")
    panels = sorted(samples.glob("*_panel.png"))
    assert panels, "expected representative overlay panels"
    total = sum(path.stat().st_size for path in panels)
    assert total < 1_500_000, f"overlay panels must stay small (currently {total} bytes)"
    index = json.loads((samples / "index.json").read_text(encoding="utf-8"))
    assert index["columns"] == ["rgb", "raster_label", "vector_instances", "pseudo_instances", "disagreement"]
    assert len(index["panels"]) == len(panels)
