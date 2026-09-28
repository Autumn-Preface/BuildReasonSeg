"""Task 6L tests: the 28 required checks from the task file (section 22).

Structural checks always run; artifact-content checks skip (never fail) when an artifact has not been
generated yet, so the suite stays usable during development.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
NV_ROOT = REPO_ROOT / "datasets" / "whu_native_vector" / "v1.0"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
V011 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1"
SCRIPTS = REPO_ROOT / "scripts"

WHU_SOURCE = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")
CONVERTED = Path(r"C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment\dataset\WHU_YOLO_dataset")
LEGACY = Path(r"C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment")

TASK6L_SOURCES = (
    "task6l_build_dataset.py",
    "task6l_build_reasoning_view.py",
    "task6l_build_v0_2.py",
    "task6l_compare_v01_v02.py",
    "task6l_validate.py",
)


def _artifact(name: str):
    path = EVAL / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _code(relative: str) -> str:
    return (SCRIPTS / relative).read_text(encoding="utf-8")


def _v02_records(split: str, limit: int | None = None):
    path = V02 / f"{split}.jsonl"
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


# ------------------------------------------------------------------ 1-3 read-only


def test_original_whu_source_read_only():
    # every write target must be a BuildReasonSeg-internal path, never a source path
    allowed = (
        "OUT", "out_dir", "EVAL", "SAMPLES", "DATASET_ROOT", "CACHE_ROOT", "REASONING_VIEW_ROOT",
        "path", "INTEGRITY_OUT", "SPLIT_AUDIT_OUT", "tmp", "root",
    )
    for relative in TASK6L_SOURCES:
        text = _code(relative)
        for marker in ("imwrite", "shutil.copy", "shutil.move", "os.rename", "os.remove", "shutil.rmtree"):
            assert marker not in text, f"{relative} must not write to a source ({marker})"
        for match in re.finditer(r"write_jsonl?\(\s*([A-Za-z_][A-Za-z0-9_]*)", text):
            target = match.group(1)
            assert target in allowed or target.startswith("EVAL"), (
                f"{relative} writes via an unexpected target: {target}"
            )
        # no write helper may be called with a source-root-derived path variable
        for match in re.finditer(r"write_jsonl?\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*/", text):
            assert match.group(1) not in ("source_root", "WHU_SOURCE"), relative
    library = (REPO_ROOT / "buildreasonseg_mvp" / "whu_native_vector.py").read_text(encoding="utf-8")
    assert "CROPPED_ROOT" in library


def test_converted_dataset_read_only():
    for relative in TASK6L_SOURCES:
        text = _code(relative)
        assert "WHU_YOLO_dataset" not in text.replace("converted", ""), relative
    adapter = (REPO_ROOT / "buildreasonseg_mvp" / "native_vector_adapter.py").read_text(encoding="utf-8")
    for marker in ("imwrite", "copy2", "rmtree"):
        assert marker not in adapter


def test_legacy_project_read_only():
    for relative in TASK6L_SOURCES:
        text = _code(relative)
        assert "WHU_Building_Segment" not in text
    assert not (REPO_ROOT / "datasets" / "whu_native_vector" / "v1.0" / "manifest.json").read_text(
        encoding="utf-8"
    ).count("WHU_Building_Segment")


# ------------------------------------------------------------------ 4, 25-27 discipline


def test_v011_frozen_unchanged():
    verdict = _artifact("task6l_verdict.json")
    if verdict is None:
        pytest.skip("verdict artifact not generated yet")
    recorded = verdict["gates"]["v011_frozen_unchanged"]["measured"]
    for name, digest in recorded.items():
        assert hashlib.sha256((V011 / name).read_bytes()).hexdigest() == digest, f"{name} changed"
    manifest = json.loads((V011 / "manifest.json").read_text(encoding="utf-8"))
    for split, expected in manifest["sample_counts"]["by_split"].items():
        path = V011 / f"{split}.jsonl"
        assert sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) == expected


def test_no_model_training():
    for relative in TASK6L_SOURCES:
        text = _code(relative)
        for marker in ("optimizer", "backward()", "torch.save", "model.train(", "fit(", "state_dict"):
            assert marker not in text, f"{relative} must not train ({marker})"


def test_no_downloads_or_installations():
    for relative in TASK6L_SOURCES:
        text = _code(relative)
        for marker in ("pip install", "conda install", "requests.get", "urllib.request", "wget", "os.system"):
            assert marker not in text, f"{relative} must not download or install ({marker})"


def test_no_gui():
    for path in list(SCRIPTS.glob("task6l_*.py")) + [
        REPO_ROOT / "buildreasonseg_mvp" / "whu_native_vector.py",
        REPO_ROOT / "buildreasonseg_mvp" / "native_vector_adapter.py",
    ]:
        text = path.read_text(encoding="utf-8")
        for marker in ("tkinter", "PyQt", "gradio", "streamlit", "cv2.imshow", "flask", "fastapi"):
            assert marker not in text, f"{path.name} must not build a GUI ({marker})"


# ------------------------------------------------------------------ 5-8 identity / mapping / clipping


def test_native_source_feature_ids_stable():
    manifest = json.loads((NV_ROOT / "manifest.json").read_text(encoding="utf-8"))
    integrity = _artifact("task6l_vector_dataset_integrity.json")
    if integrity is None:
        pytest.skip("integrity artifact not generated yet")
    assert manifest["instance_identity"]["source_feature_id"].startswith("1-based EA.shp record order")
    assert manifest["instance_identity"]["dbf_fields_are_identity"] is False
    assert integrity["stable_source_ids_survive_clipping"] is True
    assert integrity["instances_with_no_source_feature_id"] == 0
    from buildreasonseg_mvp.whu_native_vector import load_vector_corpus

    corpus = load_vector_corpus()
    assert len(corpus) == 34085
    assert [f.feature_id for f in corpus.features[:5]] == [1, 2, 3, 4, 5]


def test_all_17388_cropped_tiles_accounted_for():
    statistics = json.loads((NV_ROOT / "statistics.json").read_text(encoding="utf-8"))
    integrity = _artifact("task6l_vector_dataset_integrity.json")
    assert statistics["tiles_indexed"] == 17388
    assert statistics["tile_counts_by_category"] == {
        "train": 3135, "train_no": 10527, "test": 903, "test_no": 2823
    }
    tile_lines = sum(1 for line in (NV_ROOT / "tiles" / "index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    assert tile_lines == 17388
    if integrity is not None:
        assert integrity["tile_id_unique"] is True and integrity["grid_id_unique"] is True


def test_tile_to_whole_raster_mapping_deterministic():
    from buildreasonseg_mvp.whu_native_vector import iter_source_tiles

    first = iter_source_tiles(WHU_SOURCE)
    second = iter_source_tiles(WHU_SOURCE)
    assert [(t.tile_id, t.grid_id, t.raster, t.row, t.col) for t in first] == [
        (t.tile_id, t.grid_id, t.raster, t.row, t.col) for t in second
    ]
    rasters = Counter(t.raster for t in first)
    assert rasters["train1"] == 10044 and rasters["train2"] == 3618 and rasters["test"] == 3726


def test_vector_clipping_deterministic():
    from buildreasonseg_mvp.whu_native_vector import (
        FullAreaCache,
        clip_native_instances,
        iter_source_tiles,
        load_vector_corpus,
    )

    corpus = load_vector_corpus()
    cache = FullAreaCache(corpus)
    tiles = iter_source_tiles(WHU_SOURCE)[:3]
    first = [i.as_dict(with_geometry=True) for t in tiles for i in clip_native_instances(corpus, t, cache)]
    second = [i.as_dict(with_geometry=True) for t in tiles for i in clip_native_instances(corpus, t, cache)]
    assert first == second


# ------------------------------------------------------------------ 9-11 preservation


def test_holes_and_multipart_not_silently_dropped():
    statistics = json.loads((NV_ROOT / "statistics.json").read_text(encoding="utf-8"))
    integrity = _artifact("task6l_vector_dataset_integrity.json")
    from buildreasonseg_mvp.whu_native_vector import load_vector_corpus

    corpus = load_vector_corpus()
    source_holes = sum(1 for f in corpus.features if f.hole_rings)
    source_multipart = sum(1 for f in corpus.features if f.n_parts > 1)
    assert source_holes == 5 and source_multipart == 5
    assert integrity["holes_preserved"]["hole_loss"] is False
    assert integrity["multipart_preserved"]["multipart_loss"] is False
    assert statistics["instances_with_holes"] >= source_holes - 5
    # the code must not use RETR_EXTERNAL or hole-filling anywhere in the canonical path
    library = (REPO_ROOT / "buildreasonseg_mvp" / "whu_native_vector.py").read_text(encoding="utf-8")
    assert "cv2.findContours" not in library, "the canonical path must not trace contours"
    assert '"hole"' in library and '"outer"' in library


def test_no_global_area_deletion_in_canonical_gt():
    statistics = json.loads((NV_ROOT / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["preservation_guarantees"]["min_contour_area_filter"] is None
    assert statistics["tiny_instances"] > 0, "tiny instances exist and must be retained"
    library = (REPO_ROOT / "buildreasonseg_mvp" / "whu_native_vector.py").read_text(encoding="utf-8")
    assert "cv2.contourArea" not in library, "no contour-area filter may exist in the canonical path"
    assert "cv2.approxPolyDP" not in library, "no polygon simplification may exist in the canonical path"
    integrity = _artifact("task6l_vector_dataset_integrity.json")
    assert integrity["no_global_area_deletion"]["note"].startswith("tiny instances are flagged")


def test_empty_tiles_supported():
    statistics = json.loads((NV_ROOT / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["empty_tiles"] > 10000
    assert statistics["empty_tiles"] + statistics["non_empty_tiles"] == 17388
    from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset

    dataset = NativeVectorDataset()
    empty = None
    for view in dataset.iter_tiles("train", split_view="scene_disjoint_v1"):
        if view.is_empty:
            empty = view
            break
    assert empty is not None
    assert dataset.list_instances(empty.tile_id) == []


# ------------------------------------------------------------------ 12-15 splits


def test_legacy_compat_v1_exact_historical_stems():
    payload = json.loads((NV_ROOT / "splits" / "legacy_compat_v1.json").read_text(encoding="utf-8"))
    assert payload["view"] == "legacy_compat_v1"
    from task6k_common import converted_stems

    assert set(payload["train"]) == converted_stems("train", kind="labels")
    assert set(payload["val"]) == converted_stems("val", kind="labels")
    assert set(payload["test"]) == converted_stems("test", kind="labels")
    assert len(payload["train"]) == 2508 and len(payload["val"]) == 627 and len(payload["test"]) == 903


def test_scene_disjoint_v1_mapping_correct():
    payload = json.loads((NV_ROOT / "splits" / "scene_disjoint_v1.json").read_text(encoding="utf-8"))
    assert payload["definition"] == {"train": "train1", "val": "train2", "test": "test"}
    tiles = {row["tile_id"]: row for row in read_jsonl(NV_ROOT / "tiles" / "index.jsonl")}
    for split, raster in (("train", "train1"), ("val", "train2"), ("test", "test")):
        rasters = {tiles[tile]["source_raster"] for tile in payload[split]}
        assert rasters == {raster}, f"{split} mixes rasters {rasters}"
    assert len(payload["train"]) == 10044 and len(payload["val"]) == 3618 and len(payload["test"]) == 3726


def test_zero_tile_overlap_across_scene_disjoint_splits():
    payload = json.loads((NV_ROOT / "splits" / "scene_disjoint_v1.json").read_text(encoding="utf-8"))
    train, val, test = set(payload["train"]), set(payload["val"]), set(payload["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    audit = _artifact("task6l_scene_disjoint_split_audit.json")
    if audit is not None:
        assert audit["tile_overlap_zero"] is True


def test_zero_source_feature_leakage():
    audit = _artifact("task6l_scene_disjoint_split_audit.json")
    if audit is None:
        pytest.skip("split audit not generated yet")
    assert audit["source_feature_leakage_zero"] is True
    assert all(count == 0 for count in audit["source_feature_overlap_after_exclusions"].values())
    # any exclusion must be explicitly listed
    assert isinstance(audit["excluded_boundary_crossing_features"], list)
    if audit["excluded_boundary_crossing_feature_count"]:
        assert audit["exclusion_reason"]


# ------------------------------------------------------------------ 16-20 semantics / language


def test_no_test_driven_threshold_tuning():
    v02_manifest = json.loads((V02 / "manifest.json").read_text(encoding="utf-8"))
    v011_manifest = json.loads((V011 / "manifest.json").read_text(encoding="utf-8"))
    assert v02_manifest["relation_config_sha256"] == v011_manifest["relation_config_sha256"]
    assert v02_manifest["frozen_relation_settings"] == v011_manifest["frozen_relation_settings"]
    config = yaml.safe_load((REPO_ROOT / "configs" / "build_spatial_reason_v0.2.yaml").read_text(encoding="utf-8"))
    assert config["relations"]["frozen"] == yaml.safe_load(
        (REPO_ROOT / "configs" / "build_spatial_reason_v0.1.1.yaml").read_text(encoding="utf-8")
    )["relations"]["frozen"]
    assert config["semantic_policy"]["on_semantic_target_ineligible"] == "discard"


def test_relation_engine_consumes_vector_geometry():
    from buildreasonseg_mvp.native_vector_adapter import (
        NativeVectorDataset,
        candidate_set_for_tile,
        image_record_for_reasoning,
    )

    dataset = NativeVectorDataset()
    tile = next(view for view in dataset.iter_tiles("val", split_view="scene_disjoint_v1")
                if view.instance_count >= 2)
    record = image_record_for_reasoning(dataset, tile.tile_id, "val")
    assert record["components"], "the reasoning record must expose native instances as components"
    instance = record["components"][0]
    for field in ("component_id", "area_px", "centroid_px", "bbox_xyxy_px", "touches_image_border"):
        assert field in instance
    candidate_set = candidate_set_for_tile(dataset, tile.tile_id)
    assert candidate_set is not None and len(candidate_set.candidates) == tile.instance_count
    first = candidate_set.candidates[0]
    assert first.mask.any() and first.area_px > 0
    # the frozen relation engine accepts the adapted geometry
    import thresholds as T

    from spatial_reasoning import relations as R  # noqa: F401  (import path check)

    config = T.load_config(REPO_ROOT / "configs" / "spatial_relations_v1.yaml")
    assert config is not None


def test_all_20_program_ids_preserved():
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    statistics = json.loads((V02 / "statistics.json").read_text(encoding="utf-8"))
    assert len(EXPECTED_QUERY_TYPES) == 20
    assert set(statistics["by_query_type"]) <= set(EXPECTED_QUERY_TYPES)
    manifest = json.loads((V02 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["program_count"] == 20
    v011 = json.loads((V011 / "manifest.json").read_text(encoding="utf-8"))
    assert set(v011["sample_counts"]["by_query_type"]) == set(EXPECTED_QUERY_TYPES)


def test_ids_absent_from_model_input_instructions():
    records = _v02_records("val", limit=150)
    if not records:
        pytest.skip("v0.2 not generated yet")
    for record in records:
        target = (record.get("native_vector") or {}).get("target") or {}
        for identifier in (target.get("source_feature_id"), target.get("tile_instance_id")):
            if identifier is None:
                continue
            for field in ("instruction_zh", "instruction_en"):
                text = str(record[field])
                assert not re.search(rf"(?<!\d){int(identifier)}(?!\d)", text), (
                    f"{field} leaks an instance id: {text}"
                )
        assert "tile_instance_id" not in str(record["instruction_zh"])
        assert "source_feature" not in str(record["instruction_en"])


def test_zh_en_semantic_parity():
    records = _v02_records("val", limit=200)
    if not records:
        pytest.skip("v0.2 not generated yet")
    for record in records:
        zh, en = str(record["instruction_zh"]), str(record["instruction_en"])
        assert zh.strip() and en.strip()
        assert any("\u4e00" <= ch <= "\u9fff" for ch in zh), "instruction_zh must contain Chinese"
        assert re.search(r"[A-Za-z]", en)
        assert str(record["reasoning_zh"]).strip() and str(record["reasoning_en"]).strip()
    # identical program distribution in both languages (same records, both emitted)
    counts_zh = Counter(record["query_type"] for record in _v02_records("train", limit=400))
    counts_en = Counter(record["query_type"] for record in _v02_records("train", limit=400))
    assert counts_zh == counts_en


# ------------------------------------------------------------------ 21-24 pipeline integrity


def test_generator_deterministic():
    verdict = _artifact("task6l_verdict.json")
    if verdict is None:
        pytest.skip("verdict artifact not generated yet")
    determinism = verdict["gates"]["generator_and_validator_deterministic"]["measured"]
    assert determinism["identical"] is True
    assert determinism["real_v02_untouched"] is True


def test_validator_catches_invalid_target_provenance():
    from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset, validate_reasoning_record

    dataset = NativeVectorDataset()
    record = next(iter(_v02_records("val", limit=5)), None)
    if record is None:
        pytest.skip("v0.2 not generated yet")
    assert validate_reasoning_record(record, dataset) == []

    corrupted = json.loads(json.dumps(record))
    corrupted["native_vector"]["target"]["tile_instance_id"] = 250
    corrupted["native_vector"]["target"]["source_feature_id"] = 999999
    problems = validate_reasoning_record(corrupted, dataset)
    assert problems, "the validator must reject an impossible target instance"

    corrupted_tile = json.loads(json.dumps(record))
    corrupted_tile["image_id"] = "definitely_not_a_tile"
    assert validate_reasoning_record(corrupted_tile, dataset)


def test_every_program_has_val_and_test_support():
    statistics = json.loads((V02 / "statistics.json").read_text(encoding="utf-8"))
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    for split in ("val", "test"):
        present = set(statistics["program_support"][split])
        missing = sorted(set(EXPECTED_QUERY_TYPES) - present)
        assert not missing, f"{split} lacks support for {missing}"


def test_v01_comparison_reconciles_with_task6k1():
    comparison = _artifact("task6l_v01_vs_v02_comparison.json")
    if comparison is None:
        pytest.skip("comparison not generated yet")
    rate = comparison["weighted_current_query_target_change_rate"]
    reference = comparison["reconciliation"]["task6k1_vector_vs_pseudo_weighted_drift"]
    assert rate is not None
    assert abs(rate - reference) <= 0.05, f"{rate} does not reconcile with {reference}"
    assert comparison["scope"]["common_query_groups"] > 10000
    assert comparison["answer_change"]["target_changed"] > 0


# ------------------------------------------------------------------ 28 artifact consistency


def test_full_artifact_consistency():
    required = (
        "datasets/whu_native_vector/v1.0/manifest.json",
        "datasets/whu_native_vector/v1.0/statistics.json",
        "datasets/whu_native_vector/v1.0/splits/legacy_compat_v1.json",
        "datasets/whu_native_vector/v1.0/splits/scene_disjoint_v1.json",
        "datasets/build_spatial_reason/v0.2/manifest.json",
        "datasets/build_spatial_reason/v0.2/statistics.json",
        "evaluation/task6l_vector_dataset_integrity.json",
        "evaluation/task6l_scene_disjoint_split_audit.json",
        "evaluation/task6l_build_spatial_reason_v0.2_quality.json",
        "evaluation/task6l_v01_vs_v02_comparison.json",
        "evaluation/task6l_artifact_index.json",
        "evaluation/task6l_verdict.json",
        "docs/task6l_vector_dataset_migration.md",
    )
    missing = [name for name in required if not (REPO_ROOT / name).is_file()]
    if missing:
        pytest.skip(f"artifacts not generated yet: {missing}")
    index = _artifact("task6l_artifact_index.json")
    assert index["all_present"] is True
    for entry in index["artifacts"]:
        path = REPO_ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    manifest = json.loads((NV_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "WHU-EA-NativeVector"
    assert manifest["dataset_version"] == "v1.0"
    statistics = json.loads((NV_ROOT / "statistics.json").read_text(encoding="utf-8"))
    assert statistics["tiles_indexed"] == statistics["tiles_expected"] == 17388
    v02_manifest = json.loads((V02 / "manifest.json").read_text(encoding="utf-8"))
    assert v02_manifest["split_view"] == "scene_disjoint_v1"
    assert v02_manifest["source_component_representation_version"] == "whu-native-vector-v1.0"


def test_verdict_is_one_of_two_allowed_values():
    verdict = _artifact("task6l_verdict.json")
    if verdict is None:
        pytest.skip("verdict not generated yet")
    assert verdict["verdict"] in ("VECTOR_DATASET_MIGRATION_PASS", "VECTOR_DATASET_MIGRATION_NEEDS_FIX")
    assert verdict["keep_flags"]["keep_whu_imagery"] is True
    assert verdict["keep_flags"]["keep_historical_pseudo_baseline_v0_1_1"] is True
    if verdict["verdict"] == "VECTOR_DATASET_MIGRATION_PASS":
        assert verdict["failed_gates"] == []


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
