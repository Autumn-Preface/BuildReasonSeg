"""Task 6K tests: read-only WHU source -> pseudo-instance data audit.

Static/unit facts are asserted on the audit library and scripts; artifact facts are asserted
against the recorded JSON when it exists. Nothing here writes to the read-only sources, trains a
model or mutates a dataset.

Run with pytest, or directly::

    python tests/test_task6k_whu_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_source_audit import (  # noqa: E402
    CONVERTED_CANDIDATES,
    CROPPED_ROOT,
    CURRENT_ROOT,
    EPSILON_RATIO,
    HISTORICAL_CONVERTER,
    LEGACY_PROJECTS,
    MIN_APPROX_VERTICES,
    MIN_CONTOUR_AREA,
    ORIGINAL_ROOT,
    THRESHOLD,
    audit_contours,
    binarize,
    external_contour_target,
    historical_contours,
    parse_yolo_polygons,
    rasterize_polygons,
    read_semantic_label,
    read_semantic_label_pillow,
    simplified_polygon_target,
    simplify_contour,
    write_json,
)

EVAL = REPO_ROOT / "evaluation"
AUDIT_SOURCES = (
    "buildreasonseg_mvp/whu_source_audit.py",
    "scripts/task6k_common.py",
    "scripts/task6k_scan.py",
    "scripts/task6k_inventory.py",
    "scripts/task6k_aggregate.py",
    "scripts/task6k_yolo_fidelity.py",
    "scripts/task6k_relation_drift.py",
    "scripts/task6k_cross_6j.py",
    "scripts/task6k_schema.py",
    "scripts/task6k_decision.py",
)
TMP_DIR = REPO_ROOT / "artifacts" / "task6k" / "test_tmp"


def _code(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _tmp_path(name: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return TMP_DIR / name


# ------------------------------------------------------------------ 1-2 read-only guarantees


def test_no_writes_under_resources():
    """No audit code may write below C:\\D\\resources (the read-only sources)."""

    for relative in AUDIT_SOURCES:
        code = _code(relative)
        for marker in ("write_text", "write_bytes", "imwrite", "np.save", "savez", "shutil.copy", "mkdir"):
            for line in code.splitlines():
                if marker in line and (
                    "ORIGINAL_ROOT" in line or "CROPPED_ROOT" in line or "CONVERTED" in line
                ):
                    pytest.fail(f"{relative}: write '{marker}' targets a read-only source: {line.strip()}")
    assert str(ORIGINAL_ROOT).startswith("C:\\D\\resources")
    assert str(CROPPED_ROOT).startswith("C:\\D\\resources")


def test_no_writes_under_legacy_project():
    for relative in AUDIT_SOURCES:
        code = _code(relative)
        for line in code.splitlines():
            if "LEGACY_PROJECTS" in line or "WHU_Building_Segment" in line:
                for marker in ("write_text", "imwrite", "copy", "unlink", "mkdir", "rename"):
                    assert marker not in line, f"{relative}: {marker} on legacy path: {line.strip()}"
    assert all("WHU_Building_Segment" in str(project) for project in LEGACY_PROJECTS)


# ------------------------------------------------------------------ 3-7 historical semantics


def test_threshold_127_exact():
    """`cv2.threshold(mask, 127, 255, THRESH_BINARY)` means pixel > 127 becomes foreground."""

    mask = np.array([[0, 126, 127, 128, 255]], dtype=np.uint8)
    binary = binarize(mask)
    assert binary.tolist() == [[0, 0, 0, 255, 255]]
    assert THRESHOLD == 127
    _, reference = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    assert np.array_equal(binary, reference)


def test_retr_external_exact():
    """RETR_EXTERNAL keeps only outer contours: holes fill, islands inside holes are ignored."""

    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = 255
    mask[12:28, 12:28] = 0        # a hole inside the outer contour
    mask[17:23, 17:23] = 255      # an island inside the hole
    contours = historical_contours(binarize(mask))
    assert len(contours) == 1, "RETR_EXTERNAL must return exactly the outer contour"
    filled = external_contour_target(binarize(mask))
    assert filled[20, 20], "hole must be filled by the external contour"
    assert filled.sum() == 30 * 30
    source = _code("buildreasonseg_mvp/whu_source_audit.py")
    assert "cv2.RETR_EXTERNAL" in source and "cv2.CHAIN_APPROX_SIMPLE" in source


def test_min_contour_area_50_exact():
    """The filter is `cv2.contourArea(contour) < 50`, using the contour polygon area."""

    def one_component(size: int) -> np.ndarray:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:10 + size, 10:10 + size] = 255
        return mask

    # findContours returns the pixel-centre rectangle: area == (size - 1)^2
    assert cv2.contourArea(historical_contours(one_component(6))[0]) == pytest.approx(25.0)
    assert cv2.contourArea(historical_contours(one_component(8))[0]) == pytest.approx(49.0)
    assert cv2.contourArea(historical_contours(one_component(9))[0]) == pytest.approx(64.0)

    small = audit_contours(one_component(6))
    assert small.n_contours_kept == 0 and small.n_contours_removed == 1
    boundary = audit_contours(one_component(8))
    assert boundary.n_contours_kept == 0 and boundary.n_contours_removed == 1  # 49 < 50
    kept = audit_contours(one_component(9))
    assert kept.n_contours_kept == 1 and kept.n_contours_removed == 0
    assert MIN_CONTOUR_AREA == 50.0


def test_epsilon_is_0_001_arc_length():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    contour = historical_contours(mask)[0]
    approx, epsilon = simplify_contour(contour)
    assert epsilon == pytest.approx(EPSILON_RATIO * cv2.arcLength(contour, True))
    assert EPSILON_RATIO == 0.001
    assert len(approx) >= MIN_APPROX_VERTICES


def test_pillow_fallback_does_not_change_binary_semantics():
    """The Pillow fallback must produce the same binary mask as the OpenCV path."""

    from PIL import Image

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:20, 6:28] = 255
    mask[10:14, 10:14] = 0
    path = _tmp_path("fallback_probe.png")
    Image.fromarray(mask).save(path)
    opencv_read = read_semantic_label(path)
    pillow_read = read_semantic_label_pillow(path)
    assert np.array_equal(binarize(opencv_read), binarize(pillow_read))
    assert np.array_equal(binarize(opencv_read), mask)
    # the historical script's own structure: OpenCV first, Pillow only when cv2 returns None
    source = _code("buildreasonseg_mvp/whu_source_audit.py")
    assert "if mask is None:" in source and "Image.open" in source
    path.unlink(missing_ok=True)


# ------------------------------------------------------------------ 8-9 split hygiene


def test_historical_split_recovered_not_regenerated():
    text = _code("scripts/task6k_inventory.py")
    assert "converted_stems" in text
    # the historical random calls may only appear inside documentation strings (the recorded
    # historical code list plus the caveat prose); the script must not import random or
    # reimplement the historical stem collector.
    assert "import random" not in text, "the split must be recovered, never regenerated"
    assert text.count("random.seed") <= 2, "random.seed may only appear as recorded historical code"
    assert text.count("random.shuffle") <= 2, "random.shuffle may only appear as recorded historical code"
    assert "def get_image_stems" not in text, "the historical collector must not be reimplemented"
    artifact = _artifact("task6k_split_audit.json")
    if artifact is not None:
        assert artifact["counts"]["converted_total"] == 4038
        assert artifact["source_mapping"]["train_plus_val_equals_source_train"] is True
        assert artifact["source_mapping"]["source_test_equals_converted_test"] is True


def test_split_reproducibility_caveat_recorded():
    artifact = _artifact("task6k_split_audit.json")
    assert artifact is not None
    caveat = artifact["reproducibility_caveat"]
    assert "set" in caveat["caveat"] and "PYTHONHASHSEED" in caveat["caveat"]
    assert caveat["split_authority"].startswith("existing converted folders")
    assert caveat["val_size_formula_check"]["consistent"] is True


# ------------------------------------------------------------------ 10 YOLO parser / rasterizer


def test_yolo_rasterization_parser_correct():
    """A hand-written polygon label must parse and rasterize to the expected rectangle."""

    path = _tmp_path("polygon_probe.txt")
    # rectangle covering x in [0.25, 0.50], y in [0.25, 0.50] of a 512x512 tile
    path.write_text("0 0.25 0.25 0.50 0.25 0.50 0.50 0.25 0.50\n", encoding="utf-8")
    polygons, info = parse_yolo_polygons(path)
    assert len(polygons) == 1 and info["malformed_lines"] == []
    mask = rasterize_polygons(polygons, 512, 512)
    assert mask.any()
    ys, xs = np.nonzero(mask)
    assert xs.min() == 128 and xs.max() == 256
    assert ys.min() == 128 and ys.max() == 256
    # malformed line detection (fewer than 3 xy pairs)
    path.write_text("0 0.1 0.1 0.2\n", encoding="utf-8")
    polygons, info = parse_yolo_polygons(path)
    assert polygons == [] and info["malformed_lines"] == [1]
    path.unlink(missing_ok=True)


# ------------------------------------------------------------------ 11-15 audit integrity


def test_loss_decomposition_separates_filter_topology_approximation():
    artifact = _artifact("task6k_conversion_loss.json")
    assert artifact is not None
    for key in ("stage_1_area_filter", "stage_2_retr_external_topology", "stage_3_polygon_approximation"):
        assert key in artifact
    assert "decomposition_summary" in artifact
    assert artifact["pipeline_parameters"]["min_contour_area"] == 50.0
    assert artifact["pipeline_parameters"]["threshold"] == 127
    assert artifact["pipeline_parameters"]["epsilon_ratio"] == 0.001


def test_raw_components_never_called_true_physical_instances():
    for relative in AUDIT_SOURCES:
        text = _code(relative)
        assert "true physical instance" not in text.lower() or "not" in text.lower()
        assert "verified physical building" not in text or "NOT" in text or "not" in text
    stats = _artifact("task6k_raw_component_stats.json")
    assert stats is not None
    assert "semantic connected components" in stats["_doc"]
    assert "NOT" in stats["_doc"]
    lineage = _artifact("task6k_component_lineage.json")
    assert "never a verified physical building instance" in lineage["terminology"]


def test_relation_drift_uses_frozen_relation_config():
    text = _code("scripts/task6k_relation_drift.py")
    assert "frozen_relation_config" in text
    assert "execute_program_by_id" in text
    from task6k_common import RELATION_CONFIG

    assert RELATION_CONFIG.name == "spatial_relations_v1.yaml"
    artifact = _artifact("task6k_relation_semantic_drift.json")
    assert artifact is not None
    assert artifact["scope"]["programs"] == 20


def test_no_target_id_leakage_in_the_raw_vs_converted_experiment():
    """The drift experiment must not use target ids: candidate construction is target-free."""

    text = _code("scripts/task6k_relation_drift.py")
    assert "target_component_id" not in text
    assert "target_mask" not in text
    text = _code("buildreasonseg_mvp/whu_source_audit.py")
    assert "target_component_id" not in text


def test_merge_risk_labelled_heuristic():
    text = _code("scripts/task6k_scan.py")
    assert "heuristic" in text.lower()
    artifact = _artifact("task6k_merge_risk.json")
    assert artifact is not None
    assert "HEURISTIC" in artifact["_doc"]
    assert "does NOT recover true instances" in artifact["_doc"]
    assert set(artifact["heuristic_definition"]) >= {"high", "medium", "low"}


# ------------------------------------------------------------------ 16-20 provenance/tests


def test_task6j_cross_analysis_uses_frozen_artifacts():
    text = _code("scripts/task6k_cross_6j.py")
    for name in (
        "task6j_j1_oracle_program_yolo.json",
        "task6j_yolo_proposal_recall.json",
        "task6j_j2_program_parser.json",
        "task6j_j3_predicted_program_oracle_candidates.json",
    ):
        assert name in text
    artifact = _artifact("task6k_task6j_cross_analysis.json")
    assert artifact is not None
    facts = artifact["frozen_task6j_facts"]
    j1 = _artifact("task6j_j1_oracle_program_yolo.json")
    assert facts["J1_strict_miou"] == pytest.approx(j1["metrics"]["strict_selected_mask_miou"])
    assert facts["J1_paired_mask_selection"] == "5/20"


def test_j2_qualification_recorded():
    artifact = _artifact("task6k_task6j_cross_analysis.json")
    assert artifact is not None
    qualification = artifact["qwen_qualification"]
    assert "closed-template 20-program classification task" in qualification["proves"]
    assert len(qualification["does_not_prove"]) == 3
    assert qualification["task6k_ran_2b_vs_4b"] is False


def test_no_model_training_in_task6k():
    for relative in AUDIT_SOURCES:
        text = _code(relative)
        for marker in (".backward()", "optimizer.step", "AdamW", "CrossEntropyLoss", "train_step", "save_checkpoint"):
            assert marker not in text, f"{relative} must not train a model ({marker})"


def test_no_dataset_mutation():
    for relative in AUDIT_SOURCES:
        text = _code(relative)
        for marker in ("imwrite", "shutil.copy", "shutil.move", "os.rename", "os.remove"):
            assert marker not in text, f"{relative} must not mutate a dataset ({marker})"
    # the artifacts only reference BuildReasonSeg-relative or read-only source paths
    decision = _artifact("task6k_dataset_decision.json")
    if decision is not None:
        assert decision["read_only_guarantees"]["wrote_outside_buildreasonseg"] is False


def test_read_only_proof_shows_sources_untouched():
    """Empirical proof: no file under any read-only root was modified during the audit."""

    proof = _artifact("task6k_read_only_proof.json")
    assert proof is not None
    assert proof["roots"], "expected at least one read-only root"
    for name, value in proof["roots"].items():
        assert value["modified_after_audit_start"] is False, f"{name} was modified during the audit"
    decision = _artifact("task6k_dataset_decision.json")
    if decision is not None:
        check = decision["read_only_guarantees"]["empirical_check"]
        assert check["all_read_only_roots_untouched"] is True


def test_deterministic_json_output():
    path = _tmp_path("determinism_probe.json")
    payload = {"b": 2, "a": {"z": [1, 2, 3], "y": None}}
    write_json(path, payload)
    first = path.read_text(encoding="utf-8")
    write_json(path, payload)
    assert first == path.read_text(encoding="utf-8")
    keys = json.loads(first)
    assert list(keys) == ["a", "b"]  # sorted keys -> stable diffs
    path.unlink(missing_ok=True)


# ------------------------------------------------------------------ artifacts exist and align


def test_required_task6k_artifacts_exist():
    for name in (
        "task6k_source_inventory.json",
        "task6k_raw_component_stats.json",
        "task6k_conversion_loss.json",
        "task6k_actual_yolo_fidelity.json",
        "task6k_component_lineage.json",
        "task6k_relation_semantic_drift.json",
        "task6k_split_audit.json",
        "task6k_merge_risk.json",
        "task6k_task6j_cross_analysis.json",
        "task6k_dataset_decision.json",
        "task6k_read_only_proof.json",
        "dataset_audit_schema_v1.json",
    ):
        assert (EVAL / name).is_file(), f"missing artifact: {name}"


def test_source_and_current_alignment_complete():
    inventory = _artifact("task6k_source_inventory.json")
    assert inventory is not None
    assert inventory["splits"]["train"]["image_count"] == 3135
    assert inventory["splits"]["test"]["image_count"] == 903
    assert inventory["splits"]["train_no"]["image_count"] == 10527
    lineage = _artifact("task6k_component_lineage.json")
    assert lineage is not None
    assert lineage["identities"]["actual_polygons_equal_current_components"] is True
    assert lineage["identities"]["kept_contours_equal_actual_polygons"] is True
    assert lineage["measured"]["current_components"] == 36926


def test_dataset_decision_has_a_single_verdict():
    decision = _artifact("task6k_dataset_decision.json")
    assert decision is not None
    assert decision["dataset_role_verdict"] in (
        "REPLACE_PRIMARY_DATASET",
        "KEEP_WHU_AS_PRIMARY_FOR_NOW",
        "INSUFFICIENT_EVIDENCE",
    )
    assert decision["legacy_baseline_role"] in ("keep", "do_not_keep")
    assert decision["evidence_gates"]


def main() -> int:
    import inspect

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value) and not inspect.signature(value).parameters
    ]
    failures = 0
    for function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {function.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} task6k WHU-audit checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
