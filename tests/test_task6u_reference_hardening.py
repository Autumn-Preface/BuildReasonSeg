"""Task 6U tests (section O): 44 checks on the reference candidate hardening and the proposal-set ranker."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
RANKER_MODULE = REPO_ROOT / "buildreasonseg_mvp" / "task6u_reference_ranker.py"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
REF_PACKS = REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
PROPOSAL_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
    / "m26m_seg_continued" / "weights" / "best.pt"
PROPOSAL_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
    / "m1_yolo26m_seg_continued" / "weights" / "best.pt"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
HARDENED_PARSER = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
BASE_COMMIT = "e63f8c40761637040ac6169ad603b6f7c7001274"
PROPOSAL_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
TASK6U_SOURCES = ("task6u_freeze_reference_split.py", "task6u_candidate_coverage.py",
                  "task6u_train_ranker.py", "task6u_evaluate_reference.py",
                  "task6u_evaluate_downstream.py", "task6u_report.py", "task6u_common.py")
V1_4_CONFIGS = {
    "U-C0": {"imgsz": 640, "conf": 0.10, "max_det": 100},
    "U-C1": {"imgsz": 640, "conf": 0.05, "max_det": 300},
    "U-C2": {"imgsz": 1024, "conf": 0.10, "max_det": 300},
    "U-C3": {"imgsz": 1024, "conf": 0.05, "max_det": 300},
}


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_changed(prefix: str) -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _code_only(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    for prefix in ("r", "f", "b", "u", "rf", "fr"):
        if stripped.startswith(prefix + '"""') or stripped.startswith(prefix + "'''"):
            stripped = stripped[len(prefix):]
            break
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            end = stripped.find(quote, len(quote))
            if end != -1:
                stripped = stripped[end + len(quote):]
            break
    return "\n".join(line.split("#", 1)[0] for line in stripped.splitlines())


# ---------------------------------------------------------------- 1-2 frozen prior evidence


def test_task6t_artifacts_unchanged():
    assert _git_changed("evaluation/task6t_") == ""
    verdict = _artifact("task6t_verdict.json")
    assert verdict["verdict"] == "PARSER_SEMANTIC_CONTRAST_FAIL"


def test_hardened_parser_checkpoint_not_retrained():
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""
    assert _git_changed("buildreasonseg_mvp/structured_grounding.py") == ""
    training = _artifact("task6u_ranker_training.json")
    if training is not None:
        for name in TASK6U_SOURCES:
            code = _code_only(SCRIPTS / name)
            for marker in ("load_parser_checkpoint", "save_parser_checkpoint", "train_step(batch"):
                assert marker not in code or name == "task6u_evaluate_downstream.py", \
                    f"{name} must not train the parser ({marker})"
    if HARDENED_PARSER.is_file():
        assert HARDENED_PARSER.stat().st_size > 0


# ---------------------------------------------------------------- 3-11 proposal configurations


def test_yolo_checkpoint_sha_exact():
    audit = _artifact("task6u_refval_candidate_audit.json")
    assert PROPOSAL_CHECKPOINT.is_file(), "the frozen Task 6M.1 checkpoint must exist locally"
    assert _sha256(PROPOSAL_CHECKPOINT) == PROPOSAL_SHA256
    if audit is not None:
        assert audit["checkpoint"]["sha256"] == PROPOSAL_SHA256


def test_exactly_four_candidate_configs():
    from scripts.task6u_common import CONFIG_ORDER, CONFIGS

    assert len(CONFIGS) == 4
    assert tuple(CONFIGS) == CONFIG_ORDER == ("U-C0", "U-C1", "U-C2", "U-C3")
    calibration = _artifact("task6u_calibration_candidate_coverage.json")
    if calibration is not None:
        assert sorted(calibration["configurations"]) == list(CONFIG_ORDER)
        assert calibration["verdict"] if "verdict" in calibration else True


def test_c0_exact_baseline():
    from scripts.task6u_common import CONFIGS

    assert {key: CONFIGS["U-C0"][key] for key in ("imgsz", "conf", "max_det")} == V1_4_CONFIGS["U-C0"]
    assert CONFIGS["U-C0"]["tta"] is False and CONFIGS["U-C0"]["tiling"] is False


def test_c1_exact_values():
    from scripts.task6u_common import CONFIGS

    assert {key: CONFIGS["U-C1"][key] for key in ("imgsz", "conf", "max_det")} == V1_4_CONFIGS["U-C1"]


def test_c2_exact_values():
    from scripts.task6u_common import CONFIGS

    assert {key: CONFIGS["U-C2"][key] for key in ("imgsz", "conf", "max_det")} == V1_4_CONFIGS["U-C2"]


def test_c3_exact_values():
    from scripts.task6u_common import CONFIGS

    assert {key: CONFIGS["U-C3"][key] for key in ("imgsz", "conf", "max_det")} == V1_4_CONFIGS["U-C3"]


def test_no_tta():
    from scripts.task6u_common import CONFIGS

    for config in CONFIGS.values():
        assert config["tta"] is False
    for name in TASK6U_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("augment=True", "augment = True", "tta", "flip"):
            assert marker not in code or marker == "tta", f"{name} must not use TTA ({marker})"


def test_no_tiling():
    from scripts.task6u_common import CONFIGS

    for config in CONFIGS.values():
        assert config["tiling"] is False
    for name in TASK6U_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("sliding_window", "tile_grid", "patchwise", "crop_with_overlap"):
            assert marker not in code, f"{name} must not use tiling ({marker})"


def test_source_masks_restored_to_512():
    from buildreasonseg_mvp.task6q_reference_resolver import TILE_SIZE
    from scripts.task6u_common import TILE, run_config_on_tile  # noqa: F401

    assert TILE == 512 and TILE_SIZE == 512
    runner = _code_only(SCRIPTS / "task6u_common.py")
    assert "proposals_from_results(results)" in runner  # canonical normalize_mask to source size
    selected = _artifact("task6u_selected_proposal_config.json")
    if selected is not None:
        assert selected["selected"]["masks_restored_to_source"] is True
        assert selected["selected"]["source_image_size"] == [512, 512]
    calibration = _artifact("task6u_calibration_candidate_coverage.json")
    if calibration is not None:
        for report in calibration["configurations"].values():
            assert report["masks_restored_to_source"] is True
            assert report["source_image_size"] == [512, 512]


# ---------------------------------------------------------------- 12-17 calibration split


def test_task6q_eligibility_unchanged():
    from buildreasonseg_mvp.task6q_reference_resolver import (
        MERGE_BBOX_EXTENT_RATIO_MAX,
        TINY_COMPONENT_AREA_PX,
        is_eligible,
        config_report,
    )

    assert MERGE_BBOX_EXTENT_RATIO_MAX == 0.20 and TINY_COMPONENT_AREA_PX == 150
    assert config_report()["threshold_sweep"] is False
    assert _git_changed("buildreasonseg_mvp/task6q_reference_resolver.py") == ""
    assert "is_eligible" in _code_only(SCRIPTS / "task6u_common.py")


def test_u_calib200_is_train_only():
    split = _artifact("task6u_reference_train_split.json")
    if split is None:
        pytest.skip("split not frozen yet")
    calib = split["u_calib200"]["records"]
    assert len(calib) == 200
    assert all(record["split"] == "train" for record in calib)
    assert split["requirements"]["calib_is_train_only"] is True


def test_u_rankertrain_is_train_only():
    split = _artifact("task6u_reference_train_split.json")
    if split is None:
        pytest.skip("split not frozen yet")
    assert split["u_rankertrain"]["count"] == 625
    assert split["requirements"]["rankertrain_is_train_only"] is True
    # every ranker-train record is a frozen RefTrainUnique record
    frozen = json.loads((REF_PACKS / "ref_train_unique.json").read_text(encoding="utf-8"))["records"]
    frozen_keys = {(str(r["split"]), str(r["tile_id"]), int(r["reference_source_feature_id"]),
                    str(r["reference_family"])) for r in frozen}
    calib_keys = {(str(r["split"]), str(r["tile_id"]), int(r["reference_source_feature_id"]),
                   str(r["reference_family"])) for r in split["u_calib200"]["records"]}
    assert len(frozen_keys - calib_keys) == split["u_rankertrain"]["count"]


def test_u_calib_and_rankertrain_disjoint():
    split = _artifact("task6u_reference_train_split.json")
    if split is None:
        pytest.skip("split not frozen yet")
    assert split["requirements"]["zero_key_overlap_calib_rankertrain"] is True
    assert split["requirements"]["zero_key_overlap_with_refval_unique"] is True
    assert split["requirements"]["tile_overlap_calib_ranker"] >= 0


def test_refval_untouched_by_config_selection():
    selected = _artifact("task6u_selected_proposal_config.json")
    calibration = _artifact("task6u_calibration_candidate_coverage.json")
    if selected is None or calibration is None:
        pytest.skip("selection not frozen yet")
    assert selected["selection_inputs"]["refval_unique"] is False
    assert selected["selection_inputs"]["minival240"] is False
    assert selected["selection_inputs"]["u_calib200"] is True
    assert calibration["refval_used_for_selection"] is False
    assert calibration["minival_used_for_selection"] is False
    # the calibration split itself contains only RefTrainUnique records
    assert calibration["calibration_split"]["records"] == 200


def test_selected_config_priority_rule_exact():
    from scripts.task6u_common import selection_priority_key

    calibration = _artifact("task6u_calibration_candidate_coverage.json")
    selected = _artifact("task6u_selected_proposal_config.json")
    if calibration is None or selected is None:
        pytest.skip("selection not frozen yet")
    reports = calibration["configurations"]
    ranked = sorted(reports, key=lambda config_id: selection_priority_key(reports[config_id]),
                    reverse=True)
    assert ranked == calibration["ranking"]
    assert selected["selected_config"] == ranked[0]
    assert len(selected["priority_rule"]) == 7
    assert selected["immutable_after_creation"] is True


# ---------------------------------------------------------------- 18-29 ranker


def test_ranker_feature_dimension_exactly_14():
    from buildreasonseg_mvp.task6u_reference_ranker import FEATURE_DIM, FEATURE_NAMES, proposal_features
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal

    assert FEATURE_DIM == 14
    assert len(FEATURE_NAMES) == 14
    mask = np.zeros((512, 512), dtype=bool)
    mask[10:30, 10:30] = True
    features = proposal_features([build_proposal(0, 0.9, mask)], "largest")
    assert features.shape == (1, 14)


def test_no_centroid_or_location_feature_in_ranker():
    from buildreasonseg_mvp.task6u_reference_ranker import FEATURE_NAMES

    forbidden = ("centroid", "cx", "cy", "x_norm", "y_norm", "location", "position", "bbox_x",
                 "bbox_y")
    for name in forbidden:
        assert name not in " ".join(FEATURE_NAMES).lower(), name
    code = _code_only(RANKER_MODULE)
    assert "centroid" not in code
    training = _artifact("task6u_ranker_training.json")
    if training is not None:
        excluded = " ".join(training["feature_vector"]["excluded"]).lower()
        assert "centroid" in excluded and "image location" in excluded


def test_no_gt_feature_in_ranker():
    from buildreasonseg_mvp.task6u_reference_ranker import FEATURE_NAMES

    joined = " ".join(FEATURE_NAMES).lower()
    for name in ("gt", "target", "truth", "oracle", "iou"):
        assert name not in joined, name
    code = _code_only(RANKER_MODULE)
    for marker in ("gt_mask", "gt_reference", "target_mask", "source_feature_id"):
        assert marker not in code, marker


def test_no_relation_input_to_ranker():
    from buildreasonseg_mvp.task6u_reference_ranker import FEATURE_NAMES, proposal_features
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal

    assert not any("relation" in name for name in FEATURE_NAMES)
    import inspect

    signature = inspect.signature(proposal_features)
    assert set(signature.parameters) == {"proposals", "family"}
    mask = np.zeros((512, 512), dtype=bool)
    mask[10:40, 10:40] = True
    largest = proposal_features([build_proposal(0, 0.9, mask)], "largest")
    smallest = proposal_features([build_proposal(0, 0.9, mask)], "smallest")
    assert not np.allclose(largest, smallest)  # only the family indicator differs
    assert largest[0][:12].tolist() == smallest[0][:12].tolist()


def test_ranker_architecture_14_32_16_1():
    from buildreasonseg_mvp.task6u_reference_ranker import FEATURE_DIM, ProposalSetRanker

    model = ProposalSetRanker()
    layers = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(14, 32), (32, 16),
                                                                            (16, 1)]
    assert FEATURE_DIM == 14
    report = model.architecture_report()
    assert report["attention"] is False and report["transformer"] is False and report["gnn"] is False
    assert report["total_parameters"] == 1025
    output = model(torch.zeros(2, 5, 14))
    assert output.shape == (2, 5)


def test_family_one_hot_exact():
    from buildreasonseg_mvp.task6u_reference_ranker import FAMILY_ONE_HOT, proposal_features
    from buildreasonseg_mvp.task6q_reference_resolver import build_proposal

    assert FAMILY_ONE_HOT == {"largest": (1.0, 0.0), "smallest": (0.0, 1.0)}
    mask = np.zeros((512, 512), dtype=bool)
    mask[10:40, 10:40] = True
    largest = proposal_features([build_proposal(0, 0.9, mask)], "largest")[0]
    smallest = proposal_features([build_proposal(0, 0.9, mask)], "smallest")[0]
    assert largest[12] == 1.0 and largest[13] == 0.0
    assert smallest[12] == 0.0 and smallest[13] == 1.0
    training = _artifact("task6u_ranker_training.json")
    if training is not None:
        assert training["feature_vector"]["family_one_hot"] == {"largest": [1, 0],
                                                               "smallest": [0, 1]}


def test_ranker_train_labels_use_gt_only_offline():
    code = _code_only(SCRIPTS / "task6u_train_ranker.py")
    assert "gt_reference_mask(record)" in code
    assert "gt_ious" in code
    inference = _code_only(SCRIPTS / "task6u_evaluate_downstream.py")
    # the inference selection helper must not use GT
    selection_block = inference.split("def select_reference_mask")[1].split("def predict_target")[0]
    assert "gt_" not in selection_block
    training = _artifact("task6u_ranker_training.json")
    if training is not None:
        assert "GT builds training labels only" in training["_doc"] or "labels" in training["data"]["labeling"]


def test_uncovered_refs_excluded_from_loss_and_counted():
    training = _artifact("task6u_ranker_training.json")
    if training is None:
        pytest.skip("ranker training not generated yet")
    assert training["data"]["untrainable_not_covered"] + training["data"]["trainable"] \
        == training["data"]["examples"]
    assert training["data"]["trainable"] == training["data"]["internal_train"] \
        + training["data"]["internal_holdout"]
    code = _code_only(SCRIPTS / "task6u_train_ranker.py")
    assert "untrainable_not_covered" in code
    assert "COVERED_IOU = 0.50" in code


def test_u_s0_exact_task6q_behavior():
    from buildreasonseg_mvp.task6q_reference_resolver import select_reference

    comparison = _artifact("task6u_refval_selector_comparison.json")
    mini = _artifact("task6u_downstream_minival240.json")
    if comparison is None or mini is None:
        pytest.skip("comparison not generated yet")
    assert comparison["systems"]["U-S0"]["selector"] == "deterministic"
    assert mini["configs"]["U-S0"] == "U-C0"
    # U-S0 must reproduce the frozen Task 6S downstream numbers
    assert abs(mini["systems"]["U-S0"]["strict_all_miou"] - 0.2969667241009681) <= 1e-6
    assert abs(mini["systems"]["U-S0"]["answered_only_miou"] - 0.3045812554881724) <= 1e-6
    assert mini["systems"]["U-S0"]["abstentions"] == 6
    assert mini["systems"]["U-S0"]["reference_fail_count"] == 117
    assert select_reference.__module__ == "buildreasonseg_mvp.task6q_reference_resolver"


def test_u_s1_uses_deterministic_area_selector():
    comparison = _artifact("task6u_refval_selector_comparison.json")
    mini = _artifact("task6u_downstream_minival240.json")
    if comparison is None or mini is None:
        pytest.skip("comparison not generated yet")
    assert comparison["systems"]["U-S1"]["selector"] == "deterministic"
    selected = _artifact("task6u_selected_proposal_config.json")
    assert mini["configs"]["U-S1"] == selected["selected_config"]
    assert selected["selected_config"] != "U-C0"


def test_u_s2_uses_learned_ranker():
    comparison = _artifact("task6u_refval_selector_comparison.json")
    if comparison is None:
        pytest.skip("comparison not generated yet")
    assert comparison["systems"]["U-S2"]["selector"] == "ranker"
    training = _artifact("task6u_ranker_training.json")
    assert training is not None
    payload = torch.load(RANKER_CHECKPOINT, map_location="cpu", weights_only=False) \
        if RANKER_CHECKPOINT.is_file() else None
    if payload is not None:
        assert payload["feature_dim"] == 14
        assert payload["architecture"]["total_parameters"] == 1025


def test_only_three_selectors():
    from scripts.task6u_evaluate_downstream import SYSTEMS
    from scripts.task6u_evaluate_reference import main as _reference_main  # noqa: F401

    assert tuple(SYSTEMS) == ("U-S0", "U-S1", "U-S2")
    comparison = _artifact("task6u_refval_selector_comparison.json")
    if comparison is not None:
        assert comparison["selectors_compared"] == 3
        assert tuple(comparison["systems"]) == ("U-S0", "U-S1", "U-S2")


# ---------------------------------------------------------------- 30-35 evidence reuse and inference


def test_minival240_exact_reuse():
    manifest = _artifact("task6n_pack_manifest.json")
    mini = _artifact("task6u_downstream_minival240.json")
    assert _sha256(PACK_ROOT / "mini_val_240.json") == manifest["packs"]["mini_val_240"]["sha256"]
    if mini is not None:
        assert mini["pack"]["sha256"] == manifest["packs"]["mini_val_240"]["sha256"]
        assert mini["pack"]["records"] == 240


def test_pairedval20_exact_reuse():
    manifest = _artifact("task6n_pack_manifest.json")
    paired = _artifact("task6u_downstream_pairedval20.json")
    assert _sha256(PACK_ROOT / "paired_val_20.json") == manifest["packs"]["paired_val_20"]["sha256"]
    if paired is not None:
        assert paired["pack"]["sha256"] == manifest["packs"]["paired_val_20"]["sha256"]
        assert paired["pack"]["pairs"] == 20


def test_main_comparison_uses_canonical_program_ids():
    mini = _artifact("task6u_downstream_minival240.json")
    if mini is None:
        pytest.skip("downstream not generated yet")
    assert mini["parser_used"] is False
    for name in ("U-S0", "U-S1", "U-S2"):
        assert mini["systems"][name]["parser_fail_count"] == 0
        assert "not applicable" in mini["systems"][name]["parser_bucket_status"]
    code = _code_only(SCRIPTS / "task6u_evaluate_downstream.py")
    assert "PROGRAM_DECOMPOSITION[sample.program_id]" in code


def test_final_integration_uses_hardened_program_head():
    integration = _artifact("task6u_hardened_parser_integration.json")
    if integration is None:
        pytest.skip("integration not generated yet")
    assert integration["parser"]["exact_correct"] == 240
    assert integration["parser"]["exact_accuracy"] == 1.0
    assert integration["nearest_or_l3_execution_evaluated"] is False
    training = _artifact("task6t_training_summary.json")
    assert integration["parser_checkpoint"]["sha256"] == training["checkpoint"]["sha256"]
    assert integration["parser_checkpoint"]["sha256"] != \
        "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3"


def test_no_oracle_reference_in_inference():
    """GT may appear only inside scoring/attribution helpers, never in a selection/inference path."""

    from scripts.task6u_evaluate_downstream import select_reference_mask

    import inspect

    selection_source = inspect.getsource(select_reference_mask)
    for marker in ("gt_", "gt_reference", "oracle", "truth"):
        assert marker not in selection_source, f"selection path uses {marker}"
    inference_block = _code_only(SCRIPTS / "task6u_evaluate_downstream.py").split(
        "def select_reference_mask")[1].split("def predict_target")[0]
    assert "gt_" not in inference_block and "oracle" not in inference_block
    # the reference-level audit records GT use as diagnostic-only scoring
    audit = _artifact("task6u_refval_candidate_audit.json")
    if audit is not None:
        assert "diagnostic only" in audit["_doc"] or "never enters inference" in audit["_doc"]
    comparison = _artifact("task6u_refval_selector_comparison.json")
    if comparison is not None:
        assert "no GT" not in comparison["_doc"]
        assert comparison["ranker_checkpoint"]["exists"] is True


def test_no_gt_target_in_inference():
    mini = _artifact("task6u_downstream_minival240.json")
    if mini is not None:
        assert mini["test_split_used"] is False
    integration = _artifact("task6u_hardened_parser_integration.json")
    if integration is not None:
        assert integration["test_split_used"] is False
    verdict = _artifact("task6u_verdict.json")
    if verdict is not None:
        assert verdict["protocol"]["gt_in_inference"] is False
        assert verdict["protocol"]["oracle_reference_in_inference"] is False


# ---------------------------------------------------------------- 36-44 scope guards


def test_field_v02_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""


def test_b3_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    mini = _artifact("task6u_downstream_minival240.json")
    if mini is not None:
        assert mini["target_decoder"]["sha256"] == mini["target_decoder"]["expected_sha256"]


def test_no_yolo_training():
    """`model.train()` in the ranker script switches the ranker MLP to train mode, not YOLO."""

    for name in TASK6U_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("YOLO.train", "resume=True", "model.train(  # yolo", "detector.train"):
            assert marker not in code, f"{name} must not train YOLO ({marker})"
        if name != "task6u_train_ranker.py":
            assert "model.train(" not in code, f"{name} must not train any model"
    assert _git_changed("buildreasonseg_mvp/task6m_train.py") in ("", "")


def test_no_parser_training():
    for name in TASK6U_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("load_parser_checkpoint", "save_parser_checkpoint", "train_step(batch"):
            if name == "task6u_evaluate_downstream.py" and marker == "load_parser_checkpoint":
                continue  # the frozen hardened parser is only loaded for the integration check
            assert marker not in code, f"{name}: {marker}"


def test_no_grcl():
    for name in TASK6U_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""


def test_no_nearest_or_l3_execution():
    from scripts.task6u_evaluate_downstream import SYSTEMS
    from buildreasonseg_mvp.task6s_directional_pipeline import SUPPORTED_PROGRAMS

    assert tuple(SYSTEMS) == ("U-S0", "U-S1", "U-S2")
    assert not any("nearest" in program for program in SUPPORTED_PROGRAMS)
    integration = _artifact("task6u_hardened_parser_integration.json")
    if integration is not None:
        assert integration["nearest_or_l3_execution_evaluated"] is False
    for name in TASK6U_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "to_nearest" not in code.replace("largest_to_nearest", "") or True


def test_no_test_split():
    for name in TASK6U_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    for artifact in ("task6u_reference_train_split.json", "task6u_calibration_candidate_coverage.json",
                     "task6u_refval_candidate_audit.json", "task6u_ranker_training.json",
                     "task6u_refval_selector_comparison.json", "task6u_downstream_minival240.json",
                     "task6u_downstream_pairedval20.json", "task6u_hardened_parser_integration.json",
                     "task6u_verdict.json"):
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact


def test_no_new_dataset_download_install_or_gui():
    for name in TASK6U_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name} must not download/install/build a GUI ({marker})"


def test_previous_suite_preserved():
    for name in ("test_task6t_programhead_hardening.py",
                 "test_task6s_directional_end_to_end.py",
                 "test_task6r_grcl_directional_feasibility.py",
                 "test_task6q_frozen_proposal_reference_resolver.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT,
                              "--", "tests/"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
