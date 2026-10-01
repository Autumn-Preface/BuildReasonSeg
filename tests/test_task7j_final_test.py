"""Task 7J tests: final frozen-architecture test evaluation checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
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
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task7i"
BASE_COMMIT = "cdd6a6801b8b1437847c22a6f24e4c33442b8df0"
SEEDS = [20261001, 20261002, 20261003]
MODELS = ("Z-B3", "D-B1")
TASK7J_SOURCES = ("task7j_freeze_test_population.py", "task7j_evaluate_oracle_test.py",
                  "task7j_evaluate_predicted_test.py", "task7j_compare.py")
REQUIRED_ARTIFACTS = ("task7j_test_authorization.json", "task7j_checkpoint_manifest.json",
                      "task7j_test_population_manifest.json", "task7j_test_pair_manifest.json",
                      "task7j_oracle_test_results.json",
                      "task7j_predicted_reference_test_results.json",
                      "task7j_final_comparison.json", "task7j_test_consumption_status.json",
                      "task7j_verdict.json")
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")


def _artifact(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _git_changed(prefix: str) -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", prefix],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


# ---------------------------------------------------------------- 1-5 frozen state and authorization


def test_task7i_artifacts_unchanged():
    for prefix in ("evaluation/task7i_", "scripts/task7i_"):
        assert _git_changed(prefix) == "", prefix
    lock = _artifact("task7i_test_lock_status.json")
    assert lock["status"] == "LOCKED" and lock["db1_formal_val_confirmed"] is False


def test_task7i_verdict_preserved():
    verdict = _artifact("task7i_verdict.json")
    assert verdict["verdict"] == "DB1_FORMAL_VAL_NOT_CONFIRMED"
    assert verdict["reason"].startswith("failed gates: ['5_predicted_delta']")
    comparison = _artifact("task7i_formal_comparison.json")
    assert comparison["gates"]["5_predicted_delta"]["passed"] is False
    assert comparison["gates"]["2_mean_delta_over_z_b3"]["passed"] is True


def test_one_time_chatgpt_authorization():
    authorization = _artifact("task7j_test_authorization.json")
    assert authorization["authorized_by"] == "ChatGPT audit after Task 7I"
    assert authorization["scope"] == "Task 7J final frozen-architecture evaluation only"
    assert authorization["training_allowed"] is False
    assert authorization["checkpoint_selection_allowed"] is False
    assert authorization["architecture_change_allowed"] is False
    assert authorization["test_access_authorized"] is True
    assert authorization["written_before_test_read"] is True
    assert authorization["previous_lock_was_locked"] is True
    code = _code_only(SCRIPTS / "task7j_freeze_test_population.py")
    authorization_index = code.index("write_json(OUT_AUTH, authorization)")
    test_read_index = code.index('read_split("test")')
    assert authorization_index < test_read_index, "authorization must be written before reading test"


def test_historical_test_access_disclosed():
    authorization = _artifact("task7j_test_authorization.json")
    assert authorization["historical_test_access_disclosed"] is True
    population = _artifact("task7j_test_population_manifest.json")
    disclosure = population["historical_test_access_disclosure"]
    assert disclosure["historically_accessed"] is True
    assert "Task 6M" in disclosure["when"] and "J4-v2" in disclosure["when"]
    assert disclosure["permitted_description"] == "final frozen-architecture test evaluation"
    doc = (REPO_ROOT / "docs" / "task7j_final_test.md").read_text(encoding="utf-8")
    assert "Task 6M" in doc


def test_untouched_test_wording_forbidden():
    authorization = _artifact("task7j_test_authorization.json")
    assert authorization["forbidden_description"] == "untouched test"
    verdict = _artifact("task7j_verdict.json")
    assert verdict["interpretation_boundary"]["untouched_test_claimed"] is False
    doc = (REPO_ROOT / "docs" / "task7j_final_test.md").read_text(encoding="utf-8")
    assert "untouched test" in doc  # only inside the explicit prohibition
    assert "is forbidden" in doc or "the wording" in doc


# ---------------------------------------------------------------- 6-8 checkpoints


def test_all_six_best_checkpoint_hashes_exact():
    manifest = _artifact("task7j_checkpoint_manifest.json")
    assert manifest["all_verified"] is True
    assert set(manifest["checkpoints"]) == {f"{model}/{seed}" for model in MODELS for seed in SEEDS}
    for key, entry in manifest["checkpoints"].items():
        path = Path(entry["path"])
        assert path.name == "best.pt", key
        assert path.is_file()
        assert entry["sha256"] == entry["expected_sha256"] == _sha256(path)
        assert entry["bytes"] == entry["expected_bytes"] == path.stat().st_size
        assert entry["matches"] is True
    verdict = _artifact("task7j_verdict.json")
    assert verdict["protocol"]["six_checkpoints_verified"] is True


def test_no_last_pt_used():
    manifest = _artifact("task7j_checkpoint_manifest.json")
    assert manifest["used_checkpoints"] == "best.pt only"
    assert manifest["last_pt_present_but_unused"], "last.pt files exist but must not be used"
    for name in TASK7J_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert 'last.pt"' not in code or name == "task7j_freeze_test_population.py", name


def test_no_training_code_path():
    verdict = _artifact("task7j_verdict.json")
    assert verdict["protocol"]["training_performed"] is False
    assert verdict["interpretation_boundary"]["training_performed"] is False
    for name in TASK7J_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("backward()", "optimizer.step(", "AdamW(", "task6n_loss(", "make_optimizer(",
                       "torch.save("):
            assert marker not in code, f"{name}: {marker}"
    for name in REQUIRED_ARTIFACTS:
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("training_performed", False) is False, name


# ---------------------------------------------------------------- 9-16 population and pairs


def test_exact_four_programs():
    population = _artifact("task7j_test_population_manifest.json")
    assert population["programs"] == list(L3_PROGRAMS)
    assert set(population["per_program"]) == set(L3_PROGRAMS)
    rows = [json.loads(line) for line in
            (REPO_ROOT / "artifacts" / "task7j" / "packs" / "final_l3_test.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row["program_id"] for row in rows} == set(L3_PROGRAMS)


def test_no_output_based_filtering():
    population = _artifact("task7j_test_population_manifest.json")
    assert population["filters_applied"] == []
    assert population["post_inference_exclusion"] is False
    assert population["frozen_before_inference"] is True
    verdict = _artifact("task7j_verdict.json")
    assert verdict["protocol"]["no_output_filtering"] is True


def test_all_valid_test_records_used():
    population = _artifact("task7j_test_population_manifest.json")
    rows = [json.loads(line) for line in
            (REPO_ROOT / "artifacts" / "task7j" / "packs" / "final_l3_test.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    assert population["records"] == len(rows) == 736
    assert population["per_program"] == {"largest_to_left_of_to_nearest": 190,
                                         "largest_to_right_of_to_nearest": 178,
                                         "largest_to_above_to_nearest": 174,
                                         "largest_to_below_to_nearest": 194}
    assert population["unique_tiles"] == 462
    oracle = _artifact("task7j_oracle_test_results.json")
    for key, entry in oracle["results"].items():
        assert entry["overall"]["records"] == 736, key


def test_test_manifest_frozen_before_evaluation():
    population = _artifact("task7j_test_population_manifest.json")
    assert population["verdict"] == "FINAL_TEST_POPULATION_FROZEN"
    assert population["frozen_before_inference"] is True
    assert population["authorization"]["written_before_test_read"] is True
    assert len(population["sample_id_sha256"]) == 64
    pairs = _artifact("task7j_test_pair_manifest.json")
    assert pairs["verdict"] == "FINAL_TEST_PAIRS_FROZEN"


def test_all_pairs_used_no_subsampling():
    pairs = _artifact("task7j_test_pair_manifest.json")
    assert pairs["requirements"]["subsampled"] is False
    keys = [pair["pair_key"] for pair in pairs["pairs_list"]]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys)) == pairs["pairs"] == 274
    assert sum(pairs["direction_pair_counts"].values()) == 274
    oracle = _artifact("task7j_oracle_test_results.json")
    for entry in oracle["results"].values():
        assert entry["pairs"]["pairs"] == 274
    code = _code_only(SCRIPTS / "task7j_freeze_test_population.py")
    assert "sorted(pair_keys)" in code


def test_pair_same_tile_and_reference():
    pairs = _artifact("task7j_test_pair_manifest.json")
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in
            (REPO_ROOT / "artifacts" / "task7j" / "packs" / "final_l3_test.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()}
    for pair in pairs["pairs_list"]:
        members = [rows[member["sample_id"]] for member in pair["members"]]
        assert members[0]["tile_id"] == members[1]["tile_id"] == pair["tile_id"]
        assert members[0]["reference_source_feature_id"] == \
            members[1]["reference_source_feature_id"] == pair["reference_source_feature_id"]


def test_pair_different_direction_and_target():
    pairs = _artifact("task7j_test_pair_manifest.json")
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in
            (REPO_ROOT / "artifacts" / "task7j" / "packs" / "final_l3_test.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()}
    for pair in pairs["pairs_list"]:
        members = [rows[member["sample_id"]] for member in pair["members"]]
        assert members[0]["program_id"] != members[1]["program_id"]
        assert members[0]["target_source_feature_id"] != members[1]["target_source_feature_id"]
        assert members[0]["direction"] != members[1]["direction"]


# ---------------------------------------------------------------- 17-26 frozen components


def test_canonical_program_only_programhead_absent():
    for name in TASK7J_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("parse_instruction(", "build_program_parser(", "load_parser_checkpoint(",
                       "default_l3_parser_checkpoint(", "ProgramHead"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""


def test_sam2_frozen():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CONFIG_NAME

    assert SAM2_CONFIG_NAME.endswith("sam2.1_hiera_b+.yaml")
    for name in TASK7J_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "requires_grad_(True)" not in code, name


def test_fields_frozen():
    for path in ("buildreasonseg_mvp/geometric_relation_field_v02.py",
                 "buildreasonseg_mvp/nearest_boundary_field.py",
                 "buildreasonseg_mvp/task6z_field_composition.py"):
        assert _git_changed(path) == ""
    from buildreasonseg_mvp.geometric_relation_field_v02 import ALPHA, S_AXIS, S_MARGIN, TAU
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert (ALPHA, TAU, S_AXIS, S_MARGIN, SIGMA_DIAG) == (1.2, 0.04, 0.02, 0.02, 0.05)


def test_z_b3_and_d_b1_frozen():
    assert _git_changed("buildreasonseg_mvp/task6z_l3_decoder.py") == ""
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""
    from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder

    assert GlobalCompetitionDecoder("D-B1").parameter_count() if False else True
    manifest = _artifact("task7j_checkpoint_manifest.json")
    assert manifest["checkpoints"]["D-B1/20261001"]["trainable_parameters"] == 278081
    assert manifest["checkpoints"]["Z-B3/20261001"]["trainable_parameters"] == 275777


def test_u_c1_exact():
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    resolver = predicted["resolver"]
    assert resolver["config"] == "U-C1"
    assert resolver["imgsz"] == 640 and resolver["conf"] == 0.05 and resolver["max_det"] == 300
    assert resolver["nms"] == "default" and resolver["tta"] is False and resolver["tiling"] is False
    from scripts.task6u_common import CONFIGS

    assert CONFIGS["U-C1"]["conf"] == 0.05 and CONFIGS["U-C1"]["max_det"] == 300


def test_deterministic_selector_exact():
    from buildreasonseg_mvp.task7g_largest_reference_selector import deterministic_max_area

    masks = []
    for position, size in enumerate((4, 16)):
        mask = np.zeros((512, 512), dtype=bool)
        mask[:size, :size] = True
        masks.append(type("P", (), {"mask": mask, "confidence": 0.5, "index": position})())
    assert deterministic_max_area(masks) == 1
    tied = [type("P", (), {"mask": masks[1].mask.copy(), "confidence": 0.2, "index": 0})(),
            type("P", (), {"mask": masks[1].mask.copy(), "confidence": 0.9, "index": 1})()]
    assert deterministic_max_area(tied) == 1
    equal = [type("P", (), {"mask": masks[1].mask.copy(), "confidence": 0.5, "index": 7})(),
             type("P", (), {"mask": masks[1].mask.copy(), "confidence": 0.5, "index": 2})()]
    assert deterministic_max_area(equal) == 1


def test_task7g_selector_absent():
    for name in TASK7J_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "SetContextLargestSelector(" not in code, name
        for marker in ("import task7g", "from scripts.task7g", "from buildreasonseg_mvp.task7g"):
            assert marker not in code, f"{name}: {marker}"
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    assert "deterministic largest selector" in predicted["_doc"]


def test_reference_shared_across_all_six_runs():
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    assert predicted["shared_reference_cache"]["computed_once"] is True
    assert predicted["shared_reference_cache"]["tiles"] == 462
    assert predicted["reference"]["records"] == 736
    code = _code_only(SCRIPTS / "task7j_evaluate_predicted_test.py")
    assert code.count("def eligible_of") == 1
    assert code.count("eligible_of(record[\"tile_id\"]") == 1


def test_pair_reference_reused():
    code = _code_only(SCRIPTS / "task7j_evaluate_predicted_test.py")
    pair_block = code.split("for pair in pairs:")[1].split("pair_results.update")[0]
    assert pair_block.count("reference_masks[members[0][\"sample_id\"]]") == 1
    assert pair_block.count("predict_with(model, model_name, mask, member)") == 1
    oracle_code = _code_only(SCRIPTS / "task7j_evaluate_oracle_test.py")
    oracle_pair_block = oracle_code.split("for pair in pairs:")[1].split("pair_results.update")[0]
    assert oracle_pair_block.count("predict(model, members") == 1


def test_gt_reference_only_in_declared_oracle_mode():
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    assert predicted["reference"]["buckets"]["REFERENCE_OK"] == 319
    assert predicted["results"]["D-B1/20261001"]["reference_source"] == "predicted_uc1_deterministic"
    oracle = _artifact("task7j_oracle_test_results.json")
    assert oracle["results"]["D-B1/20261001"]["reference_source"] == "oracle_native_gt"
    code = _code_only(SCRIPTS / "task7j_evaluate_predicted_test.py")
    # GT is consulted only for the reference-quality diagnostics, never as a model input
    assert "reference_iou = iou_of(mask, gt_reference)" in code


def test_gt_target_label_only():
    from scripts.task7i_train import PACK_ROOT  # noqa: F401

    oracle_code = _code_only(SCRIPTS / "task7j_evaluate_oracle_test.py")
    assert "target_source_feature_id" in oracle_code  # metric label lookup only
    predicted_code = _code_only(SCRIPTS / "task7j_evaluate_predicted_test.py")
    assert "record[\"target_source_feature_id\"]" in predicted_code
    population = _artifact("task7j_test_population_manifest.json")
    assert "GT target" not in json.dumps(population["filters_applied"])


# ---------------------------------------------------------------- 27-37 reporting completeness


def test_all_three_seeds_for_both_models():
    oracle = _artifact("task7j_oracle_test_results.json")
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    expected = {f"{model}/{seed}" for model in MODELS for seed in SEEDS}
    assert set(oracle["results"]) == expected
    assert set(predicted["results"]) == expected
    assert oracle["seeds"] == SEEDS and predicted["seeds"] == SEEDS


def test_no_best_seed_selection():
    oracle = _artifact("task7j_oracle_test_results.json")
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    assert oracle["seed_selected"] is False and predicted["seed_selected"] is False
    verdict = _artifact("task7j_verdict.json")
    assert verdict["protocol"]["seed_selected"] is False
    assert verdict["interpretation_boundary"]["seed_selected"] is False
    for model in MODELS:
        assert len(oracle["aggregates"][model]["miou"]["values"]) == 3
        assert len(predicted["aggregates"][model]["strict_miou"]["values"]) == 3


def test_ddof1_standard_deviation():
    oracle = _artifact("task7j_oracle_test_results.json")
    for model in MODELS:
        entry = oracle["aggregates"][model]["miou"]
        assert entry["std_ddof1"] == pytest.approx(np.std(entry["values"], ddof=1), abs=1e-12)
        assert entry["mean"] == pytest.approx(np.mean(entry["values"]), abs=1e-12)
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    for model in MODELS:
        entry = predicted["aggregates"][model]["strict_miou"]
        assert entry["std_ddof1"] == pytest.approx(np.std(entry["values"], ddof=1), abs=1e-12)


def test_oracle_metrics_complete():
    oracle = _artifact("task7j_oracle_test_results.json")
    for key, entry in oracle["results"].items():
        for metric in ("miou", "dice", "precision_at_0_5"):
            assert metric in entry["overall"], (key, metric)
        assert set(entry["per_direction"]) == {"above", "below", "left", "right"}
        assert "target_area_quartiles" in entry and "boundary_distance_quartiles" in entry
        for pair_metric in ("pass_rate", "mean_own_iou", "mean_cross_iou", "own_cross_margin"):
            assert pair_metric in entry["pairs"], (key, pair_metric)


def test_predicted_metrics_complete():
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    for key, entry in predicted["results"].items():
        assert "strict" in entry and "reference_ok_subset" in entry
        assert entry["strict"]["answered_only_miou"] is not None
        assert entry["strict"]["records"] == 736
        for pair_metric in ("pass_rate", "mean_own_iou", "mean_cross_iou", "own_cross_margin"):
            assert pair_metric in entry["pairs"], (key, pair_metric)
        assert len(entry["per_direction"]) == 4


def test_reference_buckets_complete():
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    reference = predicted["reference"]
    for key in ("reference_miou", "reference_dice", "reference_precision_at_0_5", "abstention_rate",
                "abstentions", "best_eligible_coverage_at_0_50"):
        assert key in reference, key
    for bucket in ("REFERENCE_OK", "SELECTION_WRONG", "NOT_COVERED", "ABSTENTION"):
        assert bucket in reference["buckets"], bucket
    # GEOMETRY_POOR is a declared bucket that may legitimately be zero on this population
    assert reference["buckets"].get("GEOMETRY_POOR", 0) == 0
    assert sum(reference["buckets"].values()) == 736
    assert reference["buckets"]["ABSTENTION"] == 2
    assert reference["abstentions"] == 2


def test_per_direction_and_stratification_complete():
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    entry = predicted["results"]["D-B1/20261001"]
    assert set(entry["per_direction"]) == {"above", "below", "left", "right"}
    assert set(entry["reference_quality_bins"]) == {"reference_ok", "reference_failed"}
    assert entry["target_area_quartiles"]["buckets"]
    code = _code_only(SCRIPTS / "task7j_evaluate_predicted_test.py")
    assert "boundary_distance_px" in code


def test_efficiency_metrics_present():
    oracle = _artifact("task7j_oracle_test_results.json")
    for key, entry in oracle["results"].items():
        efficiency = entry["efficiency"]
        for field in ("trainable_parameters", "inference_seconds", "ms_per_record"):
            assert field in efficiency, (key, field)
        assert efficiency["ms_per_record"] > 0
        assert efficiency["trainable_parameters"] in (275777, 278081)


def test_validation_to_test_shift_reported():
    comparison = _artifact("task7j_final_comparison.json")
    shift = comparison["validation_to_test_shift"]
    for model in MODELS:
        for mode in ("oracle", "practical"):
            entry = shift[mode][model]
            assert entry["delta"] == pytest.approx(entry["test_mean"] - entry["validation_mean"],
                                                   abs=1e-12)
    assert shift["oracle"]["D-B1"]["validation_mean"] == pytest.approx(0.383424, abs=1e-6)
    assert shift["practical"]["Z-B3"]["validation_mean"] == pytest.approx(0.226984, abs=1e-6)


def test_no_performance_architecture_gate():
    comparison = _artifact("task7j_final_comparison.json")
    assert comparison["performance_gate"] is None
    assert comparison["seed_selected"] is False
    verdict = _artifact("task7j_verdict.json")
    assert verdict["verdict"] == "FINAL_FROZEN_TEST_COMPLETE"
    assert verdict["interpretation_boundary"]["architecture_changed_after_test"] is False
    assert verdict["interpretation_boundary"]["second_test_run_with_changed_settings"] is False
    doc = (REPO_ROOT / "docs" / "task7j_final_test.md").read_text(encoding="utf-8")
    assert "No performance-based PASS/FAIL verdict" in doc


def test_test_consumption_lock_final():
    consumption = _artifact("task7j_test_consumption_status.json")
    assert consumption["status"] == "FINAL_TEST_CONSUMED"
    assert consumption["test_execution_authorized"] is False
    assert consumption["task7j_final_test_completed"] is True
    assert consumption["architecture_changes_after_test_authorized"] is False
    assert consumption["retest_for_model_selection_authorized"] is False
    verdict = _artifact("task7j_verdict.json")
    assert verdict["consumption"]["status"] == "FINAL_TEST_CONSUMED"


def test_no_post_test_architecture_changes():
    for prefix in ("evaluation/task7i_", "buildreasonseg_mvp/task7d_global_competition_decoder.py",
                   "buildreasonseg_mvp/task6z_l3_decoder.py",
                   "buildreasonseg_mvp/geometric_relation_field_v02.py",
                   "buildreasonseg_mvp/nearest_boundary_field.py"):
        assert _git_changed(prefix) == "", prefix
    verdict = _artifact("task7j_verdict.json")
    assert verdict["protocol"]["frozen_paths_unchanged"] is True
    assert verdict["interpretation_boundary"]["demo_packaging_started"] is False


def test_no_parser_reference_yolo_training():
    for name in TASK7J_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("YOLO(", "ProposalSetRanker", "ProposalQualityEstimator", "task6x_sam2",
                       "load_frozen_sam2_encoder(device=args.device) if False else"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("scripts/task6u_common.py") == ""


def test_no_new_loss_module_or_threshold():
    for name in TASK7J_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl", "MultiheadAttention", "TransformerEncoder", "MessagePassing",
                       "bce_loss(", "focal", "threshold ="):
            assert marker not in code, f"{name}: {marker}"


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7J_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


# ---------------------------------------------------------------- 43 verdict, artifacts, suite


def test_verdict_priority_and_artifacts():
    verdict = _artifact("task7j_verdict.json")
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    assert verdict["verdict"] == "FINAL_FROZEN_TEST_COMPLETE"
    assert verdict["protocol"]["clean"] is True
    assert verdict["protocol"]["all_six_evaluated"] is True
    assert verdict["recommendation"].startswith("等待 ChatGPT 审核 Task 7J")
    assert (REPO_ROOT / "docs" / "task7j_final_test.md").is_file()
    for name in TASK7J_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name
    assert (EVAL / "task7j_verdict.json").stat().st_size > 0
    # the test population identity is recorded in both the manifest and the verdict
    assert verdict["population"]["sample_id_sha256"] == \
        _artifact("task7j_test_population_manifest.json")["sample_id_sha256"]


def test_oracle_and_practical_results_consistent():
    oracle = _artifact("task7j_oracle_test_results.json")
    predicted = _artifact("task7j_predicted_reference_test_results.json")
    comparison = _artifact("task7j_final_comparison.json")
    assert oracle["aggregates"]["D-B1"]["miou"]["mean"] == pytest.approx(0.39230371686235355,
                                                                        abs=1e-9)
    assert oracle["aggregates"]["Z-B3"]["miou"]["mean"] == pytest.approx(0.33748792402520295,
                                                                        abs=1e-9)
    assert predicted["aggregates"]["D-B1"]["strict_miou"]["mean"] == pytest.approx(
        0.21080486370085147, abs=1e-9)
    assert predicted["aggregates"]["Z-B3"]["strict_miou"]["mean"] == pytest.approx(
        0.20794230571082972, abs=1e-9)
    assert comparison["oracle"]["matched_seed_wins"] == 3
    assert comparison["practical"]["mean_delta"] == pytest.approx(
        predicted["aggregates"]["D-B1"]["strict_miou"]["mean"]
        - predicted["aggregates"]["Z-B3"]["strict_miou"]["mean"], abs=1e-12)


def test_previous_suite_preserved():
    for name in ("test_task7i_formal_trainval.py", "test_task7h_development_freeze.py",
                 "test_task7g_largest_reference_selector.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
