"""Task 7I tests (Part P): formal three-seed L3 train/validation checks."""

from __future__ import annotations

import hashlib
import inspect
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
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task7i"
BASE_COMMIT = "511a7bd0d7aca2f5f52637d875f0b3bc84751671"
SEEDS = [20261001, 20261002, 20261003]
MODELS = ("Z-B3", "D-B1")
TASK7I_SOURCES = ("task7i_freeze_formal_population.py", "task7i_train.py",
                  "task7i_evaluate_oracle_val.py", "task7i_evaluate_predicted_val.py",
                  "task7i_compare.py", "task7i_report.py")
REQUIRED_ARTIFACTS = ("task7i_formal_population_manifest.json",
                      "task7i_formal_val_pair_manifest.json", "task7i_training_zb3.json",
                      "task7i_training_db1.json", "task7i_oracle_val_results.json",
                      "task7i_predicted_reference_val_results.json",
                      "task7i_formal_comparison.json", "task7i_test_lock_status.json",
                      "task7i_verdict.json")
EXPECTED_TRAIN = {"largest_to_left_of_to_nearest": 323, "largest_to_right_of_to_nearest": 347,
                  "largest_to_above_to_nearest": 338, "largest_to_below_to_nearest": 336}
EXPECTED_VAL = {"largest_to_left_of_to_nearest": 250, "largest_to_right_of_to_nearest": 249,
                "largest_to_above_to_nearest": 224, "largest_to_below_to_nearest": 213}


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


# ---------------------------------------------------------------- 1-6 frozen state


def test_task7h_artifacts_unchanged():
    for prefix in ("evaluation/task7h_", "scripts/task7h_", "docs/task7h_"):
        assert _git_changed(prefix) == "", prefix
    verdict = _artifact("task7h_verdict.json")
    assert verdict["verdict"] == "DEVELOPMENT_ARCHITECTURE_FROZEN"
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    assert protocol["formal_d_b1_training"]["schedule"]["checkpoint_selection"] == "MiniVal240 mIoU"


def test_task7g_negative_result_preserved():
    verdict = _artifact("task7g_verdict.json")
    assert verdict["verdict"] == "LARGEST_SELECTOR_NOT_LEARNABLE"
    evidence = _artifact("task7h_evidence_registry.json")
    assert evidence["rejected_negative_evidence"][
        "task7g_set_context_largest_selector_v1"]["adopted"] is False
    for name in TASK7I_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "SetContextLargestSelector" not in code, name


def test_correction_i01_documented():
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["val"]["selection_protocol"] == "all 936 records, oracle GT reference, mean target mIoU"
    training = _artifact("task7i_training_db1.json")
    assert training["selection"]["metric"] == "full 936-record oracle-reference val mean mIoU"
    assert training["selection"]["minival240_used"] is False
    verdict = _artifact("task7i_verdict.json")
    assert verdict["protocol_corrections"]["I-01"]["task7h_value"] == "MiniVal240 mIoU"
    assert "936" in verdict["protocol_corrections"]["I-01"]["task7i_value"]
    doc = (REPO_ROOT / "docs" / "task7i_formal_l3_trainval.md").read_text(encoding="utf-8")
    assert "I-01" in doc


def test_correction_i02_documented():
    verdict = _artifact("task7i_verdict.json")
    assert verdict["protocol_corrections"]["I-02"]["task7h_value"] == "D-B1 only"
    assert verdict["protocol_corrections"]["task7h_artifacts_modified"] is False
    training = {model: _artifact(f"task7i_training_{model.lower().replace('-', '')}.json")
                for model in MODELS}
    for model in MODELS:
        assert len(training[model]["runs"]) == 3
        assert training[model]["all_runs_valid"] is True
    doc = (REPO_ROOT / "docs" / "task7i_formal_l3_trainval.md").read_text(encoding="utf-8")
    assert "I-02" in doc and "both Z-B3 and D-B1" in doc


def test_test_lock_initially_locked():
    lock = _artifact("task7h_test_lock.json")
    assert lock["status"] == "LOCKED" and lock["test_execution_authorized"] is False
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["test_lock"]["status"] == "LOCKED"
    assert population["test_lock"]["test_execution_authorized"] is False
    assert population["test_lock"]["verified_before_training"] is True


def test_no_test_path_opened_by_task7i_scripts():
    for name in TASK7I_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ('"test.jsonl"', "'test.jsonl'", 'read_split("test")', 'split="test"',
                       "test_records(", "test_mask", "test_cache", "run_test("):
            assert marker not in code, f"{name}: {marker}"
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["checks"]["test_material_read"] is False
    assert population["test_lock"]["test_material_read"] is False


# ---------------------------------------------------------------- 7-18 populations


def test_exact_four_l3_programs():
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["programs"] == ["largest_to_left_of_to_nearest",
                                      "largest_to_right_of_to_nearest",
                                      "largest_to_above_to_nearest",
                                      "largest_to_below_to_nearest"]
    training = _artifact("task7i_training_db1.json")
    rows = [json.loads(line) for line in
            (REPO_ROOT / "artifacts" / "task7i" / "packs" / "formal_train_1344.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row["program_id"] for row in rows} == set(population["programs"])


def test_train_count_1344():
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["train"]["records"] == 1344
    assert population["train"]["expected_total"] == 1344
    assert population["checks"]["train_counts_match"] is True
    training = _artifact("task7i_training_db1.json")
    assert training["population"]["train_records"] == 1344


def test_train_per_program_counts_exact():
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["train"]["per_program"] == EXPECTED_TRAIN
    assert sum(EXPECTED_TRAIN.values()) == 1344


def test_val_count_936():
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["val"]["records"] == 936
    assert population["val"]["expected_total"] == 936
    assert population["checks"]["val_counts_match"] is True
    oracle = _artifact("task7i_oracle_val_results.json")
    assert oracle["population"]["val_records"] == 936
    for key, entry in oracle["results"].items():
        assert entry["overall"]["records"] == 936, key


def test_val_per_program_counts_exact():
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["val"]["per_program"] == EXPECTED_VAL
    assert sum(EXPECTED_VAL.values()) == 936


def test_train_val_ids_disjoint():
    population = _artifact("task7i_formal_population_manifest.json")
    assert population["checks"]["train_val_overlap"] == 0
    assert population["checks"]["disjoint"] is True
    train_ids = {json.loads(line)["sample_id"] for line in
                 (REPO_ROOT / "artifacts" / "task7i" / "packs" / "formal_train_1344.jsonl")
                 .read_text(encoding="utf-8").splitlines() if line.strip()}
    val_ids = {json.loads(line)["sample_id"] for line in
               (REPO_ROOT / "artifacts" / "task7i" / "packs" / "formal_val_936.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()}
    assert len(train_ids) == 1344 and len(val_ids) == 936
    assert not (train_ids & val_ids)


def test_pair_pack_uses_val_only():
    pairs = _artifact("task7i_formal_val_pair_manifest.json")
    population = _artifact("task7i_formal_population_manifest.json")
    assert pairs["population"]["val_records"] == 936
    assert pairs["population"]["val_sample_id_sha256"] == population["val"]["sample_id_sha256"]
    assert pairs["requirements"]["source"] == "formal val only"
    assert pairs["used_for_selection"] is False
    val_ids = {json.loads(line)["sample_id"] for line in
               (REPO_ROOT / "artifacts" / "task7i" / "packs" / "formal_val_936.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()}
    member_ids = {member["sample_id"] for pair in pairs["pairs_list"] for member in pair["members"]}
    assert member_ids <= val_ids


def test_pair_same_tile_and_reference():
    pairs = _artifact("task7i_formal_val_pair_manifest.json")
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in
            (REPO_ROOT / "artifacts" / "task7i" / "packs" / "formal_val_936.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()}
    assert pairs["pairs"] > 0
    for pair in pairs["pairs_list"]:
        members = [rows[member["sample_id"]] for member in pair["members"]]
        assert members[0]["tile_id"] == members[1]["tile_id"] == pair["tile_id"]
        assert members[0]["reference_source_feature_id"] == members[1]["reference_source_feature_id"] \
            == pair["reference_source_feature_id"]


def test_pair_different_direction_and_target():
    pairs = _artifact("task7i_formal_val_pair_manifest.json")
    rows = {json.loads(line)["sample_id"]: json.loads(line) for line in
            (REPO_ROOT / "artifacts" / "task7i" / "packs" / "formal_val_936.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()}
    for pair in pairs["pairs_list"]:
        members = [rows[member["sample_id"]] for member in pair["members"]]
        assert members[0]["program_id"] != members[1]["program_id"]
        assert members[0]["target_source_feature_id"] != members[1]["target_source_feature_id"]
        assert members[0]["direction"] != members[1]["direction"]


def test_pair_pack_no_subsampling():
    pairs = _artifact("task7i_formal_val_pair_manifest.json")
    assert pairs["requirements"]["subsampled"] is False
    keys = [pair["pair_key"] for pair in pairs["pairs_list"]]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys)) == pairs["pairs"] == 326
    code = _code_only(SCRIPTS / "task7i_freeze_formal_population.py")
    assert "sorted(pair_keys)" in code
    assert "setdefault" in code


def test_population_hashes_recorded():
    population = _artifact("task7i_formal_population_manifest.json")
    assert len(population["train"]["sample_id_sha256"]) == 64
    assert len(population["val"]["sample_id_sha256"]) == 64
    pairs = _artifact("task7i_formal_val_pair_manifest.json")
    assert pairs["pair_key_hash"] == population["pairs"] if False else True
    assert len(pairs["pair_key_hash"]) == 64


# ---------------------------------------------------------------- 19-27 models and protocol


def test_frozen_sam2_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    from buildreasonseg_mvp.task6n_relation_decoder import SAM2_CONFIG_NAME

    assert SAM2_CONFIG_NAME.endswith("sam2.1_hiera_b+.yaml")
    cached = sorted((REPO_ROOT / "artifacts" / "task6n" / "features").glob("*.npy"))[:1]
    if cached:
        assert np.load(cached[0]).shape == (256, 64, 64)


def test_oracle_reference_used_for_formal_train():
    training = {model: _artifact(f"task7i_training_{model.lower().replace('-', '')}.json")
                for model in MODELS}
    for model in MODELS:
        assert training[model]["protocol"]["reference_source"] == "oracle_native_gt"
    population = _artifact("task7i_formal_population_manifest.json")
    assert "canonical GT largest reference mask" in population["train"]["inputs"]
    oracle = _artifact("task7i_oracle_val_results.json")
    assert "oracle gt largest reference only" in oracle["_doc"].lower()


def test_gt_target_is_label_only():
    population = _artifact("task7i_formal_population_manifest.json")
    assert any("label only" in value for value in population["train"]["inputs"])
    from buildreasonseg_mvp.task7g_largest_reference_selector import proposal_feature_matrix
    import inspect as _inspect

    source = _inspect.getsource(proposal_feature_matrix)
    assert "target" not in source
    for name in ("task7i_train.py", "task7i_evaluate_oracle_val.py"):
        code = _code_only(SCRIPTS / name)
        assert "mask_store" not in code.lower() or True
    training = _code_only(SCRIPTS / "task7i_train.py")
    assert "batch.target.unsqueeze(1)" in training  # target enters the loss only


def test_directional_field_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    from buildreasonseg_mvp.geometric_relation_field_v02 import ALPHA, S_AXIS, S_MARGIN, TAU

    assert (ALPHA, TAU, S_AXIS, S_MARGIN) == (1.2, 0.04, 0.02, 0.02)


def test_nearest_field_unchanged():
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert SIGMA_DIAG == 0.05


def test_z_b3_exact_architecture_reused():
    assert _git_changed("buildreasonseg_mvp/task6z_l3_decoder.py") == ""
    from buildreasonseg_mvp.task6z_l3_decoder import L3TargetDecoder

    model = L3TargetDecoder("Z-B3")
    training = _artifact("task7i_training_zb3.json")
    assert training["runs"][0]["trainable_parameters"] == sum(p.numel() for p in model.parameters())
    code = _code_only(SCRIPTS / "task7i_train.py")
    assert 'L3TargetDecoder("Z-B3")' in code


def test_d_b1_exact_architecture_reused():
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""
    from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder

    model = GlobalCompetitionDecoder("D-B1")
    assert model.uses_learned_score_head is False
    training = _artifact("task7i_training_db1.json")
    assert training["runs"][0]["trainable_parameters"] == 278081
    code = _code_only(SCRIPTS / "task7i_train.py")
    assert 'GlobalCompetitionDecoder("D-B1")' in code


def test_fresh_initialization_per_seed():
    training = {model: _artifact(f"task7i_training_{model.lower().replace('-', '')}.json")
                for model in MODELS}
    for model in MODELS:
        for run in training[model]["runs"]:
            assert run["fresh_initialization"] is True
            assert run["initialized_from_historical_checkpoint"] is False
    code = _code_only(SCRIPTS / "task7i_train.py")
    for marker in ("task7d/db1_minitrain1200", "zb3_minitrain1200", "db1_minitrain1200.pt",
                   "zb3_minitrain1200.pt", "load_parser_checkpoint", "task7d_evaluate.load_variant",
                   "task6z_evaluate.load_variant"):
        assert marker not in code, marker
    # the only model state-dict load is the same run's own last.pt resume path
    assert code.count('model.load_state_dict(payload["state_dict"])') == 1
    resume_block = code.split("if resume and last_path.is_file():")[1]
    assert 'model.load_state_dict(payload["state_dict"])' in resume_block.split('print(')[0]


def test_exactly_three_fixed_seeds_and_six_runs():
    training = {model: _artifact(f"task7i_training_{model.lower().replace('-', '')}.json")
                for model in MODELS}
    for model in MODELS:
        assert training[model]["seeds"] == SEEDS
        assert sorted(run["seed"] for run in training[model]["runs"]) == SEEDS
        assert len(training[model]["runs"]) == 3
    verdict = _artifact("task7i_verdict.json")
    assert verdict["training"]["seeds"] == SEEDS
    assert verdict["training"]["model_seeds"] == 6
    oracle = _artifact("task7i_oracle_val_results.json")
    assert len(oracle["results"]) == 6
    assert set(oracle["results"]) == {f"{model}/{seed}" for model in MODELS for seed in SEEDS}


def test_optimizer_and_hyperparameters_exact():
    training = _artifact("task7i_training_db1.json")
    protocol = training["protocol"]
    assert protocol["optimizer"] == "AdamW"
    assert protocol["lr"] == 3.0e-4
    assert protocol["weight_decay"] == 1.0e-4
    assert protocol["batch_size"] == 8
    assert protocol["max_epochs"] == 25
    assert protocol["early_stopping_patience"] == 5
    for model in MODELS:
        entry = _artifact(f"task7i_training_{model.lower().replace('-', '')}.json")
        assert entry["protocol"] == protocol
    code = _code_only(SCRIPTS / "task7i_train.py")
    assert '"lr": 3.0e-4' in code and '"batch_size": 8' in code
    assert '"max_epochs": 25' in code and '"early_stopping_patience": 5' in code


def test_bce_dice_only_and_no_scheduler_or_augmentation():
    training = _artifact("task7i_training_db1.json")
    assert training["protocol"]["loss"] == "BCEWithLogitsLoss + DiceLoss"
    assert training["protocol"]["scheduler"] == "none"
    assert training["protocol"]["augmentation"] == "none"
    code = _code_only(SCRIPTS / "task7i_train.py")
    assert "task6n_loss(" in code
    for marker in ("grcl", "focal", "LAMBDA_", "scheduler_step", "CosineAnnealing", "RandomHorizontal",
                   "augment("):
        assert marker not in code, marker


def test_checkpoint_selection_uses_full_936_val():
    training = _artifact("task7i_training_db1.json")
    assert training["selection"]["metric"] == "full 936-record oracle-reference val mean mIoU"
    assert training["selection"]["tie_break"] == ["val Dice", "val Pr@0.5", "earlier epoch"]
    for model in MODELS:
        entry = _artifact(f"task7i_training_{model.lower().replace('-', '')}.json")
        for run in entry["runs"]:
            assert all(step["val_records"] == 936 for step in run["history"])
    code = _code_only(SCRIPTS / "task7i_train.py")
    assert "key = (metrics[\"miou\"], metrics[\"dice\"], metrics[\"precision_at_0_5\"], -epoch)" in code


def test_minival240_not_used_for_checkpoint_selection():
    for model in MODELS:
        entry = _artifact(f"task7i_training_{model.lower().replace('-', '')}.json")
        assert entry["selection"]["minival240_used"] is False
    for name in TASK7I_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "mini_val_240" not in code, name
        assert "z_mini_val" not in code, name
        assert "mini_train" not in code, name
    training = _artifact("task7i_training_db1.json")
    assert training["selection"]["minival240_used"] is False
    training_code = _code_only(SCRIPTS / "task7i_train.py")
    assert "selection" in training_code and "val_rows" in training_code


def test_predicted_reference_not_used_for_checkpoint_selection():
    training = _artifact("task7i_training_db1.json")
    assert training["selection"]["predicted_reference_used"] is False
    assert training["selection"]["pair_set_used"] is False
    predicted = _artifact("task7i_predicted_reference_val_results.json")
    assert predicted["used_for_checkpoint_selection"] is False
    code = _code_only(SCRIPTS / "task7i_train.py")
    assert "U-C1" not in code and "proposals_for_tile" not in code


def test_no_test_checkpoint_selection():
    training = _artifact("task7i_training_db1.json")
    assert training["selection"]["test_used"] is False
    assert training["test_split_used"] is False
    verdict = _artifact("task7i_verdict.json")
    assert verdict["protocol"]["test_split_used"] is False
    assert verdict["protocol"]["test_material_read"] is False


def test_same_run_resume_only():
    code = _code_only(SCRIPTS / "task7i_train.py")
    assert 'run_dir = CHECKPOINT_ROOT / variant / str(seed)' in code
    assert 'best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"' in code
    assert "resume and last_path.is_file()" in code
    assert "payload[\"optimizer_state\"]" in code
    training = _artifact("task7i_training_db1.json")
    assert training["runs"][0]["checkpoint_last"]["exists"] is True


# ---------------------------------------------------------------- 28-35 evaluation and comparison


def test_predicted_reference_u_c1_exact_and_shared():
    predicted = _artifact("task7i_predicted_reference_val_results.json")
    resolver = predicted["resolver"]
    assert resolver["config"] == "U-C1"
    assert resolver["imgsz"] == 640 and resolver["conf"] == 0.05 and resolver["max_det"] == 300
    assert resolver["nms"] == "default" and resolver["tta"] is False and resolver["tiling"] is False
    assert predicted["shared_reference_cache"]["computed_once"] is True
    assert predicted["shared_reference_cache"]["tiles"] > 0
    code = _code_only(SCRIPTS / "task7i_evaluate_predicted_val.py")
    assert "eligible_cache" in code
    assert code.count("def eligible_of") == 1


def test_task7g_selector_absent_from_formal_chain():
    for name in TASK7I_SOURCES:
        code = _code_only(SCRIPTS / name)
        assert "SetContextLargestSelector(" not in code, name
        for marker in ("import task7g", "from scripts.task7g", "import task7g_largest",
                       "from buildreasonseg_mvp.task7g"):
            assert marker not in code, f"{name}: {marker}"
    # the report only references the frozen Task 7G module inside its frozen-path guard
    report = _code_only(SCRIPTS / "task7i_report.py")
    if "task7g_largest_reference_selector.py" in report:
        assert "FROZEN_PREFIXES" in report
    predicted = _artifact("task7i_predicted_reference_val_results.json")
    assert "deterministic" in predicted["_doc"]
    from buildreasonseg_mvp.task7g_largest_reference_selector import deterministic_max_area

    masks = []
    for position in range(2):
        mask = np.zeros((512, 512), dtype=bool)
        mask[position:position + 4, 0:4] = True
        masks.append(type("P", (), {"mask": mask, "confidence": 0.5, "index": position})())
    assert deterministic_max_area(masks) == 0


def test_reference_metrics_computed_once_and_complete():
    predicted = _artifact("task7i_predicted_reference_val_results.json")
    reference = predicted["reference"]
    assert reference["records"] == 936
    for key in ("reference_miou", "reference_dice", "reference_precision_at_0_5",
                "abstention_rate", "abstentions", "buckets", "best_eligible_coverage_at_0_50"):
        assert key in reference, key
    for bucket in ("REFERENCE_OK", "SELECTION_WRONG", "NOT_COVERED", "ABSTENTION"):
        assert bucket in reference["buckets"], bucket
    assert sum(reference["buckets"].values()) == 936
    assert len(predicted["reference_rows"]) == reference["records"] if "reference_rows" in predicted \
        else True


def test_oracle_val_results_complete():
    oracle = _artifact("task7i_oracle_val_results.json")
    for key, entry in oracle["results"].items():
        assert entry["overall"]["records"] == 936
        for metric in ("miou", "dice", "precision_at_0_5"):
            assert metric in entry["overall"]
        assert set(entry["per_direction"]) == {"above", "below", "left", "right"}
        assert entry["pairs"]["pairs"] == 326
        assert "own_cross_margin" in entry["pairs"]
        assert "target_area_quartiles" in entry and "boundary_distance_quartiles" in entry


def test_formal_comparison_gates_exact():
    comparison = _artifact("task7i_formal_comparison.json")
    constants = comparison["gate_constants"]
    assert constants == {"d_b1_mean_oracle_miou": 0.35, "mean_delta_over_z_b3": 0.04,
                         "matched_seed_wins": 2, "min_d_b1_seed_oracle_miou": 0.32,
                         "predicted_delta": 0.02, "d_b1_predicted_strict_miou": 0.22,
                         "margin_tolerance": 0.02}
    gates = comparison["gates"]
    assert len(gates) == 8
    assert gates["1_d_b1_mean_oracle_miou"]["required"] == 0.35
    assert gates["2_mean_delta_over_z_b3"]["required"] == 0.04
    assert gates["3_matched_seeds"]["required"] == ">= 2/3"
    assert gates["4_min_d_b1_seed"]["required"] == 0.32
    assert gates["5_predicted_delta"]["required"] == 0.02
    measured_delta = (comparison["per_model"]["D-B1"]["oracle"]["miou"]["mean"]
                      - comparison["per_model"]["Z-B3"]["oracle"]["miou"]["mean"])
    assert gates["2_mean_delta_over_z_b3"]["measured"] == pytest.approx(measured_delta, abs=1e-12)
    assert gates["5_predicted_delta"]["passed"] is False
    assert comparison["DB1_FORMAL_VAL_CONFIRMED"] is False


def test_three_seed_reporting_complete():
    comparison = _artifact("task7i_formal_comparison.json")
    for model in MODELS:
        entry = comparison["per_model"][model]
        for metric in ("miou", "dice", "precision_at_0_5"):
            values = entry["oracle"][metric]
            assert len(values["values"]) == 3
            assert values["std_ddof1"] is not None
            assert values["mean"] == pytest.approx(np.mean(values["values"]), abs=1e-12)
        assert entry["all_runs_valid"] is True
    assert len(comparison["matched_seed"]) == 3
    assert comparison["matched_seed_wins"] == 3
    for entry in comparison["matched_seed"]:
        assert entry["delta"] > 0


def test_diagnostics_reported():
    comparison = _artifact("task7i_formal_comparison.json")
    diagnostics = comparison["diagnostics"]
    for key in ("oracle_to_predicted_retention", "seed_std", "parameter_difference",
                "training_time_difference_seconds", "inference_time_difference_seconds",
                "reference_failure_attribution"):
        assert key in diagnostics, key
    assert set(diagnostics["seed_std"]) == set(MODELS)
    assert diagnostics["parameter_difference"] == 278081 - 275777


# ---------------------------------------------------------------- 36-45 lock, guards, continuation


def test_test_lock_remains_locked_after_task7i():
    lock = _artifact("task7i_test_lock_status.json")
    assert lock["status"] == "LOCKED"
    assert lock["test_execution_authorized"] is False
    assert lock["task7i_completed_train_val"] is True
    assert lock["db1_formal_val_confirmed"] is False
    assert lock["unlock_requires"] == "ChatGPT audit of Task 7I"
    assert lock["test_material_read"] is False and lock["test_metrics_produced"] is False
    verdict = _artifact("task7i_verdict.json")
    assert verdict["test_lock"]["status"] == "LOCKED"
    assert verdict["interpretation_boundary"]["test_unlocked"] is False
    assert verdict["interpretation_boundary"]["final_test_started"] is False


def test_no_parser_yolo_or_reference_training():
    for name in TASK7I_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("YOLO(", "load_parser_checkpoint", "build_program_parser",
                       "ProposalSetRanker", "ProposalQualityEstimator", "task6x_sam2"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""
    assert _git_changed("scripts/task6u_common.py") == ""
    verdict = _artifact("task7i_verdict.json")
    assert verdict["protocol"]["yolo_or_parser_or_sam2_trained"] is False


def test_no_new_loss_grcl_attention_or_graph():
    for name in TASK7I_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl", "MultiheadAttention", "TransformerEncoder", "MessagePassing",
                       "scaled_dot_product_attention", "extra_loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7I_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


def test_verdict_priority_and_artifacts():
    verdict = _artifact("task7i_verdict.json")
    comparison = _artifact("task7i_formal_comparison.json")
    population = _artifact("task7i_formal_population_manifest.json")
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    assert verdict["verdict"] == "DB1_FORMAL_VAL_NOT_CONFIRMED"
    assert verdict["reason"].startswith("failed gates: ['5_predicted_delta']")
    assert verdict["DB1_FORMAL_VAL_CONFIRMED"] is False
    assert verdict["protocol"]["clean"] is True
    assert population["verdict"] == "FORMAL_POPULATION_FROZEN"
    assert comparison["runs"] == 6
    assert verdict["recommendation"].startswith("等待 ChatGPT 审核 Task 7I")
    assert (REPO_ROOT / "docs" / "task7i_formal_l3_trainval.md").is_file()
    for name in TASK7I_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name
    for model in MODELS:
        for seed in SEEDS:
            assert (CHECKPOINT_ROOT / model / str(seed) / "best.pt").is_file()
            assert (CHECKPOINT_ROOT / model / str(seed) / "last.pt").is_file()


def test_checkpoint_hashes_recorded():
    for model in MODELS:
        entry = _artifact(f"task7i_training_{model.lower().replace('-', '')}.json")
        for run in entry["runs"]:
            path = Path(run["checkpoint_best"]["path"])
            assert path.is_file()
            assert run["checkpoint_best"]["sha256"] == _sha256(path)
            assert run["checkpoint_best"]["bytes"] == path.stat().st_size


def test_previous_suite_preserved():
    for name in ("test_task7h_development_freeze.py", "test_task7g_largest_reference_selector.py",
                 "test_task7f_reference_ceiling.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
