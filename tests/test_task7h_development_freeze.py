"""Task 7H tests (Part I): 40 checks on the development architecture freeze and formal protocol."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
BASE_COMMIT = "22401feedd42e349de425849415fc2fd4caa66d2"
YOLO_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
PARSER_SHA256 = "c150573613c421098f55776a4b2a26e1536b806dd5c210f716715fd2ded58d9a"
Z_B3_SHA256 = "74f308e11e7f7f1098dd0d092ddcf6be1220d0b322b39df39c4ce05c9fc0f0bc"
D_B1_SHA256 = "6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0"
N_B3_SHA256 = "7556e4a4862b75d47e61b5c6391c2a05689d5bde3e74586aed1d94a1c3c7d6ab"
FORMAL_SEEDS = [20261001, 20261002, 20261003]
REJECTED = ("task6p_reference_mask_head", "task6u_proposal_set_ranker_v01",
            "task6w_proposal_quality_estimator_v01", "task6x_sam2_proposal_refinement",
            "task7g_set_context_largest_selector_v1", "task7d_d_b2_learned_global_competition")
TASK7H_SOURCES = ("task7h_audit_frozen_assets.py", "task7h_build_protocol.py", "task7h_report.py")
REQUIRED_ARTIFACTS = ("task7h_evidence_registry.json", "task7h_limitation_registry.json",
                      "task7h_formal_experiment_protocol.json", "task7h_test_lock.json",
                      "task7h_claim_registry.json", "task7h_verdict.json")


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


# ---------------------------------------------------------------- 1-3 Task 7G record


def test_task7g_artifacts_unchanged():
    for prefix in ("evaluation/task7g_", "scripts/task7g_",
                   "buildreasonseg_mvp/task7g_largest_reference_selector.py"):
        assert _git_changed(prefix) == "", prefix
    verdict = _artifact("task7g_verdict.json")
    assert verdict["verdict"] == "LARGEST_SELECTOR_NOT_LEARNABLE"
    assert verdict["decision"]["TASK7G_SELECTOR_ADOPTED"] is False


def test_task7g_selector_not_adopted():
    evidence = _artifact("task7h_evidence_registry.json")
    negative = evidence["rejected_negative_evidence"]["task7g_set_context_largest_selector_v1"]
    assert negative["adopted"] is False
    assert negative["verified_result"]["verdict"] == "LARGEST_SELECTOR_NOT_LEARNABLE"
    assert negative["verified_result"]["internal_gate_passed"] is False
    internal = _artifact("task7g_internal_holdout.json")
    assert internal["gate_passed"] is False
    gate = internal["gate"]
    assert gate["1_mean_gain"]["passed"] is True
    assert gate["2_mean_selected_iou"]["passed"] is False
    assert gate["3_oracle_top1"]["passed"] is True
    assert gate["4_mean_gap"]["passed"] is False
    doc = (REPO_ROOT / "docs" / "task7h_development_architecture_freeze.md").read_text(encoding="utf-8")
    assert "0.6198878" in doc and "0.1790756" in doc


def test_external_task7g_scene_disjoint_result_not_claimed():
    external = _artifact("task7g_external_reference.json")
    assert external["executed"] is False
    assert not (EVAL / "task7g_external_paired.json").exists()
    evidence = _artifact("task7h_evidence_registry.json")
    boundary = evidence["rejected_negative_evidence"][
        "task7g_set_context_largest_selector_v1"]["claim_boundary"]
    assert "never ran" in boundary
    assert "scene-disjoint failure" in boundary
    verdict = _artifact("task7h_verdict.json")
    assert verdict["limitations"]["L-01"] == "unresolved practical bottleneck"


# ---------------------------------------------------------------- 4-9 frozen assets


def test_active_dataset_is_v02():
    evidence = _artifact("task7h_evidence_registry.json")
    data = evidence["data"]
    assert data["name"] == "BuildSpatialReason" and data["version"] == "v0.2"
    assert data["split_view"] == "scene_disjoint_v1"
    assert data["splits"]["train"]["rows"] == 12778
    assert data["splits"]["val"]["rows"] == 9111
    assert evidence["verdict"] == "FROZEN_ASSETS_VERIFIED"


def test_native_vector_v10_recorded():
    evidence = _artifact("task7h_evidence_registry.json")
    native = evidence["data"]["native_vector"]
    assert native["version"] == "v1.0"
    assert native["name"] == "WHU-EA-NativeVector"
    assert evidence["data"]["native_view"]["exists"] is True
    assert evidence["data"]["native_view"]["metadata"]["test"]["read"] is False


def test_yolo_hash_exact():
    from scripts.task6u_common import PROPOSAL_CHECKPOINT

    assert _sha256(PROPOSAL_CHECKPOINT) == YOLO_SHA256
    evidence = _artifact("task7h_evidence_registry.json")
    entry = evidence["checkpoints_and_modules"]["yolo_task6m1"]
    assert entry["sha256"] == YOLO_SHA256 and entry["matches"] is True
    assert entry["config"] == {"imgsz": 640, "conf": 0.05, "max_det": 300, "nms": "default",
                               "tta": False, "tiling": False}


def test_task7c_parser_hash_exact():
    evidence = _artifact("task7h_evidence_registry.json")
    entry = evidence["checkpoints_and_modules"]["parser_task7c"]
    assert entry["expected_sha256"] == PARSER_SHA256
    assert entry["sha256"] == PARSER_SHA256 and entry["matches"] is True
    assert entry["verified_result"]["canonical_full_val_accuracy"] == 1.0
    assert entry["verified_result"]["fixed24"] == "5/24"


def test_z_b3_hash_exact():
    evidence = _artifact("task7h_evidence_registry.json")
    entry = evidence["checkpoints_and_modules"]["z_b3_task6z"]
    assert entry["sha256"] == Z_B3_SHA256 and entry["matches"] is True
    assert entry["role"] == "frozen L3 target-decoder baseline/ablation"


def test_d_b1_hash_exact():
    evidence = _artifact("task7h_evidence_registry.json")
    entry = evidence["checkpoints_and_modules"]["d_b1_task7d"]
    assert entry["sha256"] == D_B1_SHA256 and entry["matches"] is True
    assert entry["evidence_population"].startswith("Task 7E untouched oracle-reference")
    verdict = _artifact("task7h_verdict.json")
    assert verdict["checkpoint_hash_checks"]["required"]["d_b1_task7d"] is True


def test_n_b3_recorded_with_expected_hash():
    evidence = _artifact("task7h_evidence_registry.json")
    entry = evidence["checkpoints_and_modules"]["n_b3_task6o"]
    assert entry["sha256"] == N_B3_SHA256 and entry["matches"] is True
    assert entry["role"].startswith("verified directional-field visual segmentation component")
    assert "N-B2" in entry["claim_boundary"] and "N-B4" in entry["claim_boundary"]


# ---------------------------------------------------------------- 10-18 module roles and exclusions


def test_directional_field_source_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    from buildreasonseg_mvp.geometric_relation_field_v02 import ALPHA, S_AXIS, S_MARGIN, SOFTNESS, TAU
    from buildreasonseg_mvp.task7d_global_competition_decoder import VARIANT_USES_RELATION_FIELDS

    assert (ALPHA, TAU, S_AXIS, S_MARGIN) == (1.2, 0.04, 0.02, 0.02)
    assert SOFTNESS == TAU / 2.0
    assert VARIANT_USES_RELATION_FIELDS["D-B1"] is True
    evidence = _artifact("task7h_evidence_registry.json")
    modules = evidence["checkpoints_and_modules"]["modules"]
    assert modules["direction_field"]["exists"] is True
    assert "v0.2" in modules["direction_field"]["role"]
    assert evidence["roles"]["directional_field"] == "GeometricRelationField v0.2" if "roles" in evidence \
        else True


def test_nearest_field_source_unchanged():
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG

    assert SIGMA_DIAG == 0.05


def test_d_b1_role_exact():
    evidence = _artifact("task7h_evidence_registry.json")
    verdict = _artifact("task7h_verdict.json")
    assert evidence["checkpoints_and_modules"]["d_b1_task7d"]["role"] == \
        "preferred L3 target-decoder architecture candidate"
    assert verdict["roles"]["d_b1"] == "preferred L3 target-decoder architecture candidate"
    assert verdict["conditions"]["4_d_b1_role"]["passed"] is True


def test_z_b3_baseline_role_exact():
    evidence = _artifact("task7h_evidence_registry.json")
    verdict = _artifact("task7h_verdict.json")
    assert evidence["checkpoints_and_modules"]["z_b3_task6z"]["role"] == \
        "frozen L3 target-decoder baseline/ablation"
    assert verdict["roles"]["z_b3"] == "frozen L3 target-decoder baseline/ablation"


def test_task7g_selector_excluded():
    evidence = _artifact("task7h_evidence_registry.json")
    assert "task7g_set_context_largest_selector_v1" in evidence["excluded_from_development_chain"]
    assert "task7g" not in " ".join(evidence["development_chain"]).lower()
    assert "set-context" not in " ".join(evidence["development_chain"]).lower()


def test_proposal_set_ranker_excluded():
    evidence = _artifact("task7h_evidence_registry.json")
    assert "task6u_proposal_set_ranker_v01" in evidence["excluded_from_development_chain"]
    assert "ranker" not in " ".join(evidence["development_chain"]).lower()
    assert evidence["rejected_negative_evidence"]["task6u_proposal_set_ranker_v01"]["adopted"] is False


def test_proposal_quality_estimator_excluded():
    evidence = _artifact("task7h_evidence_registry.json")
    assert "task6w_proposal_quality_estimator_v01" in evidence["excluded_from_development_chain"]
    boundary = evidence["rejected_negative_evidence"][
        "task6w_proposal_quality_estimator_v01"]["claim_boundary"]
    assert boundary == "must not appear in the development chain"
    chain = " ".join(evidence["development_chain"]).lower()
    assert "quality" not in chain


def test_sam2_refinement_excluded():
    evidence = _artifact("task7h_evidence_registry.json")
    assert "task6x_sam2_proposal_refinement" in evidence["excluded_from_development_chain"]
    assert "refinement" not in " ".join(evidence["development_chain"]).lower()
    assert evidence["rejected_negative_evidence"]["task6x_sam2_proposal_refinement"][
        "verified_result"]["verdict"] == "SAM2_REFINEMENT_NOT_HELPFUL"


def test_learned_global_competition_excluded():
    evidence = _artifact("task7h_evidence_registry.json")
    chain = " ".join(evidence["development_chain"]).lower()
    assert "task7d_d_b2_learned_global_competition" in evidence["excluded_from_development_chain"]
    assert "competition" not in chain and "d-b2" not in chain
    assert evidence["rejected_negative_evidence"][
        "task7d_d_b2_learned_global_competition"]["verified_result"]["argmax_in_target"] == 0.0


def test_deterministic_reference_policy_exact():
    evidence = _artifact("task7h_evidence_registry.json")
    chain = evidence["development_chain"]
    assert "deterministic largest reference selector" in chain
    assert any("U-C1" in step for step in chain)
    lock = _artifact("task7h_test_lock.json")
    assert lock["reference_policy"] == "U-C1 deterministic largest"
    doc = (REPO_ROOT / "docs" / "task7h_development_architecture_freeze.md").read_text(encoding="utf-8")
    assert "maximum predicted mask area" in doc
    assert "tie higher confidence" in doc
    assert "lower original index" in doc


# ---------------------------------------------------------------- 19-23 limitations


def test_controlled_language_parser_limitation_recorded():
    limitations = _artifact("task7h_limitation_registry.json")
    entry = limitations["limitations"]["L-04"]
    assert entry["status"] == "controlled-language interface only"
    assert entry["evidence"]["task7c_fixed24"] == "5/24"
    assert entry["evidence"]["task7c_fixed24_compact"] == "0/8"
    assert entry["evidence"]["task7c_stress_accuracy"] == pytest.approx(0.6666666666666666)
    doc = (REPO_ROOT / "docs" / "task7h_development_architecture_freeze.md").read_text(encoding="utf-8")
    assert "canonical-program / controlled-language development interface" in doc


def test_free_form_l3_robustness_not_claimed():
    claims = _artifact("task7h_claim_registry.json")
    assert claims["claims"]["C5"]["status"] == "NOT_SUPPORTED"
    limitations = _artifact("task7h_limitation_registry.json")
    assert "NOT verified" in limitations["limitations"]["L-04"]["claim_boundary"]
    verdict = _artifact("task7h_verdict.json")
    assert verdict["interpretation_boundary"]["unrestricted_natural_language_claimed"] is False
    assert verdict["interpretation_boundary"]["end_to_end_solved_claimed"] is False


def test_nearest_only_limitation_recorded():
    limitations = _artifact("task7h_limitation_registry.json")
    entry = limitations["limitations"]["L-05"]
    assert entry["status"] == "not validated as standalone final capability"
    assert entry["evidence"]["task6y_verdict"] == "NEAREST_FIELD_NO_MEANINGFUL_GAIN"
    assert entry["evidence"]["task6y_b2_paired"] == "6/20"
    doc = (REPO_ROOT / "docs" / "task7h_development_architecture_freeze.md").read_text(encoding="utf-8")
    assert "`experimental/limited`" in doc


def test_unseen_city_generalization_not_claimed():
    limitations = _artifact("task7h_limitation_registry.json")
    entry = limitations["limitations"]["L-06"]
    assert entry["status"] == "not established"
    assert "raster/scene separation" in entry["evidence"]["note"]
    claims = _artifact("task7h_claim_registry.json")
    assert claims["claims"]["C8"]["status"] == "NOT_SUPPORTED"
    verdict = _artifact("task7h_verdict.json")
    assert verdict["interpretation_boundary"]["unseen_city_generalization_claimed"] is False


def test_d_b1_oracle_holdout_evidence_exact():
    evidence = _artifact("task7h_evidence_registry.json")
    verified = evidence["checkpoints_and_modules"]["d_b1_task7d"]["verified_result"]
    assert verified["holdout_oracle_miou"] == 0.38549570532647004
    assert verified["holdout_delta_over_z_b3"] == 0.07138432800871886
    assert verified["holdout_bootstrap_ci"] == [0.058623364793963954, 0.08426264360286147]
    assert verified["holdout_paired"] == "18/20"
    assert verified["holdout_margin"] == 0.3193402994
    assert verified["practical_predicted_reference_strict_miou"] == 0.24540501038500215
    task7e = _artifact("task7e_oracle_holdout.json")
    assert task7e["DB1_HOLDOUT_GENERALIZES"] is True
    assert task7e["results"]["D-B1"]["overall"]["miou"] == verified["holdout_oracle_miou"]


def test_task7f_bottleneck_evidence_exact():
    limitations = _artifact("task7h_limitation_registry.json")
    selection = limitations["limitations"]["L-01"]["evidence"]
    assert selection["task7f_f_r0_strict_miou"] == 0.24540501038500215
    assert selection["task7f_f_r1_oracle_selection_miou"] == 0.36490938928549665
    assert selection["selection_gain"] == 0.1195043789004945
    coverage = limitations["limitations"]["L-02"]["evidence"]
    assert coverage["u_c1_best_eligible_coverage_at_0_50"] == 0.8460388639760837
    assert coverage["coverage_gain"] == 0.054316262681561533
    geometry = limitations["limitations"]["L-03"]["evidence"]
    assert geometry["task7f_geometry_gain_covered"] == 0.004253166957657317
    claims = _artifact("task7h_claim_registry.json")
    assert claims["claims"]["C6"]["status"] == "SUPPORTED_WITH_LIMITATION"
    assert claims["claims"]["C7"]["status"] == "NOT_SUPPORTED"


# ---------------------------------------------------------------- 24-30 formal protocol


def test_formal_l3_train_programs_exact_four():
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    population = protocol["formal_l3_population"]
    assert population["programs"] == ["largest_to_left_of_to_nearest",
                                      "largest_to_right_of_to_nearest",
                                      "largest_to_above_to_nearest",
                                      "largest_to_below_to_nearest"]
    assert population["train"]["expected_total"] == 1344
    assert population["train"]["per_program"] == {"largest_to_left_of_to_nearest": 323,
                                                  "largest_to_right_of_to_nearest": 347,
                                                  "largest_to_above_to_nearest": 338,
                                                  "largest_to_below_to_nearest": 336}
    assert population["val"]["expected_total"] == 936
    evidence = _artifact("task7h_evidence_registry.json")
    assert evidence["data"]["splits"]["train"]["matches_expected"] is True
    assert evidence["data"]["splits"]["val"]["matches_expected"] is True
    assert sum(population["train"]["per_program"].values()) == 1344


def test_formal_seeds_exact():
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    lock = _artifact("task7h_test_lock.json")
    assert protocol["formal_d_b1_training"]["formal_seeds"] == FORMAL_SEEDS
    assert lock["formal_seeds"] == FORMAL_SEEDS
    assert protocol["three_seed_reporting"]["report_each_seed_separately"] is True
    assert protocol["three_seed_reporting"]["best_test_seed_only"] is False
    doc = (REPO_ROOT / "docs" / "task7h_formal_experiment_protocol.md").read_text(encoding="utf-8")
    assert "20261001" in doc and "20261002" in doc and "20261003" in doc


def test_formal_schedule_copied_from_task7d():
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    schedule = protocol["formal_d_b1_training"]["schedule"]
    task7d = _artifact("task7d_training.json")
    assert schedule["lr"] == task7d["training"]["lr"] == 0.0003
    assert schedule["weight_decay"] == task7d["training"]["weight_decay"] == 0.0001
    assert schedule["batch"] == task7d["training"]["batch"] == 8
    assert schedule["max_epochs"] == task7d["training"]["max_epochs"] == 25
    assert schedule["patience"] == task7d["training"]["patience"] == 5
    assert schedule["checkpoint_selection"] == task7d["selection_metric"] == "MiniVal240 mIoU"
    assert schedule["source_artifact"] == "evaluation/task7d_training.json"
    assert protocol["formal_d_b1_training"]["loss"] == "BCE + Dice only"
    assert protocol["formal_d_b1_training"]["new_losses"] == []


def test_formal_metrics_complete():
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    metrics = protocol["required_metrics"]
    assert metrics["mask"] == ["mIoU", "Dice", "Pr@0.5"]
    assert metrics["counterfactual"] == ["pair pass rate", "own IoU", "cross IoU", "own-cross margin"]
    for bucket in ("NO_PROPOSALS", "NO_ELIGIBLE", "NOT_COVERED", "SELECTION_WRONG", "GEOMETRY_POOR",
                   "REFERENCE_OK"):
        assert bucket in metrics["reference"]
    assert metrics["per_relation"] == ["left", "right", "above", "below"]
    assert len(metrics["stratification"]) == 3 and len(metrics["efficiency"]) == 4


def test_practical_and_oracle_results_separated():
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    policy = protocol["reference_policy"]
    assert policy["report_both_separately"] is True
    assert policy["mixing_allowed"] is False
    assert policy["practical_predicted_reference_chain"] == ["U-C1 proposals",
                                                             "deterministic largest selector",
                                                             "D-B1 target mask"]
    assert policy["oracle_reference_diagnostic"] == ["GT reference", "D-B1 target mask"]
    assert policy["task7f_f_r1_oracle_selected_proposal_as_production_result"] is False
    assert protocol["baselines_and_ablations"]["B-L3-0"]["role"] == "baseline/ablation"
    assert protocol["baselines_and_ablations"]["B-L3-1"]["role"] == \
        "main target-decoder architecture candidate"
    assert protocol["baselines_and_ablations"]["new_retrospective_ablation_in_task7h"] is False


def test_future_val_is_model_selection_only():
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    policy = protocol["data_policy"]
    assert "checkpoint selection" in policy["val"]
    assert policy["train"] == "gradient updates only"
    assert "frozen" in policy["test"]
    assert protocol["formal_d_b1_training"]["checkpoint_selection"].startswith("val mIoU only")
    verdict = _artifact("task7h_verdict.json")
    assert verdict["formal_protocol"]["practical_and_oracle_separated"] is True


def test_test_not_authorized():
    lock = _artifact("task7h_test_lock.json")
    assert lock["test_execution_authorized"] is False
    assert lock["checked_before_test_execution"] is True
    verdict = _artifact("task7h_verdict.json")
    assert verdict["interpretation_boundary"]["test_evaluated"] is False
    assert verdict["conditions"]["10_no_test_access"]["passed"] is True
    for name in TASK7H_SOURCES:
        code = _code_only(SCRIPTS / name)
        # the test split may only be *described* (existence/read=False), never read or evaluated
        for marker in ('jsonl_rows(DATA / "test.jsonl")', 'open(DATA / "test.jsonl"',
                       'split="test"', "test_mask", "test_records(", "sha256_file(DATA"):
            assert marker not in code, f"{name}: {marker}"
    audit = _code_only(SCRIPTS / "task7h_audit_frozen_assets.py")
    assert '"read": False, "hashed": False' in audit


def test_test_lock_status_locked():
    lock = _artifact("task7h_test_lock.json")
    assert lock["status"] == "LOCKED"
    assert lock["architecture_head"] == "D-B1"
    assert lock["reference_policy"] == "U-C1 deterministic largest"
    assert lock["parser_role"] == "controlled-language/canonical interface"
    assert lock["unlock_condition"] == "ChatGPT audit after formal train/val completion"
    assert lock["task7h_test_records_read"] is False
    verdict = _artifact("task7h_verdict.json")
    assert verdict["test_lock"]["status"] == "LOCKED"
    assert verdict["conditions"]["8_test_lock"]["passed"] is True


# ---------------------------------------------------------------- 31-37 disclosure and claims


def test_historical_task6m_test_access_disclosed():
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    disclosure = protocol["data_policy"]["test_access_disclosure"]
    assert disclosure["historically_accessed"] is True
    assert "Task 6M" in disclosure["when"]
    assert "J4-v2" in disclosure["when"]
    assert disclosure["post_6m_architecture_selection_used_test_metrics"] is False
    lock = _artifact("task7h_test_lock.json")
    assert "Task 6M" in lock["test_access_disclosure"]
    doc = (REPO_ROOT / "docs" / "task7h_formal_experiment_protocol.md").read_text(encoding="utf-8")
    assert "Task 6M" in doc and "final frozen-architecture test evaluation" in doc


def test_no_untouched_test_claim():
    protocol = _artifact("task7h_formal_experiment_protocol.json")
    disclosure = protocol["data_policy"]["test_access_disclosure"]
    assert "NOT call" in disclosure["consequence"]
    assert "never previously viewed" in disclosure["consequence"]
    verdict = _artifact("task7h_verdict.json")
    assert verdict["interpretation_boundary"]["untouched_test_claimed"] is False
    for name in ("task7h_build_protocol.py", "task7h_report.py"):
        code = _code_only(SCRIPTS / name)
        # the forbidden phrasings may only appear inside an explicit prohibition
        for phrase in ("never previously viewed", "completely untouched"):
            if phrase in code:
                assert "must NOT call" in code or "NOT call" in code
    protocol_doc = (REPO_ROOT / "docs" / "task7h_formal_experiment_protocol.md").read_text(
        encoding="utf-8")
    assert "**not** call the project test split" in protocol_doc


def test_claim_registry_statuses_exact():
    claims = _artifact("task7h_claim_registry.json")
    expected = {"C1": "SUPPORTED", "C2": "SUPPORTED_WITH_LIMITATION", "C3": "SUPPORTED",
                "C4": "NOT_SUPPORTED", "C5": "NOT_SUPPORTED",
                "C6": "SUPPORTED_WITH_LIMITATION", "C7": "NOT_SUPPORTED", "C8": "NOT_SUPPORTED",
                "C9": "NOT_SUPPORTED"}
    assert {key: entry["status"] for key, entry in claims["claims"].items()} == expected
    for entry in claims["claims"].values():
        assert entry["status"] in claims["allowed_statuses"]
    assert claims["claims"]["C1"]["evidence"] == ["Task 6N", "Task 6O"]
    assert claims["claims"]["C3"]["evidence"] == ["Task 7D D-B1",
                                                  "Task 7E untouched oracle-reference holdout"]


def test_no_novelty_or_first_claim():
    claims = _artifact("task7h_claim_registry.json")
    assert claims["novelty_or_first_claim"] is False
    for entry in claims["claims"].values():
        statement = entry["statement"].lower()
        for marker in ("first", "novel", "state-of-the-art", "sota", "unprecedented"):
            assert marker not in statement, entry["statement"]
    verdict = _artifact("task7h_verdict.json")
    assert verdict["interpretation_boundary"]["novelty_claimed"] is False
    doc = (REPO_ROOT / "docs" / "task7h_development_architecture_freeze.md").read_text(encoding="utf-8")
    assert "no novelty or" in doc.lower()


# ---------------------------------------------------------------- 38-40 guards and continuation


def test_no_training_in_task7h():
    verdict = _artifact("task7h_verdict.json")
    assert verdict["interpretation_boundary"]["training_performed"] is False
    assert verdict["conditions"]["9_no_training"]["passed"] is True
    for name in REQUIRED_ARTIFACTS:
        payload = _artifact(name)
        if payload is not None:
            assert payload.get("training_performed", False) is False, name
    for name in TASK7H_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("backward()", "optimizer.step(", "train_variant(", "torch.save(", "AdamW("):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/task7d_global_competition_decoder.py") == ""
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""


def test_no_test_inference():
    verdict = _artifact("task7h_verdict.json")
    assert verdict["interpretation_boundary"]["test_records_read"] is False
    assert verdict["test_split_used"] is False
    evidence = _artifact("task7h_evidence_registry.json")
    assert evidence["data"]["splits"]["test"]["read"] is False
    assert evidence["data"]["splits"]["test"]["hashed"] is False
    assert evidence["test_records_read"] is False
    for name in TASK7H_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("masks.mask(", "load_frozen_sam2_encoder", "FrozenL3Decoder",
                       "LargestSetContextSelector(", "proposals_for_tile"):
            assert marker not in code, f"{name}: {marker}"


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7H_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7h_verdict.json")
    assert verdict["interpretation_boundary"]["downloads_or_installs"] is False
    assert verdict["interpretation_boundary"]["task7i_implementation_details_chosen"] is False


def test_verdict_and_artifacts_complete():
    verdict = _artifact("task7h_verdict.json")
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    assert verdict["verdict"] == "DEVELOPMENT_ARCHITECTURE_FROZEN"
    assert verdict["conditions_passed"] is True
    assert all(entry["passed"] for entry in verdict["conditions"].values())
    assert verdict["architecture_name"] == "BuildReasonSeg-DevFreeze-2026-10"
    assert verdict["recommendation"].startswith("等待 ChatGPT 审核 Task 7H")
    assert verdict["formal_protocol"]["executed_by_task7h"] is False
    assert (REPO_ROOT / "docs" / "task7h_development_architecture_freeze.md").is_file()
    assert (REPO_ROOT / "docs" / "task7h_formal_experiment_protocol.md").is_file()
    for name in TASK7H_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name


def test_previous_suite_preserved():
    for name in ("test_task7g_largest_reference_selector.py", "test_task7f_reference_ceiling.py",
                 "test_task7e_holdout_audit.py", "test_task7d_global_competition.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
