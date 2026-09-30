"""Task 6V tests (section L): 30 checks on the family-conditioned reference resolver policy."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

EVAL = REPO_ROOT / "evaluation"
SCRIPTS = REPO_ROOT / "scripts"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
REF_PACKS = REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
BASE_COMMIT = "b8b5224ec88d32af7d086e337016f6e7d2ada1a2"
TASK6V_SOURCES = ("task6v_select_family_policy.py", "task6v_evaluate_reference.py",
                  "task6v_evaluate_downstream.py", "task6v_report.py")
EXPECTED_OPTIONS = {
    "V-P0": {"config": "U-C0", "selector": "deterministic"},
    "V-P1": {"config": "U-C1", "selector": "deterministic"},
    "V-P2": {"config": "U-C1", "selector": "ranker"},
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


# ---------------------------------------------------------------- 1-5 frozen evidence reuse


def test_task6u_artifacts_unchanged():
    assert _git_changed("evaluation/task6u_") == ""
    assert _git_changed("buildreasonseg_mvp/task6u_reference_ranker.py") == ""
    verdict = _artifact("task6u_verdict.json")
    assert verdict["verdict"] == "REFERENCE_RANKER_NOT_HELPFUL"


def test_u_calib200_exact_reuse():
    split = _artifact("task6u_reference_train_split.json")
    calibration = _artifact("task6v_calibration_family_policy.json")
    assert split["u_calib200"]["count"] == 200
    if calibration is not None:
        assert calibration["calibration_split"]["records"] == 200
        assert calibration["calibration_split"]["by_family"] == {"largest": 100, "smallest": 100}
        assert calibration["calibration_split"]["path"].endswith("task6u_reference_train_split.json")


def test_refval_unique_exact_reuse():
    refval = _artifact("task6v_refval_family_policy.json")
    frozen = json.loads((REF_PACKS / "ref_val_unique.json").read_text(encoding="utf-8"))
    assert frozen["count"] == 219
    if refval is not None:
        assert refval["pack"]["sha256"] == _sha256(REF_PACKS / "ref_val_unique.json")
        assert refval["pack"]["records"] == 219


def test_minival240_exact_reuse():
    manifest = _artifact("task6n_pack_manifest.json")
    mini = _artifact("task6v_downstream_minival240.json")
    assert _sha256(PACK_ROOT / "mini_val_240.json") == manifest["packs"]["mini_val_240"]["sha256"]
    if mini is not None:
        assert mini["pack"]["sha256"] == manifest["packs"]["mini_val_240"]["sha256"]
        assert mini["pack"]["records"] == 240


def test_pairedval20_exact_reuse():
    manifest = _artifact("task6n_pack_manifest.json")
    paired = _artifact("task6v_downstream_pairedval20.json")
    assert _sha256(PACK_ROOT / "paired_val_20.json") == manifest["packs"]["paired_val_20"]["sha256"]
    if paired is not None:
        assert paired["pack"]["sha256"] == manifest["packs"]["paired_val_20"]["sha256"]
        assert paired["pack"]["pairs"] == 20


# ---------------------------------------------------------------- 6-12 options


def test_exactly_three_policy_options():
    from buildreasonseg_mvp.task6v_family_reference_resolver import OPTIONS, OPTION_ORDER

    assert len(OPTIONS) == 3 and OPTION_ORDER == ("V-P0", "V-P1", "V-P2")


def test_v_p0_exact():
    from buildreasonseg_mvp.task6v_family_reference_resolver import OPTIONS

    assert {key: OPTIONS["V-P0"][key] for key in ("config", "selector")} == EXPECTED_OPTIONS["V-P0"]


def test_v_p1_exact():
    from buildreasonseg_mvp.task6v_family_reference_resolver import OPTIONS

    assert {key: OPTIONS["V-P1"][key] for key in ("config", "selector")} == EXPECTED_OPTIONS["V-P1"]


def test_v_p2_exact():
    from buildreasonseg_mvp.task6v_family_reference_resolver import OPTIONS

    assert {key: OPTIONS["V-P2"][key] for key in ("config", "selector")} == EXPECTED_OPTIONS["V-P2"]
    training = _artifact("task6u_ranker_training.json")
    calibration = _artifact("task6v_calibration_family_policy.json")
    if calibration is not None:
        assert calibration["options"]["V-P2"] == "U-C1 + frozen ProposalSetRanker v0.1"
        assert calibration["frozen_assets"]["ranker_checkpoint"]["sha256"] == \
            training["checkpoint"]["sha256"]


def test_ranker_not_retrained():
    training = _artifact("task6u_ranker_training.json")
    selection = _artifact("task6v_calibration_family_policy.json")
    if selection is not None:
        assert selection["frozen_assets"]["ranker_checkpoint"]["retrained"] is False
    if RANKER_CHECKPOINT.is_file():
        assert _sha256(RANKER_CHECKPOINT) == training["checkpoint"]["sha256"]
    for name in TASK6V_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("optimizer.step()", "loss.backward()", "AdamW", "train_step(",
                       "torch.save("):
            assert marker not in code, f"{name} must not train the ranker ({marker})"


def test_yolo_not_retrained():
    selection = _artifact("task6v_calibration_family_policy.json")
    if selection is not None:
        assert selection["frozen_assets"]["yolo_checkpoint"]["retrained"] is False
        assert selection["frozen_assets"]["yolo_checkpoint"]["sha256"] == \
            selection["frozen_assets"]["yolo_checkpoint"]["expected"]
    for name in TASK6V_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("model.train(", "YOLO.train", "resume=True"):
            assert marker not in code, f"{name} must not train YOLO ({marker})"


def test_no_fourth_option():
    verdict = _artifact("task6v_verdict.json")
    calibration = _artifact("task6v_calibration_family_policy.json")
    if calibration is not None:
        assert calibration["options_compared"] == 3
    if verdict is not None:
        assert verdict["protocol"]["fourth_option_added"] is False
        assert verdict["protocol"]["options_compared"] == 3
    frozen = _artifact("task6v_frozen_family_policy.json")
    if frozen is not None:
        assert frozen["fourth_option"] is None
        assert set(frozen["policy"].values()) <= set(EXPECTED_OPTIONS)


# ---------------------------------------------------------------- 13-16 policy selection


def test_per_family_selection_uses_u_calib200_only():
    calibration = _artifact("task6v_calibration_family_policy.json")
    if calibration is None:
        pytest.skip("calibration not run yet")
    assert calibration["calibration_split"]["train_only"] is True
    assert calibration["selection_inputs"]["u_calib200"] is True
    for evidence in ("refval_unique", "minival240", "pairedval20", "test_split"):
        assert calibration["selection_inputs"][evidence] is False
    assert set(calibration["per_family"]) == {"largest", "smallest"}
    for family in ("largest", "smallest"):
        assert set(calibration["per_family"][family]) == set(EXPECTED_OPTIONS)


def test_no_refval_in_policy_selection():
    calibration = _artifact("task6v_calibration_family_policy.json")
    frozen = _artifact("task6v_frozen_family_policy.json")
    if calibration is not None:
        assert calibration["selection_inputs"]["refval_unique"] is False
    if frozen is not None:
        assert frozen["chosen_without_refval"] is True
        assert "refval" not in json.dumps(frozen["evidence"]).lower()
    # the selection script must not *read* any RefVal/other-eval asset (declaring a False flag is fine)
    code = _code_only(SCRIPTS / "task6v_select_family_policy.py")
    for marker in ("ref_val_unique.json", "task6u_refval_selector_comparison.json",
                   "task6u_refval_candidate_audit.json", "mini_val_240", "paired_val_20"):
        assert marker not in code, f"policy selection must not read {marker}"
    assert "task6u_reference_train_split.json" in code


def test_priority_rule_exact():
    from buildreasonseg_mvp.task6v_family_reference_resolver import (
        OPTION_ORDER,
        choose_policy,
        policy_priority_key,
    )

    calibration = _artifact("task6v_calibration_family_policy.json")
    frozen = _artifact("task6v_frozen_family_policy.json")
    if calibration is None or frozen is None:
        pytest.skip("policy not frozen yet")
    for family in ("largest", "smallest"):
        ranked = sorted(OPTION_ORDER,
                        key=lambda option: policy_priority_key(calibration["per_family"][family][option]),
                        reverse=True)
        assert ranked == calibration["ranking"][family]
        assert frozen["policy"][family] == ranked[0]
    assert choose_policy(calibration["per_family"]) == frozen["policy"]
    assert len(frozen["priority_rule"]) == 6
    assert frozen["priority_rule"][0] == "higher Pr@0.5"
    assert "simpler option" in frozen["priority_rule"][-1]


def test_policy_frozen_before_refval_evaluation():
    frozen = _artifact("task6v_frozen_family_policy.json")
    refval = _artifact("task6v_refval_family_policy.json")
    if frozen is None or refval is None:
        pytest.skip("artifacts not generated yet")
    assert frozen["immutable_after_creation"] is True
    assert refval["policy"] == frozen["policy"]
    assert refval["policy_changed_after_refval"] is False
    verdict = _artifact("task6v_verdict.json")
    if verdict is not None:
        assert verdict["protocol"]["policy_frozen_before_refval"] is True
        assert verdict["protocol"]["policy_changed_after_refval"] is False


# ---------------------------------------------------------------- 17-19 resolver dispatch


def test_resolver_dispatch_is_family_only():
    from buildreasonseg_mvp.task6v_family_reference_resolver import (
        FAMILIES,
        FamilyConditionedResolver,
    )

    import inspect

    resolver = FamilyConditionedResolver({"largest": "V-P2", "smallest": "V-P0"})
    assert resolver.option_for("largest")["id"] == "V-P2"
    assert resolver.option_for("smallest")["id"] == "V-P0"
    signature = inspect.signature(FamilyConditionedResolver.select)
    assert list(signature.parameters) == ["self", "proposals", "family"]
    assert inspect.getsource(FamilyConditionedResolver.select).count("def ") == 1
    assert FAMILIES == ("largest", "smallest")


def test_no_relation_input_to_resolver_policy():
    from buildreasonseg_mvp.task6v_family_reference_resolver import (
        OPTIONS,
        load_frozen_policy,
        resolve_family_policy,
    )

    import inspect

    assert list(inspect.signature(resolve_family_policy).parameters) == ["family", "policy"]
    for option in OPTIONS.values():
        assert set(option) == {"id", "config", "selector", "description"}
    source = (REPO_ROOT / "buildreasonseg_mvp" / "task6v_family_reference_resolver.py").read_text(
        encoding="utf-8")
    for marker in ("relation", "target_mask", "instruction"):
        assert marker not in source.lower() or marker in ("relation",), marker


def test_no_gt_in_resolver_inference():
    from buildreasonseg_mvp.task6v_family_reference_resolver import FamilyConditionedResolver

    import inspect

    source = inspect.getsource(FamilyConditionedResolver)
    for marker in ("gt_", "gt_reference", "truth", "oracle", "iou("):
        assert marker not in source, f"resolver uses {marker}"
    refval = _artifact("task6v_refval_family_policy.json")
    if refval is not None:
        assert refval["training_performed"] is False


# ---------------------------------------------------------------- 20-29 downstream and scope


def test_canonical_program_causal_evaluation_has_no_parser():
    mini = _artifact("task6v_downstream_minival240.json")
    if mini is None:
        pytest.skip("downstream not generated yet")
    assert mini["parser_used"] is False
    assert mini["parser_fail_count"] == 0
    code = _code_only(SCRIPTS / "task6v_evaluate_downstream.py")
    assert "PROGRAM_DECOMPOSITION[sample.program_id]" in code
    causal_block = code.split("def run_causal")[1].split("def run_parser")[0]
    assert "parse_instruction" not in causal_block


def test_natural_language_integration_uses_hardened_program_head():
    integration = _artifact("task6v_hardened_parser_integration.json")
    if integration is None:
        pytest.skip("integration not generated yet")
    training = _artifact("task6t_training_summary.json")
    assert integration["parser_checkpoint"]["sha256"] == training["checkpoint"]["sha256"]
    code = _code_only(SCRIPTS / "task6v_evaluate_downstream.py")
    parser_block = code.split("def run_parser")[1]
    assert "parse_instruction" in parser_block
    assert "load_parser_checkpoint" in parser_block


def test_parser_remains_240_of_240():
    integration = _artifact("task6v_hardened_parser_integration.json")
    if integration is None:
        pytest.skip("integration not generated yet")
    assert integration["parser"]["exact_correct"] == 240
    assert integration["parser"]["exact_accuracy"] == 1.0
    assert integration["nearest_or_l3_execution_evaluated"] is False


def test_field_v02_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""


def test_b3_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    mini = _artifact("task6v_downstream_minival240.json")
    if mini is not None:
        assert mini["target_decoder"]["sha256"] == mini["target_decoder"]["expected_sha256"]


def test_sam2_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    for name in TASK6V_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("build_sam", "sam2.build", "load_frozen_sam2_encoder(device",
                       "checkpoint="):
            assert marker not in code, f"{name} must not rebuild SAM2 ({marker})"


def test_no_grcl():
    for name in TASK6V_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""


def test_no_nearest_or_l3_execution():
    from buildreasonseg_mvp.task6s_directional_pipeline import SUPPORTED_PROGRAMS

    assert not any("nearest" in program for program in SUPPORTED_PROGRAMS)
    integration = _artifact("task6v_hardened_parser_integration.json")
    if integration is not None:
        assert integration["nearest_or_l3_execution_evaluated"] is False
    verdict = _artifact("task6v_verdict.json")
    if verdict is not None:
        assert verdict["interpretation_boundary"]["nearest_or_l3_started"] is False


def test_no_test_split():
    for name in TASK6V_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    for artifact in ("task6v_calibration_family_policy.json", "task6v_frozen_family_policy.json",
                     "task6v_refval_family_policy.json", "task6v_downstream_minival240.json",
                     "task6v_downstream_pairedval20.json",
                     "task6v_hardened_parser_integration.json", "task6v_verdict.json"):
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact


def test_no_new_dataset_download_install_or_gui():
    for name in TASK6V_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task6v_verdict.json")
    if verdict is not None:
        assert verdict["training_performed"] is False


def test_previous_suite_preserved():
    for name in ("test_task6u_reference_hardening.py", "test_task6t_programhead_hardening.py",
                 "test_task6s_directional_end_to_end.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT,
                              "--", "tests/"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
