"""Task 7B tests (section L): 39 checks on the L3 ProgramHead compositional-semantic hardening."""

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
CLI = REPO_ROOT / "predict_buildreasonseg_l3.py"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7b" / "program_parser_l3_hardened_v1.pt"
BASELINE = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
BASELINE_SHA256 = "4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e"
BASE_COMMIT = "6ee3d0d90a56fb11c19112ab51e341b9f36c1392"
TASK7B_SOURCES = ("task7b_build_parser_data.py", "task7b_train_parser.py", "task7b_eval_parser.py",
                  "task7b_scope_audit.py", "task7b_e2e_regression.py", "task7b_report.py")
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")
REQUIRED_ARTIFACTS = ("task7b_compositional_minimal_pairs.json", "task7b_l3_stress_v1.json",
                      "task7b_eval_prompt_manifest.json", "task7b_train_augmentation_spec.json",
                      "task7b_parser_leakage_audit.json", "task7b_training_summary.json",
                      "task7b_full_val.json", "task7b_z_minival240.json",
                      "task7b_z_paired_parser.json", "task7b_task7a_fixed24.json",
                      "task7b_minimal_pairs_result.json", "task7b_stress_result.json",
                      "task7b_scope_safety.json", "task7b_end_to_end_regression.json",
                      "task7b_verdict.json")
COMPACT_PROMPTS = (
    "分割面积最大的建筑物右侧最近的建筑物。",
    "segment the building nearest to the right of the largest building",
    "找出最大建筑左边最近的建筑。",
    "find the closest building to the left of the largest building",
    "找出最大建筑上方最近的建筑。",
    "find the nearest building above the largest building",
    "找出最大建筑下方最近的建筑。",
    "find the closest building below the largest building",
)


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


# ---------------------------------------------------------------- 1-6 freeze, baseline, architecture


def test_task7a_artifacts_unchanged():
    assert _git_changed("evaluation/task7a_") == ""
    assert _git_changed("scripts/task7a_") == ""
    assert _artifact("task7a_verdict.json")["verdict"] == "L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE"


def test_baseline_parser_hash_exact():
    assert BASELINE.is_file()
    assert _sha256(BASELINE) == BASELINE_SHA256
    training = _artifact("task7b_training_summary.json")
    assert training["baseline"]["matches"] is True
    assert training["baseline"]["sha256"] == BASELINE_SHA256
    verdict = _artifact("task7b_verdict.json")
    assert verdict["canonical_gates"]["1_baseline_parser_hash"]["passed"] is True


def test_same_qwen3_vl_2b():
    training = _artifact("task7b_training_summary.json")
    assert training["architecture"]["model_family"] == "Qwen3-VL-2B"
    assert training["architecture"]["total_parameters"] == 2144421888
    assert training["architecture"]["trainable_parameters"] == 17434624
    assert training["architecture"]["trainable_policy"].startswith("Task 6M policy")
    verdict = _artifact("task7b_verdict.json")
    assert verdict["protocol"]["architecture"] == "Qwen3-VL-2B"


def test_text_only_parser():
    training = _artifact("task7b_training_summary.json")
    assert training["architecture"]["text_only"] is True
    assert training["architecture"]["image_input"] is False
    for name in TASK7B_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("read_rgb_tile(", "pixel_values", "images=image"):
            assert marker not in code, f"{name}: {marker}"


def test_exactly_20_classes():
    from buildreasonseg_mvp.program_parser import EXPECTED_PROGRAM_IDS

    assert len(EXPECTED_PROGRAM_IDS) == 20
    training = _artifact("task7b_training_summary.json")
    assert training["architecture"]["classes"] == 20
    assert "unsupported" not in EXPECTED_PROGRAM_IDS


def test_no_keyword_or_regex_remap():
    for name in TASK7B_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("re.search(", "re.match(", "KEYWORD_MAP", "semantic_override",
                       "apply_keyword", "keyword_gate"):
            assert marker not in code, f"{name}: {marker}"
    scope = _artifact("task7b_scope_safety.json")
    assert scope["cli"]["keyword_or_regex_override"] is False
    verdict = _artifact("task7b_verdict.json")
    assert verdict["protocol"]["keyword_or_regex_override"] is False
    assert verdict["safety_gates"]["18_no_keyword_override"]["passed"] is True
    cli = _code_only(CLI)
    assert "keyword" not in cli.lower()


# ---------------------------------------------------------------- 7-13 frozen packs


def test_minimal_pair_pack_frozen_before_training():
    manifest = _artifact("task7b_eval_prompt_manifest.json")
    pack = _artifact("task7b_compositional_minimal_pairs.json")
    spec = _artifact("task7b_train_augmentation_spec.json")
    assert manifest["frozen_before_training"] is True
    assert pack["stage"] == "C-minimal-pairs"
    assert spec["stage"] == "D-training-data"
    code = _code_only(SCRIPTS / "task7b_build_parser_data.py")
    pack_write = code.index("write_json(OUT_MINIMAL")
    train_write = code.index("write_json(OUT_SPEC")
    assert pack_write < train_write, "packs must be frozen before the training data/spec"


def test_stress_pack_frozen_before_training():
    pack = _artifact("task7b_l3_stress_v1.json")
    assert pack["prompt_count"] == 192
    manifest = _artifact("task7b_eval_prompt_manifest.json")
    assert manifest["sources"]["stress_v1"]["sha256"] == _sha256(EVAL / "task7b_l3_stress_v1.json")
    assert manifest["sources"]["minimal_pairs"]["sha256"] == \
        _sha256(EVAL / "task7b_compositional_minimal_pairs.json")


def test_exactly_96_minimal_pair_prompts():
    pack = _artifact("task7b_compositional_minimal_pairs.json")
    assert pack["prompt_count"] == 96
    assert len(pack["rows"]) == 96
    assert pack["by_contrast_group"] == {"M1": 48, "M2": 16, "M3": 16, "M4": 16}
    assert pack["by_language"] == {"zh": 48, "en": 48}
    ids = {row["id"] for row in pack["rows"]}
    assert len(ids) == 96


def test_exactly_192_stress_prompts():
    pack = _artifact("task7b_l3_stress_v1.json")
    assert pack["prompt_count"] == 192 == len(pack["rows"])
    coverage = pack["coverage"]
    assert coverage == {"l3_classes_total": 96, "l2_direction_total": 48, "largest_to_nearest": 16,
                        "smallest_to_nearest": 8, "largest": 8, "smallest": 8,
                        "extreme_classes_total": 8}
    assert sum(coverage.values()) == 192
    assert len({row["id"] for row in pack["rows"]}) == 192


def test_bilingual_l3_stress_coverage():
    pack = _artifact("task7b_l3_stress_v1.json")
    for program in L3_PROGRAMS:
        assert pack["l3_by_language"][f"{program}|zh"] == 12
        assert pack["l3_by_language"][f"{program}|en"] == 12
    assert pack["by_language"] == {"zh": 96, "en": 96}


def test_exact_task7a_fixed24_reused():
    pack = _artifact("task7b_task7a_fixed24.json")
    fixed24 = _artifact("task7a_l3_paraphrase_pack.json")
    assert pack["total"] == 24 == len(fixed24["prompts"])
    assert [row["prompt"] for row in pack["rows"]] == [entry["text"] for entry in fixed24["prompts"]]


def test_all_8_compact_prompts_present():
    fixed24 = _artifact("task7a_l3_paraphrase_pack.json")
    texts = {entry["text"] for entry in fixed24["prompts"]}
    for prompt in COMPACT_PROMPTS:
        assert prompt in texts, prompt
    result = _artifact("task7b_task7a_fixed24.json")
    assert result["compact"]["total"] == 8


# ---------------------------------------------------------------- 14-20 training data and split


def test_original_train_eval_overlap_removed():
    spec = _artifact("task7b_train_augmentation_spec.json")
    leakage = _artifact("task7b_parser_leakage_audit.json")
    assert spec["cleaned_train_rows"] == 0
    assert spec["dropped_train_records"] == 12778
    assert leakage["cleaned_train"]["rows"] == 0
    assert leakage["cleaned_train_overlap"]["exact"] == 0
    assert leakage["cleaned_train_overlap"]["normalized"] == 0


def test_augmentation_eval_exact_overlap_zero():
    leakage = _artifact("task7b_parser_leakage_audit.json")
    assert leakage["augmentation"]["exact_overlap"] == 0
    assert leakage["verdict"] == "LEAKAGE_FREE"
    verdict = _artifact("task7b_verdict.json")
    assert verdict["protocol"]["exact_overlap"] == 0
    assert verdict["canonical_gates"]["3_leakage_zero"]["passed"] is True


def test_augmentation_eval_normalized_overlap_zero():
    leakage = _artifact("task7b_parser_leakage_audit.json")
    assert leakage["augmentation"]["normalized_overlap"] == 0
    verdict = _artifact("task7b_verdict.json")
    assert verdict["protocol"]["normalized_overlap"] == 0
    assert leakage["task7a_fixed24_in_training"] == []


def test_exactly_4800_augmentations():
    spec = _artifact("task7b_train_augmentation_spec.json")
    assert spec["augmentation_total"] == 4800
    assert sum(spec["allocation"].values()) == 4800
    training = _artifact("task7b_training_summary.json")
    assert training["data"]["rows"] == 4800


def test_exact_class_allocation():
    spec = _artifact("task7b_train_augmentation_spec.json")
    allocation = spec["allocation"]
    for program in L3_PROGRAMS:
        assert allocation[program] == 600
    for direction in ("left_of", "right_of", "above", "below"):
        assert allocation[f"largest_to_{direction}"] == 300
        assert allocation[f"smallest_to_{direction}"] == 150
    assert allocation["largest_to_nearest"] == 400
    assert allocation["smallest_to_nearest"] == 200
    training = _artifact("task7b_training_summary.json")
    assert training["data"]["class_counts"]["largest_to_left_of_to_nearest"] == 600
    assert training["data"]["class_counts"]["largest"] == 0


def test_train_internal_holdout_hash_disjoint():
    training = _artifact("task7b_training_summary.json")
    assert training["data"]["normalized_overlap"] == 0
    code = _code_only(SCRIPTS / "task7b_train_parser.py")
    assert 'by_program[row["program"]][normalize_prompt(row["prompt"])].append(row)' in code
    assert "Every row sharing a normalized prompt is assigned to the same side" in \
        (SCRIPTS / "task7b_train_parser.py").read_text(encoding="utf-8")
    assert training["data"]["train_rows"] + training["data"]["holdout_rows"] == 4800


def test_val_test_not_used_for_checkpoint_selection():
    training = _artifact("task7b_training_summary.json")
    assert training["data"]["val_or_test_used_for_selection"] is False
    assert training["protocol"]["selection"] == ["internal-holdout macro F1", "exact accuracy",
                                                 "L3-four-class macro recall"]
    code = _code_only(SCRIPTS / "task7b_train_parser.py")
    assert "evaluate(runtime, holdout_rows)" in code
    assert "z_mini_val_240" not in code
    assert "val.jsonl" not in code


def test_new_checkpoint_sha_recorded():
    training = _artifact("task7b_training_summary.json")
    assert CHECKPOINT.is_file()
    assert training["checkpoint"]["exists"] is True
    assert training["checkpoint"]["sha256"] == _sha256(CHECKPOINT)
    assert len(training["checkpoint"]["sha256"]) == 64


# ---------------------------------------------------------------- 21-28 evaluation reuse and scope


def test_full_val_all_20_classes_evaluated():
    full = _artifact("task7b_full_val.json")
    recalls = full["metrics"]["per_class_recall"]
    assert len(recalls) == 20
    assert all(program in recalls for program in
               ("leftmost", "rightmost", "topmost", "bottommost", "largest", "smallest"))
    assert full["metrics"]["count"] == 18222
    assert full["checkpoint"]["sha256"] == _sha256(CHECKPOINT)


def test_z_minival_exact_reuse():
    mini = _artifact("task7b_z_minival240.json")
    pack = json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]
    assert mini["pack"] == "z_mini_val_240"
    assert mini["metrics"]["count"] == len(pack) == 240
    assert mini["required_correct"] == 240


def test_z_paired_exact_reuse():
    paired = _artifact("task7b_z_paired_parser.json")
    pack = json.loads((PACK_ROOT / "z_paired_val20.json").read_text(encoding="utf-8"))["records"]
    assert paired["pack"] == "z_paired_val20"
    assert paired["metrics"]["count"] == len(pack) == 40
    assert paired["required_correct"] == 40


def test_scope_l2_controls_exit5():
    scope = _artifact("task7b_scope_safety.json")
    controls = {entry["expected_program"]: entry for entry in scope["out_of_scope_controls"]}
    assert "largest_to_right_of" in controls and "largest_to_left_of" in controls
    for program in ("largest_to_right_of", "largest_to_left_of"):
        assert controls[program]["exit_code"] == 5
        assert controls[program]["semantically_correct_before_exit"] is True


def test_scope_nearest_only_controls_exit5():
    scope = _artifact("task7b_scope_safety.json")
    nearest = [entry for entry in scope["out_of_scope_controls"]
               if entry["expected_program"] == "largest_to_nearest"]
    assert len(nearest) == 2
    for entry in nearest:
        assert entry["exit_code"] == 5
        assert entry["semantically_correct_before_exit"] is True
    assert scope["out_of_scope_controls_ok"] is True


def test_no_proposal_call_on_exit5():
    scope = _artifact("task7b_scope_safety.json")
    for entry in scope["out_of_scope_controls"]:
        assert entry["stopped_before"] == ["proposals", "reference", "fields", "sam2", "z_b3"]
        assert entry["only_result_json"] is True


def test_no_reference_sam2_or_zb3_call_on_exit5():
    cli = _code_only(CLI)
    stop_block = cli.split("if program not in SUPPORTED_L3_PROGRAMS:")[1].split("try:")[0]
    for marker in ("L3Pipeline", "predict_target", "resolve_reference", "proposals("):
        assert marker not in stop_block, marker
    scope = _artifact("task7b_scope_safety.json")
    assert all(entry["ground_truth_used"] is False for entry in scope["out_of_scope_controls"])


# ---------------------------------------------------------------- 29-39 frozen downstream and guards


def test_u_c1_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6u_reference_ranker.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6q_reference_resolver.py") == ""
    scope = _artifact("task7b_scope_safety.json")
    assert scope["cli"]["u_c1_resolver_changed"] is False


def test_fields_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6z_field_composition.py") == ""
    scope = _artifact("task7b_scope_safety.json")
    assert scope["cli"]["fields_changed"] is False


def test_z_b3_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6z_l3_decoder.py") == ""
    e2e = _artifact("task7b_end_to_end_regression.json")
    assert e2e["checkpoints"]["downstream_unchanged"] is True
    scope = _artifact("task7b_scope_safety.json")
    assert scope["cli"]["z_b3_changed"] is False


def test_no_yolo_training():
    for name in TASK7B_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("YOLO.train", "resume=True", "yolo.train("):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/task6m1") == ""


def test_no_downstream_training():
    assert _git_changed("buildreasonseg_mvp/task6z_l3_decoder.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    for name in TASK7B_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("L3TargetDecoder(", "RelationMaskDecoder(", "task6z_train"):
            assert marker not in code, f"{name}: {marker}"
    training = _artifact("task7b_training_summary.json")
    assert training["architecture"]["trainable_policy"].startswith("Task 6M policy")


def test_no_grcl():
    for name in TASK7B_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""


def test_no_attention_or_transformer_downstream_change():
    for name in TASK7B_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("MultiheadAttention", "TransformerEncoder", "nn.Transformer", "MessagePassing"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7b_verdict.json")
    assert verdict["interpretation_boundary"]["attention_or_global_competition_added"] is False


def test_no_test_split():
    for name in TASK7B_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"
    for artifact in REQUIRED_ARTIFACTS:
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact
    verdict = _artifact("task7b_verdict.json")
    assert verdict["test_split_used"] is False


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7B_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


def test_end_to_end_regression_tolerance_exact():
    e2e = _artifact("task7b_end_to_end_regression.json")
    assert e2e["tolerance"] == 1.0e-6
    assert e2e["measured"]["parser_correct"] == 240
    assert e2e["deltas"]["strict_all_miou"] <= 1.0e-6
    assert e2e["deltas"]["answered_only_miou"] <= 1.0e-6
    assert e2e["deltas"]["own_cross_margin"] <= 1.0e-6
    assert e2e["measured"]["abstentions"] == e2e["task7a_reference"]["abstentions"]
    assert e2e["measured"]["paired_passed"] == e2e["task7a_reference"]["paired_passed"]
    assert e2e["passed"] is True


def test_cli_default_not_moved_below_gate():
    from buildreasonseg_mvp.task7a_l3_pipeline import (default_l3_parser_checkpoint,
                                                       default_parser_checkpoint)

    verdict = _artifact("task7b_verdict.json")
    scope = _artifact("task7b_scope_safety.json")
    if not verdict["canonical_gates_passed"]:
        assert default_l3_parser_checkpoint() == default_parser_checkpoint()
        assert scope["cli"]["updated_to_task7b"] is False
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    assert verdict["recommendation"].startswith("等待 ChatGPT 根据 Task 7B")


def test_required_artifacts_and_modules_exist():
    assert (REPO_ROOT / "docs" / "task7b_l3_programhead_hardening.md").is_file()
    for name in TASK7B_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name


def test_previous_suite_preserved():
    for name in ("test_task7a_l3_integration.py", "test_task6z_l3_composition.py",
                 "test_task6y_nearest_boundary_field.py", "test_task6x_sam2_refinement.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
