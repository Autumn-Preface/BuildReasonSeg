"""Task 7C tests (section Q): 41 checks on the 20-class rehearsal ProgramHead hardening."""

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
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7c" / "program_parser_l3_rehearsal_v1.pt"
TASK7B_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7b" \
    / "program_parser_l3_hardened_v1.pt"
BASELINE = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
BASELINE_SHA256 = "4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e"
BASE_COMMIT = "b325585c18ddef1ee80cf05d4fc971c5c19b477a"
TASK7C_SOURCES = ("task7c_build_rehearsal.py", "task7c_train_parser.py", "task7c_eval_parser.py",
                  "task7c_scope_audit.py", "task7c_report.py", "task7c_e2e_regression.py")
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")
ALL_CLASSES = ("leftmost", "rightmost", "topmost", "bottommost", "largest", "smallest",
               "largest_to_nearest", "smallest_to_nearest", "largest_to_above", "largest_to_below",
               "largest_to_left_of", "largest_to_right_of", "smallest_to_above", "smallest_to_below",
               "smallest_to_left_of", "smallest_to_right_of", *L3_PROGRAMS)
REQUIRED_ARTIFACTS = ("task7c_reused_eval_manifest.json", "task7c_rehearsal_spec.json",
                      "task7c_training_data_audit.json", "task7c_training_summary.json",
                      "task7c_full_val.json", "task7c_z_minival240.json",
                      "task7c_z_paired_parser.json", "task7c_task7a_fixed24.json",
                      "task7c_minimal_pairs_result.json", "task7c_stress_result.json",
                      "task7c_scope_safety.json", "task7c_end_to_end_regression.json",
                      "task7c_verdict.json")


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


# ---------------------------------------------------------------- 1-7 baseline, init and architecture


def test_task7b_artifacts_unchanged():
    assert _git_changed("evaluation/task7b_") == ""
    assert _git_changed("scripts/task7b_") == ""
    assert _artifact("task7b_verdict.json")["verdict"] == "L3_PARSER_CANONICAL_REGRESSION"


def test_task7b_checkpoint_not_used_as_initialization():
    training = _artifact("task7c_training_summary.json")
    assert training["baseline"]["initialized_from_task7b"] is False
    assert training["baseline"]["task7b_checkpoint_used"] is False
    if TASK7B_CHECKPOINT.is_file():
        assert training["checkpoint"]["sha256"] != _sha256(TASK7B_CHECKPOINT)
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["initialized_from_task7b"] is False
    assert verdict["protocol"]["checkpoint_differs_from_task7b"] is True


def test_baseline_parser_hash_exact():
    assert BASELINE.is_file()
    assert _sha256(BASELINE) == BASELINE_SHA256
    training = _artifact("task7c_training_summary.json")
    assert training["baseline"]["matches"] is True
    assert training["baseline"]["sha256"] == BASELINE_SHA256
    assert _artifact("task7c_verdict.json")["canonical_gates"]["1_baseline_parser_hash"]["passed"] is True


def test_same_qwen3_vl_2b():
    training = _artifact("task7c_training_summary.json")
    assert training["architecture"]["model_family"] == "Qwen3-VL-2B"
    assert training["architecture"]["total_parameters"] == 2144421888
    assert training["architecture"]["trainable_parameters"] == 17434624


def test_text_only():
    training = _artifact("task7c_training_summary.json")
    assert training["architecture"]["text_only"] is True
    assert training["architecture"]["image_input"] is False
    for name in TASK7C_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("pixel_values", "images=image"):
            assert marker not in code, f"{name}: {marker}"


def test_20_classes_exact():
    from buildreasonseg_mvp.program_parser import EXPECTED_PROGRAM_IDS

    assert len(tuple(EXPECTED_PROGRAM_IDS)) == 20
    training = _artifact("task7c_training_summary.json")
    assert training["architecture"]["classes"] == 20
    assert "unsupported" not in EXPECTED_PROGRAM_IDS


def test_no_original_v02_template_row_used():
    training = _artifact("task7c_training_summary.json")
    assert training["data"]["original_v02_train_text_used"] is False
    audit = _artifact("task7c_training_data_audit.json")
    assert audit["original_v02_train_text_used_for_optimization"] is False
    spec = _artifact("task7c_rehearsal_spec.json")
    assert spec["original_v02_train_text_used_for_optimization"] is False
    source = (SCRIPTS / "task7c_build_rehearsal.py").read_text(encoding="utf-8")
    # the candidate generators are template-based and never read an instruction string
    generators = source.split("def zh_candidates")[1].split("def evaluation_prompts")[0]
    assert "instruction_" not in generators
    # the only train-file read is the class-id schema assertion
    assert 'train_classes.add(str(json.loads(line)["query_type"]))' in source
    assert source.count("train.jsonl") == 1
    rows = (REPO_ROOT / "artifacts" / "task7c" / "parser_rehearsal"
            / "parser_rehearsal_rows.jsonl")
    if rows.is_file():
        train_texts = set()
        with (REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2" / "train.jsonl").open(
                encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for key in ("instruction_en", "instruction_zh"):
                    if record.get(key):
                        train_texts.add(str(record[key]))
        rehearsal = {json.loads(line)["prompt"] for line in rows.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
        assert not (rehearsal & train_texts), "no original v0.2 train instruction string may be used"


# ---------------------------------------------------------------- 8-16 rehearsal data


def test_exactly_8000_rehearsal_prompts():
    spec = _artifact("task7c_rehearsal_spec.json")
    audit = _artifact("task7c_training_data_audit.json")
    assert spec["total_rows"] == 8000
    assert audit["rehearsal"]["rows"] == 8000
    assert sum(spec["allocation"].values()) == 8000
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["rehearsal_rows"] == 8000


def test_all_20_classes_represented():
    spec = _artifact("task7c_rehearsal_spec.json")
    assert set(spec["by_class"]) == set(ALL_CLASSES)
    assert all(spec["by_class"][program] > 0 for program in ALL_CLASSES)
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["all_classes_represented"] is True


def test_l3_800_per_class():
    spec = _artifact("task7c_rehearsal_spec.json")
    for program in L3_PROGRAMS:
        assert spec["allocation"][program] == 800
        assert spec["by_class"][program] == 800


def test_non_l3_300_per_class():
    spec = _artifact("task7c_rehearsal_spec.json")
    for program in ALL_CLASSES:
        if program in L3_PROGRAMS:
            continue
        assert spec["allocation"][program] == 300
        assert spec["by_class"][program] == 300


def test_exact_chinese_english_balance():
    spec = _artifact("task7c_rehearsal_spec.json")
    for program in ALL_CLASSES:
        expected_zh = spec["allocation"][program] // 2
        expected_en = spec["allocation"][program] - expected_zh
        assert spec["by_class_language"][f"{program}|zh"] == expected_zh, program
        assert spec["by_class_language"][f"{program}|en"] == expected_en, program
    audit = _artifact("task7c_training_data_audit.json")
    assert audit["rehearsal"]["per_class_counts_exact"] is True


def test_no_normalized_training_duplicate():
    audit = _artifact("task7c_training_data_audit.json")
    assert audit["rehearsal"]["normalized_duplicates"] == 0
    assert audit["rehearsal"]["unique_normalized"] == 8000
    verdict = _artifact("task7c_verdict.json")
    assert verdict["rehearsal"]["normalized_duplicates"] == 0


def test_exact_reused_eval_hashes():
    reused = _artifact("task7c_reused_eval_manifest.json")
    assert reused["regenerated"] is False
    fixed24 = EVAL / "task7a_l3_paraphrase_pack.json"
    minimal = EVAL / "task7b_compositional_minimal_pairs.json"
    stress = EVAL / "task7b_l3_stress_v1.json"
    assert reused["files"]["task7a_fixed24"]["sha256"] == _sha256(fixed24)
    assert reused["files"]["task7b_minimal96"]["sha256"] == _sha256(minimal)
    assert reused["files"]["task7b_stress192"]["sha256"] == _sha256(stress)
    assert reused["files"]["z_minival240"]["sha256"] == _sha256(PACK_ROOT / "z_mini_val_240.json")
    for name in ("v02_val", "task7a_fixed24", "task7b_minimal96", "task7b_stress192",
                 "z_minival240", "z_paired_val20", "task7b_scope_controls"):
        assert reused["sources"][name]["prompts"] > 0, name


def test_train_eval_exact_overlap_zero():
    audit = _artifact("task7c_training_data_audit.json")
    assert audit["rehearsal"]["exact_overlap"] == 0
    assert audit["verdict"] == "REHEARSAL_CLEAN"
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["exact_overlap"] == 0
    assert verdict["canonical_gates"]["7_exact_overlap_zero"]["passed"] is True


def test_train_eval_normalized_overlap_zero():
    audit = _artifact("task7c_training_data_audit.json")
    assert audit["rehearsal"]["normalized_overlap"] == 0
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["normalized_overlap"] == 0
    assert verdict["canonical_gates"]["8_normalized_overlap_zero"]["passed"] is True


def test_internal_holdout_all_20_classes():
    training = _artifact("task7c_training_summary.json")
    support = training["data"]["holdout_support"]
    assert set(support) == set(ALL_CLASSES)
    assert all(value > 0 for value in support.values())
    assert training["data"]["classes_without_holdout_support"] == []
    for program in L3_PROGRAMS:
        assert support[program] == 80, program
    for program in ALL_CLASSES:
        if program not in L3_PROGRAMS:
            assert support[program] == 30, program


def test_train_holdout_normalized_disjoint():
    training = _artifact("task7c_training_summary.json")
    assert training["data"]["normalized_overlap"] == 0
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["holdout_normalized_overlap"] == 0
    assert verdict["canonical_gates"]["9_holdout_all_classes"]["passed"] is True


# ---------------------------------------------------------------- 17-23 optimization protocol


def test_lora_lr_2e_5():
    training = _artifact("task7c_training_summary.json")
    assert training["protocol"]["lora_lr"] == 2.0e-5
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["lora_lr"] == 2.0e-5


def test_program_head_lr_1e_4():
    training = _artifact("task7c_training_summary.json")
    assert training["protocol"]["program_head_lr"] == 1.0e-4
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["program_head_lr"] == 1.0e-4
    code = _code_only(SCRIPTS / "task7c_train_parser.py")
    assert "trainable_parameter_groups(PROTOCOL[\"lora_lr\"], PROTOCOL[\"program_head_lr\"]" in code


def test_no_base_backbone_unfreeze():
    training = _artifact("task7c_training_summary.json")
    assert training["architecture"]["backbone_unfrozen"] is False
    assert training["architecture"]["trainable_parameters"] == 17434624
    for name in TASK7C_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("requires_grad_(True)", "unfreeze", "enable_input_require_grads"):
            assert marker not in code, f"{name}: {marker}"


def test_no_hyperparameter_sweep():
    training = _artifact("task7c_training_summary.json")
    assert training["hyperparameter_sweep"] is False
    assert training["protocol"]["sweep"] is False
    assert training["protocol"]["scheduler"] == "none"
    code = _code_only(SCRIPTS / "task7c_train_parser.py")
    assert "CONFIGS" not in code and "for config in" not in code


def test_external_eval_not_used_for_selection():
    training = _artifact("task7c_training_summary.json")
    assert training["data"]["external_eval_used_for_selection"] is False
    assert training["protocol"]["selection"][0] == "min(macro_f1_20, l3_macro_recall)"
    code = _code_only(SCRIPTS / "task7c_train_parser.py")
    assert "evaluate(runtime, holdout_rows)" in code
    for marker in ("z_mini_val_240", "task7a_l3_paraphrase_pack", "task7b_l3_stress_v1",
                   "val.jsonl"):
        assert marker not in code, marker


def test_full_val_all_20_classes_evaluated():
    full = _artifact("task7c_full_val.json")
    recalls = full["metrics"]["per_class_recall"]
    assert len(recalls) == 20
    assert all(entry is not None for entry in recalls.values())
    assert full["metrics"]["count"] == 18222
    assert full["checkpoint"]["sha256"] == _sha256(CHECKPOINT)


def test_z_minival_exact_reuse():
    mini = _artifact("task7c_z_minival240.json")
    pack = json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]
    assert mini["pack"] == "z_mini_val_240"
    assert mini["metrics"]["count"] == len(pack) == 240
    assert mini["required_correct"] == 240


def test_z_paired_exact_reuse():
    paired = _artifact("task7c_z_paired_parser.json")
    pack = json.loads((PACK_ROOT / "z_paired_val20.json").read_text(encoding="utf-8"))["records"]
    assert paired["pack"] == "z_paired_val20"
    assert paired["metrics"]["count"] == len(pack) == 40
    assert paired["required_correct"] == 40


def test_fixed24_exact_reuse():
    result = _artifact("task7c_task7a_fixed24.json")
    fixed24 = _artifact("task7a_l3_paraphrase_pack.json")
    assert result["total"] == 24 == len(fixed24["prompts"])
    assert [row["prompt"] for row in result["rows"]] == [entry["text"] for entry in fixed24["prompts"]]
    assert result["compact"]["total"] == 8


def test_minimal96_exact_reuse():
    result = _artifact("task7c_minimal_pairs_result.json")
    pack = _artifact("task7b_compositional_minimal_pairs.json")
    assert result["total"] == 96 == len(pack["rows"])
    assert [row["prompt"] for row in result["rows"]] == [row["prompt"] for row in pack["rows"]]


def test_stress192_exact_reuse():
    result = _artifact("task7c_stress_result.json")
    pack = _artifact("task7b_l3_stress_v1.json")
    assert result["metrics"]["count"] == 192 == len(pack["rows"])
    assert result["l3_classes"] == list(L3_PROGRAMS)


def test_scope_controls_exact_reuse():
    scope = _artifact("task7c_scope_safety.json")
    task7b_scope = _artifact("task7b_scope_safety.json")
    assert scope["controls_source"]["reused"] is True
    assert scope["controls_source"]["sha256"] == _sha256(EVAL / "task7b_scope_safety.json")
    assert [entry["prompt"] for entry in scope["out_of_scope_controls"]] == \
        [entry["prompt"] for entry in task7b_scope["out_of_scope_controls"]]
    assert scope["passed"] is True


# ---------------------------------------------------------------- 24-36 frozen modules and guards


def test_no_keyword_or_regex_correction():
    for name in TASK7C_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("re.search(", "re.match(", "KEYWORD_MAP", "semantic_override",
                       "apply_keyword", "keyword_gate"):
            assert marker not in code, f"{name}: {marker}"
    scope = _artifact("task7c_scope_safety.json")
    assert scope["cli"]["keyword_or_regex_override"] is False
    verdict = _artifact("task7c_verdict.json")
    assert verdict["protocol"]["keyword_or_regex_override"] is False


def test_u_c1_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6q_reference_resolver.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6u_reference_ranker.py") == ""
    scope = _artifact("task7c_scope_safety.json")
    assert scope["cli"]["u_c1_resolver_changed"] is False


def test_field_modules_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/nearest_boundary_field.py") == ""
    assert _git_changed("buildreasonseg_mvp/task6z_field_composition.py") == ""
    scope = _artifact("task7c_scope_safety.json")
    assert scope["cli"]["fields_changed"] is False


def test_z_b3_unchanged():
    assert _git_changed("buildreasonseg_mvp/task6z_l3_decoder.py") == ""
    e2e = _artifact("task7c_end_to_end_regression.json")
    assert e2e["checkpoints"]["downstream_unchanged"] is True
    scope = _artifact("task7c_scope_safety.json")
    assert scope["cli"]["z_b3_changed"] is False


def test_no_yolo_or_downstream_training():
    for name in TASK7C_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("YOLO.train", "yolo.train(", "L3TargetDecoder(", "RelationMaskDecoder("):
            assert marker not in code, f"{name}: {marker}"
    training = _artifact("task7c_training_summary.json")
    assert training["architecture"]["trainable_policy"].startswith("Task 6M/6T policy")


def test_no_grcl():
    for name in TASK7C_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name}: {marker}"
    assert _git_changed("buildreasonseg_mvp/grcl_directional.py") == ""


def test_no_attention_gnn_transformer_downstream_change():
    for name in TASK7C_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("MultiheadAttention", "TransformerEncoder", "nn.Transformer", "MessagePassing"):
            assert marker not in code, f"{name}: {marker}"
    verdict = _artifact("task7c_verdict.json")
    assert verdict["interpretation_boundary"]["attention_or_global_competition_added"] is False


def test_no_test_split():
    for name in TASK7C_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name}: {marker}"
    for artifact in REQUIRED_ARTIFACTS:
        payload = _artifact(artifact)
        if payload is not None:
            assert payload.get("test_split_used", False) is False, artifact


def test_no_new_dataset_download_install_or_gui():
    for name in TASK7C_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "load_dataset", "tkinter", "gradio", "streamlit", "flask"):
            assert marker not in code, f"{name}: {marker}"


# ---------------------------------------------------------------- 37-41 verdict, E2E, suite


def test_end_to_end_tolerance_exact():
    e2e = _artifact("task7c_end_to_end_regression.json")
    assert e2e["tolerance"] == 1.0e-6
    assert e2e["measured"]["parser_correct"] == 240
    for key in ("strict_all_miou", "answered_only_miou", "own_cross_margin"):
        assert e2e["deltas"][key] <= 1.0e-6, key
    assert e2e["measured"]["abstentions"] == e2e["task7a_reference"]["abstentions"]
    assert e2e["measured"]["paired_passed"] == e2e["task7a_reference"]["paired_passed"]
    assert e2e["passed"] is True


def test_cli_default_follows_canonical_gates():
    from buildreasonseg_mvp.task7a_l3_pipeline import (default_l3_parser_checkpoint,
                                                       default_parser_checkpoint)

    verdict = _artifact("task7c_verdict.json")
    scope = _artifact("task7c_scope_safety.json")
    if verdict["canonical_gates_passed"]:
        assert default_l3_parser_checkpoint() == CHECKPOINT
        assert scope["cli"]["canonical_gates_passed"] is True
    else:
        assert default_l3_parser_checkpoint() == default_parser_checkpoint()
    assert verdict["verdict"] in verdict["allowed_verdicts"]
    assert verdict["recommendation"].startswith("等待 ChatGPT 根据 Task 7C")


def test_required_artifacts_and_modules_exist():
    assert (REPO_ROOT / "docs" / "task7c_20class_rehearsal_l3_hardening.md").is_file()
    for name in TASK7C_SOURCES:
        assert (SCRIPTS / name).is_file(), name
    for name in REQUIRED_ARTIFACTS:
        assert (EVAL / name).is_file(), name
    assert CHECKPOINT.is_file()
    assert _sha256(CHECKPOINT) == _artifact("task7c_training_summary.json")["checkpoint"]["sha256"]


def test_verdict_gate_arithmetic():
    verdict = _artifact("task7c_verdict.json")
    canonical = all(entry["passed"] for entry in verdict["canonical_gates"].values())
    compositional = all(entry["passed"] for entry in verdict["compositional_gates"].values())
    safety = all(entry["passed"] for entry in verdict["safety_gates"].values())
    assert verdict["canonical_gates_passed"] == canonical
    assert verdict["compositional_gates_passed"] == compositional
    if not canonical:
        assert verdict["verdict"] == "L3_20CLASS_CANONICAL_REGRESSION"
    elif not compositional:
        assert verdict["verdict"] == "L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL"
    elif not safety:
        assert verdict["verdict"] in ("END_TO_END_REGRESSION", "INVALID_EXPERIMENT")
    else:
        assert verdict["verdict"] == "L3_20CLASS_REHEARSAL_PASS"


def test_previous_suite_preserved():
    for name in ("test_task7b_parser_hardening.py", "test_task7a_l3_integration.py",
                 "test_task6z_l3_composition.py", "test_task6y_nearest_boundary_field.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT, "--",
                              "tests/"], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
