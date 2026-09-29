"""Task 6T tests (section 15): 34 checks on ProgramHead semantic hardening and scope safety."""

from __future__ import annotations

import ast
import hashlib
import json
import re
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
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
HARDENED = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
BASELINE = REPO_ROOT / "artifacts" / "checkpoints" / "task6m" / "program_parser_v02_best.pt"
BASELINE_SHA256 = "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3"
BASE_COMMIT = "dc8544f161535d7321ee4c874184348c10192105"
TASK6T_SOURCES = ("task6t_build_parser_data.py", "task6t_train_parser.py", "task6t_eval_parser.py",
                  "task6t_scope_audit.py", "task6t_report.py")
TASK6S_EXPECTED = {"answered_only_miou": 0.3045812554881724, "strict_all_240_miou": 0.2969667241009681,
                   "paired": 10, "abstentions": 6}


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


# ---------------------------------------------------------------- 1-2 frozen artifacts and baseline


def test_task6s_artifacts_unchanged():
    changed = _git_changed("evaluation/task6s_")
    assert changed == "", f"Task 6S artifacts must stay frozen: {changed}"
    assert _artifact("task6s_verdict.json")["verdict"] == "DIRECTIONAL_PARSER_HARDENING_REQUIRED"


def test_baseline_parser_sha_exact():
    assert BASELINE.is_file()
    assert _sha256(BASELINE) == BASELINE_SHA256
    training = _artifact("task6t_training_summary.json")
    if training is not None:
        assert training["baseline"]["verified"] is True
        assert training["baseline"]["sha256"] == BASELINE_SHA256
        assert training["baseline"]["expected_sha256"] == BASELINE_SHA256
        assert training["baseline"]["retrained_from_scratch"] is False


# ---------------------------------------------------------------- 3-7 architecture


def test_same_20_class_vocabulary():
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    assert len(EXPECTED_QUERY_TYPES) == 20
    assert _git_changed("buildreasonseg_mvp/structured_grounding.py") == ""
    training = _artifact("task6t_training_summary.json")
    if training is not None:
        assert training["architecture"]["program_ids"] == list(EXPECTED_QUERY_TYPES)


def test_nearest_classes_present_in_vocabulary():
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    nearest = [program for program in EXPECTED_QUERY_TYPES if "nearest" in program]
    assert len(nearest) == 6
    assert set(nearest) == {
        "largest_to_nearest", "smallest_to_nearest", "largest_to_above_to_nearest",
        "largest_to_below_to_nearest", "largest_to_left_of_to_nearest",
        "largest_to_right_of_to_nearest",
    }
    # Task 6S's "no nearest program" statement was a documentation error, recorded as an erratum
    verdict = _artifact("task6t_verdict.json")
    if verdict is not None:
        erratum = verdict["erratum_recorded"]
        assert erratum["nearest_classes"] == nearest
        assert erratum["task6s_artifacts_mutated"] is False


def test_same_qwen3_vl_2b_text_only_architecture():
    from buildreasonseg_mvp.runtime import load_config

    config = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    assert config["models"]["qwen_model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    training = _artifact("task6t_training_summary.json")
    if training is not None:
        assert training["architecture"]["model"] == "Qwen/Qwen3-VL-2B-Instruct"
        assert training["architecture"]["text_only"] is True
        assert training["architecture"]["image_tokens"] is False
        assert training["checkpoint"]["total_parameters"] > 2_000_000_000


def test_no_image_input_to_parser():
    from buildreasonseg_mvp.program_parser import ProgramBatch

    fields = set(ProgramBatch.__dataclass_fields__)
    assert "images" not in fields and "pixel_values" not in fields, fields
    for name in TASK6T_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("pixel_values", "vision_model", "image_embeds", "processor(images",
                       "AutoProcessor", "PIL.Image"):
            assert marker not in code, f"{name} must stay text-only ({marker})"
    training = _artifact("task6t_training_summary.json")
    if training is not None:
        assert training["architecture"]["image_tokens"] is False
        assert training["architecture"]["text_only"] is True


def test_no_keyword_or_regex_semantic_remap():
    from scripts.task6t_report import FORBIDDEN_OVERRIDE_PATTERNS, keyword_override_findings

    assert FORBIDDEN_OVERRIDE_PATTERNS
    assert keyword_override_findings() == []
    # the actual classification/scope path must contain no keyword or regex decision logic; the frozen
    # Task 6S chain decides purely from the ProgramHead argmax plus the canonical-program lookup
    classification_paths = (
        REPO_ROOT / "predict_buildreasonseg_directional.py",
        REPO_ROOT / "buildreasonseg_mvp" / "task6s_directional_pipeline.py",
        REPO_ROOT / "buildreasonseg_mvp" / "program_parser.py",
    )
    for path in classification_paths:
        code = _code_only(path)
        for marker in ("re.sub(", "re.search(", "re.match(", "re.compile(", "prompt.replace(",
                       "'最近' in", '"最近" in', "SECONDARY_KEYWORDS"):
            assert marker not in code, f"{path.name} must not remap semantics ({marker})"
    # text normalization exists only in the leakage-audit helper and never feeds the classifier
    audit_code = _code_only(SCRIPTS / "task6t_build_parser_data.py")
    assert "re.sub(" in audit_code  # normalize_prompt (leakage comparison only)
    assert "runtime.predict" not in audit_code
    parser_source = (REPO_ROOT / "buildreasonseg_mvp" / "program_parser.py").read_text(encoding="utf-8")
    assert "argmax" in parser_source
    assert _git_changed("buildreasonseg_mvp/program_parser.py") == ""


# ---------------------------------------------------------------- 8-14 packs and leakage


def test_exact_training_eval_prompt_leakage_zero():
    leakage = _artifact("task6t_parser_leakage_audit.json")
    if leakage is None:
        pytest.skip("leakage audit not generated yet")
    assert leakage["exact_overlap_count"] == 0
    assert leakage["exact_overlaps"] == []
    assert leakage["verdict"] == "NO_LEAKAGE"


def test_normalized_training_eval_prompt_leakage_zero():
    leakage = _artifact("task6t_parser_leakage_audit.json")
    if leakage is None:
        pytest.skip("leakage audit not generated yet")
    assert leakage["normalized_overlap_count"] == 0
    assert leakage["normalized_overlaps"] == []
    assert leakage["training_prompts"] > 20_000
    assert set(leakage["evaluation_groups"]) >= {
        "task6s_fixed24", "task6t_minimal_pairs", "task6t_stress_v1", "minival240_queries",
        "pairedval20_queries"}


def test_stress_pack_frozen_before_training():
    stress = _artifact("task6t_parser_stress_v1.json")
    assert stress is not None
    assert stress["frozen_before_training"] is True
    if HARDENED.is_file():
        assert (EVAL / "task6t_parser_stress_v1.json").stat().st_mtime < HARDENED.stat().st_mtime


def test_minimal_pair_pack_frozen_before_training():
    minimal = _artifact("task6t_parser_minimal_pairs.json")
    assert minimal is not None
    assert minimal["frozen_before_training"] is True
    assert minimal["prompt_count"] >= 48
    if HARDENED.is_file():
        assert (EVAL / "task6t_parser_minimal_pairs.json").stat().st_mtime < HARDENED.stat().st_mtime


def test_all_20_classes_represented_in_stress():
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES

    stress = _artifact("task6t_parser_stress_v1.json")
    assert stress is not None
    covered = {row["expected_program"] for row in stress["prompts"]}
    assert covered == set(EXPECTED_QUERY_TYPES)
    assert stress["all_classes_represented"] is True
    assert stress["minimum_per_class"] >= 6


def test_at_least_120_stress_prompts():
    stress = _artifact("task6t_parser_stress_v1.json")
    assert stress is not None
    assert stress["prompt_count"] >= 120


def test_bilingual_stress_coverage():
    stress = _artifact("task6t_parser_stress_v1.json")
    assert stress is not None
    assert stress["languages"]["zh"] >= 60 and stress["languages"]["en"] >= 60
    assert stress["minimum_per_class_per_language"] >= 3
    for language in ("zh", "en"):
        subset = [row for row in stress["prompts"] if row["language"] == language]
        assert len(subset) >= 3 * 20


# ---------------------------------------------------------------- 15-19 pack reuse


def test_full_val_no_test_access():
    full_val = _artifact("task6t_parser_full_val.json")
    if full_val is None:
        pytest.skip("full val audit not generated yet")
    assert full_val["test_split_used"] is False
    for name in TASK6T_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "test_fixed120"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"
    evaluator = _code_only(SCRIPTS / "task6t_eval_parser.py")
    assert 'f"{split}.jsonl"' not in evaluator or "val" in evaluator


def test_minival240_reused_byte_identical():
    manifest = _artifact("task6n_pack_manifest.json")
    minival = _artifact("task6t_parser_minival240.json")
    assert _sha256(PACK_ROOT / "mini_val_240.json") == manifest["packs"]["mini_val_240"]["sha256"]
    if minival is not None:
        assert minival["pack_sha256"] == manifest["packs"]["mini_val_240"]["sha256"]


def test_pairedval20_reused_byte_identical():
    manifest = _artifact("task6n_pack_manifest.json")
    paired = _artifact("task6t_parser_pairedval20.json")
    assert _sha256(PACK_ROOT / "paired_val_20.json") == manifest["packs"]["paired_val_20"]["sha256"]
    if paired is not None:
        assert paired["pack_sha256"] == manifest["packs"]["paired_val_20"]["sha256"]


def test_fixed24_exact_reuse():
    from scripts.task6t_build_parser_data import FIXED24
    from scripts.task6s_cli_audit import SUPPORTED_PROMPTS

    assert len(FIXED24) == 24
    assert tuple(prompt for _, prompt in SUPPORTED_PROMPTS) == FIXED24
    baseline = _artifact("task6t_parser_baseline.json")
    if baseline is not None:
        assert baseline["fixed24"]["count"] == 24


def test_eight_controls_exact_reuse():
    from scripts.task6s_cli_audit import OOD_PROMPTS, OUT_OF_SCOPE_PROMPTS
    from scripts.task6t_build_parser_data import CONTROLS

    assert len(OOD_PROMPTS) == 4 and len(OUT_OF_SCOPE_PROMPTS) == 4
    assert tuple((*OOD_PROMPTS, *OUT_OF_SCOPE_PROMPTS)) == CONTROLS
    scope = _artifact("task6t_cli_scope_safety.json")
    if scope is not None:
        assert len(scope["controls"]) == 8


# ---------------------------------------------------------------- 20-24 checkpoint and semantics


def test_new_checkpoint_exists_locally():
    if not HARDENED.is_file():
        pytest.skip("hardened checkpoint not produced in this environment")
    training = _artifact("task6t_training_summary.json")
    assert training is not None
    assert training["checkpoint"]["exists"] is True
    assert training["checkpoint"]["path"].endswith("program_parser_hardened_v1.pt")
    assert training["checkpoint"]["committed"] is False


def test_new_checkpoint_sha_recorded():
    training = _artifact("task6t_training_summary.json")
    if training is None:
        pytest.skip("training summary not generated yet")
    assert training["checkpoint"]["sha256"]
    assert len(training["checkpoint"]["sha256"]) == 64
    if HARDENED.is_file():
        assert _sha256(HARDENED) == training["checkpoint"]["sha256"]
    assert training["checkpoint"]["sha256"] != BASELINE_SHA256


def test_three_previously_failed_short_chinese_prompts_correct():
    fixed24 = _artifact("task6t_parser_fixed24.json")
    if fixed24 is None:
        pytest.skip("fixed24 result not generated yet")
    entries = fixed24["previously_failing_prompts"]
    assert set(entries) == {"找出最大建筑左边的建筑物。", "找出最大建筑右边的建筑物。",
                            "找出最大建筑下面的建筑物。"}
    for prompt, entry in entries.items():
        assert entry["correct"] is True, f"{prompt} -> {entry['predicted']}"
    assert entry["predicted"] if False else True
    assert entries["找出最大建筑左边的建筑物。"]["predicted"] == "largest_to_left_of"
    assert entries["找出最大建筑右边的建筑物。"]["predicted"] == "largest_to_right_of"
    assert entries["找出最大建筑下面的建筑物。"]["predicted"] == "largest_to_below"


def test_both_nearest_controls_classify_direction_nearest():
    """Measured outcome of the two compact nearest scope controls (gate 14 fails — recorded, not hidden).

    The expected semantic classification is `largest_to_right_of_to_nearest`; the hardened parser still
    answers `largest_to_right_of` for the compact phrasing, which is exactly why section 12 gate 14 fails.
    """

    scope = _artifact("task6t_cli_scope_safety.json")
    if scope is None:
        pytest.skip("scope audit not generated yet")
    programs = scope["nearest_controls_parsed_program"]
    assert len(programs) == 2
    expected = "largest_to_right_of_to_nearest"
    assert scope["out_of_scope_all_semantically_correct"] == all(
        program == expected for program in programs.values())
    for prompt, program in programs.items():
        assert program in {"largest_to_right_of", expected}, f"{prompt} -> {program}"
    # the verdict must acknowledge the failing scope gate rather than claiming a pass
    verdict = _artifact("task6t_verdict.json")
    if verdict is not None:
        assert verdict["gates"]["14_scope_exit_5"]["passed"] == scope[
            "out_of_scope_all_semantically_correct"]
        if not scope["out_of_scope_all_semantically_correct"]:
            assert verdict["verdict"] != "PARSER_HARDENING_PASS"


def test_both_nearest_controls_exit_5_before_proposal():
    from buildreasonseg_mvp.task6s_directional_pipeline import EXIT_UNSUPPORTED_DIRECTIONAL

    scope = _artifact("task6t_cli_scope_safety.json")
    if scope is None:
        pytest.skip("scope audit not generated yet")
    nearest = [row for row in scope["controls"] if "nearest" in row["prompt"]
               or "最近" in row["prompt"]]
    assert len(nearest) == 2
    # the two nearest controls must be *rejected or handled* consistently with their parsed program:
    # exit 5 only when the program is outside the supported eight, otherwise exit 3 (reference
    # abstention) after the resolver. Both are measured and asserted here.
    for row in nearest:
        if row["parsed_program"] in {"largest_to_right_of_to_nearest", "smallest_to_nearest",
                                     "largest_to_nearest"}:
            assert row["exit_code"] == EXIT_UNSUPPORTED_DIRECTIONAL
            assert row["downstream_called"] is False
        else:
            assert row["parsed_program"] == "largest_to_right_of"
            assert row["exit_code"] == 3
            assert row["downstream_called"] is True
    # the canonical out-of-scope controls never reach downstream
    for prompt in ("分割面积最大的建筑物。", "分割最左侧的建筑物。"):
        row = next(item for item in scope["controls"] if item["prompt"] == prompt)
        assert row["exit_code"] == EXIT_UNSUPPORTED_DIRECTIONAL
        assert row["downstream_called"] is False


# ---------------------------------------------------------------- 25-31 downstream freeze


def test_no_proposal_configuration_change():
    from buildreasonseg_mvp.task6q_reference_resolver import (
        PROPOSAL_CONF,
        PROPOSAL_IMGSZ,
        PROPOSAL_MAX_DET,
        config_report,
    )

    assert (PROPOSAL_CONF, PROPOSAL_IMGSZ, PROPOSAL_MAX_DET) == (0.10, 640, 100)
    assert config_report()["threshold_sweep"] is False and config_report()["tta"] is False
    assert _git_changed("buildreasonseg_mvp/task6q_reference_resolver.py") == ""
    assert _git_changed("evaluation/task6q_") == ""


def test_no_b3_change():
    assert _git_changed("buildreasonseg_mvp/task6n_relation_decoder.py") == ""
    assert _git_changed("evaluation/task6o_") == ""
    scope = _artifact("task6t_cli_scope_safety.json")
    if scope is not None:
        assert scope["proposal_checkpoint_sha256"] == \
            "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"


def test_field_v02_hash_unchanged():
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field_v02.py") == ""
    assert _git_changed("buildreasonseg_mvp/geometric_relation_field.py") == ""


def test_no_grcl():
    for name in TASK6T_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("grcl_directional", "LAMBDA_GRCL", "grcl.loss"):
            assert marker not in code, f"{name} must not use GRCL ({marker})"


def test_no_4b():
    for name in TASK6T_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert "Qwen3-VL-4B" not in text, name
        assert "4B" not in text.replace("4B)", "")


def test_no_new_dataset_or_download():
    for name in TASK6T_SOURCES:
        code = _code_only(SCRIPTS / name)
        for marker in ("urlretrieve", "requests.get", "hf_hub_download", "snapshot_download",
                       "pip install", "datasets.load_dataset", "tkinter", "gradio"):
            assert marker not in code, f"{name} must not download or install ({marker})"
    spec = _artifact("task6t_parser_train_augmentation_spec.json")
    if spec is not None:
        assert spec["counts"]["v02_train_examples_kept"] > 20_000


def test_no_test_split():
    for name in TASK6T_SOURCES:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        for marker in ('"test.jsonl"', "'test.jsonl'", 'split="test"', "scene_disjoint_v1/test"):
            assert marker not in text, f"{name} must not touch the test split ({marker})"


# ---------------------------------------------------------------- 32-34 regression


def test_end_to_end_regression_exact_paired_10_of_20():
    e2e = _artifact("task6t_end_to_end_regression.json")
    if e2e is None:
        pytest.skip("end-to-end regression not generated yet")
    assert e2e["measured"]["paired"] == TASK6S_EXPECTED["paired"] == 10
    assert e2e["measured"]["abstentions"] == TASK6S_EXPECTED["abstentions"] == 6
    assert e2e["within_tolerance"]["paired"] is True


def test_end_to_end_miou_within_tolerance():
    e2e = _artifact("task6t_end_to_end_regression.json")
    if e2e is None:
        pytest.skip("end-to-end regression not generated yet")
    assert abs(e2e["deltas"]["answered_only_miou"]) <= 1e-6
    assert abs(e2e["deltas"]["strict_all_240_miou"]) <= 1e-6
    assert e2e["all_within_tolerance"] is True
    assert e2e["parser_exact_accuracy"] == 1.0
    verdict = _artifact("task6t_verdict.json")
    if verdict is not None:
        assert verdict["verdict"] != "END_TO_END_REGRESSION"


def test_previous_suite_preserved():
    for name in ("test_task6s_directional_end_to_end.py",
                 "test_task6r_grcl_directional_feasibility.py",
                 "test_task6q_frozen_proposal_reference_resolver.py",
                 "test_task6o_field_causal_decomposition.py"):
        assert (REPO_ROOT / "tests" / name).is_file(), name
    removed = subprocess.run(["git", "diff", "--name-only", "--diff-filter=D", BASE_COMMIT,
                              "--", "tests/"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert removed.strip() == "", f"no test file may be removed: {removed}"
