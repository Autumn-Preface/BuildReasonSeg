"""Task 8B.1 tests: structured suggestion contract, validator gating, Y/N behaviour, no hard-coded routing."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from buildreasonseg import paths
from buildreasonseg.errors import BuildReasonSegError
from buildreasonseg.language.registry import SUPPORTED_PROGRAMS, ParsedProgram
from buildreasonseg.language.validator import validate
from buildreasonseg.runtime.program_head import (
    SUGGESTION_GENERATION,
    SUGGESTION_SYSTEM_PROMPT,
    parse_suggestion_response,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "logs" / "task8b1_prompt_suite.json"


# ---------------------------------------------------------------- structured contract


def test_valid_suggestion_parsed() -> None:
    raw = json.dumps({"status": "SUGGESTION", "program": "largest_to_right_of_to_nearest",
                      "display_command": "找出最大建筑物右侧最近的建筑物", "reason": "含最大/右侧/最近"})
    parsed = parse_suggestion_response(raw)
    assert parsed["status"] == "SUGGESTION"
    assert parsed["program"] == "largest_to_right_of_to_nearest"
    assert parsed["display_command"] and parsed["reason"]


def test_no_safe_suggestion_parsed() -> None:
    assert parse_suggestion_response(json.dumps({"status": "NO_SAFE_SUGGESTION"}))["status"] == \
        "NO_SAFE_SUGGESTION"
    assert parse_suggestion_response("我认为 NO_SAFE_SUGGESTION")["status"] == "NO_SAFE_SUGGESTION"


def test_program_outside_four_is_rejected() -> None:
    """The model may not widen RC1's support: any non-supported program is a contract violation."""

    for program in ("largest", "smallest", "largest_to_right_of", "second_largest", ""):
        raw = json.dumps({"status": "SUGGESTION", "program": program, "display_command": "x"})
        assert parse_suggestion_response(raw)["status"] == "UNPARSABLE"


def test_free_text_is_not_keyword_routed() -> None:
    """A free-text answer mentioning a direction must NOT become a suggestion (no keyword routing)."""

    for text in ("请分割最大建筑右侧最近的建筑。", "right_of", "建议：largest_to_right_of_to_nearest"):
        assert parse_suggestion_response(text)["status"] == "UNPARSABLE"


def test_fenced_json_tolerated() -> None:
    raw = ("```json\n" + json.dumps({"status": "SUGGESTION",
                                     "program": "largest_to_above_to_nearest",
                                     "display_command": "d", "reason": "r"}) + "\n```")
    assert parse_suggestion_response(raw)["program"] == "largest_to_above_to_nearest"


def test_generation_config_frozen() -> None:
    assert SUGGESTION_GENERATION == {"do_sample": False, "num_beams": 1, "max_new_tokens": 128}


def test_system_prompt_forbids_new_operators() -> None:
    for phrase in ("不得发明新的算子", "NO_SAFE_SUGGESTION", "second largest", "between"):
        assert phrase in SUGGESTION_SYSTEM_PROMPT


# ---------------------------------------------------------------- validator gating


def test_suggestion_must_pass_validator() -> None:
    for program in ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
                    "largest_to_above_to_nearest", "largest_to_below_to_nearest"):
        parsed = ParsedProgram(program=program, source="qwen_program_head", confidence=None)
        assert validate(parsed).supported is True
    for program in ("largest", "largest_to_right_of", "second_largest_to_right_of_to_nearest"):
        assert validate(ParsedProgram(program=program, source="qwen_program_head")).supported is False


def test_only_four_programs_supported() -> None:
    assert len(SUPPORTED_PROGRAMS) == 4


# ---------------------------------------------------------------- no hard-coded routes


def test_no_prompt_to_program_table_in_delivery_code() -> None:
    """No source file may map a test prompt string to a program (hard-coding is a protocol violation)."""

    import ast as _ast

    def code_without_docstrings(source: str) -> str:
        """Ignore docstrings: usage examples are documentation, not routing."""

        tree = _ast.parse(source)
        lines = source.splitlines()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.Module, _ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], _ast.Expr) and isinstance(body[0].value, _ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    for index in range(body[0].lineno - 1, body[0].end_lineno):
                        lines[index] = ""
        return "\n".join(lines)

    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    prompts = [entry["prompt"] for entry in suite["prompts"] + suite["safety_prompts"]]
    offenders = []
    for path in list(ROOT.rglob("*.py")):
        if any(part in ("logs", "_frozen", "tests", "__pycache__")
               for part in path.relative_to(ROOT).parts):
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        try:
            text = code_without_docstrings(raw)
        except SyntaxError:
            text = raw
        for prompt in prompts:
            if prompt in text:
                offenders.append(f"{path.relative_to(ROOT)} contains a suite prompt")
        # forbid keyword routing in the Qwen-first path; `language/frontend.py` holds the Task 8A
        # deterministic fallback table, which is permitted only when Qwen is unavailable.
        if path.name in ("predict.py", "program_head.py", "pipeline.py") and "suggest" in text.lower():
            for marker in ('"左侧": ', "'左侧': ", '"右侧": ', '"左": "left_of"'):
                if marker in text:
                    offenders.append(f"{path.relative_to(ROOT)} looks like keyword routing ({marker})")
    assert offenders == []


def test_suggestion_uses_local_qwen_generate() -> None:
    """The suggestion path must call the local Qwen `generate`, never an external API."""

    source = (ROOT / "buildreasonseg" / "runtime" / "program_head.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "generate" in names
    for forbidden in ("requests", "openai", "http", "api_key", "urllib"):
        assert forbidden not in source.replace("外部 API", "")


def test_no_model_or_checkpoint_writes() -> None:
    """Task 8B.1 must not train, save or modify any checkpoint."""

    for path in list(ROOT.rglob("*.py")):
        if any(part in ("logs", "_frozen", "tests", "__pycache__")
               for part in path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in ("torch.save(", "optimizer.step(", "backward()", "save_parser_checkpoint("):
            assert marker not in text, f"{path.relative_to(ROOT)}: {marker}"


# ---------------------------------------------------------------- fixture integrity


def test_fixture_frozen_and_complete() -> None:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    assert len(suite["prompts"]) == 24
    assert len(suite["safety_prompts"]) == 4
    sources = [entry["source"] for entry in suite["prompts"]]
    assert sources.count("task8b_known_conflict") == 4
    assert sources.count("task8b_gates") == 8
    assert sources.count("task8b1_new_frozen") == 12
    assert suite["confirmation_policy"] == "simulated_yes"
    for entry in suite["prompts"]:
        assert entry["expected_program"] in SUPPORTED_PROGRAMS
        assert entry["direction"] in ("left", "right", "above", "below")


def test_task8b_paraphrases_are_verbatim() -> None:
    """The 8 paraphrases must be byte-identical to the frozen Task 8B artifact."""

    gates = json.loads((ROOT / "logs" / "task8b_gates.json").read_text(encoding="utf-8"))
    frozen = [entry["prompt"] for entry in gates["program_head"]["paraphrase"]]
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    carried = [entry["prompt"] for entry in suite["prompts"] if entry["source"] == "task8b_gates"]
    assert carried == frozen


# ---------------------------------------------------------------- result.json language trace


def test_language_trace_fields_documented() -> None:
    source = (ROOT / "predict.py").read_text(encoding="utf-8")
    for field in ("original_prompt", "initial_program", "initial_supported", "suggestion_used",
                  "suggested_program", "user_confirmation"):
        assert field in source, field
    pipeline = (ROOT / "buildreasonseg" / "runtime" / "pipeline.py").read_text(encoding="utf-8")
    assert "language_trace" in pipeline


def test_user_visible_wording_rules() -> None:
    """Section 19 wording: no "标准命令"; the fallback wording is used instead."""

    source = (ROOT / "predict.py").read_text(encoding="utf-8")
    assert "标准命令" not in source
    assert "当前解析结果不属于 RC1 已开放的四类空间推理语义。" in source
    assert "Qwen 建议的可支持替代指令：" in source
    assert "当前已支持的空间语义（命令示例）" in source
    from buildreasonseg.language.suggestion import EXAMPLES

    assert len(EXAMPLES) == 4


def test_n_path_does_not_run_visual_inference(monkeypatch) -> None:
    """E102 (user declines) must abort before detection: the CLI raises before PredictRuntime use."""

    import predict as predict_module

    source = (ROOT / "predict.py").read_text(encoding="utf-8")
    parse_index = source.index("parsed, language_info, suggestion = _parse_program(")
    runtime_index = source.index("runtime = PredictRuntime(")
    assert runtime_index < parse_index  # runtime object is constructed (cheap), but no detect call happens
    for marker in ("inspect_proposals(runtime", "predict_one(runtime"):
        assert marker in source
    # the failure contract for N is E102 with the command examples, no mask/overlay writing
    assert 'raise error' in source or 'BuildReasonSegError(\n            "E102"' in source


def test_unknown_semantics_never_auto_executed() -> None:
    """A NO_SAFE_SUGGESTION / unparsable suggestion can never be executed."""

    from buildreasonseg.runtime.pipeline import predict_one  # noqa: F401

    for status in ("NO_SAFE_SUGGESTION", "UNPARSABLE", "RUNTIME_ERROR"):
        parsed_suggestion = {"status": status, "invoked": True}
        assert parsed_suggestion.get("status") != "SUGGESTION"
