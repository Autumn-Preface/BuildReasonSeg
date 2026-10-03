"""Language-contract tests (Task 8A section 19.4)."""

from __future__ import annotations

import pytest

from buildreasonseg.errors import BuildReasonSegError, NotInTask8AError
from buildreasonseg.language import (FALLBACK_BANNER, DeterministicFallbackFrontend,
                                     QwenProgramHeadFrontend, SUPPORTED_PROGRAMS, ParsedProgram,
                                     fallback_status, parse_prompt, registry,
                                     run_suggestion_flow, suggest_supported, validate)
from buildreasonseg.language.suggestion import SuggestionFlow


def test_normal_path_is_qwen_first() -> None:
    frontend = QwenProgramHeadFrontend()
    assert frontend.name == "qwen_program_head"
    available, reason = frontend.available()
    assert available is True, reason
    # the real inference call belongs to Task 8B: the interface must not fake a parse
    with pytest.raises(NotInTask8AError):
        frontend.parse("最大建筑左侧最近的建筑")


def test_supported_registry_exactly_four_l3_programs() -> None:
    assert len(SUPPORTED_PROGRAMS) == 4
    assert set(SUPPORTED_PROGRAMS) == {
        "largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
        "largest_to_above_to_nearest", "largest_to_below_to_nearest"}
    assert registry()["executable_program_count"] == 4


def test_smallest_programs_are_not_supported() -> None:
    parsed = ParsedProgram(program="smallest_to_left_of_to_nearest", source="qwen_program_head")
    result = validate(parsed)
    assert result.supported is False
    assert result.code == "E102"
    assert result.suggestion in SUPPORTED_PROGRAMS


def test_l2_program_is_unsupported_but_suggested() -> None:
    parsed = ParsedProgram(program="largest_to_left_of", source="qwen_program_head")
    result = validate(parsed)
    assert result.supported is False
    assert result.suggestion == "largest_to_left_of_to_nearest"


def test_supported_program_passes_validator() -> None:
    for program in SUPPORTED_PROGRAMS:
        result = validate(ParsedProgram(program=program, source="qwen_program_head"))
        assert result.supported is True
        assert result.suggestion is None
        assert result.code is None


def test_suggestion_must_pass_validator() -> None:
    from buildreasonseg.language.validator import validate_suggestion

    assert validate_suggestion("largest_to_left_of_to_nearest") is True
    assert validate_suggestion("smallest_to_left_of_to_nearest") is False
    assert suggest_supported(ParsedProgram(program="totally_unknown",
                                           source="qwen_program_head")) in (None, *SUPPORTED_PROGRAMS)


def test_suggestion_requires_yn() -> None:
    from buildreasonseg.language.suggestion import SUGGESTION_QUESTION

    parsed = ParsedProgram(program="smallest_to_above_to_nearest", source="qwen_program_head")
    echoed: list[str] = []
    questions: list[str] = []

    def reader(question: str) -> str:
        questions.append(question)
        return "y"

    flow = run_suggestion_flow(parsed, reader=reader, echo=echoed.append)
    assert isinstance(flow, SuggestionFlow)
    assert flow.confirmed is True
    assert flow.executed_program == "largest_to_above_to_nearest"
    assert any("建议指令" in line for line in echoed)
    assert questions and SUGGESTION_QUESTION in questions[0] and "[Y/N]" in questions[0]

    declined = run_suggestion_flow(parsed, reader=lambda question: "n", echo=echoed.append)
    assert declined.confirmed is False
    assert any("命令示例" in line for line in echoed)


def test_suggestion_flow_does_not_auto_rewrite() -> None:
    parsed = ParsedProgram(program="smallest_to_below_to_nearest", source="qwen_program_head")
    flow = run_suggestion_flow(parsed, reader=lambda question: "n", echo=lambda line: None)
    assert flow.parsed.program == "smallest_to_below_to_nearest"
    assert flow.executed_program is None
    assert "user_declined" in flow.history


def test_fallback_cannot_become_normal_path() -> None:
    status = fallback_status()
    assert status["qwen_first"] is True
    assert status["fallback_may_become_normal_path"] is False
    assert status["requires_user_confirmation"] is True
    assert "user_prefers_shortcut" in status["forbidden_reasons"]
    with pytest.raises(ValueError):
        DeterministicFallbackFrontend("user_prefers_shortcut")
    permitted = DeterministicFallbackFrontend("qwen_asset_missing")
    permitted.reason = "qwen_asset_missing"
    parsed = permitted.parse("最大建筑左侧最近的建筑")
    assert parsed.source == "deterministic_fallback"
    assert parsed.program == "largest_to_left_of_to_nearest"


def test_fallback_banner_constant() -> None:
    assert FALLBACK_BANNER == "MODE: FALLBACK COMMAND PARSER"


def test_parse_prompt_uses_fallback_only_when_qwen_unavailable(capsys) -> None:
    parsed, validation = parse_prompt("最大建筑上方最近的建筑",
                                      qwen_failure_reason="qwen_load_failed")
    captured = capsys.readouterr().out
    assert FALLBACK_BANNER in captured
    assert parsed.source == "deterministic_fallback"
    assert validation.supported is True
    note = " ".join(validation.notes)
    assert "fallback" in note


def test_parse_prompt_empty_raises_invalid_command() -> None:
    with pytest.raises(BuildReasonSegError) as error:
        parse_prompt("   ", qwen_failure_reason="qwen_asset_missing")
    assert error.value.code == "E101"


def test_parsed_program_schema_fields() -> None:
    parsed = ParsedProgram(program="largest_to_below_to_nearest", source="qwen_program_head",
                           confidence=0.93, raw_text="最大建筑下方最近的建筑")
    assert parsed.reference_family == "largest"
    assert parsed.direction == "below"
    assert parsed.terminal == "nearest"
    assert parsed.is_supported is True
    assert parsed.is_qwen_parsed is True
    payload = parsed.to_dict()
    assert payload["program"] == "largest_to_below_to_nearest"
    with pytest.raises(ValueError):
        ParsedProgram(program="largest", source="not_a_source")
