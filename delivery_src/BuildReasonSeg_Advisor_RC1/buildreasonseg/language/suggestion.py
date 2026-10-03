"""Suggestion flow for understood-but-unsupported programs (Task 8A section 7.3).

The flow never rewrites the user's command on its own: it shows the parsed result, offers the closest
**supported** program, asks `是否使用建议指令继续？ [Y/N]`, and only a Y continues. A rejected suggestion ends the
sample with command examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .registry import SUPPORTED_PROGRAMS, ParsedProgram
from .validator import ValidationResult, validate, validate_suggestion

SUGGESTION_QUESTION = "是否使用建议指令继续？"
EXAMPLES = (
    "最大建筑左侧最近的建筑",
    "最大建筑右侧最近的建筑",
    "最大建筑上方最近的建筑",
    "最大建筑下方最近的建筑",
)


@dataclass
class SuggestionFlow:
    parsed: ParsedProgram
    validation: ValidationResult
    confirmed: bool = False
    executed_program: str | None = None
    history: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        lines = [f"解析结果: {self.parsed.program or '(空)'} (来源: {self.parsed.source})"]
        if self.parsed.confidence is not None:
            lines.append(f"置信度: {self.parsed.confidence:.3f}")
        lines.append(f"状态: {self.validation.reason}")
        if self.validation.suggestion:
            lines.append(f"建议指令: {self.validation.suggestion}")
        return lines

    def confirm_suggestion(self, reader: Callable[[str], str] = input) -> bool:
        """Validate the suggestion, ask Y/N, and record the outcome. Only Y continues."""

        suggestion = self.validation.suggestion
        if not suggestion or not validate_suggestion(suggestion):
            self.history.append("suggestion_rejected_by_validator")
            return False
        self.history.append(f"suggested:{suggestion}")
        answer = (reader(f"{SUGGESTION_QUESTION} [Y/N]: ") or "").strip().lower()
        if answer in ("y", "yes", "是"):
            self.confirmed = True
            self.executed_program = suggestion
            self.history.append("user_confirmed")
            return True
        self.history.append("user_declined")
        return False

    def examples_text(self) -> str:
        return "可用命令示例: " + " | ".join(EXAMPLES) + \
            f"（正式支持: {', '.join(SUPPORTED_PROGRAMS)}）"


def run_suggestion_flow(parsed: ParsedProgram, reader: Callable[[str], str] = input,
                        *, echo: Callable[[str], None] = print) -> SuggestionFlow:
    validation = validate(parsed)
    flow = SuggestionFlow(parsed=parsed, validation=validation)
    for line in flow.describe():
        echo(line)
    if validation.supported:
        flow.confirmed = True
        flow.executed_program = parsed.program
        flow.history.append("supported_directly")
        return flow
    if validation.suggestion:
        flow.confirm_suggestion(reader)
    if not flow.confirmed:
        echo(flow.examples_text())
    return flow


__all__ = ["EXAMPLES", "SUGGESTION_QUESTION", "SuggestionFlow", "run_suggestion_flow"]
