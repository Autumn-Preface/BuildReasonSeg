"""Hard validator: the only gate between a parsed program and execution (Task 8A section 7.1-7.3).

Rules:
* a supported program passes;
* an unsupported (but understood) program must go through the suggestion flow and the user must confirm;
* a suggestion must itself be one of the supported programs and must be re-validated;
* a deterministic-fallback parse is never silently promoted to the normal Qwen path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .registry import SUPPORTED_PROGRAMS, ParsedProgram


@dataclass
class ValidationResult:
    parsed: ParsedProgram
    supported: bool
    reason: str = ""
    suggestion: str | None = None
    code: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def requires_user_confirmation(self) -> bool:
        return bool(self.suggestion) and not self.supported

    def to_dict(self) -> dict:
        return {"program": self.parsed.program, "supported": self.supported, "reason": self.reason,
                "suggestion": self.suggestion, "code": self.code, "notes": list(self.notes),
                "source": self.parsed.source}


def validate(parsed: ParsedProgram) -> ValidationResult:
    if not parsed.program:
        return ValidationResult(parsed=parsed, supported=False, code="E101",
                                reason="解析结果为空，无法确定空间关系程序。")
    if parsed.program in SUPPORTED_PROGRAMS:
        notes = []
        if parsed.source == "deterministic_fallback":
            notes.append("fallback parse: 仅在 Qwen 不可用时允许，且需用户确认后继续。")
        return ValidationResult(parsed=parsed, supported=True,
                                reason="program 在 RC1 正式支持列表内。", notes=notes)
    return ValidationResult(parsed=parsed, supported=False, code="E102",
                            reason=f"program {parsed.program!r} 不在 RC1 正式支持列表内。",
                            suggestion=suggest_supported(parsed))


def suggest_supported(parsed: ParsedProgram) -> str | None:
    """Suggest the closest *supported* program, using only the frozen support list."""

    if not SUPPORTED_PROGRAMS:
        return None
    program = parsed.program or ""
    if program in SUPPORTED_PROGRAMS:
        return program
    # smallest -> largest with the same direction/terminal shape is the closest supported analogue
    if program.startswith("smallest_to_"):
        candidate = "largest_to_" + program[len("smallest_to_"):]
        if candidate in SUPPORTED_PROGRAMS:
            return candidate
    if program.startswith("largest_to_") and program.endswith("_to_nearest"):
        candidate = program.replace("_to_nearest", "") + "_to_nearest"
        if candidate in SUPPORTED_PROGRAMS:
            return candidate
    if program.startswith("largest_to_") and program not in SUPPORTED_PROGRAMS:
        candidate = program + "_to_nearest"
        if candidate in SUPPORTED_PROGRAMS:
            return candidate
    for direction in ("left_of", "right_of", "above", "below"):
        candidate = f"largest_to_{direction}_to_nearest"
        if candidate in SUPPORTED_PROGRAMS and direction in program:
            return candidate
    return None


def validate_suggestion(suggestion: str) -> bool:
    """A suggestion is only usable when it is itself a supported program."""

    return suggestion in SUPPORTED_PROGRAMS


__all__ = ["ValidationResult", "suggest_supported", "validate", "validate_suggestion"]
