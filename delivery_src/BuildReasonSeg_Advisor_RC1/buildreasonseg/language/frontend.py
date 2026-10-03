"""Language front end interface, Qwen-first contract and the deterministic fallback (Task 8A section 7).

Normal chain (frozen):

    user prompt -> Qwen / ProgramHead -> parsed program -> hard validator
                -> supported: execution     | unsupported: suggestion flow + user confirmation

The deterministic fallback exists **only** for the three permitted cases (Qwen asset missing, Qwen load
failure, Qwen runtime failure). It prints `MODE: FALLBACK COMMAND PARSER`, requires explicit user confirmation,
and can never silently replace the Qwen path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from buildreasonseg import component_dir, paths
from buildreasonseg.errors import BuildReasonSegError, NotInTask8AError

from .registry import ParsedProgram
from .validator import ValidationResult, validate

FALLBACK_BANNER = "MODE: FALLBACK COMMAND PARSER"

QWEN_FAMILY = "Qwen3-VL-2B-Instruct"
PROGRAM_HEAD_CHECKPOINT = "program_parser_l3_rerehearsal_v1.pt"  # replaced below by the exact file name
PROGRAM_HEAD_CHECKPOINT = "program_parser_l3_rehearsal_v1.pt"
QWEN_ASSET_DIRNAME = "Qwen3-VL-2B-Instruct"


@dataclass
class FrontendStatus:
    qwen_available: bool
    reason: str = ""
    missing_assets: list[str] = field(default_factory=list)
    checkpoint_path: str | None = None
    qwen_asset_root: str | None = None


def qwen_assets_present() -> bool:
    root = component_dir("program_head")
    return (root / PROGRAM_HEAD_CHECKPOINT).is_file() and (root / QWEN_ASSET_DIRNAME).is_dir()


def frontend_status() -> FrontendStatus:
    root = component_dir("program_head")
    missing = [name for name in (PROGRAM_HEAD_CHECKPOINT, QWEN_ASSET_DIRNAME)
               if not (root / name).exists()]
    available = not missing
    return FrontendStatus(qwen_available=available,
                          reason="" if available else "Qwen / ProgramHead 资产缺失",
                          missing_assets=missing,
                          checkpoint_path=str(root / PROGRAM_HEAD_CHECKPOINT),
                          qwen_asset_root=str(root / QWEN_ASSET_DIRNAME))


class LanguageFrontend(Protocol):
    """The frozen front-end interface. Implementations must return `ParsedProgram` only."""

    name: str

    def parse(self, prompt: str) -> ParsedProgram:  # pragma: no cover - protocol
        ...

    def available(self) -> tuple[bool, str]:  # pragma: no cover - protocol
        ...


class QwenProgramHeadFrontend:
    """Qwen-first front end.

    Task 8A freezes the interface and the asset contract. The real inference call belongs to Task 8B: until it
    exists this implementation raises `NOT_IMPLEMENTED_IN_TASK_8A` instead of faking a parse.
    """

    name = "qwen_program_head"

    def available(self) -> tuple[bool, str]:
        status = frontend_status()
        if status.qwen_available:
            return True, f"{QWEN_FAMILY} + ProgramHead 资产齐备"
        return False, f"缺少资产: {', '.join(status.missing_assets)}"

    def parse(self, prompt: str) -> ParsedProgram:  # noqa: D401 - contract method
        if not prompt or not prompt.strip():
            raise BuildReasonSegError("E101", detail="指令为空。")
        available, reason = self.available()
        if not available:
            raise BuildReasonSegError("E302", detail=f"Qwen 前端不可用: {reason}")
        raise NotInTask8AError(
            detail="Qwen / ProgramHead 真实推理集成将在 Task 8B 实现；Task 8A 仅冻结接口与资产契约。")


class DeterministicFallbackFrontend:
    """Fallback parser: keyword/structure matching against the frozen supported-program list only."""

    name = "deterministic_fallback"

    #: explicit trigger reasons that permit fallback
    PERMITTED_REASONS = ("qwen_asset_missing", "qwen_load_failed", "qwen_runtime_failed")

    DIRECTIONS_ZH = {"左侧": "left_of", "左边": "left_of", "左方": "left_of", "right": None,
                     "右侧": "right_of", "右边": "right_of", "右方": "right_of",
                     "上方": "above", "上面": "above", "上侧": "above",
                     "下方": "below", "下面": "below", "下侧": "below"}
    DIRECTIONS_EN = {"left": "left_of", "right": "right_of", "above": "above", "below": "below",
                     "upper": "above", "lower": "below"}

    def __init__(self, reason: str) -> None:
        if reason not in self.PERMITTED_REASONS:
            raise ValueError(f"fallback not permitted for reason {reason!r}")

    def available(self) -> tuple[bool, str]:
        return True, f"fallback permitted: {self.reason}" if hasattr(self, "reason") else "fallback"

    def parse(self, prompt: str) -> ParsedProgram:
        text = (prompt or "").strip().lower()
        if not text:
            raise BuildReasonSegError("E101", detail="指令为空。")
        direction = None
        for token, value in self.DIRECTIONS_ZH.items():
            if token in text:
                direction = value
                break
        if direction is None:
            for token, value in self.DIRECTIONS_EN.items():
                if token in text:
                    direction = value
                    break
        family = "largest" if ("largest" in text or "最大" in text or "面积最大" in text) else None
        terminal = "nearest" if ("nearest" in text or "最近" in text) else None
        if family == "largest" and direction and terminal == "nearest":
            program = f"largest_to_{direction}_to_nearest"
        elif family == "largest" and direction:
            program = f"largest_to_{direction}"
        elif family == "largest":
            program = "largest"
        else:
            program = ""
        return ParsedProgram(program=program, source="deterministic_fallback", confidence=None,
                            raw_text=prompt, normalized_text=text, diagnostics={
                                "fallback_reason": getattr(self, "reason", "unspecified"),
                                "matched_direction": direction, "matched_family": family,
                                "matched_terminal": terminal})


def parse_prompt(prompt: str, *, qwen_failure_reason: str | None = None) -> tuple[ParsedProgram,
                                                                                 ValidationResult]:
    """Run the frozen chain: Qwen-first, validator-controlled, deterministic fallback only when unavailable."""

    frontend = QwenProgramHeadFrontend()
    available, reason = frontend.available()
    if available and qwen_failure_reason is None:
        parsed = frontend.parse(prompt)
    else:
        if qwen_failure_reason is None:
            qwen_failure_reason = "qwen_asset_missing" if not available else "qwen_runtime_failed"
        print(FALLBACK_BANNER)
        fallback = DeterministicFallbackFrontend(qwen_failure_reason)
        fallback.reason = qwen_failure_reason
        parsed = fallback.parse(prompt)
    return parsed, validate(parsed)


def confirm(question: str, reader=input) -> bool:
    """Ask a Y/N question; only an explicit Y/是 continues (Task 8A sections 7.3/7.4/13)."""

    while True:
        answer = (reader(f"{question} [Y/N]: ") or "").strip().lower()
        if answer in ("y", "yes", "是"):
            return True
        if answer in ("n", "no", "否", ""):
            return False


__all__ = ["FALLBACK_BANNER", "FrontendStatus", "LanguageFrontend", "PROGRAM_HEAD_CHECKPOINT",
           "QWEN_ASSET_DIRNAME", "QWEN_FAMILY", "DeterministicFallbackFrontend",
           "QwenProgramHeadFrontend", "confirm", "frontend_status", "parse_prompt",
           "qwen_assets_present"]
