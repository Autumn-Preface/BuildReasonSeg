"""Language layer: Qwen-first front end, hard validator, suggestion flow, deterministic fallback."""

from __future__ import annotations

from .fallback import fallback_status  # noqa: F401
from .frontend import (  # noqa: F401
    FALLBACK_BANNER,
    DeterministicFallbackFrontend,
    QwenProgramHeadFrontend,
    confirm,
    frontend_status,
    parse_prompt,
    qwen_assets_present,
)
from .registry import SUPPORTED_PROGRAMS, ParsedProgram, registry  # noqa: F401
from .suggestion import SuggestionFlow, run_suggestion_flow  # noqa: F401
from .validator import ValidationResult, suggest_supported, validate  # noqa: F401

__all__ = ["FALLBACK_BANNER", "DeterministicFallbackFrontend", "ParsedProgram",
           "QwenProgramHeadFrontend", "SUPPORTED_PROGRAMS", "SuggestionFlow", "ValidationResult",
           "confirm", "fallback_status", "frontend_status", "parse_prompt",
           "qwen_assets_present", "registry", "run_suggestion_flow", "suggest_supported",
           "validate"]
