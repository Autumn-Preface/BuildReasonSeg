"""Fallback availability contract (Task 8A section 7.4).

The deterministic parser is only permitted for the three declared cases and always announces itself with
`MODE: FALLBACK COMMAND PARSER` plus an explicit user confirmation. This module exposes that contract so the CLI
and the tests share one source of truth.
"""

from __future__ import annotations

from .frontend import FALLBACK_BANNER, DeterministicFallbackFrontend, frontend_status

PERMITTED_REASONS = ("qwen_asset_missing", "qwen_load_failed", "qwen_runtime_failed")

NORMAL_PATH_FORBIDDEN_REASONS = ("user_prefers_shortcut", "regex_faster", "prompt_looks_canonical")


def fallback_status() -> dict:
    status = frontend_status()
    qwen_missing = not status.qwen_available
    return {
        "banner": FALLBACK_BANNER,
        "qwen_first": True,
        "permitted_reasons": list(PERMITTED_REASONS),
        "forbidden_reasons": list(NORMAL_PATH_FORBIDDEN_REASONS),
        "qwen_available": status.qwen_available,
        "fallback_currently_available": qwen_missing,
        "fallback_reason_now": "qwen_asset_missing" if qwen_missing else None,
        "fallback_may_become_normal_path": False,
        "requires_user_confirmation": True,
        "implementation": DeterministicFallbackFrontend.__name__,
    }


__all__ = ["NORMAL_PATH_FORBIDDEN_REASONS", "PERMITTED_REASONS", "fallback_status"]
