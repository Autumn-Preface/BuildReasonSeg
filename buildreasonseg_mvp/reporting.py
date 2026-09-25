"""Merging report writer for the Task 6A stages.

`evaluation/task6a_smoke_report.json` is the machine-readable deliverable. Each
stage script contributes its own section and the file is merged rather than
overwritten, so running the stages separately or in sequence produces the same
result.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_REPORT = REPO_ROOT / "evaluation" / "task6a_smoke_report.json"
SUBSET_IDS = REPO_ROOT / "evaluation" / "task6a_subset_ids.json"
CHECKPOINT_MANIFEST = REPO_ROOT / "evaluation" / "task6a_checkpoint_manifest.json"

VERDICTS = ("PASS", "PASS_WITH_WARNINGS", "FAIL_REQUIRES_DEBUG")


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def update_report(section: str, payload: dict, extra: dict | None = None) -> dict:
    """Merge one section into the smoke report and rewrite it."""

    report = _read(SMOKE_REPORT)
    report.setdefault(
        "_doc",
        "Task 6A machine-readable smoke report. Sections are written by the stage scripts.",
    )
    report["task"] = "6A"
    report[section] = payload
    for key, value in (extra or {}).items():
        report[key] = value
    SMOKE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return report


def set_verdict(verdict: str, blockers: list[str] | None = None, warnings: list[str] | None = None) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    return update_report(
        "final_verdict",
        {"verdict": verdict, "blockers": blockers or [], "warnings": warnings or []},
    )


def write_subset_ids(payload: dict) -> None:
    SUBSET_IDS.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_subset_ids() -> dict:
    return _read(SUBSET_IDS)
