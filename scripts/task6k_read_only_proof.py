"""Empirical read-only proof: no file under the read-only roots was modified during the audit."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp.whu_source_audit import (  # noqa: E402
    CONVERTED_CANDIDATES,
    CROPPED_ROOT,
    LEGACY_PROJECTS,
    ORIGINAL_ROOT,
)

# The audit artifacts were produced during this session; compare source mtimes against the
# oldest artifact mtime as the session reference point.
reference = min(
    path.stat().st_mtime
    for path in (REPO_ROOT / "evaluation").glob("task6k_*.json")
)

roots = {
    "original_whu_root": ORIGINAL_ROOT,
    "cropped_subtree": CROPPED_ROOT,
    "converted_dataset": CONVERTED_CANDIDATES[1],
    "legacy_project": LEGACY_PROJECTS[0],
}

report = {"reference_artifact_mtime": datetime.fromtimestamp(reference, timezone.utc).isoformat(), "roots": {}}
for name, root in roots.items():
    newest = None
    newest_path = None
    count = 0
    for path in root.rglob("*"):
        if path.is_file():
            count += 1
            mtime = path.stat().st_mtime
            if newest is None or mtime > newest:
                newest = mtime
                newest_path = path
    report["roots"][name] = {
        "path": str(root),
        "exists": root.is_dir(),
        "files": count,
        "newest_mtime_utc": None if newest is None else datetime.fromtimestamp(newest, timezone.utc).isoformat(),
        "newest_file": None if newest_path is None else str(newest_path),
        "modified_after_audit_start": bool(newest is not None and newest > reference),
    }

out = REPO_ROOT / "evaluation" / "task6k_read_only_proof.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
(REPO_ROOT / "artifacts" / "task6k").mkdir(parents=True, exist_ok=True)
print(json.dumps(report, indent=2, ensure_ascii=False))
print("wrote", out)
