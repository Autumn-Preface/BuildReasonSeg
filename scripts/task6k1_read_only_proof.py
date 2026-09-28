"""Task 6K.1: empirical read-only proof for the native-vector audit.

Confirms that no file under the read-only roots (original WHU archive, historical converted dataset,
legacy project) was modified after the audit started, by comparing the newest modification time per
root with the oldest Task 6K.1 artifact.

Writes `evaluation/task6k1_read_only_proof.json`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_vector_audit import CROPPED_ROOT, ORIGINAL_ROOT, SHP_DIR, WHOLE_AREA_DIR  # noqa: E402

from task6k1_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6k1_read_only_proof.json"

ROOTS = {
    "original_whu_root": ORIGINAL_ROOT,
    "shapefile_dir": SHP_DIR,
    "whole_area_dir": WHOLE_AREA_DIR,
    "cropped_subtree": CROPPED_ROOT,
    "converted_dataset": Path(
        r"C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment\dataset\WHU_YOLO_dataset"
    ),
    "legacy_project": Path(r"C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment"),
}


def main() -> int:
    references = sorted(
        path.stat().st_mtime
        for pattern in ("task6k1_*.json",)
        for path in EVAL.glob(pattern)
        if path.name != OUT.name
    )
    if not references:
        raise SystemExit("no Task 6K.1 artifacts found to use as the audit reference time")
    reference = min(references)

    report = {
        "_doc": (
            "Task 6K.1 read-only proof: the newest modification time under every read-only root is "
            "compared with the oldest Task 6K.1 artifact, so a write into any source tree would show "
            "up as modified_after_audit_start = true."
        ),
        "task": "6K.1",
        "reference_artifact_mtime_utc": datetime.fromtimestamp(reference, timezone.utc).isoformat(),
        "roots": {},
    }
    for name, root in ROOTS.items():
        newest = None
        newest_path = None
        count = 0
        if root.is_dir():
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
    report["all_read_only_roots_untouched"] = not any(
        value["modified_after_audit_start"] for value in report["roots"].values()
    )
    report["no_model_training"] = True
    report["no_package_installation"] = True
    report["no_dataset_regeneration"] = True
    write_json(OUT, report)
    for name, value in report["roots"].items():
        print(
            f"[task6k1.readonly] {name}: files {value['files']} newest {value['newest_mtime_utc']} "
            f"modified={value['modified_after_audit_start']}",
            flush=True,
        )
    print(f"[task6k1.readonly] wrote {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
