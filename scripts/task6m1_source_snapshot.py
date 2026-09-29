"""Task 6M.1 Part A (section 4): verify and snapshot the Task 6M 18-epoch checkpoint state.

Recomputes the SHA256 of the Task 6M `best.pt` / `last.pt`, requires an exact match with the hashes
recorded by Task 6M, copies (never moves) both checkpoints plus `results.csv` and the minimum resume
metadata into a Task 6M.1-local snapshot, re-verifies the copies, and writes
`evaluation/task6m1_source_checkpoint_audit.json`.

The snapshot is never overwritten: if it already exists, this script verifies it instead of replacing
it, so the 18-epoch state stays provably frozen.

    python scripts/task6m1_source_snapshot.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import CHECKPOINT_ROOT, EVAL, write_json  # noqa: E402

#: Task 6M recorded hashes (handoff/TO_DSH.md section 4).
EXPECTED = {
    "best.pt": "fd407db634a8a7ef83f09f8096686e73407095105f1b45c70c623d18dbf4ea44",
    "last.pt": "ea998bda37dd2dcb2cc7bb5e19f6d15b7a205a137c9cc2866c508a45513f860e",
}
SOURCE_RUN = CHECKPOINT_ROOT / "runs" / "m1_yolo26m_seg"
SNAPSHOT = CHECKPOINT_ROOT.parent / "task6m1" / "source_epoch18_snapshot"
OUT = EVAL / "task6m1_source_checkpoint_audit.json"
RESUME_METADATA = ("args.yaml", "results.csv", "labels.jpg", "train_batch0.jpg")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-rematch", action="store_true",
                        help="permit a hash that matches the snapshot but not the recorded constant")
    args = parser.parse_args(argv)

    started = time.time()
    checks = {}
    for name, expected in EXPECTED.items():
        path = SOURCE_RUN / "weights" / name
        if not path.is_file():
            write_json(
                OUT,
                {
                    "_doc": "Task 6M.1 section 4 source checkpoint audit.",
                    "verdict": "SOURCE_CHECKPOINT_MISMATCH",
                    "reason": f"missing {path}",
                },
            )
            print(f"[6m1.snapshot] SOURCE_CHECKPOINT_MISMATCH: missing {path}", flush=True)
            return 2
        actual = sha256_file(path)
        checks[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }

    all_match = all(entry["matches"] for entry in checks.values())
    if not all_match:
        write_json(
            OUT,
            {
                "_doc": "Task 6M.1 section 4 source checkpoint audit.",
                "task": "6M.1",
                "verdict": "SOURCE_CHECKPOINT_MISMATCH",
                "checkpoints": checks,
                "reason": "recomputed Task 6M checkpoint hashes do not match the recorded ones",
            },
        )
        print(f"[6m1.snapshot] SOURCE_CHECKPOINT_MISMATCH: {checks}", flush=True)
        return 2

    # ---------------- snapshot (copy, never move; never overwrite an existing snapshot)
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT / "weights").mkdir(parents=True, exist_ok=True)
    snapshot_records = {}
    for name in EXPECTED:
        source = SOURCE_RUN / "weights" / name
        target = SNAPSHOT / "weights" / name
        if target.exists():
            status = "already_present_verified"
        else:
            shutil.copy2(source, target)
            status = "copied"
        digest = sha256_file(target)
        snapshot_records[name] = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": digest,
            "matches_original": digest == sha256_file(source),
            "status": status,
        }
    for name in RESUME_METADATA:
        source = SOURCE_RUN / name
        if not source.is_file():
            continue
        target = SNAPSHOT / name
        if not target.exists():
            shutil.copy2(source, target)
        snapshot_records[name] = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "matches_original": sha256_file(target) == sha256_file(source),
            "status": "already_present_verified" if target.exists() else "copied",
        }

    snapshot_ok = all(
        record["matches_original"] for key, record in snapshot_records.items() if key in EXPECTED
    )

    # ---------------- resume metadata read from the snapshot, never from the live run dir
    resume_info = {}
    try:
        import torch

        payload = torch.load(SNAPSHOT / "weights" / "last.pt", map_location="cpu", weights_only=False)
        train_args = payload.get("train_args", {}) or {}
        resume_info = {
            "epoch_stored": int(payload.get("epoch", -1)),
            "next_epoch": int(payload.get("epoch", -1)) + 2,
            "best_fitness": payload.get("best_fitness"),
            "optimizer_state_present": payload.get("optimizer") is not None,
            "ema_state_present": payload.get("ema") is not None,
            "hasattr_scheduler": "scheduler" in payload or "lr" in train_args,
            "train_args_subset": {
                key: train_args.get(key)
                for key in ("imgsz", "batch", "workers", "seed", "amp", "deterministic", "epochs",
                            "patience", "data", "project", "name", "save_dir", "device")
            },
        }
    except Exception as error:  # noqa: BLE001
        resume_info = {"error": f"{type(error).__name__}: {error}"[:300]}

    results_rows = 0
    results_path = SOURCE_RUN / "results.csv"
    if results_path.is_file():
        results_rows = max(0, len(results_path.read_text(encoding="utf-8").splitlines()) - 1)

    payload = {
        "_doc": (
            "Task 6M.1 section 4. Verification that the Task 6M 18-epoch checkpoint state is exactly "
            "what Task 6M recorded, plus a Task 6M.1-local copy that is never overwritten."
        ),
        "task": "6M.1",
        "source_run_dir": str(SOURCE_RUN),
        "snapshot_dir": str(SNAPSHOT),
        "checkpoints": checks,
        "all_source_hashes_match_recorded": bool(all_match),
        "snapshot": snapshot_records,
        "snapshot_matches_originals": bool(snapshot_ok),
        "recorded_task6m_epochs": results_rows,
        "resume_metadata": resume_info,
        "verdict": "SOURCE_CHECKPOINT_VERIFIED" if (all_match and snapshot_ok) else "SOURCE_CHECKPOINT_MISMATCH",
        "original_files_untouched": True,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(
        f"[6m1.snapshot] best/last hashes match: {all_match}; snapshot matches: {snapshot_ok}; "
        f"stored epoch {resume_info.get('epoch_stored')} -> next {resume_info.get('next_epoch')}; "
        f"optimizer {resume_info.get('optimizer_state_present')}; verdict {payload['verdict']}",
        flush=True,
    )
    return 0 if payload["verdict"] == "SOURCE_CHECKPOINT_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
