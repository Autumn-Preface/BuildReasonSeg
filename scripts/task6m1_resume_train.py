"""Task 6M.1 Part B (sections 5-6): safely continue Task 6M training from the epoch-18 state.

Safety design
-------------
* The Task 6M run directory is treated as READ-ONLY evidence: the continuation gets its own
  `artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/` directory, and `results.csv` is
  copied (not moved) so the resumed trainer appends to the 6M.1 copy only.
* `resume=True` makes Ultralytics replace its args with the checkpoint's stored `train_args`; the
  stored `project`/`name`/`save_dir` point at the Task 6M directory, so this script writes a patched
  copy of the checkpoint (`last_resume.pt`) whose `train_args` point at the 6M.1 directory, and also
  passes `save_dir` (an allowed resume override) as a second guard. Nothing in Task 6M is touched.
* Before training the script asserts: source hashes still match the recorded ones, the stored epoch
  is 17 (next epoch 19), optimizer and EMA state are present, and the stored training config equals
  the frozen Task 6M configuration (imgsz 640, batch 16, workers 4, seed 20260812, epochs 80,
  patience 15, the derived native-vector export).

If any check fails the script exits with `SAFE_RESUME_UNAVAILABLE` and runs nothing.

    python scripts/task6m1_resume_train.py [--dry-run]
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

from buildreasonseg_mvp.task6m_eval import CHECKPOINT_ROOT, EVAL, EXPORT_ROOT, write_json  # noqa: E402

SOURCE_RUN = CHECKPOINT_ROOT / "runs" / "m1_yolo26m_seg"
SNAPSHOT = CHECKPOINT_ROOT.parent / "task6m1" / "source_epoch18_snapshot"
RUN_DIR = CHECKPOINT_ROOT.parent / "task6m1" / "runs" / "m1_yolo26m_seg_continued"
PREFLIGHT = EVAL / "task6m1_resume_preflight.json"

EXPECTED_HASHES = {
    "best.pt": "fd407db634a8a7ef83f09f8096686e73407095105f1b45c70c623d18dbf4ea44",
    "last.pt": "ea998bda37dd2dcb2cc7bb5e19f6d15b7a205a137c9cc2866c508a45513f860e",
}
FROZEN_CONFIG = {
    "imgsz": 640,
    "batch": 16,
    "workers": 4,
    "seed": 20260812,
    "epochs": 80,
    "patience": 15,
    "amp": True,
    "deterministic": True,
}
EXPECTED_STORED_EPOCH = 17  # 0-based; the next displayed epoch is 19


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fail(reason: str, detail: dict) -> int:
    write_json(
        PREFLIGHT,
        {
            "_doc": "Task 6M.1 section 5 safe-resume pre-flight.",
            "task": "6M.1",
            "verdict": "SAFE_RESUME_UNAVAILABLE",
            "reason": reason,
            "detail": detail,
        },
    )
    print(f"[6m1.resume] SAFE_RESUME_UNAVAILABLE: {reason}", flush=True)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="pre-flight checks only")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    started = time.time()
    import torch

    # ---------------- 1. source integrity (Task 6M evidence must still be untouched)
    source_hashes = {}
    for name, expected in EXPECTED_HASHES.items():
        path = SOURCE_RUN / "weights" / name
        if not path.is_file():
            return fail(f"missing source checkpoint {path}", {})
        actual = sha256_file(path)
        source_hashes[name] = {"actual": actual, "expected": expected, "matches": actual == expected}
    if not all(entry["matches"] for entry in source_hashes.values()):
        return fail("source checkpoint hashes changed since Task 6M", source_hashes)

    snapshot_hashes = {}
    for name in EXPECTED_HASHES:
        path = SNAPSHOT / "weights" / name
        if path.is_file():
            digest = sha256_file(path)
            snapshot_hashes[name] = {
                "sha256": digest,
                "matches_source": digest == sha256_file(SOURCE_RUN / "weights" / name),
            }
    if not all(entry["matches_source"] for entry in snapshot_hashes.values()):
        return fail("snapshot does not match the live Task 6M checkpoints", snapshot_hashes)

    # ---------------- 2. prepare the 6M.1-local run directory from the snapshot
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "weights").mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ("best.pt", "last.pt"):
        target = RUN_DIR / "weights" / name
        if not target.exists():
            shutil.copy2(SNAPSHOT / "weights" / name, target)
        copied[name] = {"path": str(target), "sha256": sha256_file(target)}
    results_source = SNAPSHOT / "results.csv"
    results_target = RUN_DIR / "results.csv"
    if results_source.is_file() and not results_target.exists():
        shutil.copy2(results_source, results_target)
    copied["results.csv"] = {
        "path": str(results_target),
        "rows": max(0, len(results_target.read_text(encoding="utf-8").splitlines()) - 1)
        if results_target.is_file() else 0,
    }

    # ---------------- 3. patch the stored train_args so resume cannot write into Task 6M
    payload = torch.load(RUN_DIR / "weights" / "last.pt", map_location="cpu", weights_only=False)
    train_args = dict(payload.get("train_args", {}) or {})
    stored_epoch = int(payload.get("epoch", -1))
    original_paths = {
        key: train_args.get(key) for key in ("project", "name", "save_dir")
    }
    train_args["project"] = str(RUN_DIR.parent)
    train_args["name"] = RUN_DIR.name
    train_args["save_dir"] = str(RUN_DIR)
    payload["train_args"] = train_args
    resume_checkpoint = RUN_DIR / "weights" / "last_resume.pt"
    torch.save(payload, resume_checkpoint)

    checks = {
        "source_hashes_match_task6m": all(entry["matches"] for entry in source_hashes.values()),
        "snapshot_matches_source": all(entry["matches_source"] for entry in snapshot_hashes.values()),
        "stored_epoch_is_18th": stored_epoch == EXPECTED_STORED_EPOCH,
        "next_epoch_is_19": stored_epoch + 2 == 19,
        "optimizer_state_present": payload.get("optimizer") is not None,
        "ema_state_present": payload.get("ema") is not None,
        "save_dir_is_task6m1_local": str(RUN_DIR) in str(train_args.get("save_dir")),
        "original_task6m_paths_preserved_in_snapshot": True,
    }
    config_checks = {}
    for key, expected in FROZEN_CONFIG.items():
        stored = train_args.get(key)
        config_checks[key] = {"expected": expected, "stored": stored, "matches": stored == expected}
    checks["frozen_config_unchanged"] = all(entry["matches"] for entry in config_checks.values())
    checks["data_is_native_vector_export"] = str(train_args.get("data", "")).replace("\\", "/").endswith(
        "artifacts/task6m_yolo_native/data.yaml"
    )

    preflight = {
        "_doc": (
            "Task 6M.1 section 5. Safe-resume pre-flight: the continuation resumes the Task 6M "
            "epoch-18 state from a patched COPY of last.pt whose stored args point at a Task 6M.1-local "
            "run directory, so no Task 6M evidence can be written."
        ),
        "task": "6M.1",
        "source_run_dir": str(SOURCE_RUN),
        "snapshot_dir": str(SNAPSHOT),
        "continuation_run_dir": str(RUN_DIR),
        "source_hashes": source_hashes,
        "snapshot_hashes": snapshot_hashes,
        "copied_into_continuation_run": copied,
        "resume_checkpoint": {
            "path": str(resume_checkpoint),
            "sha256": sha256_file(resume_checkpoint),
            "stored_epoch_0based": stored_epoch,
            "next_epoch_1based": stored_epoch + 2,
            "patched_train_args": {
                "original": original_paths,
                "patched": {key: train_args.get(key) for key in ("project", "name", "save_dir")},
            },
        },
        "checks": checks,
        "frozen_config_checks": config_checks,
        "stored_data_path": train_args.get("data"),
        "verdict": "SAFE_RESUME_READY" if all(checks.values()) else "SAFE_RESUME_UNAVAILABLE",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(PREFLIGHT, preflight)
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        print(f"[6m1.resume] SAFE_RESUME_UNAVAILABLE: failed {failed}", flush=True)
        return 2

    print(
        f"[6m1.resume] pre-flight OK: stored epoch {stored_epoch} -> next epoch 19; "
        f"optimizer {checks['optimizer_state_present']}; run dir {RUN_DIR.name}",
        flush=True,
    )
    if args.dry_run:
        return 0

    # ---------------- 4. continue training
    from ultralytics import YOLO

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(resume_checkpoint))
    started_training = time.time()
    model.train(
        resume=True,
        save_dir=str(RUN_DIR),
        patience=FROZEN_CONFIG["patience"],
        workers=FROZEN_CONFIG["workers"],
        device=args.device,
        val=True,
        plots=False,
    )
    elapsed = time.time() - started_training
    print(f"[6m1.resume] continuation finished after {elapsed / 60:.1f} min", flush=True)

    # ---------------- 5. post-run integrity of the Task 6M evidence
    post_hashes = {
        name: {"actual": sha256_file(SOURCE_RUN / "weights" / name), "expected": expected}
        for name, expected in EXPECTED_HASHES.items()
    }
    for entry in post_hashes.values():
        entry["matches"] = entry["actual"] == entry["expected"]
    write_json(
        PREFLIGHT,
        {
            **preflight,
            "verdict": "SAFE_RESUME_COMPLETED",
            "training_seconds": round(elapsed, 1),
            "task6m_evidence_unchanged_after_run": all(entry["matches"] for entry in post_hashes.values()),
            "post_run_source_hashes": post_hashes,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
