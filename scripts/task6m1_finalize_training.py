"""Task 6M.1 Part B: finalize the continuation training summary (combined Task 6M + 6M.1 lineage).

Reads the Task 6M `results.csv` (epochs 1-18) and the Task 6M.1 continuation `results.csv`
(epoch 19 onwards), and writes `evaluation/task6m1_training_summary.json` with every field Task 6M.1
section 6 requires: source checkpoint hashes, start/end epoch, stop reason, best epoch in the combined
lineage, best/final mask mAP50 and mAP50-95, precision/recall, losses, wall time, mean epoch time,
peak VRAM, NaN/Inf, and the final best/last paths with SHA256.

Works whether the continuation finished normally (epoch cap / early stop) or was interrupted by the
session limit; the stop reason is passed on the command line by whoever stops it.

    python scripts/task6m1_finalize_training.py --stop-reason session_wall_clock_limit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import CHECKPOINT_ROOT, EVAL, write_json  # noqa: E402

SOURCE_RUN = CHECKPOINT_ROOT / "runs" / "m1_yolo26m_seg"
RUN_DIR = CHECKPOINT_ROOT.parent / "task6m1" / "runs" / "m1_yolo26m_seg_continued"
PREFLIGHT = EVAL / "task6m1_resume_preflight.json"
TRAIN_LOG = REPO_ROOT / "artifacts" / "task6m1_train.log"
OUT = EVAL / "task6m1_training_summary.json"

EXPECTED_HASHES = {
    "best.pt": "fd407db634a8a7ef83f09f8096686e73407095105f1b45c70c623d18dbf4ea44",
    "last.pt": "ea998bda37dd2dcb2cc7bb5e19f6d15b7a205a137c9cc2866c508a45513f860e",
}
STOP_NORMAL = ("epoch_cap_reached", "early_stopping_fired", "training_completed")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]
    cleaned = []
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", 0)))
        except (TypeError, ValueError):
            continue
        row["_epoch"] = epoch
        cleaned.append(row)
    return cleaned


def metric(row: dict, needle: str) -> float | None:
    for key, value in row.items():
        if needle in key and key.startswith("metrics/"):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def epoch_durations(rows: list[dict]) -> list[float]:
    times = []
    previous = 0.0
    for row in rows:
        try:
            current = float(row.get("time", 0.0))
        except (TypeError, ValueError):
            continue
        times.append(round(current - previous, 1))
        previous = current
    return times


def read_log_text() -> str:
    """Read the training log regardless of the encoding PowerShell redirection used (UTF-8/UTF-16)."""

    if not TRAIN_LOG.is_file():
        return ""
    raw = TRAIN_LOG.read_bytes()
    for encoding in ("utf-8", "utf-16", "utf-16-le", "cp1252"):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if "\x00" not in text and ("/80" in text or "Epoch" in text):
            return text
    return raw.decode("utf-8", errors="ignore").replace("\x00", "")


def peak_vram_gb() -> float | None:
    text = read_log_text()
    if not text:
        return None
    values = [float(match) for match in re.findall(r"(\d+\.\d+)\s*G\b", text)]
    return max(values) if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-reason", default="session_wall_clock_limit")
    parser.add_argument("--note", default=None)
    args = parser.parse_args(argv)

    source_rows = read_results(SOURCE_RUN / "results.csv")
    continued_rows = read_results(RUN_DIR / "results.csv")
    # the continuation copy starts with the Task 6M rows; keep only epochs beyond them
    source_epochs = {row["_epoch"] for row in source_rows}
    new_rows = [row for row in continued_rows if row["_epoch"] not in source_epochs]
    combined = sorted(source_rows + new_rows, key=lambda row: row["_epoch"])

    def summarise(rows: list[dict]) -> dict:
        if not rows:
            return {}
        by_epoch = {row["_epoch"]: row for row in rows}

        def best_of(needle: str) -> tuple[int | None, float | None]:
            best_epoch = None
            best_value = None
            for row in rows:
                value = metric(row, needle)
                if value is not None and (best_value is None or value > best_value):
                    best_value = value
                    best_epoch = row["_epoch"]
            return best_epoch, best_value

        # Ultralytics selects `best.pt` by fitness, which is box-dominated; report both criteria.
        best_box_epoch, best_box_value = best_of("mAP50-95(B)")
        best_mask_epoch, best_mask_value = best_of("mAP50-95(M)")
        final = rows[-1]
        best_row = by_epoch.get(best_box_epoch or final["_epoch"], final)
        mask_best_row = by_epoch.get(best_mask_epoch or final["_epoch"], final)
        return {
            "epochs": len(rows),
            "first_epoch": rows[0]["_epoch"],
            "last_epoch": final["_epoch"],
            "best_epoch": best_box_epoch,
            "best_metric_column": "metrics/mAP50-95(B) (Ultralytics fitness proxy, selects best.pt)",
            "best_mAP50_95": best_box_value,
            "best_epoch_by_mask": best_mask_epoch,
            "best_mask_mAP50": metric(mask_best_row, "mAP50(M)"),
            "best_mask_mAP50_95": best_mask_value,
            "best_box_mAP50": metric(best_row, "mAP50(B)"),
            "best_box_mAP50_95": metric(best_row, "mAP50-95(B)"),
            "best_mask_precision": metric(best_row, "precision(M)"),
            "best_mask_recall": metric(best_row, "recall(M)"),
            "final_mask_mAP50": metric(final, "mAP50(M)"),
            "final_mask_mAP50_95": metric(final, "mAP50-95(M)"),
            "final_box_mAP50": metric(final, "mAP50(B)"),
            "final_box_mAP50_95": metric(final, "mAP50-95(B)"),
            "final_mask_precision": metric(final, "precision(M)"),
            "final_mask_recall": metric(final, "recall(M)"),
            "final_box_precision": metric(final, "precision(B)"),
            "final_box_recall": metric(final, "recall(B)"),
            "final_losses": {
                key: final.get(key) for key in final if key.startswith("train/") or key.startswith("val/")
            },
        }

    def series(rows: list[dict], needle: str) -> dict:
        values = {row["_epoch"]: metric(row, needle) for row in rows if metric(row, needle) is not None}
        return values

    source_hashes = {}
    for name, expected in EXPECTED_HASHES.items():
        path = SOURCE_RUN / "weights" / name
        actual = sha256_file(path) if path.is_file() else None
        source_hashes[name] = {"path": str(path), "expected": expected, "actual": actual,
                               "matches": actual == expected}

    final_best = RUN_DIR / "weights" / "best.pt"
    final_last = RUN_DIR / "weights" / "last.pt"
    durations_new = epoch_durations(new_rows)
    durations_source = epoch_durations(source_rows)

    stop_reason = args.stop_reason
    normal = stop_reason in STOP_NORMAL
    payload = {
        "_doc": (
            "Task 6M.1 section 6. Continuation training summary for the SAME YOLO26m-seg configuration: "
            "Task 6M epochs 1-18 plus the Task 6M.1 continuation from the epoch-18 state, with the "
            "combined lineage, resources and final checkpoint hashes."
        ),
        "task": "6M.1",
        "continuation": {
            "source_run_dir": str(SOURCE_RUN),
            "continuation_run_dir": str(RUN_DIR),
            "source_checkpoint_used": str(RUN_DIR / "weights" / "last_resume.pt"),
            "start_epoch": (new_rows[0]["_epoch"] if new_rows else None),
            "end_epoch": (new_rows[-1]["_epoch"] if new_rows else None),
            "epochs_completed_this_task": len(new_rows),
            "frozen_config": {
                "model": "YOLO26m-seg",
                "imgsz": 640,
                "batch": 16,
                "workers": 4,
                "seed": 20260812,
                "amp": True,
                "deterministic": True,
                "max_epoch_horizon": 80,
                "patience": 15,
                "train": "train1 (scene_disjoint_v1)",
                "val": "train2 (scene_disjoint_v1)",
                "test": "not used in this task's validation phase",
            },
        },
        "source_checkpoint_hashes": source_hashes,
        "source_checkpoint_integrity": all(entry["matches"] for entry in source_hashes.values()),
        "stop": {
            "stop_reason": stop_reason,
            "normal_completion": normal,
            "note": args.note or (
                "continuation reached its configured stop condition"
                if normal else
                "continuation interrupted by the orchestration session limit; the latest completed "
                "state is preserved, convergence is NOT claimed, and downstream graded evaluation is "
                "not run for this outcome"
            ),
        },
        "source_stage": summarise(source_rows),
        "continuation_stage": summarise(new_rows),
        "combined_lineage": summarise(combined),
        "combined_metric_series": {
            "mask_mAP50": series(combined, "mAP50(M)"),
            "mask_mAP50_95": series(combined, "mAP50-95(M)"),
            "box_mAP50": series(combined, "mAP50(B)"),
        },
        "resources": {
            "source_stage_seconds": round(sum(durations_source), 1),
            "source_stage_mean_epoch_seconds": round(
                sum(durations_source) / max(len(durations_source), 1), 1
            ),
            "continuation_seconds": round(sum(durations_new), 1),
            "continuation_hours": round(sum(durations_new) / 3600, 2),
            "continuation_mean_epoch_seconds": round(sum(durations_new) / max(len(durations_new), 1), 1),
            "peak_vram_gb_from_log": peak_vram_gb(),
        },
        "nan_or_inf": any(
            str(value).lower() in ("nan", "inf", "-inf") for row in continued_rows for value in row.values()
        ),
        "final_checkpoints": {
            "best": {
                "path": str(final_best),
                "sha256": sha256_file(final_best) if final_best.is_file() else None,
                "bytes": final_best.stat().st_size if final_best.is_file() else None,
            },
            "last": {
                "path": str(final_last),
                "sha256": sha256_file(final_last) if final_last.is_file() else None,
                "bytes": final_last.stat().st_size if final_last.is_file() else None,
            },
        },
        "task6m_evidence_preserved": {
            "source_hashes_unchanged": all(entry["matches"] for entry in source_hashes.values()),
            "snapshot_dir": str(CHECKPOINT_ROOT.parent / "task6m1" / "source_epoch18_snapshot"),
            "preflight_artifact": str(PREFLIGHT),
        },
        "verdict": "CONTINUATION_COMPLETE" if normal else "CONTINUATION_INTERRUPTED",
    }
    write_json(OUT, payload)
    combined_summary = payload["combined_lineage"]
    print(
        f"[6m1.summary] stop={stop_reason} normal={normal}; epochs {payload['continuation']['start_epoch']}"
        f"-{payload['continuation']['end_epoch']}; combined best epoch {combined_summary.get('best_epoch')} "
        f"(mAP50-95 {combined_summary.get('best_mAP50_95')}); peak VRAM "
        f"{payload['resources']['peak_vram_gb_from_log']} GB -> {payload['verdict']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
