"""Task 6M section 9: finalize the M1 training summary.

Works whether the run finished on its own (early stop / epoch cap) or was stopped by the session's
wall-clock budget: in both cases the summary records the epochs actually completed, the best epoch,
the metrics, the resources and the exact checkpoint hashes.

    python scripts/task6m_training_summary.py [--run m1_yolo26m_seg] [--stop-reason TEXT]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import CHECKPOINT_ROOT, EVAL, EXPORT_ROOT, write_json  # noqa: E402

RUNS = CHECKPOINT_ROOT / "runs"
PRETRAINED = CHECKPOINT_ROOT / "pretrained"
OUT = EVAL / "task6m_training_summary.json"


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
    parser.add_argument("--run", default="m1_yolo26m_seg")
    parser.add_argument("--stop-reason", default=None)
    args = parser.parse_args(argv)

    run_dir = RUNS / args.run
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        print(f"error: no results.csv under {run_dir}", file=sys.stderr)
        return 2

    with results_path.open(encoding="utf-8") as handle:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        print("error: empty results.csv", file=sys.stderr)
        return 2

    metric_keys = [key for key in rows[0] if key.startswith("metrics/")]
    primary = next((key for key in metric_keys if "mAP50-95" in key and "M" in key), None) or next(
        (key for key in metric_keys if "mAP50-95" in key), None
    )
    best_epoch = None
    best_value = None
    for index, row in enumerate(rows, start=1):
        try:
            value = float(row[primary]) if primary else None
        except (TypeError, ValueError):
            value = None
        if value is not None and (best_value is None or value > best_value):
            best_value = value
            best_epoch = index

    def epoch_seconds() -> list[float]:
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

    durations = epoch_seconds()
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    args_yaml = run_dir / "args.yaml"
    config = {}
    if args_yaml.is_file():
        for line in args_yaml.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                if key.strip() in {"epochs", "batch", "imgsz", "workers", "seed", "amp", "deterministic",
                                   "patience", "device", "pretrained", "fraction"}:
                    config[key.strip()] = value.strip()

    weights = PRETRAINED / "yolo26m-seg.pt"
    summary = {
        "_doc": (
            "Task 6M section 9. M1 training of YOLO26m-seg on the derived native-vector export "
            "(scene_disjoint_v1: train=train1, val=train2). Records the epochs actually completed, "
            "the best epoch, metrics, resources and checkpoint hashes. The test split was never used."
        ),
        "task": "6M",
        "run": args.run,
        "run_dir": str(run_dir),
        "config": config,
        "pretrained_weights": {
            "path": str(weights),
            "sha256": sha256_file(weights) if weights.is_file() else None,
            "bytes": weights.stat().st_size if weights.is_file() else None,
        },
        "data": {"export_root": str(EXPORT_ROOT), "data_yaml": str(EXPORT_ROOT / "data.yaml")},
        "results": {
            "epochs_recorded": len(rows),
            "metric_columns": metric_keys,
            "best": {"epoch": best_epoch, "column": primary, "value": best_value},
            "final": rows[-1],
            "nan_or_inf": any(
                str(value).lower() in ("nan", "inf", "-inf") for row in rows for value in row.values()
            ),
            "per_epoch_seconds": durations,
            "total_seconds": round(sum(durations), 1),
            "total_hours": round(sum(durations) / 3600, 2),
        },
        "resources": {
            "reported_train_seconds": round(sum(durations), 1),
            "epochs_completed": len(rows),
            "mean_epoch_seconds": round(sum(durations) / max(len(durations), 1), 1),
        },
        "stop": {
            "requested_epochs": int(config.get("epochs", 80)) if str(config.get("epochs", 80)).isdigit() else 80,
            "patience": int(config.get("patience", 15)) if str(config.get("patience", 15)).isdigit() else 15,
            "stopped_early": len(rows) < (int(config.get("epochs", 80)) if str(config.get("epochs", 80)).isdigit() else 80),
            "stop_reason": args.stop_reason or "training completed or early-stopped within the configured cap",
        },
        "checkpoints": {
            "best": {"path": str(best), "sha256": sha256_file(best), "bytes": best.stat().st_size}
            if best.is_file() else None,
            "last": {"path": str(last), "sha256": sha256_file(last), "bytes": last.stat().st_size}
            if last.is_file() else None,
        },
        "test_split_used": False,
        "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(OUT, summary)
    print(f"[6m.train.summary] epochs {len(rows)}; best epoch {best_epoch} ({primary} {best_value}); "
          f"mean epoch {summary['resources']['mean_epoch_seconds']}s; wrote {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
