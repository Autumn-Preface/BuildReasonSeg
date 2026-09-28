"""Task 6M section 9: M1 full training of the native-vector proposal model (YOLO26m-seg).

Trains on the derived export of `WHU-EA-NativeVector v1.0` under `scene_disjoint_v1`
(train = train1, val = train2). The test split is never touched here.

Config: COCO-pretrained checkpoint, imgsz 640, max 80 epochs, patience 15, fixed seed, AMP when
stable, conservative Windows-safe workers, best+last checkpoints, deterministic settings.

A short memory probe runs first and the largest stable batch is chosen; on OOM the batch is reduced
(never imgsz, and any reduction is reported).

    python scripts/task6m_train.py [--epochs 80] [--batch auto] [--probe-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import CHECKPOINT_ROOT, EVAL, EXPORT_ROOT, write_json  # noqa: E402

PRETRAINED = CHECKPOINT_ROOT / "pretrained"
RUNS = CHECKPOINT_ROOT / "runs"
OUT = EVAL / "task6m_training_summary.json"
SEED = 20260812
DATA_YAML = EXPORT_ROOT / "data.yaml"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class VramSampler(threading.Thread):
    """Samples CUDA memory usage while training runs (max VRAM is the recorded metric)."""

    def __init__(self, interval: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.peak_allocated = 0
        self.peak_reserved = 0

    def run(self) -> None:
        import torch

        while not self.stop_flag.is_set():
            try:
                self.peak_allocated = max(self.peak_allocated, torch.cuda.memory_allocated())
                self.peak_reserved = max(self.peak_reserved, torch.cuda.memory_reserved())
            except Exception:  # noqa: BLE001
                pass
            self.stop_flag.wait(self.interval)


def read_results_csv(path: Path) -> dict:
    """Summarise Ultralytics' results.csv (losses and metrics per epoch)."""

    if not path.is_file():
        return {}
    import csv

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    cleaned = [{key.strip(): value for key, value in row.items()} for row in rows]
    metric_keys = [key for key in cleaned[0] if key.startswith("metrics/")]
    loss_keys = [key for key in cleaned[0] if key.startswith("train/") or key.startswith("val/")]
    best_epoch = None
    best_value = None
    primary = next((key for key in metric_keys if "mAP50-95" in key and key.startswith("metrics/mAP")), None)
    if primary:
        for index, row in enumerate(cleaned, start=1):
            try:
                value = float(row[primary])
            except (TypeError, ValueError):
                continue
            if best_value is None or value > best_value:
                best_value = value
                best_epoch = index
    return {
        "epochs_recorded": len(cleaned),
        "metric_columns": metric_keys,
        "loss_columns": loss_keys,
        "final": cleaned[-1],
        "best": {"epoch": best_epoch, "column": primary, "value": best_value},
        "nan_or_inf": any(
            str(value).lower() in ("nan", "inf", "-inf") for row in cleaned for value in row.values()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26m-seg.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="auto", help="'auto' probes, or an integer")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--name", default="m1_yolo26m_seg")
    args = parser.parse_args(argv)

    started = time.time()
    import torch
    import ultralytics
    from ultralytics import YOLO

    weights = PRETRAINED / args.model
    if not weights.is_file():
        print(f"error: missing pretrained weights {weights}", file=sys.stderr)
        return 2
    if not DATA_YAML.is_file():
        print(f"error: missing export data.yaml {DATA_YAML}", file=sys.stderr)
        return 2

    RUNS.mkdir(parents=True, exist_ok=True)
    batch = args.batch
    probe_records = []
    if str(batch) == "auto" or args.probe_only:
        # short memory probe: try decreasing batch sizes until one completes an epoch
        for candidate in (16, 12, 8, 6, 4, 2):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            probe_name = f"probe_b{candidate}"
            try:
                model = YOLO(str(weights))
                model.train(
                    data=str(DATA_YAML), epochs=1, imgsz=args.imgsz, batch=candidate,
                    workers=0, device=args.device, project=str(RUNS), name=probe_name,
                    exist_ok=True, verbose=False, plots=False, val=False, save=False, seed=SEED,
                    fraction=0.02,
                )
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                probe_records.append({"batch": candidate, "ok": True, "peak_vram_gb": round(peak, 2)})
                print(f"[6m.train] memory probe batch {candidate}: OK, peak {peak:.2f} GB", flush=True)
                if batch == "auto":
                    batch = candidate
                    break
            except RuntimeError as error:
                message = str(error)[:200]
                probe_records.append({"batch": candidate, "ok": False, "error": message})
                print(f"[6m.train] memory probe batch {candidate}: OOM ({message[:80]})", flush=True)
                if args.probe_only:
                    continue
        if batch == "auto" and not any(record["ok"] for record in probe_records):
            print("error: no batch size completed the memory probe", file=sys.stderr)
            return 2
    if args.probe_only:
        write_json(
            OUT,
            {
                "_doc": "Task 6M section 9 memory probe only (no full training).",
                "task": "6M",
                "mode": "probe_only",
                "probes": probe_records,
                "chosen_batch": batch,
                "imgsz": args.imgsz,
            },
        )
        print(f"[6m.train] probe-only complete; chosen batch {batch}", flush=True)
        return 0

    sampler = VramSampler()
    sampler.start()
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(weights))
    results = model.train(
        data=str(DATA_YAML),
        epochs=int(args.epochs),
        patience=int(args.patience),
        imgsz=int(args.imgsz),
        batch=int(batch),
        workers=int(args.workers),
        device=args.device,
        project=str(RUNS),
        name=args.name,
        exist_ok=True,
        seed=SEED,
        deterministic=True,
        amp=bool(args.amp),
        plots=False,
        val=True,
        save=True,
        verbose=True,
        pretrained=True,
        cos_lr=True,
        close_mosaic=10,
    )
    sampler.stop_flag.set()
    sampler.join(timeout=5)

    save_dir = Path(getattr(results, "save_dir", RUNS / args.name))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    csv_summary = read_results_csv(save_dir / "results.csv")
    peak_allocated = max(sampler.peak_allocated, torch.cuda.max_memory_allocated()) / (1024 ** 3)
    peak_reserved = max(sampler.peak_reserved, torch.cuda.max_memory_reserved()) / (1024 ** 3)

    summary = {
        "_doc": (
            "Task 6M section 9. M1 full training of the native-vector proposal model on "
            "scene_disjoint_v1 (train=train1, val=train2). The test split was never used."
        ),
        "task": "6M",
        "model": args.model,
        "pretrained_weights": {
            "path": str(weights),
            "sha256": sha256_file(weights),
            "bytes": weights.stat().st_size,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
        },
        "config": {
            "data": str(DATA_YAML),
            "epochs_max": int(args.epochs),
            "patience": int(args.patience),
            "imgsz": int(args.imgsz),
            "batch": int(batch),
            "workers": int(args.workers),
            "amp": bool(args.amp),
            "seed": SEED,
            "deterministic": True,
            "device": args.device,
        },
        "memory_probe": probe_records,
        "resources": {
            "wall_time_seconds": round(time.time() - started, 1),
            "wall_time_hours": round((time.time() - started) / 3600, 2),
            "peak_vram_allocated_gb": round(peak_allocated, 2),
            "peak_vram_reserved_gb": round(peak_reserved, 2),
        },
        "results": csv_summary,
        "checkpoints": {
            "best": {"path": str(best), "sha256": sha256_file(best), "bytes": best.stat().st_size}
            if best.is_file() else None,
            "last": {"path": str(last), "sha256": sha256_file(last), "bytes": last.stat().st_size}
            if last.is_file() else None,
            "save_dir": str(save_dir),
        },
        "test_split_used": False,
    }
    write_json(OUT, summary)
    print(
        f"[6m.train] done: {csv_summary.get('epochs_recorded')} epochs; best "
        f"{csv_summary.get('best')}; peak VRAM {peak_allocated:.2f} GB; "
        f"wall {(time.time() - started) / 60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
