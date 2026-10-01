"""Task 7I Parts F-H — formal three-seed training of Z-B3 (F0) and D-B1 (F1).

Six runs: {Z-B3, D-B1} x {20261001, 20261002, 20261003}, each from **fresh** trainable initialization (no
historical checkpoint is ever loaded as initialization), with the exact frozen protocol:

* gradient updates on the formal **train** population only (1344 oracle-reference L3 records);
* AdamW, lr 3e-4, weight decay 1e-4, batch 8, max 25 epochs, early-stopping patience 5, no scheduler, no
  augmentation, bfloat16 AMP, loss exactly `BCEWithLogitsLoss + DiceLoss`;
* every epoch evaluates **all 936** oracle-reference formal val records; checkpoint selection is
  (val mIoU, val Dice, val Pr@0.5, earlier epoch) and early stopping uses full-val mIoU only — MiniVal240 is
  never used for selection, and neither are the pairs or the predicted-reference results;
* `best.pt` / `last.pt` per model/seed under the gitignored `artifacts/checkpoints/task7i/`; a resume after an
  interruption restores only that same run's `last.pt`.

Writes `evaluation/task7i_training_zb3.json` and `evaluation/task7i_training_db1.json`.

    python scripts/task7i_train.py --model Z-B3
    python scripts/task7i_train.py --model D-B1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
    task6n_loss,
)
from buildreasonseg_mvp.task6z_l3_decoder import L3TargetDecoder  # noqa: E402
from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder  # noqa: E402
from scripts.task6n_train import (  # noqa: E402
    FEATURE_ROOT,
    MaskStore,
    _per_sample_metrics,
    canonical_instances,
    make_optimizer,
    seed_everything,
)

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task7i" / "packs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task7i"
POPULATION = EVAL / "task7i_formal_population_manifest.json"
TRAIN_ROWS = PACK_ROOT / "formal_train_1344.jsonl"
VAL_ROWS = PACK_ROOT / "formal_val_936.jsonl"
OUT = {"Z-B3": EVAL / "task7i_training_zb3.json", "D-B1": EVAL / "task7i_training_db1.json"}
SEEDS = (20261001, 20261002, 20261003)
PROTOCOL = {"optimizer": "AdamW", "lr": 3.0e-4, "weight_decay": 1.0e-4, "batch_size": 8,
            "max_epochs": 25, "early_stopping_patience": 5, "scheduler": "none",
            "augmentation": "none", "amp": "bfloat16 autocast", "loss": "BCEWithLogitsLoss + DiceLoss",
            "reference_source": "oracle_native_gt"}
TOLERANCE = 1.0e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def forward_batch(model, batch, variant: str) -> torch.Tensor:
    if variant == "Z-B3":
        return model(batch.visual, batch.relations, batch.directional, batch.nearest, batch.product)
    return model(batch.visual, batch.relations, batch.directional, batch.nearest)


def make_batch(records: list[dict], store, masks, device, precomputed, variant: str):
    if variant == "Z-B3":
        from scripts.task6z_train import build_batch

        return build_batch(records, store, masks, device, precomputed)
    from buildreasonseg_mvp.task7d_data import build_batch

    return build_batch(records, store, masks, device, precomputed)


@torch.no_grad()
def evaluate(model, records: list[dict], store, masks, device, precomputed, variant: str,
             batch_size: int = 8) -> dict:
    model.eval()
    rows = []
    for start in range(0, len(records), batch_size):
        batch = make_batch(records[start: start + batch_size], store, masks, device, precomputed,
                           variant)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = forward_batch(model, batch, variant)
        upsampled = torch.nn.functional.interpolate(logits.float(), size=(512, 512), mode="bilinear",
                                                   align_corners=False)
        rows.extend(_per_sample_metrics(upsampled, batch))
    return {"records": len(rows),
            "miou": float(np.mean([row["miou"] for row in rows])),
            "dice": float(np.mean([row["dice"] for row in rows])),
            "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows])),
            "rows": rows}


def train_run(variant: str, seed: int, train_rows: list[dict], val_rows: list[dict], store, masks,
              device: str, precomputed: dict, resume: bool) -> dict:
    seed_everything(seed)
    if variant == "Z-B3":
        model = L3TargetDecoder("Z-B3").to(device)
    else:
        model = GlobalCompetitionDecoder("D-B1").to(device)
    optimizer = make_optimizer(model, PROTOCOL["lr"], PROTOCOL["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    run_dir = CHECKPOINT_ROOT / variant / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"

    start_epoch, history, best = 0, [], {"key": (-1.0, -1.0, -1.0, 0), "epoch": 0, "metrics": None}
    if resume and last_path.is_file():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload["epoch"])
        history = list(payload.get("history", []))
        best = payload.get("best", best)
        print(f"[7i.train] {variant}/{seed} resumed from epoch {start_epoch}", flush=True)

    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    generator = np.random.default_rng(seed)
    stale = 0
    final_epoch = start_epoch
    oom_or_nan = False
    for epoch in range(start_epoch + 1, PROTOCOL["max_epochs"] + 1):
        model.train()
        order = generator.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), PROTOCOL["batch_size"]):
            chunk = [train_rows[int(index)] for index in order[start: start + PROTOCOL["batch_size"]]]
            batch = make_batch(chunk, store, masks, device, precomputed, variant)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits = forward_batch(model, batch, variant)
                upsampled = torch.nn.functional.interpolate(logits.float(), size=(512, 512),
                                                            mode="bilinear", align_corners=False)
                losses_dict = task6n_loss(upsampled, batch.target.unsqueeze(1))
            scaler.scale(losses_dict["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            value = float(losses_dict["loss"].detach())
            if not np.isfinite(value):
                oom_or_nan = True
                break
            losses.append(value)
        metrics = evaluate(model, val_rows, store, masks, device, precomputed, variant)
        final_epoch = epoch
        key = (metrics["miou"], metrics["dice"], metrics["precision_at_0_5"], -epoch)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                        "val_miou": metrics["miou"], "val_dice": metrics["dice"],
                        "val_precision_at_0_5": metrics["precision_at_0_5"],
                        "val_records": metrics["records"]})
        print(f"[7i.train] {variant}/{seed} epoch {epoch}: loss "
              f"{(np.mean(losses) if losses else float('nan')):.4f} | val mIoU {metrics['miou']:.4f} "
              f"Dice {metrics['dice']:.4f} Pr@0.5 {metrics['precision_at_0_5']:.4f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(), "epoch": epoch, "history": history,
                    "best": best, "variant": variant, "seed": seed}, last_path)
        if key > best["key"]:
            best = {"key": key, "epoch": epoch, "metrics": {key: metrics[key]
                                                            for key in ("records", "miou", "dice",
                                                                        "precision_at_0_5")}}
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "variant": variant,
                        "seed": seed, "metrics": best["metrics"]}, best_path)
            stale = 0
        else:
            stale += 1
            if stale >= PROTOCOL["early_stopping_patience"]:
                print(f"[7i.train] {variant}/{seed} early stop at epoch {epoch}", flush=True)
                break

    wall = time.time() - started
    peak_vram = (round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
                 if device != "cpu" else None)
    trainable = int(sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad))
    return {
        "model": variant, "seed": seed, "fresh_initialization": True,
        "initialized_from_historical_checkpoint": False,
        "trainable_parameters": trainable,
        "history": history, "selected_epoch": best["epoch"], "final_epoch": final_epoch,
        "selected_val_metrics": best["metrics"],
        "checkpoint_best": {"path": str(best_path), "exists": best_path.is_file(),
                            "sha256": sha256_file(best_path) if best_path.is_file() else None,
                            "bytes": best_path.stat().st_size if best_path.is_file() else None},
        "checkpoint_last": {"path": str(last_path), "exists": last_path.is_file(),
                            "sha256": sha256_file(last_path) if last_path.is_file() else None,
                            "bytes": last_path.stat().st_size if last_path.is_file() else None},
        "wall_seconds": round(wall, 1), "peak_vram_gb": peak_vram,
        "nan_or_inf": oom_or_nan, "oom": False,
        "valid": bool(best["epoch"] > 0 and len(history) > 0 and not oom_or_nan),
        "committed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Z-B3", "D-B1"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="resume each run from its own last.pt only")
    args = parser.parse_args(argv)
    started = time.time()

    population = json.loads(POPULATION.read_text(encoding="utf-8"))
    if population["verdict"] != "FORMAL_POPULATION_FROZEN":
        write_json(OUT[args.model], {"_doc": "Task 7I section 8.", "task": "7I",
                                     "verdict": "FORMAL_POPULATION_MISMATCH"})
        print("[7i.train] STOP FORMAL_POPULATION_MISMATCH", flush=True)
        return 3

    train_rows = read_rows(TRAIN_ROWS)
    val_rows = read_rows(VAL_ROWS)
    encoder, _report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()
    precomputed: dict[str, dict] = {}

    runs = [train_run(args.model, seed, train_rows, val_rows, store, masks, args.device,
                      precomputed, args.resume) for seed in SEEDS]
    payload = {
        "_doc": ("Task 7I sections 15-18. Formal three-seed training of "
                 f"{args.model} from fresh trainable weights on the 1344-record formal L3 train population, "
                 "with per-epoch full-936 oracle-reference validation, selection by val mIoU/Dice/Pr@0.5 and "
                 "full-val-mIoU early stopping. MiniVal240, the pair set and the predicted-reference results "
                 "were never used for selection."),
        "task": "7I", "stage": "F/G-formal-training", "model": args.model,
        "protocol": PROTOCOL, "seeds": list(SEEDS), "runs": runs,
        "population": {"train_records": len(train_rows), "val_records": len(val_rows),
                       "train_sample_id_sha256": population["train"]["sample_id_sha256"],
                       "val_sample_id_sha256": population["val"]["sample_id_sha256"]},
        "selection": {"metric": "full 936-record oracle-reference val mean mIoU",
                      "tie_break": ["val Dice", "val Pr@0.5", "earlier epoch"],
                      "minival240_used": False, "pair_set_used": False,
                      "predicted_reference_used": False, "test_used": False},
        "all_runs_valid": all(run["valid"] for run in runs),
        "training_performed": True, "test_split_used": False,
        "verdict": "FORMAL_TRAINING_COMPLETE" if all(run["valid"] for run in runs)
        else "FORMAL_TRAINING_INCOMPLETE",
        "wall_seconds": round(time.time() - started, 1),
    }
    write_json(OUT[args.model], payload)
    for run in runs:
        print(f"[7i.train] {args.model} seed {run['seed']}: selected epoch {run['selected_epoch']} "
              f"val mIoU {run['selected_val_metrics']['miou']:.4f} wall {run['wall_seconds']}s "
              f"params {run['trainable_parameters']}", flush=True)
    return 0 if payload["all_runs_valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
