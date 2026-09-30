"""Task 6W Part F — train ProposalQualityEstimator v0.1.

Protocol (sections 14-15): tile-disjoint 80/20 train/internal-holdout split (seed 20260930), AdamW with
lr 5e-4, weight decay 1e-4, batch 256 proposals, max 40 epochs, early-stopping patience 5, seed 20260930,
AMP allowed, no augmentation, no scheduler, no hyperparameter sweep. Loss is
`BCEWithLogitsLoss(pos_weight = N_negative / max(1, N_positive))` with `pos_weight` computed from the
**training split only**.

Checkpoint selection: highest internal-holdout AUROC, tie-break highest F1 at threshold 0.50, tie-break
lower BCE. Checkpoint: `artifacts/checkpoints/task6w/proposal_quality_v01.pt` (gitignored).
Writes `evaluation/task6w_quality_training.json`.

    python scripts/task6w_train_quality.py
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
from buildreasonseg_mvp.task6w_proposal_quality import (  # noqa: E402
    QUALITY_THRESHOLD,
    ProposalQualityEstimator,
    auprc,
    auroc,
    binary_metrics,
    pos_weight_from_labels,
    quality_probabilities,
)

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6w_quality_training.json"
MANIFEST = EVAL / "task6w_quality_dataset_manifest.json"
DATASET = REPO_ROOT / "artifacts" / "task6w" / "quality_dataset" / "quality_rows.npz"
OUT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6w" / "proposal_quality_v01.pt"
SEED = 20260930
TRAIN_TILE_HASH_SEED = 20260930
HOLDOUT_FRACTION = 0.20
BATCH = 256
MAX_EPOCHS = 40
PATIENCE = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tile_hash(tile_id: str) -> str:
    return hashlib.sha256(f"{TRAIN_TILE_HASH_SEED}:{tile_id}".encode("utf-8")).hexdigest()


def evaluate_split(model, visual, geometry, labels, device) -> dict:
    probabilities = quality_probabilities(model, visual, geometry, device=device)
    metrics = binary_metrics(labels, probabilities, QUALITY_THRESHOLD)
    return {
        "proposals": int(len(labels)),
        "positives": int((np.asarray(labels) >= 0.5).sum()),
        "negatives": int((np.asarray(labels) < 0.5).sum()),
        "auroc": auroc(labels, probabilities),
        "auprc": auprc(labels, probabilities),
        **metrics,
        "probabilities_mean": float(np.mean(probabilities)) if len(probabilities) else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    if not DATASET.is_file():
        write_json(OUT, {"_doc": "Task 6W sections 14-15.", "task": "6W",
                         "verdict": "QUALITY_ESTIMATOR_NOT_LEARNABLE",
                         "reason": "quality dataset missing"})
        return 2
    payload = np.load(DATASET, allow_pickle=False)
    geometry = payload["geometry"].astype(np.float32)
    visual = payload["visual"].astype(np.float32)
    labels = payload["labels"].astype(np.float32)
    tiles = [str(tile) for tile in payload["tiles"].tolist()]
    unique_tiles = sorted(set(tiles))
    holdout_tiles = {tile_id for tile_id in unique_tiles
                     if int(tile_hash(tile_id)[:8], 16) / 0xFFFFFFFF < HOLDOUT_FRACTION}
    train_mask = np.asarray([tile not in holdout_tiles for tile in tiles])
    holdout_mask = ~train_mask
    train_pos_weight = pos_weight_from_labels(labels[train_mask])

    torch.manual_seed(SEED)
    model = ProposalQualityEstimator().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_pos_weight], dtype=torch.float32, device=args.device))
    use_amp = str(args.device) != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_visual = torch.as_tensor(visual[train_mask], device=args.device)
    train_geometry = torch.as_tensor(geometry[train_mask], device=args.device)
    train_labels = torch.as_tensor(labels[train_mask], device=args.device)
    holdout_visual_np, holdout_geometry_np = visual[holdout_mask], geometry[holdout_mask]
    holdout_labels = labels[holdout_mask]

    if str(args.device) != "cpu":
        torch.cuda.reset_peak_memory_stats()
    best = {"key": (-1.0, -1.0, float("inf")), "epoch": 0, "metrics": None}
    history = []
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        generator = np.random.default_rng(SEED + epoch)
        order = generator.permutation(len(train_labels))
        losses = []
        for start in range(0, len(order), BATCH):
            index = torch.as_tensor(order[start: start + BATCH].astype(np.int64),
                                    device=args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(train_visual[index], train_geometry[index])
                loss = loss_function(logits, train_labels[index])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        holdout = evaluate_split(model, holdout_visual_np, holdout_geometry_np, holdout_labels,
                                 args.device)
        key = (holdout["auroc"] if np.isfinite(holdout["auroc"]) else 0.0,
               holdout["f1"], -float(np.mean(losses)))
        history.append({"epoch": epoch, "train_bce": float(np.mean(losses)),
                        "holdout_auroc": holdout["auroc"], "holdout_auprc": holdout["auprc"],
                        "holdout_f1": holdout["f1"], "holdout_accuracy": holdout["accuracy"]})
        print(f"[6w.train] epoch {epoch}: train BCE {np.mean(losses):.4f} holdout AUROC "
              f"{holdout['auroc']:.4f} AUPRC {holdout['auprc']:.4f} F1 {holdout['f1']:.4f}", flush=True)
        if key > best["key"]:
            best = {"key": key, "epoch": epoch, "metrics": holdout}
            OUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "architecture": model.architecture_report(),
                        "pos_weight": train_pos_weight, "seed": SEED,
                        "quality_threshold": QUALITY_THRESHOLD}, OUT_CHECKPOINT)
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[6w.train] early stop at epoch {epoch}", flush=True)
                break

    if OUT_CHECKPOINT.is_file():
        model.load_state_dict(torch.load(OUT_CHECKPOINT, map_location=args.device,
                                         weights_only=False)["state_dict"])
    train_final = evaluate_split(model, visual[train_mask], geometry[train_mask], labels[train_mask],
                                args.device)
    holdout_final = evaluate_split(model, holdout_visual_np, holdout_geometry_np, holdout_labels,
                                   args.device)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}

    result = {
        "_doc": (
            "Task 6W sections 14-15. ProposalQualityEstimator v0.1 training: tile-disjoint 80/20 "
            "train/internal-holdout split, BCEWithLogits with train-only pos_weight, checkpoint selected "
            "by internal-holdout AUROC (tie-break F1@0.50, then lower BCE). Threshold 0.50 is frozen; no "
            "sweep, no augmentation, no scheduler."
        ),
        "task": "6W", "stage": "F-quality-training",
        "data": {
            "dataset": str(DATASET),
            "manifest": str(MANIFEST),
            "tiles": {"unique": len(unique_tiles), "train": len(unique_tiles) - len(holdout_tiles),
                      "holdout": len(holdout_tiles),
                      "overlap": len({tile for tile in tiles if tile not in holdout_tiles}
                                     & holdout_tiles)},
            "proposals": {"total": int(len(labels)), "train": int(train_mask.sum()),
                          "holdout": int(holdout_mask.sum())},
            "class_counts": {
                "train": {"positives": int((labels[train_mask] >= 0.5).sum()),
                          "negatives": int((labels[train_mask] < 0.5).sum())},
                "holdout": {"positives": int((labels[holdout_mask] >= 0.5).sum()),
                            "negatives": int((labels[holdout_mask] < 0.5).sum())},
            },
            "pos_weight_train_only": train_pos_weight,
            "split_by": "tile id hash, seed 20260930, 80/20",
        },
        "architecture": model.architecture_report(),
        "optimizer": {"name": "AdamW", "lr": 5.0e-4, "weight_decay": 1.0e-4, "batch_proposals": BATCH,
                      "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE, "seed": SEED,
                      "amp": use_amp, "augmentation": False, "scheduler": "none", "sweep": False,
                      "loss": "BCEWithLogitsLoss(pos_weight = N_negative / max(1, N_positive))",
                      "selection": ["internal-holdout AUROC", "tie-break F1@0.50",
                                    "tie-break lower BCE"]},
        "history": history,
        "selected_epoch": best["epoch"],
        "internal_holdout": holdout_final,
        "internal_train_final": train_final,
        "gates": {
            "auroc_min": 0.80, "f1_min": 0.65,
            "measured_auroc": holdout_final["auroc"], "measured_f1": holdout_final["f1"],
            "no_tile_overlap": len({tile for tile in tiles if tile not in holdout_tiles}
                                   & holdout_tiles) == 0,
        },
        "checkpoint": {
            "path": str(OUT_CHECKPOINT), "exists": OUT_CHECKPOINT.is_file(),
            "sha256": sha256_file(OUT_CHECKPOINT) if OUT_CHECKPOINT.is_file() else None,
            "committed": False,
        },
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if str(args.device) != "cpu" else None,
        "wall_seconds": round(time.time() - started, 1),
        "training_performed": True,
        "refval_used_for_training": False,
        "calib_used_for_training": False,
        "test_split_used": False,
    }
    result["gates"]["passed"] = bool(
        (holdout_final["auroc"] or 0.0) >= result["gates"]["auroc_min"]
        and (holdout_final["f1"] or 0.0) >= result["gates"]["f1_min"]
        and result["gates"]["no_tile_overlap"])
    if manifest:
        result["dataset_manifest_hash"] = manifest.get("features", {}).get("visual_dim")
    result["verdict"] = ("QUALITY_ESTIMATOR_ADEQUATE" if result["gates"]["passed"]
                         else "QUALITY_ESTIMATOR_NOT_LEARNABLE")
    write_json(OUT, result)
    print(f"[6w.train] selected epoch {best['epoch']}: holdout AUROC {holdout_final['auroc']:.4f} "
          f"AUPRC {holdout_final['auprc']:.4f} F1 {holdout_final['f1']:.4f} acc "
          f"{holdout_final['accuracy']:.4f} -> {result['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
