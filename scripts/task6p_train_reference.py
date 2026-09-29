"""Task 6P Part F (sections 10-11) — ReferenceMaskHead training.

**Stage P1 (Overfit20)** — AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler / no
augmentation / seed 20260929 / evaluate every 100 steps. Gate: train mIoU >= 0.85 and Dice >= 0.90,
otherwise the task stops with `REFERENCE_HEAD_NOT_LEARNABLE`.

**Stage P2 (RefTrainUnique -> RefValUnique)** — fresh initialisation, AdamW / lr 3e-4 / wd 1e-4 /
batch 8 / max 30 epochs / early stopping patience 6 on RefValUnique mIoU / seed 20260929 / model
selection = highest RefValUnique mIoU.

The head receives only the frozen SAM2 feature and the reference family id; the target building, the
target relation id, the relation field and any proposal never enter. The test split is never read.

    python scripts/task6p_train_reference.py --stage p1
    python scripts/task6p_train_reference.py --stage p2
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6p_reference_head import (  # noqa: E402
    FAMILY_TO_INDEX,
    ReferenceMaskHead,
    reference_loss,
)
from scripts.task6n_train import (  # noqa: E402
    FEATURE_ROOT,
    SEED,
    FrozenFeatureStore,
    MaskStore,
    load_frozen_sam2_encoder,
    make_optimizer,
    seed_everything,
)

EVAL = REPO_ROOT / "evaluation"
REF_ROOT = REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "task6p" / "checkpoints"
OUT_P1 = EVAL / "task6p_reference_overfit20.json"
OUT_P2 = EVAL / "task6p_reference_val.json"
TARGET_SIZE = (512, 512)


def read_reference_pack(name: str) -> list[dict]:
    payload = json.loads((REF_ROOT / name).read_text(encoding="utf-8"))
    return payload["records"]


def reference_batch(samples: list[dict], store: FrozenFeatureStore, masks: MaskStore, device: str):
    """Build (visual, family_index, reference target) — no target identity and no relation id."""

    visuals, families, targets = [], [], []
    for sample in samples:
        visual = store.get(sample["tile_id"], Path(sample["image_path"])).float()
        mask = masks.mask(sample["tile_id"], sample["reference_source_feature_id"])
        target = torch.as_tensor(np.asarray(mask, dtype=np.float32))
        visuals.append(visual)
        families.append(FAMILY_TO_INDEX[sample["reference_family"]])
        targets.append(target.unsqueeze(0))
    return (
        torch.stack(visuals).to(device),
        torch.as_tensor(families, dtype=torch.long, device=device),
        torch.stack(targets).to(device),
    )


def normalized_centroid(probability: torch.Tensor) -> tuple[float, float]:
    """Probability-weighted centroid in normalized [0, 1] pixel-centre coordinates."""

    height, width = probability.shape
    x = (torch.arange(width, device=probability.device, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, device=probability.device, dtype=torch.float32) + 0.5) / height
    mass = probability.sum() + 1e-6
    cx = float((probability.sum(dim=0) * x).sum() / mass)
    cy = float((probability.sum(dim=1) * y).sum() / mass)
    return cx, cy


def binary_centroid(mask: torch.Tensor) -> tuple[float, float]:
    height, width = mask.shape
    x = (torch.arange(width, device=mask.device, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, device=mask.device, dtype=torch.float32) + 0.5) / height
    mass = mask.sum() + 1e-6
    cx = float((mask.sum(dim=0) * x).sum() / mass)
    cy = float((mask.sum(dim=1) * y).sum() / mass)
    return cx, cy


@torch.no_grad()
def evaluate_reference(model, samples, store, masks, device, batch_size=8) -> dict:
    model.eval()
    rows = []
    diagonal = float(np.sqrt(2.0))
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        visual, family, target = reference_batch(chunk, store, masks, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(visual, family)
        upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                  align_corners=False)
        probability = torch.sigmoid(upsampled)
        prediction = upsampled > 0.0
        gt = target > 0.5
        for index, sample in enumerate(chunk):
            pred = prediction[index, 0]
            truth = gt[index, 0]
            intersection = float((pred & truth).sum())
            union = float((pred | truth).sum())
            predicted_area = float(pred.sum())
            gt_area = float(truth.sum())
            pred_cx, pred_cy = normalized_centroid(probability[index, 0])
            gt_cx, gt_cy = binary_centroid(truth.float())
            centroid_error = float(np.hypot(pred_cx - gt_cx, pred_cy - gt_cy)) / diagonal
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "tile_id": sample["tile_id"],
                    "reference_family": sample["reference_family"],
                    "miou": (intersection + 1e-6) / (union + 1e-6),
                    "dice": (2.0 * intersection + 1e-6) / (predicted_area + gt_area + 1e-6),
                    "precision_at_0_5": (intersection + 1e-6) / (predicted_area + 1e-6),
                    "centroid_error_normalized": centroid_error,
                    "pred_area": predicted_area,
                    "gt_area": gt_area,
                    "area_ratio": predicted_area / max(gt_area, 1.0),
                    "touches_border": bool(_touches_border(truth)),
                    "tiny": bool(gt_area < 50),
                }
            )
    return _aggregate(rows)


def _touches_border(mask: torch.Tensor) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    errors = np.asarray([row["centroid_error_normalized"] for row in rows], dtype=np.float64)
    ratios = np.asarray([row["area_ratio"] for row in rows], dtype=np.float64)
    summary = {
        "records": len(rows),
        "miou": float(np.mean([row["miou"] for row in rows])),
        "dice": float(np.mean([row["dice"] for row in rows])),
        "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows])),
        "centroid_error_normalized": {
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
        },
        "median_area_ratio": float(np.median(ratios)),
        "per_family": {
            family: {
                "records": sum(1 for row in rows if row["reference_family"] == family),
                "miou": float(np.mean([row["miou"] for row in rows
                                       if row["reference_family"] == family]))
                if any(row["reference_family"] == family for row in rows) else None,
                "dice": float(np.mean([row["dice"] for row in rows
                                       if row["reference_family"] == family]))
                if any(row["reference_family"] == family for row in rows) else None,
            }
            for family in ("largest", "smallest")
        },
        "border_reference": {
            "records": sum(1 for row in rows if row["touches_border"]),
            "miou": float(np.mean([row["miou"] for row in rows if row["touches_border"]]))
            if any(row["touches_border"] for row in rows) else None,
        },
        "non_border_reference": {
            "records": sum(1 for row in rows if not row["touches_border"]),
            "miou": float(np.mean([row["miou"] for row in rows if not row["touches_border"]]))
            if any(not row["touches_border"] for row in rows) else None,
        },
        "tiny_reference": {
            "records": sum(1 for row in rows if row["tiny"]),
            "miou": float(np.mean([row["miou"] for row in rows if row["tiny"]]))
            if any(row["tiny"] for row in rows) else None,
        },
    }
    return summary


def run_p1(args) -> int:
    config = {
        "stage": "P1", "pack": "reference_overfit20", "optimizer": "AdamW", "lr": 1.0e-3,
        "weight_decay": 1.0e-4, "max_steps": 1200, "batch_size": 4, "scheduler": "none",
        "augmentation": "none", "seed": SEED, "amp": "bfloat16 autocast (same policy as 6N/6O)",
        "evaluate_every_steps": 100, "gate": {"train_miou_min": 0.85, "train_dice_min": 0.90},
    }
    started = time.time()
    samples = read_reference_pack("reference_overfit20.json")
    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)

    seed_everything(SEED)
    model = ReferenceMaskHead().to(args.device)
    parameters = model.parameter_report()
    optimizer = make_optimizer(model, config["lr"], config["weight_decay"])
    order = list(range(len(samples)))
    generator = np.random.default_rng(SEED)
    history = []
    best = {"miou": -1.0, "dice": -1.0, "step": 0}
    step = 0
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    while step < config["max_steps"]:
        model.train()
        generator.shuffle(order)
        chunk = [samples[index] for index in order[: config["batch_size"]]]
        visual, family, target = reference_batch(chunk, store, masks, args.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device != "cpu"):
            logits = model(visual, family)
        upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear", align_corners=False)
        losses = reference_loss(upsampled, target)
        losses["loss"].backward()
        optimizer.step()
        step += 1
        if step % config["evaluate_every_steps"] == 0 or step == config["max_steps"]:
            metrics = evaluate_reference(model, samples, store, masks, args.device, config["batch_size"])
            history.append({"step": step, "loss": float(losses["loss"].detach()),
                            "miou": metrics["miou"], "dice": metrics["dice"]})
            if metrics["miou"] > best["miou"]:
                best = {"miou": metrics["miou"], "dice": metrics["dice"], "step": step}
            print(f"[6p.p1] step {step}: loss {float(losses['loss'].detach()):.4f} "
                  f"mIoU {metrics['miou']:.4f} Dice {metrics['dice']:.4f}", flush=True)

    final = evaluate_reference(model, samples, store, masks, args.device, config["batch_size"])
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_ROOT / "p1_reference_head.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, checkpoint_path)
    gate = {
        "miou_min": config["gate"]["train_miou_min"], "dice_min": config["gate"]["train_dice_min"],
        "best_miou": best["miou"], "best_dice": best["dice"],
        "final_miou": final["miou"], "final_dice": final["dice"],
    }
    gate["passed"] = bool(max(gate["best_miou"], gate["final_miou"]) >= gate["miou_min"]
                          and max(gate["best_dice"], gate["final_dice"]) >= gate["dice_min"])
    payload = {
        "_doc": (
            "Task 6P section 10. Stage P1: ReferenceMaskHead trained on 20 unique train references "
            "(10 per family) with the fixed P1 optimiser/step budget. The head never sees the target "
            "building, the target relation id or the relation field."
        ),
        "task": "6P", "stage": "P1", "reference_source": "oracle_native_gt",
        "config": config,
        "pack": {"path": str(REF_ROOT / "reference_overfit20.json"), "count": len(samples),
                 "sha256": hashlib.sha256((REF_ROOT / "reference_overfit20.json").read_bytes()).hexdigest()},
        "parameters": parameters,
        "history": history, "best": best,
        "final": {key: final[key] for key in ("miou", "dice", "precision_at_0_5", "records")},
        "final_detail": final,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if args.device != "cpu" else None,
        "wall_seconds": round(time.time() - started, 1),
        "checkpoint": {"path": str(checkpoint_path),
                       "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
        "gate": gate,
        "verdict": "P1_PASS" if gate["passed"] else "REFERENCE_HEAD_NOT_LEARNABLE",
        "test_split_used": False,
    }
    write_json(OUT_P1, payload)
    print(f"[6p.p1] gate {gate['passed']} (mIoU {gate['best_miou']:.4f} Dice {gate['best_dice']:.4f}) "
          f"-> {payload['verdict']}", flush=True)
    return 0 if gate["passed"] else 2


def run_p2(args) -> int:
    config = {
        "stage": "P2", "train_pack": "ref_train_unique", "val_pack": "ref_val_unique",
        "optimizer": "AdamW", "lr": 3.0e-4, "weight_decay": 1.0e-4, "batch_size": 8,
        "max_epochs": 30, "early_stopping_patience": 6, "early_stopping_monitor": "ref_val_unique_miou",
        "model_selection": "highest RefValUnique mIoU", "scheduler": "none", "augmentation": "none",
        "seed": SEED, "amp": "bfloat16 autocast (same policy as 6N/6O)", "fresh_initialisation": True,
    }
    started = time.time()
    train_samples = read_reference_pack("ref_train_unique.json")
    val_samples = read_reference_pack("ref_val_unique.json")
    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    print(f"[6p.p2] train {len(train_samples)} / val {len(val_samples)} unique references", flush=True)

    seed_everything(SEED)
    model = ReferenceMaskHead().to(args.device)
    parameters = model.parameter_report()
    optimizer = make_optimizer(model, config["lr"], config["weight_decay"])
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    best = {"miou": -1.0, "dice": -1.0, "epoch": 0, "state": None, "detail": None}
    epochs = []
    generator = np.random.default_rng(SEED)
    stale = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = generator.permutation(len(train_samples)).tolist()
        losses = []
        for start in range(0, len(order), config["batch_size"]):
            chunk = [train_samples[index] for index in order[start: start + config["batch_size"]]]
            if not chunk:
                continue
            visual, family, target = reference_batch(chunk, store, masks, args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device != "cpu"):
                logits = model(visual, family)
            upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                      align_corners=False)
            loss = reference_loss(upsampled, target)["loss"]
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val = evaluate_reference(model, val_samples, store, masks, args.device, config["batch_size"])
        epochs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_miou": val["miou"],
                       "val_dice": val["dice"],
                       "val_centroid_median": val["centroid_error_normalized"]["median"]})
        print(f"[6p.p2] epoch {epoch}: loss {np.mean(losses):.4f} val mIoU {val['miou']:.4f} "
              f"Dice {val['dice']:.4f}", flush=True)
        if val["miou"] > best["miou"]:
            best = {"miou": val["miou"], "dice": val["dice"], "epoch": epoch,
                    "state": {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()},
                    "detail": val}
            stale = 0
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]:
                print(f"[6p.p2] early stop at epoch {epoch} "
                      f"(patience {config['early_stopping_patience']})", flush=True)
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    final = evaluate_reference(model, val_samples, store, masks, args.device, config["batch_size"])
    checkpoint_path = CHECKPOINT_ROOT / "p2_reference_head.pt"
    torch.save({"state_dict": model.state_dict(), "config": config, "best_epoch": best["epoch"]},
               checkpoint_path)
    payload = {
        "_doc": (
            "Task 6P section 11. Stage P2: ReferenceMaskHead trained from fresh initialisation on "
            "RefTrainUnique and selected on RefValUnique mIoU. All head parameters are frozen after "
            "this stage; it is never jointly trained with the target decoder B3."
        ),
        "task": "6P", "stage": "P2", "reference_source": "oracle_native_gt",
        "config": config,
        "packs": {
            "train": {"path": str(REF_ROOT / "ref_train_unique.json"), "count": len(train_samples),
                      "sha256": hashlib.sha256((REF_ROOT / "ref_train_unique.json").read_bytes()).hexdigest()},
            "val": {"path": str(REF_ROOT / "ref_val_unique.json"), "count": len(val_samples),
                    "sha256": hashlib.sha256((REF_ROOT / "ref_val_unique.json").read_bytes()).hexdigest()},
        },
        "parameters": parameters,
        "epochs": epochs,
        "epochs_run": len(epochs),
        "best_epoch": best["epoch"],
        "selected_model": final,
        "best_selection_detail": best.get("detail"),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if args.device != "cpu" else None,
        "wall_seconds": round(time.time() - started, 1),
        "checkpoint": {"path": str(checkpoint_path),
                       "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
        "frozen_after_training": True,
        "joint_training_with_b3": False,
        "test_split_used": False,
    }
    write_json(OUT_P2, payload)
    print(f"[6p.p2] selected epoch {best['epoch']} val mIoU {final['miou']:.6f} "
          f"Dice {final['dice']:.6f} | centroid median "
          f"{final['centroid_error_normalized']['median']:.6f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("p1", "p2"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return run_p1(args) if args.stage == "p1" else run_p2(args)


if __name__ == "__main__":
    raise SystemExit(main())
