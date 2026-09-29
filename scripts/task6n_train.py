"""Task 6N sections 14-16 — Stage N1 (Overfit20) and Stage N2 (MiniTrain1000 -> MiniVal240).

Trains exactly the three controlled variants of one decoder family:

* ``B0`` relation-aware visual baseline (visual + relation embedding) — first conv 144 channels
* ``B1`` reference-mask baseline (visual + M_ref_down + relation) — first conv 145 channels
* ``B2`` full GeometricRelationField (visual + M_ref_down + P_rel + relation) — first conv 146 channels

Frozen settings (never swept):

* N1: AdamW, lr 1e-3, weight_decay 1e-4, max 1200 steps, batch 4, no scheduler, no augmentation,
  seed 20260929, evaluate every 100 steps; gate: B2 train mIoU >= 0.85 and Dice >= 0.90.
* N2: AdamW, lr 3e-4, weight_decay 1e-4, batch 8, max 25 epochs, early stopping patience 5 on
  MiniVal240 mIoU, fresh initialisation, seed 20260929, model selection = best MiniVal240 mIoU.

The target mask is used only as the training label; it never enters the model input. The test split is
never read.

    python scripts/task6n_train.py --stage n1
    python scripts/task6n_train.py --stage n2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import canonical_instances, write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    Task6NSample,
    VARIANTS,
    VARIANT_LABELS,
    collate_samples,
    load_frozen_sam2_encoder,
    read_pack,
    reference_centroid,
    task6n_loss,
    upsampled_logits,
)

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "task6n" / "checkpoints"
FEATURE_ROOT = REPO_ROOT / "artifacts" / "task6n" / "features"
OUT_N1 = EVAL / "task6n_overfit20.json"
OUT_N2 = EVAL / "task6n_mini_val.json"

SEED = 20260929
FEATURE_SIZE = (64, 64)
TARGET_SIZE = (512, 512)


# --------------------------------------------------------------------------- mask storage


class MaskStore:
    """Canonical native-instance masks per (tile, source_feature_id), memoised."""

    def __init__(self) -> None:
        self._instances: dict[str, list] = {}
        self._by_feature: dict[str, dict[int, np.ndarray]] = {}

    def _load(self, tile_id: str) -> dict[int, np.ndarray]:
        if tile_id not in self._by_feature:
            instances = canonical_instances(tile_id)
            self._instances[tile_id] = instances
            self._by_feature[tile_id] = {instance.source_feature_id: instance.mask for instance in instances}
        return self._by_feature[tile_id]

    def mask(self, tile_id: str, source_feature_id: int) -> np.ndarray:
        table = self._load(tile_id)
        if source_feature_id not in table:
            raise KeyError(f"{tile_id}: source_feature_id {source_feature_id} not in the native store")
        return table[source_feature_id]

    def instances(self, tile_id: str) -> list:
        """Canonical native instances of one tile (for derived properties such as border/tiny)."""

        self._load(tile_id)
        return self._instances[tile_id]


# --------------------------------------------------------------------------- metrics


def binary_metrics(logits: torch.Tensor, target: torch.Tensor, size=TARGET_SIZE) -> dict:
    """mIoU / Dice / precision at the canonical target-mask resolution (bilinear upsample)."""

    upsampled = upsampled_logits(logits, size)
    prediction = upsampled > 0.0
    target_bool = target.to(prediction.device) > 0.5
    if target_bool.dim() == 3:
        target_bool = target_bool.unsqueeze(1)
    dims = tuple(range(1, prediction.dim()))
    intersection = (prediction & target_bool).sum(dim=dims).float()
    union = (prediction | target_bool).sum(dim=dims).float()
    predicted = prediction.sum(dim=dims).float()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    dice = ((2.0 * intersection + 1e-6) / (predicted + target_bool.sum(dim=dims).float() + 1e-6)).mean().item()
    precision = ((intersection + 1e-6) / (predicted + 1e-6)).mean().item()
    return {"miou": iou, "dice": dice, "precision_at_0_5": precision}


@torch.no_grad()
def evaluate(
    model: RelationMaskDecoder,
    samples: list[Task6NSample],
    store: FrozenFeatureStore,
    masks: MaskStore,
    device: str,
    batch_size: int = 8,
) -> dict:
    model.eval()
    rows = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        batch = _build_batch(chunk, store, masks, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(batch.visual, batch.relation_index, batch.mask_ref_down, batch.field)
        per_sample = _per_sample_metrics(logits, batch)
        rows.extend(per_sample)
    return _aggregate(rows)


def _per_sample_metrics(logits: torch.Tensor, batch) -> list[dict]:
    upsampled = upsampled_logits(logits, batch.target_size)
    prediction = upsampled > 0.0
    target = batch.target > 0.5
    if target.dim() == 3:
        target = target.unsqueeze(1)
    rows = []
    for index, sample_id in enumerate(batch.sample_ids):
        pred = prediction[index]
        gt = target[index]
        intersection = float((pred & gt).sum())
        union = float((pred | gt).sum())
        predicted = float(pred.sum())
        gt_area = float(gt.sum())
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2.0 * intersection + 1e-6) / (predicted + gt_area + 1e-6)
        precision = (intersection + 1e-6) / (predicted + 1e-6)
        rows.append(
            {
                "sample_id": sample_id,
                "program_id": batch.programs[index],
                "miou": iou,
                "dice": dice,
                "precision_at_0_5": precision,
                "target_area_px": gt_area,
            }
        )
    return rows


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    values = lambda key: [row[key] for row in rows]  # noqa: E731
    summary = {
        "records": len(rows),
        "miou": float(np.mean(values("miou"))),
        "dice": float(np.mean(values("dice"))),
        "precision_at_0_5": float(np.mean(values("precision_at_0_5"))),
    }
    per_relation = {}
    for relation in ("left_of", "right_of", "above", "below"):
        subset = [row for row in rows if row["program_id"].endswith(f"to_{relation}")]
        if subset:
            per_relation[relation] = {
                "records": len(subset),
                "miou": float(np.mean([row["miou"] for row in subset])),
                "dice": float(np.mean([row["dice"] for row in subset])),
            }
    per_family = {}
    for family in ("largest", "smallest"):
        subset = [row for row in rows if row["program_id"].startswith(family)]
        if subset:
            per_family[family] = {
                "records": len(subset),
                "miou": float(np.mean([row["miou"] for row in subset])),
                "dice": float(np.mean([row["dice"] for row in subset])),
            }
    per_program = {}
    for program in sorted({row["program_id"] for row in rows}):
        subset = [row for row in rows if row["program_id"] == program]
        per_program[program] = {
            "records": len(subset),
            "miou": float(np.mean([row["miou"] for row in subset])),
        }
    summary["per_relation"] = per_relation
    summary["per_reference_family"] = per_family
    summary["per_program"] = per_program
    return summary


def _build_batch(samples: list[Task6NSample], store: FrozenFeatureStore, masks: MaskStore, device: str):
    reference_masks = {sample.sample_id: masks.mask(sample.tile_id, sample.reference_source_feature_id)
                       for sample in samples}
    target_masks = {sample.sample_id: masks.mask(sample.tile_id, sample.target_source_feature_id)
                    for sample in samples}
    return collate_samples(samples, store, FEATURE_SIZE, target_masks, reference_masks, device)


# --------------------------------------------------------------------------- training helpers


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def paired_own_cross(
    model: RelationMaskDecoder,
    pack_path: Path,
    samples: list[Task6NSample],
    store: FrozenFeatureStore,
    masks: MaskStore,
    device: str,
) -> dict:
    """Own-target vs cross-target IoU on the pack's same-image counterfactual pairs."""

    payload = json.loads(Path(pack_path).read_text(encoding="utf-8"))
    pairs = payload.get("detail", {}).get("counterfactual_pairs") or []
    if not pairs:
        return {"pairs": 0, "available": False}
    by_id = {sample.sample_id: sample for sample in samples}
    rows = []
    for pair in pairs:
        first = by_id.get(pair["a"]["sample_id"])
        second = by_id.get(pair["b"]["sample_id"])
        if first is None or second is None:
            continue
        members = [first, second]
        batch = _build_batch(members, store, masks, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(batch.visual, batch.relation_index, batch.mask_ref_down, batch.field)
        upsampled = upsampled_logits(logits, batch.target_size) > 0.0
        own = []
        cross = []
        for index in range(len(members)):
            own.append(float((upsampled[index] & (batch.target[index] > 0.5)).sum())
                       / float((upsampled[index] | (batch.target[index] > 0.5)).sum() + 1e-6))
            other = 1 - index
            other_target = masks.mask(members[other].tile_id, members[other].target_source_feature_id)
            other_tensor = torch.as_tensor(other_target, dtype=torch.bool, device=upsampled.device)
            cross.append(float((upsampled[index] & other_tensor).sum())
                         / float((upsampled[index] | other_tensor).sum() + 1e-6))
        rows.append(
            {
                "tile_id": members[0].tile_id,
                "own": own,
                "cross": cross,
                "passes": all(own[i] > cross[i] for i in range(2)),
            }
        )
    passed = sum(1 for row in rows if row["passes"])
    return {
        "pairs": len(rows),
        "available": True,
        "passed": passed,
        "mean_own_iou": float(np.mean([value for row in rows for value in row["own"]])),
        "mean_cross_iou": float(np.mean([value for row in rows for value in row["cross"]])),
        "own_cross_margin": float(
            np.mean([value for row in rows for value in row["own"]])
            - np.mean([value for row in rows for value in row["cross"]])
        ),
        "rows": rows,
    }


def run_n1(args) -> int:
    config = {
        "variant": "all",
        "optimizer": "AdamW",
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "max_steps": 1200,
        "batch_size": 4,
        "scheduler": "none",
        "augmentation": "none",
        "seed": SEED,
        "amp": "bfloat16 autocast, identical across variants",
        "evaluate_every_steps": 100,
        "gate": {"train_miou_min": 0.85, "train_dice_min": 0.90},
    }
    started = time.time()
    pack_path = PACK_ROOT / "overfit20.json"
    samples = read_pack(pack_path)
    masks = MaskStore()
    encoder, sam_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    print(f"[6n.n1] Overfit20 records {len(samples)}; visual C x h x w = 256 x 64 x 64", flush=True)

    results = {}
    for variant in VARIANTS:
        seed_everything(SEED)
        model = RelationMaskDecoder(variant, DecoderConfig()).to(args.device)
        optimizer = make_optimizer(model, config["lr"], config["weight_decay"])
        parameters = model.parameter_report()
        order = list(range(len(samples)))
        generator = np.random.default_rng(SEED)
        history = []
        best = {"miou": -1.0, "dice": -1.0, "step": 0}
        step = 0
        torch.cuda.reset_peak_memory_stats() if args.device != "cpu" else None
        variant_started = time.time()
        while step < config["max_steps"]:
            model.train()
            generator.shuffle(order)
            batch_indices = order[: config["batch_size"]]
            chunk = [samples[index] for index in batch_indices]
            batch = _build_batch(chunk, store, masks, args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device != "cpu"):
                logits = model(batch.visual, batch.relation_index, batch.mask_ref_down, batch.field)
            losses = task6n_loss(upsampled_logits(logits, batch.target_size), batch.target)
            losses["loss"].backward()
            optimizer.step()
            step += 1

            if step % config["evaluate_every_steps"] == 0 or step == config["max_steps"]:
                metrics = evaluate(model, samples, store, masks, args.device, config["batch_size"])
                history.append(
                    {"step": step, "loss": float(losses["loss"].detach()),
                     "miou": metrics["miou"], "dice": metrics["dice"]}
                )
                if metrics["miou"] > best["miou"]:
                    best = {"miou": metrics["miou"], "dice": metrics["dice"], "step": step}
                print(
                    f"[6n.n1] {variant} step {step}: loss {float(losses['loss'].detach()):.4f} "
                    f"mIoU {metrics['miou']:.4f} Dice {metrics['dice']:.4f}",
                    flush=True,
                )

        final = evaluate(model, samples, store, masks, args.device, config["batch_size"])
        pairing = paired_own_cross(model, pack_path, samples, store, masks, args.device)
        checkpoint_path = CHECKPOINT_ROOT / f"n1_{variant}.pt"
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        torch.save({"variant": variant, "state_dict": model.state_dict(), "config": config}, checkpoint_path)
        results[variant] = {
            "label": VARIANT_LABELS[variant],
            "parameters": parameters,
            "history": history,
            "best": best,
            "final": {key: final[key] for key in ("miou", "dice", "precision_at_0_5", "records")},
            "final_per_relation": final["per_relation"],
            "paired_own_cross": pairing,
            "wall_seconds": round(time.time() - variant_started, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if args.device != "cpu" else None,
            "checkpoint": {"path": str(checkpoint_path),
                           "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
        }
        print(
            f"[6n.n1] {variant} done: best mIoU {best['miou']:.4f} (step {best['step']}), "
            f"final mIoU {final['miou']:.4f} Dice {final['dice']:.4f}",
            flush=True,
        )

    b2 = results["B2"]["final"]
    gate = {
        "train_miou_min": config["gate"]["train_miou_min"],
        "train_dice_min": config["gate"]["train_dice_min"],
        "b2_best_miou": results["B2"]["best"]["miou"],
        "b2_best_dice": results["B2"]["best"]["dice"],
        "b2_final_miou": b2["miou"],
        "b2_final_dice": b2["dice"],
    }
    gate["passed"] = bool(
        max(gate["b2_best_miou"], gate["b2_final_miou"]) >= gate["train_miou_min"]
        and max(gate["b2_best_dice"], gate["b2_final_dice"]) >= gate["train_dice_min"]
    )
    payload = {
        "_doc": (
            "Task 6N section 14. Stage N1: the three variants trained separately on the exact same "
            "Overfit20 pack with the frozen N1 optimiser/step budget. The N1 gate asks whether B2 can "
            "fit 20 samples; failure stops the task with "
            "GEOMETRIC_RELATION_FIELD_NOT_LEARNABLE_IN_CURRENT_DECODER."
        ),
        "task": "6N",
        "stage": "N1",
        "reference_source": "oracle_native_gt",
        "config": config,
        "pack": {"path": str(pack_path), "count": len(samples),
                 "sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest()},
        "visual": store.provenance(),
        "sam2_load_report": sam_report.as_dict() if hasattr(sam_report, "as_dict") else str(sam_report),
        "variants": results,
        "gate": gate,
        "verdict": "N1_PASS" if gate["passed"] else "GEOMETRIC_RELATION_FIELD_NOT_LEARNABLE_IN_CURRENT_DECODER",
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_N1, payload)
    print(f"[6n.n1] gate {gate['passed']} (B2 mIoU {gate['b2_best_miou']:.4f} / Dice "
          f"{gate['b2_best_dice']:.4f}) -> {payload['verdict']}", flush=True)
    return 0 if gate["passed"] else 2


def run_n2(args) -> int:
    config = {
        "train_pack": "mini_train_1000",
        "val_pack": "mini_val_240",
        "optimizer": "AdamW",
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "batch_size": 8,
        "max_epochs": 25,
        "early_stopping_patience": 5,
        "early_stopping_monitor": "val_miou",
        "model_selection": "highest MiniVal240 mIoU",
        "scheduler": "none",
        "augmentation": "none",
        "seed": SEED,
        "amp": "bfloat16 autocast, identical across variants",
        "fresh_initialisation": True,
    }
    started = time.time()
    train_path = PACK_ROOT / "mini_train_1000.json"
    val_path = PACK_ROOT / "mini_val_240.json"
    train_samples = read_pack(train_path)
    val_samples = read_pack(val_path)
    masks = MaskStore()
    encoder, sam_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    print(f"[6n.n2] train {len(train_samples)} / val {len(val_samples)}", flush=True)

    results = {}
    for variant in VARIANTS:
        seed_everything(SEED)
        model = RelationMaskDecoder(variant, DecoderConfig()).to(args.device)
        optimizer = make_optimizer(model, config["lr"], config["weight_decay"])
        parameters = model.parameter_report()
        torch.cuda.reset_peak_memory_stats() if args.device != "cpu" else None
        variant_started = time.time()
        best = {"miou": -1.0, "dice": -1.0, "epoch": 0, "state": None}
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
                batch = _build_batch(chunk, store, masks, args.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device != "cpu"):
                    logits = model(batch.visual, batch.relation_index, batch.mask_ref_down, batch.field)
                loss = task6n_loss(upsampled_logits(logits, batch.target_size), batch.target)["loss"]
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            val = evaluate(model, val_samples, store, masks, args.device, config["batch_size"])
            epochs.append(
                {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_miou": val["miou"],
                 "val_dice": val["dice"], "val_precision_at_0_5": val["precision_at_0_5"]}
            )
            print(
                f"[6n.n2] {variant} epoch {epoch}: loss {np.mean(losses):.4f} "
                f"val mIoU {val['miou']:.4f} Dice {val['dice']:.4f}",
                flush=True,
            )
            if val["miou"] > best["miou"]:
                best = {
                    "miou": val["miou"], "dice": val["dice"], "epoch": epoch,
                    "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "per_relation": val["per_relation"], "per_program": val["per_program"],
                }
                stale = 0
            else:
                stale += 1
                if stale >= config["early_stopping_patience"]:
                    print(f"[6n.n2] {variant}: early stop at epoch {epoch} (patience "
                          f"{config['early_stopping_patience']})", flush=True)
                    break

        # model selection = highest MiniVal240 mIoU
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        final = evaluate(model, val_samples, store, masks, args.device, config["batch_size"])
        checkpoint_path = CHECKPOINT_ROOT / f"n2_{variant}.pt"
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        torch.save({"variant": variant, "state_dict": model.state_dict(), "config": config,
                    "best_epoch": best["epoch"]}, checkpoint_path)
        results[variant] = {
            "label": VARIANT_LABELS[variant],
            "parameters": parameters,
            "epochs": epochs,
            "epochs_run": len(epochs),
            "best": {"epoch": best["epoch"], "miou": best["miou"], "dice": best["dice"],
                     "per_relation": best.get("per_relation"), "per_program": best.get("per_program")},
            "selected_model": {
                "miou": final["miou"], "dice": final["dice"],
                "precision_at_0_5": final["precision_at_0_5"],
                "per_relation": final["per_relation"],
                "per_reference_family": final["per_reference_family"],
                "per_program": final["per_program"],
                "records": final["records"],
            },
            "wall_seconds": round(time.time() - variant_started, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if args.device != "cpu" else None,
            "checkpoint": {"path": str(checkpoint_path),
                           "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
        }
        print(f"[6n.n2] {variant} selected epoch {best['epoch']} val mIoU {final['miou']:.4f}", flush=True)

    payload = {
        "_doc": (
            "Task 6N sections 15-16. Stage N2: the three controlled variants trained from fresh "
            "initialisation on MiniTrain1000 and selected on MiniVal240 mIoU. Oracle reference masks "
            "are inputs; the GT target is the label only."
        ),
        "task": "6N",
        "stage": "N2",
        "reference_source": "oracle_native_gt",
        "config": config,
        "packs": {
            "train": {"path": str(train_path), "count": len(train_samples),
                      "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()},
            "val": {"path": str(val_path), "count": len(val_samples),
                    "sha256": hashlib.sha256(val_path.read_bytes()).hexdigest()},
        },
        "visual": store.provenance(),
        "sam2_load_report": sam_report.as_dict() if hasattr(sam_report, "as_dict") else str(sam_report),
        "variants": results,
        "comparisons": {
            "b2_minus_b0_miou": results["B2"]["selected_model"]["miou"] - results["B0"]["selected_model"]["miou"],
            "b2_minus_b1_miou": results["B2"]["selected_model"]["miou"] - results["B1"]["selected_model"]["miou"],
            "b1_minus_b0_miou": results["B1"]["selected_model"]["miou"] - results["B0"]["selected_model"]["miou"],
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_N2, payload)
    print(
        f"[6n.n2] mIoU B0 {results['B0']['selected_model']['miou']:.4f} | "
        f"B1 {results['B1']['selected_model']['miou']:.4f} | "
        f"B2 {results['B2']['selected_model']['miou']:.4f}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("n1", "n2"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "n1":
        return run_n1(args)
    return run_n2(args)


if __name__ == "__main__":
    raise SystemExit(main())
