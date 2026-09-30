"""Task 6Z Parts H-I — Stage Z1 (Overfit20) and Stage Z2 (MiniTrain1200 -> MiniVal240).

All six predeclared variants are trained separately from fresh initialization on the frozen L3 packs, with
the frozen SAM2 visual representation, the frozen directional field v0.2 and the frozen Task 6Y nearest
field v0.1 (product field = exact clamped multiplication).

* Z1 (section 23): AdamW, lr 1e-3, weight_decay 1e-4, batch 4, max 1200 steps, no scheduler, no
  augmentation, seed 20260930, bfloat16 AMP, evaluate every 100 steps. Primary gate: Z-B3 mIoU >= 0.85 and
  Dice >= 0.90, otherwise STOP `L3_LEARNED_COMPOSITION_NOT_LEARNABLE` (Z2 then never runs). Z-B4/B5 have no
  stop gate.
* Z2 (section 25): AdamW, lr 3e-4, weight_decay 1e-4, batch 8, max 25 epochs, early-stopping patience 5,
  model selection = MiniVal240 mIoU, seed 20260930, no scheduler, no augmentation, same AMP.

Loss exactly `BCEWithLogitsLoss + DiceLoss`; the GT target mask is a label only. No candidate masks, no
predicted reference, no attention, no GRCL, no test split.

    python scripts/task6z_train.py --stage z1
    python scripts/task6z_train.py --stage z2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
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
    upsampled_logits,
)
from buildreasonseg_mvp.task6z_field_composition import PROGRAM_TO_RELATION, field_report  # noqa: E402
from buildreasonseg_mvp.task6z_l3_decoder import (  # noqa: E402
    ALL_VARIANTS,
    L3TargetDecoder,
    variant_report,
)
from scripts.task6n_train import (  # noqa: E402
    FEATURE_ROOT,
    MaskStore,
    _aggregate,
    _per_sample_metrics,
    make_optimizer,
    seed_everything,
)
from scripts.task6z_fields import field_bundle  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task6z"
OUT_Z1 = EVAL / "task6z_overfit20.json"
OUT_Z2 = EVAL / "task6z_training.json"
SEED = 20260930
Z1 = {"lr": 1.0e-3, "weight_decay": 1.0e-4, "batch": 4, "max_steps": 1200, "eval_every": 100,
      "gate_miou": 0.85, "gate_dice": 0.90}
Z2 = {"lr": 3.0e-4, "weight_decay": 1.0e-4, "batch": 8, "max_epochs": 25, "patience": 5}


@dataclass
class L3Batch:
    visual: torch.Tensor | None
    directional: torch.Tensor
    nearest: torch.Tensor
    product: torch.Tensor
    relations: list[str]
    target: torch.Tensor
    sample_ids: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    target_size: tuple[int, int] = (512, 512)


def load_pack(name: str) -> list[dict]:
    return json.loads((PACK_ROOT / f"{name}.json").read_text(encoding="utf-8"))["records"]


def image_path_of(record: dict) -> Path:
    path = Path(record["image_path"])
    return path if path.is_absolute() else REPO_ROOT / path


def build_batch(records: list[dict], store: FrozenFeatureStore, masks: MaskStore, device: str,
                precomputed: dict[str, dict]) -> L3Batch:
    visuals, directionals, nearests, products, targets = [], [], [], [], []
    for record in records:
        sample_id = record["sample_id"]
        entry = precomputed.get(sample_id)
        if entry is None:
            reference_mask = np.asarray(masks.mask(record["tile_id"],
                                                   record["reference_source_feature_id"]),
                                        dtype=bool)
            entry = field_bundle(reference_mask, record["program_id"])
            precomputed[sample_id] = entry
        directionals.append(entry["P_dir_64"][None])
        nearests.append(entry["P_near_64"][None])
        products.append(entry["P_prod_64"][None])
        target = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                            dtype=np.float32)
        targets.append(torch.as_tensor(target))
        visuals.append(store.get(record["tile_id"], image_path_of(record)).float())
    return L3Batch(
        visual=torch.stack(visuals).to(device),
        directional=torch.stack(directionals).to(device),
        nearest=torch.stack(nearests).to(device),
        product=torch.stack(products).to(device),
        relations=[PROGRAM_TO_RELATION[record["program_id"]] for record in records],
        target=torch.stack(targets).to(device),
        sample_ids=[record["sample_id"] for record in records],
        programs=[record["program_id"] for record in records],
        target_size=(int(targets[0].shape[0]), int(targets[0].shape[1])),
    )


def forward_variant(model: L3TargetDecoder, batch: L3Batch) -> torch.Tensor:
    """Hand each variant exactly the inputs it is allowed to see."""

    return model(batch.visual if model.uses_visual else None, batch.relations,
                 batch.directional if model.uses_directional_field else None,
                 batch.nearest if model.uses_nearest_field else None,
                 batch.product if model.uses_product_field else None)


@torch.no_grad()
def evaluate_variant(model: L3TargetDecoder, records: list[dict], store, masks, device,
                     precomputed, batch_size: int = 8) -> dict:
    model.eval()
    rows = []
    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        batch = build_batch(chunk, store, masks, device, precomputed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = forward_variant(model, batch)
        rows.extend(_per_sample_metrics(logits, batch))
    return _aggregate(rows)


def by_direction_metrics(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        direction = str(row["program_id"]).split("_to_")[1].replace("_to_nearest", "")
        groups.setdefault(direction, []).append(row)
    return {direction: {"records": len(subset),
                        "miou": float(np.mean([row["miou"] for row in subset])),
                        "dice": float(np.mean([row["dice"] for row in subset])),
                        "precision_at_0_5": float(np.mean([row["precision_at_0_5"]
                                                           for row in subset]))}
            for direction, subset in sorted(groups.items())}


def train_one(variant: str, train_records: list[dict], store, masks, device, precomputed, *,
              lr: float, weight_decay: float, batch_size: int, max_steps: int | None,
              max_epochs: int | None, patience: int | None, eval_every: int | None,
              selection_records: list[dict], checkpoint_path: Path) -> dict:
    seed_everything(SEED)
    model = L3TargetDecoder(variant).to(device)
    optimizer = make_optimizer(model, lr, weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = []
    best = {"key": -1.0, "step": 0, "epoch": 0, "metrics": None, "state": None}
    step = 0
    epoch = 0
    stale = 0
    stop = False
    while not stop:
        epoch += 1
        generator = np.random.default_rng(SEED + epoch)
        order = generator.permutation(len(train_records))
        for start in range(0, len(order), batch_size):
            index = order[start: start + batch_size]
            chunk = [train_records[int(value)] for value in index]
            batch = build_batch(chunk, store, masks, device, precomputed)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits = forward_variant(model, batch)
                losses = task6n_loss(upsampled_logits(logits, batch.target_size),
                                     batch.target.unsqueeze(1))
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if eval_every is not None and step % eval_every == 0:
                metrics = evaluate_variant(model, train_records, store, masks, device, precomputed)
                history.append({"step": step, "epoch": epoch, "miou": metrics["miou"],
                                "dice": metrics["dice"], "loss": float(losses["loss"].detach())})
                print(f"[6z.train] {variant} step {step}: mIoU {metrics['miou']:.4f} "
                      f"Dice {metrics['dice']:.4f}", flush=True)
                if metrics["miou"] > best["key"]:
                    best = {"key": metrics["miou"], "step": step, "epoch": epoch, "metrics": metrics,
                            "state": {key: value.detach().cpu().clone()
                                      for key, value in model.state_dict().items()}}
            if max_steps is not None and step >= max_steps:
                stop = True
                break
        if max_steps is None and selection_records:
            metrics = evaluate_variant(model, selection_records, store, masks, device, precomputed)
            history.append({"step": step, "epoch": epoch, "miou": metrics["miou"],
                            "dice": metrics["dice"], "loss": None})
            print(f"[6z.train] {variant} epoch {epoch}: MiniVal mIoU {metrics['miou']:.4f} "
                  f"Dice {metrics['dice']:.4f}", flush=True)
            if metrics["miou"] > best["key"]:
                best = {"key": metrics["miou"], "step": step, "epoch": epoch, "metrics": metrics,
                        "state": {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    print(f"[6z.train] {variant} early stop at epoch {epoch}", flush=True)
                    stop = True
        if max_epochs is not None and epoch >= max_epochs:
            stop = True
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    @torch.no_grad()
    def rows_for(records: list[dict]) -> list[dict]:
        model.eval()
        collected = []
        for start in range(0, len(records), 8):
            chunk = records[start: start + 8]
            batch = build_batch(chunk, store, masks, device, precomputed)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits = forward_variant(model, batch)
            collected.extend(_per_sample_metrics(logits, batch))
        return collected

    final_rows = rows_for(selection_records or train_records)
    final = _aggregate(final_rows)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": variant, "state_dict": model.state_dict(), "seed": SEED, "lr": lr,
                "weight_decay": weight_decay, "batch": batch_size, "best_step": best["step"],
                "best_epoch": best["epoch"]}, checkpoint_path)
    return {
        "variant": variant,
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
        "best": {"miou": best["metrics"]["miou"] if best["metrics"] else None,
                 "dice": best["metrics"]["dice"] if best["metrics"] else None,
                 "step": best["step"], "epoch": best["epoch"]},
        "final": final, "final_by_direction": by_direction_metrics(final_rows),
        "steps": step, "epochs": epoch,
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if device != "cpu" else None,
        "checkpoint": {"path": str(checkpoint_path),
                       "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                       "bytes": checkpoint_path.stat().st_size, "committed": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("z1", "z2"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    encoder, encoder_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()

    if args.stage == "z1":
        records = load_pack("z_overfit20")
        precomputed: dict[str, dict] = {}
        results = {}
        for variant in ALL_VARIANTS:
            results[variant] = train_one(
                variant, records, store, masks, args.device, precomputed, lr=Z1["lr"],
                weight_decay=Z1["weight_decay"], batch_size=Z1["batch"], max_steps=Z1["max_steps"],
                max_epochs=None, patience=None, eval_every=Z1["eval_every"],
                selection_records=records,
                checkpoint_path=CHECKPOINT_ROOT / f"{variant.lower().replace('-', '')}_overfit20.pt")
        b3 = results["Z-B3"]
        gate = {"miou_min": Z1["gate_miou"], "dice_min": Z1["gate_dice"],
                "measured_miou": b3["best"]["miou"], "measured_dice": b3["best"]["dice"],
                "applies_to": "Z-B3",
                "passed": bool(b3["best"]["miou"] >= Z1["gate_miou"]
                               and b3["best"]["dice"] >= Z1["gate_dice"])}
        payload = {
            "_doc": (
                "Task 6Z section 23. Stage Z1 Overfit20: the six predeclared L3 variants trained from "
                "fresh init on the frozen 20-record pack (5 per direction) with the oracle largest "
                "reference, bfloat16 AMP and exactly BCE+Dice. Primary gate: Z-B3 mIoU >= 0.85, Dice "
                ">= 0.90."
            ),
            "task": "6Z", "stage": "Z1-overfit20", "seed": SEED,
            "pack": {"name": "z_overfit20", "records": len(records),
                     "by_direction": {direction: sum(1 for record in records
                                                     if record["direction"] == direction)
                                      for direction in ("above", "below", "left", "right")}},
            "training": Z1, "variants": variant_report(), "fields": field_report(),
            "results": results, "gate": gate, "z_b3_overfit_passed": gate["passed"],
            "verdict": ("L3_OVERFIT_PASS" if gate["passed"] else "L3_LEARNED_COMPOSITION_NOT_LEARNABLE"),
            "encoder": encoder_report, "training_performed": True, "test_split_used": False,
            "runtime_seconds": round(time.time() - started, 1),
        }
        write_json(OUT_Z1, payload)
        print(f"[6z.train] Z1 Z-B3 mIoU {gate['measured_miou']:.4f} Dice {gate['measured_dice']:.4f} "
              f"-> {payload['verdict']}", flush=True)
        return 0 if gate["passed"] else 3

    z1 = json.loads(OUT_Z1.read_text(encoding="utf-8")) if OUT_Z1.is_file() else None
    if z1 is None or not z1.get("z_b3_overfit_passed"):
        write_json(OUT_Z2, {"_doc": "Task 6Z section 25.", "task": "6Z",
                            "verdict": "L3_LEARNED_COMPOSITION_NOT_LEARNABLE",
                            "reason": "the Z-B3 Overfit20 gate did not pass; Z2 never runs"})
        print("[6z.train] STOP L3_LEARNED_COMPOSITION_NOT_LEARNABLE", flush=True)
        return 3

    train_records = load_pack("z_mini_train_1200")
    val_records = load_pack("z_mini_val_240")
    precomputed = {}
    results = {}
    for variant in ALL_VARIANTS:
        results[variant] = train_one(
            variant, train_records, store, masks, args.device, precomputed, lr=Z2["lr"],
            weight_decay=Z2["weight_decay"], batch_size=Z2["batch"], max_steps=None,
            max_epochs=Z2["max_epochs"], patience=Z2["patience"], eval_every=None,
            selection_records=val_records,
            checkpoint_path=CHECKPOINT_ROOT / f"{variant.lower().replace('-', '')}_minitrain1200.pt")
    payload = {
        "_doc": (
            "Task 6Z section 25. Stage Z2 MiniTrain1200 -> MiniVal240: the six predeclared L3 variants "
            "trained from fresh init with the oracle largest reference, selection by MiniVal240 mIoU, "
            "patience 5 and exactly BCE+Dice. No test split."
        ),
        "task": "6Z", "stage": "Z2-minitrain1200", "seed": SEED,
        "packs": {"train": {"name": "z_mini_train_1200", "records": len(train_records)},
                  "selection": {"name": "z_mini_val_240", "records": len(val_records)}},
        "training": Z2, "variants": variant_report(), "fields": field_report(),
        "results": results, "selection_metric": "MiniVal240 mIoU",
        "encoder": encoder_report, "training_performed": True, "test_split_used": False,
        "verdict": "L3_TRAINING_COMPLETE", "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_Z2, payload)
    for variant in ALL_VARIANTS:
        entry = results[variant]
        print(f"[6z.train] {variant}: best MiniVal mIoU {entry['best']['miou']:.4f} Dice "
              f"{entry['best']['dice']:.4f} epoch {entry['best']['epoch']} params {entry['params']}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
