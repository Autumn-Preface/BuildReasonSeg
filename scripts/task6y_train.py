"""Task 6Y Parts H-I — Stage Y1 (Overfit20) and Stage Y2 (MiniTrain1000 -> MiniVal240).

Exactly the four predeclared variants are trained separately from fresh initialization on the frozen
nearest-only packs, with the frozen SAM2 visual representation and the frozen oracle-reference
`NearestBoundaryField v0.1`:

* Y1 (section 21): AdamW, lr 1e-3, weight_decay 1e-4, batch 4, max 1200 steps, no scheduler, no
  augmentation, seed 20260930, bfloat16 AMP, evaluate every 100 steps. B2 gate: mIoU >= 0.85 and
  Dice >= 0.90, otherwise STOP `NEAREST_FIELD_NOT_LEARNABLE` (Y2 then never runs).
* Y2 (section 22): AdamW, lr 3e-4, weight_decay 1e-4, batch 8, max 25 epochs, early-stopping patience 5,
  selection metric = MiniVal240 mIoU, seed 20260930, no scheduler, no augmentation, same AMP.

The loss is exactly `BCEWithLogitsLoss + DiceLoss` (the frozen Task 6N `task6n_loss`); the GT target mask is
used only as the training label and never as an input. No test split, no GRCL, no ranking/auxiliary loss.

    python scripts/task6y_train.py --stage o1
    python scripts/task6y_train.py --stage o2
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
from buildreasonseg_mvp.nearest_boundary_field import (  # noqa: E402
    SIGMA_DIAG,
    field_to_decoder_resolution,
    nearest_boundary_field_512,
    reference_area_to_decoder_resolution,
)
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
    upsampled_logits,
)
from buildreasonseg_mvp.task6y_nearest_decoder import (  # noqa: E402
    ALL_VARIANTS,
    NearestTargetDecoder,
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
from buildreasonseg_mvp.task6n_relation_decoder import task6n_loss  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6y" / "packs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task6y"
OUT_O1 = EVAL / "task6y_overfit20.json"
OUT_O2 = EVAL / "task6y_training.json"
SEED = 20260930
O1 = {"lr": 1.0e-3, "weight_decay": 1.0e-4, "batch": 4, "max_steps": 1200, "eval_every": 100,
      "gate_miou": 0.85, "gate_dice": 0.90}
O2 = {"lr": 3.0e-4, "weight_decay": 1.0e-4, "batch": 8, "max_epochs": 25, "patience": 5}


@dataclass
class NearestBatch:
    """Everything one forward/loss step needs; the GT target is a label only."""

    visual: torch.Tensor | None
    reference_mask: torch.Tensor | None
    nearest_field: torch.Tensor | None
    target: torch.Tensor
    sample_ids: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    target_size: tuple[int, int] = (512, 512)


def load_pack(name: str) -> list[dict]:
    payload = json.loads((PACK_ROOT / f"{name}.json").read_text(encoding="utf-8"))
    return payload["records"]


def image_path_of(record: dict) -> Path:
    path = Path(record["image_path"])
    return path if path.is_absolute() else REPO_ROOT / path


def build_batch(records: list[dict], store: FrozenFeatureStore, masks: MaskStore, device: str,
                precomputed: dict[str, dict]) -> NearestBatch:
    visuals, fields, references, targets = [], [], [], []
    for record in records:
        sample_id = record["sample_id"]
        entry = precomputed.get(sample_id)
        if entry is None:
            reference_mask = np.asarray(masks.mask(record["tile_id"],
                                                   record["reference_source_feature_id"]),
                                        dtype=bool)
            entry = {
                "nearest_field": field_to_decoder_resolution(
                    nearest_boundary_field_512(reference_mask)),
                "reference_mask": reference_area_to_decoder_resolution(reference_mask),
            }
            precomputed[sample_id] = entry
        fields.append(entry["nearest_field"][None])
        references.append(entry["reference_mask"][None])
        target = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                            dtype=np.float32)
        targets.append(torch.as_tensor(target))
        visual = store.get(record["tile_id"], image_path_of(record))
        visuals.append(visual.float())
    return NearestBatch(
        visual=torch.stack(visuals).to(device),
        reference_mask=torch.stack(references).to(device),
        nearest_field=torch.stack(fields).to(device),
        target=torch.stack(targets).to(device),
        sample_ids=[record["sample_id"] for record in records],
        programs=[record["program_id"] for record in records],
        target_size=(int(targets[0].shape[0]), int(targets[0].shape[1])),
    )


def forward_variant(model: NearestTargetDecoder, batch: NearestBatch) -> torch.Tensor:
    """Hand each variant exactly the inputs it is allowed to see."""

    return model(batch.visual if model.uses_visual else None,
                 batch.reference_mask if model.uses_reference_mask else None,
                 batch.nearest_field if model.uses_nearest_field else None)


@torch.no_grad()
def evaluate_variant(model: NearestTargetDecoder, records: list[dict], store, masks, device,
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


def train_one(variant: str, train_records: list[dict], store, masks, device, precomputed,
              *, lr: float, weight_decay: float, batch_size: int, max_steps: int | None,
              max_epochs: int | None, patience: int | None, eval_every: int | None,
              selection_records: list[dict], checkpoint_path: Path) -> dict:
    seed_everything(SEED)
    model = NearestTargetDecoder(variant).to(device)
    optimizer = make_optimizer(model, lr, weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = []
    best = {"key": -1.0, "step": 0, "epoch": 0, "metrics": None,
            "state": None}
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
                upsampled = upsampled_logits(logits, batch.target_size)
                losses = task6n_loss(upsampled, batch.target.unsqueeze(1))
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if eval_every is not None and step % eval_every == 0:
                metrics = evaluate_variant(model, train_records, store, masks, device, precomputed)
                history.append({"step": step, "epoch": epoch, "miou": metrics["miou"],
                                "dice": metrics["dice"], "loss": float(losses["loss"].detach())})
                print(f"[6y.train] {variant} step {step}: mIoU {metrics['miou']:.4f} "
                      f"Dice {metrics['dice']:.4f}", flush=True)
                if metrics["miou"] > best["key"]:
                    best = {"key": metrics["miou"], "step": step, "epoch": epoch,
                            "metrics": metrics,
                            "state": {key: value.detach().cpu().clone()
                                      for key, value in model.state_dict().items()}}
            if max_steps is not None and step >= max_steps:
                stop = True
                break
        if max_steps is None and selection_records:
            metrics = evaluate_variant(model, selection_records, store, masks, device, precomputed)
            history.append({"step": step, "epoch": epoch, "miou": metrics["miou"],
                            "dice": metrics["dice"], "loss": None})
            print(f"[6y.train] {variant} epoch {epoch}: MiniVal mIoU {metrics['miou']:.4f} "
                  f"Dice {metrics['dice']:.4f}", flush=True)
            if metrics["miou"] > best["key"]:
                best = {"key": metrics["miou"], "step": step, "epoch": epoch, "metrics": metrics,
                        "state": {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    print(f"[6y.train] {variant} early stop at epoch {epoch}", flush=True)
                    stop = True
        if max_epochs is not None and epoch >= max_epochs:
            stop = True
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    final = evaluate_variant(model, selection_records or train_records, store, masks, device,
                            precomputed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"variant": variant, "state_dict": model.state_dict(), "seed": SEED,
                "lr": lr, "weight_decay": weight_decay, "batch": batch_size,
                "best_step": best["step"], "best_epoch": best["epoch"],
                "sigma_diag": SIGMA_DIAG}, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return {
        "variant": variant,
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
        "best": {"miou": best["metrics"]["miou"] if best["metrics"] else None,
                 "dice": best["metrics"]["dice"] if best["metrics"] else None,
                 "step": best["step"], "epoch": best["epoch"]},
        "final": final,
        "steps": step, "epochs": epoch,
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if device != "cpu" else None,
        "checkpoint": {"path": str(checkpoint_path), "sha256": digest,
                       "bytes": checkpoint_path.stat().st_size, "committed": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("o1", "o2"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    encoder, encoder_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()

    if args.stage == "o1":
        records = load_pack("y_overfit20")
        precomputed: dict[str, dict] = {}
        results = {}
        for variant in ALL_VARIANTS:
            results[variant] = train_one(
                variant, records, store, masks, args.device, precomputed,
                lr=O1["lr"], weight_decay=O1["weight_decay"], batch_size=O1["batch"],
                max_steps=O1["max_steps"], max_epochs=None, patience=None,
                eval_every=O1["eval_every"], selection_records=records,
                checkpoint_path=CHECKPOINT_ROOT / f"{variant.lower()}_overfit20.pt")
        b2 = results["Y-B2"]
        gate = {"miou_min": O1["gate_miou"], "dice_min": O1["gate_dice"],
                "measured_miou": b2["best"]["miou"], "measured_dice": b2["best"]["dice"],
                "passed": bool(b2["best"]["miou"] >= O1["gate_miou"]
                               and b2["best"]["dice"] >= O1["gate_dice"])}
        payload = {
            "_doc": (
                "Task 6Y section 21. Stage Y1 Overfit20: the four predeclared variants trained from fresh "
                "init on the frozen 20-record nearest pack (10 largest_to_nearest + 10 smallest_to_nearest) "
                "with the oracle reference, bfloat16 AMP and exactly BCE+Dice. B2 gate mIoU >= 0.85 and "
                "Dice >= 0.90."
            ),
            "task": "6Y", "stage": "Y1-overfit20", "seed": SEED,
            "pack": {"name": "y_overfit20", "records": len(records),
                     "by_program": {program: sum(1 for record in records
                                                 if record["program_id"] == program)
                                    for program in ("largest_to_nearest", "smallest_to_nearest")}},
            "training": O1, "variants": variant_report(), "results": results, "gate": gate,
            "b2_overfit_passed": gate["passed"],
            "verdict": ("NEAREST_OVERFIT_PASS" if gate["passed"] else "NEAREST_FIELD_NOT_LEARNABLE"),
            "encoder": encoder_report,
            "field": {"sigma_diag": SIGMA_DIAG, "reference": "oracle_native_gt"},
            "training_performed": True, "test_split_used": False,
            "runtime_seconds": round(time.time() - started, 1),
        }
        write_json(OUT_O1, payload)
        print(f"[6y.train] Y1 B2 mIoU {gate['measured_miou']:.4f} Dice {gate['measured_dice']:.4f} "
              f"-> {payload['verdict']}", flush=True)
        return 0 if gate["passed"] else 3

    o1 = json.loads(OUT_O1.read_text(encoding="utf-8")) if OUT_O1.is_file() else None
    if o1 is None or not o1.get("b2_overfit_passed"):
        write_json(OUT_O2, {"_doc": "Task 6Y section 22.", "task": "6Y",
                            "verdict": "NEAREST_FIELD_NOT_LEARNABLE",
                            "reason": "the B2 Overfit20 gate did not pass; Y2 never runs"})
        print("[6y.train] STOP NEAREST_FIELD_NOT_LEARNABLE (B2 overfit gate failed)", flush=True)
        return 3

    train_records = load_pack("y_mini_train_1000")
    val_records = load_pack("y_mini_val_240")
    precomputed = {}
    results = {}
    for variant in ALL_VARIANTS:
        results[variant] = train_one(
            variant, train_records, store, masks, args.device, precomputed,
            lr=O2["lr"], weight_decay=O2["weight_decay"], batch_size=O2["batch"],
            max_steps=None, max_epochs=O2["max_epochs"], patience=O2["patience"],
            eval_every=None, selection_records=val_records,
            checkpoint_path=CHECKPOINT_ROOT / f"{variant.lower()}_minitrain1000.pt")
    payload = {
        "_doc": (
            "Task 6Y section 22. Stage Y2 MiniTrain1000 -> MiniVal240: the four predeclared variants "
            "trained from fresh init with the oracle reference, selection by MiniVal240 mIoU, patience 5, "
            "and exactly BCE+Dice. No test split."
        ),
        "task": "6Y", "stage": "Y2-minitrain1000", "seed": SEED,
        "packs": {"train": {"name": "y_mini_train_1000", "records": len(train_records),
                            "by_program": {program: sum(1 for record in train_records
                                                        if record["program_id"] == program)
                                           for program in ("largest_to_nearest",
                                                           "smallest_to_nearest")}},
                  "selection": {"name": "y_mini_val_240", "records": len(val_records)}},
        "training": O2, "variants": variant_report(), "results": results,
        "selection_metric": "MiniVal240 mIoU",
        "encoder": encoder_report,
        "field": {"sigma_diag": SIGMA_DIAG, "reference": "oracle_native_gt"},
        "training_performed": True, "test_split_used": False,
        "verdict": "NEAREST_TRAINING_COMPLETE",
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_O2, payload)
    for variant in ALL_VARIANTS:
        entry = results[variant]
        print(f"[6y.train] {variant}: best MiniVal mIoU {entry['best']['miou']:.4f} Dice "
              f"{entry['best']['dice']:.4f} epoch {entry['best']['epoch']} params {entry['params']}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
