"""Task 7D Parts G/J/K — D-B0 reproduction, Stage D1 (Overfit20) and Stage D2 (MiniTrain1200 -> MiniVal240).

Part G (section 16): re-run the frozen Task 6Z Z-B3 decoder on the exact Z-MiniVal240 / Z-PairedVal20 packs and
require exact reproduction (mIoU/Dice/margin deltas <= 1e-6, Paired exactly 15/20). Failure stops the task
with `TASK6Z_BASELINE_REPRODUCTION_FAIL`.

Stage D1 (section 23): train D-B1/D-B2/D-B3/D-B4 from fresh initialization on Z-Overfit20 — AdamW, lr 1e-3,
weight decay 1e-4, batch 4, max 1200 steps, no scheduler, no augmentation, seed 20261001, bfloat16 AMP,
evaluate every 100 steps. D-B2 gate: mIoU >= 0.85 and Dice >= 0.90, otherwise STOP
`GLOBAL_COMPETITION_NOT_LEARNABLE`.

Stage D2 (section 25): only after the D-B2 gate passes — AdamW, lr 3e-4, weight decay 1e-4, batch 8, max 25
epochs, early-stopping patience 5, checkpoint selection by MiniVal240 mIoU, seed 20261001, same AMP.

The loss is exactly `BCEWithLogitsLoss + DiceLoss` (the frozen Task 6N `task6n_loss`); the competition map is
never supervised. GT targets and the oracle reference are labels only.

    python scripts/task7d_train.py --stage baseline
    python scripts/task7d_train.py --stage d1
    python scripts/task7d_train.py --stage d2
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
    upsampled_logits,
)
from buildreasonseg_mvp.task7d_data import (  # noqa: E402
    CHECKPOINT_ROOT,
    PACK_MANIFEST,
    PACK_ROOT,
    TRAINABLE_VARIANTS,
    build_batch,
    forward_variant,
    load_pack,
)
from buildreasonseg_mvp.task7d_global_competition_decoder import (  # noqa: E402
    COMPETITION_TEMPERATURE,
    SPATIAL_TOKENS,
    GlobalCompetitionDecoder,
)
from scripts.task6n_train import (  # noqa: E402
    FEATURE_ROOT,
    MaskStore,
    _aggregate,
    _per_sample_metrics,
    make_optimizer,
    seed_everything,
)

EVAL = REPO_ROOT / "evaluation"
OUT_BASELINE = EVAL / "task7d_baseline_reproduction.json"
OUT_D1 = EVAL / "task7d_overfit20.json"
OUT_D2 = EVAL / "task7d_training.json"
TASK6Z_VAL = EVAL / "task6z_mini_val.json"
TASK6Z_PAIRED = EVAL / "task6z_paired_val.json"
TASK6Z_TRAINING = EVAL / "task6z_training.json"
Z_B3_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6z" / "zb3_minitrain1200.pt"
TOLERANCE = 1.0e-6
D1 = {"lr": 1.0e-3, "weight_decay": 1.0e-4, "batch": 4, "max_steps": 1200, "eval_every": 100,
      "gate_miou": 0.85, "gate_dice": 0.90}
D2 = {"lr": 3.0e-4, "weight_decay": 1.0e-4, "batch": 8, "max_epochs": 25, "patience": 5}
SEED = 20261001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_packs() -> dict:
    manifest = json.loads(PACK_MANIFEST.read_text(encoding="utf-8"))
    report = {}
    for name in ("z_overfit20", "z_mini_train_1200", "z_mini_val_240", "z_paired_val20"):
        path = PACK_ROOT / f"{name}.json"
        digest = sha256_file(path)
        expected = manifest["packs"][name]["sha256"]
        report[name] = {"path": str(path), "sha256": digest, "expected_sha256": expected,
                        "matches": digest == expected}
    return report


@torch.no_grad()
def evaluate_variant(model, records: list[dict], store, masks, device,
                     precomputed: dict[str, dict], batch_size: int = 8) -> dict:
    import torch

    model.eval()
    rows = []
    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        batch = build_batch(chunk, store, masks, device, precomputed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = forward_variant(model, batch)
        rows.extend(_per_sample_metrics(logits, batch))
    return _aggregate(rows)


def train_variant(variant: str, train_records: list[dict], selection_records: list[dict], store, masks,
                  device: str, precomputed: dict, *, lr: float, weight_decay: float, batch_size: int,
                  max_steps: int | None, max_epochs: int | None, patience: int | None,
                  eval_every: int | None, checkpoint_path: Path) -> dict:
    import torch

    seed_everything(SEED)
    model = GlobalCompetitionDecoder(variant).to(device)
    optimizer = make_optimizer(model, lr, weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device != "cpu")
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = []
    best = {"key": -1.0, "step": 0, "epoch": 0, "metrics": None, "state": None}
    step, epoch, stale, stop = 0, 0, 0, False
    while not stop:
        epoch += 1
        generator = np.random.default_rng(SEED + epoch)
        order = generator.permutation(len(train_records))
        for start in range(0, len(order), batch_size):
            chunk = [train_records[int(value)] for value in order[start: start + batch_size]]
            batch = build_batch(chunk, store, masks, device, precomputed)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                logits = forward_variant(model, batch)
                losses = task6n_loss(upsampled_logits(logits, batch.target.shape[-2:]),
                                     batch.target.unsqueeze(1))
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if eval_every is not None and step % eval_every == 0:
                metrics = evaluate_variant(model, train_records, store, masks, device, precomputed)
                history.append({"step": step, "epoch": epoch, "miou": metrics["miou"],
                                "dice": metrics["dice"], "loss": float(losses["loss"].detach())})
                print(f"[7d.train] {variant} step {step}: mIoU {metrics['miou']:.4f} "
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
            print(f"[7d.train] {variant} epoch {epoch}: MiniVal mIoU {metrics['miou']:.4f} "
                  f"Dice {metrics['dice']:.4f}", flush=True)
            if metrics["miou"] > best["key"]:
                best = {"key": metrics["miou"], "step": step, "epoch": epoch, "metrics": metrics,
                        "state": {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    print(f"[7d.train] {variant} early stop at epoch {epoch}", flush=True)
                    stop = True
        if max_epochs is not None and epoch >= max_epochs:
            stop = True
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    final = evaluate_variant(model, selection_records or train_records, store, masks, device,
                             precomputed)
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
        "final": final,
        "steps": step, "epochs": epoch,
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if device != "cpu" else None,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                       "bytes": checkpoint_path.stat().st_size, "committed": False},
    }


def run_baseline(device: str) -> int:
    """Section 16: exact frozen Z-B3 reproduction through the Task 6Z evaluation path."""

    from scripts.task6z_evaluate import load_variant, predict

    started = time.time()
    task6z_training = json.loads(TASK6Z_TRAINING.read_text(encoding="utf-8"))
    expected_sha = task6z_training["results"]["Z-B3"]["checkpoint"]["sha256"]
    if not Z_B3_CHECKPOINT.is_file() or sha256_file(Z_B3_CHECKPOINT) != expected_sha:
        write_json(OUT_BASELINE, {"_doc": "Task 7D section 16.", "task": "7D",
                                  "verdict": "INVALID_EXPERIMENT",
                                  "reason": "Z-B3 checkpoint unavailable"})
        return 2
    encoder, _report = load_frozen_sam2_encoder(device=device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=device)
    masks = MaskStore()
    model = load_variant("Z-B3", device)
    records = load_pack("z_mini_val_240")
    precomputed: dict[str, dict] = {}
    predictions = predict(model, records, store, masks, device, precomputed, batch_size=8)
    rows = []
    for record, prediction in zip(records, predictions):
        truth = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                           dtype=bool)
        intersection = float((prediction & truth).sum())
        rows.append({"miou": (intersection + TOLERANCE) / (float((prediction | truth).sum())
                                                           + TOLERANCE),
                     "dice": (2.0 * intersection + TOLERANCE)
                     / (float(prediction.sum()) + float(truth.sum()) + TOLERANCE),
                     "precision_at_0_5": (intersection + TOLERANCE)
                     / (float(prediction.sum()) + TOLERANCE)})
    miou = float(np.mean([row["miou"] for row in rows]))
    dice = float(np.mean([row["dice"] for row in rows]))

    paired_records = load_pack("z_paired_val20")
    pairs = [{"tile_id": paired_records[index]["tile_id"], "a": paired_records[index],
              "b": paired_records[index + 1]} for index in range(0, len(paired_records), 2)]
    passed = 0
    own_values, cross_values = [], []
    for entry in pairs:
        members = [entry["a"], entry["b"]]
        pair_predictions = predict(model, members, store, masks, device, precomputed, batch_size=2)
        own, cross = [], []
        for index, member in enumerate(members):
            own_mask = np.asarray(masks.mask(member["tile_id"], member["target_source_feature_id"]),
                                 dtype=bool)
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other["tile_id"], other["target_source_feature_id"]),
                                   dtype=bool)
            own.append(float((pair_predictions[index] & own_mask).sum())
                       / (float((pair_predictions[index] | own_mask).sum()) + TOLERANCE))
            cross.append(float((pair_predictions[index] & other_mask).sum())
                         / (float((pair_predictions[index] | other_mask).sum()) + TOLERANCE))
        if own[0] > cross[0] and own[1] > cross[1]:
            passed += 1
        own_values.extend(own)
        cross_values.extend(cross)
    margin = float(np.mean(own_values) - np.mean(cross_values))

    frozen = json.loads(TASK6Z_VAL.read_text(encoding="utf-8"))["results"]["Z-B3"]["overall"]
    frozen_paired = json.loads(TASK6Z_PAIRED.read_text(encoding="utf-8"))["systems"]["Z-B3"]
    deltas = {"miou": abs(miou - frozen["miou"]), "dice": abs(dice - frozen["dice"]),
              "margin": abs(margin - frozen_paired["own_cross_margin"])}
    ok = bool(all(value <= TOLERANCE for value in deltas.values()) and passed == 15)
    write_json(OUT_BASELINE, {
        "_doc": ("Task 7D section 16. D-B0: frozen Task 6Z Z-B3 baseline reproduction through the frozen "
                 "Task 6Z evaluation path (same model, batching, pack order and oracle reference)."),
        "task": "7D", "stage": "G-baseline-reproduction",
        "checkpoint": {"path": str(Z_B3_CHECKPOINT), "sha256": sha256_file(Z_B3_CHECKPOINT),
                       "expected_sha256": expected_sha, "variant": "Z-B3", "retrained": False},
        "recomputed": {"miou": miou, "dice": dice, "paired_passed": passed, "pairs": len(pairs),
                       "own_cross_margin": margin},
        "frozen_task6z": {"miou": frozen["miou"], "dice": frozen["dice"],
                          "paired_passed": frozen_paired["passed"],
                          "own_cross_margin": frozen_paired["own_cross_margin"]},
        "deltas": deltas, "tolerance": TOLERANCE, "reproduction_passed": ok,
        "verdict": "TASK6Z_BASELINE_REPRODUCTION_PASS" if ok else "TASK6Z_BASELINE_REPRODUCTION_FAIL",
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7d.base] mIoU {miou:.10f} (Δ{deltas['miou']:.2e}) Dice {dice:.10f} "
          f"(Δ{deltas['dice']:.2e}) paired {passed}/20 margin {margin:.10f} "
          f"(Δ{deltas['margin']:.2e}) -> {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "d1", "d2"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    packs = verify_packs()
    if not all(entry["matches"] for entry in packs.values()):
        write_json(OUT_BASELINE if args.stage == "baseline" else OUT_D1,
                   {"_doc": "Task 7D section 6.", "task": "7D", "verdict": "TASK6Z_PACK_MISMATCH",
                    "packs": packs})
        print("[7d] STOP TASK6Z_PACK_MISMATCH", flush=True)
        return 2
    if args.stage == "baseline":
        return run_baseline(args.device)

    import torch  # noqa: F401

    encoder, encoder_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()

    if args.stage == "d1":
        records = load_pack("z_overfit20")
        precomputed: dict[str, dict] = {}
        results = {}
        for variant in TRAINABLE_VARIANTS:
            results[variant] = train_variant(
                variant, records, records, store, masks, args.device, precomputed, lr=D1["lr"],
                weight_decay=D1["weight_decay"], batch_size=D1["batch"], max_steps=D1["max_steps"],
                max_epochs=None, patience=None, eval_every=D1["eval_every"],
                checkpoint_path=CHECKPOINT_ROOT / f"{variant.lower().replace('-', '')}_overfit20.pt")
        primary = results["D-B2"]
        gate = {"miou_min": D1["gate_miou"], "dice_min": D1["gate_dice"],
                "measured_miou": primary["best"]["miou"], "measured_dice": primary["best"]["dice"],
                "applies_to": "D-B2",
                "passed": bool(primary["best"]["miou"] >= D1["gate_miou"]
                               and primary["best"]["dice"] >= D1["gate_dice"])}
        write_json(OUT_D1, {
            "_doc": ("Task 7D section 23. Stage D1 Overfit20: D-B1/D-B2/D-B3/D-B4 trained from fresh "
                     "initialization on the exact frozen Z-Overfit20 pack with the section-23 protocol. "
                     "D-B2 gate: mIoU >= 0.85 and Dice >= 0.90."),
            "task": "7D", "stage": "J-overfit20", "seed": SEED,
            "packs": packs, "pack": {"name": "z_overfit20", "records": len(records)},
            "training": D1, "results": results, "gate": gate,
            "d_b2_overfit_passed": gate["passed"],
            "verdict": ("GLOBAL_COMPETITION_OVERFIT_PASS" if gate["passed"]
                        else "GLOBAL_COMPETITION_NOT_LEARNABLE"),
            "constants": {"competition_temperature": COMPETITION_TEMPERATURE,
                          "spatial_tokens": SPATIAL_TOKENS},
            "encoder": encoder_report, "training_performed": True, "test_split_used": False,
            "runtime_seconds": round(time.time() - started, 1),
        })
        print(f"[7d.d1] D-B2 mIoU {gate['measured_miou']:.4f} Dice {gate['measured_dice']:.4f} -> "
              f"{'PASS' if gate['passed'] else 'FAIL'}", flush=True)
        return 0 if gate["passed"] else 3

    d1 = json.loads(OUT_D1.read_text(encoding="utf-8")) if OUT_D1.is_file() else None
    if d1 is None or not d1.get("d_b2_overfit_passed"):
        write_json(OUT_D2, {"_doc": "Task 7D section 25.", "task": "7D",
                            "verdict": "GLOBAL_COMPETITION_NOT_LEARNABLE",
                            "reason": "the D-B2 Overfit20 gate did not pass; D2 never runs"})
        print("[7d.d2] STOP GLOBAL_COMPETITION_NOT_LEARNABLE", flush=True)
        return 3
    train_records = load_pack("z_mini_train_1200")
    val_records = load_pack("z_mini_val_240")
    precomputed = {}
    results = {}
    for variant in TRAINABLE_VARIANTS:
        results[variant] = train_variant(
            variant, train_records, val_records, store, masks, args.device, precomputed, lr=D2["lr"],
            weight_decay=D2["weight_decay"], batch_size=D2["batch"], max_steps=None,
            max_epochs=D2["max_epochs"], patience=D2["patience"], eval_every=None,
            checkpoint_path=CHECKPOINT_ROOT / f"{variant.lower().replace('-', '')}_minitrain1200.pt")
    write_json(OUT_D2, {
        "_doc": ("Task 7D section 25. Stage D2 MiniTrain1200 -> MiniVal240: D-B1/D-B2/D-B3/D-B4 trained "
                 "from fresh initialization with checkpoint selection by MiniVal240 mIoU and patience 5."),
        "task": "7D", "stage": "K-minitrain1200", "seed": SEED, "packs": packs,
        "packs_used": {"train": {"name": "z_mini_train_1200", "records": len(train_records)},
                       "selection": {"name": "z_mini_val_240", "records": len(val_records)}},
        "training": D2, "results": results, "selection_metric": "MiniVal240 mIoU",
        "encoder": encoder_report, "training_performed": True, "test_split_used": False,
        "verdict": "GLOBAL_COMPETITION_TRAINING_COMPLETE",
        "runtime_seconds": round(time.time() - started, 1),
    })
    for variant in TRAINABLE_VARIANTS:
        entry = results[variant]
        print(f"[7d.d2] {variant}: best MiniVal mIoU {entry['best']['miou']:.4f} Dice "
              f"{entry['best']['dice']:.4f} epoch {entry['best']['epoch']} params {entry['params']}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
