"""Task 6O sections 10-11 — Stage O1 (Overfit20) and Stage O2 (MiniTrain1000 -> MiniVal240).

Trains the two causal-decomposition variants on the **byte-identical frozen Task 6N packs** with the
**exact Task 6N N1/N2 settings**:

* **N-B3** — visual + `P_rel` + relation embedding (145-channel fusion), **no direct reference channel**;
* **N-B4** — `P_rel` projected 1 -> 128 + relation embedding (144-channel fusion), **no visual feature and
  no direct reference channel** (geometry-only capacity control).

O1 gate (section 10): B3 must reach mIoU >= 0.85 and Dice >= 0.90, otherwise the task stops with
`FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE`. B4 has no gate. O2 runs only if B3 passes O1.

The frozen B0/B1/B2 variants, the packs, the field and the evaluator are untouched. The test split is
never read.

    python scripts/task6o_train.py --stage o1
    python scripts/task6o_train.py --stage o2
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

from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    TASK6O_VARIANTS,
    VARIANT_LABELS,
    VARIANT_USES_VISUAL,
    load_frozen_sam2_encoder,
    read_pack,
    task6n_loss,
    upsampled_logits,
)
from scripts.task6n_train import (  # noqa: E402
    FEATURE_ROOT,
    PACK_ROOT,
    SEED,
    MaskStore,
    _aggregate,
    _build_batch,
    _per_sample_metrics,
    make_optimizer,
    paired_own_cross,
    seed_everything,
)

EVAL = REPO_ROOT / "evaluation"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "task6o" / "checkpoints"
OUT_O1 = EVAL / "task6o_overfit20.json"
OUT_O2 = EVAL / "task6o_o2_training.json"


def variant_forward(model: RelationMaskDecoder, batch):
    """Call the decoder with exactly the inputs its variant is allowed to receive.

    B4 (geometry-only) is called with ``visual=None`` and ``mask_ref_down=None`` so the RGB-derived
    feature is never even handed to the module; B3 is called without ``mask_ref_down``.
    """

    if not VARIANT_USES_VISUAL[model.variant]:
        return model(None, batch.relation_index, None, batch.field)
    return model(batch.visual, batch.relation_index, batch.mask_ref_down, batch.field)


@torch.no_grad()
def evaluate_variant(model, samples, store, masks, device, batch_size=8) -> dict:
    """Identical metric math to the frozen Task 6N `evaluate`, with variant-appropriate inputs."""

    model.eval()
    rows = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        batch = _build_batch(chunk, store, masks, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = variant_forward(model, batch)
        rows.extend(_per_sample_metrics(logits, batch))
    return _aggregate(rows)


def paired_variant(model, pack_path: Path, samples, store, masks, device) -> dict:
    """Own-vs-cross on the Overfit20 counterfactual pairs, variant-appropriate forward."""

    payload = json.loads(Path(pack_path).read_text(encoding="utf-8"))
    pairs = payload.get("detail", {}).get("counterfactual_pairs") or []
    if not pairs:
        return {"pairs": 0, "available": False}
    by_id = {sample.sample_id: sample for sample in samples}
    rows = []
    for pair in pairs:
        members = [by_id.get(pair["a"]["sample_id"]), by_id.get(pair["b"]["sample_id"])]
        if any(member is None for member in members):
            continue
        batch = _build_batch(members, store, masks, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=device != "cpu"):
            logits = variant_forward(model, batch)
        upsampled = upsampled_logits(logits, batch.target_size) > 0.0
        own, cross = [], []
        for index in range(len(members)):
            own_target = batch.target[index] > 0.5
            own.append(float((upsampled[index] & own_target).sum())
                       / float((upsampled[index] | own_target).sum() + 1e-6))
            other = members[1 - index]
            other_mask = torch.as_tensor(
                masks.mask(other.tile_id, other.target_source_feature_id), dtype=torch.bool,
                device=upsampled.device,
            )
            cross.append(float((upsampled[index] & other_mask).sum())
                         / float((upsampled[index] | other_mask).sum() + 1e-6))
        rows.append({"tile_id": members[0].tile_id, "own": own, "cross": cross,
                     "passes": bool(own[0] > cross[0] and own[1] > cross[1])})
    passed = sum(1 for row in rows if row["passes"])
    mean_own = float(np.mean([value for row in rows for value in row["own"]]))
    mean_cross = float(np.mean([value for row in rows for value in row["cross"]]))
    return {
        "pairs": len(rows), "available": True, "passed": passed,
        "mean_own_iou": mean_own, "mean_cross_iou": mean_cross,
        "own_cross_margin": mean_own - mean_cross, "rows": rows,
    }


def run_o1(args) -> int:
    config = {
        "stage": "O1",
        "variants": list(TASK6O_VARIANTS),
        "optimizer": "AdamW",
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "max_steps": 1200,
        "batch_size": 4,
        "scheduler": "none",
        "augmentation": "none",
        "seed": SEED,
        "amp": "bfloat16 autocast, identical to Task 6N",
        "evaluate_every_steps": 100,
        "identical_to": "Task 6N N1 (section 14)",
        "gate": {"train_miou_min": 0.85, "train_dice_min": 0.90, "applies_to": "B3"},
    }
    started = time.time()
    pack_path = PACK_ROOT / "overfit20.json"
    samples = read_pack(pack_path)
    masks = MaskStore()
    encoder, sam_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    print(f"[6o.o1] Overfit20 records {len(samples)}; variants {TASK6O_VARIANTS}", flush=True)

    results = {}
    for variant in TASK6O_VARIANTS:
        seed_everything(SEED)
        model = RelationMaskDecoder(variant, DecoderConfig()).to(args.device)
        optimizer = make_optimizer(model, config["lr"], config["weight_decay"])
        parameters = model.parameter_report()
        order = list(range(len(samples)))
        generator = np.random.default_rng(SEED)
        history = []
        best = {"miou": -1.0, "dice": -1.0, "step": 0}
        step = 0
        if args.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        variant_started = time.time()
        while step < config["max_steps"]:
            model.train()
            generator.shuffle(order)
            chunk = [samples[index] for index in order[: config["batch_size"]]]
            batch = _build_batch(chunk, store, masks, args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device != "cpu"):
                logits = variant_forward(model, batch)
            losses = task6n_loss(upsampled_logits(logits, batch.target_size), batch.target)
            losses["loss"].backward()
            optimizer.step()
            step += 1
            if step % config["evaluate_every_steps"] == 0 or step == config["max_steps"]:
                metrics = evaluate_variant(model, samples, store, masks, args.device, config["batch_size"])
                history.append({"step": step, "loss": float(losses["loss"].detach()),
                                "miou": metrics["miou"], "dice": metrics["dice"]})
                if metrics["miou"] > best["miou"]:
                    best = {"miou": metrics["miou"], "dice": metrics["dice"], "step": step}
                print(f"[6o.o1] {variant} step {step}: loss {float(losses['loss'].detach()):.4f} "
                      f"mIoU {metrics['miou']:.4f} Dice {metrics['dice']:.4f}", flush=True)

        final = evaluate_variant(model, samples, store, masks, args.device, config["batch_size"])
        pairing = paired_variant(model, pack_path, samples, store, masks, args.device)
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_ROOT / f"o1_{variant}.pt"
        torch.save({"variant": variant, "state_dict": model.state_dict(), "config": config}, checkpoint_path)
        results[variant] = {
            "label": VARIANT_LABELS[variant],
            "parameters": parameters,
            "history": history,
            "best": best,
            "final": {key: final[key] for key in ("miou", "dice", "precision_at_0_5", "records")},
            "final_per_relation": final["per_relation"],
            "paired_own_cross": {key: value for key, value in pairing.items() if key != "rows"},
            "wall_seconds": round(time.time() - variant_started, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if args.device != "cpu" else None,
            "checkpoint": {"path": str(checkpoint_path),
                           "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
        }
        print(f"[6o.o1] {variant} done: best mIoU {best['miou']:.4f} (step {best['step']}), "
              f"final mIoU {final['miou']:.4f} Dice {final['dice']:.4f}", flush=True)

    b3_best = results["B3"]["best"]
    b3_final = results["B3"]["final"]
    gate = {
        "applies_to": "B3",
        "miou_min": config["gate"]["train_miou_min"],
        "dice_min": config["gate"]["train_dice_min"],
        "b3_best_miou": b3_best["miou"],
        "b3_best_dice": b3_best["dice"],
        "b3_final_miou": b3_final["miou"],
        "b3_final_dice": b3_final["dice"],
        "b4_best_miou": results["B4"]["best"]["miou"],
        "b4_best_dice": results["B4"]["best"]["dice"],
        "b4_has_no_gate": True,
    }
    gate["passed"] = bool(
        max(gate["b3_best_miou"], gate["b3_final_miou"]) >= gate["miou_min"]
        and max(gate["b3_best_dice"], gate["b3_final_dice"]) >= gate["dice_min"]
    )
    payload = {
        "_doc": (
            "Task 6O section 10. Stage O1: B3 (visual + field + relation, no direct reference channel) "
            "and B4 (field + relation only, no visual) trained separately on the byte-identical frozen "
            "Task 6N Overfit20 with the exact Task 6N N1 settings. The gate applies to B3 only."
        ),
        "task": "6O",
        "stage": "O1",
        "reference_source": "oracle_native_gt",
        "config": config,
        "pack": {"path": str(pack_path), "count": len(samples),
                 "sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest()},
        "visual": store.provenance(),
        "variants": results,
        "gate": gate,
        "verdict": "O1_PASS" if gate["passed"] else "FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_O1, payload)
    print(f"[6o.o1] gate {gate['passed']} (B3 mIoU {gate['b3_best_miou']:.4f} / Dice "
          f"{gate['b3_best_dice']:.4f}) -> {payload['verdict']}", flush=True)
    return 0 if gate["passed"] else 2


def run_o2(args) -> int:
    config = {
        "stage": "O2",
        "variants": list(TASK6O_VARIANTS),
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
        "amp": "bfloat16 autocast, identical to Task 6N",
        "fresh_initialisation": True,
        "identical_to": "Task 6N N2 (section 15)",
    }
    started = time.time()
    train_path = PACK_ROOT / "mini_train_1000.json"
    val_path = PACK_ROOT / "mini_val_240.json"
    train_samples = read_pack(train_path)
    val_samples = read_pack(val_path)
    masks = MaskStore()
    encoder, sam_report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    print(f"[6o.o2] train {len(train_samples)} / val {len(val_samples)}", flush=True)

    results = {}
    for variant in TASK6O_VARIANTS:
        seed_everything(SEED)
        model = RelationMaskDecoder(variant, DecoderConfig()).to(args.device)
        optimizer = make_optimizer(model, config["lr"], config["weight_decay"])
        parameters = model.parameter_report()
        if args.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=args.device != "cpu"):
                    logits = variant_forward(model, batch)
                loss = task6n_loss(upsampled_logits(logits, batch.target_size), batch.target)["loss"]
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            val = evaluate_variant(model, val_samples, store, masks, args.device, config["batch_size"])
            epochs.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                           "val_miou": val["miou"], "val_dice": val["dice"],
                           "val_precision_at_0_5": val["precision_at_0_5"]})
            print(f"[6o.o2] {variant} epoch {epoch}: loss {np.mean(losses):.4f} "
                  f"val mIoU {val['miou']:.4f} Dice {val['dice']:.4f}", flush=True)
            if val["miou"] > best["miou"]:
                best = {"miou": val["miou"], "dice": val["dice"], "epoch": epoch,
                        "state": {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if stale >= config["early_stopping_patience"]:
                    print(f"[6o.o2] {variant}: early stop at epoch {epoch} "
                          f"(patience {config['early_stopping_patience']})", flush=True)
                    break

        if best["state"] is not None:
            model.load_state_dict(best["state"])
        final = evaluate_variant(model, val_samples, store, masks, args.device, config["batch_size"])
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_ROOT / f"o2_{variant}.pt"
        torch.save({"variant": variant, "state_dict": model.state_dict(), "config": config,
                    "best_epoch": best["epoch"]}, checkpoint_path)
        results[variant] = {
            "label": VARIANT_LABELS[variant],
            "parameters": parameters,
            "epochs": epochs,
            "epochs_run": len(epochs),
            "best": {"epoch": best["epoch"], "miou": best["miou"], "dice": best["dice"]},
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
        print(f"[6o.o2] {variant} selected epoch {best['epoch']} val mIoU {final['miou']:.4f}", flush=True)

    payload = {
        "_doc": (
            "Task 6O section 11. Stage O2: B3 and B4 trained from fresh initialisation on the frozen "
            "Task 6N MiniTrain1000 and selected on MiniVal240 mIoU with the exact Task 6N N2 settings."
        ),
        "task": "6O",
        "stage": "O2",
        "reference_source": "oracle_native_gt",
        "config": config,
        "packs": {
            "train": {"path": str(train_path), "count": len(train_samples),
                      "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()},
            "val": {"path": str(val_path), "count": len(val_samples),
                    "sha256": hashlib.sha256(val_path.read_bytes()).hexdigest()},
        },
        "visual": store.provenance(),
        "variants": results,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_O2, payload)
    print(f"[6o.o2] mIoU B3 {results['B3']['selected_model']['miou']:.6f} | "
          f"B4 {results['B4']['selected_model']['miou']:.6f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("o1", "o2"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return run_o1(args) if args.stage == "o1" else run_o2(args)


if __name__ == "__main__":
    raise SystemExit(main())
