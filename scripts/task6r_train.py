"""Task 6R Parts E-G — controlled training of R1 (B3 + GRCL) and R2 (no-field + GRCL).

**R1** = exactly the Task 6O B3 architecture (frozen SAM2 visual feature + GeometricRelationField v0.2
from the oracle reference + relation embedding, **no** direct reference-mask decoder channel) trained
with `L_BCE + L_Dice + 0.5 * L_GRCL`.

**R2** = exactly the Task 6N B0 architecture (visual + relation embedding, **no** reference mask and
**no** field input) trained with the same objective. The oracle reference mask is used by GRCL and by
relation evaluation only — it is **not** an R2 model input.

Stage R1 (Overfit20): AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler / no
augmentation / seed 20260929 / evaluate every 100 steps. R1 gate: mIoU >= 0.85, Dice >= 0.90 and hard
relation accuracy >= 0.95, otherwise the task stops with `GRCL_OVERFIT_FAIL`. R2 has no gate.

Stage R2 (MiniTrain1000 -> MiniVal240): fresh initialisation, AdamW / lr 3e-4 / wd 1e-4 / batch 8 /
max 25 epochs / early stopping patience 5 / model selection = MiniVal240 mIoU / seed 20260929.

The test split is never read and the GT target is only ever a label.

    python scripts/task6r_train.py --stage overfit
    python scripts/task6r_train.py --stage mini
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

from buildreasonseg_mvp.geometric_relation_field_v02 import geometric_relation_field_v02  # noqa: E402
from buildreasonseg_mvp.grcl_directional import (  # noqa: E402
    ALPHA,
    LAMBDA_GRCL,
    TAU,
    grcl_directional,
    hard_relation_metrics,
    summarise_relation_metrics,
)
from buildreasonseg_mvp.losses import mask_bce_with_logits, mask_soft_dice  # noqa: E402
from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    VARIANT_USES_FIELD,
    VARIANT_USES_VISUAL,
    load_frozen_sam2_encoder,
    read_pack,
)
from scripts.task6n_train import (  # noqa: E402
    FEATURE_ROOT,
    PACK_ROOT,
    SEED,
    MaskStore,
    make_optimizer,
    seed_everything,
)

EVAL = REPO_ROOT / "evaluation"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "task6r" / "checkpoints"
OUT_OVERFIT = EVAL / "task6r_overfit20.json"
OUT_TRAINING = EVAL / "task6r_training.json"
FIELD_SIZE = (64, 64)
TARGET_SIZE = (512, 512)
RELATION_ORDER = ("left_of", "right_of", "above", "below")

#: The two controlled variants of Task 6R (architecture frozen; only the loss differs from 6O/6N).
VARIANTS = {
    "R1": {"decoder": "B3", "uses_field": True, "description": "B3 + GRCL (field-guided target decoder)"},
    "R2": {"decoder": "B0", "uses_field": False, "description": "no-field + GRCL (B0 control)"},
}


def model_for(variant: str, device: str) -> RelationMaskDecoder:
    return RelationMaskDecoder(VARIANTS[variant]["decoder"], DecoderConfig()).to(device)


def variant_forward(model, variant: str, visual, relation_index, field):
    """R1 receives the field; R2 must not receive any field or reference channel."""

    if VARIANTS[variant]["uses_field"]:
        return model(visual, relation_index, None, field)
    return model(visual, relation_index, None, None)


def build_batch(samples, store: FrozenFeatureStore, masks: MaskStore, device: str):
    """(visual, field, relation index, target 512, oracle reference 512) — target is a label only."""

    visuals, fields, indices, targets, references = [], [], [], [], []
    for sample in samples:
        visual = store.get(sample.tile_id, Path(sample.image_path)).float()
        reference = masks.mask(sample.tile_id, sample.reference_source_feature_id)
        reference_tensor = torch.as_tensor(np.asarray(reference, dtype=np.float32))
        field = geometric_relation_field_v02(reference_tensor, sample.relation, FIELD_SIZE)
        target = masks.mask(sample.tile_id, sample.target_source_feature_id)
        visuals.append(visual)
        fields.append(field[0])
        indices.append(RELATION_ORDER.index(sample.relation))
        targets.append(torch.as_tensor(np.asarray(target, dtype=np.float32)).unsqueeze(0))
        references.append(reference_tensor.unsqueeze(0))
    return (
        torch.stack(visuals).to(device),
        torch.stack(fields).to(device),
        torch.as_tensor(indices, dtype=torch.long, device=device),
        torch.stack(targets).to(device),
        torch.stack(references).to(device),
    )


def batch_loss(logits_64, target, reference, samples, *, with_grcl: bool = True) -> dict:
    """`L_BCE + L_Dice + 0.5 * L_GRCL` at the canonical 512x512 resolution."""

    logits_512 = F.interpolate(logits_64.float(), size=TARGET_SIZE, mode="bilinear", align_corners=False)
    bce = mask_bce_with_logits(logits_512, target)
    dice = mask_soft_dice(logits_512, target)
    total = bce + dice
    grcl_value = torch.zeros((), device=logits_64.device)
    if with_grcl:
        relations = [sample.relation for sample in samples]
        report = grcl_directional(logits_512, reference, relations, alpha=ALPHA, tau=TAU)
        grcl_value = report.loss
        total = total + LAMBDA_GRCL * report.loss
    return {"loss": total, "bce": bce.detach(), "dice": dice.detach(), "grcl": grcl_value.detach()}


@torch.no_grad()
def evaluate_variant(model, variant: str, samples, store, masks, device, batch_size: int = 8) -> dict:
    """Mask metrics + hard relation metrics + mean GRCL for one variant."""

    model.eval()
    rows = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        visual, field, indices, target, reference = build_batch(chunk, store, masks, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = variant_forward(model, variant, visual, indices, field)
        logits_512 = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                   align_corners=False)
        probabilities = torch.sigmoid(logits_512)
        prediction = probabilities > 0.5
        truth = target > 0.5
        grcl = grcl_directional(logits_512, reference, [sample.relation for sample in chunk],
                                alpha=ALPHA, tau=TAU)
        relation = hard_relation_metrics(probabilities, reference,
                                         [sample.relation for sample in chunk])
        for index, sample in enumerate(chunk):
            pred = prediction[index, 0]
            gt = truth[index, 0]
            intersection = float((pred & gt).sum())
            union = float((pred | gt).sum())
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "program_id": sample.program_id,
                    "relation": sample.relation,
                    "reference_family": sample.program_id.split("_", 1)[0],
                    "miou": (intersection + 1e-6) / (union + 1e-6),
                    "dice": (2.0 * intersection + 1e-6) / (float(pred.sum()) + float(gt.sum()) + 1e-6),
                    "precision_at_0_5": (intersection + 1e-6) / (float(pred.sum()) + 1e-6),
                    "correct": bool(relation["correct"][index]),
                    "empty": bool(relation["empty"][index]),
                    "margin": float(relation["margin"][index]),
                    "axis_violation": bool(relation["axis_violation"][index]),
                    "grcl": float(grcl.loss.detach()),
                }
            )
    if not rows:
        return {"records": 0}
    relation_summary = summarise_relation_metrics(rows)
    return {
        "records": len(rows),
        "miou": float(np.mean([row["miou"] for row in rows])),
        "dice": float(np.mean([row["dice"] for row in rows])),
        "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows])),
        "mean_grcl": float(np.mean([row["grcl"] for row in rows])),
        "relation_accuracy": relation_summary["relation_accuracy"],
        "relation_per_direction": relation_summary["per_relation"],
        "empty_predictions": relation_summary["empty_predictions"],
        "mean_positive_signed_margin": relation_summary["mean_positive_signed_margin"],
        "axis_violation_rate": relation_summary["axis_violation_rate"],
        "rows": rows,
    }


def paired_own_cross(model, variant: str, pack_path: Path, samples, store, masks, device) -> dict:
    """Own-vs-cross IoU on a pack's same-image counterfactual pairs (oracle reference)."""

    from buildreasonseg_mvp.task6n_relation_decoder import Task6NSample

    payload = json.loads(Path(pack_path).read_text(encoding="utf-8"))
    pairs = payload.get("detail", {}).get("counterfactual_pairs") or payload.get("pairs") or []
    if not pairs:
        return {"pairs": 0, "available": False}
    by_id = {sample.sample_id: sample for sample in samples}
    rows = []
    for pair in pairs:
        if "a" in pair and isinstance(pair["a"], dict) and "program_id" in pair["a"]:
            members = [Task6NSample(**{key: value for key, value in pair[side].items()
                                       if key in Task6NSample.__dataclass_fields__})
                       for side in ("a", "b")]
        else:
            members = [by_id.get(pair["a"]["sample_id"]), by_id.get(pair["b"]["sample_id"])]
            if any(member is None for member in members):
                continue
        visual, field, indices, target, reference = build_batch(members, store, masks, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=device != "cpu"):
            logits = variant_forward(model, variant, visual, indices, field)
        upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                  align_corners=False) > 0.0
        own, cross = [], []
        for index in range(len(members)):
            own_mask = target[index] > 0.5
            own.append(float((upsampled[index, 0] & own_mask).sum())
                       / float((upsampled[index, 0] | own_mask).sum() + 1e-6))
            other_target = target[1 - index] > 0.5
            cross.append(float((upsampled[index, 0] & other_target).sum())
                         / float((upsampled[index, 0] | other_target).sum() + 1e-6))
        rows.append({"tile_id": members[0].tile_id, "own": own, "cross": cross,
                     "passes": bool(own[0] > cross[0] and own[1] > cross[1])})
    passed = sum(1 for row in rows if row["passes"])
    mean_own = float(np.mean([value for row in rows for value in row["own"]]))
    mean_cross = float(np.mean([value for row in rows for value in row["cross"]]))
    return {"pairs": len(rows), "available": True, "passed": passed, "mean_own_iou": mean_own,
            "mean_cross_iou": mean_cross, "own_cross_margin": mean_own - mean_cross, "rows": rows}


def _train_steps(model, variant, samples, store, masks, device, *, steps, batch_size, lr,
                 weight_decay, evaluate_every, evaluate_fn):
    seed_everything(SEED)
    optimizer = make_optimizer(model, lr, weight_decay)
    order = list(range(len(samples)))
    generator = np.random.default_rng(SEED)
    history = []
    step = 0
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    while step < steps:
        model.train()
        generator.shuffle(order)
        chunk = [samples[index] for index in order[:batch_size]]
        visual, field, indices, target, reference = build_batch(chunk, store, masks, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = variant_forward(model, variant, visual, indices, field)
        losses = batch_loss(logits, target, reference, chunk, with_grcl=True)
        losses["loss"].backward()
        optimizer.step()
        step += 1
        if step % evaluate_every == 0 or step == steps:
            metrics = evaluate_fn()
            history.append({"step": step, "loss": float(losses["loss"].detach()),
                            "bce": float(losses["bce"]), "dice_loss": float(losses["dice"]),
                            "grcl": float(losses["grcl"]), "miou": metrics["miou"],
                            "dice": metrics["dice"],
                            "relation_accuracy": metrics["relation_accuracy"]})
            print(f"[6r.train] {variant} step {step}: loss {float(losses['loss'].detach()):.4f} "
                  f"GRCL {float(losses['grcl']):.4f} mIoU {metrics['miou']:.4f} "
                  f"Dice {metrics['dice']:.4f} rel-acc {metrics['relation_accuracy']:.4f}", flush=True)
    return history


def run_overfit(args) -> int:
    config = {
        "stage": "overfit20", "variants": list(VARIANTS), "optimizer": "AdamW", "lr": 1.0e-3,
        "weight_decay": 1.0e-4, "max_steps": 1200, "batch_size": 4, "scheduler": "none",
        "augmentation": "none", "seed": SEED, "amp": "bfloat16 autocast (same policy as 6N/6O)",
        "evaluate_every_steps": 100, "lambda_grcl": LAMBDA_GRCL, "alpha": ALPHA, "tau": TAU,
        "loss": "BCE + Dice + 0.5 * GRCL",
        "gate": {"applies_to": "R1", "miou_min": 0.85, "dice_min": 0.90,
                 "relation_accuracy_min": 0.95},
    }
    started = time.time()
    pack_path = PACK_ROOT / "overfit20.json"
    samples = read_pack(pack_path)
    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    print(f"[6r.train] Overfit20 records {len(samples)}; variants {list(VARIANTS)}", flush=True)

    results = {}
    for variant in VARIANTS:
        model = model_for(variant, args.device)
        parameters = model.parameter_report()
        variant_started = time.time()
        history = _train_steps(
            model, variant, samples, store, masks, args.device,
            steps=config["max_steps"], batch_size=config["batch_size"], lr=config["lr"],
            weight_decay=config["weight_decay"], evaluate_every=config["evaluate_every_steps"],
            evaluate_fn=lambda: evaluate_variant(model, variant, samples, store, masks, args.device,
                                                 config["batch_size"]),
        )
        final = evaluate_variant(model, variant, samples, store, masks, args.device,
                                 config["batch_size"])
        pairing = paired_own_cross(model, variant, pack_path, samples, store, masks, args.device)
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_ROOT / f"overfit_{variant}.pt"
        torch.save({"variant": variant, "state_dict": model.state_dict(), "config": config},
                   checkpoint_path)
        results[variant] = {
            "description": VARIANTS[variant]["description"],
            "decoder": VARIANTS[variant]["decoder"],
            "uses_field": VARIANT_USES_FIELD[VARIANTS[variant]["decoder"]],
            "parameters": parameters,
            "history": history,
            "final": {key: final[key] for key in ("miou", "dice", "precision_at_0_5", "mean_grcl",
                                                  "relation_accuracy", "axis_violation_rate",
                                                  "mean_positive_signed_margin", "records")},
            "relation_per_direction": final["relation_per_direction"],
            "paired_own_cross": {key: value for key, value in pairing.items() if key != "rows"},
            "wall_seconds": round(time.time() - variant_started, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if args.device != "cpu" else None,
            "checkpoint": {"path": str(checkpoint_path),
                           "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
        }
        print(f"[6r.train] {variant} done: mIoU {final['miou']:.4f} Dice {final['dice']:.4f} "
              f"rel-acc {final['relation_accuracy']:.4f}", flush=True)

    r1 = results["R1"]["final"]
    gate = {
        "applies_to": "R1", "miou_min": config["gate"]["miou_min"],
        "dice_min": config["gate"]["dice_min"],
        "relation_accuracy_min": config["gate"]["relation_accuracy_min"],
        "r1_miou": r1["miou"], "r1_dice": r1["dice"],
        "r1_relation_accuracy": r1["relation_accuracy"],
    }
    gate["passed"] = bool(r1["miou"] >= gate["miou_min"] and r1["dice"] >= gate["dice_min"]
                          and r1["relation_accuracy"] >= gate["relation_accuracy_min"])
    payload = {
        "_doc": (
            "Task 6R Part F. Stage R1: R1 (B3 + GRCL) and R2 (no-field + GRCL) trained separately on "
            "the byte-identical frozen Task 6N Overfit20 with the fixed optimiser/step budget. The gate "
            "applies to R1 only; R2 has no gate. Oracle reference: input to R1's field, and used by "
            "GRCL/evaluation only for R2."
        ),
        "task": "6R", "stage": "R1-overfit20", "reference_source": "oracle_native_gt",
        "config": config,
        "pack": {"path": str(pack_path), "count": len(samples),
                 "sha256": hashlib.sha256(pack_path.read_bytes()).hexdigest()},
        "variants": results,
        "gate": gate,
        "verdict": "R1_OVERFIT_PASS" if gate["passed"] else "GRCL_OVERFIT_FAIL",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_OVERFIT, payload)
    print(f"[6r.train] overfit gate {gate['passed']} (mIoU {gate['r1_miou']:.4f} Dice "
          f"{gate['r1_dice']:.4f} rel-acc {gate['r1_relation_accuracy']:.4f}) -> "
          f"{payload['verdict']}", flush=True)
    return 0 if gate["passed"] else 2


def run_mini(args) -> int:
    config = {
        "stage": "mini", "train_pack": "mini_train_1000", "val_pack": "mini_val_240",
        "variants": list(VARIANTS), "optimizer": "AdamW", "lr": 3.0e-4, "weight_decay": 1.0e-4,
        "batch_size": 8, "max_epochs": 25, "early_stopping_patience": 5,
        "early_stopping_monitor": "mini_val_240_miou", "model_selection": "highest MiniVal240 mIoU",
        "scheduler": "none", "augmentation": "none", "seed": SEED,
        "amp": "bfloat16 autocast (same policy as 6N/6O)", "fresh_initialisation": True,
        "lambda_grcl": LAMBDA_GRCL, "alpha": ALPHA, "tau": TAU, "loss": "BCE + Dice + 0.5 * GRCL",
    }
    started = time.time()
    train_samples = read_pack(PACK_ROOT / "mini_train_1000.json")
    val_samples = read_pack(PACK_ROOT / "mini_val_240.json")
    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    print(f"[6r.train] mini train {len(train_samples)} / val {len(val_samples)}", flush=True)

    results = {}
    for variant in VARIANTS:
        seed_everything(SEED)
        model = model_for(variant, args.device)
        parameters = model.parameter_report()
        optimizer = make_optimizer(model, config["lr"], config["weight_decay"])
        if args.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        variant_started = time.time()
        best = {"miou": -1.0, "epoch": 0, "state": None, "metrics": None}
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
                visual, field, indices, target, reference = build_batch(chunk, store, masks,
                                                                        args.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=args.device != "cpu"):
                    logits = variant_forward(model, variant, visual, indices, field)
                output = batch_loss(logits, target, reference, chunk, with_grcl=True)
                output["loss"].backward()
                optimizer.step()
                losses.append(float(output["loss"].detach()))
            val = evaluate_variant(model, variant, val_samples, store, masks, args.device,
                                   config["batch_size"])
            epochs.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                           "val_miou": val["miou"], "val_dice": val["dice"],
                           "val_relation_accuracy": val["relation_accuracy"],
                           "val_grcl": val["mean_grcl"]})
            print(f"[6r.train] {variant} epoch {epoch}: loss {np.mean(losses):.4f} val mIoU "
                  f"{val['miou']:.4f} Dice {val['dice']:.4f} rel-acc {val['relation_accuracy']:.4f}",
                  flush=True)
            if val["miou"] > best["miou"]:
                best = {"miou": val["miou"], "epoch": epoch, "metrics": val,
                        "state": {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if stale >= config["early_stopping_patience"]:
                    print(f"[6r.train] {variant}: early stop at epoch {epoch}", flush=True)
                    break
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        final = evaluate_variant(model, variant, val_samples, store, masks, args.device,
                                 config["batch_size"])
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_ROOT / f"mini_{variant}.pt"
        torch.save({"variant": variant, "state_dict": model.state_dict(), "config": config,
                    "best_epoch": best["epoch"]}, checkpoint_path)
        results[variant] = {
            "description": VARIANTS[variant]["description"],
            "decoder": VARIANTS[variant]["decoder"],
            "uses_field": VARIANT_USES_FIELD[VARIANTS[variant]["decoder"]],
            "parameters": parameters,
            "epochs": epochs,
            "epochs_run": len(epochs),
            "best_epoch": best["epoch"],
            "selected_model": {key: final[key] for key in
                               ("miou", "dice", "precision_at_0_5", "mean_grcl", "relation_accuracy",
                                "axis_violation_rate", "mean_positive_signed_margin", "records")},
            "relation_per_direction": final["relation_per_direction"],
            "wall_seconds": round(time.time() - variant_started, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if args.device != "cpu" else None,
            "checkpoint": {"path": str(checkpoint_path),
                           "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()},
        }
        print(f"[6r.train] {variant} selected epoch {best['epoch']} val mIoU {final['miou']:.4f}",
              flush=True)

    payload = {
        "_doc": (
            "Task 6R Part G. Stage R2: R1 and R2 trained from fresh initialisation on the frozen Task "
            "6N MiniTrain1000 and selected on MiniVal240 mIoU with the fixed 25-epoch / patience-5 "
            "schedule. Identical settings for both variants."
        ),
        "task": "6R", "stage": "R2-mini", "reference_source": "oracle_native_gt",
        "config": config,
        "packs": {
            "train": {"path": str(PACK_ROOT / "mini_train_1000.json"), "count": len(train_samples),
                      "sha256": hashlib.sha256((PACK_ROOT / "mini_train_1000.json").read_bytes()).hexdigest()},
            "val": {"path": str(PACK_ROOT / "mini_val_240.json"), "count": len(val_samples),
                    "sha256": hashlib.sha256((PACK_ROOT / "mini_val_240.json").read_bytes()).hexdigest()},
        },
        "variants": results,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_TRAINING, payload)
    print(f"[6r.train] mini selected: R1 {results['R1']['selected_model']['miou']:.6f} | "
          f"R2 {results['R2']['selected_model']['miou']:.6f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("overfit", "mini"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return run_overfit(args) if args.stage == "overfit" else run_mini(args)


if __name__ == "__main__":
    raise SystemExit(main())
