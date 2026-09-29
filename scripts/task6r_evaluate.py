"""Task 6R Parts D and H — frozen B3 relation baseline and the main R0/R1/R2 evaluation.

`--stage baseline` (sections 10-11): verify the Task 6O B3 checkpoint hash, reproduce MiniVal240
mIoU/Dice within 1e-6, and compute the hard relation-correctness metric for frozen B3 on MiniVal240 and
PairedVal20 → `evaluation/task6r_b3_relation_baseline.json`. R0 is never trained.

`--stage final` (sections 17-18): MiniVal240 and PairedVal20 tables for R0 (frozen B3) / R1 (B3+GRCL) /
R2 (no-field+GRCL) → `evaluation/task6r_mini_val.json` and `evaluation/task6r_paired_val.json`.

The oracle reference is the input to R1's field and is used by GRCL/evaluation; it is never an R2 model
input. GT targets are labels/scores only and the test split is never read.

    python scripts/task6r_evaluate.py --stage baseline
    python scripts/task6r_evaluate.py --stage final
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

from buildreasonseg_mvp.grcl_directional import hard_relation_metrics, summarise_relation_metrics  # noqa: E402
from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    Task6NSample,
    load_frozen_sam2_encoder,
    read_pack,
)
from scripts.task6n_train import FEATURE_ROOT, PACK_ROOT, MaskStore  # noqa: E402
from scripts.task6r_train import (  # noqa: E402
    CHECKPOINT_ROOT,
    TARGET_SIZE,
    VARIANTS,
    build_batch,
    evaluate_variant,
    model_for,
    paired_own_cross,
    variant_forward,
)

EVAL = REPO_ROOT / "evaluation"
OUT_BASELINE = EVAL / "task6r_b3_relation_baseline.json"
OUT_MINI = EVAL / "task6r_mini_val.json"
OUT_PAIRED = EVAL / "task6r_paired_val.json"
TOLERANCE = 1e-6
STORED_B3 = {"miou": 0.4299680351479113, "dice": 0.5401551056992948}
ORACLE_MIOU = 0.4299680351479113
MASK_RETENTION_THRESHOLD = 0.4099680351479113


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_b3(device: str):
    artifact = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))
    checkpoint = artifact["variants"]["B3"]["training"]["checkpoint"]
    path = Path(checkpoint["path"])
    present = path.is_file()
    actual = sha256_file(path) if present else None
    info = {"path": str(path), "present": present, "expected_sha256": checkpoint["sha256"],
            "actual_sha256": actual, "matches": actual == checkpoint["sha256"], "retrained": False}
    if not (present and actual == checkpoint["sha256"]):
        return None, info
    model = model_for("R1", device)  # R1 has exactly the B3 architecture
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
    model.eval()
    return model, info


def run_baseline(args) -> int:
    started = time.time()
    model, info = load_b3(args.device)
    if model is None:
        write_json(OUT_BASELINE, {"_doc": "Task 6R section 10.", "task": "6R",
                                  "checkpoint": info, "verdict": "INVALID_EXPERIMENT",
                                  "reason": "frozen Task 6O B3 checkpoint unavailable"})
        print("[6r.eval] STOP: B3 checkpoint unavailable", flush=True)
        return 2
    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    val_samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))

    val = evaluate_variant(model, "R1", val_samples, store, masks, args.device, 8)
    comparisons = {
        "miou": {"stored_task6o": STORED_B3["miou"], "reproduced": val["miou"],
                 "abs_delta": abs(val["miou"] - STORED_B3["miou"]),
                 "within_tolerance": abs(val["miou"] - STORED_B3["miou"]) <= TOLERANCE},
        "dice": {"stored_task6o": STORED_B3["dice"], "reproduced": val["dice"],
                 "abs_delta": abs(val["dice"] - STORED_B3["dice"]),
                 "within_tolerance": abs(val["dice"] - STORED_B3["dice"]) <= TOLERANCE},
    }
    reproduced = all(entry["within_tolerance"] for entry in comparisons.values())

    # ---------------- hard relation metric for R0 on PairedVal20 members
    pair_rows = []
    for pair in pairs["pairs"]:
        members = [Task6NSample(**{key: value for key, value in pair[side].items()
                                   if key in Task6NSample.__dataclass_fields__})
                   for side in ("a", "b")]
        visual, field, indices, target, reference = build_batch(members, store, masks, args.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=args.device != "cpu"):
            logits = variant_forward(model, "R1", visual, indices, field)
        probabilities = torch.sigmoid(
            F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear", align_corners=False)
        )
        relation = hard_relation_metrics(probabilities, reference,
                                         [member.relation for member in members])
        for index, member in enumerate(members):
            pair_rows.append({"sample_id": member.sample_id, "relation": member.relation,
                              "correct": bool(relation["correct"][index]),
                              "empty": bool(relation["empty"][index]),
                              "margin": float(relation["margin"][index]),
                              "axis_violation": bool(relation["axis_violation"][index])})

    payload = {
        "_doc": (
            "Task 6R sections 10-11. Frozen Task 6O B3 (R0) reproduced on MiniVal240 within 1e-6, plus "
            "the hard relation-correctness metric (threshold 0.5, evaluation only) on MiniVal240 and "
            "PairedVal20. R0 is not trained in Task 6R."
        ),
        "task": "6R",
        "stage": "D-b3-relation-baseline",
        "reference_source": "oracle_native_gt",
        "checkpoint": info,
        "tolerance": TOLERANCE,
        "reproduction": {
            "comparisons": comparisons,
            "reproduced": reproduced,
            "verdict": "B3_REPRODUCED" if reproduced else "INVALID_EXPERIMENT",
        },
        "mini_val_240": {
            "miou": val["miou"], "dice": val["dice"], "precision_at_0_5": val["precision_at_0_5"],
            "relation_accuracy": val["relation_accuracy"],
            "relation_per_direction": val["relation_per_direction"],
            "mean_positive_signed_margin": val["mean_positive_signed_margin"],
            "axis_violation_rate": val["axis_violation_rate"],
            "mean_grcl": val["mean_grcl"],
            "empty_predictions": val["empty_predictions"],
            "records": val["records"],
        },
        "paired_val_20": {
            "member_predictions": len(pair_rows),
            "relation_accuracy": summarise_relation_metrics(pair_rows)["relation_accuracy"],
            "per_relation": summarise_relation_metrics(pair_rows)["per_relation"],
            "mean_positive_signed_margin":
                summarise_relation_metrics(pair_rows)["mean_positive_signed_margin"],
            "axis_violation_rate": summarise_relation_metrics(pair_rows)["axis_violation_rate"],
            "empty_predictions": sum(1 for row in pair_rows if row["empty"]),
        },
        "hard_relation_metric": {
            "definition": "signed >= tau AND signed >= alpha * orth, threshold 0.5, empty mask incorrect",
            "alpha": 1.2, "tau": 0.04, "threshold": 0.5,
        },
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_BASELINE, payload)
    print(f"[6r.eval] R0 reproduction mIoU {val['miou']:.9f} (stored {STORED_B3['miou']:.9f}) dice "
          f"{val['dice']:.9f}; relation accuracy {val['relation_accuracy']:.4f} -> "
          f"{payload['reproduction']['verdict']}", flush=True)
    return 0 if reproduced else 2


def _checkpoint_for(variant: str) -> Path:
    training = json.loads((EVAL / "task6r_training.json").read_text(encoding="utf-8"))
    return Path(training["variants"][variant]["checkpoint"]["path"])


def run_final(args) -> int:
    started = time.time()
    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    val_samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pair_pack = PACK_ROOT / "paired_val_20.json"
    training = json.loads((EVAL / "task6r_training.json").read_text(encoding="utf-8"))
    baseline = json.loads(OUT_BASELINE.read_text(encoding="utf-8"))

    models = {}
    frozen_b3, b3_info = load_b3(args.device)
    if frozen_b3 is None:
        write_json(OUT_MINI, {"_doc": "Task 6R section 17.", "task": "6R",
                              "verdict": "INVALID_EXPERIMENT",
                              "reason": "frozen B3 checkpoint unavailable"})
        print("[6r.eval] STOP: B3 unavailable", flush=True)
        return 2
    models["R0"] = (frozen_b3, "R1", None)
    for variant in VARIANTS:
        path = _checkpoint_for(variant)
        model = model_for(variant, args.device)
        model.load_state_dict(torch.load(path, map_location=args.device, weights_only=False)["state_dict"])
        model.eval()
        models[variant] = (model, variant, training["variants"][variant])

    mini = {}
    paired = {}
    for name, (model, forward_variant, meta) in models.items():
        metrics = evaluate_variant(model, forward_variant, val_samples, store, masks, args.device, 8)
        mini[name] = {
            "label": "R0 frozen B3 (BCE+Dice only)" if name == "R0" else
                     f"{name} ({VARIANTS[name]['description']})",
            "decoder": "B3" if name in ("R0", "R1") else "B0",
            "uses_field": name in ("R0", "R1"),
            "loss": "BCE + Dice" if name == "R0" else "BCE + Dice + 0.5 * GRCL",
            "miou": metrics["miou"], "dice": metrics["dice"],
            "precision_at_0_5": metrics["precision_at_0_5"],
            "relation_accuracy": metrics["relation_accuracy"],
            "relation_per_direction": metrics["relation_per_direction"],
            "per_relation_miou": {
                relation: float(np.mean([row["miou"] for row in metrics["rows"]
                                         if row["relation"] == relation]))
                for relation in ("left_of", "right_of", "above", "below")
            },
            "per_reference_family_miou": {
                family: float(np.mean([row["miou"] for row in metrics["rows"]
                                       if row["reference_family"] == family]))
                for family in ("largest", "smallest")
            },
            "mean_grcl": metrics["mean_grcl"],
            "mean_positive_signed_margin": metrics["mean_positive_signed_margin"],
            "axis_violation_rate": metrics["axis_violation_rate"],
            "empty_predictions": metrics["empty_predictions"],
            "parameters": (meta["parameters"]["total_parameters"] if meta is not None else
                           json.loads((EVAL / "task6o_mini_val.json").read_text(
                               encoding="utf-8"))["variants"]["B3"]["parameters"]["total_parameters"]),
            "selected_epoch": (meta["best_epoch"] if meta is not None else
                               json.loads((EVAL / "task6o_mini_val.json").read_text(
                                   encoding="utf-8"))["variants"]["B3"]["training"]["selected_epoch"]),
            "wall_seconds": meta["wall_seconds"] if meta is not None else None,
            "peak_vram_gb": meta["peak_vram_gb"] if meta is not None else None,
            "border_target": {
                "records": sum(1 for row in metrics["rows"] if row["program_id"]),
                "miou": float(np.mean([row["miou"] for row in metrics["rows"]])),
            },
            "tiny_target": {"records": 0, "miou": None},
        }
        pairing = paired_own_cross(model, forward_variant, pair_pack, val_samples, store, masks,
                                   args.device)
        # hard relation accuracy over the 40 pair-member predictions
        pair_rows = []
        for pair in json.loads(pair_pack.read_text(encoding="utf-8"))["pairs"]:
            members = [Task6NSample(**{key: value for key, value in pair[side].items()
                                       if key in Task6NSample.__dataclass_fields__})
                       for side in ("a", "b")]
            visual, field, indices, target, reference = build_batch(members, store, masks, args.device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                                 enabled=args.device != "cpu"):
                logits = variant_forward(model, forward_variant, visual, indices, field)
            probabilities = torch.sigmoid(
                F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear", align_corners=False)
            )
            relation = hard_relation_metrics(probabilities, reference,
                                             [member.relation for member in members])
            for index, member in enumerate(members):
                pair_rows.append({"relation": member.relation,
                                  "correct": bool(relation["correct"][index]),
                                  "empty": bool(relation["empty"][index]),
                                  "margin": float(relation["margin"][index]),
                                  "axis_violation": bool(relation["axis_violation"][index])})
        paired[name] = {
            "passed": pairing["passed"], "pairs": pairing["pairs"],
            "mean_own_iou": pairing["mean_own_iou"], "mean_cross_iou": pairing["mean_cross_iou"],
            "own_cross_margin": pairing["own_cross_margin"],
            "relation_accuracy_members": summarise_relation_metrics(pair_rows)["relation_accuracy"],
            "member_predictions": len(pair_rows),
            "empty_predictions": sum(1 for row in pair_rows if row["empty"]),
        }
        print(f"[6r.eval] {name}: mIoU {metrics['miou']:.6f} rel-acc "
              f"{metrics['relation_accuracy']:.4f} | paired {pairing['passed']}/{pairing['pairs']} "
              f"margin {pairing['own_cross_margin']:+.6f}", flush=True)

    r0_relation = baseline["mini_val_240"]["relation_accuracy"]
    r0_miou = baseline["mini_val_240"]["miou"]
    criteria = {
        "1_r1_overfit_gate": None,
        "2_mask_retention": {
            "requirement": "R1 mIoU >= R0 mIoU - 0.02",
            "numerical_threshold": MASK_RETENTION_THRESHOLD,
            "r0_miou": r0_miou, "r1_miou": mini["R1"]["miou"],
            "passed": mini["R1"]["miou"] >= MASK_RETENTION_THRESHOLD,
        },
        "3_relation_gain": {
            "requirement": "R1 relation_accuracy >= R0 relation_accuracy + 0.08",
            "r0_relation_accuracy": r0_relation,
            "r1_relation_accuracy": mini["R1"]["relation_accuracy"],
            "gain": mini["R1"]["relation_accuracy"] - r0_relation,
            "passed": mini["R1"]["relation_accuracy"] >= r0_relation + 0.08,
        },
        "4_paired_discrimination": {
            "requirement": "R1 PairedVal >= 16/20", "measured": paired["R1"]["passed"],
            "passed": paired["R1"]["passed"] >= 16,
        },
        "5_own_cross_margin": {
            "requirement": "R1 own-cross margin >= 0.38", "measured": paired["R1"]["own_cross_margin"],
            "passed": paired["R1"]["own_cross_margin"] >= 0.38,
        },
        "6_field_remains_useful": {
            "requirement": "R1 mIoU - R2 mIoU >= 0.10",
            "r1_miou": mini["R1"]["miou"], "r2_miou": mini["R2"]["miou"],
            "delta": mini["R1"]["miou"] - mini["R2"]["miou"],
            "passed": (mini["R1"]["miou"] - mini["R2"]["miou"]) >= 0.10,
        },
    }
    if (OUT_BASELINE.parent / "task6r_overfit20.json").is_file():
        overfit = json.loads((EVAL / "task6r_overfit20.json").read_text(encoding="utf-8"))
        criteria["1_r1_overfit_gate"] = {"measured": overfit["gate"]["passed"],
                                         "passed": bool(overfit["gate"]["passed"])}
    payload = {
        "_doc": (
            "Task 6R Part H. MiniVal240 metrics for R0 (frozen B3), R1 (B3 + GRCL) and R2 "
            "(no-field + GRCL): mask quality, hard relation correctness, GRCL value, signed-margin "
            "statistics, axis violations, per-relation and per-family breakdowns and resources."
        ),
        "task": "6R",
        "stage": "H-mini-val",
        "reference_source": "oracle_native_gt",
        "pack": {"path": str(PACK_ROOT / "mini_val_240.json"), "count": len(val_samples),
                 "sha256": sha256_file(PACK_ROOT / "mini_val_240.json")},
        "variants": mini,
        "predeclared_criteria": criteria,
        "r0_reference": {"miou": r0_miou, "relation_accuracy": r0_relation,
                         "source": "evaluation/task6r_b3_relation_baseline.json"},
        "strong_mask_gain": bool(mini["R1"]["miou"] >= r0_miou + 0.02),
        "test_split_used": False,
    }
    paired_payload = {
        "_doc": (
            "Task 6R section 18. PairedVal20 for R0/R1/R2 on the exact frozen Task 6N pack with the "
            "oracle reference: a pair passes only if both members prefer their own GT target over the "
            "paired alternative by IoU."
        ),
        "task": "6R",
        "stage": "H-paired-val",
        "reference_source": "oracle_native_gt",
        "pack": {"path": str(pair_pack), "pairs": 20, "sha256": sha256_file(pair_pack)},
        "variants": paired,
        "test_split_used": False,
    }
    write_json(OUT_MINI, payload)
    write_json(OUT_PAIRED, paired_payload)
    print(f"[6r.eval] criteria: " + ", ".join(
        f"{name}={entry['passed']}" for name, entry in criteria.items()), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "final"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return run_baseline(args) if args.stage == "baseline" else run_final(args)


if __name__ == "__main__":
    raise SystemExit(main())
