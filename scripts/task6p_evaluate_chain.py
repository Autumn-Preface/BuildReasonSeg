"""Task 6P Parts C and G-H — B3 reproduction with field v0.2 and the predicted-reference chain.

**Stage `b3-reproduction` (section 5)** — verify the frozen Task 6O B3 checkpoint (path + SHA256 from
the Task 6O artifact), re-run B3 once on MiniVal240 using the **v0.2 oracle field** and require
mIoU/Dice absolute delta <= 1e-6 against the frozen Task 6O B3 result. B3 is never retrained.

**Stage `chain` (sections 13-16)** — for every frozen MiniVal240 record: frozen SAM2 feature ->
frozen ReferenceMaskHead with the canonical `largest`/`smallest` family -> soft reference probability
map at 512 x 512 -> `P_rel_pred` via GeometricRelationField v0.2 -> frozen Task 6O B3 -> target mask.
Also writes the oracle-vs-predicted field diagnostics and the PairedVal20 report (the same predicted
reference mask is reused for a pair that shares the image and the reference source).

No oracle reference mask may enter the predicted-reference path; the GT reference is used only for
reference-head metrics and the GT target only for target scoring.

    python scripts/task6p_evaluate_chain.py --stage b3-reproduction
    python scripts/task6p_evaluate_chain.py --stage chain
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

from buildreasonseg_mvp.geometric_relation_field import geometric_relation_field as v01_field  # noqa: E402
from buildreasonseg_mvp.geometric_relation_field_v02 import (  # noqa: E402
    geometric_relation_field_v02,
    soft_centroid,
)
from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    DecoderConfig,
    FrozenFeatureStore,
    RelationMaskDecoder,
    load_frozen_sam2_encoder,
    read_pack,
)
from buildreasonseg_mvp.task6p_reference_head import (  # noqa: E402
    FAMILY_TO_INDEX,
    ReferenceMaskHead,
    family_of_program,
)
from scripts.task6n_evaluate import breakdown, per_family, per_relation, target_flags  # noqa: E402
from scripts.task6n_train import FEATURE_ROOT, PACK_ROOT, MaskStore  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "task6p" / "checkpoints"
OUT_B3 = EVAL / "task6p_b3_reproduction.json"
OUT_TARGET = EVAL / "task6p_predicted_reference_target_val.json"
OUT_DIAGNOSTICS = EVAL / "task6p_field_propagation_diagnostics.json"
OUT_PAIRED = EVAL / "task6p_predicted_reference_paired_val.json"
TOLERANCE = 1e-6
FIELD_SIZE = (64, 64)
TARGET_SIZE = (512, 512)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_b3(device: str) -> tuple[RelationMaskDecoder, dict]:
    artifact = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))
    checkpoint = artifact["variants"]["B3"]["training"]["checkpoint"]
    path = Path(checkpoint["path"])
    return _load_b3_from(path, checkpoint["sha256"], device)


def _load_b3_from(path: Path, expected_sha: str, device: str) -> tuple[RelationMaskDecoder, dict]:
    present = path.is_file()
    actual = sha256_file(path) if present else None
    info = {"path": str(path), "present": present, "expected_sha256": expected_sha,
            "actual_sha256": actual, "matches": actual == expected_sha, "retrained": False}
    if not (present and actual == expected_sha):
        return None, info
    model = RelationMaskDecoder("B3", DecoderConfig()).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, info


@torch.no_grad()
def b3_metrics(model, samples, store, masks, device, *, field_source: str,
               reference_masks: dict | None = None, batch_size: int = 8) -> dict:
    """B3 target metrics with either the oracle field (v0.1 or v0.2) or a supplied predicted field."""

    rows = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        visuals, fields, indices, targets = [], [], [], []
        for sample in chunk:
            visual = store.get(sample.tile_id, Path(sample.image_path)).float()
            if field_source == "predicted":
                field = reference_masks[sample.sample_id]
            else:
                mask_ref = masks.mask(sample.tile_id, sample.reference_source_feature_id)
                tensor = torch.as_tensor(np.asarray(mask_ref, dtype=np.float32))
                field = (v01_field(tensor, sample.relation, FIELD_SIZE) if field_source == "v01"
                         else geometric_relation_field_v02(tensor, sample.relation, FIELD_SIZE))
            target = masks.mask(sample.tile_id, sample.target_source_feature_id)
            visuals.append(visual)
            fields.append(field[0])
            indices.append(("left_of", "right_of", "above", "below").index(sample.relation))
            targets.append(torch.as_tensor(np.asarray(target, dtype=np.float32)).unsqueeze(0))
        batch_visual = torch.stack(visuals).to(device)
        batch_field = torch.stack(fields).to(device)
        batch_index = torch.as_tensor(indices, dtype=torch.long, device=device)
        batch_target = torch.stack(targets).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = model(batch_visual, batch_index, None, batch_field)
        upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                  align_corners=False) > 0.0
        gt = batch_target > 0.5
        for index, sample in enumerate(chunk):
            prediction = upsampled[index]
            truth = gt[index]
            intersection = float((prediction & truth).sum())
            union = float((prediction | truth).sum())
            predicted_area = float(prediction.sum())
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "program_id": sample.program_id,
                    "relation": sample.relation,
                    "reference_family": family_of_program(sample.program_id),
                    "miou": (intersection + 1e-6) / (union + 1e-6),
                    "dice": (2.0 * intersection + 1e-6) / (predicted_area + float(truth.sum()) + 1e-6),
                    "precision_at_0_5": (intersection + 1e-6) / (predicted_area + 1e-6),
                    **target_flags(masks, sample),
                }
            )
    return rows


def run_b3_reproduction(args) -> int:
    started = time.time()
    model, info = load_b3(args.device)
    if model is None:
        write_json(OUT_B3, {
            "_doc": "Task 6P section 5. Frozen Task 6O B3 checkpoint verification.",
            "task": "6P", "stage": "C-b3-verification", "checkpoint": info,
            "verdict": "TASK6O_B3_REPRODUCTION_FAIL",
        })
        print("[6p.b3] STOP TASK6O_B3_REPRODUCTION_FAIL (checkpoint unavailable)", flush=True)
        return 2

    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    rows = b3_metrics(model, samples, store, masks, args.device, field_source="v02")
    reproduced = breakdown(rows)
    frozen = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))
    stored = frozen["variants"]["B3"]["overall"]

    comparisons = {
        "miou": {"stored_task6o": stored["miou"], "reproduced": reproduced["miou"],
                 "abs_delta": abs(reproduced["miou"] - stored["miou"]),
                 "within_tolerance": abs(reproduced["miou"] - stored["miou"]) <= TOLERANCE},
        "dice": {"stored_task6o": stored["dice"], "reproduced": reproduced["dice"],
                 "abs_delta": abs(reproduced["dice"] - stored["dice"]),
                 "within_tolerance": abs(reproduced["dice"] - stored["dice"]) <= TOLERANCE},
    }
    passed = all(entry["within_tolerance"] for entry in comparisons.values())
    payload = {
        "_doc": (
            "Task 6P section 5. The frozen Task 6O B3 checkpoint, re-evaluated exactly once on "
            "MiniVal240 with the oracle reference processed by GeometricRelationField v0.2 instead of "
            "v0.1. B3 is never retrained; the tolerance is 1e-6 on mIoU and Dice."
        ),
        "task": "6P", "stage": "C-b3-reproduction", "reference_source": "oracle_native_gt",
        "tolerance": TOLERANCE,
        "checkpoint": info,
        "field_used": "geometric_relation_field_v02 (oracle reference mask)",
        "reproduced": {key: reproduced[key] for key in ("miou", "dice", "precision_at_0_5", "records")},
        "stored_task6o": {"miou": stored["miou"], "dice": stored["dice"]},
        "comparisons": comparisons,
        "b3_retrained": False,
        "verdict": "B3_REPRODUCED" if passed else "TASK6O_B3_REPRODUCTION_FAIL",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_B3, payload)
    print(f"[6p.b3] reproduced mIoU {reproduced['miou']:.9f} (stored {stored['miou']:.9f}) | Dice "
          f"{reproduced['dice']:.9f} (stored {stored['dice']:.9f}) -> {payload['verdict']}", flush=True)
    return 0 if passed else 2


@torch.no_grad()
def predicted_field_for(sample, head, store, device) -> tuple[torch.Tensor, np.ndarray]:
    """`M_ref_pred` (512) and `P_rel_pred` (64) from the frozen reference head — no oracle mask."""

    visual = store.get(sample.tile_id, Path(sample.image_path)).float().unsqueeze(0).to(device)
    family = torch.as_tensor([FAMILY_TO_INDEX[family_of_program(sample.program_id)]],
                             dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
        logits = head(visual, family)
    upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear", align_corners=False)
    soft_reference = torch.sigmoid(upsampled)[0, 0]
    field = geometric_relation_field_v02(soft_reference.unsqueeze(0).unsqueeze(0), sample.relation,
                                         FIELD_SIZE)
    return field, soft_reference.detach().cpu().numpy()


def run_chain(args) -> int:
    started = time.time()
    training = json.loads((EVAL / "task6p_reference_val.json").read_text(encoding="utf-8"))
    head_path = Path(training["checkpoint"]["path"])
    head = ReferenceMaskHead().to(args.device)
    state = torch.load(head_path, map_location=args.device, weights_only=False)
    head.load_state_dict(state["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    b3, b3_info = load_b3(args.device)
    if b3 is None:
        print("[6p.chain] STOP: B3 checkpoint unavailable", flush=True)
        return 2

    masks = MaskStore()
    encoder, _ = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    samples = read_pack(PACK_ROOT / "mini_val_240.json")
    pairs = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))

    # ---------------- predicted field per sample + propagation diagnostics
    predicted_fields: dict[str, torch.Tensor] = {}
    diagnostics_rows = []
    for index, sample in enumerate(samples, start=1):
        field_pred, soft_reference = predicted_field_for(sample, head, store, args.device)
        predicted_fields[sample.sample_id] = field_pred[0].detach().cpu()
        oracle_mask = torch.as_tensor(
            np.asarray(masks.mask(sample.tile_id, sample.reference_source_feature_id), dtype=np.float32)
        )
        field_oracle = geometric_relation_field_v02(oracle_mask, sample.relation, FIELD_SIZE)[0, 0]
        field_pred_2d = field_pred[0, 0].detach().cpu()
        difference = (field_pred_2d - field_oracle).detach().numpy()
        flat_pred = field_pred_2d.numpy().reshape(-1)
        flat_oracle = field_oracle.numpy().reshape(-1)
        pearson = float(np.corrcoef(flat_pred, flat_oracle)[0, 1]) if flat_pred.std() > 0 else None
        pred_cx, pred_cy = [float(value) for value in
                            torch.cat([item.view(-1) for item in soft_centroid(
                                torch.as_tensor(soft_reference, dtype=torch.float32))])]
        gt_cx, gt_cy = [float(value) for value in
                        torch.cat([item.view(-1) for item in soft_centroid(oracle_mask)])]
        diagnostics_rows.append(
            {
                "sample_id": sample.sample_id,
                "relation": sample.relation,
                "reference_family": family_of_program(sample.program_id),
                "mae": float(np.abs(difference).mean()),
                "rmse": float(np.sqrt((difference ** 2).mean())),
                "pearson": pearson,
                "predicted_centroid": [pred_cx, pred_cy],
                "oracle_centroid": [gt_cx, gt_cy],
                "centroid_error_normalized": float(np.hypot(pred_cx - gt_cx, pred_cy - gt_cy))
                / float(np.sqrt(2.0)),
            }
        )
        if index % 60 == 0:
            print(f"[6p.chain] predicted fields {index}/{len(samples)}", flush=True)

    def summarise_diagnostics(rows):
        pearsons = [row["pearson"] for row in rows if row["pearson"] is not None]
        errors = np.asarray([row["centroid_error_normalized"] for row in rows], dtype=np.float64)
        return {
            "records": len(rows),
            "mae": float(np.mean([row["mae"] for row in rows])),
            "rmse": float(np.mean([row["rmse"] for row in rows])),
            "mean_pearson": float(np.mean(pearsons)) if pearsons else None,
            "centroid_error_normalized": {
                "mean": float(errors.mean()), "median": float(np.median(errors)),
                "p90": float(np.percentile(errors, 90)),
            },
        }

    diagnostics = {
        "_doc": (
            "Task 6P section 14. Predicted-reference field versus oracle-reference field "
            "(GeometricRelationField v0.2) on MiniVal240: MAE, RMSE, per-record Pearson correlation "
            "then mean, predicted-vs-oracle reference centroid error, per family and per relation. "
            "Diagnostic only, no threshold tuning."
        ),
        "task": "6P", "stage": "G-field-propagation",
        "reference_source": "predicted (frozen ReferenceMaskHead) vs oracle_native_gt",
        "overall": summarise_diagnostics(diagnostics_rows),
        "per_family": {family: summarise_diagnostics(
            [row for row in diagnostics_rows if row["reference_family"] == family])
            for family in ("largest", "smallest")},
        "per_relation": {relation: summarise_diagnostics(
            [row for row in diagnostics_rows if row["relation"] == relation])
            for relation in ("left_of", "right_of", "above", "below")},
        "records": diagnostics_rows,
        "threshold_tuning": False,
    }
    write_json(OUT_DIAGNOSTICS, diagnostics)

    # ---------------- downstream target metrics with the predicted field
    rows = b3_metrics(b3, samples, store, masks, args.device, field_source="predicted",
                      reference_masks={key: value.unsqueeze(0) for key, value in
                                       predicted_fields.items()})
    predicted_target = breakdown(rows)
    frozen_b3 = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))["variants"]["B3"]
    target_payload = {
        "_doc": (
            "Task 6P sections 13/15. Predicted-reference substitution: frozen ReferenceMaskHead -> soft "
            "reference map -> GeometricRelationField v0.2 -> frozen Task 6O B3 -> target mask. No "
            "oracle reference mask enters this path; the GT target is used only for scoring."
        ),
        "task": "6P", "stage": "G-predicted-reference-chain",
        "reference_source": "predicted_reference_mask",
        "chain": ["frozen_sam2_feature", "frozen_reference_mask_head(largest/smallest)",
                  "sigmoid_upsample_512", "geometric_relation_field_v02", "frozen_task6o_b3"],
        "reference_head": {"checkpoint": training["checkpoint"], "frozen": True,
                           "joint_training_with_b3": False},
        "b3": {"checkpoint": b3_info, "retrained": False},
        "predicted_target": {
            "overall": predicted_target,
            "per_relation": per_relation(rows),
            "per_reference_family": per_family(rows),
        },
        "oracle_reference_b3_frozen": {
            "miou": frozen_b3["overall"]["miou"], "dice": frozen_b3["overall"]["dice"],
            "paired_pass": 14, "own_cross_margin": 0.39719566349802166,
        },
        "oracle_vs_predicted_miou_delta": predicted_target["miou"] - frozen_b3["overall"]["miou"],
        "records": rows,
        "test_split_used": False,
    }
    write_json(OUT_TARGET, target_payload)

    # ---------------- PairedVal20 with the predicted reference (same image/reference reused)
    #
    # The frozen PairedVal20 pack covers val records beyond MiniVal240, so a pair's predicted reference
    # field is taken from the cache when available and otherwise computed on demand through the exact
    # same chain (frozen head -> v0.2). A pair sharing (tile, reference source) always reuses one mask.
    from buildreasonseg_mvp.task6n_relation_decoder import Task6NSample

    cache: dict[tuple, torch.Tensor] = {
        (sample.tile_id, sample.reference_source_feature_id): predicted_fields[sample.sample_id]
        for sample in samples
    }
    pair_rows = []
    family_mismatch = 0
    cache_hits = 0
    for pair in pairs["pairs"]:
        members = [
            Task6NSample(**{key: value for key, value in pair[side].items()
                            if key in Task6NSample.__dataclass_fields__})
            for side in ("a", "b")
        ]
        families = {family_of_program(member.program_id) for member in members}
        if len(families) > 1:
            family_mismatch += 1
        key = (members[0].tile_id, members[0].reference_source_feature_id)
        if key in cache:
            cache_hits += 1
        else:
            field, _ = predicted_field_for(members[0], head, store, args.device)
            cache[key] = field[0].detach().cpu()
        visual = torch.stack([
            store.get(member.tile_id, Path(member.image_path)).float() for member in members
        ]).to(args.device)
        field = cache[key].to(args.device).unsqueeze(0).expand(len(members), -1, -1, -1)
        indices = torch.as_tensor([("left_of", "right_of", "above", "below").index(member.relation)
                                   for member in members], dtype=torch.long, device=args.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                             enabled=args.device != "cpu"):
            logits = b3(visual, indices, None, field)
        upsampled = F.interpolate(logits.float(), size=TARGET_SIZE, mode="bilinear",
                                  align_corners=False) > 0.0
        own, cross = [], []
        for index, member in enumerate(members):
            own_mask = torch.as_tensor(
                np.asarray(masks.mask(member.tile_id, member.target_source_feature_id), dtype=np.float32)
            ).to(upsampled.device) > 0.5
            own.append(float((upsampled[index, 0] & own_mask).sum())
                       / float((upsampled[index, 0] | own_mask).sum() + 1e-6))
            other = members[1 - index]
            other_mask = torch.as_tensor(
                np.asarray(masks.mask(other.tile_id, other.target_source_feature_id), dtype=np.float32)
            ).to(upsampled.device) > 0.5
            cross.append(float((upsampled[index, 0] & other_mask).sum())
                         / float((upsampled[index, 0] | other_mask).sum() + 1e-6))
        pair_rows.append(
            {
                "tile_id": members[0].tile_id,
                "reference_source_feature_id": members[0].reference_source_feature_id,
                "same_predicted_reference_reused": True,
                "a": {"sample_id": members[0].sample_id, "relation": members[0].relation,
                      "own_iou": own[0], "cross_iou": cross[0], "prefers_own": own[0] > cross[0]},
                "b": {"sample_id": members[1].sample_id, "relation": members[1].relation,
                      "own_iou": own[1], "cross_iou": cross[1], "prefers_own": own[1] > cross[1]},
                "passes": bool(own[0] > cross[0] and own[1] > cross[1]),
            }
        )
    passed = sum(1 for row in pair_rows if row["passes"])
    mean_own = float(np.mean([row["a"]["own_iou"] for row in pair_rows]
                             + [row["b"]["own_iou"] for row in pair_rows]))
    mean_cross = float(np.mean([row["a"]["cross_iou"] for row in pair_rows]
                               + [row["b"]["cross_iou"] for row in pair_rows]))
    paired_payload = {
        "_doc": (
            "Task 6P section 16. PairedVal20 with the predicted-reference chain: for a pair sharing the "
            "image and the reference source the SAME predicted reference mask is reused. A pair passes "
            "only if both members prefer their own GT target over the paired alternative by IoU."
        ),
        "task": "6P", "stage": "H-paired",
        "reference_source": "predicted_reference_mask",
        "pairs": len(pair_rows), "passed": passed,
        "pass_rate": passed / len(pair_rows) if pair_rows else None,
        "mean_own_iou": mean_own, "mean_cross_iou": mean_cross,
        "own_cross_margin": mean_own - mean_cross,
        "pair_pass_rule": "both members prefer their own GT target over the paired alternative by IoU",
        "same_predicted_reference_reused_for_pairs": True,
        "pair_field_cache_hits_from_mini_val": cache_hits,
        "family_mismatch_pairs": family_mismatch,
        "oracle_reference_b3_frozen": {"paired_pass": 14, "own_cross_margin": 0.39719566349802166},
        "rows": pair_rows,
        "test_split_used": False,
    }
    write_json(OUT_PAIRED, paired_payload)
    print(f"[6p.chain] predicted target mIoU {predicted_target['miou']:.6f} Dice "
          f"{predicted_target['dice']:.6f} | paired {passed}/{len(pair_rows)} margin "
          f"{mean_own - mean_cross:+.6f}", flush=True)
    print(f"[6p.chain] field MAE {diagnostics['overall']['mae']:.6f} RMSE "
          f"{diagnostics['overall']['rmse']:.6f} mean Pearson "
          f"{diagnostics['overall']['mean_pearson']:.6f}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("b3-reproduction", "chain"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return run_b3_reproduction(args) if args.stage == "b3-reproduction" else run_chain(args)


if __name__ == "__main__":
    raise SystemExit(main())
