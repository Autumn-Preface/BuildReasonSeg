"""Task 7I Part J — practical predicted-reference full-val evaluation of the six frozen checkpoints.

Frozen practical reference (no Task 7G selector): U-C1 proposals (imgsz 640, conf 0.05, max_det 300, default
NMS, no TTA, no tiling) → deterministic largest selector (max predicted mask area, tie higher confidence, then
lower index). Proposal inference and reference selection are computed **once** over all 936 val records and
reused by every model/seed.

Reference metrics (computed once), target metrics per model/seed (strict, answered-only, reference-OK subset,
per direction) and the `I-FormalValPairsAll` counterfactual set with the shared predicted reference per pair.
Predicted-reference results never influence checkpoint selection.

Writes `evaluation/task7i_predicted_reference_val_results.json`.

    python scripts/task7i_evaluate_predicted_val.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6z_l3_decoder import L3TargetDecoder  # noqa: E402
from buildreasonseg_mvp.task7d_global_competition_decoder import (  # noqa: E402
    GlobalCompetitionDecoder,
)
from scripts.task6n_train import MaskStore  # noqa: E402
from scripts.task7e_evaluate_predicted_reference import (  # noqa: E402
    BUCKET_MAP,
    _ReferenceOverride,
    effective_resolver_report,
)
from buildreasonseg_mvp.task7f_reference_ceiling import (  # noqa: E402
    dice_of,
    iou_of,
    sha256_file,
)
from scripts.task7i_train import PACK_ROOT, SEEDS, read_rows  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task7i"
VAL_ROWS = PACK_ROOT / "formal_val_936.jsonl"
PAIRS = EVAL / "task7i_formal_val_pair_manifest.json"
OUT = EVAL / "task7i_predicted_reference_val_results.json"
TOLERANCE = 1.0e-6
DIRECTIONS = ("above", "below", "left", "right")
MODELS = ("Z-B3", "D-B1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    from buildreasonseg_mvp.task6q_reference_resolver import is_eligible
    from buildreasonseg_mvp.task7a_l3_pipeline import L3Pipeline

    records = read_rows(VAL_ROWS)
    pair_payload = json.loads(PAIRS.read_text(encoding="utf-8"))
    pairs = pair_payload["pairs_list"]
    by_id = {record["sample_id"]: record for record in records}
    missing = [f"{model}/{seed}" for model in MODELS for seed in SEEDS
               if not (CHECKPOINT_ROOT / model / str(seed) / "best.pt").is_file()]
    if missing:
        write_json(OUT, {"_doc": "Task 7I section 23.", "task": "7I",
                         "verdict": "FORMAL_TRAINING_INCOMPLETE", "missing_checkpoints": missing})
        print(f"[7i.pred] STOP FORMAL_TRAINING_INCOMPLETE ({missing})", flush=True)
        return 4

    pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device)
    store = pipeline.store
    masks = MaskStore()
    eligible_cache: dict[str, list] = {}

    def eligible_of(tile_id: str, image_path) -> list:
        """One shared U-C1 proposal pass per tile, reused by every model/seed."""

        if tile_id not in eligible_cache:
            proposals = pipeline.proposals(tile_id, Path(image_path))
            eligible_cache[tile_id] = [proposal for proposal in proposals
                                       if is_eligible(proposal, "largest")]
        return eligible_cache[tile_id]

    # ---------------- reference metrics computed once, shared across all six runs
    reference_rows = []
    reference_masks: dict[str, np.ndarray | None] = {}
    for record in records:
        eligible = eligible_of(record["tile_id"], record["image_path"])
        gt_reference = np.asarray(masks.mask(record["tile_id"],
                                             record["reference_source_feature_id"]), dtype=bool)
        best_iou = max((iou_of(np.asarray(proposal.mask, dtype=bool), gt_reference)
                        for proposal in eligible), default=0.0)
        if not eligible:
            bucket = "NO_PROPOSALS" if not pipeline.proposals(record["tile_id"],
                                                              Path(record["image_path"])) \
                else "NO_ELIGIBLE_PROPOSALS"
            reference_masks[record["sample_id"]] = None
            reference_rows.append({"sample_id": record["sample_id"], "abstained": True,
                                   "bucket": bucket, "bucket_semantic": BUCKET_MAP[bucket],
                                   "reference_iou": None, "reference_dice": None,
                                   "reference_precision_at_0_5": None, "best_eligible_iou": best_iou})
            continue
        selected = max(eligible, key=lambda proposal: (int(np.asarray(proposal.mask).sum()),
                                                       float(proposal.confidence),
                                                       -int(getattr(proposal, "index", 0))))
        mask = np.asarray(selected.mask, dtype=bool)
        reference_masks[record["sample_id"]] = mask
        intersection = float((mask & gt_reference).sum())
        reference_iou = iou_of(mask, gt_reference)
        if reference_iou < 0.50:
            bucket = ("REFERENCE_SELECTION_WRONG" if best_iou >= 0.50
                      else "REFERENCE_NOT_COVERED_IOU50")
        else:
            bucket = "REFERENCE_OK"
        reference_rows.append({
            "sample_id": record["sample_id"], "abstained": False, "bucket": bucket,
            "bucket_semantic": BUCKET_MAP[bucket], "reference_iou": reference_iou,
            "reference_dice": dice_of(mask, gt_reference),
            "reference_precision_at_0_5": (intersection + TOLERANCE)
            / (float(mask.sum()) + TOLERANCE),
            "best_eligible_iou": best_iou})
    answered = [row for row in reference_rows if not row["abstained"]]
    reference_metrics = {
        "records": len(reference_rows), "answered": len(answered),
        "abstentions": len(reference_rows) - len(answered),
        "abstention_rate": (len(reference_rows) - len(answered)) / max(1, len(reference_rows)),
        "reference_miou": float(np.mean([row["reference_iou"] for row in answered])),
        "reference_dice": float(np.mean([row["reference_dice"] for row in answered])),
        "reference_precision_at_0_5": float(np.mean([row["reference_precision_at_0_5"]
                                                    for row in answered])),
        "buckets": dict(Counter(row["bucket_semantic"] for row in reference_rows)),
        "raw_buckets": dict(Counter(row["bucket"] for row in reference_rows)),
        "best_eligible_coverage_at_0_50": float(np.mean([row["best_eligible_iou"] >= 0.50
                                                         for row in reference_rows])),
    }

    # ---------------- per model/seed target metrics (Task 7I checkpoints, frozen prediction paths)
    def predict_with(model, model_name: str, mask: np.ndarray, record: dict) -> np.ndarray:
        synthetic_id = -9
        overridden = dict(record)
        overridden["reference_source_feature_id"] = synthetic_id
        overridden["sample_id"] = f"{record['sample_id']}:i"
        wrapper = _ReferenceOverride(masks, {(record["tile_id"], synthetic_id): mask})
        if model_name == "Z-B3":
            from scripts.task6z_evaluate import predict as predict_z_b3

            return predict_z_b3(model, [overridden], store, wrapper, args.device, {},
                                batch_size=1)[0]
        from scripts.task7d_evaluate import predict as predict_d_b1

        return predict_d_b1(model, [overridden], store, wrapper, args.device, {}, batch_size=1)[0]

    results = {}
    for model_name in MODELS:
        for seed in SEEDS:
            key = f"{model_name}/{seed}"
            checkpoint = CHECKPOINT_ROOT / model_name / str(seed) / "best.pt"
            payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
            model = (L3TargetDecoder("Z-B3") if model_name == "Z-B3"
                     else GlobalCompetitionDecoder("D-B1")).to(args.device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            rows = []
            for record in records:
                mask = reference_masks[record["sample_id"]]
                truth = np.asarray(masks.mask(record["tile_id"],
                                              record["target_source_feature_id"]), dtype=bool)
                if mask is None:
                    rows.append({"sample_id": record["sample_id"], "direction": record["direction"],
                                 "miou": 0.0, "dice": 0.0, "precision_at_0_5": 0.0,
                                 "abstained": True, "reference_ok": False,
                                 "reference_iou": 0.0, "program_id": record["program_id"]})
                    continue
                prediction = predict_with(model, model_name, mask, record)
                intersection = float((prediction & truth).sum())
                reference_row = next(row for row in reference_rows
                                     if row["sample_id"] == record["sample_id"])
                rows.append({"sample_id": record["sample_id"], "direction": record["direction"],
                             "program_id": record["program_id"],
                             "miou": (intersection + TOLERANCE)
                             / (float((prediction | truth).sum()) + TOLERANCE),
                             "dice": dice_of(prediction, truth),
                             "precision_at_0_5": (intersection + TOLERANCE)
                             / (float(prediction.sum()) + TOLERANCE),
                             "abstained": False,
                             "reference_ok": reference_row["bucket"] == "REFERENCE_OK",
                             "reference_iou": reference_row["reference_iou"]})

            def summarise(subset: list[dict]) -> dict:
                if not subset:
                    return {"records": 0}
                answered_subset = [row for row in subset if not row["abstained"]]
                return {"records": len(subset),
                        "miou": float(np.mean([row["miou"] for row in subset])),
                        "dice": float(np.mean([row["dice"] for row in subset])),
                        "precision_at_0_5": float(np.mean([row["precision_at_0_5"]
                                                          for row in subset])),
                        "answered": len(answered_subset),
                        "answered_only_miou": float(np.mean([row["miou"]
                                                             for row in answered_subset]))
                        if answered_subset else None,
                        "answered_only_dice": float(np.mean([row["dice"]
                                                             for row in answered_subset]))
                        if answered_subset else None}

            # counterfactual pairs share one predicted reference per pair
            pair_results = {"passed": 0, "pairs": len(pairs), "reference_abstention_pairs": 0}
            own_values, cross_values = [], []
            for pair in pairs:
                members = [by_id[member["sample_id"]] for member in pair["members"]]
                mask = reference_masks[members[0]["sample_id"]]
                if mask is None:
                    pair_results["reference_abstention_pairs"] += 1
                    continue
                own, cross = [], []
                for position, member in enumerate(members):
                    prediction = predict_with(model, model_name, mask, member)
                    own_mask = np.asarray(masks.mask(member["tile_id"],
                                                     member["target_source_feature_id"]), dtype=bool)
                    other = members[1 - position]
                    other_mask = np.asarray(masks.mask(other["tile_id"],
                                                       other["target_source_feature_id"]), dtype=bool)
                    own.append(iou_of(prediction, own_mask))
                    cross.append(iou_of(prediction, other_mask))
                if own[0] > cross[0] and own[1] > cross[1]:
                    pair_results["passed"] += 1
                own_values.extend(own)
                cross_values.extend(cross)
            pair_results.update({
                "pass_rate": pair_results["passed"] / max(1, len(pairs)),
                "mean_own_iou": float(np.mean(own_values)) if own_values else None,
                "mean_cross_iou": float(np.mean(cross_values)) if cross_values else None,
                "own_cross_margin": (float(np.mean(own_values) - np.mean(cross_values))
                                     if own_values else None)})

            results[key] = {
                "model": model_name, "seed": seed,
                "checkpoint_sha256": sha256_file(CHECKPOINT_ROOT / model_name / str(seed)
                                                 / "best.pt"),
                "strict": summarise(rows),
                "reference_ok_subset": summarise([row for row in rows if row["reference_ok"]]),
                "per_direction": {direction: summarise([row for row in rows
                                                        if row["direction"] == direction])
                                  for direction in DIRECTIONS},
                "pairs": pair_results,
            }
            print(f"[7i.pred] {key}: strict {results[key]['strict']['miou']:.4f} answered-only "
                  f"{(results[key]['strict']['answered_only_miou'] or 0.0):.4f} refOK "
                  f"{(results[key]['reference_ok_subset']['miou'] or 0.0):.4f} pairs "
                  f"{pair_results['passed']}/{pair_results['pairs']}", flush=True)
            del model
            if args.device != "cpu":
                torch.cuda.empty_cache()

    write_json(OUT, {
        "_doc": ("Task 7I sections 21-23. Practical predicted-reference full-val evaluation of the six frozen "
                 "formal checkpoints: frozen U-C1 proposals + deterministic largest selector (no Task 7G "
                 "selector), reference metrics computed once over all 936 records and reused, target metrics "
                 "per model/seed and the complete I-FormalValPairsAll set with a shared predicted reference "
                 "per pair. Predicted-reference results did not affect checkpoint selection."),
        "task": "7I", "stage": "J-predicted-reference-val-evaluation",
        "resolver": effective_resolver_report(),
        "population": {"val_records": len(records), "pairs": len(pairs)},
        "reference": reference_metrics, "reference_rows": reference_rows,
        "results": results, "models": list(MODELS), "seeds": list(SEEDS),
        "shared_reference_cache": {"tiles": len(eligible_cache), "computed_once": True},
        "used_for_checkpoint_selection": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7i.pred] reference mIoU {reference_metrics['reference_miou']:.4f} abstentions "
          f"{reference_metrics['abstentions']} buckets {reference_metrics['buckets']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
