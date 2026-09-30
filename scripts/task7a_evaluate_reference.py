"""Task 7A Parts D-E — A0 oracle reproduction and the A1 canonical predicted-reference chain.

A0 (section 7): re-run the frozen Task 6Z Z-B3 decoder on the exact Z-MiniVal240 and Z-PairedVal20 packs
with the **oracle** reference and require byte-level agreement with the frozen Task 6Z metrics
(mIoU/Dice absolute delta <= 1e-6, paired exactly 15/20, margin delta <= 1e-6). Failure stops the task with
`TASK6Z_REPRODUCTION_FAIL`.

A1 (sections 8-11): replace the oracle reference with the frozen U-C1 deterministic `largest` resolver and
report reference diagnostics vs GT (scoring only), target metrics, and the same-reference paired
counterfactual where the predicted reference is computed once per tile and reused for both directions.

Writes `evaluation/task7a_oracle_reproduction.json`,
`evaluation/task7a_predicted_reference_quality.json`,
`evaluation/task7a_canonical_predicted_reference_val.json` and
`evaluation/task7a_canonical_predicted_reference_paired.json`.

    python scripts/task7a_evaluate_reference.py
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
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import (  # noqa: E402
    PROGRAM_DECOMPOSITION,
    SUPPORTED_L3_PROGRAMS,
    L3Pipeline,
    default_target_checkpoint,
    pipeline_report,
    sha256_file,
)
from scripts.task6u_common import centroid_error, dice, iou, percentile  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
OUT_A0 = EVAL / "task7a_oracle_reproduction.json"
OUT_REF = EVAL / "task7a_predicted_reference_quality.json"
OUT_VAL = EVAL / "task7a_canonical_predicted_reference_val.json"
OUT_PAIRED = EVAL / "task7a_canonical_predicted_reference_paired.json"
TASK6Z = EVAL / "task6z_mini_val.json"
TASK6Z_PAIRED = EVAL / "task6z_paired_val.json"
TASK6Z_TRAINING = EVAL / "task6z_training.json"
TOLERANCE = 1.0e-6
REFERENCE_BUCKETS = ("NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "REFERENCE_NOT_COVERED_IOU50",
                     "REFERENCE_SELECTION_WRONG", "REFERENCE_GEOMETRY_POOR", "REFERENCE_OK")


def load_pack(name: str) -> list[dict]:
    return json.loads((PACK_ROOT / f"{name}.json").read_text(encoding="utf-8"))["records"]


def image_path_of(record: dict) -> Path:
    path = Path(record["image_path"])
    return path if path.is_absolute() else REPO_ROOT / path


def gt_reference_mask(masks, record: dict) -> np.ndarray:
    return np.asarray(masks.mask(record["tile_id"], record["reference_source_feature_id"]),
                      dtype=bool)


def oracle_predictions_6z(records: list[dict], store, masks, device, precomputed: dict,
                          batch_size: int = 8) -> list[np.ndarray]:
    """Reproduce the frozen Task 6Z inference path bit-for-bit (same model, batching and pack order)."""

    from scripts.task6z_evaluate import load_variant, predict

    model = load_variant("Z-B3", device)
    return predict(model, records, store, masks, device, precomputed, batch_size=batch_size)


def target_row(prediction: np.ndarray, truth: np.ndarray, record: dict) -> dict:
    intersection = float((prediction & truth).sum())
    union = float((prediction | truth).sum())
    return {
        "sample_id": record["sample_id"], "direction": record["direction"],
        "program_id": record["program_id"],
        "miou": (intersection + TOLERANCE) / (union + TOLERANCE),
        "dice": (2.0 * intersection + TOLERANCE)
        / (float(prediction.sum()) + float(truth.sum()) + TOLERANCE),
        "precision_at_0_5": (intersection + TOLERANCE) / (float(prediction.sum()) + TOLERANCE),
        "target_area_px": float(truth.sum()),
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    answered = [row for row in rows if not row.get("abstained", False)]
    return {
        "records": len(rows),
        "miou": float(np.mean([row["miou"] for row in rows])),
        "dice": float(np.mean([row["dice"] for row in rows])),
        "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows])),
        "answered": len(answered),
        "answered_only_miou": float(np.mean([row["miou"] for row in answered]))
        if answered else None,
        "answered_only_dice": float(np.mean([row["dice"] for row in answered]))
        if answered else None,
    }


def reference_bucket(*, proposals: int, eligible: int, selected_iou: float | None,
                     centroid: float | None, best_eligible_iou: float) -> str:
    """Section 9 buckets: NOT_COVERED needs *no* eligible proposal at IoU >= 0.50."""

    if proposals == 0:
        return "NO_PROPOSALS"
    if eligible == 0:
        return "NO_ELIGIBLE_PROPOSALS"
    if selected_iou is None:
        return "NO_ELIGIBLE_PROPOSALS"
    if selected_iou < 0.50:
        return ("REFERENCE_SELECTION_WRONG" if best_eligible_iou >= 0.50
                else "REFERENCE_NOT_COVERED_IOU50")
    if centroid is not None and centroid > 0.05:
        return "REFERENCE_GEOMETRY_POOR"
    return "REFERENCE_OK"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    task6z_training = json.loads(TASK6Z_TRAINING.read_text(encoding="utf-8"))
    expected_sha = task6z_training["results"]["Z-B3"]["checkpoint"]["sha256"]
    target_path = default_target_checkpoint()
    if not target_path.is_file() or sha256_file(target_path) != expected_sha:
        write_json(OUT_A0, {"_doc": "Task 7A section 3.", "task": "7A",
                            "verdict": "L3_CHECKPOINT_UNAVAILABLE",
                            "expected_sha256": expected_sha})
        print("[7a.a0] STOP L3_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2

    pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device,
                          target_checkpoint=target_path, expected_target_sha256=expected_sha)
    from task6n_train import MaskStore

    masks = MaskStore()
    records = load_pack("z_mini_val_240")
    paired_records = load_pack("z_paired_val20")
    pairs = [{"tile_id": paired_records[index]["tile_id"], "a": paired_records[index],
              "b": paired_records[index + 1]} for index in range(0, len(paired_records), 2)]

    # ---------------------------------------------------------------- A0 oracle reproduction
    task6z = json.loads(TASK6Z.read_text(encoding="utf-8"))
    task6z_paired = json.loads(TASK6Z_PAIRED.read_text(encoding="utf-8"))
    precomputed_oracle: dict[str, dict] = {}
    oracle_predictions = oracle_predictions_6z(records, pipeline.store, masks, args.device,
                                               precomputed_oracle)
    oracle_rows = []
    for record, prediction in zip(records, oracle_predictions):
        truth = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                           dtype=bool)
        oracle_rows.append(target_row(prediction, truth, record))
    oracle_metrics = aggregate(oracle_rows)
    frozen_b3 = task6z["results"]["Z-B3"]["overall"]
    reference_miou = frozen_b3["miou"]
    reference_dice = frozen_b3["dice"]
    miou_delta = abs(oracle_metrics["miou"] - reference_miou)
    dice_delta = abs(oracle_metrics["dice"] - reference_dice)

    paired_oracle = []
    for entry in pairs:
        members = [entry["a"], entry["b"]]
        pair_predictions = oracle_predictions_6z(members, pipeline.store, masks, args.device,
                                                 precomputed_oracle, batch_size=2)
        own, cross = [], []
        for index, member in enumerate(members):
            prediction = pair_predictions[index]
            own_mask = np.asarray(masks.mask(member["tile_id"], member["target_source_feature_id"]),
                                 dtype=bool)
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other["tile_id"], other["target_source_feature_id"]),
                                    dtype=bool)
            intersection_own = float((prediction & own_mask).sum())
            intersection_cross = float((prediction & other_mask).sum())
            own.append(intersection_own / (float((prediction | own_mask).sum()) + TOLERANCE))
            cross.append(intersection_cross / (float((prediction | other_mask).sum()) + TOLERANCE))
        paired_oracle.append({"tile_id": entry["tile_id"], "own": own, "cross": cross,
                              "passes": bool(own[0] > cross[0] and own[1] > cross[1])})
    paired_passed = sum(1 for row in paired_oracle if row["passes"])
    own_mean = float(np.mean([value for row in paired_oracle for value in row["own"]]))
    cross_mean = float(np.mean([value for row in paired_oracle for value in row["cross"]]))
    margin = own_mean - cross_mean
    frozen_margin = task6z_paired["systems"]["Z-B3"]["own_cross_margin"]
    margin_delta = abs(margin - frozen_margin)

    reproduction_ok = bool(miou_delta <= TOLERANCE and dice_delta <= TOLERANCE
                           and paired_passed == 15 and margin_delta <= TOLERANCE)
    write_json(OUT_A0, {
        "_doc": ("Task 7A section 7. A0 oracle reproduction: the frozen Task 6Z Z-B3 decoder re-run on the "
                 "exact Z-MiniVal240 and Z-PairedVal20 packs with the oracle largest reference. Every "
                 "tolerance is absolute and predeclared (1e-6); no threshold was tuned."),
        "task": "7A", "stage": "A0-oracle-reproduction",
        "checkpoint": {"path": str(target_path), "sha256": sha256_file(target_path),
                       "expected_sha256": expected_sha, "variant": pipeline.target_variant,
                       "matches": sha256_file(target_path) == expected_sha},
        "pack": {"mini_val_240": len(records), "paired_val20_pairs": len(pairs)},
        "recomputed": {"miou": oracle_metrics["miou"], "dice": oracle_metrics["dice"],
                       "paired_passed": paired_passed, "paired_pairs": len(pairs),
                       "own_cross_margin": margin, "own_mean": own_mean, "cross_mean": cross_mean},
        "frozen_task6z": {"miou": reference_miou, "dice": reference_dice,
                          "paired_passed": task6z_paired["systems"]["Z-B3"]["passed"],
                          "own_cross_margin": frozen_margin},
        "deltas": {"miou": miou_delta, "dice": dice_delta, "margin": margin_delta},
        "tolerance": TOLERANCE,
        "reproduction_passed": reproduction_ok,
        "verdict": "TASK6Z_REPRODUCTION_PASS" if reproduction_ok else "TASK6Z_REPRODUCTION_FAIL",
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7a.a0] mIoU {oracle_metrics['miou']:.6f} (Δ{miou_delta:.2e}) Dice "
          f"{oracle_metrics['dice']:.6f} (Δ{dice_delta:.2e}) paired {paired_passed}/20 margin "
          f"{margin:+.6f} (Δ{margin_delta:.2e}) -> "
          f"{'PASS' if reproduction_ok else 'FAIL'}", flush=True)
    if not reproduction_ok:
        return 3

    # ---------------------------------------------------------------- A1 predicted reference
    reference_rows, target_rows = [], []
    for record in records:
        truth_reference = gt_reference_mask(masks, record)
        proposals = pipeline.proposals(record["tile_id"], image_path_of(record))
        bundle = pipeline.resolve_reference(proposals)
        row = {"sample_id": record["sample_id"], "direction": record["direction"],
               "tile_id": record["tile_id"], "proposal_count": len(proposals),
               "eligible_count": len(bundle.eligible),
               "abstained": bundle.selection is None,
               "abstention_reason": bundle.abstention_reason}
        best_eligible = max((iou(proposal.mask, truth_reference) for proposal in bundle.eligible),
                            default=0.0)
        row["best_eligible_coverage_at_0_5"] = float(best_eligible >= 0.50)
        if bundle.selection is None:
            row.update({"reference_iou": None, "reference_dice": None, "centroid_error": None,
                        "bucket": "NO_PROPOSALS" if not proposals else "NO_ELIGIBLE_PROPOSALS"})
        else:
            selected = np.asarray(bundle.selection["mask"], dtype=bool)
            row.update({"reference_iou": iou(selected, truth_reference),
                        "reference_dice": dice(selected, truth_reference),
                        "centroid_error": centroid_error(selected, truth_reference)})
            row["bucket"] = reference_bucket(proposals=len(proposals),
                                             eligible=len(bundle.eligible),
                                             selected_iou=row["reference_iou"],
                                             centroid=row["centroid_error"],
                                             best_eligible_iou=best_eligible)
        reference_rows.append(row)
        if bundle.selection is None:
            target_rows.append({"sample_id": record["sample_id"], "direction": record["direction"],
                                "miou": 0.0, "dice": 0.0, "precision_at_0_5": 0.0,
                                "abstained": True, "bucket": row["bucket"]})
            continue
        prediction = pipeline.predict_target(image_path_of(record), record["tile_id"],
                                             record["program_id"],
                                             np.asarray(bundle.selection["mask"], dtype=bool))
        truth_target = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                                  dtype=bool)
        target = target_row(prediction["mask"], truth_target, record)
        target["abstained"] = False
        target["bucket"] = row["bucket"]
        target["reference_iou"] = row["reference_iou"]
        target_rows.append(target)

    answered_reference = [row for row in reference_rows if row["reference_iou"] is not None]
    reference_metrics = {
        "records": len(reference_rows),
        "proposal_count_mean": float(np.mean([row["proposal_count"] for row in reference_rows])),
        "eligible_count_mean": float(np.mean([row["eligible_count"] for row in reference_rows])),
        "reference_abstentions": sum(1 for row in reference_rows if row["abstained"]),
        "reference_abstention_rate": sum(1 for row in reference_rows if row["abstained"])
        / len(reference_rows),
        "selected_reference_miou": float(np.mean([row["reference_iou"] for row in answered_reference]))
        if answered_reference else None,
        "selected_reference_dice": float(np.mean([row["reference_dice"]
                                                  for row in answered_reference]))
        if answered_reference else None,
        "precision_at_0_5": float(np.mean([row["reference_iou"] >= 0.50
                                           for row in answered_reference]))
        if answered_reference else None,
        "centroid_error_mean": float(np.mean([row["centroid_error"]
                                              for row in answered_reference]))
        if answered_reference else None,
        "centroid_error_median": float(np.median([row["centroid_error"]
                                                  for row in answered_reference]))
        if answered_reference else None,
        "centroid_error_p90": percentile([row["centroid_error"] for row in answered_reference], 0.90),
        "best_eligible_coverage_at_0_5": float(np.mean([row["best_eligible_coverage_at_0_5"]
                                                        for row in reference_rows])),
        "buckets": {name: sum(1 for row in reference_rows if row["bucket"] == name)
                    for name in REFERENCE_BUCKETS},
    }
    write_json(OUT_REF, {
        "_doc": ("Task 7A section 9. Predicted-reference diagnostics for the frozen U-C1 deterministic "
                 "largest resolver on Z-MiniVal240. The GT reference is used for scoring only and never "
                 "enters inference."),
        "task": "7A", "stage": "A1-reference-quality",
        "pack": {"name": "z_mini_val_240", "records": len(records)},
        "pipeline": pipeline_report(),
        "reference": reference_metrics,
        "buckets_order": list(REFERENCE_BUCKETS),
        "rows": reference_rows,
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })

    strict = aggregate(target_rows)
    answered = [row for row in target_rows if not row["abstained"]]
    reference_ok_rows = [row for row in target_rows if row["bucket"] == "REFERENCE_OK"
                         and not row["abstained"]]
    reference_fail_rows = [row for row in target_rows if row["bucket"] != "REFERENCE_OK"]
    val_payload = {
        "_doc": ("Task 7A section 10. A1 canonical-program predicted-reference chain on Z-MiniVal240: the "
                 "canonical program id plus the source image are the only inputs; the oracle reference, GT "
                 "target, source feature ids and GT candidate ids are never provided."),
        "task": "7A", "stage": "A1-canonical-predicted-reference-val",
        "pack": {"name": "z_mini_val_240", "records": len(records)},
        "strict": strict,
        "by_direction": {direction: aggregate([row for row in target_rows
                                               if row["direction"] == direction])
                         for direction in ("above", "below", "left", "right")},
        "reference_ok_subset": aggregate(reference_ok_rows),
        "reference_fail_subset": aggregate(reference_fail_rows),
        "answered_only": aggregate(answered),
        "abstentions": len(target_rows) - len(answered),
        "abstention_rate": (len(target_rows) - len(answered)) / len(target_rows),
        "oracle_task6z_miou": reference_miou,
        "retention": strict["miou"] / reference_miou if reference_miou else None,
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_VAL, val_payload)

    paired_rows = []
    reference_abstention_pairs = 0
    for entry in pairs:
        tile_id = entry["tile_id"]
        image_path = image_path_of(entry["a"])
        proposals = pipeline.proposals(tile_id, image_path)
        bundle = pipeline.resolve_reference(proposals)  # once per pair/tile
        row = {"tile_id": tile_id, "abstained": bundle.selection is None}
        if bundle.selection is None:
            reference_abstention_pairs += 1
            row.update({"own": [0.0, 0.0], "cross": [0.0, 0.0], "passes": False})
            paired_rows.append(row)
            continue
        reference_mask = np.asarray(bundle.selection["mask"], dtype=bool)
        members = [entry["a"], entry["b"]]
        own, cross = [], []
        for index, member in enumerate(members):
            prediction = pipeline.predict_target(image_path, member["tile_id"], member["program_id"],
                                                 reference_mask)["mask"]
            own_mask = np.asarray(masks.mask(member["tile_id"], member["target_source_feature_id"]),
                                 dtype=bool)
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other["tile_id"], other["target_source_feature_id"]),
                                   dtype=bool)
            own.append(float((prediction & own_mask).sum())
                       / (float((prediction | own_mask).sum()) + TOLERANCE))
            cross.append(float((prediction & other_mask).sum())
                         / (float((prediction | other_mask).sum()) + TOLERANCE))
        row.update({"own": own, "cross": cross,
                    "passes": bool(own[0] > cross[0] and own[1] > cross[1])})
        paired_rows.append(row)
    passed = sum(1 for row in paired_rows if row["passes"])
    own_mean = float(np.mean([value for row in paired_rows for value in row["own"]]))
    cross_mean = float(np.mean([value for row in paired_rows for value in row["cross"]]))
    paired_payload = {
        "_doc": ("Task 7A section 11. A1 same-reference paired counterfactual on Z-PairedVal20: the "
                 "predicted largest reference is computed once per tile and reused for both member "
                 "programs, so only the direction changes."),
        "task": "7A", "stage": "A1-canonical-predicted-reference-paired",
        "pack": {"name": "z_paired_val20", "pairs": len(paired_rows)},
        "passed": passed, "pairs": len(paired_rows), "pass_rate": passed / max(1, len(paired_rows)),
        "mean_own_iou": own_mean, "mean_cross_iou": cross_mean,
        "own_cross_margin": own_mean - cross_mean,
        "reference_abstention_pairs": reference_abstention_pairs,
        "reference_reused_within_pair": True,
        "rows": paired_rows,
        "training_performed": False, "test_split_used": False,
    }
    write_json(OUT_PAIRED, paired_payload)
    print(f"[7a.a1] reference mIoU {reference_metrics['selected_reference_miou']:.4f} Pr@0.5 "
          f"{reference_metrics['precision_at_0_5']:.4f} abstain "
          f"{reference_metrics['reference_abstention_rate']:.4f} buckets "
          f"{reference_metrics['buckets']}", flush=True)
    print(f"[7a.a1] strict mIoU {strict['miou']:.4f} answered {strict['answered_only_miou']:.4f} "
          f"retention {val_payload['retention']:.4f} | paired {passed}/{len(paired_rows)} margin "
          f"{paired_payload['own_cross_margin']:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
