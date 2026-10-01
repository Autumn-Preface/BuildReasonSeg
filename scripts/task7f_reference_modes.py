"""Task 7F Parts C-E — verify the Task 7E holdout, build the four reference modes and audit them.

Verifies the Task 7E manifest hashes (record-id hash, pair-id hash, zero test, zero Task 6Z MiniVal/Paired
overlap) and stops with `TASK7E_HOLDOUT_MISMATCH` otherwise. Then, for every `E-HoldoutL3` record, runs the
frozen U-C1 proposal inference **once** per tile and builds F-R0/F-R1/F-R2/F-R3 from that single proposal set
(plus the canonical GT reference, used only for the declared diagnostic ceilings), audits the reference modes
and runs the frozen D-B1 downstream once per mode, caching every per-record row in the gitignored
`artifacts/task7f/`. Finally it measures the paired counterfactual ceilings on `E-PairedHoldout20`, resolving
the reference once per pair and reusing it for both members and all four modes.

Writes `evaluation/task7f_reference_modes.json` and `evaluation/task7f_paired_reference_modes.json`.

    python scripts/task7f_reference_modes.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import L3Pipeline  # noqa: E402
from buildreasonseg_mvp.task7e_l3_decoder_adapter import FrozenL3Decoder, frozen_metadata  # noqa: E402
from buildreasonseg_mvp.task7f_reference_ceiling import (  # noqa: E402
    CACHE_PATH,
    COVERAGE_BANDS,
    COVERAGE_IOU,
    D_B1_SHA256,
    DIRECTIONS,
    HOLDOUT_ROOT,
    MODES,
    MODE_LABELS,
    TOLERANCE,
    YOLO_SHA256,
    best_eligible_iou,
    build_mode_masks,
    centroid_error,
    dice_of,
    iou_of,
    load_holdout_rows,
    load_pairs,
    sha256_file,
    sha256_text,
    write_cache,
)
from scripts.task6n_train import MaskStore  # noqa: E402
from scripts.task6u_common import CONFIGS, PROPOSAL_CHECKPOINT  # noqa: E402
from scripts.task7e_evaluate_predicted_reference import (  # noqa: E402
    _ReferenceOverride,
    effective_resolver_report,
)

EVAL = REPO_ROOT / "evaluation"
OUT_MODES = EVAL / "task7f_reference_modes.json"
OUT_PAIRED = EVAL / "task7f_paired_reference_modes.json"
MANIFEST = EVAL / "task7e_holdout_manifest.json"


def reference_row(record: dict, mode: str, mask, gt_reference: np.ndarray, detail: dict) -> dict:
    abstained = mask is None
    row = {"sample_id": record["sample_id"], "mode": mode, "direction": record["direction"],
           "program_id": record["program_id"], "tile_id": record["tile_id"],
           "abstained": abstained,
           "reference_iou": None, "reference_dice": None, "reference_precision_at_0_5": None,
           "centroid_error": None,
           "selected_confidence": detail.get("selected_confidence"),
           "selected_area": detail.get("selected_area"),
           "selected_vs_best_iou": detail.get("selected_vs_best_iou"),
           "reference_iou_gap": detail.get("reference_iou_gap"),
           "oracle_selected_confidence": detail.get("oracle_selected_confidence"),
           "oracle_selected_area": detail.get("oracle_selected_area"),
           "best_eligible_iou": detail["best_eligible_iou"],
           "covered50": detail["covered50"], "eligible_count": detail["eligible_count"]}
    if not abstained:
        reference = np.asarray(mask, dtype=bool)
        intersection = float((reference & gt_reference).sum())
        row.update({
            "reference_iou": (intersection + TOLERANCE)
            / (float((reference | gt_reference).sum()) + TOLERANCE),
            "reference_dice": dice_of(reference, gt_reference),
            "reference_precision_at_0_5": (intersection + TOLERANCE)
            / (float(reference.sum()) + TOLERANCE),
            "centroid_error": centroid_error(reference, gt_reference),
        })
    return row


def summarise_references(rows: list[dict]) -> dict:
    answered = [row for row in rows if not row["abstained"]]
    centroids = [row["centroid_error"] for row in answered if row["centroid_error"] is not None]
    return {
        "records": len(rows),
        "answered": len(answered),
        "abstentions": len(rows) - len(answered),
        "abstention_rate": (len(rows) - len(answered)) / max(1, len(rows)),
        "reference_miou": float(np.mean([row["reference_iou"] for row in answered]))
        if answered else None,
        "reference_dice": float(np.mean([row["reference_dice"] for row in answered]))
        if answered else None,
        "reference_precision_at_0_5": float(np.mean([row["reference_precision_at_0_5"]
                                                    for row in answered])) if answered else None,
        "centroid_error_median": float(np.median(centroids)) if centroids else None,
        "centroid_error_p90": float(np.percentile(centroids, 90)) if centroids else None,
    }


def summarise_targets(rows: list[dict]) -> dict:
    answered = [row for row in rows if not row["abstained"]]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metadata = frozen_metadata()
    records = load_holdout_rows()
    pairs = load_pairs()
    record_id_hash = sha256_text("\n".join(sorted(record["sample_id"] for record in records)))
    pair_id_hash = sha256_text("\n".join(sorted(pair["pair_id"] for pair in pairs)))
    checks = {
        "records": len(records), "records_expected": manifest["holdout"]["records"],
        "record_id_hash": record_id_hash,
        "record_id_hash_expected": manifest["holdout"]["record_id_hash"],
        "record_id_hash_ok": record_id_hash == manifest["holdout"]["record_id_hash"],
        "pairs": len(pairs), "pairs_expected": manifest["paired"]["pairs"],
        "pair_id_hash": pair_id_hash,
        "pair_id_hash_expected": manifest["paired"]["pair_id_hash"],
        "pair_id_hash_ok": pair_id_hash == manifest["paired"]["pair_id_hash"],
        "no_test": manifest["checks"]["test_records"] == 0,
        "overlap_minival": manifest["checks"]["overlap_with_z_mini_val_240"] == 0,
        "overlap_pairedval": manifest["checks"]["overlap_with_z_paired_val20"] == 0,
        "d_b1_sha256": metadata["d_b1"]["sha256"], "d_b1_sha256_ok": metadata["d_b1"]["matches"],
        "yolo_sha256": sha256_file(PROPOSAL_CHECKPOINT), "yolo_sha256_ok":
            sha256_file(PROPOSAL_CHECKPOINT) == YOLO_SHA256,
    }
    if not all([checks["record_id_hash_ok"], checks["pair_id_hash_ok"], checks["no_test"],
                checks["overlap_minival"], checks["overlap_pairedval"], checks["d_b1_sha256_ok"]]):
        write_json(OUT_MODES, {"_doc": "Task 7F section 5.", "task": "7F",
                               "verdict": "TASK7E_HOLDOUT_MISMATCH", "checks": checks})
        print("[7f.modes] STOP TASK7E_HOLDOUT_MISMATCH", flush=True)
        return 2

    pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device)
    store = pipeline.store
    masks = MaskStore()
    decoder = FrozenL3Decoder("D-B1", args.device)
    reference_cache: dict[str, list] = {}

    def proposals_of(tile_id: str, image_path) -> list:
        if tile_id not in reference_cache:
            raw = pipeline.proposals(tile_id, Path(image_path))
            reference_cache[tile_id] = [proposal for proposal in raw
                                        if _is_eligible(proposal)]
        return reference_cache[tile_id]

    def predict_with(reference_mask: np.ndarray, record: dict) -> np.ndarray:
        synthetic_id = -1
        overridden = dict(record)
        overridden["reference_source_feature_id"] = synthetic_id
        overridden["sample_id"] = f"{record['sample_id']}:f7f"
        wrapper = _ReferenceOverride(masks, {(record["tile_id"], synthetic_id): reference_mask})
        return decoder.predict([overridden], store, wrapper, {}, batch_size=1)[0]

    reference_rows: dict[str, list] = {mode: [] for mode in MODES}
    target_rows: dict[str, list] = {mode: [] for mode in MODES}
    coverage_stats = {"covered50": 0, "uncovered": 0, "bands": {str(band): 0 for band in COVERAGE_BANDS}}
    per_record_modes = []
    for record in records:
        eligible = proposals_of(record["tile_id"], record["image_path"])
        gt_reference = np.asarray(masks.mask(record["tile_id"],
                                             record["reference_source_feature_id"]), dtype=bool)
        built = build_mode_masks(eligible, gt_reference)
        detail = built["detail"]
        if detail["covered50"]:
            coverage_stats["covered50"] += 1
        else:
            coverage_stats["uncovered"] += 1
        best = best_eligible_iou(eligible, gt_reference)
        for band in COVERAGE_BANDS:
            if best >= band:
                coverage_stats["bands"][str(band)] += 1
        gt_target = np.asarray(masks.mask(record["tile_id"],
                                          record["target_source_feature_id"]), dtype=bool)
        entry = {"sample_id": record["sample_id"], "covered50": detail["covered50"],
                 "best_eligible_iou": best}
        for mode in MODES:
            mask = built["masks"][mode]
            reference_rows[mode].append(reference_row(record, mode, mask, gt_reference, detail))
            if mask is None:
                target_rows[mode].append({"sample_id": record["sample_id"],
                                          "direction": record["direction"], "mode": mode,
                                          "miou": 0.0, "dice": 0.0, "precision_at_0_5": 0.0,
                                          "abstained": True})
                entry[mode] = {"miou": 0.0, "dice": 0.0, "precision_at_0_5": 0.0, "abstained": True}
                continue
            prediction = predict_with(mask, record)
            intersection = float((prediction & gt_target).sum())
            row = {"sample_id": record["sample_id"], "direction": record["direction"], "mode": mode,
                   "miou": (intersection + TOLERANCE)
                   / (float((prediction | gt_target).sum()) + TOLERANCE),
                   "dice": dice_of(prediction, gt_target),
                   "precision_at_0_5": (intersection + TOLERANCE)
                   / (float(prediction.sum()) + TOLERANCE),
                   "abstained": False}
            target_rows[mode].append(row)
            entry[mode] = {"miou": row["miou"], "dice": row["dice"],
                           "precision_at_0_5": row["precision_at_0_5"], "abstained": False}
        per_record_modes.append(entry)

    reference_summary = {}
    for mode in MODES:
        rows = reference_rows[mode]
        summary = summarise_references(rows)
        if mode in ("F-R0", "F-R1"):
            selected = [row for row in rows if not row["abstained"]]
            summary.update({
                "selected_confidence_mean": float(np.mean([row["selected_confidence"]
                                                           for row in selected])),
                "selected_area_mean": float(np.mean([row["selected_area"] for row in selected])),
                "selected_vs_best_iou_mean": float(np.mean([row["selected_vs_best_iou"]
                                                            for row in selected])),
                "reference_iou_gap_mean": float(np.mean([row["reference_iou_gap"]
                                                         for row in selected])),
            })
        if mode == "F-R1":
            summary["best_eligible_coverage"] = {
                str(band): coverage_stats["bands"][str(band)] / max(1, len(rows))
                for band in COVERAGE_BANDS}
        if mode == "F-R2":
            summary.update({"covered_records": coverage_stats["covered50"],
                            "uncovered_records": coverage_stats["uncovered"],
                            "coverage_rate": coverage_stats["covered50"] / max(1, len(rows)),
                            "coverage_threshold_iou": COVERAGE_IOU})
        summary["per_direction"] = {direction: summarise_references(
            [row for row in rows if row["direction"] == direction]) for direction in DIRECTIONS}
        reference_summary[mode] = summary

    write_json(OUT_MODES, {
        "_doc": ("Task 7F sections 5-10. Four exact reference modes over the frozen Task 7E E-HoldoutL3 "
                 "population, built from a single frozen U-C1 proposal inference per tile. F-R2/F-R3 use the "
                 "canonical GT reference only as the declared diagnostic ceiling; F-R1 uses GT only to choose "
                 "among existing eligible proposals and still returns a predicted mask."),
        "task": "7F", "stage": "C-E-reference-modes",
        "checks": checks, "checkpoints": metadata,
        "resolver": effective_resolver_report(), "u_c1_config": dict(CONFIGS["U-C1"]),
        "modes": {mode: MODE_LABELS[mode] for mode in MODES},
        "mode_definitions": {
            "F-R0": "max predicted mask area -> tie higher confidence -> lower original index",
            "F-R1": "max IoU to GT reference among the same eligible proposals -> tie higher confidence -> "
                    "lower index, returning the predicted proposal mask",
            "F-R2": f"canonical GT mask when best eligible IoU >= {COVERAGE_IOU}, else abstain",
            "F-R3": "canonical GT mask for every record",
        },
        "reference": reference_summary, "coverage": coverage_stats,
        "records": len(records), "training_performed": False, "test_split_used": False,
    })
    for mode in MODES:
        summary = reference_summary[mode]
        print(f"[7f.modes] {mode} ({MODE_LABELS[mode]}): answered {summary['answered']}/"
              f"{summary['records']} ref mIoU "
              f"{(summary['reference_miou'] if summary['reference_miou'] is not None else 0.0):.4f} "
              f"Pr@0.5 {(summary['reference_precision_at_0_5'] or 0.0):.4f}", flush=True)
    print(f"[7f.modes] coverage50 {coverage_stats['covered50']}/{len(records)} "
          f"({coverage_stats['covered50'] / max(1, len(records)):.4f}) bands {coverage_stats['bands']}",
          flush=True)

    # ---------------- paired counterfactual ceilings
    paired_rows = []
    for pair in pairs:
        members = [next(record for record in records if record["sample_id"] == member["sample_id"])
                   for member in pair["members"]]
        eligible = proposals_of(pair["tile_id"], members[0]["image_path"])
        gt_reference = np.asarray(masks.mask(pair["tile_id"], pair["reference_source_feature_id"]),
                                  dtype=bool)
        built = build_mode_masks(eligible, gt_reference)
        entry = {"pair_id": pair["pair_id"], "eligible_count": built["detail"]["eligible_count"],
                 "covered50": built["detail"]["covered50"], "modes": {}}
        for mode in MODES:
            mask = built["masks"][mode]
            if mask is None:
                entry["modes"][mode] = {"abstained": True, "passed": False, "own_iou": [0.0, 0.0],
                                        "cross_iou": [0.0, 0.0], "margin": 0.0}
                continue
            own, cross = [], []
            for index, member in enumerate(members):
                prediction = predict_with(mask, member)
                own_mask = np.asarray(masks.mask(member["tile_id"],
                                                 member["target_source_feature_id"]), dtype=bool)
                other = members[1 - index]
                other_mask = np.asarray(masks.mask(other["tile_id"],
                                                   other["target_source_feature_id"]), dtype=bool)
                own.append(iou_of(prediction, own_mask))
                cross.append(iou_of(prediction, other_mask))
            entry["modes"][mode] = {
                "abstained": False, "passed": bool(own[0] > cross[0] and own[1] > cross[1]),
                "own_iou": own, "cross_iou": cross,
                "margin": float(np.mean(own) - np.mean(cross))}
        paired_rows.append(entry)

    paired_summary = {}
    for mode in MODES:
        answered = [row for row in paired_rows if not row["modes"][mode]["abstained"]]
        own_values = [value for row in answered for value in row["modes"][mode]["own_iou"]]
        cross_values = [value for row in answered for value in row["modes"][mode]["cross_iou"]]
        paired_summary[mode] = {
            "passed": sum(1 for row in answered if row["modes"][mode]["passed"]),
            "pairs": len(paired_rows), "answered_pairs": len(answered),
            "abstention_pairs": len(paired_rows) - len(answered),
            "pass_rate": sum(1 for row in answered if row["modes"][mode]["passed"])
            / max(1, len(paired_rows)),
            "mean_own_iou": float(np.mean(own_values)) if own_values else None,
            "mean_cross_iou": float(np.mean(cross_values)) if cross_values else None,
            "own_cross_margin": (float(np.mean(own_values) - np.mean(cross_values))
                                 if own_values else None),
        }
    write_json(OUT_PAIRED, {
        "_doc": ("Task 7F section 14. Paired counterfactual ceilings on the frozen E-PairedHoldout20 for "
                 "F-R0 ... F-R3. Each pair shares one tile and one oracle reference, so the reference is built "
                 "once per pair and reused for both direction programs and every mode."),
        "task": "7F", "stage": "H-paired-reference-modes",
        "pairs": len(paired_rows), "checks": checks,
        "modes": {mode: MODE_LABELS[mode] for mode in MODES},
        "results": paired_summary, "rows": paired_rows,
        "training_performed": False, "test_split_used": False,
    })
    for mode in MODES:
        summary = paired_summary[mode]
        print(f"[7f.paired] {mode}: {summary['passed']}/{summary['pairs']} "
              f"(answered {summary['answered_pairs']}, abstention pairs {summary['abstention_pairs']}) "
              f"margin {(summary['own_cross_margin'] or 0.0):+.4f}", flush=True)

    write_cache({
        "_doc": ("Task 7F shared measurement cache: per-record reference rows, per-record D-B1 target rows "
                 "for all four modes, the coverage statistics and the per-pair rows. Gitignored."),
        "task": "7F", "checks": checks, "records": len(records), "pairs": len(pairs),
        "reference_rows": reference_rows, "target_rows": target_rows,
        "per_record_modes": per_record_modes, "paired_rows": paired_rows,
        "coverage": coverage_stats, "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7f.modes] cache written ({CACHE_PATH.name}); runtime {time.time() - started:.1f}s", flush=True)
    return 0


def _is_eligible(proposal) -> bool:
    from buildreasonseg_mvp.task6q_reference_resolver import is_eligible

    return bool(is_eligible(proposal, "largest"))


if __name__ == "__main__":
    raise SystemExit(main())
