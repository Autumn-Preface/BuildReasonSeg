"""Task 6W Part H — three reference diagnostics on the untouched RefValUnique.

* **W-S0** — Task 6U U-S1: U-C1 proposals + deterministic Task 6Q selector.
* **W-SQ** — U-C1 proposals + learned quality filter (ProposalQualityEstimator v0.1, threshold 0.50) +
  deterministic Task 6Q selector.
* **W-ORACLE** — U-C1 proposals + oracle `q_gt >= 0.50` filter + deterministic Task 6Q selector
  (evaluation ceiling only; never inference).

Metrics per system: selected-reference mIoU, Dice, Pr@0.5, centroid mean/median/p90, area-ratio median,
abstention rate, largest/smallest breakdowns and the failure buckets `NO_PROPOSALS`,
`NO_ELIGIBLE_PROPOSALS`, `QUALITY_FILTER_ALL_REJECTED`, `REFERENCE_NOT_COVERED_IOU50`,
`REFERENCE_SELECTION_WRONG`, `SELECTED_MASK_GEOMETRY_POOR`, `REFERENCE_OK`.

Writes `evaluation/task6w_refval_quality_filter.json`.

    python scripts/task6w_evaluate_reference.py --device 0
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
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
)
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible, select_reference  # noqa: E402
from buildreasonseg_mvp.task6w_proposal_quality import ProposalQualityEstimator  # noqa: E402
from buildreasonseg_mvp.task6w_quality_reference_resolver import (  # noqa: E402
    QualityReferenceResolver,
    deterministic_by_area,
)
from scripts.task6u_common import (  # noqa: E402
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    centroid_error,
    dice,
    gt_masks,
    group_records_by_tile,
    iou,
    percentile,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)
from task6n_train import FEATURE_ROOT  # noqa: E402

OUT = EVAL / "task6w_refval_quality_filter.json"
PACK = REPO_ROOT / "artifacts" / "task6p" / "reference_packs" / "ref_val_unique.json"
QUALITY_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6w" / "proposal_quality_v01.pt"
TRAINING = EVAL / "task6w_quality_training.json"
W0 = EVAL / "task6w_oracle_quality_filter_calib.json"
TASK6U_COMPARISON = EVAL / "task6u_refval_selector_comparison.json"
THRESHOLD = 0.50
BUCKETS = ("NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "QUALITY_FILTER_ALL_REJECTED",
           "REFERENCE_NOT_COVERED_IOU50", "REFERENCE_SELECTION_WRONG",
           "SELECTED_MASK_GEOMETRY_POOR", "REFERENCE_OK")


def load_quality_model(device: str):
    payload = torch.load(QUALITY_CHECKPOINT, map_location=device, weights_only=False)
    model = ProposalQualityEstimator().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--torch-device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256 or not QUALITY_CHECKPOINT.is_file():
        write_json(OUT, {"_doc": "Task 6W section 17.", "task": "6W",
                         "verdict": "FROZEN_ASSET_UNAVAILABLE"})
        print("[6w.ref] STOP FROZEN_ASSET_UNAVAILABLE", flush=True)
        return 2

    model, quality_payload = load_quality_model(args.torch_device)
    encoder, _report = load_frozen_sam2_encoder(device=args.torch_device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.torch_device)
    resolver = QualityReferenceResolver(model, store, device=args.torch_device, threshold=THRESHOLD)

    records = json.loads(PACK.read_text(encoding="utf-8"))["records"]
    grouped = group_records_by_tile(records)
    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    proposals_by_tile: dict[str, list] = {}
    for tile_id, tile_records in grouped.items():
        proposals, _run = proposals_for_tile(yolo, CONFIGS["U-C1"], tile_id,
                                            record_image_path(tile_records[0]), args.device,
                                            use_cache=True)
        proposals_by_tile[tile_id] = proposals
    print(f"[6w.ref] RefValUnique {len(records)} references over {len(grouped)} tiles", flush=True)

    def evaluate(mode: str) -> dict:
        rows = []
        for record in records:
            family = str(record["reference_family"])
            tile_id = str(record["tile_id"])
            proposals = proposals_by_tile[tile_id]
            truth = gt_masks(tile_id)[int(record["reference_source_feature_id"])]
            eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
            best_eligible = max((iou(proposal.mask, truth) for proposal in eligible), default=0.0)
            row = {"reference_family": family, "proposal_count": len(proposals),
                   "eligible_count": len(eligible), "best_eligible_iou": best_eligible}

            if mode == "W-S0":
                selection = select_reference(proposals, family)
                selected = None if selection.abstained else selection.proposal
                reason = selection.reason if selection.abstained else None
                filtered_all = False
            elif mode == "W-SQ":
                outcome = resolver.select(proposals, family, tile_id,
                                          record_image_path(record))
                selected = None if outcome["abstained"] else _find(eligible, outcome["selected_index"])
                reason = outcome["reason"]
                filtered_all = reason == "no_quality_eligible_proposals"
            else:  # W-ORACLE
                instances = gt_masks(tile_id)
                kept = [proposal for proposal in eligible
                        if max((iou(proposal.mask, item) for item in instances.values()),
                               default=0.0) >= THRESHOLD]
                selected = deterministic_by_area(kept, family) if kept else None
                reason = (None if kept else
                          ("no_proposals" if not proposals else
                           "no_eligible_proposals" if not eligible else
                           "no_quality_eligible_proposals"))
                filtered_all = reason == "no_quality_eligible_proposals"

            row.update({"abstained": selected is None, "abstention_reason": reason})
            if selected is None:
                row.update({"selected_iou": None, "selected_dice": None, "centroid_error": None,
                            "area_ratio": None})
            else:
                selected_iou = iou(selected.mask, truth)
                row.update({
                    "selected_iou": selected_iou,
                    "selected_dice": dice(selected.mask, truth),
                    "centroid_error": centroid_error(selected.mask, truth),
                    "area_ratio": float(selected.mask.sum()) / max(1.0, float(truth.sum())),
                })
            if not proposals:
                row["bucket"] = "NO_PROPOSALS"
            elif not eligible:
                row["bucket"] = "NO_ELIGIBLE_PROPOSALS"
            elif filtered_all:
                row["bucket"] = "QUALITY_FILTER_ALL_REJECTED"
            elif best_eligible < 0.50:
                row["bucket"] = "REFERENCE_NOT_COVERED_IOU50"
            elif row["abstained"] or row["selected_iou"] < 0.50:
                row["bucket"] = "REFERENCE_SELECTION_WRONG"
            elif row["centroid_error"] > 0.05:
                row["bucket"] = "SELECTED_MASK_GEOMETRY_POOR"
            else:
                row["bucket"] = "REFERENCE_OK"
            rows.append(row)

        def summarise(subset: list[dict]) -> dict:
            if not subset:
                return {"records": 0}
            answered = [row for row in subset if not row["abstained"]]
            ious = [row["selected_iou"] for row in answered]
            errors = [row["centroid_error"] for row in answered]
            ratios = [row["area_ratio"] for row in answered]
            buckets = Counter(row["bucket"] for row in subset)
            return {
                "records": len(subset), "answered": len(answered),
                "selected_reference_miou": float(np.mean(ious)) if ious else None,
                "selected_reference_dice": float(np.mean([row["selected_dice"] for row in answered]))
                if answered else None,
                "precision_at_0_5": float(np.mean([value >= 0.50 for value in ious]))
                if ious else None,
                "centroid_error_mean": float(np.mean(errors)) if errors else None,
                "centroid_error_median": float(np.median(errors)) if errors else None,
                "centroid_error_p90": percentile(errors, 0.90),
                "area_ratio_median": float(np.median(ratios)) if ratios else None,
                "abstention_rate": sum(1 for row in subset if row["abstained"]) / len(subset),
                "buckets": {name: buckets.get(name, 0) for name in BUCKETS},
                "REFERENCE_NOT_COVERED_IOU50": buckets.get("REFERENCE_NOT_COVERED_IOU50", 0),
                "REFERENCE_SELECTION_WRONG": buckets.get("REFERENCE_SELECTION_WRONG", 0),
                "REFERENCE_OK": buckets.get("REFERENCE_OK", 0),
            }

        return {"overall": summarise(rows),
                "largest": summarise([row for row in rows if row["reference_family"] == "largest"]),
                "smallest": summarise([row for row in rows
                                       if row["reference_family"] == "smallest"])}

    systems = {mode: evaluate(mode) for mode in ("W-S0", "W-SQ", "W-ORACLE")}
    task6u = json.loads(TASK6U_COMPARISON.read_text(encoding="utf-8"))
    s0 = systems["W-S0"]["overall"]
    sq = systems["W-SQ"]["overall"]
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    w0 = json.loads(W0.read_text(encoding="utf-8"))

    payload = {
        "_doc": (
            "Task 6W section 17. Three reference diagnostics on the untouched Task 6P RefValUnique: "
            "W-S0 (U-C1 + deterministic Task 6Q selector; identical to Task 6U U-S1), W-SQ (U-C1 + "
            "ProposalQualityEstimator v0.1 quality>=0.50 filter + deterministic selector) and W-ORACLE "
            "(U-C1 + oracle q_gt>=0.50 filter, evaluation ceiling only, never inference)."
        ),
        "task": "6W", "stage": "H-refval-quality-filter",
        "pack": {"path": str(PACK), "records": len(records), "sha256": sha256_file(PACK)},
        "config": {key: CONFIGS["U-C1"][key]
                   for key in ("id", "imgsz", "conf", "max_det", "nms", "tta", "tiling")},
        "quality_checkpoint": {"path": str(QUALITY_CHECKPOINT),
                               "sha256": sha256_file(QUALITY_CHECKPOINT),
                               "expected": training["checkpoint"]["sha256"],
                               "threshold": THRESHOLD, "trained_epoch": quality_payload["epoch"]},
        "systems": systems,
        "bucket_order": list(BUCKETS),
        "comparison_task6u_u_s1": task6u["systems"]["U-S1"]["overall"],
        "w0_mechanism_gate_passed": w0["mechanism_gate_passed"],
        "improvement_inputs": {
            "miou_required": (s0["selected_reference_miou"] or 0.0) + 0.04,
            "miou_measured": sq["selected_reference_miou"],
            "selection_wrong_required": int(0.75 * s0["REFERENCE_SELECTION_WRONG"]),
            "selection_wrong_measured": sq["REFERENCE_SELECTION_WRONG"],
            "reference_ok_required": s0["REFERENCE_OK"] + 8,
            "reference_ok_measured": sq["REFERENCE_OK"],
            "abstention_rate_max": 0.10,
            "abstention_rate_measured": sq["abstention_rate"],
        },
        "quality_reference_improved": bool(
            (sq["selected_reference_miou"] or 0.0) >= (s0["selected_reference_miou"] or 0.0) + 0.04
            and sq["REFERENCE_SELECTION_WRONG"] <= 0.75 * s0["REFERENCE_SELECTION_WRONG"]
            and sq["REFERENCE_OK"] >= s0["REFERENCE_OK"] + 8
            and sq["abstention_rate"] <= 0.10),
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    for mode in ("W-S0", "W-SQ", "W-ORACLE"):
        overall = systems[mode]["overall"]
        print(f"[6w.ref] {mode}: mIoU {overall['selected_reference_miou']:.4f} Pr@0.5 "
              f"{overall['precision_at_0_5']:.4f} centroid med "
              f"{overall['centroid_error_median']:.4f} abstain {overall['abstention_rate']:.4f} "
              f"OK {overall['REFERENCE_OK']} SW {overall['REFERENCE_SELECTION_WRONG']} NC "
              f"{overall['REFERENCE_NOT_COVERED_IOU50']} REJ "
              f"{overall['buckets']['QUALITY_FILTER_ALL_REJECTED']}", flush=True)
    print(f"[6w.ref] quality_reference_improved {payload['quality_reference_improved']}", flush=True)
    return 0


def _find(proposals: list, index: int):
    for proposal in proposals:
        if proposal.index == index:
            return proposal
    return None


if __name__ == "__main__":
    raise SystemExit(main())
