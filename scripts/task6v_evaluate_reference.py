"""Task 6V Part E — frozen family-conditioned resolver on the untouched RefValUnique.

Evaluates the frozen per-family policy (`evaluation/task6v_frozen_family_policy.json`) on the exact Task 6P
RefValUnique (219 references) and reports overall and per-family metrics with the six Task 6Q failure
buckets, compared against the frozen Task 6U U-S0 / U-S1 / U-S2 numbers and the U-C1 oracle-selection
ceiling (diagnostic only). The policy is never modified here.

Writes `evaluation/task6v_refval_family_policy.json`.

    python scripts/task6v_evaluate_reference.py --device 0
"""

from __future__ import annotations

import argparse
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
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible  # noqa: E402
from buildreasonseg_mvp.task6u_reference_ranker import ProposalSetRanker  # noqa: E402
from buildreasonseg_mvp.task6v_family_reference_resolver import (  # noqa: E402
    FAMILIES,
    OPTIONS,
    FamilyConditionedResolver,
    load_frozen_policy,
)
from scripts.task6u_common import (  # noqa: E402
    BUCKET_ORDER,
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    attribute_failure,
    bucket_counts,
    centroid_error,
    coverage_for_records,
    dice,
    gt_reference_mask,
    group_records_by_tile,
    iou,
    percentile,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)

OUT = EVAL / "task6v_refval_family_policy.json"
PACK = REPO_ROOT / "artifacts" / "task6p" / "reference_packs" / "ref_val_unique.json"
FROZEN_POLICY = EVAL / "task6v_frozen_family_policy.json"
CALIBRATION = EVAL / "task6v_calibration_family_policy.json"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
TASK6U_COMPARISON = EVAL / "task6u_refval_selector_comparison.json"
TASK6U_AUDIT = EVAL / "task6u_refval_candidate_audit.json"


def load_ranker():
    payload = torch.load(RANKER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ProposalSetRanker().to("cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256 or not FROZEN_POLICY.is_file() \
            or not RANKER_CHECKPOINT.is_file():
        write_json(OUT, {"_doc": "Task 6V section 9.", "task": "6V",
                         "verdict": "FROZEN_RESOLVER_ASSET_UNAVAILABLE"})
        print("[6v.ref] STOP FROZEN_RESOLVER_ASSET_UNAVAILABLE", flush=True)
        return 2

    policy = load_frozen_policy(FROZEN_POLICY)
    resolver = FamilyConditionedResolver(policy, ranker=load_ranker())
    records = json.loads(PACK.read_text(encoding="utf-8"))["records"]
    print(f"[6v.ref] RefValUnique {len(records)} references; frozen policy {policy}", flush=True)

    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    grouped = group_records_by_tile(records)
    caches: dict[str, dict] = {config_id: {} for config_id in ("U-C0", "U-C1")}
    for config_id in caches:
        for tile_id, tile_records in grouped.items():
            proposals, _run = proposals_for_tile(yolo, CONFIGS[config_id], tile_id,
                                                 record_image_path(tile_records[0]), args.device,
                                                 use_cache=True)
            for record in tile_records:
                key = (tile_id, int(record["reference_source_feature_id"]),
                       str(record["reference_family"]))
                caches[config_id][key] = proposals

    rows = []
    for record in records:
        family = str(record["reference_family"])
        option = resolver.option_for(family)
        key = (str(record["tile_id"]), int(record["reference_source_feature_id"]), family)
        proposals = caches[option["config"]].get(key, [])
        truth = gt_reference_mask(record)
        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        best_eligible = max((iou(proposal.mask, truth) for proposal in eligible), default=0.0)
        outcome = resolver.select(proposals, family)
        row = {"reference_key": list(key), "reference_family": family, "option": outcome["option"],
               "config": outcome["config"], "selector": outcome["selector"],
               "proposal_count": len(proposals), "eligible_count": len(eligible),
               "best_eligible_iou": best_eligible, "abstained": outcome["abstained"],
               "abstention_reason": outcome["reason"]}
        if outcome["abstained"]:
            row.update({"selected_iou": None, "selected_dice": None, "centroid_error": None,
                        "area_ratio": None,
                        "bucket": attribute_failure(proposal_count=len(proposals),
                                                    eligible_count=len(eligible),
                                                    best_eligible_iou=best_eligible,
                                                    selected_iou=None,
                                                    selected_centroid_error=None)})
        else:
            selected_iou = iou(outcome["mask"], truth)
            error = centroid_error(outcome["mask"], truth)
            area_ratio = float(outcome["mask"].sum()) / max(1.0, float(truth.sum()))
            row.update({"selected_iou": selected_iou, "selected_dice": dice(outcome["mask"], truth),
                        "centroid_error": error, "area_ratio": area_ratio,
                        "bucket": attribute_failure(proposal_count=len(proposals),
                                                    eligible_count=len(eligible),
                                                    best_eligible_iou=best_eligible,
                                                    selected_iou=selected_iou,
                                                    selected_centroid_error=error)})
        rows.append(row)

    def summarise(subset: list[dict]) -> dict:
        if not subset:
            return {"records": 0}
        answered = [row for row in subset if not row["abstained"]]
        ious = [row["selected_iou"] for row in answered]
        errors = [row["centroid_error"] for row in answered]
        ratios = [row["area_ratio"] for row in answered]
        return {
            "records": len(subset), "answered": len(answered),
            "selected_reference_miou": float(np.mean(ious)) if ious else None,
            "selected_reference_dice": float(np.mean([row["selected_dice"] for row in answered]))
            if answered else None,
            "precision_at_0_5": float(np.mean([value >= 0.50 for value in ious])) if ious else None,
            "centroid_error_mean": float(np.mean(errors)) if errors else None,
            "centroid_error_median": float(np.median(errors)) if errors else None,
            "centroid_error_p90": percentile(errors, 0.90),
            "area_ratio_median": float(np.median(ratios)) if ratios else None,
            "abstention_rate": sum(1 for row in subset if row["abstained"]) / len(subset),
            "buckets": bucket_counts(subset),
        }

    overall = summarise(rows)
    per_family = {family: summarise([row for row in rows if row["reference_family"] == family])
                  for family in FAMILIES}
    task6u = json.loads(TASK6U_COMPARISON.read_text(encoding="utf-8"))
    ceiling = json.loads(TASK6U_AUDIT.read_text(encoding="utf-8"))["oracle_selection_ceiling"]["selected"]
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    s1 = task6u["systems"]["U-S1"]["overall"]
    payload = {
        "_doc": (
            "Task 6V section 9. The frozen family-conditioned resolver (largest and smallest policies "
            "selected on the train-only U-Calib200 split) evaluated on the untouched Task 6P "
            "RefValUnique, with the six Task 6Q failure buckets and comparisons against the frozen "
            "Task 6U U-S0/U-S1/U-S2 and the U-C1 oracle-selection ceiling (diagnostic only). The policy "
            "is not modified here."
        ),
        "task": "6V", "stage": "E-refval-family-policy",
        "pack": {"path": str(PACK), "records": len(records), "sha256": sha256_file(PACK)},
        "policy": policy,
        "policy_source": {"frozen": str(FROZEN_POLICY), "calibration": str(CALIBRATION),
                          "train_only_calibration": True},
        "option_usage": {family: resolver.option_for(family)["id"] for family in FAMILIES},
        "checkpoint": {"path": str(PROPOSAL_CHECKPOINT), "sha256": checkpoint_sha},
        "overall": overall,
        "largest": per_family["largest"],
        "smallest": per_family["smallest"],
        "buckets": overall["buckets"],
        "bucket_order": list(BUCKET_ORDER),
        "comparison_task6u": {
            "U-S0": task6u["systems"]["U-S0"]["overall"],
            "U-S1": task6u["systems"]["U-S1"]["overall"],
            "U-S2": task6u["systems"]["U-S2"]["overall"],
            "U-S0_largest": task6u["systems"]["U-S0"]["largest"],
            "U-S1_largest": task6u["systems"]["U-S1"]["largest"],
            "U-S2_largest": task6u["systems"]["U-S2"]["largest"],
            "U-S0_smallest": task6u["systems"]["U-S0"]["smallest"],
            "U-S1_smallest": task6u["systems"]["U-S1"]["smallest"],
            "U-S2_smallest": task6u["systems"]["U-S2"]["smallest"],
            "U-C1_oracle_selection_ceiling": ceiling["overall"],
        },
        "improvement_flag_inputs": {
            "miou_required": (s1["selected_reference_miou"] or 0.0) + 0.015,
            "miou_measured": overall["selected_reference_miou"],
            "reference_ok_required": s1["buckets"]["REFERENCE_OK"] + 5,
            "reference_ok_measured": overall["buckets"]["REFERENCE_OK"],
            "selection_wrong_required": s1["buckets"]["REFERENCE_SELECTION_WRONG"] - 5,
            "selection_wrong_measured": overall["buckets"]["REFERENCE_SELECTION_WRONG"],
            "abstention_rate_max": 0.05,
            "abstention_rate_measured": overall["abstention_rate"],
        },
        "family_policy_improved": bool(
            (overall["selected_reference_miou"] or 0.0) >= (s1["selected_reference_miou"] or 0.0) + 0.015
            and overall["buckets"]["REFERENCE_OK"] >= s1["buckets"]["REFERENCE_OK"] + 5
            and overall["buckets"]["REFERENCE_SELECTION_WRONG"]
            <= s1["buckets"]["REFERENCE_SELECTION_WRONG"] - 5
            and overall["abstention_rate"] <= 0.05),
        "policy_changed_after_refval": False,
        "training_performed": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[6v.ref] frozen policy {policy}: mIoU "
          f"{overall['selected_reference_miou']:.4f} Pr@0.5 {overall['precision_at_0_5']:.4f} "
          f"REFERENCE_OK {overall['buckets']['REFERENCE_OK']} SELECTION_WRONG "
          f"{overall['buckets']['REFERENCE_SELECTION_WRONG']} NOT_COVERED "
          f"{overall['buckets']['REFERENCE_NOT_COVERED_IOU50']} abstain "
          f"{overall['abstention_rate']:.4f} (U-S1 mIoU {s1['selected_reference_miou']:.4f}) | "
          f"improved {payload['family_policy_improved']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
