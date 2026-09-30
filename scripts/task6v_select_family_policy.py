"""Task 6V Parts C-D — train-only per-family resolver policy selection on U-Calib200.

For each reference family independently, evaluates exactly the three predeclared frozen options

* **V-P0** = U-C0 + frozen Task 6Q deterministic area selector
* **V-P1** = U-C1 + frozen Task 6Q deterministic area selector
* **V-P2** = U-C1 + frozen ProposalSetRanker v0.1

on the frozen train-only U-Calib200 split (100 largest + 100 smallest) and freezes one option per family
with the exact section 7 priority:

1. higher Pr@0.5; 2. higher selected-reference mIoU; 3. lower `REFERENCE_SELECTION_WRONG`;
4. lower centroid median; 5. lower abstention rate; 6. simpler option (V-P0 → V-P1 → V-P2).

RefValUnique / MiniVal240 / PairedVal20 never influence the choice. No model is trained.

Writes `evaluation/task6v_calibration_family_policy.json` and
`evaluation/task6v_frozen_family_policy.json`.

    python scripts/task6v_select_family_policy.py --device 0
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
from buildreasonseg_mvp.task6u_reference_ranker import ProposalSetRanker  # noqa: E402
from buildreasonseg_mvp.task6v_family_reference_resolver import (  # noqa: E402
    FAMILIES,
    OPTIONS,
    OPTION_ORDER,
    choose_policy,
    policy_priority_key,
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
    dice,
    gt_reference_mask,
    group_records_by_tile,
    iou,
    percentile,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)
from scripts.task6u_evaluate_reference import evaluate_selector  # noqa: E402

OUT_CALIB = EVAL / "task6v_calibration_family_policy.json"
OUT_FROZEN = EVAL / "task6v_frozen_family_policy.json"
SPLIT = EVAL / "task6u_reference_train_split.json"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
TASK6U_TRAINING = EVAL / "task6u_ranker_training.json"


def load_ranker() -> tuple[object, str]:
    payload = torch.load(RANKER_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ProposalSetRanker().to("cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, sha256_file(RANKER_CHECKPOINT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    training = json.loads(TASK6U_TRAINING.read_text(encoding="utf-8"))
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256 or not RANKER_CHECKPOINT.is_file():
        write_json(OUT_CALIB, {"_doc": "Task 6V sections 4-7.", "task": "6V",
                               "verdict": "FROZEN_RESOLVER_ASSET_UNAVAILABLE",
                               "yolo_sha256": checkpoint_sha})
        print("[6v.select] STOP FROZEN_RESOLVER_ASSET_UNAVAILABLE", flush=True)
        return 2
    ranker, ranker_sha = load_ranker()
    if ranker_sha != training["checkpoint"]["sha256"]:
        write_json(OUT_CALIB, {"_doc": "Task 6V sections 4-7.", "task": "6V",
                               "verdict": "FROZEN_RESOLVER_ASSET_UNAVAILABLE",
                               "ranker_sha256": ranker_sha})
        print("[6v.select] STOP: ranker hash mismatch", flush=True)
        return 2

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    records = split["u_calib200"]["records"]
    by_family = {family: [record for record in records
                          if str(record["reference_family"]) == family] for family in FAMILIES}
    print(f"[6v.select] U-Calib200 {len(records)} references "
          f"({ {family: len(rows) for family, rows in by_family.items()} })", flush=True)

    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    grouped = group_records_by_tile(records)
    proposals_by_config: dict[str, dict] = {"U-C0": {}, "U-C1": {}}
    for config_id in ("U-C0", "U-C1"):
        config = CONFIGS[config_id]
        cache: dict[tuple, list] = {}
        for tile_id, tile_records in grouped.items():
            proposals, _run = proposals_for_tile(yolo, config, tile_id,
                                                 record_image_path(tile_records[0]), args.device,
                                                 use_cache=True)
            for record in tile_records:
                key = (tile_id, int(record["reference_source_feature_id"]),
                       str(record["reference_family"]))
                cache[key] = proposals
        proposals_by_config[config_id] = cache
        print(f"[6v.select] {config_id}: {len(grouped)} tiles cached", flush=True)

    def metrics_for(family: str, option: str) -> dict:
        option_spec = OPTIONS[option]
        cache = proposals_by_config[option_spec["config"]]
        rows = []
        for record in by_family[family]:
            key = (str(record["tile_id"]), int(record["reference_source_feature_id"]), family)
            proposals = cache.get(key, [])
            truth = gt_reference_mask(record)
            eligible = [proposal for proposal in proposals
                        if __import__("buildreasonseg_mvp.task6q_reference_resolver",
                                      fromlist=["is_eligible"]).is_eligible(proposal, family)]
            best_eligible = max((iou(proposal.mask, truth) for proposal in eligible), default=0.0)
            outcome = select_option(option, proposals, family, ranker)
            row = {"reference_family": family, "proposal_count": len(proposals),
                   "eligible_count": len(eligible), "best_eligible_iou": best_eligible,
                   "abstained": outcome["abstained"]}
            if outcome["abstained"]:
                row.update({"selected_iou": None, "selected_dice": None, "centroid_error": None,
                            "bucket": attribute_failure(proposal_count=len(proposals),
                                                        eligible_count=len(eligible),
                                                        best_eligible_iou=best_eligible,
                                                        selected_iou=None,
                                                        selected_centroid_error=None)})
            else:
                selected_iou = iou(outcome["mask"], truth)
                error = centroid_error(outcome["mask"], truth)
                row.update({"selected_iou": selected_iou, "selected_dice": dice(outcome["mask"], truth),
                            "centroid_error": error,
                            "bucket": attribute_failure(proposal_count=len(proposals),
                                                        eligible_count=len(eligible),
                                                        best_eligible_iou=best_eligible,
                                                        selected_iou=selected_iou,
                                                        selected_centroid_error=error)})
            rows.append(row)
        answered = [row for row in rows if not row["abstained"]]
        ious = [row["selected_iou"] for row in answered]
        errors = [row["centroid_error"] for row in answered]
        buckets = bucket_counts(rows)
        return {
            "option": option, "family": family, "description": option_spec["description"],
            "config": option_spec["config"], "selector": option_spec["selector"],
            "records": len(rows), "answered": len(answered),
            "selected_reference_miou": float(np.mean(ious)) if ious else None,
            "selected_reference_dice": float(np.mean([row["selected_dice"] for row in answered]))
            if answered else None,
            "precision_at_0_5": float(np.mean([value >= 0.50 for value in ious])) if ious else None,
            "centroid_median": float(np.median(errors)) if errors else None,
            "centroid_p90": percentile(errors, 0.90),
            "abstention_rate": sum(1 for row in rows if row["abstained"]) / len(rows),
            "REFERENCE_NOT_COVERED_IOU50": buckets["REFERENCE_NOT_COVERED_IOU50"],
            "REFERENCE_SELECTION_WRONG": buckets["REFERENCE_SELECTION_WRONG"],
            "REFERENCE_OK": buckets["REFERENCE_OK"],
            "buckets": buckets,
        }

    def select_option(option: str, proposals, family: str, ranker_model) -> dict:
        from buildreasonseg_mvp.task6q_reference_resolver import is_eligible, select_reference

        if OPTIONS[option]["selector"] == "deterministic":
            selection = select_reference(proposals, family)
            if selection.abstained:
                return {"abstained": True}
            return {"abstained": False, "mask": selection.mask}
        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        if not eligible:
            return {"abstained": True}
        from buildreasonseg_mvp.task6u_reference_ranker import select_with_ranker

        outcome = select_with_ranker(ranker_model, eligible, family, device="cpu")
        return {"abstained": False, "mask": eligible[outcome["selected_index"]].mask}

    per_family = {}
    for family in FAMILIES:
        per_family[family] = {option: metrics_for(family, option) for option in OPTION_ORDER}
        for option in OPTION_ORDER:
            entry = per_family[family][option]
            print(f"[6v.select] {family:8s} {option}: Pr@0.5 "
                  f"{entry['precision_at_0_5']:.4f} mIoU {entry['selected_reference_miou']:.4f} "
                  f"SELECTION_WRONG {entry['REFERENCE_SELECTION_WRONG']} "
                  f"centroid_med {entry['centroid_median']:.4f} abstain "
                  f"{entry['abstention_rate']:.4f}", flush=True)

    policy = choose_policy(per_family)
    calibration = {
        "_doc": (
            "Task 6V sections 4-7. Train-only per-family resolver policy selection on the frozen Task 6U "
            "U-Calib200 split (100 largest + 100 smallest): exactly the three predeclared frozen options "
            "V-P0/V-P1/V-P2 are compared per family with the unchanged Task 6Q eligibility. GT reference "
            "masks score the result only; no model is trained in Task 6V."
        ),
        "task": "6V", "stage": "C-D-calibration-family-policy",
        "frozen_assets": {
            "yolo_checkpoint": {"path": str(PROPOSAL_CHECKPOINT), "sha256": checkpoint_sha,
                                "expected": PROPOSAL_CHECKPOINT_SHA256, "retrained": False},
            "ranker_checkpoint": {"path": str(RANKER_CHECKPOINT), "sha256": ranker_sha,
                                  "expected": training["checkpoint"]["sha256"], "retrained": False},
            "configs": {config_id: {key: CONFIGS[config_id][key]
                                    for key in ("imgsz", "conf", "max_det", "nms", "tta", "tiling")}
                        for config_id in ("U-C0", "U-C1")},
        },
        "options": {option: OPTIONS[option]["description"] for option in OPTION_ORDER},
        "options_compared": len(OPTION_ORDER),
        "calibration_split": {"path": str(SPLIT), "records": len(records),
                              "by_family": {family: len(rows)
                                            for family, rows in by_family.items()},
                              "train_only": True},
        "per_family": per_family,
        "priority_rule": [
            "higher Pr@0.5", "higher selected-reference mIoU",
            "lower REFERENCE_SELECTION_WRONG", "lower centroid median",
            "lower abstention rate", "simpler option (V-P0 -> V-P1 -> V-P2)",
        ],
        "ranking": {family: sorted(OPTION_ORDER,
                                   key=lambda option: policy_priority_key(per_family[family][option]),
                                   reverse=True) for family in FAMILIES},
        "selected_policy": policy,
        "selection_inputs": {"u_calib200": True, "refval_unique": False, "minival240": False,
                             "pairedval20": False, "test_split": False},
        "training_performed": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_CALIB, calibration)

    frozen = {
        "_doc": (
            "Task 6V section 7. The frozen single resolver option per reference family, chosen on the "
            "train-only U-Calib200 split by the exact section 7 priority and immutable after creation. "
            "RefValUnique/MiniVal240/PairedVal20 results are never used to change it."
        ),
        "task": "6V", "stage": "D-frozen-family-policy",
        "policy": policy,
        "policy_description": {family: OPTIONS[policy[family]]["description"] for family in FAMILIES},
        "options": {option: OPTIONS[option]["description"] for option in OPTION_ORDER},
        "priority_rule": calibration["priority_rule"],
        "ranking": calibration["ranking"],
        "selected_metrics": {
            family: {key: per_family[family][policy[family]][key] for key in (
                "selected_reference_miou", "precision_at_0_5", "centroid_median",
                "abstention_rate", "REFERENCE_SELECTION_WRONG", "REFERENCE_OK")}
            for family in FAMILIES},
        "evidence": {"calibration": str(OUT_CALIB)},
        "frozen": True, "immutable_after_creation": True,
        "chosen_without_refval": True,
        "fourth_option": None,
        "training_performed": False,
        "test_split_used": False,
    }
    write_json(OUT_FROZEN, frozen)
    print(f"[6v.select] ranking {calibration['ranking']} -> frozen policy {policy}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
