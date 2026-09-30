"""Task 6W Part C — W0 oracle proposal-quality-filter mechanism diagnostic (U-Calib200 only).

Before any training, this tests whether **perfect knowledge of proposal quality** would make the
quality-filter + deterministic-area mechanism useful. For every U-C1 proposal on a U-Calib200 tile:

```text
q_gt = max IoU(proposal_mask, every native GT building instance on that tile)
```

Oracle quality filter for a reference query: Task 6Q family eligibility first → keep only `q_gt >= 0.50`
→ abstain if none remain → largest = max area / smallest = min area → Task 6Q tie-break (higher YOLO
confidence, then lower original index).

The W0 mechanism gate (section 7) requires ALL against the U-C1 deterministic baseline on the same
U-Calib200:

* oracle overall selected-reference mIoU >= baseline + 0.08
* oracle smallest mIoU >= baseline smallest + 0.10
* oracle `REFERENCE_SELECTION_WRONG` <= 0.60 * baseline
* oracle abstention rate <= 0.10

If any fails the task stops with `QUALITY_FILTER_MECHANISM_INSUFFICIENT` and no estimator is trained.

Writes `evaluation/task6w_oracle_quality_filter_calib.json`.

    python scripts/task6w_oracle_quality_diagnostic.py --device 0
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
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible, select_reference  # noqa: E402
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

OUT = EVAL / "task6w_oracle_quality_filter_calib.json"
SPLIT = EVAL / "task6u_reference_train_split.json"
QUALITY_THRESHOLD = 0.50
GATE = {"overall_miou_delta_min": 0.08, "smallest_miou_delta_min": 0.10,
        "selection_wrong_ratio_max": 0.60, "abstention_rate_max": 0.10}
FAMILIES = ("largest", "smallest")


def proposal_quality(proposal_mask: np.ndarray, instances: dict[int, np.ndarray]) -> float:
    """q_gt = max IoU of the proposal mask with any native GT building instance on the tile."""

    best = 0.0
    for truth in instances.values():
        value = iou(proposal_mask, truth)
        if value > best:
            best = value
    return float(best)


def choose_by_area(proposals: list, family: str):
    """Deterministic largest/smallest area selection with the Task 6Q tie-break."""

    if not proposals:
        return None
    if family == "largest":
        return max(proposals, key=lambda proposal: (proposal.area_px, proposal.confidence,
                                                    -proposal.index))
    return min(proposals, key=lambda proposal: (proposal.area_px, -proposal.confidence,
                                                proposal.index))


def evaluate_system(records: list[dict], proposals_by_key: dict, qualities_by_key: dict,
                    mode: str) -> dict:
    """mode: 'baseline' (U-C1 deterministic) or 'oracle' (q_gt >= 0.50 filter first)."""

    rows = []
    for record in records:
        family = str(record["reference_family"])
        key = (str(record["tile_id"]), int(record["reference_source_feature_id"]), family)
        proposals = proposals_by_key.get(key, [])
        qualities = qualities_by_key.get(key, [])
        truth = gt_masks(str(record["tile_id"]))[int(record["reference_source_feature_id"])]
        eligible = [(proposal, quality) for proposal, quality in zip(proposals, qualities)
                    if is_eligible(proposal, family)]
        best_eligible = max((iou(proposal.mask, truth) for proposal, _ in eligible), default=0.0)
        if mode == "oracle":
            kept = [proposal for proposal, quality in eligible if quality >= QUALITY_THRESHOLD]
            selected = choose_by_area(kept, family)
            if not proposals:
                reason = "no_proposals"
            elif not eligible:
                reason = "no_eligible_proposals"
            elif not kept:
                reason = "no_quality_eligible_proposals"
            else:
                reason = None
        else:
            selection = select_reference(proposals, family)
            selected = None if selection.abstained else selection.proposal
            reason = selection.reason if selection.abstained else None

        row = {"reference_key": list(key), "reference_family": family,
               "proposal_count": len(proposals), "eligible_count": len(eligible),
               "best_eligible_iou": best_eligible, "abstained": selected is None,
               "abstention_reason": reason}
        if selected is None:
            row.update({"selected_iou": None, "selected_dice": None, "centroid_error": None})
        else:
            selected_iou = iou(selected.mask, truth)
            row.update({"selected_iou": selected_iou, "selected_dice": dice(selected.mask, truth),
                        "centroid_error": centroid_error(selected.mask, truth)})
        if not proposals:
            row["bucket"] = "NO_PROPOSALS"
        elif not eligible:
            row["bucket"] = "NO_ELIGIBLE_PROPOSALS"
        elif mode == "oracle" and row["abstention_reason"] == "no_quality_eligible_proposals":
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
        buckets = Counter(row["bucket"] for row in subset)
        return {
            "records": len(subset), "answered": len(answered),
            "selected_reference_miou": float(np.mean(ious)) if ious else None,
            "selected_reference_dice": float(np.mean([row["selected_dice"] for row in answered]))
            if answered else None,
            "precision_at_0_5": float(np.mean([value >= 0.50 for value in ious])) if ious else None,
            "centroid_median": float(np.median(errors)) if errors else None,
            "centroid_p90": percentile(errors, 0.90),
            "abstention_rate": sum(1 for row in subset if row["abstained"]) / len(subset),
            "REFERENCE_NOT_COVERED_IOU50": buckets.get("REFERENCE_NOT_COVERED_IOU50", 0),
            "REFERENCE_SELECTION_WRONG": buckets.get("REFERENCE_SELECTION_WRONG", 0),
            "QUALITY_FILTER_ALL_REJECTED": buckets.get("QUALITY_FILTER_ALL_REJECTED", 0),
            "REFERENCE_OK": buckets.get("REFERENCE_OK", 0),
        }

    return {"overall": summarise(rows),
            "largest": summarise([row for row in rows if row["reference_family"] == "largest"]),
            "smallest": summarise([row for row in rows if row["reference_family"] == "smallest"]),
            "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256:
        write_json(OUT, {"_doc": "Task 6W section 7.", "task": "6W",
                         "verdict": "FROZEN_ASSET_UNAVAILABLE", "yolo_sha256": checkpoint_sha})
        print("[6w.w0] STOP FROZEN_ASSET_UNAVAILABLE", flush=True)
        return 2

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    records = split["u_calib200"]["records"]
    grouped = group_records_by_tile(records)
    print(f"[6w.w0] U-Calib200 {len(records)} references over {len(grouped)} tiles; U-C1 "
          f"(imgsz {CONFIGS['U-C1']['imgsz']}, conf {CONFIGS['U-C1']['conf']}, "
          f"max_det {CONFIGS['U-C1']['max_det']})", flush=True)

    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    proposals_by_key: dict[tuple, list] = {}
    qualities_by_key: dict[tuple, list] = {}
    quality_histogram = Counter()
    for tile_id, tile_records in grouped.items():
        proposals, _run = proposals_for_tile(yolo, CONFIGS["U-C1"], tile_id,
                                             record_image_path(tile_records[0]), args.device,
                                             use_cache=True)
        instances = gt_masks(tile_id)
        qualities = [proposal_quality(proposal.mask, instances) for proposal in proposals]
        for quality in qualities:
            quality_histogram[f"{min(int(quality * 10) / 10, 0.9):.1f}"] += 1
        for record in tile_records:
            key = (tile_id, int(record["reference_source_feature_id"]),
                   str(record["reference_family"]))
            proposals_by_key[key] = proposals
            qualities_by_key[key] = qualities

    baseline = evaluate_system(records, proposals_by_key, qualities_by_key, "baseline")
    oracle = evaluate_system(records, proposals_by_key, qualities_by_key, "oracle")

    base_overall = baseline["overall"]
    base_smallest = baseline["smallest"]
    oracle_overall = oracle["overall"]
    oracle_smallest = oracle["smallest"]
    gate = {
        "1_overall_miou": {
            "requirement": f"oracle overall mIoU >= baseline + {GATE['overall_miou_delta_min']}",
            "threshold": base_overall["selected_reference_miou"] + GATE["overall_miou_delta_min"],
            "baseline": base_overall["selected_reference_miou"],
            "measured": oracle_overall["selected_reference_miou"],
            "delta": oracle_overall["selected_reference_miou"] - base_overall["selected_reference_miou"],
        },
        "2_smallest_miou": {
            "requirement": f"oracle smallest mIoU >= baseline smallest + "
                           f"{GATE['smallest_miou_delta_min']}",
            "threshold": base_smallest["selected_reference_miou"]
            + GATE["smallest_miou_delta_min"],
            "baseline": base_smallest["selected_reference_miou"],
            "measured": oracle_smallest["selected_reference_miou"],
            "delta": oracle_smallest["selected_reference_miou"]
            - base_smallest["selected_reference_miou"],
        },
        "3_selection_wrong": {
            "requirement": f"oracle SELECTION_WRONG <= {GATE['selection_wrong_ratio_max']} * baseline",
            "threshold": int(GATE["selection_wrong_ratio_max"]
                             * base_overall["REFERENCE_SELECTION_WRONG"]),
            "baseline": base_overall["REFERENCE_SELECTION_WRONG"],
            "measured": oracle_overall["REFERENCE_SELECTION_WRONG"],
        },
        "4_abstention": {
            "requirement": f"oracle abstention rate <= {GATE['abstention_rate_max']}",
            "threshold": GATE["abstention_rate_max"],
            "measured": oracle_overall["abstention_rate"],
        },
    }
    gate["1_overall_miou"]["passed"] = (gate["1_overall_miou"]["delta"]
                                        >= GATE["overall_miou_delta_min"])
    gate["2_smallest_miou"]["passed"] = (gate["2_smallest_miou"]["delta"]
                                         >= GATE["smallest_miou_delta_min"])
    gate["3_selection_wrong"]["passed"] = (gate["3_selection_wrong"]["measured"]
                                           <= gate["3_selection_wrong"]["threshold"])
    gate["4_abstention"]["passed"] = (gate["4_abstention"]["measured"]
                                      <= GATE["abstention_rate_max"])
    passed = all(entry["passed"] for entry in gate.values())

    payload = {
        "_doc": (
            "Task 6W section 7. W0 oracle proposal-quality-filter mechanism diagnostic on the train-only "
            "U-Calib200 split: does perfect knowledge of proposal quality (q_gt = max IoU with any native "
            "GT building instance on the tile) make the quality-filter + deterministic-area mechanism "
            "useful? GT is diagnostic metadata only and never enters inference. No estimator is trained "
            "unless every W0 condition passes."
        ),
        "task": "6W", "stage": "C-W0-oracle-quality-diagnostic",
        "checkpoint": {"path": str(PROPOSAL_CHECKPOINT), "sha256": checkpoint_sha,
                       "expected": PROPOSAL_CHECKPOINT_SHA256, "retrained": False},
        "config": {key: CONFIGS["U-C1"][key]
                   for key in ("id", "imgsz", "conf", "max_det", "nms", "tta", "tiling")},
        "calibration_split": {"path": str(SPLIT), "records": len(records), "tiles": len(grouped),
                              "by_family": dict(Counter(record["reference_family"]
                                                        for record in records)),
                              "train_only": True},
        "quality_definition": "q_gt = max IoU(proposal_mask, every native GT building instance on tile)",
        "quality_threshold": QUALITY_THRESHOLD,
        "q_gt_histogram": {key: quality_histogram[key] for key in sorted(quality_histogram)},
        "baseline_u_c1_deterministic": {key: value for key, value in baseline.items()
                                        if key != "rows"},
        "oracle_quality_filter": {key: value for key, value in oracle.items() if key != "rows"},
        "gate_constants": GATE,
        "gate": gate,
        "mechanism_gate_passed": passed,
        "training_allowed": passed,
        "estimator_trained": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    payload["verdict"] = "MECHANISM_FEASIBLE" if passed else "QUALITY_FILTER_MECHANISM_INSUFFICIENT"
    write_json(OUT, payload)

    print(f"[6w.w0] baseline overall mIoU {base_overall['selected_reference_miou']:.4f} "
          f"smallest {base_smallest['selected_reference_miou']:.4f} SW "
          f"{base_overall['REFERENCE_SELECTION_WRONG']} OK {base_overall['REFERENCE_OK']} abstain "
          f"{base_overall['abstention_rate']:.4f}", flush=True)
    print(f"[6w.w0] oracle   overall mIoU {oracle_overall['selected_reference_miou']:.4f} "
          f"smallest {oracle_smallest['selected_reference_miou']:.4f} SW "
          f"{oracle_overall['REFERENCE_SELECTION_WRONG']} OK {oracle_overall['REFERENCE_OK']} abstain "
          f"{oracle_overall['abstention_rate']:.4f} rejected {oracle_overall['QUALITY_FILTER_ALL_REJECTED']}",
          flush=True)
    print(f"[6w.w0] gate { {name: entry['passed'] for name, entry in gate.items()} } -> "
          f"{payload['verdict']}", flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
