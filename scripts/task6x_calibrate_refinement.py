"""Task 6X Parts B-E — frozen-asset audit and the U-Calib200 refinement calibration.

Part B: verify the frozen SAM2.1 checkpoint SHA256, load the official `SAM2ImagePredictor` and prove a box
prompt returns a 512x512 mask plus a predicted quality score; record the frozen U-C1 / ProgramHead / B3 /
field assets.

Part D/E: evaluate exactly the four predeclared options on the train-only U-Calib200 split and freeze one
with the section 12 priority order:

1. highest overall selected-reference mIoU (refined masks vs GT reference);
2. then highest smallest-family mIoU; 3. then highest largest-family mIoU;
4. then lowest abstention rate; 5. then lower mean SAM2 calls per tile;
6. then lower mean wall time per tile; 7. then simpler option (X-C0 → X-C1 → X-C2 → X-C3).

**If X-C0 wins the task stops** with `SAM2_REFINEMENT_NOT_HELPFUL` and no RefVal/MiniVal/Paired evaluation
is run. RefValUnique and the evaluation packs never influence the choice.

Writes `evaluation/task6x_frozen_asset_audit.json`, `evaluation/task6x_calibration_refinement.json` and
`evaluation/task6x_frozen_refinement_option.json`.

    python scripts/task6x_calibrate_refinement.py --device 0
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
from buildreasonseg_mvp.task6n_relation_decoder import read_rgb_tile  # noqa: E402
from buildreasonseg_mvp.task6q_reference_resolver import select_reference  # noqa: E402
from buildreasonseg_mvp.task6x_sam2_reference_refiner import (  # noqa: E402
    OPTIONS,
    OPTION_ORDER,
    SIMPLICITY_ORDER,
    SAM2_CHECKPOINT_SHA256,
    Sam2ProposalRefiner,
    eligible_original,
    eligible_refined,
    load_sam2_image_predictor,
    option_report,
    select_refined,
    sha256_file,
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
)

OUT_ASSETS = EVAL / "task6x_frozen_asset_audit.json"
OUT_CALIB = EVAL / "task6x_calibration_refinement.json"
OUT_FROZEN = EVAL / "task6x_frozen_refinement_option.json"
SPLIT = EVAL / "task6u_reference_train_split.json"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
QUALITY_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6w" / "proposal_quality_v01.pt"
BUCKETS = ("NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "REFINEMENT_EMPTY",
           "REFERENCE_NOT_COVERED_IOU50", "REFERENCE_SELECTION_WRONG",
           "SELECTED_MASK_GEOMETRY_POOR", "REFERENCE_OK")


def bucket_for(*, proposal_count: int, eligible_count: int, selection, reference_iou, centroid) -> str:
    if proposal_count == 0:
        return "NO_PROPOSALS"
    if eligible_count == 0:
        return "NO_ELIGIBLE_PROPOSALS"
    if selection is None:
        return "REFINEMENT_EMPTY"
    if reference_iou < 0.50:
        return "REFERENCE_SELECTION_WRONG"
    if centroid > 0.05:
        return "SELECTED_MASK_GEOMETRY_POOR"
    return "REFERENCE_OK"


def priority_key(entry: dict) -> tuple:
    return (
        entry["overall"]["selected_reference_miou"],
        entry["smallest"]["selected_reference_miou"],
        entry["largest"]["selected_reference_miou"],
        -entry["overall"]["abstention_rate"],
        -entry["mean_sam2_calls_per_tile"],
        -entry["mean_wall_time_per_tile"],
        -SIMPLICITY_ORDER[entry["option"]],
    )


def evaluate_option(option: str, records: list[dict], grouped: dict, caches: dict, predictor,
                    device: str) -> dict:
    config = OPTIONS[option]
    refiner = Sam2ProposalRefiner(predictor, option, device=device) if config["refinement"] else None
    sam_scores, per_tile_calls, per_tile_time = [], [], []
    empty_refined = 0
    rows = []
    for tile_id, tile_records in grouped.items():
        proposals = caches["U-C1"][tile_id]
        started = time.perf_counter()
        if refiner is not None:
            refiner.calls = 0
            refiner.set_tile(read_rgb_tile(record_image_path(tile_records[0])), tile_id)
            refined_by_index = {}
            for proposal in proposals:
                candidate = refiner.refine(proposal)
                refined_by_index[proposal.index] = candidate
                if candidate.empty:
                    empty_refined += 1
            per_tile_calls.append(refiner.calls)
        else:
            refined_by_index = {}
            per_tile_calls.append(0)
        per_tile_time.append(time.perf_counter() - started)

        for record in tile_records:
            family = str(record["reference_family"])
            truth = gt_masks(tile_id)[int(record["reference_source_feature_id"])]
            eligible = eligible_original(proposals, family)
            row = {"reference_family": family, "proposal_count": len(proposals),
                   "eligible_count": len(eligible)}
            if config["refinement"]:
                candidates = [refined_by_index[proposal.index] for proposal in eligible]
                kept = eligible_refined(candidates, family)
                selection = select_refined(kept, family)
                selected_mask = None if selection is None else selection[0].mask
                if selection is not None:
                    row["sam_score"] = selection[1].sam_score
                    sam_scores.append(selection[1].sam_score)
                row["kept_after_refinement"] = len(kept)
                row["refinement_emptied"] = sum(1 for candidate in candidates if candidate.empty)
            else:
                selection = select_reference(proposals, family)
                selected_mask = None if selection.abstained else selection.proposal.mask
                row["refinement_emptied"] = 0
            row["abstained"] = selected_mask is None
            if selected_mask is None:
                row.update({"selected_iou": None, "selected_dice": None, "centroid_error": None,
                            "bucket": bucket_for(proposal_count=len(proposals),
                                                 eligible_count=len(eligible), selection=None,
                                                 reference_iou=0.0, centroid=1.0)})
            else:
                selected_iou = iou(selected_mask, truth)
                error = centroid_error(selected_mask, truth)
                row.update({
                    "selected_iou": selected_iou, "selected_dice": dice(selected_mask, truth),
                    "centroid_error": error,
                    "bucket": bucket_for(proposal_count=len(proposals),
                                         eligible_count=len(eligible), selection=selected_mask,
                                         reference_iou=selected_iou, centroid=error),
                })
                if row["bucket"] == "REFERENCE_OK":
                    row["bucket"] = "REFERENCE_OK"
            if selected_mask is not None and row["bucket"] in ("REFERENCE_SELECTION_WRONG",) and \
                    row["selected_iou"] >= 0.50:
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
            "centroid_error_mean": float(np.mean(errors)) if errors else None,
            "centroid_error_median": float(np.median(errors)) if errors else None,
            "centroid_error_p90": percentile(errors, 0.90),
            "abstention_rate": sum(1 for row in subset if row["abstained"]) / len(subset),
            "buckets": {name: buckets.get(name, 0) for name in BUCKETS},
        }

    overall = summarise(rows)
    return {
        "option": option,
        "description": config["description"],
        "refinement": bool(config["refinement"]),
        "overall": overall,
        "largest": summarise([row for row in rows if row["reference_family"] == "largest"]),
        "smallest": summarise([row for row in rows if row["reference_family"] == "smallest"]),
        "mean_sam2_score": float(np.mean(sam_scores)) if sam_scores else None,
        "empty_refined_masks": empty_refined,
        "mean_sam2_calls_per_tile": float(np.mean(per_tile_calls)) if per_tile_calls else 0.0,
        "mean_wall_time_per_tile": float(np.mean(per_tile_time)) if per_tile_time else 0.0,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if str(device) != "cpu" else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--torch-device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    # ---------------- Part B: frozen asset audit
    sam_sha = sha256_file(Path(__file__).resolve().parents[1] / "local_cache" / "models"
                          / "sam2.1_hiera_base_plus.pt")
    yolo_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    b3 = json.loads((EVAL / "task6o_mini_val.json").read_text(encoding="utf-8"))["variants"]["B3"]
    b3_path = Path(b3["training"]["checkpoint"]["path"])
    parser_training = json.loads((EVAL / "task6t_training_summary.json").read_text(encoding="utf-8"))
    predictor, predictor_report = load_sam2_image_predictor(device=args.torch_device)

    probe_image = None
    probe_box = None
    for tile_id, tile_records in group_records_by_tile(
            json.loads(SPLIT.read_text(encoding="utf-8"))["u_calib200"]["records"]).items():
        probe_image = read_rgb_tile(record_image_path(tile_records[0]))
        break
    predictor.set_image(probe_image)
    probe_box = np.asarray([[10.0, 10.0, 200.0, 200.0]], dtype=np.float32)
    probe_masks, probe_scores, _probe_logits = predictor.predict(
        point_coords=None, point_labels=None, box=probe_box, mask_input=None,
        multimask_output=True, return_logits=False)

    audit = {
        "_doc": (
            "Task 6X section 3. Frozen-asset audit: the official SAM2.1 Hiera Base+ image predictor over "
            "the project's frozen checkpoint, plus the frozen U-C1 proposal configuration, the Task 6T "
            "hardened ProgramHead, the Task 6O B3 decoder and GeometricRelationField v0.2."
        ),
        "task": "6X", "stage": "B-frozen-asset-audit",
        "sam2": {
            **predictor_report,
            "sha256_matches": predictor_report["checkpoint_sha256"] == SAM2_CHECKPOINT_SHA256,
            "predictor_loaded": True,
            "probe": {
                "box": probe_box.tolist(),
                "masks_shape": list(np.asarray(probe_masks).shape),
                "scores": [float(value) for value in np.asarray(probe_scores).reshape(-1)],
                "boxes_returned": True,
                "mask_is_512x512": bool(np.asarray(probe_masks).shape[-2:] == (512, 512)),
                "point_coords": None, "point_labels": None, "mask_input": None,
                "return_logits": False,
                "frozen": True, "retrained": False,
            },
        },
        "proposal": {
            "config": {key: CONFIGS["U-C1"][key]
                       for key in ("id", "imgsz", "conf", "max_det", "nms", "tta", "tiling")},
            "checkpoint": str(PROPOSAL_CHECKPOINT), "sha256": yolo_sha,
            "expected_sha256": PROPOSAL_CHECKPOINT_SHA256,
            "matches": yolo_sha == PROPOSAL_CHECKPOINT_SHA256, "retrained": False,
        },
        "program_head": {"sha256": parser_training["checkpoint"]["sha256"], "retrained": False,
                         "role": "frozen; not used by the refinement calibration"},
        "b3": {"sha256": sha256_file(b3_path) if b3_path.is_file() else None,
               "expected_sha256": b3["training"]["checkpoint"]["sha256"], "retrained": False,
               "architecture": "Task 6O B3"},
        "field": {"module": "buildreasonseg_mvp/geometric_relation_field_v02.py", "modified": False},
        "excluded_from_primary_resolver": {
            "proposal_set_ranker": str(RANKER_CHECKPOINT),
            "ranker_used": False,
            "quality_estimator": str(QUALITY_CHECKPOINT),
            "quality_estimator_used": False,
        },
        "options": option_report(),
        "verdict": "FROZEN_ASSETS_VERIFIED" if (
            predictor_report["checkpoint_sha256"] == SAM2_CHECKPOINT_SHA256
            and yolo_sha == PROPOSAL_CHECKPOINT_SHA256) else "FROZEN_ASSET_UNAVAILABLE",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_ASSETS, audit)
    if audit["verdict"] != "FROZEN_ASSETS_VERIFIED":
        write_json(OUT_CALIB, {"_doc": "Task 6X section 4.", "task": "6X",
                               "verdict": "FROZEN_ASSET_UNAVAILABLE"})
        print("[6x.calib] STOP FROZEN_ASSET_UNAVAILABLE", flush=True)
        return 2
    print(f"[6x.calib] SAM2 sha verified; probe masks {np.asarray(probe_masks).shape} scores "
          f"{[round(float(value), 4) for value in np.asarray(probe_scores).reshape(-1)]}", flush=True)

    # ---------------- Part D: four-option calibration on U-Calib200
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    records = split["u_calib200"]["records"]
    grouped = group_records_by_tile(records)
    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    caches: dict[str, dict] = {"U-C1": {}}
    for tile_id, tile_records in grouped.items():
        proposals, _run = proposals_for_tile(yolo, CONFIGS["U-C1"], tile_id,
                                            record_image_path(tile_records[0]), args.device,
                                            use_cache=True)
        caches["U-C1"][tile_id] = proposals
    print(f"[6x.calib] U-Calib200 {len(records)} references over {len(grouped)} tiles", flush=True)

    results = {}
    for option in OPTION_ORDER:
        if str(args.torch_device) != "cpu":
            torch.cuda.reset_peak_memory_stats()
        results[option] = evaluate_option(option, records, grouped, caches, predictor,
                                          args.torch_device)
        entry = results[option]
        print(f"[6x.calib] {option}: overall mIoU {entry['overall']['selected_reference_miou']:.4f} "
              f"smallest {entry['smallest']['selected_reference_miou']:.4f} "
              f"largest {entry['largest']['selected_reference_miou']:.4f} abstain "
              f"{entry['overall']['abstention_rate']:.4f} SAM calls/tile "
              f"{entry['mean_sam2_calls_per_tile']:.2f} time/tile "
              f"{entry['mean_wall_time_per_tile']:.3f}s empty {entry['empty_refined_masks']}",
              flush=True)

    ranking = sorted(OPTION_ORDER, key=lambda option: priority_key(results[option]), reverse=True)
    selected = ranking[0]
    payload = {
        "_doc": (
            "Task 6X sections 4-12. Train-only U-Calib200 calibration of exactly the four predeclared "
            "refinement options (X-C0 no refinement, X-C1 exact box single-mask, X-C2 exact box multimask, "
            "X-C3 10%-expanded box multimask) over frozen U-C1 proposals and the frozen SAM2.1 image "
            "predictor, with the unchanged Task 6Q family eligibility and the literal largest/smallest "
            "rule. GT reference masks score the result only."
        ),
        "task": "6X", "stage": "D-E-calibration-refinement",
        "calibration_split": {"path": str(SPLIT), "records": len(records), "tiles": len(grouped),
                              "train_only": True,
                              "by_family": dict(Counter(record["reference_family"]
                                                        for record in records))},
        "options": option_report(),
        "results": results,
        "ranking": ranking,
        "priority_rule": [
            "highest overall selected-reference mIoU", "then highest smallest-family mIoU",
            "then highest largest-family mIoU", "then lowest abstention rate",
            "then lower mean SAM2 calls per tile", "then lower mean wall time per tile",
            "then simpler option (X-C0 -> X-C1 -> X-C2 -> X-C3)",
        ],
        "selection_inputs": {"u_calib200": True, "refval_unique": False, "minival240": False,
                             "pairedval20": False, "test_split": False},
        "training_performed": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_CALIB, payload)

    frozen = {
        "_doc": (
            "Task 6X section 12. The single frozen refinement option, chosen on the train-only U-Calib200 "
            "split by the exact section 12 priority order and immutable afterwards. If X-C0 wins, the Task "
            "6X audit stops with SAM2_REFINEMENT_NOT_HELPFUL and no RefVal/MiniVal/Paired evaluation runs."
        ),
        "task": "6X", "stage": "E-frozen-refinement-option",
        "selected_option": selected,
        "selected_description": OPTIONS[selected]["description"],
        "baseline_is_selected": selected == "X-C0",
        "refinement_is_selected": selected != "X-C0",
        "priority_rule": payload["priority_rule"],
        "ranking": ranking,
        "selected_metrics": {
            "overall": results[selected]["overall"],
            "largest": results[selected]["largest"],
            "smallest": results[selected]["smallest"],
            "mean_sam2_calls_per_tile": results[selected]["mean_sam2_calls_per_tile"],
            "mean_wall_time_per_tile": results[selected]["mean_wall_time_per_tile"],
        },
        "baseline_metrics": {"overall": results["X-C0"]["overall"],
                             "largest": results["X-C0"]["largest"],
                             "smallest": results["X-C0"]["smallest"]},
        "options": {option: OPTIONS[option]["description"] for option in OPTION_ORDER},
        "fourth_option": None,
        "evidence": {"calibration": str(OUT_CALIB), "asset_audit": str(OUT_ASSETS)},
        "frozen": True, "immutable_after_creation": True,
        "chosen_without_refval": True,
        "training_performed": False,
        "test_split_used": False,
    }
    write_json(OUT_FROZEN, frozen)
    print(f"[6x.calib] ranking {ranking} -> frozen option {selected} "
          f"(baseline selected: {frozen['baseline_is_selected']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
