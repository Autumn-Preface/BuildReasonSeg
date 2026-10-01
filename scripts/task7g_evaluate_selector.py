"""Task 7G Parts H-J — external scene-disjoint selector evaluation (guarded by the section-23 gate).

Section 24 reuses the Task 7E `E-HoldoutL3` (669 records) and `E-PairedHoldout20` byte-for-byte with hashes
re-verified against the Task 7F artifacts. Sections 25-26 compare:

* **G-S0** the frozen production-like selector (max predicted mask area, tie higher confidence then lower index)
  which must reproduce Task 7F F-R0;
* **G-S1** the frozen Task 7G learned selector over the exact same eligible U-C1 proposals (18-D features,
  highest score, exact-float tie → higher confidence → lower index), returning the selected **predicted** mask;
* **G-ORACLE** the frozen Task 7F F-R1 diagnostic value (reused, never recomputed with GT influence).

The script refuses to run when the internal gate did not pass (section 23), so no external artifact is
fabricated after a STOP.

    python scripts/task7g_evaluate_selector.py
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
from buildreasonseg_mvp.task7g_largest_reference_selector import (  # noqa: E402
    SetContextLargestSelector,
    deterministic_max_area,
    proposal_feature_matrix,
    select_with_scores,
    selector_report,
)
from scripts.task7e_evaluate_oracle import read_holdout_rows, read_pairs  # noqa: E402
from buildreasonseg_mvp.task7f_reference_ceiling import (  # noqa: E402
    YOLO_SHA256,
    centroid_error,
    dice_of,
    iou_of,
    sha256_file,
)

EVAL = REPO_ROOT / "evaluation"
OUT_REFERENCE = EVAL / "task7g_external_reference.json"
OUT_PAIRED = EVAL / "task7g_external_paired.json"
INTERNAL = EVAL / "task7g_internal_holdout.json"
TRAINING = EVAL / "task7g_training.json"
MANIFEST = EVAL / "task7g_training_dataset_manifest.json"
TASK7F_MODES = EVAL / "task7f_reference_modes.json"
TASK7F_PAIRED = EVAL / "task7f_paired_reference_modes.json"
TASK7F_VERDICT = EVAL / "task7f_verdict.json"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7g" / "largest_set_context_selector_v1.pt"
TOLERANCE = 1.0e-6
DIRECTIONS = ("above", "below", "left", "right")


def load_selector(device: str = "cpu") -> SetContextLargestSelector:
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = SetContextLargestSelector().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def reference_row(record: dict, mode: str, mask, gt_reference: np.ndarray, scores: dict) -> dict:
    abstained = mask is None
    row = {"sample_id": record["sample_id"], "mode": mode, "direction": record["direction"],
           "program_id": record["program_id"], "tile_id": record["tile_id"], "abstained": abstained,
           "reference_iou": None, "reference_dice": None, "reference_precision_at_0_5": None,
           "centroid_error": None, **scores}
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


def summarise(rows: list[dict]) -> dict:
    answered = [row for row in rows if not row["abstained"]]
    centroids = [row["centroid_error"] for row in answered if row["centroid_error"] is not None]
    if not answered:
        return {"records": len(rows), "answered": 0, "abstentions": len(rows),
                "reference_miou": None}
    return {
        "records": len(rows), "answered": len(answered), "abstentions": len(rows) - len(answered),
        "abstention_rate": (len(rows) - len(answered)) / max(1, len(rows)),
        "reference_miou": float(np.mean([row["reference_iou"] for row in answered])),
        "reference_dice": float(np.mean([row["reference_dice"] for row in answered])),
        "reference_precision_at_0_5": float(np.mean([row["reference_precision_at_0_5"]
                                                    for row in answered])),
        "centroid_error_median": float(np.median(centroids)) if centroids else None,
        "centroid_error_p90": float(np.percentile(centroids, 90)) if centroids else None,
        "oracle_best_top1": float(np.mean([row.get("oracle_best_top1", 0.0) for row in answered])),
        "mean_best_eligible_iou": float(np.mean([row["best_eligible_iou"] for row in answered])),
        "mean_selected_gt_iou": float(np.mean([row["selected_gt_iou"] for row in answered])),
        "mean_best_minus_selected": float(np.mean([row["best_minus_selected"]
                                                   for row in answered])),
        "per_direction": {direction: {
            "records": sum(1 for row in answered if row["direction"] == direction),
            "reference_miou": float(np.mean([row["reference_iou"] for row in answered
                                             if row["direction"] == direction]))
            if any(row["direction"] == direction for row in answered) else None}
            for direction in DIRECTIONS},
    }


def guarded() -> tuple[bool, dict]:
    if not INTERNAL.is_file():
        return False, {"reason": "internal holdout artifact missing"}
    internal = json.loads(INTERNAL.read_text(encoding="utf-8"))
    return bool(internal.get("gate_passed")), internal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    passed, internal = guarded()
    if not passed:
        write_json(OUT_REFERENCE, {
            "_doc": ("Task 7G sections 24-26. External scene-disjoint selection audit, which is conditional on "
                     "the section-23 internal gate. That gate did not pass, so the external stage did not run."),
            "task": "7G", "stage": "H-external-reference", "executed": False,
            "reason": "section 23 gate failed: LARGEST_SELECTOR_NOT_LEARNABLE",
            "internal_gate": internal.get("gate"),
            "training_performed": False, "test_split_used": False,
        })
        print("[7g.ext] not executed: section 23 gate failed (LARGEST_SELECTOR_NOT_LEARNABLE)", flush=True)
        return 4

    from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: F401
        load_frozen_sam2_encoder,
    )
    from buildreasonseg_mvp.task7a_l3_pipeline import L3Pipeline
    from scripts.task6n_train import MaskStore
    from scripts.task6u_common import CONFIGS, PROPOSAL_CHECKPOINT

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    task7f_modes = json.loads(TASK7F_MODES.read_text(encoding="utf-8"))
    task7f_paired = json.loads(TASK7F_PAIRED.read_text(encoding="utf-8"))
    records = read_holdout_rows()
    pairs = read_pairs()
    model = load_selector("cpu")
    pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device)
    masks = MaskStore()
    from buildreasonseg_mvp.task6q_reference_resolver import is_eligible

    eligible_cache: dict[str, list] = {}

    def eligible_of(tile_id: str, image_path) -> list:
        if tile_id not in eligible_cache:
            proposals = pipeline.proposals(tile_id, Path(image_path))
            eligible_cache[tile_id] = [proposal for proposal in proposals
                                       if is_eligible(proposal, "largest")]
        return eligible_cache[tile_id]

    rows = {"G-S0": [], "G-S1": []}
    gt_iou_by_record = {}
    for record in records:
        eligible = eligible_of(record["tile_id"], record["image_path"])
        gt_reference = np.asarray(masks.mask(record["tile_id"],
                                             record["reference_source_feature_id"]), dtype=bool)
        if not eligible:
            for mode in ("G-S0", "G-S1"):
                rows[mode].append(reference_row(record, mode, None, gt_reference,
                                                {"best_eligible_iou": 0.0, "selected_gt_iou": 0.0,
                                                 "best_minus_selected": 0.0,
                                                 "oracle_best_top1": 0.0}))
            gt_iou_by_record[record["sample_id"]] = {"G-S0": 0.0, "G-S1": 0.0}
            continue
        ious = [iou_of(np.asarray(proposal.mask, dtype=bool), gt_reference)
                for proposal in eligible]
        best = float(max(ious))
        target = int(np.argmax(ious))
        features = torch.as_tensor(proposal_feature_matrix(eligible))
        with torch.no_grad():
            scores = model(features).numpy()
        selected = {"G-S0": deterministic_max_area(eligible),
                    "G-S1": select_with_scores(scores, eligible)}
        gt_iou_by_record[record["sample_id"]] = {mode: float(ious[index])
                                                 for mode, index in selected.items()}
        for mode, index in selected.items():
            rows[mode].append(reference_row(
                record, mode, np.asarray(eligible[index].mask, dtype=bool), gt_reference,
                {"best_eligible_iou": best, "selected_gt_iou": float(ious[index]),
                 "best_minus_selected": best - float(ious[index]),
                 "oracle_best_top1": 1.0 if index == target else 0.0}))

    # G-S0 must reproduce Task 7F F-R0
    f_r0 = task7f_modes["reference"]["F-R0"]
    g_s0 = summarise(rows["G-S0"])
    g_s1 = summarise(rows["G-S1"])
    g_oracle = {"reference_miou": task7f_modes["reference"]["F-R1"]["reference_miou"],
                "source": "frozen Task 7F F-R1 diagnostic value",
                "selection_ceiling_miou": task7f_modes["reference"]["F-R1"]["reference_miou"]}
    reproduction = {
        "reference_miou": {"measured": g_s0["reference_miou"], "expected": f_r0["reference_miou"],
                           "delta": abs(g_s0["reference_miou"] - f_r0["reference_miou"]),
                           "passed": abs(g_s0["reference_miou"] - f_r0["reference_miou"]) <= TOLERANCE},
        "answered": {"measured": g_s0["answered"], "expected": f_r0["answered"],
                     "passed": g_s0["answered"] == f_r0["answered"]},
        "abstentions": {"measured": g_s0["abstentions"], "expected": f_r0["abstentions"],
                        "passed": g_s0["abstentions"] == f_r0["abstentions"]},
    }
    reproduction["passed"] = all(entry["passed"] for entry in reproduction.values()
                                 if isinstance(entry, dict))
    improved = sum(1 for record in records
                   if gt_iou_by_record[record["sample_id"]]["G-S1"]
                   - gt_iou_by_record[record["sample_id"]]["G-S0"] >= 0.10)
    worsened = sum(1 for record in records
                   if gt_iou_by_record[record["sample_id"]]["G-S0"]
                   - gt_iou_by_record[record["sample_id"]]["G-S1"] >= 0.10)
    write_json(OUT_REFERENCE, {
        "_doc": ("Task 7G sections 24-26. External scene-disjoint reference audit on the frozen Task 7E "
                 "E-HoldoutL3 with the deterministic G-S0 baseline, the learned G-S1 selector and the frozen "
                 "Task 7F F-R1 diagnostic ceiling. GT is used for offline metrics only."),
        "task": "7G", "stage": "H-external-reference",
        "holdout": {"records": len(records), "source": "task7e_holdout_manifest / task7f checks",
                    "record_id_hash": manifest["rows_path"] and
                    json.loads((EVAL / "task7e_holdout_manifest.json").read_text(encoding="utf-8"))
                    ["holdout"]["record_id_hash"]},
        "checkpoint": {"path": str(CHECKPOINT), "sha256": sha256_file(CHECKPOINT),
                       "selected_epoch": training["selected_epoch"]},
        "selector": selector_report(), "proposal_config": dict(CONFIGS["U-C1"]),
        "yolo_sha256": sha256_file(PROPOSAL_CHECKPOINT), "yolo_sha256_expected": YOLO_SHA256,
        "g_s0": g_s0, "g_s1": g_s1, "g_oracle": g_oracle,
        "improved_by_0_10": improved, "worsened_by_0_10": worsened,
        "best_eligible_coverage": {
            "at_0_25": float(np.mean([row["best_eligible_iou"] >= 0.25 for row in rows["G-S0"]])),
            "at_0_50": float(np.mean([row["best_eligible_iou"] >= 0.50 for row in rows["G-S0"]])),
            "at_0_75": float(np.mean([row["best_eligible_iou"] >= 0.75 for row in rows["G-S0"]])),
        },
        "reproduction": reproduction, "reproduction_passed": reproduction["passed"],
        "test_split_used": False, "training_performed": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7g.ext] G-S0 ref mIoU {g_s0['reference_miou']:.4f} (Δ"
          f"{reproduction['reference_miou']['delta']:.2e}) | G-S1 {g_s1['reference_miou']:.4f} | "
          f"oracle {g_oracle['reference_miou']:.4f} | improved {improved} worsened {worsened}",
          flush=True)

    paired_rows = []
    for pair in pairs:
        members = [next(record for record in records if record["sample_id"] == member["sample_id"])
                   for member in pair["members"]]
        eligible = eligible_of(pair["tile_id"], members[0]["image_path"])
        entry = {"pair_id": pair["pair_id"], "eligible_count": len(eligible), "modes": {}}
        if not eligible:
            for mode in ("G-S0", "G-S1"):
                entry["modes"][mode] = {"abstained": True, "passed": False, "margin": 0.0}
            paired_rows.append(entry)
            continue
        features = torch.as_tensor(proposal_feature_matrix(eligible))
        with torch.no_grad():
            scores = model(features).numpy()
        selected = {"G-S0": deterministic_max_area(eligible),
                    "G-S1": select_with_scores(scores, eligible)}
        for mode, index in selected.items():
            mask = np.asarray(eligible[index].mask, dtype=bool)
            own, cross = [], []
            for position, member in enumerate(members):
                reference = mask
                own_mask = np.asarray(masks.mask(member["tile_id"],
                                                 member["target_source_feature_id"]), dtype=bool)
                other = members[1 - position]
                other_mask = np.asarray(masks.mask(other["tile_id"],
                                                   other["target_source_feature_id"]), dtype=bool)
                own.append(_target_iou(member, reference, own_mask, pipeline, masks, args))
                cross.append(_target_iou(other, reference, other_mask, pipeline, masks, args))
            entry["modes"][mode] = {"abstained": False,
                                    "passed": bool(own[0] > cross[0] and own[1] > cross[1]),
                                    "own_iou": own, "cross_iou": cross,
                                    "margin": float(np.mean(own) - np.mean(cross))}
        paired_rows.append(entry)

    paired_summary = {}
    for mode in ("G-S0", "G-S1"):
        answered = [row for row in paired_rows if not row["modes"][mode]["abstained"]]
        own = [value for row in answered for value in row["modes"][mode]["own_iou"]]
        cross = [value for row in answered for value in row["modes"][mode]["cross_iou"]]
        paired_summary[mode] = {
            "passed": sum(1 for row in answered if row["modes"][mode]["passed"]),
            "pairs": len(paired_rows), "answered_pairs": len(answered),
            "abstention_pairs": len(paired_rows) - len(answered),
            "mean_own_iou": float(np.mean(own)) if own else None,
            "mean_cross_iou": float(np.mean(cross)) if cross else None,
            "own_cross_margin": float(np.mean(own) - np.mean(cross)) if own else None}
    f_r0_paired = task7f_paired["results"]["F-R0"]
    paired_summary["G-ORACLE"] = {"passed": task7f_paired["results"]["F-R1"]["passed"],
                                  "own_cross_margin": task7f_paired["results"]["F-R1"]["own_cross_margin"],
                                  "source": "frozen Task 7F F-R1"}
    paired_reproduction = {
        "passed": {"measured": paired_summary["G-S0"]["passed"],
                   "expected": f_r0_paired["passed"],
                   "ok": paired_summary["G-S0"]["passed"] == f_r0_paired["passed"]},
        "margin": {"measured": paired_summary["G-S0"]["own_cross_margin"],
                   "expected": f_r0_paired["own_cross_margin"],
                   "delta": abs(paired_summary["G-S0"]["own_cross_margin"]
                                - f_r0_paired["own_cross_margin"]),
                   "ok": abs(paired_summary["G-S0"]["own_cross_margin"]
                             - f_r0_paired["own_cross_margin"]) <= TOLERANCE},
    }
    paired_reproduction["passed_ok"] = paired_reproduction["passed"]["ok"] \
        and paired_reproduction["margin"]["ok"]
    write_json(OUT_PAIRED, {
        "_doc": ("Task 7G section 29. External paired counterfactual on the frozen E-PairedHoldout20: the "
                 "selector runs once per shared tile/reference and the selected predicted mask is reused for "
                 "both direction programs."),
        "task": "7G", "stage": "J-external-paired",
        "pairs": len(paired_rows), "results": paired_summary, "rows": paired_rows,
        "reproduction": paired_reproduction,
        "reproduction_passed": paired_reproduction["passed_ok"],
        "test_split_used": False, "training_performed": False,
    })
    print(f"[7g.ext-paired] G-S0 {paired_summary['G-S0']['passed']}/20 "
          f"(+{paired_summary['G-S0']['own_cross_margin']:.4f}) | G-S1 "
          f"{paired_summary['G-S1']['passed']}/20 "
          f"(+{paired_summary['G-S1']['own_cross_margin']:.4f})", flush=True)
    return 0


def _target_iou(record: dict, reference: np.ndarray, target: np.ndarray, pipeline, masks, args) -> float:
    from buildreasonseg_mvp.task7d_data import CompetitionBatch

    fields = _fields(reference, record)
    visual = pipeline.store.get(record["tile_id"], record["image_path"]).float()[None]
    relation = {"left": "left_of", "right": "right_of", "above": "above",
                "below": "below"}[record["direction"]]
    batch = CompetitionBatch(
        visual=visual.to(args.device),
        directional=torch.as_tensor(fields["P_dir_64"])[None, None].to(args.device),
        nearest=torch.as_tensor(fields["P_near_64"])[None, None].to(args.device),
        relations=[relation], target=torch.zeros(1, 512, 512, device=args.device))
    from buildreasonseg_mvp.task7e_l3_decoder_adapter import FrozenL3Decoder

    decoder = _decoder_cache.setdefault("D-B1", FrozenL3Decoder("D-B1", args.device))
    with torch.no_grad():
        logits = decoder.model(batch.visual, batch.relations, batch.directional, batch.nearest)
        prediction = (decoder.model.upsampled(logits, 512) > 0.0).cpu().numpy()[0, 0]
    return iou_of(prediction, target)


def _fields(reference: np.ndarray, record: dict) -> dict:
    from buildreasonseg_mvp.task6z_field_composition import record_fields

    return record_fields(np.asarray(reference, dtype=bool), record["program_id"])


_decoder_cache: dict = {}

if __name__ == "__main__":
    raise SystemExit(main())
