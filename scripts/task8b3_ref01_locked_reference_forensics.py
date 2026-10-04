"""Task 8B.3-REF01-F1-R2 forensic script (proposal mask = rerun GlobalProposal.mask_crop)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
CANON = REPO / "delivery_src" / "BuildReasonSeg_Advisor_RC1"
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
CACHE = REPO / "artifacts" / "whu_native_vector" / "instances"
EVIDENCE = REPO / "evaluation" / "task8b3_ref01_locked_reference_forensics.json"
ROOT = Path(r"C:\D\resources\Satellite dataset \Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image")
THRESHOLD = 0.50
CASES = [("right", "1010", 4, 6, 6, 4), ("left", "1003", 26, 66, 53, 42),
         ("above", "1008", 4, 9, 9, 4), ("below", "1009", 3, 7, 6, 3)]

sys.path.insert(0, str(CANON))
from buildreasonseg.runtime.detector import (DetectorRuntime, TILE_SIZE, TILE_OVERLAP, IMGSZ, CONF, MAX_DET,
                                            FROZEN_THRESHOLD, eligible_proposals, select_reference)


def gt_mask(tile: str, instance: int) -> np.ndarray:
    with np.load(CACHE / f"{tile}.npz", allow_pickle=False) as archive:
        label = archive["label_map"]
    return label == instance


def proposal_mask(proposal, size: int = 512) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    top, left, _bottom, _right = proposal.global_bbox
    crop = proposal.mask_crop
    mask[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return mask


def iou(first: np.ndarray, second: np.ndarray) -> float:
    inter = float(np.logical_and(first, second).sum())
    union = float(np.logical_or(first, second).sum())
    return 0.0 if union == 0 else inter / union


def main() -> int:
    runtime = DetectorRuntime(checkpoint=str(EXTERNAL / "model" / "buildreasonseg_advisor" / "detector.pt"),
                              device="cpu")
    records = []
    for relation, tile, instance, _raw, _merged, _eligible in CASES:
        raster = np.asarray(Image.open(ROOT / f"{tile}.tif").convert("RGB"))
        detection = runtime.detect_global(raster)
        merged = detection["merged"]
        reference = gt_mask(tile, instance)
        eligible = eligible_proposals(merged)
        selected = select_reference(merged)
        selected_iou = iou(proposal_mask(selected), reference) if selected else 0.0
        best_eligible_iou = max((iou(proposal_mask(p), reference) for p in eligible), default=0.0)
        best_any_iou = max((iou(proposal_mask(p), reference) for p in merged), default=0.0)
        if selected_iou >= THRESHOLD:
            klass = "REFERENCE_SELECTED_CORRECT"
        elif best_eligible_iou >= THRESHOLD:
            klass = "REFERENCE_SELECTION_WRONG_COVERED"
        elif best_any_iou >= THRESHOLD:
            klass = "REFERENCE_ELIGIBILITY_BLOCKED"
        else:
            klass = "REFERENCE_COVERAGE_MISSING"
        records.append({
            "relation": relation, "tile": tile, "gt_reference_instance": instance,
            "gt_reference_pixels": int(reference.sum()),
            "raw_proposal_count": detection["raw_count"], "merged_proposal_count": len(merged),
            "eligible_count": len(eligible),
            "selected_id": None if selected is None else selected.proposal_id,
            "selected_iou": selected_iou, "best_eligible_iou": best_eligible_iou,
            "best_any_iou": best_any_iou, "classification": klass,
            "proposals": [{"proposal_id": p.proposal_id, "mask_area": p.mask_area,
                           "touches_border": p.touches_image_border,
                           "bbox_extent_ratio": p.bbox_extent_ratio,
                           "iou_to_gt": iou(proposal_mask(p), reference)} for p in merged]})
    counts = {}
    for record in records:
        counts[record["classification"]] = counts.get(record["classification"], 0) + 1
    if counts.get("REFERENCE_COVERAGE_MISSING"):
        outcome, blocker, nxt = ("REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE", "COVERAGE",
                                 "REF01_COVERAGE_FRAGMENTATION_FORENSICS")
    elif counts.get("REFERENCE_ELIGIBILITY_BLOCKED"):
        outcome, blocker, nxt = ("REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE", "ELIGIBILITY",
                                 "REF01_ELIGIBILITY_FORENSICS")
    elif counts.get("REFERENCE_SELECTION_WRONG_COVERED"):
        outcome, blocker, nxt = ("REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE", "SELECTION",
                                 "REF01_SELECTION_REPAIR_DESIGN")
    else:
        outcome, blocker, nxt = ("REF01_LOCKED_DEMO_REFERENCE_GATE_PASS", "NONE_IN_LOCKED_REFERENCE_SET",
                                 "MASK01_LOCKED_DEMO_END_TO_END_FORENSICS")
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "starting_head": "141af9cfd943dd13366947ea3a31637a3f8c7bc9",
        "scientific_reuse_disclosure": "GT access purpose: REFERENCE_FORENSICS_ONLY (native-vector annotations used "
                                      "only to measure whether the frozen proposal set covers the canonical GT "
                                      "reference; no target GT metric, no repair, no candidate replacement)",
        "detector_module": str(CANON / "buildreasonseg" / "runtime" / "detector.py"),
        "detector_config": {"TILE_SIZE": TILE_SIZE, "TILE_OVERLAP": TILE_OVERLAP, "IMGSZ": IMGSZ, "CONF": CONF,
                            "MAX_DET": MAX_DET, "FROZEN_THRESHOLD": FROZEN_THRESHOLD},
        "detector_call_count": len(records), "coverage_threshold": THRESHOLD,
        "proposal_mask_source": "RERUN_GLOBALPROPOSAL_MASK_CROP",
        "records": records, "class_counts": counts, "outcome": outcome,
        "dominant_next_blocker": blocker, "next": nxt}, indent=1), encoding="utf-8")
    print(json.dumps({"class_counts": counts, "outcome": outcome, "blocker": blocker, "next": nxt}))
    for record in records:
        print(json.dumps({k: v for k, v in record.items() if k != "proposals"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
