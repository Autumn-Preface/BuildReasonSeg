"""Task 8B.3-REF01-F1-R3 forensic harness — locked candidate GT-reference IoU classification.

Mask source: the rerun detector's in-memory GlobalProposal.mask_crop (proposals.json is metadata only).
Coverage threshold: frozen Task 7F IoU 0.50. Exactly one detector pass per locked candidate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
CACHE = REPO / "artifacts" / "whu_native_vector" / "instances"
DIAG = EXTERNAL / "inference" / "output" / "diagnostics"
EVIDENCE = REPO / "evaluation" / "task8b3_ref01_locked_reference_forensics.json"
RASTER_ROOT = (Path(r"C:\D\resources") / "Satellite dataset " + chr(0x2161) + " (East Asia)"
               / "1. The cropped image data and raster labels" / "test" / "image")
THRESHOLD = 0.50
CASES = [("right", "1010", 4, 6, 6, 4), ("left", "1003", 26, 66, 53, 42),
         ("above", "1008", 4, 9, 9, 4), ("below", "1009", 3, 7, 6, 3)]

sys.path.insert(0, str(EXTERNAL))
from buildreasonseg.runtime.detector import (DetectorRuntime, TILE_SIZE, TILE_OVERLAP, IMGSZ, CONF, MAX_DET,
                                            FROZEN_THRESHOLD, eligible_proposals, select_reference)


def gt_mask(tile: str, instance: int) -> np.ndarray:
    with np.load(CACHE / f"{tile}.npz", allow_pickle=False) as archive:
        label = archive["label_map"]
    return label == instance


def proposal_mask(proposal, size: int = TILE_SIZE) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    top, left, _bottom, _right = proposal.global_bbox
    crop = proposal.mask_crop
    mask[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return mask


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = float(np.logical_or(first, second).sum())
    return 0.0 if union == 0 else float(np.logical_and(first, second).sum()) / union


def main() -> int:
    runtime = DetectorRuntime(checkpoint=str(EXTERNAL / "model" / "buildreasonseg_advisor" / "detector.pt"),
                              device="cpu")
    detector_calls = 0
    records = []
    reproduction = []
    for relation, tile, instance, exp_raw, exp_merged, exp_eligible in CASES:
        raster_path = RASTER_ROOT / f"{tile}.tif"
        raster = np.asarray(Image.open(raster_path).convert("RGB"))
        detection = runtime.detect_global(raster)
        detector_calls += 1
        merged = detection["merged"]
        eligible = eligible_proposals(merged)
        selected = select_reference(merged)
        reference = gt_mask(tile, instance)
        selected_iou = iou(proposal_mask(selected), reference) if selected is not None else 0.0
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
        stored = json.loads((DIAG / tile / "result.json").read_text(encoding="utf-8"))
        stored_items = json.loads((DIAG / tile / "proposals.json").read_text(encoding="utf-8"))
        items = stored_items.get("items") or stored_items.get("proposals") or []
        stored_eligible = sum(1 for item in items
                              if (item.get("mask_area", 0) or 0) > 0
                              and item.get("touches_image_border") is False
                              and (item.get("bbox_extent_ratio", 1) or 1) <= 0.20)
        match = (detection["raw_count"] == exp_raw and len(merged) == exp_merged
                 and len(eligible) == exp_eligible and stored.get("raw_proposal_count") == exp_raw
                 and stored.get("merged_proposal_count") == exp_merged and stored_eligible == exp_eligible)
        reproduction.append({"relation": relation, "rerun_raw": detection["raw_count"],
                             "rerun_merged": len(merged), "rerun_eligible": len(eligible),
                             "stored_raw": stored.get("raw_proposal_count"),
                             "stored_merged": stored.get("merged_proposal_count"),
                             "stored_eligible": stored_eligible, "match": bool(match)})
        records.append({"relation": relation, "tile": tile, "gt_reference_instance": instance,
                        "gt_reference_pixels": int(reference.sum()), "expected_raw": exp_raw,
                        "expected_merged": exp_merged, "expected_eligible": exp_eligible,
                        "raw_proposal_count": detection["raw_count"], "merged_proposal_count": len(merged),
                        "eligible_count": len(eligible),
                        "selected_id": None if selected is None else selected.proposal_id,
                        "selected_iou": selected_iou, "best_eligible_iou": best_eligible_iou,
                        "best_any_iou": best_any_iou, "classification": klass,
                        "proposals": [{"proposal_id": p.proposal_id, "mask_area": p.mask_area,
                                       "touches_image_border": p.touches_image_border,
                                       "bbox_extent_ratio": p.bbox_extent_ratio,
                                       "iou_to_gt": iou(proposal_mask(p), reference)} for p in merged]})
    if not all(row["match"] for row in reproduction):
        print("REPRODUCTION_GATE = FAIL")
        return 2
    counts: dict[str, int] = {}
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
        "starting_head": "ac1e4a9f8f452581c509ab0d73024c44c5de1be0",
        "scientific_reuse_disclosure": "GT access purpose: REFERENCE_FORENSICS_ONLY (native-vector annotations used "
                                      "only to measure whether the frozen proposal set covers the canonical GT "
                                      "reference; no target GT metric, no repair, no candidate replacement, no "
                                      "manual visual judgement)",
        "detector_module": str(EXTERNAL / "buildreasonseg" / "runtime" / "detector.py"),
        "detector_config": {"TILE_SIZE": TILE_SIZE, "TILE_OVERLAP": TILE_OVERLAP, "IMGSZ": IMGSZ, "CONF": CONF,
                            "MAX_DET": MAX_DET, "FROZEN_THRESHOLD": FROZEN_THRESHOLD},
        "detector_call_count": detector_calls, "coverage_threshold": THRESHOLD,
        "proposal_mask_source": "RERUN_GLOBALPROPOSAL_MASK_CROP",
        "proposals_json_role": "METADATA_REPRODUCTION_ONLY",
        "reproduction": reproduction, "records": records, "class_counts": counts, "outcome": outcome,
        "dominant_next_blocker": blocker, "next": nxt}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("REPRODUCTION_GATE = PASS")
    print("CLASS_COUNTS " + json.dumps(counts))
    print("OUTCOME " + json.dumps({"outcome": outcome, "blocker": blocker, "next": nxt}))
    for record in records:
        print("RESULT " + json.dumps({k: v for k, v in record.items() if k != "proposals"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
