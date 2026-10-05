"""Task 8B.3-REF01-E1 frozen-eligibility replay (tracked, read-only, zero model calls).

Replays the frozen eligibility policies P0/P1/P2/P3 over the existing P1D12 proposal metadata. Production selection
uses the frozen production ranking (-mask_area, -confidence, proposal_id) over each policy's eligible set; GT coverage
is reported separately as the maximum IoU-to-GT proposal with its predicate fail reasons and extent margin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
DIAG = EXTERNAL / "inference" / "output" / "diagnostics"
REF01 = REPO / "evaluation" / "task8b3_ref01_locked_reference_forensics.json"
THRESHOLD = 0.50
EXTENT_CAP = 0.20
RELATIONS = ("right", "left", "above", "below")
POLICIES = {"P0_FROZEN": (True, True), "P1_BORDER_RELAXED_ONLY": (False, True),
            "P2_EXTENT_RELAXED_ONLY": (True, False), "P3_BOTH_RELAXED": (False, False)}


def production_rank(item):
    return (-int(item.get("mask_area", 0)), -float(item.get("confidence", 0.0)), int(item["proposal_id"]))


def gt_rank(item):
    return (-item["_iou"], -float(item.get("confidence", 0.0)), int(item["proposal_id"]))


def fail_reasons(item) -> list[str]:
    reasons = []
    if int(item.get("mask_area", 0)) <= 0:
        reasons.append("MASK_AREA_ZERO")
    if item.get("touches_image_border") is not False:
        reasons.append("TOUCHES_IMAGE_BORDER")
    if float(item.get("bbox_extent_ratio", 1.0)) > EXTENT_CAP:
        reasons.append("BBOX_EXTENT_CAP")
    return reasons


def main() -> int:
    ref01 = json.loads(REF01.read_text(encoding="utf-8"))
    stored = {relation: {int(p["proposal_id"]): float(p["iou_to_gt"])
                         for p in ref01["historical_iou_by_proposal"][relation]} for relation in RELATIONS}
    candidates = {c["relation"]: c for c in ref01["candidates"]}
    out = {}
    for relation in RELATIONS:
        tile = candidates[relation]["tile"]
        live = json.loads((DIAG / tile / "proposals.json").read_text(encoding="utf-8"))
        items = live.get("items") or live.get("proposals") or []
        for item in items:
            item["_iou"] = stored[relation].get(int(item["proposal_id"]), 0.0)
            item["_reasons"] = fail_reasons(item)
        policies = {}
        for policy, (use_border, use_extent) in POLICIES.items():
            eligible = [i for i in items
                        if "MASK_AREA_ZERO" not in i["_reasons"]
                        and (not use_border or "TOUCHES_IMAGE_BORDER" not in i["_reasons"])
                        and (not use_extent or "BBOX_EXTENT_CAP" not in i["_reasons"])]
            selected = min(eligible, key=production_rank) if eligible else None
            policies[policy] = {
                "eligible_count": len(eligible),
                "selected_id": None if selected is None else int(selected["proposal_id"]),
                "selected_mask_area": None if selected is None else int(selected["mask_area"]),
                "selected_confidence": None if selected is None else selected.get("confidence"),
                "selected_iou": 0.0 if selected is None else selected["_iou"]}
        gt_best = min(items, key=gt_rank) if items else None
        out[relation] = {"tile": tile, "proposal_count": len(items), "policies": policies,
                         "gt_best": {"proposal_id": None if gt_best is None else int(gt_best["proposal_id"]),
                                     "iou": 0.0 if gt_best is None else gt_best["_iou"],
                                     "mask_area": None if gt_best is None else gt_best.get("mask_area"),
                                     "confidence": None if gt_best is None else gt_best.get("confidence"),
                                     "border": None if gt_best is None else gt_best.get("touches_image_border"),
                                     "extent_ratio": None if gt_best is None
                                     else float(gt_best.get("bbox_extent_ratio", 0.0)),
                                     "extent_margin": None if gt_best is None
                                     else round(float(gt_best.get("bbox_extent_ratio", 0.0)) - EXTENT_CAP, 12),
                                     "fail_reasons": [] if gt_best is None else gt_best["_reasons"]},
                         "classification": candidates[relation]["classification"]}
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
