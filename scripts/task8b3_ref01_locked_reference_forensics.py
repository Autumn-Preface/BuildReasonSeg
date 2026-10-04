"""Task 8B.3-REF01 locked-candidate reference forensics — read-only replay verifier (R8).

This script performs NO detector or model call. It replays the frozen R4/R5/R6 measurements from the consolidated
canonical evidence file, re-derives the mechanical classification from the stored per-proposal IoU values, and asserts
that the recorded class, the selected/best-eligible/best-any proposal IDs and the R4=R5=R6 agreement all hold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "evaluation" / "task8b3_ref01_locked_reference_forensics.json"
THRESHOLD = 0.50
TOLERANCE = 1e-6
RELATIONS = ("right", "left", "above", "below")


def classify(selected_iou: float, best_eligible_iou: float, best_any_iou: float) -> str:
    if selected_iou >= THRESHOLD:
        return "REFERENCE_SELECTED_CORRECT"
    if best_eligible_iou >= THRESHOLD:
        return "REFERENCE_SELECTION_WRONG_COVERED"
    if best_any_iou >= THRESHOLD:
        return "REFERENCE_ELIGIBILITY_BLOCKED"
    return "REFERENCE_COVERAGE_MISSING"


def main() -> int:
    evidence = json.loads(CANONICAL.read_text(encoding="utf-8"))
    records = {row["relation"]: row for row in evidence["r4_records"]}
    failures = []
    for relation in RELATIONS:
        record = records[relation]
        proposals = record["proposals"]
        eligible = [p for p in proposals
                    if p["mask_area"] > 0 and p.get("touches_border", p.get("touches_image_border")) is False and p["bbox_extent_ratio"] <= 0.20]
        selected_id = record["selected_id"]
        best_eligible = max(eligible, key=lambda p: p["iou_to_gt"]) if eligible else None
        best_any = max(proposals, key=lambda p: p["iou_to_gt"]) if proposals else None
        replay = {
            "relation": relation,
            "selected_id": selected_id,
            "selected_iou": record["selected_iou"],
            "best_eligible_id": None if best_eligible is None else best_eligible["proposal_id"],
            "best_eligible_iou": 0.0 if best_eligible is None else best_eligible["iou_to_gt"],
            "best_any_id": None if best_any is None else best_any["proposal_id"],
            "best_any_iou": 0.0 if best_any is None else best_any["iou_to_gt"],
        }
        replay["classification"] = classify(replay["selected_iou"], replay["best_eligible_iou"],
                                            replay["best_any_iou"])
        replay["class_reproduced"] = replay["classification"] == record["classification"]
        replay["iou_within_tolerance"] = (abs(replay["selected_iou"] - record["selected_iou"]) <= TOLERANCE
                                          and abs(replay["best_eligible_iou"] - record["best_eligible_iou"])
                                          <= TOLERANCE
                                          and abs(replay["best_any_iou"] - record["best_any_iou"]) <= TOLERANCE)
        print(json.dumps(replay, ensure_ascii=False))
        if not (replay["class_reproduced"] and replay["iou_within_tolerance"]):
            failures.append(relation)
    print("REPLAY_VERIFIER:", "PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
