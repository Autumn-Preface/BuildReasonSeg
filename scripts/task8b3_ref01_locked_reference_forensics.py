"""Task 8B.3-REF01 locked-candidate reference forensics — live-metadata read-only replay verifier (R9).

No detector or model is ever called. The verifier reads the frozen P1D12 proposal metadata straight from the external
diagnostics directories, combines it with the historically measured per-proposal IoU values stored in the canonical
evidence, and replays selected / bestEligible / bestAny with the full frozen tie-break (-IoU, -confidence, proposal_id).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "evaluation" / "task8b3_ref01_locked_reference_forensics.json"
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
DIAG = EXTERNAL / "inference" / "output" / "diagnostics"
THRESHOLD = 0.50


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
    failures = []
    for candidate in evidence["candidates"]:
        relation, tile = candidate["relation"], candidate["tile"]
        live = json.loads((DIAG / tile / "proposals.json").read_text(encoding="utf-8"))
        items = live.get("items") or live.get("proposals") or []
        stored = {int(p["proposal_id"]): float(p["iou_to_gt"])
                  for p in evidence["historical_iou_by_proposal"][relation]}
        for item in items:
            item["_iou"] = stored.get(int(item["proposal_id"]), 0.0)
            item["_eligible"] = (int(item.get("mask_area", 0)) > 0
                                 and item.get("touches_image_border") is False
                                 and float(item.get("bbox_extent_ratio", 1.0)) <= 0.20)
        key = lambda item: (-item["_iou"], -float(item.get("confidence", 0.0)), int(item["proposal_id"]))
        eligible = [i for i in items if i["_eligible"]]
        best_eligible = min(eligible, key=key) if eligible else None
        best_any = min(items, key=key) if items else None
        replay = {
            "relation": relation,
            "best_eligible_id": None if best_eligible is None else int(best_eligible["proposal_id"]),
            "best_eligible_iou": 0.0 if best_eligible is None else best_eligible["_iou"],
            "best_any_id": None if best_any is None else int(best_any["proposal_id"]),
            "best_any_iou": 0.0 if best_any is None else best_any["_iou"],
            "selected_id": candidate["selected"]["proposal_id"],
            "selected_iou": candidate["selected"]["iou"],
        }
        replay["classification"] = classify(replay["selected_iou"], replay["best_eligible_iou"],
                                            replay["best_any_iou"])
        replay["classification_matches_evidence"] = replay["classification"] == candidate["classification"]
        replay["best_eligible_matches_evidence"] = (replay["best_eligible_id"]
                                                    == candidate["best_eligible"]["proposal_id"])
        replay["best_any_matches_evidence"] = replay["best_any_id"] == candidate["best_any"]["proposal_id"]
        print(json.dumps(replay, ensure_ascii=False))
        if not all((replay["classification_matches_evidence"], replay["best_eligible_matches_evidence"],
                    replay["best_any_matches_evidence"])):
            failures.append(relation)
    print("REPLAY_VERIFIER:", "PASS" if not failures else f"FAIL {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
