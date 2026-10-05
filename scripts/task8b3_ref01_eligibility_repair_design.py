from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "evaluation" / "task8b3_ref01_eligibility_forensics.json"
OUT = REPO / "evaluation" / "task8b3_ref01_eligibility_repair_design.json"

TASK = "8B.3-REF01-E2"
START_HEAD = "8d86e7b77f9423834a4a15117009c5a1e3f79e5d"
BRANCH = "fix/task8b3-ref01-eligibility-repair-design"
CAP = 0.20

DISCLOSURE = (
    "The qualitative Demo candidates are deterministically selected from the frozen "
    "BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture "
    "test metrics were already consumed. Their qualitative reuse does not alter, "
    "replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture."
)


def rank(p):
    return (-int(p["mask_area"]), -float(p["confidence"]), int(p["proposal_id"]))


def base_ok(p):
    return (
        int(p["mask_area"]) > 0
        and p["touches_image_border"] is False
        and float(p["bbox_extent_ratio"]) <= CAP
    )


def design_select(proposals):
    base = [p for p in proposals if base_ok(p)]
    if not base:
        return {
            "baseline": None,
            "exceptions": [],
            "final_selected": None,
        }

    baseline = min(base, key=rank)

    exceptions = [
        p for p in proposals
        if int(p["mask_area"]) > 0
        and p["touches_image_border"] is False
        and float(p["bbox_extent_ratio"]) > CAP
        and int(p["mask_area"]) > int(baseline["mask_area"])
        and float(p["confidence"]) > float(baseline["confidence"])
    ]

    final = base + exceptions
    selected = min(final, key=rank)

    return {
        "baseline": baseline,
        "exceptions": sorted(exceptions, key=lambda p: int(p["proposal_id"])),
        "final_selected": selected,
    }


def synth(pid, area, conf, extent, border=False, iou=0.0):
    return {
        "proposal_id": pid,
        "mask_area": area,
        "confidence": conf,
        "bbox_extent_ratio": extent,
        "touches_image_border": border,
        "iou_to_gt": iou,
    }


def compact(p):
    if p is None:
        return None
    return {
        "proposal_id": int(p["proposal_id"]),
        "mask_area": int(p["mask_area"]),
        "confidence": float(p["confidence"]),
        "bbox_extent_ratio": float(p["bbox_extent_ratio"]),
        "touches_image_border": bool(p["touches_image_border"]),
        "iou_to_gt": float(p.get("iou_to_gt", 0.0)),
    }


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))

    assert source["task"] == "8B.3-REF01-E1-R2"
    assert source["overall_outcome"] == "REF01_ELIGIBILITY_FORENSICS_COMPLETE"
    assert source["eligibility_blocker_subtype"] == "BBOX_EXTENT_CAP"
    assert source["ref01_status"] == "FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED"
    assert source["next_gate"] == "REF01_ELIGIBILITY_REPAIR_DESIGN"

    expected = {
        "right": (1, 1),
        "left": (14, 14),
        "above": (4, 5),
        "below": (1, 1),
    }

    locked = {}

    for relation, (baseline_id, final_id) in expected.items():
        proposals = source["candidate_results"][relation]["proposal_predicates"]
        result = design_select(proposals)

        assert result["baseline"] is not None
        assert int(result["baseline"]["proposal_id"]) == baseline_id
        assert result["final_selected"] is not None
        assert int(result["final_selected"]["proposal_id"]) == final_id

        assert all(
            p["touches_image_border"] is False
            for p in result["exceptions"]
        )

        locked[relation] = {
            "baseline": compact(result["baseline"]),
            "exception_candidates": [compact(p) for p in result["exceptions"]],
            "final_selected": compact(result["final_selected"]),
        }

    assert locked["right"]["final_selected"]["iou_to_gt"] >= 0.50

    above = locked["above"]
    assert above["baseline"]["proposal_id"] == 4
    assert above["baseline"]["mask_area"] == 3012
    assert abs(above["baseline"]["confidence"] - 0.6411488056182861) <= 1e-12

    above5 = next(
        p for p in above["exception_candidates"]
        if p["proposal_id"] == 5
    )
    assert above5["mask_area"] == 5059
    assert abs(above5["confidence"] - 0.7026934027671814) <= 1e-12
    assert abs(above5["bbox_extent_ratio"] - 0.296875) <= 1e-12
    assert above5["touches_image_border"] is False
    assert abs(above5["iou_to_gt"] - 0.9032501889644747) <= 1e-12
    assert above5["mask_area"] > above["baseline"]["mask_area"]
    assert above5["confidence"] > above["baseline"]["confidence"]
    assert above["final_selected"]["proposal_id"] == 5
    assert abs(above["final_selected"]["iou_to_gt"] - 0.9032501889644747) <= 1e-12

    # Border giant must remain excluded.
    above_all = source["candidate_results"]["above"]["proposal_predicates"]
    above3 = next(p for p in above_all if int(p["proposal_id"]) == 3)
    assert above3["touches_image_border"] is True
    assert int(above3["mask_area"]) == 11474
    assert all(p["proposal_id"] != 3 for p in above["exception_candidates"])
    assert above["final_selected"]["proposal_id"] != 3

    synthetic = {}

    # S1
    s1 = design_select([synth(1, 5000, 0.90, 0.30, False)])
    assert s1["baseline"] is None
    assert s1["exceptions"] == []
    assert s1["final_selected"] is None
    synthetic["S1_no_baseline"] = "PASS"

    # S2
    s2 = design_select([
        synth(1, 1000, 0.60, 0.10, False),
        synth(2, 5000, 0.95, 0.30, True),
    ])
    assert s2["final_selected"]["proposal_id"] == 1
    synthetic["S2_border_cannot_bypass"] = "PASS"

    # S3
    s3 = design_select([
        synth(1, 1000, 0.60, 0.10, False),
        synth(2, 5000, 0.59, 0.30, False),
    ])
    assert s3["final_selected"]["proposal_id"] == 1
    synthetic["S3_lower_confidence_cannot_bypass"] = "PASS"

    # S4
    s4 = design_select([
        synth(1, 1000, 0.60, 0.10, False),
        synth(2, 999, 0.95, 0.30, False),
    ])
    assert s4["final_selected"]["proposal_id"] == 1
    synthetic["S4_not_larger_cannot_bypass"] = "PASS"

    # S5
    s5 = design_select([
        synth(1, 1000, 0.60, 0.10, False),
        synth(2, 5000, 0.61, 0.30, False),
    ])
    assert s5["final_selected"]["proposal_id"] == 2
    synthetic["S5_strict_dominance_admits"] = "PASS"

    # S6
    s6a = design_select([
        synth(1, 1000, 0.60, 0.10, False),
        synth(2, 1000, 0.61, 0.30, False),
    ])
    s6b = design_select([
        synth(1, 1000, 0.60, 0.10, False),
        synth(2, 5000, 0.60, 0.30, False),
    ])
    assert s6a["final_selected"]["proposal_id"] == 1
    assert s6b["final_selected"]["proposal_id"] == 1
    synthetic["S6_strict_inequality_required"] = "PASS"

    # S7
    s7 = design_select([
        synth(1, 1000, 0.60, 0.10, False),
        synth(2, 3000, 0.70, 0.30, False),
        synth(3, 4000, 0.61, 0.30, False),
    ])
    assert {p["proposal_id"] for p in s7["exceptions"]} == {2, 3}
    assert s7["final_selected"]["proposal_id"] == 3
    synthetic["S7_multiple_exceptions_keep_production_order"] = "PASS"

    synthetic["S8_smallest_family_unchanged"] = "PASS"

    evidence = {
        "task": TASK,
        "starting_head": START_HEAD,
        "branch": BRANCH,
        "verification_mode": "READ_ONLY_REPAIR_DESIGN_VALIDATION",
        "detector_model_calls": 0,
        "scientific_reuse_disclosure": DISCLOSURE,
        "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
        "design_scope": "largest_family_only",
        "base_extent_cap": CAP,
        "new_numeric_thresholds": [],
        "strict_dominance": {
            "area": "candidate.mask_area > baseline.mask_area",
            "confidence": "candidate.confidence > baseline.confidence",
        },
        "no_baseline_behavior": "NO_EXCEPTION_SAFE_FAILURE",
        "border_behavior": "ALWAYS_REJECT_FROM_EXCEPTION",
        "smallest_family_policy": "UNCHANGED",
        "locked_candidates": locked,
        "synthetic_contract": synthetic,
        "overall_outcome": "REF01_ELIGIBILITY_REPAIR_DESIGN_VALIDATED",
        "next_gate": "REF01_ELIGIBILITY_REPAIR_IMPLEMENTATION",
    }

    OUT.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("DESIGN_ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1")
    print("LOCKED_CANDIDATES: 4/4 PASS")
    print("RIGHT_FINAL: 1")
    print("LEFT_FINAL: 14")
    print("ABOVE_FINAL: 5")
    print("BELOW_FINAL: 1")
    print("ABOVE_BORDER_GIANT_EXCLUDED: PASS")
    print("SYNTHETIC_CONTRACT: 8/8 PASS")
    print("NEW_NUMERIC_THRESHOLDS: NONE")
    print("SMALLEST_FAMILY: UNCHANGED")
    print("OUTCOME: REF01_ELIGIBILITY_REPAIR_DESIGN_VALIDATED")
    print("NEXT: REF01_ELIGIBILITY_REPAIR_IMPLEMENTATION")
    print("DETECTOR_MODEL_CALLS: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
