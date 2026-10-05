from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
DIAG = EXTERNAL / "inference" / "output" / "diagnostics"
REF01 = REPO / "evaluation" / "task8b3_ref01_locked_reference_forensics.json"
OUT = REPO / "evaluation" / "task8b3_ref01_eligibility_forensics.json"
DETECTOR = EXTERNAL / "buildreasonseg" / "runtime" / "detector.py"

TASK = "8B.3-REF01-E1-R2"
STARTING_HEAD = "1594f1ef96223264999d2c938dd5b2d2bb1b629c"
BRANCH = "fix/task8b3-ref01-eligibility-forensics"

THRESHOLD = 0.50
EXTENT_CAP = 0.20
DETECTOR_SHA = "82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738"

DISCLOSURE = (
    "The qualitative Demo candidates are deterministically selected from the frozen "
    "BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture "
    "test metrics were already consumed. Their qualitative reuse does not alter, "
    "replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture."
)

RELATIONS = ("right", "left", "above", "below")

EXPECTED_CLASSES = {
    "right": "REFERENCE_SELECTED_CORRECT",
    "left": "REFERENCE_SELECTION_WRONG_COVERED",
    "above": "REFERENCE_ELIGIBILITY_BLOCKED",
    "below": "REFERENCE_SELECTION_WRONG_COVERED",
}

EXPECTED_P0_SELECTED = {
    "right": 1,
    "left": 14,
    "above": 4,
    "below": 1,
}

POLICIES = {
    "P0_FROZEN": {"require_border": True, "require_extent": True},
    "P1_BORDER_RELAXED_ONLY": {"require_border": False, "require_extent": True},
    "P2_EXTENT_RELAXED_ONLY": {"require_border": True, "require_extent": False},
    "P3_BOTH_RELAXED": {"require_border": False, "require_extent": False},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def production_rank(p: dict):
    return (
        -int(p["mask_area"]),
        -float(p["confidence"]),
        int(p["proposal_id"]),
    )


def coverage_rank(p: dict):
    return (
        -float(p["iou_to_gt"]),
        -float(p["confidence"]),
        int(p["proposal_id"]),
    )


def predicate_state(p: dict) -> dict:
    nonempty = int(p["mask_area"]) > 0
    border = p["touches_image_border"] is False
    extent = float(p["bbox_extent_ratio"]) <= EXTENT_CAP

    if nonempty and border and extent:
        reason = "NONE"
    elif (not nonempty) and border and extent:
        reason = "ZERO_AREA_ONLY"
    elif nonempty and (not border) and extent:
        reason = "BORDER_ONLY"
    elif nonempty and border and (not extent):
        reason = "EXTENT_ONLY"
    elif nonempty and (not border) and (not extent):
        reason = "BORDER_AND_EXTENT"
    else:
        reason = "ZERO_AREA_PLUS_OTHER"

    return {
        "nonempty_pass": nonempty,
        "border_pass": border,
        "extent_pass": extent,
        "frozen_eligible": nonempty and border and extent,
        "fail_reason": reason,
    }


def allowed_by_policy(p: dict, policy: dict) -> bool:
    state = p["_predicate"]
    if not state["nonempty_pass"]:
        return False
    if policy["require_border"] and not state["border_pass"]:
        return False
    if policy["require_extent"] and not state["extent_pass"]:
        return False
    return True


def proposal_record(p: dict) -> dict:
    state = p["_predicate"]
    return {
        "proposal_id": int(p["proposal_id"]),
        "mask_area": int(p["mask_area"]),
        "confidence": float(p["confidence"]),
        "global_bbox": p["global_bbox"],
        "touches_image_border": bool(p["touches_image_border"]),
        "bbox_extent_ratio": float(p["bbox_extent_ratio"]),
        "iou_to_gt": float(p["iou_to_gt"]),
        "nonempty_pass": state["nonempty_pass"],
        "border_pass": state["border_pass"],
        "extent_pass": state["extent_pass"],
        "frozen_eligible": state["frozen_eligible"],
        "fail_reason": state["fail_reason"],
    }


def role_record(p: dict | None) -> dict | None:
    if p is None:
        return None
    return {
        "proposal_id": int(p["proposal_id"]),
        "mask_area": int(p["mask_area"]),
        "confidence": float(p["confidence"]),
        "iou": float(p["iou_to_gt"]),
    }


def main() -> int:
    # Gate A — source identity.
    assert DETECTOR.is_file()
    actual_detector_sha = sha256(DETECTOR)
    assert actual_detector_sha == DETECTOR_SHA

    detector_text = DETECTOR.read_text(encoding="utf-8")
    assert "MERGE_BBOX_EXTENT_RATIO_MAX = 0.20" in detector_text
    assert "if proposal.mask_area <= 0 or not proposal.mask_crop.any():" in detector_text
    assert "if proposal.touches_image_border:" in detector_text
    assert "if proposal.bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX:" in detector_text
    assert "return sorted(candidates, key=lambda proposal: (-proposal.mask_area, -proposal.confidence," in detector_text

    # Gate B — canonical REF01 contract.
    ref = json.loads(REF01.read_text(encoding="utf-8"))
    assert ref["task"] == "8B.3-REF01-F1-R9"
    assert ref["overall_outcome"] == "REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE"
    assert ref["dominant_next_blocker"] == "ELIGIBILITY"
    assert ref["next_gate"] == "REF01_ELIGIBILITY_FORENSICS"

    candidate_by_relation = {c["relation"]: c for c in ref["candidates"]}
    assert set(candidate_by_relation) == set(RELATIONS)

    iou_by_relation = {
        relation: {
            int(row["proposal_id"]): float(row["iou_to_gt"])
            for row in ref["historical_iou_by_proposal"][relation]
        }
        for relation in RELATIONS
    }

    results = {}

    for relation in RELATIONS:
        candidate = candidate_by_relation[relation]
        assert candidate["classification"] == EXPECTED_CLASSES[relation]

        tile = str(candidate["tile"])
        proposal_path = DIAG / tile / "proposals.json"
        assert proposal_path.is_file()

        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        items = payload.get("items") or payload.get("proposals") or []
        assert items

        ids_live = {int(item["proposal_id"]) for item in items}
        ids_iou = set(iou_by_relation[relation])
        assert ids_live == ids_iou
        assert len(ids_live) == len(items)

        prepared = []
        for raw in items:
            for required_key in (
                "proposal_id",
                "confidence",
                "mask_area",
                "global_bbox",
                "touches_image_border",
                "bbox_extent_ratio",
            ):
                assert required_key in raw

            p = dict(raw)
            p["proposal_id"] = int(p["proposal_id"])
            p["confidence"] = float(p["confidence"])
            p["mask_area"] = int(p["mask_area"])
            p["bbox_extent_ratio"] = float(p["bbox_extent_ratio"])
            p["iou_to_gt"] = float(iou_by_relation[relation][p["proposal_id"]])
            p["_predicate"] = predicate_state(p)
            prepared.append(p)

        per_policy = {}

        for policy_name, policy in POLICIES.items():
            allowed = [p for p in prepared if allowed_by_policy(p, policy)]
            assert allowed

            production_selected = min(allowed, key=production_rank)
            best_coverage = min(allowed, key=coverage_rank)

            per_policy[policy_name] = {
                "eligible_count": len(allowed),
                "production_selected": role_record(production_selected),
                "best_coverage": role_record(best_coverage),
            }

        assert per_policy["P0_FROZEN"]["production_selected"]["proposal_id"] == EXPECTED_P0_SELECTED[relation]

        results[relation] = {
            "tile": tile,
            "frozen_reference_class": candidate["classification"],
            "proposal_count": len(prepared),
            "proposal_predicates": [proposal_record(p) for p in sorted(prepared, key=lambda x: x["proposal_id"])],
            "policies": per_policy,
        }

    # Gate C — above blocker isolation.
    above = results["above"]
    proposal5 = next(
        p for p in above["proposal_predicates"]
        if p["proposal_id"] == 5
    )

    assert close(proposal5["iou_to_gt"], 0.9032501889644747)
    assert proposal5["touches_image_border"] is False
    assert close(proposal5["bbox_extent_ratio"], 0.296875)
    assert proposal5["mask_area"] == 5059
    assert proposal5["fail_reason"] == "EXTENT_ONLY"

    extent_excess_absolute = proposal5["bbox_extent_ratio"] - EXTENT_CAP
    extent_excess_relative = proposal5["bbox_extent_ratio"] / EXTENT_CAP - 1.0

    assert close(extent_excess_absolute, 0.096875)
    assert close(extent_excess_relative, 0.484375)

    p0 = above["policies"]["P0_FROZEN"]["production_selected"]
    p1 = above["policies"]["P1_BORDER_RELAXED_ONLY"]["production_selected"]
    p2 = above["policies"]["P2_EXTENT_RELAXED_ONLY"]["production_selected"]
    p3 = above["policies"]["P3_BOTH_RELAXED"]["production_selected"]

    assert p0["proposal_id"] == 4
    assert p1["proposal_id"] == 4
    assert p2["proposal_id"] == 5
    assert p3["proposal_id"] == 3

    assert p0["iou"] < THRESHOLD
    assert p1["iou"] < THRESHOLD
    assert p2["iou"] >= THRESHOLD

    best_any = min(
        above["proposal_predicates"],
        key=lambda p: (
            -float(p["iou_to_gt"]),
            -float(p["confidence"]),
            int(p["proposal_id"]),
        ),
    )

    assert best_any["proposal_id"] == 5
    assert best_any["iou_to_gt"] >= THRESHOLD
    assert best_any["fail_reason"] == "EXTENT_ONLY"

    blocker_subtype = "BBOX_EXTENT_CAP"

    evidence = {
        "task": TASK,
        "starting_head": STARTING_HEAD,
        "branch": BRANCH,
        "verification_mode": "READ_ONLY_ELIGIBILITY_COUNTERFACTUAL",
        "detector_model_calls": 0,
        "scientific_reuse_disclosure": DISCLOSURE,
        "coverage_threshold": THRESHOLD,
        "frozen_extent_cap": EXTENT_CAP,
        "frozen_source_identity": {
            "path": str(DETECTOR),
            "sha256": actual_detector_sha,
            "largest_eligibility": (
                "mask_area>0 AND touches_image_border==False "
                "AND bbox_extent_ratio<=0.20"
            ),
            "largest_selection_order": (
                "mask_area desc, confidence desc, proposal_id asc"
            ),
        },
        "candidate_results": results,
        "above_isolation": {
            "best_any_proposal_id": 5,
            "best_any_iou": 0.9032501889644747,
            "best_any_fail_reason": "EXTENT_ONLY",
            "extent_excess_absolute": extent_excess_absolute,
            "extent_excess_relative": extent_excess_relative,
            "P0_production_selected": p0,
            "P1_production_selected": p1,
            "P2_production_selected": p2,
            "P3_production_selected": p3,
        },
        "eligibility_blocker_subtype": blocker_subtype,
        "overall_outcome": "REF01_ELIGIBILITY_FORENSICS_COMPLETE",
        "ref01_status": "FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED",
        "next_gate": "REF01_ELIGIBILITY_REPAIR_DESIGN",
    }

    OUT.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SOURCE_IDENTITY: PASS")
    print("REF01_INPUT_CONTRACT: PASS")
    print("PROPOSAL_ID_JOINS: 4/4 PASS")
    print("P0_PRODUCTION_SELECTED: 1/14/4/1 PASS")
    print("ABOVE_PROPOSAL5_FAIL_REASON: EXTENT_ONLY")
    print("ABOVE_EXTENT_EXCESS_ABSOLUTE: 0.096875")
    print("ABOVE_EXTENT_EXCESS_RELATIVE: 0.484375")
    print("ABOVE_P0_PRODUCTION_SELECTED: 4 / 0.0")
    print("ABOVE_P1_PRODUCTION_SELECTED: 4 / 0.0")
    print("ABOVE_P2_PRODUCTION_SELECTED: 5 / 0.9032501889644747")
    print("ABOVE_P3_PRODUCTION_SELECTED: 3 / 0.0")
    print("ELIGIBILITY_BLOCKER_SUBTYPE: BBOX_EXTENT_CAP")
    print("OUTCOME: REF01_ELIGIBILITY_FORENSICS_COMPLETE")
    print("NEXT: REF01_ELIGIBILITY_REPAIR_DESIGN")
    print("DETECTOR_MODEL_CALLS: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
