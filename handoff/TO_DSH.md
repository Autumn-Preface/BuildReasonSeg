# TO_DSH — Task 8B.3-REF01-E1-R2: Exact Eligibility Attribution Replay

> Status: ACTIVE
> Role boundary: ChatGPT is the decision maker. DSH is an executor only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-forensics`
> Required starting HEAD: `1594f1ef96223264999d2c938dd5b2d2bb1b629c`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

---

# 0. EXECUTION CONTRACT — MANDATORY

This task book is intentionally prescriptive.

DSH MUST NOT:
- choose a different algorithm;
- simplify a check;
- replace an enum;
- alter an output schema;
- reinterpret an unexpected condition;
- "fix" a data mismatch;
- infer a missing value;
- choose a different threshold;
- introduce an alternate ranking;
- change the task scope.

If any step below cannot be executed exactly as written, DSH MUST:
1. STOP immediately;
2. record the exact observed fact;
3. update `handoff/FROM_DSH.md`;
4. commit/push the STOP result when Git is safe;
5. wait for ChatGPT.

No autonomous technical decision is permitted.

---

# 1. CHATGPT AUDIT OF E1-R1

E1-R1 is NOT APPROVED.

The following corrections from E1-R1 are accepted:
- production selection now uses `(-mask_area, -confidence, proposal_id)`;
- P0 production IDs are corrected to `1/14/4/1`;
- `above` P2 production-selected proposal is id 5;
- `above` P3 production-selected proposal is id 3;
- tracked replay script now exists;
- detector/model calls remained zero.

However E1-R1 still violates the contract:

```text
R1-D1
The script does not verify detector source identity or frozen selector semantics.

R1-D2
The script does not write the evidence JSON it claims to reproduce; it only prints stdout.

R1-D3
The required proposal-level fail reason is a SINGLE enum:
NONE
ZERO_AREA_ONLY
BORDER_ONLY
EXTENT_ONLY
BORDER_AND_EXTENT
ZERO_AREA_PLUS_OTHER

E1-R1 instead stores lists such as ["BBOX_EXTENT_CAP"].

R1-D4
Proposal fail reason and aggregate blocker subtype are still conflated.

R1-D5
Per-policy `best_coverage` is absent.

R1-D6
The complete per-proposal predicate table is absent.

R1-D7
The exact scientific reuse disclosure is absent.

R1-D8
The required extent relative excess 0.484375 is absent.

R1-D9
The required overall outcome enum is wrong.
Current wrong value:
REF01_ELIGIBILITY_BLOCKER_ISOLATED_BBOX_EXTENT_CAP

Required:
REF01_ELIGIBILITY_FORENSICS_COMPLETE

R1-D10
Required `ref01_status` is absent.
```

R2 fixes only these evidence/replay defects.

No detector/model inference is authorized.

---

# 2. FROZEN SCIENTIFIC RESULT ENTERING R2

Do not change:

```text
right = REFERENCE_SELECTED_CORRECT
left  = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED

PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
```

Frozen coverage threshold:

```text
IoU = 0.50
```

Frozen largest-family eligibility:

```text
mask_area > 0
AND touches_image_border == False
AND bbox_extent_ratio <= 0.20
```

Frozen production largest ranking:

```text
mask_area descending
confidence descending
proposal_id ascending
```

Equivalent Python rank tuple:

```python
(-mask_area, -confidence, proposal_id)
```

GT coverage diagnostic ranking:

```python
(-iou_to_gt, -confidence, proposal_id)
```

GT IoU MUST NEVER participate in production selection.

---

# 3. GIT GATE

Before any work, require:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-forensics

git rev-parse HEAD
= 1594f1ef96223264999d2c938dd5b2d2bb1b629c
```

Allowed initial tracked changes:

```text
clean
OR
only M handoff/TO_DSH.md
```

Any other tracked modification:
- STOP.
- Do not stash/reset/clean.

Forbidden Git actions:
- branch creation;
- checkout to another branch;
- merge;
- rebase;
- reset;
- stash;
- clean;
- cherry-pick;
- force push.

---

# 4. ALLOWED TRACKED PATHS

Only these may change:

```text
scripts/task8b3_ref01_eligibility_forensics.py
evaluation/task8b3_ref01_eligibility_forensics.json
docs/task8b3_ref01_eligibility_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Any other tracked diff:
- STOP before commit.

---

# 5. ABSOLUTE PROHIBITIONS

Do NOT:
- instantiate `DetectorRuntime`;
- call `detect_global`;
- call `model.predict`;
- import/run YOLO for inference;
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- modify detector code/config/weights;
- modify eligibility code;
- modify extent cap `0.20`;
- test a replacement threshold;
- modify canonical RC1;
- modify external RC1;
- run sync write;
- replace locked candidates;
- regenerate datasets/caches;
- visually inspect source images or proposal PNGs;
- implement a product repair;
- execute `REF01_ELIGIBILITY_REPAIR_DESIGN`;
- update `main`.

Detector/model call count must remain:

```text
0
```

---

# 6. EXACT SCIENTIFIC REUSE DISCLOSURE

The final evidence and report MUST contain this English sentence exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

The report must also contain a faithful Chinese translation.

Do not paraphrase the English sentence.

---

# 7. FIXED INPUT PATHS

Repository:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
```

Canonical REF01 evidence:

```text
evaluation\task8b3_ref01_locked_reference_forensics.json
```

External proposal metadata:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\proposals.json
```

External detector source:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

Required detector source SHA256:

```text
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
```

---

# 8. REPLACE THE REPLAY SCRIPT WITH THE EXACT CODE BELOW

DSH MUST replace the entire contents of:

```text
scripts/task8b3_ref01_eligibility_forensics.py
```

with the following code.

Do NOT edit the algorithm.
Do NOT rename keys.
Do NOT replace enums.
Do NOT add fallback logic.

```python
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
```

---

# 9. STATIC GATE BEFORE RUNNING THE SCRIPT

Run:

```text
<project proposal env python> -m py_compile scripts/task8b3_ref01_eligibility_forensics.py
```

Require exit code 0.

Then inspect the source text and require:

```text
"DetectorRuntime(" absent
"detect_global(" absent
"model.predict(" absent
"YOLO(" absent
"production_rank" present
"coverage_rank" present
"EXTENT_ONLY" present
"REF01_ELIGIBILITY_FORENSICS_COMPLETE" present
```

If any static condition fails:
- STOP.
- Do not run the script.

---

# 10. EXECUTE THE SCRIPT EXACTLY ONCE

Run exactly once:

```text
<project proposal env python> scripts/task8b3_ref01_eligibility_forensics.py
```

No retry after a logic/data assertion failure.

Required stdout lines:

```text
SOURCE_IDENTITY: PASS
REF01_INPUT_CONTRACT: PASS
PROPOSAL_ID_JOINS: 4/4 PASS
P0_PRODUCTION_SELECTED: 1/14/4/1 PASS
ABOVE_PROPOSAL5_FAIL_REASON: EXTENT_ONLY
ABOVE_EXTENT_EXCESS_ABSOLUTE: 0.096875
ABOVE_EXTENT_EXCESS_RELATIVE: 0.484375
ABOVE_P0_PRODUCTION_SELECTED: 4 / 0.0
ABOVE_P1_PRODUCTION_SELECTED: 4 / 0.0
ABOVE_P2_PRODUCTION_SELECTED: 5 / 0.9032501889644747
ABOVE_P3_PRODUCTION_SELECTED: 3 / 0.0
ELIGIBILITY_BLOCKER_SUBTYPE: BBOX_EXTENT_CAP
OUTCOME: REF01_ELIGIBILITY_FORENSICS_COMPLETE
NEXT: REF01_ELIGIBILITY_REPAIR_DESIGN
DETECTOR_MODEL_CALLS: 0
```

If script exits non-zero:
- STOP.
- Do not edit data or algorithm.
- Do not rerun.

---

# 11. POST-RUN EVIDENCE ASSERTIONS

Read:

```text
evaluation/task8b3_ref01_eligibility_forensics.json
```

Require exactly:

```text
task = 8B.3-REF01-E1-R2
starting_head = 1594f1ef96223264999d2c938dd5b2d2bb1b629c
branch = fix/task8b3-ref01-eligibility-forensics
verification_mode = READ_ONLY_ELIGIBILITY_COUNTERFACTUAL
detector_model_calls = 0
coverage_threshold = 0.50
frozen_extent_cap = 0.20
eligibility_blocker_subtype = BBOX_EXTENT_CAP
overall_outcome = REF01_ELIGIBILITY_FORENSICS_COMPLETE
ref01_status = FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED
next_gate = REF01_ELIGIBILITY_REPAIR_DESIGN
```

Require exact disclosure equality to §6.

Require:

```text
candidate_results.right.P0_FROZEN.production_selected.proposal_id = 1
candidate_results.left.P0_FROZEN.production_selected.proposal_id = 14
candidate_results.above.P0_FROZEN.production_selected.proposal_id = 4
candidate_results.below.P0_FROZEN.production_selected.proposal_id = 1
```

Require above:

```text
proposal 5 fail_reason = EXTENT_ONLY
extent_excess_absolute = 0.096875
extent_excess_relative = 0.484375
P0 prod selected = 4 / IoU 0.0
P1 prod selected = 4 / IoU 0.0
P2 prod selected = 5 / IoU 0.9032501889644747
P3 prod selected = 3 / IoU 0.0
```

Require every candidate has:
- full `proposal_predicates`;
- all P0/P1/P2/P3;
- both `production_selected`;
- `best_coverage`.

If any key is missing:
- STOP.

---

# 12. REPORT — FIXED CONTENT REQUIREMENTS

Update:

```text
docs/task8b3_ref01_eligibility_forensics.md
```

Do not delete E1/E1-R1 historical sections.

Append:

```text
## E1-R2 authoritative corrected replay
```

The section MUST state:

```text
E1 initial replay was invalid because GT-IoU ranking was used as production selection.

E1-R1 fixed production ranking but still had incomplete evidence schema and fail-reason semantics.

E1-R2 is the authoritative zero-call eligibility-forensics replay.
```

Include table:

```text
relation | policy | eligible | production-selected id/IoU | best-coverage id/IoU
```

Include all 16 relation×policy rows.

Include an `above` subsection with exactly:
- proposal 5 id;
- IoU;
- border flag;
- extent ratio;
- mask area;
- fail reason = `EXTENT_ONLY`;
- absolute extent excess = `0.096875`;
- relative extent excess = `0.484375`;
- P0/P1/P2/P3 production-selected IDs and IoUs.

Required conclusion:

```text
Outcome = REF01_ELIGIBILITY_FORENSICS_COMPLETE
Eligibility blocker subtype = BBOX_EXTENT_CAP
REF-01 status = FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED
NEXT = REF01_ELIGIBILITY_REPAIR_DESIGN
```

Do not execute NEXT.

---

# 13. NONCLAIMS — MUST APPEAR VERBATIM IN REPORT

```text
No new threshold was selected.
No eligibility rule was changed.
No product repair was implemented.
Counterfactual policies are diagnostic only.
P2 success does NOT establish that removing the 0.20 cap globally is safe.
P3 demonstrates that removing both eligibility predicates can expose a larger wrong proposal.
Border exclusion remains frozen.
Task 7J metrics, model, threshold, seed, and architecture are unchanged.
PROP-01 remains PROP01_OPEN_ENGINEERING_DEFECT.
The left and below selection errors remain separate unresolved reference-selection issues.
```

---

# 14. FROM_DSH — REQUIRED FIELDS

Preserve the ARTIFACT-FACTS block exactly.

The active report must contain:

```text
Task: 8B.3-REF01-E1-R2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-forensics
Starting HEAD: 1594f1ef96223264999d2c938dd5b2d2bb1b629c
Detector/model calls: 0
Tracked replay script committed: YES / NO
Script executed exactly once: YES / NO
Source identity: PASS / FAIL
Frozen extent cap: 0.20
Coverage threshold: 0.50_TASK7F_FROZEN
P0 production selected right/left/above/below: 1/14/4/1 / other
above proposal 5 IoU: 0.9032501889644747 / other
above proposal 5 border: False / other
above proposal 5 extent: 0.296875 / other
above proposal 5 mask area: 5059 / other
above proposal 5 fail reason: EXTENT_ONLY / other
above extent excess absolute: 0.096875 / other
above extent excess relative: 0.484375 / other
above P0 production selected: 4 / 0.0 / other
above P1 production selected: 4 / 0.0 / other
above P2 production selected: 5 / 0.9032501889644747 / other
above P3 production selected: 3 / 0.0 / other
GT IoU used in production selection: NO
Eligibility blocker subtype: BBOX_EXTENT_CAP / other
No production threshold change: YES
No product repair: YES
Qwen/SAM2/relation/D-B1/target: NONE/NONE/NONE/NONE/NONE
Manual visual inspection: NO
Candidate replacement: NO
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED / other
Outcome: REF01_ELIGIBILITY_FORENSICS_COMPLETE / other
Next gate: REF01_ELIGIBILITY_REPAIR_DESIGN / other
Evidence: evaluation/task8b3_ref01_eligibility_forensics.json
Report: docs/task8b3_ref01_eligibility_forensics.md
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

---

# 15. FINAL DIFF GATE

Before commit:

```text
git diff --name-only 1594f1ef96223264999d2c938dd5b2d2bb1b629c
```

Allowed names ONLY:

```text
scripts/task8b3_ref01_eligibility_forensics.py
evaluation/task8b3_ref01_eligibility_forensics.json
docs/task8b3_ref01_eligibility_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Any other path:
- STOP.
- Do not commit.

---

# 16. COMMIT / PUSH

If COMPLETE, commit exactly:

```text
docs(rc1): finalize eligibility forensics replay
```

If STOP/FAILED, commit exactly:

```text
docs(rc1): record eligibility forensics replay stop
```

Push only:

```text
fix/task8b3-ref01-eligibility-forensics
```

No force push.

Do not update `main`.

After push:
- STOP;
- wait for ChatGPT.

---

# 17. COMPLETE DEFINITION

COMPLETE only if ALL are true:

- exact branch/start HEAD;
- only five allowed paths changed;
- detector/model calls = 0;
- exact script in §8 is committed;
- script py_compile PASS;
- script executed exactly once;
- detector source SHA PASS;
- frozen eligibility semantics source snippets PASS;
- exact proposal-ID joins 4/4;
- every proposal has exactly one required fail-reason enum;
- P0 production IDs = 1/14/4/1;
- production selection uses no GT IoU;
- per-policy best_coverage exists separately;
- complete 16 policy rows exist;
- above proposal 5 = EXTENT_ONLY;
- extent margins = 0.096875 / 0.484375;
- above P0/P1/P2/P3 production outputs = 4/4/5/3 with IoUs 0/0/0.903250.../0;
- blocker subtype = BBOX_EXTENT_CAP;
- overall outcome = REF01_ELIGIBILITY_FORENSICS_COMPLETE;
- ref01_status = FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED;
- NEXT = REF01_ELIGIBILITY_REPAIR_DESIGN;
- no threshold/rule/repair change;
- evidence/report/FROM_DSH committed and pushed;
- NEXT not executed;
- STOP.
