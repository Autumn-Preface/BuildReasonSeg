# TO_DSH — Task 8B.3-REF01-E2: Freeze Largest Extent-Dominance Repair Design

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required base branch: `fix/task8b3-ref01-eligibility-forensics`
> Required base HEAD: `8d86e7b77f9423834a4a15117009c5a1e3f79e5d`
> New task branch: `fix/task8b3-ref01-eligibility-repair-design`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. EXECUTOR CONTRACT

DSH has NO authority to choose or alter the repair design.

The repair design is already decided in this task book.

DSH may only:
1. reproduce the declared inputs;
2. execute the exact fixed validation script in this task book;
3. record its outputs;
4. commit/push the corresponding COMPLETE or STOP result.

DSH MUST NOT:
- invent an alternative rule;
- compare several repair designs and select one;
- tune any threshold;
- add a confidence threshold;
- add a new size threshold;
- change strict `>` to `>=`;
- change the baseline definition;
- change the family scope;
- modify product code;
- interpret a failed assertion and repair it.

Any unexpected result or failed assertion => STOP.

# 1. CHATGPT DECISION

The simple policy `P2_EXTENT_RELAXED_ONLY` is NOT selected as the production repair.

Reason:
- it fixes `above`;
- but it admits every non-border extent-violating proposal;
- this changes `left` production selection from id 14 to id 27;
- global safety is not established.

The selected repair design candidate is fixed as:

```text
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
```

This is a parameter-free, largest-family-only exception.

# 2. EXACT DESIGN

Let:

```text
base_candidates =
current frozen largest-family candidates:
mask_area > 0
AND mask_crop non-empty
AND touches_image_border == False
AND bbox_extent_ratio <= 0.20
```

Let:

```text
baseline =
production-largest selection over base_candidates:
mask_area descending
confidence descending
proposal_id ascending
```

If `base_candidates` is empty:

```text
final_candidates = []
selected = None
```

No exception is allowed when no frozen baseline exists.

If a baseline exists, an otherwise extent-blocked proposal `p` is an exception candidate if and only if ALL are true:

```text
p.mask_area > 0
p.mask_crop / metadata represents a non-empty proposal
p.touches_image_border == False
p.bbox_extent_ratio > 0.20
p.mask_area > baseline.mask_area
p.confidence > baseline.confidence
```

Note the strict operators:

```text
mask_area > baseline.mask_area
confidence > baseline.confidence
```

They MUST NOT be changed to `>=`.

Final largest-family candidates:

```text
final_candidates =
base_candidates
UNION
dominance_exception_candidates
```

Final selection:

```text
mask_area descending
confidence descending
proposal_id ascending
```

# 3. SCOPE

This design applies ONLY to:

```text
family = "largest"
```

It does NOT change:
- `family="smallest"`;
- border exclusion;
- non-empty requirement;
- the frozen 0.20 cap as the ordinary/base rule;
- detector threshold/confidence;
- proposal generation;
- merge;
- model weights.

This task is DESIGN VALIDATION ONLY.

No product implementation is allowed.

# 4. SCIENTIFIC / ENGINEERING NONCLAIM

Preserve exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

This repair design is an RC1 engineering rule. It is not a new scientific model-selection result.

# 5. GIT GATE

Require:

```text
branch = fix/task8b3-ref01-eligibility-forensics
HEAD = 8d86e7b77f9423834a4a15117009c5a1e3f79e5d
```

Then create exactly:

```text
fix/task8b3-ref01-eligibility-repair-design
```

No other branch name.

No merge/rebase/reset/stash/clean/cherry-pick.

# 6. ALLOWED TRACKED CHANGES ONLY

```text
scripts/task8b3_ref01_eligibility_repair_design.py
evaluation/task8b3_ref01_eligibility_repair_design.json
docs/task8b3_ref01_eligibility_repair_design.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No product source file may change.

# 7. ABSOLUTE PROHIBITIONS

Do NOT:
- modify `buildreasonseg/runtime/detector.py`;
- modify canonical/external RC1;
- run sync write;
- instantiate DetectorRuntime;
- run detector/model inference;
- run Qwen/SAM2/relation/D-B1/target segmentation;
- inspect source images visually;
- replace candidates;
- test a new numeric threshold;
- implement the repair;
- execute NEXT;
- update main;
- force push.

Detector/model calls = 0.

# 8. INPUT EVIDENCE

Use read-only:

```text
evaluation/task8b3_ref01_eligibility_forensics.json
```

Require:

```text
task = 8B.3-REF01-E1-R2
overall_outcome = REF01_ELIGIBILITY_FORENSICS_COMPLETE
eligibility_blocker_subtype = BBOX_EXTENT_CAP
ref01_status = FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED
next_gate = REF01_ELIGIBILITY_REPAIR_DESIGN
```

Use `candidate_results.*.proposal_predicates`.

No detector run.

# 9. LOCKED-CANDIDATE DESIGN GATES

Apply the exact design from §2 to the four candidate proposal tables.

The following are SUCCESS requirements.

## right

Require:

```text
baseline selected id = 1
final selected id = 1
final selected IoU >= 0.50
```

No regression.

## left

Require:

```text
baseline selected id = 14
final selected id = 14
```

The low-confidence extent-violating proposal that caused the unconditional P2 selection change must NOT replace baseline.

If final selected id is not 14:
- STOP.

## above

Require:

```text
baseline selected id = 4
final selected id = 5
final selected IoU = 0.9032501889644747
```

Require proposal 5:

```text
mask_area = 5059
confidence = 0.7026934027671814
bbox_extent_ratio = 0.296875
touches_image_border = False
```

Require baseline proposal 4:

```text
mask_area = 3012
confidence = 0.6411488056182861
```

Therefore require both:

```text
5059 > 3012
0.7026934027671814 > 0.6411488056182861
```

## below

Require:

```text
baseline selected id = 1
final selected id = 1
```

# 10. BORDER SAFETY GATE

For every candidate:

Any proposal with:

```text
touches_image_border == True
```

MUST NOT appear in:

```text
dominance_exception_candidates
```

In particular `above` proposal 3:

```text
proposal_id = 3
mask_area = 11474
touches_image_border = True
```

MUST remain excluded.

Require:

```text
above final selected != 3
```

# 11. SYNTHETIC CONTRACT CASES

The fixed validation script must also validate these exact synthetic cases.

## S1 — no baseline

Input:
- only one non-border, extent>0.20 proposal;
- area 5000;
- confidence 0.90.

Expected:

```text
baseline = None
exception_candidates = []
selected = None
```

## S2 — border candidate cannot bypass

Baseline:
```text
area=1000 confidence=0.60 extent=0.10 border=False
```

Large candidate:
```text
area=5000 confidence=0.95 extent=0.30 border=True
```

Expected selected = baseline.

## S3 — larger but lower confidence cannot bypass

Baseline:
```text
area=1000 confidence=0.60
```

Large candidate:
```text
area=5000 confidence=0.59 extent=0.30 border=False
```

Expected selected = baseline.

## S4 — higher confidence but not larger cannot bypass

Baseline:
```text
area=1000 confidence=0.60
```

Candidate:
```text
area=999 confidence=0.95 extent=0.30 border=False
```

Expected selected = baseline.

## S5 — strict dominance admits

Baseline:
```text
area=1000 confidence=0.60
```

Candidate:
```text
area=5000 confidence=0.61 extent=0.30 border=False
```

Expected selected = candidate.

## S6 — equality is not enough

Two subcases:

```text
area=1000 confidence=0.61
```
against baseline area 1000 => not admitted.

```text
area=5000 confidence=0.60
```
against baseline confidence 0.60 => not admitted.

## S7 — multiple admitted exceptions retain production ordering

Baseline:
```text
id=1 area=1000 confidence=0.60
```

Exception A:
```text
id=2 area=3000 confidence=0.70
```

Exception B:
```text
id=3 area=4000 confidence=0.61
```

Both admitted.

Expected final selected:

```text
id=3
```

because area remains the primary production ranking key.

## S8 — smallest family unchanged

The design validator must report:

```text
smallest_family_policy = UNCHANGED
```

No simulation that alters smallest semantics is permitted.

# 12. FIXED VALIDATION SCRIPT

Create exactly:

```text
scripts/task8b3_ref01_eligibility_repair_design.py
```

with this code:

```python
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
```

# 13. STATIC GATE

Compile:

```text
<proposal env python> -m py_compile scripts/task8b3_ref01_eligibility_repair_design.py
```

Require exit 0.

Source must NOT contain:

```text
DetectorRuntime(
detect_global(
model.predict(
YOLO(
```

If fail => STOP.

# 14. EXECUTE EXACTLY ONCE

Run:

```text
<proposal env python> scripts/task8b3_ref01_eligibility_repair_design.py
```

No retry after assertion failure.

Required stdout:

```text
DESIGN_ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
LOCKED_CANDIDATES: 4/4 PASS
RIGHT_FINAL: 1
LEFT_FINAL: 14
ABOVE_FINAL: 5
BELOW_FINAL: 1
ABOVE_BORDER_GIANT_EXCLUDED: PASS
SYNTHETIC_CONTRACT: 8/8 PASS
NEW_NUMERIC_THRESHOLDS: NONE
SMALLEST_FAMILY: UNCHANGED
OUTCOME: REF01_ELIGIBILITY_REPAIR_DESIGN_VALIDATED
NEXT: REF01_ELIGIBILITY_REPAIR_IMPLEMENTATION
DETECTOR_MODEL_CALLS: 0
```

Any mismatch => STOP.

# 15. REPORT

Create:

```text
docs/task8b3_ref01_eligibility_repair_design.md
```

The report must state that ChatGPT preselected the design and DSH did not choose among alternatives.

Required sections:
1. frozen E1-R2 evidence;
2. selected design;
3. exact decision rule;
4. four locked-candidate results;
5. eight synthetic contract results;
6. risks/nonclaims;
7. next gate.

Required nonclaims:

```text
This task does not implement the repair.
This task does not select a new numeric threshold.
This task does not prove global generalization.
The four locked candidates are engineering evidence only.
The border exclusion remains mandatory.
The smallest-family behavior remains unchanged.
The left and below reference-selection defects remain unresolved.
PROP-01 remains PROP01_OPEN_ENGINEERING_DEFECT.
```

# 16. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required fields:

```text
Task: 8B.3-REF01-E2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-design
Starting HEAD: 8d86e7b77f9423834a4a15117009c5a1e3f79e5d
Detector/model calls: 0
Design selected by: CHATGPT
DSH design choice performed: NO
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
New numeric threshold selected: NO
Right baseline/final: 1/1
Left baseline/final: 14/14
Above baseline/final: 4/5
Below baseline/final: 1/1
Above proposal 3 border giant excluded: YES
Synthetic contract: 8/8 PASS
Smallest family: UNCHANGED
Outcome: REF01_ELIGIBILITY_REPAIR_DESIGN_VALIDATED / other
Next gate: REF01_ELIGIBILITY_REPAIR_IMPLEMENTATION / other
External/canonical product files modified: NO / NO
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 17. STATUS → COMMIT MESSAGE CONSISTENCY GATE

This gate exists because E1-R2 had a COMPLETE/commit-message mismatch.

Before commit, determine status mechanically:

```text
If:
script exit = 0
AND all required stdout matches
AND evidence assertions pass
AND final diff contains only allowed paths

then:
STATUS = COMPLETE
COMMIT_MESSAGE = docs(rc1): freeze eligibility repair design
```

Otherwise:

```text
STATUS = STOP
COMMIT_MESSAGE = docs(rc1): record eligibility repair design stop
```

Write both values into `handoff/FROM_DSH.md`.

Then assert:

```text
STATUS == COMPLETE
=> commit message MUST equal:
docs(rc1): freeze eligibility repair design

STATUS == STOP
=> commit message MUST equal:
docs(rc1): record eligibility repair design stop
```

Do not use the wrong message for the status.

# 18. FINAL DIFF GATE

Allowed only:

```text
scripts/task8b3_ref01_eligibility_repair_design.py
evaluation/task8b3_ref01_eligibility_repair_design.json
docs/task8b3_ref01_eligibility_repair_design.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Any other path => STOP.

# 19. COMMIT / PUSH

Use the mechanically selected message from §17.

Push only:

```text
fix/task8b3-ref01-eligibility-repair-design
```

No force push.
Do not update main.

Then STOP and wait for ChatGPT.

# 20. COMPLETE DEFINITION

COMPLETE only if:
- exact base head and branch creation;
- only five allowed paths changed;
- detector/model calls 0;
- exact fixed validator used;
- design was not altered by DSH;
- locked candidates 4/4 pass;
- right 1→1;
- left 14→14;
- above 4→5;
- below 1→1;
- above border giant excluded;
- synthetic contract 8/8 pass;
- no new numeric threshold;
- smallest unchanged;
- no product code changed;
- status/commit-message consistency gate passes;
- evidence/report/FROM_DSH committed/pushed;
- NEXT not executed;
- STOP.
