# TO_DSH — Task 8B.3-REF01-E1: Locked Demo Eligibility Forensics

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Base branch: `fix/task8b3-ref01-reference-forensics`
> Required base HEAD: `798a2fcb50054a67b8101581c5007cbbfa46c36a`
> New task branch: `fix/task8b3-ref01-eligibility-forensics`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Frozen audit disposition entering E1

ChatGPT has independently closed the reference-forensics classification.

Formal frozen result:

```text
Task 8B.3-REF01-F1:
APPROVED

REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE

right:
REFERENCE_SELECTED_CORRECT

left:
REFERENCE_SELECTION_WRONG_COVERED

above:
REFERENCE_ELIGIBILITY_BLOCKED

below:
REFERENCE_SELECTION_WRONG_COVERED

Dominant next blocker:
ELIGIBILITY

PROP-01:
PROP01_OPEN_ENGINEERING_DEFECT
```

This E1 task is **diagnostic only**.

It must identify which frozen eligibility predicate blocks the high-IoU `above` reference proposal and quantify the counterfactual effect of relaxing each predicate separately.

It must NOT design, tune, or implement a repair.

# 1. Scientific-use disclosure

The report/evidence MUST preserve the exact disclosure:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

# 2. Git gate

Require:

```text
current branch = fix/task8b3-ref01-reference-forensics
HEAD = 798a2fcb50054a67b8101581c5007cbbfa46c36a
```

Then create exactly:

```text
fix/task8b3-ref01-eligibility-forensics
```

from that HEAD.

No merge/rebase/reset/stash/clean/cherry-pick.

# 3. Absolute prohibitions

Do NOT:
- run detector inference;
- instantiate `DetectorRuntime`;
- call `detect_global`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- modify detector code/config/weights;
- modify `eligible()` / `eligible_proposals()` / `select_reference()`;
- change `MERGE_BBOX_EXTENT_RATIO_MAX = 0.20`;
- propose or test a new production threshold;
- modify canonical RC1 or external RC1;
- sync write;
- replace candidates;
- visually inspect images;
- regenerate datasets/caches;
- update main;
- force push.

Detector/model calls must be exactly:

```text
0
```

# 4. Allowed tracked changes ONLY

```text
scripts/task8b3_ref01_eligibility_forensics.py
evaluation/task8b3_ref01_eligibility_forensics.json
docs/task8b3_ref01_eligibility_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other tracked path may change.

# 5. Frozen inputs

Read-only canonical reference evidence:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Require:

```text
task = 8B.3-REF01-F1-R9
overall_outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
dominant_next_blocker = ELIGIBILITY
next_gate = REF01_ELIGIBILITY_FORENSICS
```

Read-only external proposal metadata:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\proposals.json
...\1003\proposals.json
...\1008\proposals.json
...\1009\proposals.json
```

Read-only frozen detector source:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

Require source identity against manifest:

```text
sha256 =
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
```

# 6. Frozen eligibility semantics

Do not reimplement from memory until source assertions pass.

From external `detector.py`, require:

```text
MERGE_BBOX_EXTENT_RATIO_MAX = 0.20
```

For `family="largest"`, frozen eligibility is exactly:

```text
mask_area > 0
AND touches_image_border == False
AND bbox_extent_ratio <= 0.20
```

Frozen production reference ordering among eligible proposals:

```text
mask_area descending
confidence descending
proposal_id ascending
```

No smallest-family logic is relevant.

# 7. Four diagnostic policies

These are **forensic counterfactuals only**, not production proposals.

Define exactly:

```text
P0_FROZEN:
  mask_area > 0
  AND not border
  AND extent <= 0.20

P1_BORDER_RELAXED_ONLY:
  mask_area > 0
  AND extent <= 0.20

P2_EXTENT_RELAXED_ONLY:
  mask_area > 0
  AND not border

P3_BOTH_RELAXED:
  mask_area > 0
```

For every policy, production-largest ordering remains:

```text
(-mask_area, -confidence, proposal_id)
```

Do not invent any alternate threshold.

# 8. Per-proposal forensic attribution

For every merged proposal in all four locked candidates, join:
- live P1D12 metadata;
- historical `iou_to_gt` from canonical REF01 evidence.

Require exact proposal-ID-set equality.

For each proposal compute frozen predicate state:

```text
nonempty_pass
border_pass
extent_pass
frozen_eligible
```

Compute fail reason exactly one of:

```text
NONE
ZERO_AREA_ONLY
BORDER_ONLY
EXTENT_ONLY
BORDER_AND_EXTENT
ZERO_AREA_PLUS_OTHER
```

Record at minimum:

```text
proposal_id
mask_area
confidence
global_bbox
touches_image_border
bbox_extent_ratio
iou_to_gt
fail_reason
```

# 9. Per-policy replay

For each relation × each policy P0/P1/P2/P3 compute:

```text
eligible_count
selected_id
selected_mask_area
selected_confidence
selected_iou
best_coverage_id
best_coverage_iou
```

`selected_id`:
production largest ordering.

`best_coverage_id`:
rank by:

```text
(-iou_to_gt, -confidence, proposal_id)
```

No GT-based value is used for production selection.

# 10. Required above-case forensic facts

For `above / tile 1008`, canonical historical facts already establish:

```text
frozen selected_id = 4
frozen selected_iou = 0.0

best_any_id = 5
best_any_iou = 0.9032501889644747
```

E1 must independently establish from live P1D12 metadata:

```text
proposal 5:
touches_image_border = False
bbox_extent_ratio = 0.296875
mask_area = 5059
```

Then mechanically classify proposal 5's frozen fail reason.

Expected only if the data confirm it:

```text
EXTENT_ONLY
```

Do not hard-code PASS if live data disagree.

Record extent-cap excess:

```text
extent_excess_absolute =
bbox_extent_ratio - 0.20

extent_excess_relative =
(bbox_extent_ratio / 0.20) - 1
```

For proposal 5 this is expected to be:

```text
0.096875
0.484375
```

These are diagnostic margins only.

They are NOT a proposed new threshold.

# 11. Isolation test for the eligibility blocker

For `above`, compare the four policies.

Mechanically determine:

```text
P0 frozen selected IoU
P1 border-relaxed-only selected IoU
P2 extent-relaxed-only selected IoU
P3 both-relaxed selected IoU
```

Define blocker subtype:

```text
BBOX_EXTENT_CAP
if:
  frozen selected_iou < 0.50
  AND best_any_iou >= 0.50
  AND best_any fail_reason == EXTENT_ONLY
  AND P2 selected_iou >= 0.50
  AND P1 selected_iou < 0.50

BORDER_EXCLUSION
if analogous BORDER_ONLY pattern

COMBINED_ELIGIBILITY
if neither isolated predicate alone restores selected_iou >= 0.50
but P3 does

ELIGIBILITY_FORENSICS_INCONSISTENT
otherwise
```

If subtype is `ELIGIBILITY_FORENSICS_INCONSISTENT`, status must be STOP.

# 12. Cross-candidate guard

For right / left / below, report the same P0/P1/P2/P3 table.

This is NOT a repair evaluation.

Purpose:
- detect whether relaxing border or extent would expose obviously larger competing proposals;
- quantify whether the above-case phenomenon is isolated or shared.

Do not interpret a counterfactual policy as recommended.

Record for every relation:

```text
frozen_class
P0 selected_id/iou
P1 selected_id/iou
P2 selected_id/iou
P3 selected_id/iou
```

# 13. Diagnostic conclusion contract

If the `above` data satisfy §11 `BBOX_EXTENT_CAP`, final E1 outcome must be:

```text
Outcome =
REF01_ELIGIBILITY_FORENSICS_COMPLETE

Eligibility blocker subtype =
BBOX_EXTENT_CAP

Primary evidence =
above best-any proposal 5 is non-border,
IoU >= 0.50,
excluded only by bbox_extent_ratio > 0.20,
and extent-only relaxation restores a covered production-largest selection.

REF-01 status =
FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED

NEXT =
REF01_ELIGIBILITY_REPAIR_DESIGN
```

If another subtype is mechanically established, use the corresponding subtype and:

```text
NEXT = REF01_ELIGIBILITY_REPAIR_DESIGN
```

Do NOT execute NEXT.

# 14. Nonclaims

Report must explicitly state:

```text
No new threshold was selected.
No eligibility rule was changed.
No product repair was implemented.
Counterfactual policies are diagnostic only.
This task does not establish that removing the 0.20 cap globally is safe.
This task does not establish that border exclusion should be removed.
This task does not alter Task 7J metrics/model/seed/architecture.
PROP-01 remains PROP01_OPEN_ENGINEERING_DEFECT.
left and below selection errors remain separate REF-01 selection issues.
```

# 15. Script

Create:

```text
scripts/task8b3_ref01_eligibility_forensics.py
```

Requirements:
- read-only inputs only;
- no detector/model imports needed except source-text/identity inspection;
- zero inference;
- deterministic;
- writes only:
  `evaluation/task8b3_ref01_eligibility_forensics.json`.

# 16. Evidence JSON

Create:

```text
evaluation/task8b3_ref01_eligibility_forensics.json
```

Required top-level:

```text
task = 8B.3-REF01-E1
starting_head
branch
verification_mode = READ_ONLY_ELIGIBILITY_COUNTERFACTUAL
detector_model_calls = 0
coverage_threshold = 0.50
frozen_extent_cap = 0.20
scientific_reuse_disclosure
candidate_results
above_isolation
eligibility_blocker_subtype
overall_outcome
next_gate
```

For each candidate include:
- relation/tile;
- frozen class;
- proposal predicate table;
- P0/P1/P2/P3 results.

# 17. Report

Create:

```text
docs/task8b3_ref01_eligibility_forensics.md
```

Required sections:
1. scope / prohibitions;
2. frozen eligibility rule;
3. per-proposal fail-reason summary;
4. P0/P1/P2/P3 counterfactual table;
5. above blocker isolation;
6. cross-candidate guard;
7. conclusion / nonclaims;
8. exact scientific reuse disclosure.

Required compact table:

```text
relation | P0 sel/IoU | P1 sel/IoU | P2 sel/IoU | P3 sel/IoU | GT-best fail reason
```

# 18. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-E1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-forensics
Starting HEAD: 798a2fcb50054a67b8101581c5007cbbfa46c36a
Detector/model calls: 0
Frozen extent cap: 0.20
Coverage threshold: 0.50_TASK7F_FROZEN
Proposal metadata source: EXISTING_P1D12_JSON_ONLY
Historical IoU source: CANONICAL_REF01_EVIDENCE
above best-any id: <id>
above best-any IoU: <value>
above best-any border flag: <value>
above best-any extent ratio: <value>
above best-any fail reason: <enum>
above P0 selected/id IoU: <id> / <iou>
above P1 selected/id IoU: <id> / <iou>
above P2 selected/id IoU: <id> / <iou>
above P3 selected/id IoU: <id> / <iou>
Eligibility blocker subtype: <enum>
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

# 19. Commit / push

If COMPLETE:

```text
docs(rc1): isolate reference eligibility blocker
```

If STOP/FAILED:

```text
docs(rc1): record eligibility forensics stop
```

Push only:

```text
fix/task8b3-ref01-eligibility-forensics
```

No force push.
Do not update main.

Then STOP.

# 20. COMPLETE definition

COMPLETE only if:
- exact base branch/head;
- new task branch created exactly as specified;
- only five allowed tracked paths changed;
- detector/model calls = 0;
- frozen eligibility semantics verified from source;
- proposal-ID sets join exactly;
- fail reason assigned mechanically to every proposal;
- P0/P1/P2/P3 replay complete for all four;
- above best-any blocker isolated mechanically;
- no production threshold/rule change;
- no product repair;
- evidence/report/FROM_DSH committed and pushed;
- NEXT not executed;
- STOP.
