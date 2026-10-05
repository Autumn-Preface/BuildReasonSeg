# TO_DSH — Task 8B.3-REF01-E1-R1: Correct Eligibility Counterfactual Replay

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-forensics`
> Required starting HEAD: `3c497bf3d427366e00b5890fdcfab1a35e68b28c`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. ChatGPT audit disposition

Task 8B.3-REF01-E1 is NOT approved.

The E1 attempt is retained as historical evidence but its policy replay is invalid.

Confirmed defects:

```text
D1. Policy `selected` was ranked by:
    (-IoU, -confidence, proposal_id)

    This is GT-best diagnostic ranking, NOT frozen production selection.

D2. Production policy selection must always be:
    (-mask_area, -confidence, proposal_id)

D3. The required tracked script
    scripts/task8b3_ref01_eligibility_forensics.py
    was not committed.

D4. Proposal fail-reason enum was conflated with blocker subtype:
    proposal fail reason must be EXTENT_ONLY / BORDER_ONLY / ...
    blocker subtype may be BBOX_EXTENT_CAP.

D5. E1 used the wrong overall outcome enum.

D6. Required extent-excess diagnostics and complete per-proposal predicate table
    were not present.

D7. Required exact scientific-reuse disclosure was not preserved in the evidence.
```

No detector/model rerun is authorized.

# 1. Frozen prior REF01 result

Do not alter:

```text
right = REFERENCE_SELECTED_CORRECT
left  = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED

REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY

PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
```

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-eligibility-forensics
HEAD = 3c497bf3d427366e00b5890fdcfab1a35e68b28c
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No new branch.
No merge/rebase/reset/stash/clean/cherry-pick.

# 3. Strict prohibitions

Do NOT:
- instantiate DetectorRuntime;
- call detect_global;
- run any detector/model inference;
- run predict.py;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- alter detector source/config/weights;
- change eligibility rules;
- change extent cap 0.20;
- select a replacement threshold;
- modify external or canonical RC1;
- run sync write;
- replace candidates;
- visually inspect images;
- regenerate datasets/caches;
- implement a repair;
- update main;
- force push.

Detector/model calls this task = exactly 0.

# 4. Allowed tracked changes ONLY

```text
scripts/task8b3_ref01_eligibility_forensics.py
evaluation/task8b3_ref01_eligibility_forensics.json
docs/task8b3_ref01_eligibility_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other tracked path may change.

# 5. Exact scientific reuse disclosure

Evidence and report MUST contain exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Also include a faithful Chinese translation in the report.

# 6. Frozen source semantics gate

Read external detector source only:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

Require external source identity:

```text
sha256 =
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
```

Require source constants/semantics:

```text
MERGE_BBOX_EXTENT_RATIO_MAX = 0.20

largest eligibility:
mask_area > 0
AND touches_image_border == False
AND bbox_extent_ratio <= 0.20

production largest ordering:
mask_area descending
confidence descending
proposal_id ascending
```

Store source identity and frozen rule in evidence.

# 7. Inputs

Read only:

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

Read external existing P1D12 metadata only:

```text
...\diagnostics\1010\proposals.json
...\diagnostics\1003\proposals.json
...\diagnostics\1008\proposals.json
...\diagnostics\1009\proposals.json
```

Join each live proposal to:

```text
historical_iou_by_proposal[relation]
```

by exact `proposal_id`.

Require proposal ID set equality for each candidate.

# 8. Per-proposal frozen predicate attribution

For EVERY merged proposal compute:

```text
nonempty_pass = mask_area > 0
border_pass = touches_image_border == False
extent_pass = bbox_extent_ratio <= 0.20
frozen_eligible = nonempty_pass AND border_pass AND extent_pass
```

Assign exactly one `fail_reason`:

```text
NONE
ZERO_AREA_ONLY
BORDER_ONLY
EXTENT_ONLY
BORDER_AND_EXTENT
ZERO_AREA_PLUS_OTHER
```

Rules:

```text
NONE:
nonempty_pass AND border_pass AND extent_pass

ZERO_AREA_ONLY:
NOT nonempty_pass AND border_pass AND extent_pass

BORDER_ONLY:
nonempty_pass AND NOT border_pass AND extent_pass

EXTENT_ONLY:
nonempty_pass AND border_pass AND NOT extent_pass

BORDER_AND_EXTENT:
nonempty_pass AND NOT border_pass AND NOT extent_pass

ZERO_AREA_PLUS_OTHER:
NOT nonempty_pass AND (NOT border_pass OR NOT extent_pass)
```

Record for every proposal:

```text
proposal_id
mask_area
confidence
global_bbox
touches_image_border
bbox_extent_ratio
iou_to_gt
nonempty_pass
border_pass
extent_pass
frozen_eligible
fail_reason
```

# 9. Four diagnostic policies

Counterfactual policies are diagnostic only:

```text
P0_FROZEN:
nonempty AND border_pass AND extent_pass

P1_BORDER_RELAXED_ONLY:
nonempty AND extent_pass

P2_EXTENT_RELAXED_ONLY:
nonempty AND border_pass

P3_BOTH_RELAXED:
nonempty
```

# 10. IMPORTANT — production selection and GT-best must be separate

For EACH policy compute TWO distinct outputs.

## 10.1 `production_selected`

Rank policy-eligible proposals by exactly:

```python
(
    -mask_area,
    -confidence,
    proposal_id,
)
```

Record:

```text
production_selected_id
production_selected_mask_area
production_selected_confidence
production_selected_iou
```

GT IoU MUST NOT participate in production selection.

## 10.2 `best_coverage`

Diagnostic only.

Rank policy-eligible proposals by:

```python
(
    -iou_to_gt,
    -confidence,
    proposal_id,
)
```

Record:

```text
best_coverage_id
best_coverage_iou
best_coverage_confidence
```

Never call `best_coverage` the policy `selected` proposal.

# 11. Mandatory consistency with frozen P0 production behavior

P0 production-selected IDs MUST reproduce the already closed reference forensics:

```text
right = 1
left  = 14
above = 4
below = 1
```

If any P0 production-selected ID differs:
- Status = STOP
- do not classify eligibility blocker
- no repair/tuning.

This is the primary guard against repeating E1's GT-IoU-selection error.

# 12. `above` isolated blocker mechanics

From live metadata independently establish proposal 5:

```text
proposal_id = 5
iou_to_gt = 0.9032501889644747
touches_image_border = False
bbox_extent_ratio = 0.296875
mask_area = 5059
```

Its proposal-level fail reason MUST be mechanically computed.

Expected if data agree:

```text
EXTENT_ONLY
```

Record:

```text
extent_excess_absolute =
0.296875 - 0.20
= 0.096875

extent_excess_relative =
0.296875 / 0.20 - 1
= 0.484375
```

These are margins only, NOT new threshold proposals.

# 13. `above` counterfactual isolation

Use **production_selected_iou** for blocker isolation.

Required mechanics:

```text
P0.production_selected_iou < 0.50
P1.production_selected_iou < 0.50
P2.production_selected_iou >= 0.50
```

Then require the covered best-any proposal:
- has IoU >= 0.50;
- has fail_reason = EXTENT_ONLY.

If all true:

```text
eligibility_blocker_subtype = BBOX_EXTENT_CAP
```

P3 is NOT required to restore correctness.

P3 must still be reported using production largest selection.

Important:
- If P3 selects a larger bad proposal, report it faithfully.
- Do NOT overwrite it with the GT-best proposal.

If isolation conditions fail:

```text
eligibility_blocker_subtype =
ELIGIBILITY_FORENSICS_INCONSISTENT
```

and Status = STOP.

# 14. Cross-candidate guard

For all four candidates report, for P0/P1/P2/P3:

```text
eligible_count
production_selected_id
production_selected_iou
best_coverage_id
best_coverage_iou
```

This guard is diagnostic only.

Also preserve frozen reference class:

```text
right = REFERENCE_SELECTED_CORRECT
left = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Do not claim left/below selection defects are fixed by E1-R1.

# 15. Required corrected outcomes

If §13 establishes `BBOX_EXTENT_CAP`, use exactly:

```text
overall_outcome =
REF01_ELIGIBILITY_FORENSICS_COMPLETE

eligibility_blocker_subtype =
BBOX_EXTENT_CAP

ref01_status =
FORENSICS_COMPLETE_ELIGIBILITY_BLOCKER_ISOLATED

next_gate =
REF01_ELIGIBILITY_REPAIR_DESIGN
```

Do NOT use:

```text
REF01_ELIGIBILITY_BLOCKER_ISOLATED_BBOX_EXTENT_CAP
```

as the overall outcome enum.

Do not execute NEXT.

# 16. Script artifact — mandatory

Create and COMMIT:

```text
scripts/task8b3_ref01_eligibility_forensics.py
```

Requirements:
- deterministic read-only replay;
- zero detector/model inference;
- reads canonical REF01 evidence + external proposals JSON;
- checks external detector source identity/rules;
- writes only:
  `evaluation/task8b3_ref01_eligibility_forensics.json`.

If this tracked script is absent from the final commit:
- task is NOT COMPLETE.

# 17. Corrected evidence JSON

Replace:

```text
evaluation/task8b3_ref01_eligibility_forensics.json
```

Required top-level keys:

```text
task = 8B.3-REF01-E1-R1
starting_head
branch
verification_mode = READ_ONLY_ELIGIBILITY_COUNTERFACTUAL
detector_model_calls = 0
scientific_reuse_disclosure
coverage_threshold = 0.50
frozen_extent_cap = 0.20
frozen_source_identity
candidate_results
above_isolation
eligibility_blocker_subtype
overall_outcome
ref01_status
next_gate
```

Each candidate must include:
- frozen reference class;
- complete per-proposal predicate table;
- P0/P1/P2/P3;
- production_selected and best_coverage as separate objects.

# 18. Report

Update:

```text
docs/task8b3_ref01_eligibility_forensics.md
```

Preserve E1 history, then add a clearly labeled R1 correction section:

```text
E1 initial replay = INVALID because GT-IoU ranking was used as policy selection.
E1-R1 = corrected production-selection counterfactual.
```

Required table:

```text
relation | policy | eligible | prod-selected id/IoU | GT-best id/IoU
```

Required `above` section:
- proposal 5 predicate facts;
- fail reason;
- extent margins;
- P0/P1/P2/P3 production results;
- isolated subtype.

# 19. Nonclaims

Report must explicitly state:

```text
No new threshold was selected.
No eligibility rule was changed.
No product repair was implemented.
Counterfactual policies are diagnostic only.
P2 success does NOT by itself establish that removing the 0.20 cap globally is safe.
P3 behavior is diagnostic and may expose larger invalid proposals.
Border exclusion remains frozen.
Task 7J metrics/model/threshold/seed/architecture are unchanged.
PROP-01 remains PROP01_OPEN_ENGINEERING_DEFECT.
left and below selection errors remain separate unresolved reference-selection issues.
```

# 20. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-E1-R1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-forensics
Starting HEAD: 3c497bf3d427366e00b5890fdcfab1a35e68b28c
Detector/model calls: 0
Tracked replay script committed: YES / NO
Frozen extent cap: 0.20
Coverage threshold: 0.50_TASK7F_FROZEN
P0 production selected right/left/above/below: 1/14/4/1 / other
above best-any id: 5 / other
above best-any IoU: 0.9032501889644747 / other
above proposal 5 border: False / other
above proposal 5 extent: 0.296875 / other
above proposal 5 fail reason: EXTENT_ONLY / other
above extent excess absolute: 0.096875 / other
above extent excess relative: 0.484375 / other
above P0 production selected: <id>/<iou>
above P1 production selected: <id>/<iou>
above P2 production selected: <id>/<iou>
above P3 production selected: <id>/<iou>
Eligibility blocker subtype: BBOX_EXTENT_CAP / other
GT IoU used in production selection: NO
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

# 21. Commit / push

If COMPLETE:

```text
docs(rc1): correct eligibility counterfactual replay
```

If STOP/FAILED:

```text
docs(rc1): record eligibility replay correction stop
```

Push only:

```text
fix/task8b3-ref01-eligibility-forensics
```

No force push.
Do not update main.

Then STOP.

# 22. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only five allowed paths changed;
- tracked replay script exists in commit;
- detector/model calls = 0;
- source eligibility semantics verified;
- proposal-ID joins exact;
- every proposal gets exact fail_reason enum;
- P0 production IDs reproduce 1/14/4/1;
- policy production selection never uses GT IoU;
- best_coverage is reported separately;
- all P0/P1/P2/P3 policies replayed for all four candidates;
- above proposal 5 is mechanically attributed;
- blocker subtype mechanically isolated;
- corrected exact outcome/ref01/NEXT enums used;
- no threshold/rule change or repair;
- evidence/report/FROM_DSH committed and pushed;
- NEXT not executed;
- STOP.
