# Task 8B.3-REF01-E1 — Frozen Eligibility Attribution

## 1. Task, branch and scope

```text
base branch / head = fix/task8b3-ref01-reference-forensics / 798a2fcb50054a67b8101581c5007cbbfa46c36a
task branch        = fix/task8b3-ref01-eligibility-forensics
detector / model calls this task = 0
proposal metadata source = EXISTING_P1D12_JSON_ONLY
canonical REF01 evidence = evaluation\task8b3_ref01_locked_reference_forensics.json (sha256 f7495796577cb26a..., read-only)
coverage threshold = 0.50 (frozen Task 7F, unchanged)
```

No threshold was changed, no rule was modified, no repair was implemented, no image was inspected and NEXT was not
executed. The analysis replays the frozen eligibility predicates over the existing P1D12 proposal metadata combined
with the canonical REF01 per-proposal IoU values.

## 2. Policy definitions

| policy | predicate |
|---|---|
| `P0_FROZEN` | `mask_area > 0` AND not `touches_image_border` AND `bbox_extent_ratio <= 0.20` |
| `P1_BORDER_RELAXED_ONLY` | `mask_area > 0` AND `bbox_extent_ratio <= 0.20` |
| `P2_EXTENT_RELAXED_ONLY` | `mask_area > 0` AND not `touches_image_border` |
| `P3_BOTH_RELAXED` | `mask_area > 0` |

Policy `selected` is the frozen ranking `(-IoU, -confidence, proposal_id)` applied to that policy's eligible set.

## 3. Counterfactual table (all four candidates)

| relation | P0 sel/IoU | P1 sel/IoU | P2 sel/IoU | P3 sel/IoU | GT-best id/IoU | GT-best fail reason |
|---|---|---|---|---|---|---|
| right | 1 / 0.5589 | 1 / 0.5589 | 1 / 0.5589 | 1 / 0.5589 | 1 / 0.5589 | NONE |
| left | 30 / 0.6501 | 30 / 0.6501 | 30 / 0.6501 | 30 / 0.6501 | 30 / 0.6501 | NONE |
| above | 4 / 0.0000 | 4 / 0.0000 | 5 / 0.9033 | 5 / 0.9033 | 5 / 0.9033 | BBOX_EXTENT_CAP |
| below | 2 / 0.6165 | 2 / 0.6165 | 2 / 0.6165 | 2 / 0.6165 | 2 / 0.6165 | NONE |

## 4. `above` blocker isolation

```text
above best-any id      = 5
above best-any IoU     = 0.903250
above best-any border  = False
above best-any extent  = 0.296875
above best-any mask_area = 5059
above best-any fail reason = BBOX_EXTENT_CAP
above P0 selected/id   = 4 / 0.000000
above P1 selected/id   = 4 / 0.000000
above P2 selected/id   = 5 / 0.903250
above P3 selected/id   = 5 / 0.903250
bbox-extent-cap isolation condition satisfied = True
shared with other candidates = ['above']
```

The `above` blocker is therefore attributed mechanically: the high-IoU proposal is a non-border proposal that the
**bbox-extent cap** removes from the eligible set, because relaxing the extent predicate alone lifts the policy
selection to at least the frozen 0.50 coverage threshold while relaxing the border predicate alone does not.

## 5. Explicit non-execution

```text
threshold change / rule modification / repair implementation = NONE / NONE / NONE
manual visual inspection / candidate replacement            = NO / NO
detector / model / Qwen / SAM2 / D-B1 / target segmentation = NONE
NEXT executed = NO
```

## 6. Outcome

```text
Outcome = REF01_ELIGIBILITY_BLOCKER_ISOLATED_BBOX_EXTENT_CAP
NEXT = REF01_ELIGIBILITY_REPAIR_DESIGN
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT (unchanged)
```


---

## 7. E1-R1 correction — production selection rule, GT coverage, enum and margin

```text
branch = fix/task8b3-ref01-eligibility-forensics
HEAD   = 3c497bf3d427366e00b5890fdcfab1a35e68b28c
detector / model calls = 0
tracked replay script = scripts/task8b3_ref01_eligibility_forensics.py
```

**Corrected defect.** E1 (commit `3c497bf`) ranked each policy's `selected` proposal by **GT IoU**, which conflated GT
coverage with the production selection. E1-R1 replays the production selection with the **frozen production ranking
`(-mask_area, -confidence, proposal_id)`** over each policy's eligible set — matching the `largest_*` query semantics —
and records GT-best coverage **separately**.

### 7.1 Corrected counterfactual table

| relation | P0 sel id/IoU | P1 sel id/IoU | P2 sel id/IoU | P3 sel id/IoU | GT-best id/IoU | GT-best fail reason | extent margin |
|---|---|---|---|---|---|---|---|
| right | 1 / 0.5589 | 1 / 0.5589 | 1 / 0.5589 | 1 / 0.5589 | 1 / 0.5589 | ELIGIBLE | -0.076953125 |
| left | 14 / 0.0000 | 14 / 0.0000 | 27 / 0.0000 | 27 / 0.0000 | 30 / 0.6501 | ELIGIBLE | -0.053515625 |
| above | 4 / 0.0000 | 4 / 0.0000 | 5 / 0.9033 | 3 / 0.0000 | 5 / 0.9033 | BBOX_EXTENT_CAP | 0.096875 |
| below | 1 / 0.0000 | 1 / 0.0000 | 1 / 0.0000 | 1 / 0.0000 | 2 / 0.6165 | ELIGIBLE | -0.061328125 |

### 7.2 Fail-reason enum and extent margin

Per-proposal predicates are reported as the enum `MASK_AREA_ZERO` / `TOUCHES_IMAGE_BORDER` / `BBOX_EXTENT_CAP` (empty =
`ELIGIBLE`). The extent margin is `bbox_extent_ratio - 0.2` for the GT-best proposal.

### 7.3 `above` isolation (corrected)

```text
above GT-best id          = 5
above GT-best IoU         = 0.903250
above GT-best mask_area   = 5059
above GT-best border      = False
above GT-best extent      = 0.296875
above GT-best extent margin = 0.096875
above GT-best fail reason = BBOX_EXTENT_CAP
above P0 production selected = 4 (mask_area 3012, IoU 0.000000)
above P1 production selected = 4 (IoU 0.000000)
above P2 production selected = 5 (IoU 0.903250)
above P3 production selected = 3 (IoU 0.000000)
bbox-extent-cap isolation condition satisfied = True
```

Under the corrected rule the production selection for `above` is the largest eligible proposal in each policy, while the
GT-covering proposal (id 5) remains excluded by the extent cap: its extent is
0.296875 against the frozen cap 0.2, i.e. a margin of
**0.096875**, and no border violation is involved.

### 7.4 Outcome schema

```text
branch            = fix/task8b3-ref01-eligibility-forensics
above_isolation   = BBOX_EXTENT_CAP (isolated = True)
overall_outcome   = REF01_ELIGIBILITY_BLOCKER_ISOLATED_BBOX_EXTENT_CAP
dominant_blocker  = ELIGIBILITY
next_gate         = REF01_ELIGIBILITY_REPAIR_DESIGN (not executed)
```

### 7.5 Explicit non-execution

```text
threshold change / rule modification / repair implementation = NONE / NONE / NONE
manual visual inspection / candidate replacement            = NO / NO
NEXT executed = NO
```


---

## 8. E1-R2 — verbatim execution of the task book's fixed replay code

```text
branch = fix/task8b3-ref01-eligibility-forensics
HEAD   = 1594f1ef96223264999d2c938dd5b2d2bb1b629c
detector / model calls = 0 (the fixed code performs no detection and no model inference)
fixed code source = handoff/TO_DSH.md lines 309-663 (fenced python block, copied byte-for-byte)
verbatim code sha256 = 4c3c400700fcdfe482364d36dc1a0ee7fafbc96ced4a319519955da9580af3e7
script = scripts/task8b3_ref01_eligibility_forensics.py
py_compile exit = 0
replay exit     = 0
status = COMPLETE
evidence written by the fixed code = True
```

No algorithm, enum, schema, ordering or exception handling was modified; the block was extracted and executed exactly as
published. 

Replay output (tail):

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

No execution error was raised.

### 8.1 Explicit non-execution

```text
self-repair of the fixed code / algorithm change / enum change / schema change / ordering change = NONE
manual visual inspection / threshold change / rule modification / repair implementation = NO / NONE / NONE / NONE
NEXT executed = NO
```
