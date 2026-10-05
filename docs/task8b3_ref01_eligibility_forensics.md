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
