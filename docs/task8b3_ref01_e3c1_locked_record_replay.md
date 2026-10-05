# Task 8B.3-REF01-E3C1 — Locked Record-Level Reference Selection Replay

## Scope
- starting HEAD = `6d1e8d1e69f3adefb4301948cae8918d4aec7ee1`
- design = `LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1`
- detector/model calls = 0
- proposal regeneration = NO
- detector pipeline replay = NO
- external write = NO
- product source change = NO

## Record-level adapter
This task invokes the current production `eligible_proposals()` and `select_reference()` functions using exact persisted selector scalars. Because exact `mask_crop` was not persisted, the already-frozen record fact `mask_area > 0` is represented by a synthetic `1x1 True` `mask_crop` witness. This is a record-level selector replay only and is **not** a full production-object replay.

Current selector Git-canonical identity:
- bytes = `21257`
- sha256 = `934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883`

## Locked replay result
| relation | tile | baseline | repaired | changed | repaired IoU | best IoU id | semantic status |
|---|---:|---:|---:|---|---:|---:|---|
| right | 1010 | 1 | 1 | False | 0.558869701727 | 1 | REFERENCE_SELECTED_CORRECT_STABLE |
| left | 1003 | 14 | 14 | False | 0.000000000000 | 30 | REFERENCE_SELECTION_WRONG_COVERED_REMAINS |
| above | 1008 | 4 | 5 | True | 0.903250188964 | 5 | REFERENCE_ELIGIBILITY_BLOCKER_RESOLVED |
| below | 1009 | 1 | 1 | False | 0.000000000000 | 2 | REFERENCE_SELECTION_WRONG_COVERED_REMAINS |

Frozen transition:
```text
right  1 -> 1
left  14 -> 14
above  4 -> 5
below   1 -> 1
```

## Interpretation
- `above`: the frozen extent-dominance exception changes the reference from proposal 4 to proposal 5, which is the locked best-IoU reference. The eligibility blocker is resolved.
- `right`: remains correctly selected and is not regressed.
- `left`: remains proposal 14 while proposal 30 is the locked best-covered reference; the remaining defect is selection/ranking, not eligibility.
- `below`: remains proposal 1 while proposal 2 is the locked best-covered reference; the remaining defect is selection/ranking, not eligibility.

## Status
- eligibility repair validation = `PASS`
- REF-01 = `ACTIVE`
- PROP-01 = `PROP01_OPEN_ENGINEERING_DEFECT`
- selection blockers remaining = `left`, `below`
- outcome = `REF01_ELIGIBILITY_REPAIR_LOCKED_RECORD_REPLAY_PASS`
- NEXT = `REF01_SELECTION_REPAIR_DESIGN`

## Non-claims
- no detector/model inference
- no proposal regeneration
- no full production-object replay
- no final Demo
- REF-01 is not closed
- left/below are not repaired
