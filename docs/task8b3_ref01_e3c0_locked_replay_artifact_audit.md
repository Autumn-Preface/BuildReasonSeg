# Task 8B.3-REF01-E3C0-R2-R4 — Final Locked Replay Readiness Audit

## Scope
- starting HEAD = `31006384fe65bb29247c67d3b504b53d9c1ae981`
- schema = `{"count": N, "items": [...]}`
- exact known scientific artifacts read = 9
- broad search = NO
- detector/model calls = 0
- proposal regeneration = NO
- actual replay = NO
- external write = NO

## A/B/C/D
| case | tile | records | A complete list | B scalar replay | C exact object/mask | D forensic IoU |
|---|---:|---:|---|---|---|---|
| right | 1010 | 6 | True | True | False | True |
| left | 1003 | 53 | True | True | False | True |
| above | 1008 | 9 | True | True | False | True |
| below | 1009 | 6 | True | True | False | True |

For record-level replay only, persisted `mask_area > 0` is accepted as the nonempty witness for each serialized merged proposal record. This does not satisfy C; exact `mask_crop` and full production-object reconstruction material remain mandatory for full production-object replay.

## Frozen historical consistency checks
### right
- `raw_count` = `MATCH`
- `merged_count` = `MATCH`
- `eligible_count` = `MATCH`
- `pre_selected` = `MATCH`
### left
- `raw_count` = `MATCH`
- `merged_count` = `MATCH`
- `eligible_count` = `MATCH`
- `pre_selected` = `MATCH`
- `best_covered` = `MATCH`
### above
- `raw_count` = `MATCH`
- `merged_count` = `MATCH`
- `eligible_count` = `MATCH`
- `pre_selected` = `MATCH`
- `best_covered` = `MATCH`
- `proposal5_mask_area` = `MATCH`
- `proposal5_confidence` = `MATCH`
- `proposal5_touches_image_border` = `MATCH`
- `proposal5_bbox_extent_ratio` = `MATCH`
- `proposal5_iou` = `MATCH`
- `proposal3_mask_area` = `MATCH`
- `proposal3_touches_image_border` = `MATCH`
- `proposal3_bbox_extent_ratio` = `MATCH`
### below
- `raw_count` = `MATCH`
- `merged_count` = `MATCH`
- `eligible_count` = `MATCH`
- `pre_selected` = `MATCH`
- `best_covered` = `MATCH`

## Readiness
- overall readiness = `RECORD_LEVEL_REPLAY_READY`
- production-object replay possible without detector = `False`
- record-level replay possible without detector = `True`
- historical consistency mismatch count = `0`

## Non-claims
- repaired selector replay has NOT been performed
- `right1 / left14 / above5 / below1` are NOT replay-validated by this task
- REF-01 remains `ACTIVE`
- PROP-01 remains `PROP01_OPEN_ENGINEERING_DEFECT`
- final Demo has NOT run

## Outcome / NEXT
- outcome = `REF01_LOCKED_REPLAY_READINESS_ESTABLISHED`
- NEXT = `REF01_LOCKED_RECORD_REPLAY_DESIGN`
- NEXT was not executed
