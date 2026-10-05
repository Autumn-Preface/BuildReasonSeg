# Task MASK01_F2_LOCKED_SUCCESS_ARTIFACT_FORENSICS — Locked Successful Mask Evidence

## 1. Git identity

```text
starting branch = audit/task8b3-mask01-f1-r1-corrective-forensics
starting head   = e8f1315c9afdaac818126db6a2ac76b758027f70
task branch     = audit/task8b3-mask01-f2-locked-success-artifacts
final commit sha = POST_COMMIT_EXTERNAL_FACT (recorded outside this commit by rule)
```

## 2. Scope and prohibitions

Only the saved A1/A3/A4 final masks, overlays, `result.json`, `maps.npz` and diagnostics directories were inspected.
No other case was searched, no model was run, no mask was regenerated or altered, no external file was written and no
validity repair was designed or implemented.

## 3. Artifact identity

| case | mask | overlay | result.json | maps.npz | diagnostics children |
|---|---|---|---|---|---|
| A1 | True | True | True | True | 13 |
| A3 | True | True | True | True | 13 |
| A4 | True | True | True | True | 13 |

## 4. Final mask exact characterization

Descriptive, threshold-free measurements computed from the saved PNGs (foreground = `pixel_value != 0`):

| case | status | result mask_area | foreground pixels | fraction of full image | components (8-conn) | largest component fraction | pixel-count check |
|---|---|---|---|---|---|---|---|
| A1 | SUCCESS | 101 | 101 | 9.6321106e-05 | 1 | 1.0 | MATCH |
| A3 | SUCCESS | 1439 | 1439 | 0.001372337341 | 4 | 0.990965948575 | MATCH |
| A4 | SUCCESS | 359 | 359 | 0.00034236908 | 1 | 1.0 | MATCH |

No cutoff is proposed and no mask is classified valid/invalid from these numbers.

## 5. Consistency verification

```text
foreground_pixel_count == result.json.mask_area held for: ['A1', 'A3', 'A4']
centroid vs result.json.target_centroid: ['MATCH', 'MATCH', 'MATCH']
```

## 6. Failure taxonomy

| case | primary | additional | runtime status | recorded mask_area |
|---|---|---|---|---|
| A1 | TAX_SEMANTIC_TARGET_MISMATCH | ['TAX_RUNTIME_VALID_QUALITY_POOR'] | SUCCESS | 101 |
| A3 | TAX_SEMANTIC_TARGET_MISMATCH | — | SUCCESS | 1439 |
| A4 | TAX_SEMANTIC_TARGET_MISMATCH | ['TAX_RUNTIME_VALID_QUALITY_POOR'] | SUCCESS | 359 |

## 7. Observable-signal inventory

```text
{
 "mask_full nonempty": {
  "present_in_A1": true,
  "present_in_A3": true,
  "present_in_A4": true
 },
 "mask_area": {
  "present_in_A1": true,
  "present_in_A3": true,
  "present_in_A4": true
 },
 "mask fraction": {
  "present_in_A1": true,
  "present_in_A3": true,
  "present_in_A4": true
 },
 "mask bbox": {
  "present_in_A1": true,
  "present_in_A3": true,
  "present_in_A4": true
 },
 "mask bbox fill ratio": {
  "present_in_A1": true,
  "present_in_A3": true,
  "present_in_A4": true
 },
 "mask connected components": {
  "present_in_A1": true,
  "present_in_A3": true,
  "present_in_A4": true
 }
}
connected-component metrics: NOT_CURRENTLY_AVAILABLE (DERIVABLE_FROM_RUNTIME_MASK)
```

## 8. Answers to the mandatory questions

```text
final mask PNGs exist                 = True (availability FOUND)
pixel counts match R4B/D1 mask_area   = True
masks structurally non-empty          = True
numeric threshold justified by 3 cases = False
masks regenerated / external written  = NO / NO
```

## 9. Evidence gaps

```text
none
```

## 10. Disposition

```text
task_status      = COMPLETE
repair_decision  = DEFER_TO_CHATGPT
next_gate        = CHATGPT_MASK01_F2_REVIEW
```
