# Task 8B.3-REF01-E3C3 — Selection Repair Decision Freeze

## Decision

`REJECT_FURTHER_SCALAR_RANK_REPAIR`

ChatGPT freezes the automatic largest-reference selector at the already-validated eligibility repair:

`LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1`

No additional scalar ranking weight, threshold, centroid prior, or locked-four-specific rule is authorized.

## Evidence basis

E3C2 evaluated exactly six predeclared threshold-free scalar ranking families and found no rule that is correct on all four locked cases.

### left
- current automatic reference: proposal `14`
- locked best-covered reference: proposal `30`
- none of the six predeclared scalar rules selects proposal `30`

### below
- current automatic reference: proposal `1`
- locked best-covered reference: proposal `2`
- proposal `1` dominates proposal `2` in `(mask_area, confidence)`
- proposal `1` also dominates proposal `2` in `(bbox_area, confidence)`
- all six predeclared rules still select proposal `1`

Therefore the remaining below failure is not supportably repairable by a monotone scalar re-ranking over the currently persisted proposal attributes. Searching additional weights or thresholds after observing these locked cases would be post-hoc tuning.

## Frozen automatic behavior

```text
right:  1 -> 1  correct stable
left:  14 -> 14 residual selection limitation
above:  4 -> 5  eligibility blocker resolved
below:  1 -> 1  residual selection limitation
```

## Engineering fallback

RC1 already provides an assisted reference override through `reference_id`.

The fallback is therefore:

`ASSISTED_REFERENCE_OVERRIDE`

No new product code or new model branch is introduced for this fallback.

## Explicitly rejected in RC1

- new scalar weight fitting
- new scalar threshold tuning
- relation-specific centroid/location prior
- post-hoc locked-four rule search
- new SAM/SAM2 proposal-refinement branch
- new model or architecture branch for reference selection

## Status

- detector/model calls = `0`
- proposal regeneration = `NO`
- selector repair implementation = `NO`
- product source change = `NO`
- external write = `NO`
- REF-01 = `ACTIVE_RESIDUAL_SELECTION_LIMITATION`
- PROP-01 = `PROP01_OPEN_ENGINEERING_DEFECT`
- outcome = `REF01_SCALAR_REPAIR_REJECTED_ASSISTED_FALLBACK_FROZEN`
- NEXT = `MASK01_VALIDITY_FORENSICS_DESIGN`
