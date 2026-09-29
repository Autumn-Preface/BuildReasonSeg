# Task 6Q — Frozen Proposal Reference Resolver Audit

> Task: `handoff/TO_DSH.md` (Task 6Q) · Base commit: `b80f3cc` · Predecessor: Task 6P →
> `REFERENCE_HEAD_INSUFFICIENT`
> **Verdict: `REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`**
> Tests: `tests/test_task6q_frozen_proposal_reference_resolver.py` · Evidence: `evaluation/task6q_*.json`

Diagnostic/selection task: no training, no confidence/NMS/max_det tuning, no test split. It tests a
cheaper and more interpretable reference resolver than the Task 6P dense head:

> frozen building instance proposals → semantic-policy eligibility → deterministic largest/smallest
> selection → reference mask → differentiable relation field → frozen B3 target decoder.

## 1. Literature-position note (fixed; no new search performed)

* Current RRSIS work already uses explicit object/relation/position decomposition and candidate/graph
  reasoning, e.g. **SRGFormer**.
* Current reasoning-segmentation work also commonly decouples semantic reasoning from
  grounding/segmentation through foundation-model proposals/prompts, e.g. **Think2Seg-RS**.
* Therefore **proposal-based reference grounding is treated here only as a reliable support module**.
* The project's candidate method contribution remains the **reference-conditioned geometric relation
  field guiding dense visual target segmentation**, later combined with explicit relation-level
  supervision.

No "first-ever" claim is made, and the resolver is not claimed as novelty.

## 2. Frozen proposal configuration (verified before any inference)

| Item | Value |
|---|---|
| checkpoint | `artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt` |
| SHA256 (recomputed, exact match) | `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` |
| family / imgsz / conf / max_det | YOLO26m-seg / **640** / **0.10** / **100** |
| NMS / TTA / tiling / sweep | default / none / none / **none** |
| mask threshold | canonical Task 6M binarization (Ultralytics retina mask `> 0` through `normalize_mask`) |

No threshold was tuned and the model was never retrained.

## 3. Deterministic resolver rules

Proposal normalization: every source 512×512 tile is run **exactly once**; each predicted instance mask
is restored to 512×512, binarized with the canonical Task 6M threshold, and described by `area_px`,
`bbox`, `bbox_extent_ratio = bbox_area / (512·512)` and `touches_border` (any positive pixel on row 0,
row 511, col 0 or col 511).

| Family | Eligibility | Selection |
|---|---|---|
| `largest` | NOT `touches_border` **and** `bbox_extent_ratio ≤ 0.20` (tiny **not** rejected) | maximum `area_px` |
| `smallest` | the above **and** `area_px ≥ 150` | minimum `area_px` |

Tie-break on equal area: **higher confidence**, then **lower original proposal index**. Zero proposals
or zero eligible proposals → explicit abstention (`no_proposals` / `no_eligible_proposals`). Ground
truth never influences eligibility, ranking or tie-breaks; the target identity never enters the
resolver.

## 4. Reference-level audit (RefValUnique, 219 unique references over 212 tiles)

`evaluation/task6q_reference_resolver_val.json`. Ground truth is used for evaluation only.

**Proposal coverage** (best IoU against the GT reference):

| Family | all proposals @0.25 / @0.50 / @0.75 | eligible proposals @0.25 / @0.50 / @0.75 |
|---|---|---|
| overall (219) | 0.7489 / **0.6941** / 0.4612 | 0.7397 / **0.6895** / 0.4566 |
| largest (110) | 0.9000 / **0.8364** / 0.6091 | 0.9000 / **0.8364** / 0.6000 |
| smallest (109) | 0.5963 / **0.5505** / 0.3119 | 0.5780 / **0.5413** / 0.3119 |

**Deterministic selected-reference quality**:

| Metric | overall | largest | smallest |
|---|---|---|---|
| reference mIoU | **0.413164** | 0.537910 | 0.287274 |
| Dice | 0.469992 | 0.608308 | 0.330408 |
| Pr@0.5 | 0.511415 | 0.645455 | 0.376147 |
| centroid error median | **0.015607** | 0.007302 | 0.065959 |
| centroid error p90 | **0.393517** | 0.317427 | 0.428458 |
| median area ratio | 1.081301 | 1.030109 | 1.162356 |
| abstentions (rate) | 6 (0.027397) | 1 (0.009091) | 5 (0.045872) |

**Failure attribution** (`evaluation/task6q_reference_failure_attribution.json`, one category per
record in the fixed priority order):

| Category | overall | largest | smallest |
|---|---|---|---|
| `REFERENCE_OK` | **111** | 70 | 41 |
| `REFERENCE_NOT_COVERED_IOU50` | **62** | 17 | 45 |
| `EXTREME_SELECTION_WRONG` | **39** | 21 | 18 |
| `NO_ELIGIBLE_PROPOSALS` | 3 | 0 | 3 |
| `NO_PROPOSALS` | 3 | 1 | 2 |
| `SELECTED_MASK_GEOMETRY_POOR` | 1 | 1 | 0 |

Proposal statistics from the gitignored resolver cache (212 tiles): 0–51 proposals per tile, mean 8.72,
2 tiles with zero proposals, 10 (tile, family) entries abstained, 414 cached selections.

## 5. Downstream target propagation (frozen MiniVal240)

`evaluation/task6q_target_val.json` — resolved proposal reference → frozen GeometricRelationField
**v0.2** → frozen Task 6O **B3** → target mask (the oracle reference mask, target GT, candidate target
masks and target instance id are never inputs):

| Metric | Value |
|---|---|
| target mIoU (234 answered of 240) | **0.304581** |
| Dice | 0.384945 |
| Precision@0.5 | 0.635327 |
| target abstentions | 6 (rate **0.025000**) |
| per relation mIoU (left / right / above / below) | 0.3353 / 0.2471 / 0.3291 / 0.3034 |
| per reference family (largest / smallest) | 0.3350 / 0.2731 |
| border target (n=110) | 0.304968 |
| tiny target (n=4) | ≈0 |

Comparison: Task 6O oracle B3 **0.4299680351479113** (delta **−0.125387**); Task 6P dense predicted
reference **0.24096754293919803** (delta **+0.063614**).

`evaluation/task6q_target_paired_val.json` — exact frozen PairedVal20; a pair sharing the image and the
reference source reuses exactly the same resolved proposal reference mask, only the relation changes:

| Metric | This task | Oracle B3 | Task 6P dense predicted |
|---|---|---|---|
| pass | **10/20** | 14/20 | 0/20 |
| mean own IoU | 0.277946 | 0.398969 | 0.064774 |
| mean cross IoU | 0.004245 | 0.001773 | 0.061155 |
| own − cross margin | **+0.273700** | +0.397196 | +0.003619 |
| pairs with reference abstention | 0 | — | — |

## 6. Gates and verdict

**Section 11 proposal coverage — FAIL**: eligible coverage@0.50 overall 0.6895 < **0.70**;
largest 0.8364 ≥ 0.75 ✓; smallest 0.5413 < **0.60**.

**Section 12 reference-resolver adequacy — FAIL**: mIoU 0.413164 ≥ 0.35 ✓; median centroid error
0.015607 ≤ 0.05 ✓; **p90 centroid error 0.393517 > 0.12**; abstention rate 0.027397 ≤ 0.10 ✓.

**Section 13 downstream chain retention — FAIL** (only through section 12): section 12 fails; target
mIoU 0.304581 ≥ 0.3009776246035379 ✓; PairedVal 10/20 ≥ 10 ✓; margin +0.273700 ≥ 0.20 ✓.

`evaluation/task6q_verdict.json` — priority order applied literally:

1. `INVALID_EXPERIMENT` — no.
2. `PROPOSAL_CHECKPOINT_UNAVAILABLE` — no (hash verified exactly).
3. **`REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`** — section 11 fails. ← **verdict**
4. `DETERMINISTIC_REFERENCE_SELECTOR_INSUFFICIENT` — not reached.
5. `REFERENCE_ERROR_PROPAGATION_SEVERE` — not reached (3 of its 4 sub-conditions actually pass).
6. `PROPOSAL_REFERENCE_CHAIN_FEASIBLE` — no.

## 7. Interpretation boundary (section 21)

DSH reports measurements only. This document does **not** claim the proposal resolver is novel, does
**not** decide to keep proposals permanently, does **not** propose another learned reference
architecture, and does **not** start MLLM integration, GRCL or nearest/L3. No gate or threshold was
altered.

## 8. Reproduce

```text
python scripts/task6q_reference_audit.py        # checkpoint check + Part D audit + resolver cache
python scripts/task6q_target_propagation.py     # Parts E (MiniVal240 + PairedVal20)
python scripts/task6q_report.py                 # Parts F-G (gates + verdict)
```

Run `task6q_reference_audit.py` in `.conda/buildreasonseg-proposal` (Ultralytics) and the rest in
`.conda/buildreasonseg-proposal` as well (it needs the frozen SAM2 + torch stack). The YOLO checkpoint,
the resolver cache (`artifacts/task6q/`) and the feature cache stay gitignored.
