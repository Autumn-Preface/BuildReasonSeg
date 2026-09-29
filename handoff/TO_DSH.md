# TO_DSH — Task 6Q: Frozen Proposal Reference Resolver Audit

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `b80f3cc04d237a12dbeef537280a371d217d52e3`
>
> Predecessor: Task 6P → `REFERENCE_HEAD_INSUFFICIENT`
>
> This is a diagnostic/selection task. ChatGPT has already chosen the research question and the exact resolver. DSH is an executor. Do not redesign the reference module.

## 0. DSH role

All user-facing DSH output must be Chinese.

DSH MAY:
- load the frozen Task 6M.1 YOLO26m-seg proposal checkpoint;
- implement the deterministic reference resolver specified here;
- run the exact validation-only audits;
- propagate the selected reference mask through frozen GeometricRelationField v0.2 + frozen B3;
- write tests, artifacts, docs and commits.

DSH MUST NOT:
- retrain YOLO;
- tune confidence/NMS/max_det;
- retrain the Task 6P dense ReferenceMaskHead;
- train a new reference network;
- change GeometricRelationField v0.2;
- retrain B3;
- add `[REF]`;
- add GRCL/SCL;
- add nearest/L3;
- add graph reasoning;
- access test split;
- add another dataset;
- choose Task 6R.

If an unexpected issue requires any of those changes, STOP and report.

# PART A — Research decision already made

## 1. Why Task 6Q exists

Task 6P established:

- GeometricRelationField v0.2 is valid and autograd-safe.
- Frozen B3 reproduces exactly.
- The simple dense reference head can overfit 20 examples:
  - mIoU `0.969504`
  - Dice `0.984388`
- But it generalizes poorly:
  - RefValUnique mIoU `0.220176`
  - centroid median `0.124060`
  - centroid p90 `0.319618`
- Predicted-reference target chain:
  - target mIoU `0.240968`
  - PairedVal `0/20`
  - own-cross margin `0.003619`

The likely structural issue is that `largest` / `smallest` are **instance-set ranking operations**. A dense local decoder conditioned only on a family embedding is not naturally suited to compare all building instances in the scene.

Before designing another trainable reference architecture, Task 6Q tests a cheaper and more interpretable resolver:

> frozen building instance proposals → semantic-policy eligibility → deterministic largest/smallest selection → reference mask → differentiable relation field → frozen B3 target decoder.

This proposal-based reference resolver is supporting infrastructure, not a claimed algorithmic novelty.

## 2. Literature-position note for docs

Copy this note into the Task 6Q design document; do not perform a new literature search:

- Current RRSIS work already uses explicit object/relation/position decomposition and candidate/graph reasoning, e.g. SRGFormer.
- Current reasoning-segmentation work also commonly decouples semantic reasoning from grounding/segmentation through foundation-model proposals/prompts, e.g. Think2Seg-RS.
- Therefore proposal-based reference grounding is treated here only as a reliable support module.
- The project’s candidate method contribution remains the reference-conditioned geometric relation field guiding dense visual target segmentation, later combined with explicit relation-level supervision.

No “first-ever” claim.

# PART B — Frozen assets

## 3. Freeze all previous evidence

Read-only:
- all Task 6M.1 artifacts;
- all Task 6N/6O/6P artifacts;
- GeometricRelationField v0.1 and v0.2;
- Task 6O B3 checkpoint;
- Task 6P reference packs;
- Task 6N MiniVal240 / PairedVal20 packs;
- BuildSpatialReason v0.2;
- WHU native-vector v1.0;
- frozen SAM2 feature cache;
- spatial relation configuration.

No test split.

## 4. Frozen proposal configuration

Use exactly Task 6M.1 frozen configuration:

Checkpoint:
`artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt`

Required SHA256:
`ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`

Inference:
- model family: YOLO26m-seg
- imgsz = `640`
- conf = `0.10`
- max_det = `100`
- default NMS
- no TTA
- no tiling
- no threshold sweep

Verify checkpoint hash before inference.

If unavailable/mismatch:
STOP with `PROPOSAL_CHECKPOINT_UNAVAILABLE`.

# PART C — Exact deterministic reference resolver

## 5. Proposal mask normalization

For every source 512×512 tile:

1. run the frozen proposal model exactly once;
2. convert each predicted instance mask to source-image 512×512 coordinates;
3. binarize using the same canonical segmentation-mask threshold used by Task 6M evaluation; do not invent a new threshold;
4. compute:
   - `area_px`
   - `bbox`
   - `bbox_extent_ratio = bbox_area / (512*512)`
   - `touches_border`

`touches_border = true` if any positive mask pixel touches row 0, row 511, col 0 or col 511.

## 6. Predicted eligibility

Use frozen semantic-policy concepts on predicted masks only.

Constants:
- suspected large merge bbox extent ratio threshold = `0.20`
- tiny component area threshold = `150 px`

### Largest candidates

Eligible if:
- NOT `touches_border`
- `bbox_extent_ratio <= 0.20`

Do not reject by tiny threshold.

Select eligible proposal with maximum predicted `area_px`.

### Smallest candidates

Eligible if:
- NOT `touches_border`
- `bbox_extent_ratio <= 0.20`
- `area_px >= 150`

Select eligible proposal with minimum predicted `area_px`.

### Ties

If equal area:
1. higher YOLO confidence;
2. lower original proposal index.

### Abstention

If zero proposals or zero eligible proposals for the requested family, return explicit abstention.

No GT may affect eligibility, ranking or tie-breaks.

# PART D — Reference-level audit

## 7. Use exact Task 6P RefValUnique

Use frozen Task 6P `RefValUnique` only.
Do not regenerate it.

For each unique reference record:

### Proposal coverage diagnostics

Compute against GT reference for evaluation only:
- best IoU among ALL proposals
- best IoU among ELIGIBLE proposals for that family

Aggregate:
- coverage@0.25 / 0.50 / 0.75
- eligible coverage@0.25 / 0.50 / 0.75
- by largest/smallest

### Deterministic selected-reference quality

For selected proposal:
- mask IoU vs GT reference
- Dice
- centroid error normalized by image diagonal
- selected area / GT area
- abstention

Aggregate:
- reference mIoU
- Dice
- Pr@0.5
- median centroid error
- p90 centroid error
- median area ratio
- abstention count/rate
- largest family
- smallest family

Write:
`evaluation/task6q_reference_resolver_val.json`

## 8. Failure attribution

Assign every record exactly one category, priority order:

1. `NO_PROPOSALS`
2. `NO_ELIGIBLE_PROPOSALS`
3. `REFERENCE_NOT_COVERED_IOU50`
   - eligible proposal exists, best eligible IoU < 0.50
4. `EXTREME_SELECTION_WRONG`
   - best eligible IoU >= 0.50, selected proposal IoU < 0.50
5. `SELECTED_MASK_GEOMETRY_POOR`
   - selected IoU >= 0.50 but centroid error > 0.05
6. `REFERENCE_OK`

Report counts overall and by largest/smallest.

Write:
`evaluation/task6q_reference_failure_attribution.json`

# PART E — Downstream target propagation

## 9. Frozen target chain

For each frozen Task 6N MiniVal240 record:

1. run/get cached frozen YOLO proposals;
2. derive family from program (`largest` or `smallest`);
3. resolve reference mask with section 6;
4. if abstain, target prediction is explicit abstention;
5. otherwise generate `P_rel` using GeometricRelationField v0.2;
6. feed frozen Task 6O B3:
   - frozen SAM2 visual feature
   - predicted `P_rel`
   - direction relation id
7. predict target mask.

Do not provide:
- oracle reference mask
- target GT
- candidate target masks
- target instance id

GT target is scoring only.

Write:
`evaluation/task6q_target_val.json`

Report:
- target mIoU
- Dice
- Pr@0.5
- target abstention count/rate
- per relation
- per reference family
- border target
- tiny target if present

Compare against:
- Task 6O oracle B3 mIoU = `0.4299680351479113`
- Task 6P dense predicted-reference mIoU = `0.24096754293919803`

## 10. PairedVal20

Use exact frozen Task 6N PairedVal20.

For pairs sharing same image/reference:
- reuse exactly the same resolved proposal reference mask;
- only relation changes.

Report:
- pass / 20
- mean own IoU
- mean cross IoU
- own-cross margin
- pairs with reference abstention

Write:
`evaluation/task6q_target_paired_val.json`

Compare against:
- oracle B3: 14/20, margin `0.39719566349802166`
- Task 6P dense predicted reference: 0/20, margin `0.0036186017082471683`

# PART F — Gates

## 11. Proposal coverage gate

Pass only if:
- eligible reference coverage@0.50 >= `0.70` overall
- largest eligible coverage@0.50 >= `0.75`
- smallest eligible coverage@0.50 >= `0.60`

## 12. Reference resolver adequacy

Pass only if all:
- selected-reference mIoU >= `0.35`
- median normalized centroid error <= `0.05`
- p90 normalized centroid error <= `0.12`
- abstention rate <= `0.10`

## 13. Downstream chain retention

Pass only if all:
- section 12 passes
- target mIoU >= `0.3009776246035379`
- PairedVal >= `10/20`
- own-cross margin >= `0.20`

Do not alter gates.

# PART G — Verdict

## 14. Exactly one, priority order

1. `INVALID_EXPERIMENT`
2. `PROPOSAL_CHECKPOINT_UNAVAILABLE`
3. `REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`
   - section 11 fails
4. `DETERMINISTIC_REFERENCE_SELECTOR_INSUFFICIENT`
   - section 11 passes but section 12 fails
5. `REFERENCE_ERROR_PROPAGATION_SEVERE`
   - section 12 passes but section 13 fails
6. `PROPOSAL_REFERENCE_CHAIN_FEASIBLE`
   - sections 11, 12 and 13 pass

No other verdict.

# PART H — Interpretation boundary

DSH must not:
- claim proposal reference resolver is novel;
- decide to keep proposals permanently;
- decide a new learned reference architecture;
- start MLLM integration;
- start GRCL;
- start nearest/L3.

Final handoff recommendation exactly:

`等待 ChatGPT 根据 Task 6Q 的 frozen-proposal reference resolver 结果决定 Task 6R，不自行修改 reference 架构或开始 MLLM/GRCL/nearest/L3。`

# PART I — Required artifacts

Create:

```text
evaluation/task6q_reference_resolver_val.json
evaluation/task6q_reference_failure_attribution.json
evaluation/task6q_target_val.json
evaluation/task6q_target_paired_val.json
evaluation/task6q_verdict.json

docs/task6q_frozen_proposal_reference_resolver.md

buildreasonseg_mvp/task6q_reference_resolver.py

scripts/task6q_reference_audit.py
scripts/task6q_target_propagation.py
scripts/task6q_report.py
```

Optional:
`evaluation/task6q_proposal_cache_manifest.json`

Proposal caches:
`artifacts/task6q/proposals/`
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

# PART J — Tests

## 15. Required tests

At minimum:

1. Task 6P artifacts unchanged
2. Task 6M.1 inference config unchanged
3. proposal checkpoint SHA exact
4. conf exactly 0.10
5. imgsz exactly 640
6. max_det exactly 100
7. no threshold sweep
8. masks restored to 512×512
9. border predicate exact
10. merge threshold exact 0.20
11. tiny threshold exact 150
12. largest eligibility exact
13. smallest eligibility exact
14. deterministic largest selection
15. deterministic smallest selection
16. tie-break exact
17. no GT in reference selection
18. no target id in reference selection
19. RefValUnique reused
20. MiniVal240 reused
21. PairedVal20 reused
22. no test split
23. v0.2 field unchanged
24. frozen B3 checkpoint unchanged
25. B3 not retrained
26. no oracle reference in downstream chain
27. GT reference only evaluation
28. GT target only evaluation
29. same reference reused in paired same-reference case
30. no YOLO training
31. no dense reference-head retraining
32. no `[REF]`
33. no GRCL/SCL
34. no nearest/L3
35. no graph transformer
36. no 4B
37. no new dataset/download/install/GUI
38. previous suite preserved

Run:
`python -m pytest tests/ -q`

Task 6P ended at **658 passed, 1 skipped**.
Do not reduce prior passing tests.

# PART K — Git/storage

Do not commit:
- YOLO checkpoint
- proposal cache
- SAM2 checkpoint
- feature cache
- source images/vectors
- `.conda`
- large caches

Commit code/small JSON/docs/tests/handoff only.

Recommended:
1. `feat: add frozen-proposal reference resolver`
2. `eval: audit reference proposal selection and propagation`
3. optional docs/handoff commit

# PART L — Model policy

Default:
- DeepSeek V4.1 Flash + High

Use Flash + Max only for genuine implementation/runtime bugs.
Do not use V4 Pro by default.

# PART M — STOP

After Task 6Q:
- commit
- push
- update handoff
- STOP

Do not start:
- another reference architecture
- MLLM hidden-state fusion
- joint training
- GRCL
- nearest
- L3
- full-dataset training
- GUI

Wait for ChatGPT audit.
