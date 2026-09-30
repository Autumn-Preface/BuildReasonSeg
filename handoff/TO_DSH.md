# TO_DSH — Task 6X: Frozen SAM2 Proposal Refinement Audit

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `3de2142010e335f3a9d9b1c5244fdd6271f88c4e`
>
> Predecessor: Task 6W → `QUALITY_FILTER_NOT_HELPFUL`
>
> Research decision already made by ChatGPT:
>
> 1. Keep the core BuildReasonSeg method frozen:
>    `ProgramHead → Reference → GeometricRelationField v0.2 → frozen SAM2 visual feature → B3 target decoder`.
> 2. Keep **U-C1** as the current proposal generator:
>    `YOLO26m-seg, imgsz=640, conf=0.05, max_det=300, no TTA, no tiling`.
> 3. Do NOT use ProposalSetRanker v0.1 or ProposalQualityEstimator v0.1 in the Task 6X primary resolver.
> 4. Task 6W proved:
>    - oracle proposal-quality filtering is highly useful;
>    - the learned quality classifier is not reliable enough across the scene-disjoint validation split.
> 5. Before ending reference hardening, Task 6X tests one **training-free**, foundation-model-based alternative:
>
>    **Use the frozen official SAM2.1 image predictor to refine each eligible YOLO proposal mask from its
>    geometric box prompt, then execute the literal largest/smallest rule on the refined instance masks.**
>
> 6. This is support infrastructure, not a claimed research novelty.
> 7. This is the **last predeclared reference-hardening audit before ChatGPT decides whether to freeze the
>    reference subsystem and move on to nearest/L3.**
>
> DSH is an executor. Do not redesign this experiment.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- load the frozen SAM2.1 Hiera Base+ checkpoint already used by the project;
- use the official `SAM2ImagePredictor`;
- reuse/generate exact U-C1 proposals;
- evaluate exactly the four predeclared refinement options;
- freeze one option using train-only U-Calib200;
- evaluate it on untouched RefValUnique, MiniVal240 and PairedVal20;
- solve ordinary runtime/integration bugs without changing the experiment.

DSH MUST NOT:
- train or fine-tune any model;
- retrain YOLO;
- retrain SAM2;
- retrain ProgramHead/B3/ranker/quality estimator;
- use the ProposalSetRanker or ProposalQualityEstimator in the primary Task 6X resolver;
- change U-C1;
- add a SAM-score threshold/filter;
- tune SAM2 quality-score thresholds;
- add TTA/tiling/super-resolution;
- change Task 6Q eligibility thresholds;
- add nearest/L3 execution;
- access test split;
- choose Task 6Y.

If any prohibited change is needed, STOP and report.

---

# PART A — Record Task 6W findings

## 1. Task 6W result

Record in `docs/task6x_sam2_proposal_refinement.md`:

Task 6W verdict:
`QUALITY_FILTER_NOT_HELPFUL`

Important measurements:

### W0 oracle-quality mechanism
On train-only U-Calib200:

- deterministic U-C1 reference mIoU = `0.4789276`
- oracle q>=0.50 filter mIoU = `0.6575339`
- smallest mIoU = `0.3544718 → 0.6707174`
- selection wrong = `76 → 23`
- abstention = 0

Therefore the abstract mechanism “remove invalid proposals before extreme-area selection” is valid.

### Learned quality estimator
- holdout AUROC = `0.8438013`
- F1@0.50 = `0.8251182`
- but RefVal reference mIoU = `0.3793127`, below W-S0 `0.4289355`
- selection wrong = `65`, worse than W-S0 `53`
- downstream answered mIoU = `0.2745576`, below W-S0 `0.3141364`

Therefore ProposalQualityEstimator v0.1 is not part of the primary resolver.

Do not modify Task 6W artifacts.

---

# PART B — Frozen assets

## 2. Read-only assets

Freeze:

- all Task 6U/6V/6W artifacts;
- U-C1 proposal configuration/cache;
- Task 6M.1 YOLO checkpoint;
- Task 6T hardened ProgramHead checkpoint;
- Task 6Q eligibility and deterministic extreme selector semantics;
- Task 6O B3 checkpoint;
- GeometricRelationField v0.2;
- WHU-EA-NativeVector v1.0;
- BuildSpatialReason v0.2;
- U-Calib200;
- RefValUnique;
- MiniVal240;
- PairedVal20.

No test split.

## 3. Frozen SAM2.1

Use exactly the project's existing frozen SAM2.1 Hiera Base+:

Checkpoint:
`local_cache/models/sam2.1_hiera_base_plus.pt`

Expected SHA256:
`a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`

Config:
`configs/sam2.1/sam2.1_hiera_b+.yaml`

Use official package path already installed locally:
`sam2.sam2_image_predictor.SAM2ImagePredictor`

No new checkpoint/download/install.

Before evaluation:
- verify checkpoint SHA256;
- verify predictor loads;
- verify `predict(...)` returns masks + predicted mask-quality scores.

If unavailable/mismatch:
STOP with `SAM2_PREDICTOR_UNAVAILABLE`.

---

# PART C — Fixed U-C1 proposal source

## 4. Proposal generation

Use exactly:

```text
YOLO26m-seg
checkpoint SHA256 =
ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474

imgsz = 640
conf = 0.05
max_det = 300
default NMS
no TTA
no tiling
source tile = 512×512
```

Masks and boxes must be in exact source 512×512 coordinates.

Reuse Task 6U/6W U-C1 caches where possible.

---

# PART D — Four predeclared refinement options

## 5. Common pre-filter

For every family query, begin with the exact Task 6Q eligibility rules **on the original U-C1 YOLO proposal**.

Largest:
- not border-touching;
- bbox extent ratio <= 0.20.

Smallest:
- not border-touching;
- bbox extent ratio <= 0.20;
- area >= 150 px.

Only original family-eligible YOLO proposals are sent to SAM2 refinement.

No GT is used here.

## 6. X-C0 baseline

No SAM2 refinement.

Use:
- original U-C1 YOLO proposal masks;
- exact deterministic largest/smallest selection.

This must reproduce Task 6U U-S1.

## 7. X-C1 — exact box, single-mask SAM2

For each eligible YOLO proposal:

- prompt box = exact YOLO proposal bbox in XYXY source pixels;
- `point_coords=None`
- `point_labels=None`
- `mask_input=None`
- `multimask_output=False`
- `return_logits=False`

SAM2 output mask becomes the refined candidate.

Record returned SAM2 predicted quality score but do not threshold it.

## 8. X-C2 — exact box, multimask SAM2

For each eligible YOLO proposal:

- prompt box = exact YOLO bbox;
- no points;
- no mask input;
- `multimask_output=True`;
- `return_logits=False`.

SAM2 returns multiple masks plus predicted quality scores.

Choose exactly the mask with maximum SAM2 predicted quality score.

No score threshold.

## 9. X-C3 — 10% expanded box, multimask SAM2

For each eligible YOLO proposal:

Original box:
`(x1,y1,x2,y2)`

Let:
```text
w = x2 - x1
h = y2 - y1
```

Expanded box:
```text
x1' = x1 - 0.10*w
y1' = y1 - 0.10*h
x2' = x2 + 0.10*w
y2' = y2 + 0.10*h
```

Clip to:
`[0,511]`

Use:
- expanded box only;
- no points;
- no mask input;
- `multimask_output=True`;
- choose maximum SAM2 predicted quality score.

No other expansion factor.

## 10. Refined candidate normalization

For X-C1/X-C2/X-C3:

- convert SAM2 output to 512×512 boolean mask;
- compute refined:
  - area_px
  - bbox
  - bbox_area
  - bbox_extent_ratio
  - border touch.

Then apply the exact Task 6Q family eligibility **again on the refined mask**.

If refinement produces an empty mask:
- candidate is invalid and removed.

No duplicate-mask suppression.
No IoU-based deduplication.
No SAM quality threshold.

After refined eligibility:
- largest = max refined area
- smallest = min refined area
- tie-break:
  1. higher SAM2 predicted quality score;
  2. higher original YOLO confidence;
  3. lower original YOLO proposal index.

For X-C0 preserve Task 6Q original tie-break exactly.

---

# PART E — Train-only calibration and option freeze

## 11. Use exact U-Calib200 only

Evaluate X-C0/X-C1/X-C2/X-C3 on frozen U-Calib200.

RefValUnique, MiniVal240 and PairedVal20 may NOT affect option selection.

Report for every option:

- selected-reference mIoU
- Dice
- Pr@0.5
- centroid mean/median/p90
- abstention rate
- largest mIoU / Pr@0.5
- smallest mIoU / Pr@0.5
- NO_PROPOSALS
- NO_ELIGIBLE_PROPOSALS
- REFINEMENT_EMPTY
- REFERENCE_NOT_COVERED_IOU50
- REFERENCE_SELECTION_WRONG
- SELECTED_MASK_GEOMETRY_POOR
- REFERENCE_OK
- mean SAM2 mask-quality score
- mean SAM2 decoder calls per tile
- mean wall time/tile
- peak VRAM.

GT is evaluation only.

Write:
`evaluation/task6x_calibration_refinement.json`

## 12. Freeze exactly one option

Select using this exact priority order:

1. highest selected-reference mIoU
2. highest Pr@0.5
3. lowest `REFERENCE_SELECTION_WRONG`
4. highest `REFERENCE_OK`
5. lowest abstention rate
6. lower mean wall time/tile
7. simpler option:
   - X-C0
   - X-C1
   - X-C2
   - X-C3

Write:
`evaluation/task6x_frozen_refinement_option.json`

After this file exists, do not change the option.

### 12.1 Calibration stop condition

If X-C0 is selected:
- verdict path becomes `SAM2_REFINEMENT_NOT_HELPFUL`;
- still create docs/verdict/tests;
- do NOT run RefVal/MiniVal/Paired as a new primary system because calibration already rejects the mechanism.

If X-C1/X-C2/X-C3 is selected:
continue.

---

# PART F — Untouched RefValUnique evaluation

## 13. Compare baseline and frozen refinement

If a SAM refinement option was selected, evaluate on exact RefValUnique:

### X-S0
X-C0 baseline

### X-SR
Frozen selected SAM2 refinement option

Report:
- selected-reference mIoU
- Dice
- Pr@0.5
- centroid mean/median/p90
- area ratio median
- abstention rate
- largest/smallest metrics
- all failure buckets from section 11.

Also report:
- delta SR - S0
- fraction where SAM refinement increases GT-reference IoU
- fraction where it decreases GT-reference IoU
- mean IoU change among answered records
- SAM predicted quality score vs actual refined-mask q_gt Pearson/Spearman correlation
  - diagnostic only;
  - no threshold tuning.

Write:
`evaluation/task6x_refval_refinement.json`

---

# PART G — Downstream causal evaluation

## 14. MiniVal240

Use canonical program ids; no parser in the main causal comparison.

Compare:
- X-S0
- X-SR

Pipeline:

```text
canonical program
→ family/relation
→ reference resolver
→ GeometricRelationField v0.2
→ frozen SAM2 visual feature
→ frozen B3
→ target mask
```

Important:
The SAM2 used to refine proposal references is the same frozen checkpoint, but the target B3 visual path remains unchanged.

Report:
- strict all-240 mIoU/Dice
- answered-only mIoU/Dice
- Pr@0.5
- abstentions
- reference-fail count
- target-fail-with-reference-ok
- largest/smallest
- per direction
- border/tiny targets
- total runtime/tile.

Write:
`evaluation/task6x_downstream_minival240.json`

## 15. PairedVal20

Report:
- pass /20
- mean own IoU
- mean cross IoU
- own-cross margin
- reference abstention pairs.

Write:
`evaluation/task6x_downstream_pairedval20.json`

---

# PART H — Natural-language integration

## 16. Hardened ProgramHead integration

Run:

```text
Task 6T hardened ProgramHead
→ frozen selected Task 6X resolver
→ GeometricRelationField v0.2
→ frozen SAM2 visual feature
→ B3
```

On exact MiniVal240 natural-language queries.

Require parser:
- 240/240.

Report:
- strict mIoU
- answered-only mIoU
- PairedVal
- margin
- abstentions.

Do not execute nearest/L3.

Write:
`evaluation/task6x_hardened_parser_integration.json`

---

# PART I — Predeclared gates

## 17. Reference refinement improvement

If a SAM option was selected, define:

`sam_refinement_improved = true`

iff ALL:

- RefVal selected-reference mIoU >= X-S0 + `0.03`
- RefVal `REFERENCE_SELECTION_WRONG` <= X-S0 - `8`
- RefVal `REFERENCE_OK` >= X-S0 + `8`
- RefVal abstention rate <= `0.08`

## 18. Directional hardening gate

`SAM2_REFERENCE_REFINEMENT_PASS` requires ALL:

1. SAM refinement selected over X-C0 on U-Calib200
2. `sam_refinement_improved=true`
3. RefVal selected-reference mIoU >= `0.46`
4. RefVal centroid median <= `0.03`
5. RefVal centroid p90 <= `0.32`
6. MiniVal answered-only target mIoU >= `0.325`
7. MiniVal strict target mIoU >= `0.305`
8. PairedVal >= `12/20`
9. own-cross margin >= `0.30`
10. MiniVal reference-fail count <= `105`
11. parser integration 240/240
12. no test / no GT inference.

Do not change gates.

---

# PART J — Verdict

## 19. Exactly one, priority order

1. `INVALID_EXPERIMENT`
   - test use, GT inference, post-calibration option change, frozen-module mutation, protocol violation.

2. `SAM2_PREDICTOR_UNAVAILABLE`

3. `SAM2_REFINEMENT_NOT_HELPFUL`
   - calibration selects X-C0.

4. `SAM2_REFINEMENT_GENERALIZATION_FAIL`
   - a SAM option wins calibration but `sam_refinement_improved=false` on RefVal.

5. `SAM2_REFERENCE_REFINEMENT_PARTIAL`
   - RefVal improves, but one or more section-18 gates fail.

6. `SAM2_REFERENCE_REFINEMENT_PASS`
   - all section-18 gates pass.

No other verdict.

---

# PART K — Interpretation boundary

DSH may report measurements only.

Do NOT:
- claim SAM2 refinement as project novelty;
- tune a SAM quality-score threshold;
- add points/mask prompts beyond the fixed options;
- add another box expansion;
- combine SAM score with YOLO score in a learned or hand-tuned formula;
- retrain SAM2/YOLO;
- revisit ProposalSetRanker/ProposalQualityEstimator;
- change field/B3;
- start nearest/L3.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 6X 的 SAM2 refinement 结果决定是否冻结 reference subsystem 并进入 nearest/L3；不自行继续增加 reference 模块。`

---

# PART L — Required artifacts

Create at minimum:

```text
buildreasonseg_mvp/task6x_sam2_reference_refiner.py

evaluation/task6x_frozen_asset_audit.json
evaluation/task6x_calibration_refinement.json
evaluation/task6x_frozen_refinement_option.json

# Only if X-C1/C2/C3 wins calibration:
evaluation/task6x_refval_refinement.json
evaluation/task6x_downstream_minival240.json
evaluation/task6x_downstream_pairedval20.json
evaluation/task6x_hardened_parser_integration.json

evaluation/task6x_verdict.json

docs/task6x_sam2_proposal_refinement.md

scripts/task6x_calibrate_refinement.py
scripts/task6x_evaluate_reference.py
scripts/task6x_evaluate_downstream.py
scripts/task6x_report.py
```

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

No new model checkpoint.

---

# PART M — Tests

Task 6W ended at:
`929 passed, 1 skipped`

Add tests for at least:

1. Task 6W artifacts unchanged
2. SAM2 checkpoint SHA exact
3. official SAM2ImagePredictor path used
4. no SAM2 training
5. U-C1 exact
6. exactly four options X-C0..X-C3
7. no quality threshold
8. X-C1 exact box
9. X-C1 multimask false
10. X-C2 exact box
11. X-C2 multimask true
12. X-C2 max SAM quality mask
13. X-C3 expansion exactly 10%
14. X-C3 clipped to source image
15. no point prompt
16. no mask input
17. original Task 6Q family eligibility before refinement
18. refined family eligibility after refinement
19. refined mask 512×512
20. no duplicate suppression
21. largest chooses max refined area
22. smallest chooses min refined area
23. tie-break exact
24. calibration uses only U-Calib200
25. RefVal not used to choose option
26. option frozen before RefVal
27. X-C0 calibration stop path works
28. no ProposalSetRanker in X-SR
29. no ProposalQualityEstimator in X-SR
30. no GT in inference
31. canonical-program downstream comparison has no parser
32. final integration uses hardened ProgramHead
33. parser remains 240/240
34. field v0.2 unchanged
35. B3 unchanged
36. no YOLO retraining
37. no SAM2 score threshold/tuning
38. no TTA/tiling
39. no GRCL
40. no nearest/L3
41. no test split
42. no new dataset/download/install/GUI
43. previous suite preserved.

Run:

`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART N — Git/storage

Do not commit:
- existing YOLO/SAM2/B3/parser/ranker/quality checkpoints;
- proposal/refinement caches;
- source imagery/vectors;
- `.conda`.

Commit:
- refiner code;
- small JSON artifacts;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:

1. `feat: add frozen SAM2 proposal refinement`
2. `eval: audit SAM2-refined reference selection`
3. optional docs commit

---

# PART O — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for a genuine SAM2/runtime/integration bug.

No downloads or installs.

---

# PART P — STOP

After Task 6X:
- commit;
- push;
- handoff;
- STOP.

Do NOT:
- add more reference modules;
- tune SAM quality scores;
- retrain SAM2/YOLO;
- change U-C1;
- modify field/B3;
- revisit GRCL;
- start nearest/L3 without ChatGPT audit;
- access test;
- build GUI.

Wait for ChatGPT audit.
