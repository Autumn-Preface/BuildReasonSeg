# TO_DSH — Task 6W: Proposal-Quality Filtering + Semantic Extreme Selection

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `3e185fbc5764dcc74d34d70f085e4a139a93bf37`
>
> Predecessor: Task 6V → `FAMILY_POLICY_NOT_BETTER`
>
> Research decision already made by ChatGPT:
>
> 1. Keep the core BuildReasonSeg method frozen:
>    `ProgramHead → Reference → GeometricRelationField v0.2 → frozen SAM2 visual feature → B3 target decoder`.
> 2. Keep Task 6U **U-C1** as the current high-recall proposal configuration:
>    `imgsz=640, conf=0.05, max_det=300`.
> 3. Do not use Task 6U ProposalSetRanker v0.1 in the primary resolver.
> 4. Do not use the Task 6V family-conditioned policy as the primary resolver; it did not beat U-S1 enough and sacrificed smallest-family coverage.
> 5. Task 6W tests a different hypothesis:
>
>    **Many selection errors are caused by fragmented / merged / incomplete proposals whose predicted area corrupts the literal largest/smallest rule. If low-quality proposals can be filtered first, deterministic largest/smallest semantics may become reliable again.**
>
> 6. This proposal-quality module is support infrastructure, NOT a claimed algorithmic novelty.
> 7. DSH is an executor. Do not redesign the mechanism.

All user-facing DSH output must be Chinese.

---

# 0. DSH role

DSH MAY:
- reuse frozen U-C1 proposal caches or regenerate them with the exact frozen configuration;
- run the predeclared oracle-quality-filter diagnostic;
- build train-only proposal-quality labels;
- implement and train the exact ProposalQualityEstimator v0.1 below;
- evaluate the exact resolver and downstream chain;
- solve ordinary runtime bugs without changing the protocol.

DSH MUST NOT:
- retrain YOLO;
- change YOLO weights/config beyond frozen U-C1;
- use U-C0/U-C2/U-C3 as alternatives;
- retrain ProgramHead;
- retrain ProposalSetRanker;
- use ProposalSetRanker in the primary Task 6W resolver;
- retrain SAM2/B3;
- change GeometricRelationField v0.2;
- change Task 6Q family eligibility rules;
- tune the quality threshold;
- add TTA/tiling/super-resolution;
- add nearest/L3 execution;
- use test split;
- invent another quality architecture;
- choose Task 6X.

If a prohibited change is needed, STOP and report.

---

# PART A — Task 6V audit record

## 1. Record the frozen Task 6V result

Copy into `docs/task6w_proposal_quality_filter.md`:

Task 6V verdict:
`FAMILY_POLICY_NOT_BETTER`

Frozen family policy:
- largest → V-P2 (U-C1 + frozen ranker)
- smallest → V-P0 (U-C0 + deterministic)

RefValUnique:
- family-policy mIoU = 0.4383028
- U-S1 mIoU = 0.4289355
- delta = +0.0093674 < +0.015 gate
- REFERENCE_OK = 114 < 117 gate
- REFERENCE_SELECTION_WRONG = 41
- NOT_COVERED = 59
- downstream answered mIoU = 0.3069046
- strict mIoU = 0.3005107
- paired = 10/20
- margin = +0.287855
- reference-fail = 116

Important interpretation:
- family routing slightly improves some reference metrics;
- it does not solve the dominant reference bottleneck;
- U-C1 remains valuable because Task 6U demonstrated better candidate coverage;
- ProposalSetRanker is not accepted as a global selector.

Do not modify Task 6V artifacts.

---

# PART B — Frozen assets

## 2. Read-only

Freeze:

- all Task 6U and 6V artifacts;
- U-C1 proposal configuration and caches;
- Task 6M.1 YOLO checkpoint;
- Task 6T hardened ProgramHead checkpoint;
- Task 6Q deterministic family eligibility;
- Task 6O B3 checkpoint;
- GeometricRelationField v0.2;
- frozen SAM2.1 Hiera Base+ feature cache/path;
- Task 6U U-Calib200;
- Task 6U U-RankerTrain;
- Task 6P RefValUnique;
- Task 6N MiniVal240;
- Task 6N PairedVal20;
- WHU-EA-NativeVector v1.0;
- BuildSpatialReason v0.2.

No test split.

## 3. Frozen U-C1

Use only:

```text
YOLO26m-seg checkpoint:
ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474

imgsz = 640
conf = 0.05
max_det = 300
default NMS
TTA = false
tiling = false
source image = original 512×512
```

Masks restored to exact 512×512 source coordinates.

---

# PART C — W0 Oracle quality-filter mechanism diagnostic

## 4. Purpose

Before training a quality estimator, test whether **perfect knowledge of proposal quality** would make the quality-filter + deterministic-area mechanism useful.

Use **U-Calib200 only**.

Do not use RefVal for this feasibility gate.

## 5. Ground-truth proposal quality

For every U-C1 proposal on a U-Calib200 tile:

```text
q_gt = max IoU(proposal_mask, every native GT building instance on that tile)
```

This is diagnostic/training metadata only.

No GT is allowed in inference.

## 6. Oracle quality filter

For a reference query:

1. apply the exact Task 6Q family eligibility rules first;
2. keep only proposals with `q_gt >= 0.50`;
3. if none remain → abstain;
4. largest → select maximum proposal area;
5. smallest → select minimum proposal area;
6. tie-break:
   - higher YOLO confidence;
   - lower original proposal index.

Compare with U-C1 deterministic baseline on the same U-Calib200.

Report overall/largest/smallest:
- selected-reference mIoU;
- Dice;
- Pr@0.5;
- centroid median/p90;
- abstention;
- NOT_COVERED;
- SELECTION_WRONG;
- REFERENCE_OK.

Write:
`evaluation/task6w_oracle_quality_filter_calib.json`

## 7. W0 mechanism gate

Continue to training only if ALL:

- oracle-quality-filter overall selected-reference mIoU >= U-C1 deterministic + `0.08`;
- oracle-quality-filter smallest mIoU >= U-C1 deterministic smallest + `0.10`;
- oracle-quality-filter `REFERENCE_SELECTION_WRONG` <= `0.60 * baseline`;
- oracle-quality-filter abstention rate <= `0.10`.

If any fails:
STOP with:
`QUALITY_FILTER_MECHANISM_INSUFFICIENT`

Do not train a quality estimator.

---

# PART D — Proposal-quality training dataset

## 8. Training tiles

Use only U-RankerTrain from Task 6U.

Deduplicate by tile id:
- run/read U-C1 proposals once per unique train tile;
- never duplicate a proposal because several reference records share the same tile.

No U-Calib200 tile.
No RefVal tile.
No MiniVal/PairedVal-specific evaluation record for optimization.
No test.

## 9. Candidate population

For proposal-quality training, use the union/common structural eligibility:

- proposal does NOT touch source-image border;
- bbox extent ratio <= 0.20.

Do NOT apply the smallest `area >= 150` condition to quality training.

Reason:
proposal quality is family-independent; family-specific eligibility remains a later resolver rule.

## 10. Binary label

For each training proposal:

```text
q_gt = max IoU(proposal_mask, any native GT building instance on the tile)
y_quality = 1 if q_gt >= 0.50 else 0
```

Store q_gt for analysis, but train on the binary label.

GT is never a model input.

Write local/generated dataset under:
`artifacts/task6w/quality_dataset/`

Write tracked manifest:
`evaluation/task6w_quality_dataset_manifest.json`

Must include:
- unique tiles;
- proposal count;
- positive/negative count;
- q_gt histogram;
- split hashes;
- no-overlap checks.

---

# PART E — Fixed ProposalQualityEstimator v0.1

## 11. Input feature design

Create:

`buildreasonseg_mvp/task6w_proposal_quality.py`

For each proposal, use exactly:

### 11.1 Eight geometry/confidence scalars

1. `confidence`
2. `log_area = log1p(area_px) / log1p(512*512)`
3. `area_ratio = area_px / (512*512)`
4. `bbox_extent_ratio`
5. `width_ratio = bbox_width / 512`
6. `height_ratio = bbox_height / 512`
7. `fill_ratio = area_px / max(1,bbox_area)`
8. `abs_log_aspect = abs(log((bbox_width+1)/(bbox_height+1)))`

No family.
No relation.
No area rank.
No image x/y coordinate.

### 11.2 Frozen SAM2 proposal appearance/context

Use the already-frozen SAM2 feature tensor:
`V ∈ R^(256×64×64)`

Downsample proposal mask to 64×64 using nearest-neighbor:
`M64`

Require at least one positive M64 cell. If none:
- mark proposal `feature_invalid_small`;
- quality estimator output is forced to 0 at inference;
- exclude it from quality-model training;
- count it explicitly.

Create one-cell ring:

```text
D = max_pool2d(M64, kernel=3, stride=1, padding=1)
Ring = clamp(D - M64, 0, 1)
```

Compute:
- `inside_mean`: channel-wise masked mean of V over M64 → 256 dims
- `ring_mean`: channel-wise masked mean of V over Ring → 256 dims

If Ring is empty:
- use zeros for `ring_mean`;
- record count.

Do not use GT visual features.

Final raw input:
- visual = 512 dims
- geometry = 8 dims

## 12. Exact network

### Visual branch

```text
Linear(512 → 64)
LayerNorm(64)
GELU
```

### Geometry branch

```text
Linear(8 → 16)
GELU
```

Concatenate:
`64 + 16 = 80`

Head:

```text
Linear(80 → 32)
GELU
Linear(32 → 1)
```

Output raw quality logit.

No attention.
No CNN.
No Transformer/GNN.
No other feature.

## 13. Loss

Binary label from section 10.

Use:
`BCEWithLogitsLoss(pos_weight = N_negative / max(1,N_positive))`

Compute pos_weight only from the training split.

No regression loss.
No IoU loss.
No ranking loss.

At inference:

```text
quality_prob = sigmoid(logit)
keep iff quality_prob >= 0.50
```

Threshold **0.50 is frozen**.

No threshold sweep/calibration.

---

# PART F — Training protocol

## 14. Train/holdout split

Split by unique tile id before proposal rows are assigned.

Deterministic:
- seed `20260930`
- 80% train tiles
- 20% internal holdout tiles
- hash-based split.

No tile may occur in both.

## 15. Optimization

Use:

- AdamW
- lr = `5e-4`
- weight_decay = `1e-4`
- batch = `256 proposals`
- max epochs = `40`
- early stopping patience = `5`
- seed = `20260930`
- AMP allowed
- no augmentation
- no scheduler

Checkpoint selection:
1. highest internal-holdout AUROC;
2. tie-break highest internal-holdout F1 at threshold 0.50;
3. tie-break lower BCE.

No hyperparameter sweep.

Checkpoint:
`artifacts/checkpoints/task6w/proposal_quality_v01.pt`
gitignored.

Report:
- train/holdout tiles;
- proposal class counts;
- parameter count;
- AUROC;
- AUPRC;
- accuracy;
- precision/recall/F1 at 0.50;
- confusion matrix;
- BCE;
- selected epoch;
- peak VRAM;
- wall time;
- checkpoint SHA256.

Write:
`evaluation/task6w_quality_training.json`

---

# PART G — Resolver v0.1

## 16. Quality-filtered deterministic resolver

Create:

`buildreasonseg_mvp/task6w_quality_reference_resolver.py`

Inference:

```text
U-C1 proposals
→ Task 6Q family eligibility
→ ProposalQualityEstimator
→ keep quality_prob >= 0.50
→ largest: max area
   smallest: min area
→ Task 6Q confidence/index tie-break
```

If all family-eligible proposals are filtered:
- explicit abstention
- reason = `no_quality_eligible_proposals`

No fallback to an unfiltered proposal.

No ProposalSetRanker.

---

# PART H — Untouched RefValUnique evaluation

## 17. Compare exactly three diagnostics

On RefValUnique:

### W-S0
Task 6U U-S1:
`U-C1 + deterministic selector`

### W-SQ
`U-C1 + learned quality filter + deterministic selector`

### W-ORACLE
`U-C1 + oracle q_gt>=0.50 filter + deterministic selector`

W-ORACLE is evaluation ceiling only; never inference.

Report for each:
- selected-reference mIoU
- Dice
- Pr@0.5
- centroid mean/median/p90
- area-ratio median
- abstention rate
- largest/smallest metrics
- failure buckets:
  - NO_PROPOSALS
  - NO_ELIGIBLE_PROPOSALS
  - QUALITY_FILTER_ALL_REJECTED
  - REFERENCE_NOT_COVERED_IOU50
  - REFERENCE_SELECTION_WRONG
  - SELECTED_MASK_GEOMETRY_POOR
  - REFERENCE_OK

Write:
`evaluation/task6w_refval_quality_filter.json`

---

# PART I — Downstream causal evaluation

## 18. MiniVal240

Use canonical program ids; no parser in the main comparison.

Compare:
- W-S0
- W-SQ

Pipeline:

```text
canonical program
→ family/relation
→ reference resolver
→ GeometricRelationField v0.2
→ frozen SAM2 feature
→ frozen B3
→ target mask
```

Report:
- strict all-240 mIoU/Dice
- answered-only mIoU/Dice
- Pr@0.5
- abstentions
- reference-fail count
- target-fail-with-reference-ok
- largest/smallest
- per direction
- border/tiny targets.

Write:
`evaluation/task6w_downstream_minival240.json`

## 19. PairedVal20

Report W-S0 and W-SQ:
- pass /20
- own IoU
- cross IoU
- margin
- reference-abstention pairs.

Write:
`evaluation/task6w_downstream_pairedval20.json`

---

# PART J — Hardened natural-language integration

## 20. One regression/integration run

Use:
Task 6T hardened ProgramHead → W-SQ → field v0.2 → SAM2 → B3

On exact MiniVal240 queries.

Require parser:
- 240/240.

Report:
- strict mIoU
- answered-only mIoU
- paired
- margin
- abstentions.

Do not run nearest/L3.

Write:
`evaluation/task6w_hardened_parser_integration.json`

---

# PART K — Predeclared gates

## 21. Quality-estimator adequacy

Pass if:
- internal-holdout AUROC >= `0.80`
- internal-holdout F1@0.50 >= `0.65`
- no train/holdout tile overlap.

This is diagnostic; downstream gates remain decisive.

## 22. Reference hardening

`quality_reference_improved=true` iff W-SQ vs W-S0 satisfies ALL:

- RefVal mIoU >= W-S0 + `0.04`
- `REFERENCE_SELECTION_WRONG` <= `0.75 * W-S0`
- `REFERENCE_OK` >= W-S0 + `8`
- abstention rate <= `0.10`

## 23. Downstream hardening

`PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS` requires ALL:

1. W0 oracle mechanism gate passed
2. quality-estimator adequacy passed
3. RefVal W-SQ mIoU >= `0.48`
4. RefVal centroid median <= `0.03`
5. RefVal centroid p90 <= `0.28`
6. MiniVal answered-only target mIoU >= `0.33`
7. MiniVal strict target mIoU >= `0.31`
8. PairedVal >= `12/20`
9. own-cross margin >= `0.30`
10. MiniVal reference-fail count <= `100`
11. parser integration 240/240
12. no test / no GT inference.

No threshold changes after results.

---

# PART L — Verdict

## 24. Exactly one, priority order

1. `INVALID_EXPERIMENT`
2. `FROZEN_ASSET_UNAVAILABLE`
3. `QUALITY_FILTER_MECHANISM_INSUFFICIENT`
   - W0 oracle-quality gate fails; no estimator training.
4. `QUALITY_ESTIMATOR_NOT_LEARNABLE`
   - W0 passes, but section 21 adequacy fails.
5. `QUALITY_FILTER_NOT_HELPFUL`
   - estimator adequacy passes but `quality_reference_improved=false` and downstream hardening fails.
6. `QUALITY_REFERENCE_HARDENING_PARTIAL`
   - measurable reference/downstream improvement but one or more section-23 gates fail.
7. `PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS`
   - all section-23 gates pass.

No other verdict.

---

# PART M — Interpretation boundary

DSH may report measurements only.

Do NOT:
- claim ProposalQualityEstimator as novelty;
- alter threshold 0.50;
- add family-specific quality networks;
- add ranker after quality filtering;
- train a size estimator;
- retrain YOLO;
- add TTA/tiling;
- change U-C1;
- change field/B3;
- start nearest/L3.

Final recommendation exactly:

`等待 ChatGPT 根据 Task 6W 的 oracle-quality ceiling、learned quality filter 与 downstream 结果决定 reference 是否继续硬化，不自行增加 ranker、size estimator 或 detector 改动。`

---

# PART N — Required artifacts

Create at minimum:

```text
buildreasonseg_mvp/task6w_proposal_quality.py
buildreasonseg_mvp/task6w_quality_reference_resolver.py

evaluation/task6w_oracle_quality_filter_calib.json
evaluation/task6w_quality_dataset_manifest.json
evaluation/task6w_quality_training.json
evaluation/task6w_refval_quality_filter.json
evaluation/task6w_downstream_minival240.json
evaluation/task6w_downstream_pairedval20.json
evaluation/task6w_hardened_parser_integration.json
evaluation/task6w_verdict.json

docs/task6w_proposal_quality_filter.md

scripts/task6w_oracle_quality_diagnostic.py
scripts/task6w_build_quality_dataset.py
scripts/task6w_train_quality.py
scripts/task6w_evaluate_reference.py
scripts/task6w_evaluate_downstream.py
scripts/task6w_report.py
```

If W0 stops the experiment:
- create W0 artifact, verdict, docs/handoff/tests applicable to the stopped path;
- do not fabricate downstream/training files.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

---

# PART O — Tests

Task 6V ended at:
`881 passed, 1 skipped`

Add tests for at least:

1. Task 6V artifacts unchanged
2. U-C1 exact config
3. YOLO checkpoint hash exact
4. no U-C0/C2/C3 alternative in primary resolver
5. W0 uses U-Calib200 only
6. q_gt definition exact max IoU
7. W0 threshold exact 0.50
8. W0 gate exact
9. U-RankerTrain only for quality dataset
10. proposal dataset deduplicated by tile
11. no U-Calib tile in quality training
12. no RefVal tile in quality training
13. train/holdout split by tile
14. common eligibility exact
15. quality label threshold exact 0.50
16. geometry feature dimension exactly 8
17. no family feature
18. no relation feature
19. no proposal rank feature
20. no x/y location feature
21. SAM2 feature 256×64×64
22. proposal mask nearest-downsample
23. ring construction exact 3×3 dilation-minus-mask
24. inside pooled feature 256
25. ring pooled feature 256
26. quality architecture exact
27. BCE pos_weight computed train-only
28. inference quality threshold fixed 0.50
29. no threshold sweep
30. no ranker in W-SQ
31. family eligibility before quality filter
32. deterministic area semantics after filter
33. explicit abstention if all rejected
34. RefVal untouched by training
35. MiniVal240 exact reuse
36. PairedVal20 exact reuse
37. canonical-program causal comparison has no parser
38. final integration uses hardened ProgramHead
39. parser stays 240/240
40. field v0.2 unchanged
41. B3 unchanged
42. no YOLO retraining
43. no TTA/tiling
44. no GRCL
45. no nearest/L3
46. no test
47. no new dataset/download/install/GUI
48. previous suite preserved.

Run:

`python -m pytest tests/ -q`

Do not reduce previous passing tests.

---

# PART P — Git/storage

Do not commit:
- quality checkpoint;
- YOLO/SAM2/B3/parser/ranker weights;
- proposal/feature caches;
- generated proposal-quality row data if large;
- source imagery/vectors;
- `.conda`.

Commit:
- quality model/resolver code;
- dataset manifest;
- small JSON eval artifacts;
- scripts;
- tests;
- docs;
- handoff.

Suggested commits:
1. `feat: add proposal-quality reference filter`
2. `eval: isolate proposal quality and extreme selection`
3. optional docs commit.

---

# PART Q — DSH model policy

Default:
- **DeepSeek V4.1 Flash + High**

Use Max only for a genuine runtime/data-pipeline bug.

No downloads or installs.

---

# PART R — STOP

After Task 6W:
- commit;
- push;
- handoff;
- STOP.

Do NOT:
- train another reference model;
- add a ranker after quality filtering;
- add size correction;
- change detector/config/thresholds;
- retrain YOLO;
- change field/B3/SAM2;
- revisit GRCL;
- add nearest/L3;
- access test;
- build GUI.

Wait for ChatGPT audit.
