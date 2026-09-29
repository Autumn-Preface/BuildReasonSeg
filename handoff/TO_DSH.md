# TO_DSH — Task 6P: Differentiable Relation Field + Predicted-Reference Substitution

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `595e7bb4867a6a91dd6f3cd8e0671e7862b4c6b8`
>
> Predecessor: Task 6O → `FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`
>
> This task is a tightly specified execution task. ChatGPT has already decided the research question, architecture, data, gates and verdicts. DSH must not redesign them.

## 0. Execution role

All user-facing DSH output must be Chinese.

DSH MAY:
- implement the exact v0.2 differentiable field specified here;
- implement the exact reference-mask head specified here;
- train/evaluate using the exact frozen packs and protocols;
- solve ordinary implementation/runtime bugs that do not change the experiment.

DSH MUST NOT:
- change the visual backbone;
- alter Task 6N/6O artifacts;
- alter GeometricRelationField v0.1;
- introduce a new field formula;
- add `[REF]`;
- add GRCL/SCL;
- add nearest/L3;
- add graph reasoning;
- add a dataset;
- access test split;
- optimize YOLO;
- jointly train reference and target heads;
- retrain B3;
- add MLLM hidden-state conditioning;
- choose Task 6Q.

If any such change is needed, STOP and report.

# PART A — Research question

## 1. What Task 6P must answer

Task 6O established:

- frozen B2 MiniVal240 mIoU = `0.4531265609993713`
- B3 (visual + field + relation, no direct reference channel) mIoU = `0.4299680351479113`
- B3 PairedVal20 = `14/20`
- B3 own-cross margin = `0.39719566349802166`
- geometry-only B4 mIoU = `0.046637104127446545`

Therefore the preferred directional target decoder is B3.

Task 6P tests:

> Can the oracle reference mask be replaced by a learned predicted reference mask while preserving enough of the field-guided target-segmentation signal?

Task 6P is NOT yet language-to-mask end-to-end, joint reference-target training, or a final MLLM architecture.

The reference family id (`largest` or `smallest`) is taken from the canonical program record to isolate reference grounding.

# PART B — Critical implementation correction

## 2. GeometricRelationField v0.1 must remain frozen

Do not modify:
`buildreasonseg_mvp/geometric_relation_field.py`

Task 6N/6O remain valid oracle-reference experiments.

Record in Task 6P documentation:

- v0.1 calls `.detach()` on tensor masks;
- v0.1 converts the centroid to Python `float`;
- therefore v0.1 is numerically smooth as a spatial function but is NOT differentiable with respect to the input reference mask.

This did not invalidate Task 6N/6O because the reference mask was oracle/frozen and no gradient to the reference source was required.

Future joint predicted-reference training requires a gradient-preserving implementation.

## 3. Implement GeometricRelationField v0.2

Create:
`buildreasonseg_mvp/geometric_relation_field_v02.py`

It must be numerically equivalent to v0.1 for the same binary mask/relation/size while preserving autograd from `P_rel` to soft `M_ref`.

### 3.1 Input

Accept:
- `mask_ref`: torch tensor, shape `(H,W)`, `(B,1,H,W)` or `(B,H,W)`;
- relation id(s): left/right/above/below;
- output size `(h,w)`.

Do not detach.
Do not convert centroid tensors to Python float.
Do not call NumPy inside the forward path.

### 3.2 Soft centroid

For each batch element:

```text
mass = sum(mask_ref) + eps
cx = sum(mask_ref * x_grid) / mass
cy = sum(mask_ref * y_grid) / mass
```

Use `eps = 1e-6`.

Pixel-center normalized coordinates:
- x = `(col + 0.5) / W`
- y = `(row + 0.5) / H`

### 3.3 Field formula

Exactly v0.1:

```text
alpha = 1.2
tau = 0.04
s_axis = 0.02
s_margin = 0.02

dx = x - cx
dy = y - cy
ax = abs(dx)
ay = abs(dy)
```

Horizontal:
```text
axis_score = sigmoid((ax - alpha*ay)/s_axis)
margin_score = sigmoid((ax - tau)/s_margin)
```

Vertical:
```text
axis_score = sigmoid((ay - alpha*ax)/s_axis)
margin_score = sigmoid((ay - tau)/s_margin)
```

Sign:
```text
left_of  = sigmoid((-dx)/s_margin)
right_of = sigmoid(( dx)/s_margin)
above    = sigmoid((-dy)/s_margin)
below    = sigmoid(( dy)/s_margin)
```

```text
P_rel = clamp(sign_score * axis_score * margin_score, 0, 1)
```

No learned parameter.

### 3.4 v0.2 gates

Before training:

1. Deterministically select 64 binary oracle reference masks from frozen Task 6N MiniVal240, balanced over 4 directions.
2. Compare v0.1 vs v0.2 at 64×64.
3. Require:
   - max abs error <= `1e-6`
   - mean abs error <= `1e-7`

Gradient check:
- at least 8 deterministic non-binary soft masks with `requires_grad=True`;
- all four directions represented;
- fixed nonuniform spatial weighting tensor `W`;
- `scalar = sum(P_rel * W)`;
- backprop;
- gradient must exist, be finite, and have L1 norm > `1e-8`.

Write:
`evaluation/task6p_field_v02_audit.json`

If any gate fails:
STOP with `DIFFERENTIABLE_FIELD_INVALID`.

# PART C — Frozen assets

## 4. Freeze previous evidence

Read-only:
- all Task 6N tracked artifacts;
- all Task 6O tracked artifacts;
- Task 6O B3 checkpoint;
- Task 6N exact packs;
- v0.1 field file;
- `configs/spatial_relations_v1.yaml`;
- WHU native-vector v1.0;
- BuildSpatialReason v0.2;
- frozen SAM2.1 Hiera Base+ checkpoint/cache;
- all Task 6M.1 and earlier evidence.

No test split.

## 5. Verify B3

Read B3 checkpoint path/hash from Task 6O evaluation artifact.
Require local checkpoint exists and hash matches.

Reproduce B3 once on MiniVal240 using oracle field v0.2.

Require mIoU/Dice abs delta <= `1e-6` vs frozen Task 6O B3.

Write:
`evaluation/task6p_b3_reproduction.json`

If fail:
STOP `TASK6O_B3_REPRODUCTION_FAIL`.

# PART D — Reference-head dataset

## 6. Scope

Use only the same 8 directional L2 program ids from Task 6N/6O.

Reference family:
- `largest_to_*` → `largest`
- `smallest_to_*` → `smallest`

The target building must never enter reference-head input.

## 7. Deduplicate reference examples

Build unique references from the full eligible Task 6N scope.

Unique key:

```text
(split, tile_id, reference_source_feature_id, reference_family)
```

If several relations share the same reference, keep only one reference-training record.

Create:
- `RefTrainUnique`: all unique train references
- `RefValUnique`: all unique val references

No test.

Write hashes/counts:
`evaluation/task6p_reference_pack_manifest.json`

## 8. Reference Overfit20

Deterministically select 20 unique train references:
- 10 largest if available
- 10 smallest if available
- distinct `(tile, source_feature_id)`
- no duplicate reference mask

If one family has fewer than 10, fill from the other and report.

# PART E — ReferenceMaskHead v0.1

## 9. Architecture

Create:
`buildreasonseg_mvp/task6p_reference_head.py`

Input:
- frozen SAM2 feature `V`: 256×64×64
- reference family id: `{largest, smallest}`

No target relation id.
No target mask.
No relation field.
No YOLO proposal.
No candidate mask.

Visual projection:

```text
Conv1x1(256 → 128)
GroupNorm(8,128)
GELU
```

Family embedding:
- 2 embeddings
- dim 16

Broadcast over 64×64.

Fusion input = 144:

```text
Conv3x3(144 → 128, padding=1)
GroupNorm(8,128)
GELU
Conv3x3(128 → 64, padding=1)
GroupNorm(8,64)
GELU
Conv1x1(64 → 1)
```

Upsample reference logits bilinearly to 512×512.

Loss exactly:
`BCEWithLogitsLoss + DiceLoss`

No other modules.

# PART F — Reference-head training

## 10. Stage P1 — Overfit20

Use:
- AdamW
- lr `1e-3`
- weight_decay `1e-4`
- max steps `1200`
- batch `4`
- no scheduler
- no augmentation
- seed `20260929`
- same AMP policy as 6N/6O
- evaluate every 100 steps

P1 gate:
- train mIoU >= `0.85`
- train Dice >= `0.90`

If fail:
STOP with `REFERENCE_HEAD_NOT_LEARNABLE`.

## 11. Stage P2 — RefTrainUnique → RefValUnique

Only if P1 passes.

Fresh initialization.

Use:
- AdamW
- lr `3e-4`
- weight_decay `1e-4`
- batch `8`
- max epochs `30`
- early stopping patience `6` on RefValUnique mIoU
- seed `20260929`
- no augmentation
- no scheduler
- same AMP

Model selection:
highest RefValUnique mIoU.

Write:
`evaluation/task6p_reference_val.json`

Required:
- mIoU
- Dice
- Pr@0.5
- largest mIoU/Dice
- smallest mIoU/Dice
- soft-mask centroid error normalized by image diagonal:
  - mean
  - median
  - p90
- median `pred_area / gt_area`
- border/non-border mIoU
- tiny-reference mIoU if present
- parameter count
- best epoch
- wall time
- peak VRAM

Use sigmoid probability map for predicted centroid.

# PART G — Predicted-reference substitution

## 12. Freeze ReferenceMaskHead

After P2, freeze all reference-head params.
Do not jointly train it with B3.

## 13. Predicted field generation

For each frozen MiniVal240 record:

1. get frozen SAM2 feature;
2. use largest/smallest family id to predict soft reference map:
   `M_ref_pred = sigmoid(reference_logits_up512)`;
3. generate `P_rel_pred` with GeometricRelationField v0.2;
4. feed frozen B3:
   - frozen visual feature
   - `P_rel_pred`
   - direction embedding
5. predict target mask.

No oracle reference mask may enter the predicted-reference path.

GT reference is only for reference-head metrics.
GT target is only for target scoring.

Write:
`evaluation/task6p_predicted_reference_target_val.json`

## 14. Oracle-vs-predicted field diagnostics

On MiniVal240 compare predicted-reference field vs oracle-reference v0.2 field:

- MAE
- RMSE
- per-record Pearson correlation then mean
- predicted-vs-oracle reference centroid error
- per family
- per relation

Write:
`evaluation/task6p_field_propagation_diagnostics.json`

No threshold tuning.

# PART H — Downstream metrics

## 15. MiniVal240

Report predicted-reference B3:
- target mIoU
- Dice
- Pr@0.5
- per relation
- largest-reference family
- smallest-reference family
- border target
- tiny target if present

Frozen oracle B3 reference:
- mIoU `0.4299680351479113`
- PairedVal20 `14/20`
- own-cross margin `0.39719566349802166`

## 16. PairedVal20

Use exact frozen Task 6N PairedVal20.

For a pair with same image and same reference source, reuse the same predicted reference mask.

Report:
- pass / 20
- mean own IoU
- mean cross IoU
- own-cross margin

Write:
`evaluation/task6p_predicted_reference_paired_val.json`

# PART I — Gates

## 17. Reference-head adequacy

Pass only if all:

- RefValUnique mIoU >= `0.35`
- median normalized centroid error <= `0.05`
- p90 normalized centroid error <= `0.12`

These are propagation gates, not final segmentation-quality claims.

## 18. Predicted-reference chain retention

Pass only if all:

- section 17 passes
- predicted-reference target mIoU >= `0.3009776246035379`
  (= 70% of oracle B3)
- PairedVal >= `10/20`
- own-cross margin >= `0.20`

Do not alter gates.

# PART J — Verdict

## 19. Exactly one, priority order

1. `INVALID_EXPERIMENT`
2. `DIFFERENTIABLE_FIELD_INVALID`
3. `TASK6O_B3_REPRODUCTION_FAIL`
4. `REFERENCE_HEAD_NOT_LEARNABLE`
5. `REFERENCE_HEAD_INSUFFICIENT`
6. `REFERENCE_ERROR_PROPAGATION_SEVERE`
7. `PREDICTED_REFERENCE_CHAIN_FEASIBLE`

No other verdict.

# PART K — Interpretation boundary

DSH reports measurements only.

Do NOT claim final novelty or choose joint training, `[REF]`, GRCL, nearest/L3, a new reference architecture or a different field.

Final handoff recommendation exactly:

`等待 ChatGPT 根据 Task 6P 的 predicted-reference 误差传播结果决定 Task 6Q，不自行进行联合训练、MLLM 隐状态融合、nearest/L3 或 GRCL。`

# PART L — Required artifacts

Create:

```text
buildreasonseg_mvp/geometric_relation_field_v02.py
buildreasonseg_mvp/task6p_reference_head.py

evaluation/task6p_field_v02_audit.json
evaluation/task6p_b3_reproduction.json
evaluation/task6p_reference_pack_manifest.json
evaluation/task6p_reference_overfit20.json
evaluation/task6p_reference_val.json
evaluation/task6p_predicted_reference_target_val.json
evaluation/task6p_predicted_reference_paired_val.json
evaluation/task6p_field_propagation_diagnostics.json
evaluation/task6p_verdict.json

docs/task6p_differentiable_field_predicted_reference.md

scripts/task6p_freeze_reference_packs.py
scripts/task6p_field_v02_audit.py
scripts/task6p_train_reference.py
scripts/task6p_evaluate_chain.py
scripts/task6p_report.py
```

Checkpoints:
`artifacts/task6p/checkpoints/`
gitignored.

Reuse frozen Task 6N/6O feature cache.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

# PART M — Tests

## 20. Required tests

At least:

1. Task 6N artifacts unchanged
2. Task 6O artifacts unchanged
3. v0.1 field file unchanged
4. v0.2 binary numerical equivalence
5. v0.2 finite nonzero gradient to soft M_ref
6. v0.2 no detach in field forward path
7. v0.2 no Python-float centroid in field forward path
8. no test split access
9. only 8 directional L2 programs
10. reference packs deduplicated by exact key
11. target source id never reference-head input
12. relation id never reference-head input
13. reference head input = visual + family only
14. family vocab exactly largest/smallest
15. frozen SAM2 unchanged
16. B3 checkpoint hash verified
17. B3 oracle reproduction tolerance
18. B3 not retrained
19. predicted chain uses v0.2
20. predicted chain does not use oracle M_ref
21. GT reference only for reference evaluation
22. GT target only for target scoring
23. same predicted ref reused for same image/reference in paired eval
24. no `[REF]`
25. no GRCL/SCL
26. no nearest/L3
27. no graph transformer
28. no proposal training
29. no 4B
30. no download/install/GUI
31. previous suite preserved

Run:
`python -m pytest tests/ -q`

Task 6O ended at **627 passed, 1 skipped**.
Do not reduce prior passing tests.

# PART N — Git/storage

Do not commit:
- checkpoints
- SAM2 weights
- feature cache
- source imagery/vector files
- `.conda`
- large caches

Commit code/small JSON/docs/tests/handoff only.

Recommended:
1. `fix: add autograd-safe geometric relation field v02`
2. `feat: add predicted-reference grounding head`
3. `eval: measure predicted-reference error propagation`
4. optional docs/handoff commit

# PART O — Model policy

Default:
- DeepSeek V4.1 Flash + High

Use Flash + Max only for genuine implementation/runtime bugs.
Do not use V4 Pro by default.

# PART P — STOP

After Task 6P:
- commit
- push
- handoff
- STOP

Do not start:
- joint reference-target training
- ProgramHead/MLLM hidden-state fusion
- nearest
- L3
- GRCL
- full-dataset training
- proposal optimization
- GUI

Wait for ChatGPT audit.
