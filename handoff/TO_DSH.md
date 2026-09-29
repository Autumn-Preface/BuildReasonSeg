# TO_DSH — Task 6O: Geometric Relation Field Causal Decomposition

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `90f3735cf57115cd67e15b1b3457752e1fb049a6`
>
> Predecessor: Task 6N → `GEOMETRIC_RELATION_FIELD_FEASIBLE`
>
> This task is a narrow causal-decomposition experiment. Research questions, variants, metrics and verdict logic are already decided by ChatGPT. DSH is an executor.

## 0. DSH role

All user-facing DSH output must be Chinese.

Follow this task literally.

DSH MAY:
- implement the two specified ablation variants;
- reuse exact Task 6N packs, feature cache, field implementation and evaluator;
- solve ordinary implementation/runtime bugs without changing the experiment.

DSH MUST NOT:
- choose another architecture;
- change SAM2;
- change the GeometricRelationField formula;
- change relation semantics;
- add `[REF]`;
- add GRCL/SCL;
- add nearest/L3;
- add a graph transformer;
- add a new dataset;
- touch the test split;
- optimize YOLO;
- start predicted-reference integration;
- perform a literature search;
- invent new gates or sweeps.

If any change outside this task is required, STOP and report.

## 1. Why Task 6O exists

Task 6N produced:

- B0 MiniVal240 mIoU = **0.214786**
- B1 MiniVal240 mIoU = **0.241316**
- B2 MiniVal240 mIoU = **0.453127**
- B2 − B0 = **+0.238340**
- B2 − B1 = **+0.211811**
- B2 PairedVal20 = **16/20**
- B2 mean own − cross IoU = **+0.441188**
- N0 parameter-free field target top-1 among non-reference native instances = **1.000**

Task 6N therefore established feasibility, but leaves two causal questions unresolved:

1. Does the direct `M_ref_down` channel still matter after `P_rel` is supplied?
2. Is the gain really field-guided visual segmentation, or does the handcrafted geometric field alone solve most of the directional benchmark?

Task 6O answers only these two questions.

## 2. Task 6N N0 clarification

Do not treat this as a bug:

- `mean_other_building_score` includes the GT target because section 13 asked for every non-reference building.
- `mean_best_distractor_score` excludes both reference and GT target.

Do not rewrite Task 6N artifacts.

## 3. Frozen assets

Read-only/frozen:

- all Task 6N tracked artifacts;
- Task 6N B0/B1/B2 checkpoints;
- exact Task 6N packs;
- `buildreasonseg_mvp/geometric_relation_field.py`;
- `configs/spatial_relations_v1.yaml`;
- WHU native-vector v1.0;
- BuildSpatialReason v0.2;
- Task 6M.1 and earlier artifacts;
- frozen SAM2.1 Hiera Base+ feature path/cache.

Do not regenerate Task 6N packs.

### 3.1 Exact packs to reuse byte-for-byte

Read hashes from `evaluation/task6n_pack_manifest.json` and verify:

- `artifacts/task6n/packs/overfit20.json`
- `artifacts/task6n/packs/mini_train_1000.json`
- `artifacts/task6n/packs/mini_val_240.json`
- `artifacts/task6n/packs/paired_val_20.json`

If any hash mismatches, STOP:
`TASK6N_PACK_MISMATCH`.

## 4. Scope

Same Task 6N scope only:

- `largest_to_left_of`
- `largest_to_right_of`
- `largest_to_above`
- `largest_to_below`
- `smallest_to_left_of`
- `smallest_to_right_of`
- `smallest_to_above`
- `smallest_to_below`

No nearest, L1, L3, new relation, test split, predicted reference, GRCL or new data.

Reference source remains:
`oracle_native_gt`

GT target remains label/evaluation only.

## 5. Frozen common definitions

Use exactly Task 6N:

- frozen SAM2 feature tensor `V`, 256×64×64;
- relation embedding dim 16;
- 4 direction ids;
- GeometricRelationField v0.1 unchanged;
- alpha=1.2, tau=0.04, s_axis=s_margin=0.02;
- BCEWithLogitsLoss + DiceLoss;
- bilinear upsample to 512×512 for evaluation.

No threshold tuning.

# PART A — Verify Task 6N B2 baseline

## 6. Verify local B2 checkpoint

Read B2 checkpoint path and SHA256 from:
`evaluation/task6n_mini_val.json`

Require:
- local file exists;
- SHA256 matches exactly.

If missing/mismatched:
STOP with `TASK6N_B2_CHECKPOINT_UNAVAILABLE`.

Do not retrain B2.

## 7. Re-evaluate B2 exactly once

Run frozen Task 6N evaluator on:

- MiniVal240
- PairedVal20

using frozen B2 checkpoint.

Reproduced values must match stored Task 6N values within:

- mIoU absolute tolerance `1e-6`
- Dice absolute tolerance `1e-6`
- pair pass count exact
- own/cross means tolerance `1e-6`

Write:
`evaluation/task6o_b2_reproduction.json`

If outside tolerance:
STOP with `TASK6N_B2_REPRODUCTION_FAIL`.

# PART B — New variant B3

## 8. N-B3: field-guided visual model, no direct reference channel

Purpose:
test whether `P_rel` carries useful reference-conditioned geometry without separately concatenating `M_ref_down`.

Inputs:

```text
visual_128
P_rel            # 1 channel
relation_embed   # 16 channels
```

NO `M_ref_down` input to decoder.

Fusion input = **145 channels**.

Use Task 6N visual projection and trunk:

```text
visual:
Conv1x1(256 → 128)
GroupNorm(8,128)
GELU

fusion:
Conv3x3(145 → 128, padding=1)
GroupNorm(8,128)
GELU
Conv3x3(128 → 64, padding=1)
GroupNorm(8,64)
GELU
Conv1x1(64 → 1)
```

Relation embedding remains 4×16.

`P_rel` is generated from the oracle reference mask outside the decoder.

Do not pass oracle reference mask itself into B3.

# PART C — New variant B4

## 9. N-B4: geometry-only capacity control

Purpose:
measure how much is solved by the handcrafted field without RGB-derived visual evidence.

Inputs:

```text
P_rel
relation_embed
```

NO visual feature.
NO direct reference mask.

Use a 128-channel field projection:

```text
Conv1x1(1 → 128)
GroupNorm(8,128)
GELU
```

Then broadcast relation embedding and concatenate:

```text
field_128
relation_embed_16
```

Fusion input = **144 channels**.

Then the same trunk:

```text
Conv3x3(144 → 128, padding=1)
GroupNorm(8,128)
GELU
Conv3x3(128 → 64, padding=1)
GroupNorm(8,64)
GELU
Conv1x1(64 → 1)
```

Do not provide RGB, SAM2 features, M_ref_down, candidate masks or target geometry.

Report exact params.

# PART D — Training protocol

## 10. Stage O1: Overfit20

Train B3 and B4 separately on exact Task 6N Overfit20.

Use EXACT Task 6N N1 settings:

- AdamW
- lr `1e-3`
- weight_decay `1e-4`
- max steps `1200`
- batch `4`
- no scheduler
- no augmentation
- seed `20260929`
- same AMP behavior as Task 6N
- evaluate every 100 steps

Record:
- best/final mIoU
- best/final Dice
- per relation
- paired own/cross on Overfit20 pairs
- parameter count
- wall time
- peak VRAM

### O1 B3 gate

B3 must reach:
- mIoU >= **0.85**
- Dice >= **0.90**

If B3 fails:
STOP with `FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE`.

B4 has no stop gate.

## 11. Stage O2: MiniTrain1000 → MiniVal240

Only if B3 O1 passes.

Train B3 and B4 from fresh initialization.

Use EXACT Task 6N N2 settings:

- AdamW
- lr `3e-4`
- weight_decay `1e-4`
- batch `8`
- max epochs `25`
- early stopping patience `5` on val mIoU
- seed `20260929`
- no scheduler
- no augmentation
- same AMP setting
- model selection = highest MiniVal240 mIoU

No test access.

# PART E — Evaluation

## 12. MiniVal metrics

For B3 and B4 report:

- mIoU
- Dice
- Pr@0.5
- per relation mIoU
- largest-ref vs smallest-ref
- border-target mIoU
- tiny-target mIoU if present
- total/trainable params
- peak VRAM
- wall time
- best epoch

Write:
`evaluation/task6o_mini_val.json`

## 13. PairedVal20

Run exact frozen Task 6N PairedVal20.

Report:
- pass / 20
- mean own IoU
- mean cross IoU
- own-cross margin

Write:
`evaluation/task6o_paired_val.json`

# PART F — Predeclared causal comparisons

## 14. Comparisons

Use reproduced B2 as baseline.

Define:

```text
delta_B3_B2 = B3_mIoU - B2_mIoU
delta_B3_B1 = B3_mIoU - frozen_Task6N_B1_mIoU
delta_B3_B4 = B3_mIoU - B4_mIoU
```

Also report Dice and paired deltas.

### 14.1 Direct reference-channel retention

Direct `M_ref_down` is considered unnecessary for this directional decoder if:

- `B3_mIoU >= B2_mIoU - 0.03`
- B3 paired pass >= **14/20**
- B3 own-cross margin >= **0.10**

### 14.2 Visual contribution

Visual evidence is materially necessary if:

- `B3_mIoU - B4_mIoU >= 0.10`

### 14.3 Geometry-only confound

The benchmark is considered too solvable by the handcrafted field alone if BOTH:

- `B4_mIoU >= B3_mIoU - 0.05`
- B4 paired pass >= **12/20**

Do not change thresholds.

# PART G — Verdict

## 15. Exactly one verdict, priority order

1. `INVALID_EXPERIMENT`
   - leakage, frozen-artifact mutation, target-input leakage, test access or protocol violation.

2. `FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE`
   - B3 fails O1.

3. `GEOMETRY_ONLY_BENCHMARK_CONFOUND`
   - O1 passes and section 14.3 passes.

4. `DIRECT_REFERENCE_CHANNEL_MATTERS`
   - O1 passes; geometry-only confound does not pass; and:
   `B3_mIoU < B2_mIoU - 0.05`

5. `FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`
   - B3 O1 passes;
   - section 14.1 passes;
   - section 14.2 passes;
   - section 14.3 does not pass.

6. `FIELD_CAUSAL_SIGNAL_PARTIAL`
   - any remaining valid outcome.

No other verdict.

## 16. Interpretation boundary

DSH may report measurements only.

Do not decide:
- whether this is paper novelty;
- whether predicted-reference should be implemented;
- whether field formula should change;
- whether GRCL should be added.

Final handoff recommendation must be exactly:

`等待 ChatGPT 根据 Task 6O 因果分解结果决定后续架构，不自行开始 predicted-reference、nearest、L3 或 GRCL。`

# PART H — Required artifacts

Create:

```text
evaluation/task6o_b2_reproduction.json
evaluation/task6o_overfit20.json
evaluation/task6o_mini_val.json
evaluation/task6o_paired_val.json
evaluation/task6o_causal_summary.json
evaluation/task6o_verdict.json

docs/task6o_field_causal_decomposition.md

scripts/task6o_train.py
scripts/task6o_evaluate.py
scripts/task6o_report.py
```

Modify/add model code only as needed for B3/B4 without changing frozen B0/B1/B2 behavior.

Checkpoints:
`artifacts/task6o/checkpoints/`
gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

# PART I — Tests

## 17. Required tests

At least:

1. Task 6N artifacts unchanged
2. Task 6N pack hashes exact
3. Task 6N B2 checkpoint hash matches
4. B2 reproduction within tolerance
5. no test split access
6. only 8 directional L2 programs
7. target never model input
8. B3 receives visual + field + relation only
9. B3 does not receive direct ref mask
10. B4 receives field + relation only
11. B4 does not receive visual
12. B4 does not receive direct ref mask
13. B4 field projection is 1→128
14. GeometricRelationField unchanged
15. alpha/tau/softness unchanged
16. frozen SAM2 path unchanged
17. same packs across B2/B3/B4
18. O1 settings identical to Task 6N N1
19. O2 settings identical to Task 6N N2
20. deterministic seed exact
21. no new relation
22. no nearest/L3
23. no `[REF]`
24. no GRCL/SCL
25. no graph transformer
26. no proposal training
27. no 4B
28. no download/install
29. no GUI
30. previous suite preserved

Run:
`python -m pytest tests/ -q`

Task 6N ended at **597 passed, 1 skipped**. Do not reduce prior passing tests.

# PART J — Git/storage

Do not commit:
- checkpoints
- frozen feature cache
- SAM2 weights
- source imagery/vector data
- `.conda`
- large caches

Commit code/small JSON/docs/tests/handoff only.

Recommended:
1. `feat: add geometric-field causal ablations`
2. `eval: decompose visual and reference-channel contributions`
3. optional docs commit

# PART K — Model policy

Default:
- DeepSeek V4.1 Flash + High

Only use Flash + Max for a genuine implementation/runtime bug.

Do not use V4 Pro by default.

# PART L — STOP

After Task 6O:
- commit
- push
- handoff
- STOP

Do not start:
- predicted-reference grounding
- nearest
- L3
- GRCL
- full-dataset training
- proposal optimization
- GUI

Wait for ChatGPT audit.
