# TO_DSH — Task 6N: Oracle-Reference Geometric Relation Field Feasibility

> Status: ACTIVE
>
> Repository: `BuildReasonSeg`
>
> Base commit: `5e52d95c70d321a26b5d61ee96853d1123fcb26d`
>
> Predecessor: Task 6M.1 → `PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`
>
> This task is the first execution task of the final innovation-architecture phase.
>
> Research direction, architecture, variables, losses, stages and gates in this file are already decided by ChatGPT. DSH is an executor. Do not redesign them.

## 0. DSH role and STOP rule

All user-facing DSH output must be Chinese.

Follow this file literally.

DSH MAY:
- implement the specified module;
- reuse existing frozen feature caches and utilities;
- solve ordinary code/runtime bugs without changing the experiment;
- run the exact experiments and tests defined below.

DSH MUST NOT autonomously:
- change the research question;
- select another architecture;
- add a graph transformer;
- change the visual backbone;
- add `[REF]` tokens;
- add GRCL/SCL;
- add counterfactual loss;
- add another dataset;
- change the relation semantics;
- introduce new hyperparameter sweeps;
- switch to 4B;
- optimize the YOLO proposal model;
- start Task 6O.

If an unexpected issue requires any of those changes, STOP and report it.

## 1. Research decision already frozen by ChatGPT

The Challenge Cup project is:

**“空间推理引导的建筑物结构 MLLM 相关推理分割方法”**

For this project, “建筑物结构” currently means inter-building instance spatial structure in overhead imagery:
- relative direction;
- relative distance / proximity;
- scale/extreme relations;
- compositional reference → relation → target reasoning.

It does NOT mean roof/window/wall/beam internal building parts.

### 1.1 What is NOT claimed as novelty

Do not claim novelty for:
- MLLM + segmentation;
- `[SEG]`;
- `[REF]`;
- generic reference-mask conditioning;
- spatial program execution;
- geometric fields by themselves;
- relation-aware graph reasoning;
- generic spatial attention supervision;
- counterfactual reasoning by itself.

### 1.2 Fixed novelty hypothesis to test

The long-term method hypothesis is:

> a predicted/grounded reference building mask should be converted into an explicit, differentiable, relation-conditioned geometric prior and fused into a dense visual segmentation decoder; later, relation-level geometry supervision will constrain the predicted target mask.

Task 6N tests only:

> With an oracle reference mask, does an explicit reference-conditioned geometric relation field improve dense target segmentation over equally controlled relation-aware baselines that do not receive that field?

This task does NOT test the final end-to-end architecture.

## 2. Literature-overlap statement — copy into the Task 6N design doc

Record this fixed research note; do not perform a new literature search:

1. SegLLM (ICLR 2025) already re-injects previous/reference masks and uses `[REF]`/`[SEG]` mask-aware decoding. Reference-mask conditioning or `[REF]` is not our novelty.
2. R²S (ICCV 2025) already uses a two-stage relevant-element → reasoning-prior paradigm in 3D. Generic two-stage reasoning priors are not our novelty.
3. Think2Seg-RS (ISPRS JPRS 2026) decouples LVLM reasoning from SAM geometry execution using structured geometric prompts. Semantic/geometry decoupling is not our novelty.
4. SegEarth-R2 (CVPR 2026) uses spatial-attention supervision for remote-sensing language-guided segmentation. Generic spatial supervision is not our novelty.
5. SRGFormer (Sensors 2026) decomposes target/relation/position semantics and performs relation-aware graph reasoning. Relation decomposition or graph reasoning alone is not our novelty.
6. GeoSelect (TGRS 2026) executes typed spatial programs and uses continuous geometric fields plus discrete operators over candidate sets. “Geometric field + spatial program” alone is not our novelty.
7. GeoRefer-Bench (2026) already provides executable geospatial relation queries and counterfactual pairs. Executable relation datasets/counterfactual evaluation alone are not our novelty.

The possible novelty under investigation is narrower:
**a differentiable reference-conditioned geometric relation field fused into a dense segmentation decoder and later supervised by explicit reference–target geometry consistency.**

No “first-ever” claim is allowed in Task 6N.

## 3. Frozen project assets

Treat as read-only/frozen:

- `datasets/whu_native_vector/v1.0/`
- `datasets/build_spatial_reason/v0.2/`
- `datasets/build_spatial_reason/v0.1.1/`
- `configs/spatial_relations_v1.yaml`
- Task 3B relation engine
- Task 6J artifacts
- Task 6M artifacts
- Task 6M.1 artifacts
- Task 6C.7 frozen visual-feature-cache evidence
- Task 6I artifacts
- YOLO checkpoints and proposal runs

Do not modify historical artifacts.

### 3.1 Task 6M.1 metric erratum

Do not edit frozen Task 6M.1 JSON.

Record an erratum in the Task 6N design doc:

`evaluation/task6m1_verdict.json` has:
`combined_best_mask_mAP50_95 = 0.44891`

That value is actually the **box mAP50-95 / Ultralytics fitness proxy**.

Authoritative Task 6M.1 values from the training summary:
- best box mAP50-95 = **0.44891**
- best mask mAP50-95 = **0.40475**
- best mask mAP50 = **0.73742**
- best epoch = **40**

Future reporting must use the training-summary values.

## 4. Scope: directional L2 only

Handle ONLY these 8 BuildSpatialReason v0.2 program ids:

- `largest_to_left_of`
- `largest_to_right_of`
- `largest_to_above`
- `largest_to_below`
- `smallest_to_left_of`
- `smallest_to_right_of`
- `smallest_to_above`
- `smallest_to_below`

Do NOT include nearest, L1 extremes, L3 compositions, or new relations.

Use train only for training, val only for development. **Do not use test split in Task 6N.**

## 5. Oracle-reference protocol

For each sample:

- `M_ref` = canonical native-vector GT mask of the reference building.
- `M_target` = canonical native-vector GT target mask, used only as training label / evaluation GT.
- relation id = canonical direction encoded by v0.2.
- image = source RGB tile.
- visual feature = frozen image feature from the existing project visual backbone/cache.

This is deliberately an **oracle-reference ablation**.

Every artifact must say:
`reference_source = oracle_native_gt`

Never describe Task 6N as end-to-end inference.

GT target must never be fed as an input.

## 6. Frozen visual representation

Reuse the already established **frozen SAM2 image-embedding path/cache from Task 6C.7 / 6I**.

Do not retrain SAM2, change SAM2 checkpoint, switch to YOLO feature maps, or use GT building-union masks as visual input.

The new module receives frozen dense visual feature tensor `V`.

If cached spatial size differs from 64×64, use its native cached `(h,w)` and adapt masks/fields to `(h,w)` deterministically.

Record exact feature source/checkpoint, C×h×w, and cache provenance/hash where available.

## 7. Exact geometric definitions

Respect frozen Task 3B convention:

`relation(subject, object)` means subject satisfies relation with respect to object.

For target T relative to reference R:
- left_of(T,R) → `cx_T < cx_R`
- right_of(T,R) → `cx_T > cx_R`
- above(T,R) → `cy_T < cy_R`
- below(T,R) → `cy_T > cy_R`

Use active config:
- `alpha = 1.2`
- `tau = 0.04`

Image y increases downward.

## 8. GeometricRelationField v0.1 — exact implementation

Create:
`buildreasonseg_mvp/geometric_relation_field.py`

Input:
- binary/soft `M_ref` at image resolution;
- relation id in `{left_of,right_of,above,below}`;
- output size `(h,w)`.

### 8.1 Reference geometry

From `M_ref`, compute centroid `cx_ref`, `cy_ref` normalized to `[0,1]`.

For every output location `(x,y)` normalized to `[0,1]`:

```text
dx = x - cx_ref
dy = y - cy_ref
ax = abs(dx)
ay = abs(dy)
```

### 8.2 Smooth directional score

No sweep.

```text
s_axis   = 0.02
s_margin = 0.02
alpha    = 1.2
tau      = 0.04
```

`alpha`/`tau` come from `spatial_relations_v1.yaml`.
Softness is fixed at `tau / 2`.

For horizontal relations:

```text
axis_score = sigmoid((ax - alpha * ay) / s_axis)
margin_score = sigmoid((ax - tau) / s_margin)
```

For vertical relations:

```text
axis_score = sigmoid((ay - alpha * ax) / s_axis)
margin_score = sigmoid((ay - tau) / s_margin)
```

Sign score:

```text
left_of:  sign_score = sigmoid((-dx) / s_margin)
right_of: sign_score = sigmoid(( dx) / s_margin)
above:    sign_score = sigmoid((-dy) / s_margin)
below:    sign_score = sigmoid(( dy) / s_margin)
```

Final:

```text
P_rel = sign_score * axis_score * margin_score
```

Clamp `[0,1]`.

No learned parameters in field generation.

Resize `M_ref` to `(h,w)` as a soft mask and clamp `[0,1]`.

Do not add distance transform, bbox, candidate masks or extra geometry channels in 6N.

## 9. Frozen decoder architecture

One common decoder family.

Let frozen visual feature channels = C.

### 9.1 Visual projection

```text
Conv1x1(C → 128)
GroupNorm(8,128)
GELU
```

### 9.2 Relation embedding

Trainable embedding:
- 4 relations
- dim = **16**

Broadcast over `(h,w)`.

### 9.3 Fusion trunk

Full B2 concatenates:

```text
visual_128
M_ref_down       # 1
P_rel            # 1
relation_embed   # 16
```

146 channels.

Then:

```text
Conv3x3(146 → 128, padding=1)
GroupNorm(8,128)
GELU
Conv3x3(128 → 64, padding=1)
GroupNorm(8,64)
GELU
Conv1x1(64 → 1)
```

Upsample logits bilinearly to canonical target-mask resolution before loss/evaluation.

No attention, transformer, graph block or extra MLP.

## 10. Three controlled variants

Train exactly:

### N-B0 — relation-aware visual baseline
Inputs:
- visual_128
- relation_embed

No reference mask, no field.
First fusion conv input = 144.

### N-B1 — reference-mask baseline
Inputs:
- visual_128
- `M_ref_down`
- relation_embed

No field.
First fusion conv input = 145.

### N-B2 — full GeometricRelationField
Inputs:
- visual_128
- `M_ref_down`
- `P_rel`
- relation_embed

First fusion conv input = 146.

Do not pad missing channels merely to equalize parameters.
Report exact parameter counts.

No other architectural difference.

## 11. Loss

Exactly:

```text
L = BCEWithLogitsLoss + DiceLoss
```

Reuse project canonical Dice implementation if available.

No GRCL, relation loss, counterfactual loss, auxiliary field loss, focal loss or class weighting.

## 12. Dataset construction

Create deterministic Task 6N view from BuildSpatialReason v0.2.

Eligible if:
- program is one of the 8;
- oracle reference and target source_feature_id resolve;
- RGB exists;
- frozen visual feature exists/can be generated through existing frozen cache path.

No extra tiny/border/visibility filter.
No difficulty deletion.

### 12.1 Fixed packs

Freeze before training.

#### Overfit20
20 train records:
- deterministic sorted selection;
- all 4 directions;
- both largest/smallest reference families;
- at least 4 same-image counterfactual pairs if available.

#### MiniTrain1000
First deterministic 1000 eligible train records after stable seeded stratification by direction and reference family.
If fewer than 1000, use all and report.

#### MiniVal240
240 val records:
- 30 per program id if available;
- otherwise deterministic proportional fill;
- all 8 ids represented.

#### PairedVal20
20 val same-image pairs:
- same tile;
- same reference instance/source_feature_id;
- different direction;
- different target instance/source_feature_id.

If fewer than 20 valid pairs, STOP before training:
`PAIRED_SET_INSUFFICIENT`.

Write record ids + SHA256 manifest.

## 13. Stage N0 — field sanity

No training.

On MiniVal240, for each record:
1. mean `P_rel` over target mask;
2. mean over reference mask;
3. mean over every other native building;
4. target rank among non-reference native instances by mean field score.

Report:
- top-1 rate;
- top-3 rate;
- mean target score;
- mean best distractor score;
- per relation.

Diagnostic only. No threshold tuning.

## 14. Stage N1 — Overfit20

Train B0/B1/B2 separately on exact same Overfit20.

Use:
- AdamW
- lr `1e-3`
- weight_decay `1e-4`
- max steps **1200**
- batch **4**
- no scheduler
- no augmentation
- seed **20260929**
- same AMP setting across variants

Evaluate each 100 steps.

Record best/final:
- mIoU
- Dice
- per relation
- paired own-vs-cross if available

### N1 gate

B2 must reach:
- train mIoU >= **0.85**
- train Dice >= **0.90**

If B2 fails:
- STOP
- verdict `GEOMETRIC_RELATION_FIELD_NOT_LEARNABLE_IN_CURRENT_DECODER`
- no N2.

## 15. Stage N2 — MiniTrain1000 → MiniVal240

Only if N1 passes.

Train each variant from fresh initialization:

- AdamW
- lr `3e-4`
- weight_decay `1e-4`
- batch **8**
- max epochs **25**
- early stopping patience **5** on val mIoU
- seed **20260929**
- no augmentation
- no scheduler
- same AMP setting across variants

Model selection = highest MiniVal240 mIoU.

Do not touch test.

## 16. N2 metrics

For B0/B1/B2 report:

- val mIoU
- val Dice
- Pr@0.5
- per relation mIoU
- largest-ref vs smallest-ref
- border-target mIoU
- tiny-target mIoU if present
- total/trainable params
- peak VRAM
- wall time

### PairedVal20

Run both relations for every pair with same image/reference.

Report:
- pass / 20;
- mean own-target IoU;
- mean cross-target IoU;
- own-cross margin.

Pair passes only if both members prefer their own GT target over the paired alternative by IoU.

## 17. Causal success criteria

Positive signal only if ALL:

1. B2 passes N1.
2. MiniVal:
   - `B2 - B0 mIoU >= 0.05`
   - `B2 - B1 mIoU >= 0.02`
3. B2 PairedVal >= **14/20**.
4. B2 mean own-target IoU exceeds mean cross-target IoU by >= **0.10**.
5. no GT target enters model input.

Do not alter gates.

## 18. Exactly one verdict

Allowed:

- `GEOMETRIC_RELATION_FIELD_FEASIBLE`
  - all section-17 criteria pass.

- `REFERENCE_MASK_HELPS_FIELD_DOES_NOT`
  - B1-B0 >= 0.05, but B2-B1 < 0.02.

- `RELATION_CONDITIONING_NOT_GENERALIZING`
  - N1 passes but B2-B0 < 0.05 on MiniVal.

- `GEOMETRIC_RELATION_FIELD_NOT_LEARNABLE_IN_CURRENT_DECODER`
  - B2 fails N1.

- `PAIRED_SET_INSUFFICIENT`

- `INVALID_EXPERIMENT`

No other verdict.

## 19. No research interpretation by DSH

DSH may report measurements.

Do NOT autonomously conclude that novelty is proven, SRE/GRCL should be abandoned, another architecture should replace it, or a new loss should be added.

In `FROM_DSH.md`, final Recommended next step must be exactly:

`等待 ChatGPT 根据 Task 6N 测量结果决定 Task 6O，不自行选择后续算法。`

## 20. Required artifacts

Create at minimum:

```text
configs/task6n_oracle_relation_field.yaml

evaluation/task6n_pack_manifest.json
evaluation/task6n_field_sanity.json
evaluation/task6n_overfit20.json
evaluation/task6n_mini_val.json
evaluation/task6n_paired_val.json
evaluation/task6n_ablation_summary.json
evaluation/task6n_verdict.json

docs/task6n_oracle_reference_geometric_relation_field.md

buildreasonseg_mvp/geometric_relation_field.py
buildreasonseg_mvp/task6n_relation_decoder.py

scripts/task6n_freeze_packs.py
scripts/task6n_field_sanity.py
scripts/task6n_train.py
scripts/task6n_evaluate.py
scripts/task6n_report.py
```

Checkpoints/caches under `artifacts/task6n/`, gitignored.

Update:
- `handoff/FROM_DSH.md`
- `handoff/PROJECT_STATE.md`

## 21. Required tests

At least:

1. Task 6M.1 artifacts unchanged;
2. v0.2 unchanged;
3. spatial config unchanged;
4. no test access;
5. only 8 allowed programs;
6. target never input;
7. oracle reference explicitly marked;
8. direction signs correct;
9. y-axis convention correct;
10. alpha=1.2;
11. tau=0.04;
12. softness=tau/2;
13. field bounded [0,1];
14. left/right mirror sanity;
15. above/below mirror sanity;
16. B0 no ref/field;
17. B1 ref/no field;
18. B2 ref+field;
19. same visual backbone/cache;
20. same packs;
21. same N1 optimizer/steps;
22. same N2 optimizer/epochs;
23. deterministic pack hashes;
24. PairedVal same tile/ref and different target;
25. no `[REF]` token;
26. no GRCL/SCL;
27. no graph transformer;
28. no proposal training;
29. no 4B;
30. no GUI/download/install.

Run:
`python -m pytest tests/ -q`

Task 6M.1 ended at **569 passed, 1 skipped**. Do not reduce previous passing tests.

## 22. Storage / Git

Do not commit:
- SAM2 weights
- model checkpoints
- feature caches
- `.conda`
- source imagery/vectors
- large caches

Commit code/config/small JSON/docs/tests/handoff only.

Recommended:
1. `feat: add oracle-reference geometric relation field`
2. `eval: measure relation-field ablations`
3. optional docs/handoff commit

## 23. DSH model policy

Default:
- DeepSeek V4.1 Flash + High

Use Flash + Max only for genuine implementation/runtime bugs.

Do not use V4 Pro by default.

## 24. STOP

After Task 6N:
- commit
- push
- update handoff
- STOP

Do not start:
- predicted-reference integration
- nearest
- L3
- GRCL
- counterfactual loss
- full-dataset training
- proposal optimization
- GUI

Wait for ChatGPT audit.
