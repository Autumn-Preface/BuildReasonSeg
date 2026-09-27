# TO_DSH — Task 6D: Spatial Grounding Bridge v0.1

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: return focus from performance engineering to **model capability and algorithm architecture**.
>
> Task 6C established that the direct `[SEG] hidden → Projection MLP → SAM sparse prompt` path does not produce instruction-conditioned target selection on unseen images.
>
> Task 6C.5–6C.7 are now closed as performance work.
>
> Task 6D introduces the first explicit spatial-grounding mechanism:
>
> **MLLM `[SEG]` hidden state → predicted target geometry → official SAM2 spatial prompt encoder → target mask**
>
> Ground-truth geometry is used **only as training supervision / oracle diagnostic**, never as inference-time input.

## 0. User priority

The user prioritizes:
1. model capability;
2. useful training/inference code;
3. CMD-runnable `train.py` / `predict.py` / `evaluate.py`;
4. architecture innovation and paper evidence;
5. UI/GUI only much later.

Do **not** build GUI/Web UI in this task.

Do **not** spend another task on generic performance tuning unless runtime blocks model development.

## 1. Data policy

Current BuildSpatialReason v0.1.1 / WHU-based data remains the controlled dataset for Task 6D so the architecture experiment stays comparable with Task 6C.

WHU is **not a permanent hard constraint anymore**.

The user allows:
- replacing WHU with a more suitable building dataset later;
- adapting/programmatically transforming a new dataset for BuildSpatialReason-style supervision.

Do not change datasets inside Task 6D. Record evidence if current pseudo-instances materially limit the architecture.

## 2. Hard architecture boundaries

Use:
- Qwen3-VL-2B-Instruct;
- SAM2.1 Hiera Base+;
- current text-only LoRA;
- current `[SEG]`;
- strict deterministic training;
- Task 6C paired subset `P`;
- fixed Task 6B/6C validation and paired probe;
- Task 6C.7 accepted runtime (`collect_grad_norms=false`, frozen Qwen visual cache, no redundant Phase-B transfer).

Do **not** add:
- Qwen 4B;
- `[REF]`;
- Spatial Relation Encoder;
- Spatial Consistency Loss;
- external datasets;
- true batch > 1;
- UI;
- full 15,592-sample training.

Goal: prove that **the same image can produce different correct target masks under different instructions**.

# PART A — Freeze the current failure baseline

## 3. Baseline evidence

Do not rerun all Task 6C arms.

Use the frozen Task 6C `P_C` result:
- strict e2e mIoU ≈ 0.10604;
- paired mask probe = 0/20;
- same-image prediction-to-prediction IoU ≈ 0.999;
- valid `[SEG]` emission 120/120.

Record exact artifact/hash used.

A small current-code sanity revalidation is allowed only if needed to prove accepted performance changes remain value-equivalent.

# PART B — Oracle spatial-prompt diagnostic

## 4. Oracle feasibility

Before training a geometry head, establish whether SAM2 can recover the target building when given correct geometry.

Use fixed 120-record validation subset and 20 paired validation images.

### Oracle Point

Derive one deterministic interior point from GT target mask.

Preferred:
- maximum of Euclidean distance transform inside target mask;
- deterministic tie-breaking.

Feed it through official SAM2 prompt encoder as a positive point.

### Oracle Box

Derive tight GT target bounding box.

Feed it through official SAM2 box prompt path.

### Decoder state

Use the same SAM2 decoder state that will initialize the Task 6D candidate.

Do not train on validation.

Report point and box:
- strict mask mIoU;
- Dice;
- paired mask selection /20;
- mean own-target IoU;
- mean cross-target IoU;
- mean IoU(pred_A, pred_B).

Create:
`evaluation/task6d_oracle_prompt_diagnostic.json`

## 5. Geometry choice

Choose exactly one geometry head:

1. if Oracle Point paired >= 18/20 and mIoU >= 0.50, choose **POINT**, unless Oracle Box improves mIoU by >= 0.10;
2. otherwise, if Oracle Box paired >= 18/20 and mIoU >= 0.50, choose **BOX**;
3. if neither reaches those gates, stop before training and report `ORACLE_SPATIAL_PROMPT_INSUFFICIENT`.

Do not silently choose the prettier result.

Do not use combined point+box in Task 6D.

# PART C — Spatial Grounding Head

## 6. Architecture

Implement:

`SpatialGroundingHead`

Input:
`[SEG] hidden state, 2048-d`

Minimal structure:

```text
LayerNorm(2048)
→ Linear(2048, 512)
→ GELU
→ Linear(512, D)
→ sigmoid
```

where:
- `D=2` for POINT: normalized `(x, y)`;
- `D=4` for BOX: normalized `(x1, y1, x2, y2)`.

For BOX, parameterize/canonicalize differentiably so `x1<=x2`, `y1<=y2`.

Do not use the old arbitrary projected language vector as the SAM sparse prompt in the candidate arm.

Candidate:

```text
Image + instruction
        ↓
Qwen3-VL + text LoRA
        ↓
reasoning + [SEG]
        ↓
[SEG] hidden
        ↓
SpatialGroundingHead
        ↓
predicted point / box
        ↓
official SAM2 prompt encoder
        ↓
SAM2 mask decoder
        ↓
target mask
```

Name: **Spatial Grounding Bridge v0.1**.

## 7. Geometry supervision

### POINT
Use same deterministic distance-transform interior point as oracle. Normalize to `[0,1]^2`.

### BOX
Use tight target-mask bounding box. Normalize to `[0,1]`.

GT geometry is supervision only.

At inference/evaluation the prompt must use **predicted geometry only**.

Add tests for this separation.

# PART D — Loss

## 8. Objective

Keep Task 6C headline losses:

```text
2.0 * LM CE
+ 2.0 * mask BCE
+ 1.0 * mask Dice
```

Add:

```text
lambda_ground = 5.0
```

with `SmoothL1(pred_geometry, gt_geometry)`.

This is a provisional architecture-proof weight. Do not sweep weights in Task 6D.

Record raw component magnitudes.

# PART E — Two-stage training

## 9. Stage G0 — grounding proof

Training:
- Task 6C paired `P` subset: 480 records / 240 images × 2 targets.

Train:
- text LoRA;
- `[SEG]` trainable token machinery;
- SpatialGroundingHead.

Freeze:
- SAM2 image encoder;
- SAM2 decoder for G0;
- old Projection MLP unused.

Loss:
- LM CE;
- grounding loss.

Maximum:
- 2 epochs.

Validate each epoch on fixed 120 records.

Primary metrics:

### POINT
- mean normalized coordinate error;
- predicted point inside correct GT target rate;
- predicted point inside paired other-target rate;
- paired point-selection /20.

### BOX
- mean box IoU with GT box;
- predicted-box center inside target rate;
- paired geometry-selection /20.

G0 success:
- valid `[SEG]` >= 90%;
- geometry paired >= 14/20;
- plus:
  - POINT: correct-target inside >= 70%;
  - BOX: mean GT-box IoU >= 0.35.

If G0 fails, stop:
`GROUNDING_REPRESENTATION_FAILED`.

Do not compensate with 4B or extra modules.

## 10. Stage G1 — joint segmentation

Only if G0 passes.

Train:
- text LoRA;
- `[SEG]`;
- SpatialGroundingHead;
- SAM2 mask decoder.

Keep frozen:
- Qwen base;
- Qwen visual tower;
- SAM2 image encoder;
- SAM2 prompt encoder parameters by default.

Loss:

```text
2.0 LM CE
+ 2.0 mask BCE
+ 1.0 mask Dice
+ 5.0 grounding SmoothL1
```

Maximum:
- 3 epochs;
- same 480 paired samples.

No hyperparameter sweep.

Use Task 6C.7 visual cache.

# PART F — Evaluation

## 11. Free generation is primary

For each val sample:

```text
image + instruction
→ free generated reasoning + [SEG]
→ generated [SEG] hidden
→ predicted geometry
→ official SAM2 prompt encoder
→ mask
```

Exactly one `[SEG]` required.

Invalid emission: mask IoU=0 and geometry failure.

Report:
- emission;
- strict e2e mIoU;
- Dice;
- conditional mIoU;
- L1/L2/L3;
- query-family;
- grounding metrics.

Teacher-forced remains diagnostic only.

## 12. Same-image paired probe is main gate

For masks:

```text
IoU(pred_A, GT_A) > IoU(pred_A, GT_B)
IoU(pred_B, GT_B) > IoU(pred_B, GT_A)
```

Define analogous geometry criterion.

Report:
- mask paired /20;
- geometry paired /20;
- own-target IoU;
- cross-target IoU;
- own-minus-cross margin;
- IoU(pred_A, pred_B).

# PART G — Verdict

## 13. Use exactly one

### `SPATIAL_GROUNDING_FIX_FOUND`

Require:
- G0 passes;
- `[SEG]` >= 90%;
- geometry paired >= 14/20;
- mask paired >= 14/20;
- own-minus-cross margin > 0.05;
- strict e2e mIoU >= 0.20;
- no GT leakage.

### `SPATIAL_GROUNDING_PARTIAL`

Geometry becomes instruction-conditioned but mask gate is not fully solved.

### `GROUNDING_REPRESENTATION_FAILED`

`[SEG]` hidden cannot learn reliable target geometry.

### `ORACLE_SPATIAL_PROMPT_INSUFFICIENT`

Even correct GT point/box prompts cannot produce adequate masks.

### `INVALID_EXPERIMENT`

Leakage, subset mismatch, corruption, nondeterminism or correctness failure.

# PART H — Diagnostics

## 14. Representation diagnosis

Same-image/different-instruction:
- `[SEG]` hidden cosine/L2/norm;
- predicted point/box distance between A and B;
- grounding-head penultimate representation if useful;
- mask A-vs-B IoU.

Key question:

> Does instruction variation now produce **spatially different predicted geometry** and therefore different masks?

Do not require low hidden cosine if geometry is target-specific.

# PART I — CMD/product direction

## 15. No GUI

Do not build UI.

If candidate reaches FIX_FOUND or strong PARTIAL, refactor minimal reusable API:

```python
model = load_buildreasonseg(...)
result = model.predict(image, instruction)
```

Result may expose:
- generated reasoning;
- predicted geometry;
- mask/logits.

Do not spend time on CLI cosmetics yet.

A polished `predict.py` can wait until the architecture passes paired validation.

# PART J — Data adequacy

## 16. Record WHU pseudo-instance limitations

During oracle/candidate error analysis classify failures caused by:
- merged touching buildings;
- border truncation;
- ambiguous pseudo-instance boundaries;
- insufficient instance density;
- component artifacts.

Do **not** change dataset in Task 6D.

If such errors materially dominate, recommend a dedicated dataset-selection task next.

# PART K — Artifacts

## 17. Create

```text
evaluation/task6d_oracle_prompt_diagnostic.json
evaluation/task6d_grounding_targets.json
evaluation/task6d_g0.json
evaluation/task6d_g1.json
evaluation/task6d_paired_probe.json
evaluation/task6d_representation.json
evaluation/task6d_error_analysis.json
evaluation/task6d_checkpoint_manifest.json
docs/task6d_spatial_grounding_bridge.md
```

`task6d_g1.json` only if G0 passes.

Checkpoints remain local/gitignored.

Create compact qualitative panels for at least 6 paired val images:
- image;
- instruction A/B;
- GT masks;
- predicted point/box A/B;
- predicted masks A/B.

No GUI.

# PART L — Tests

## 18. Tests

Cover:
1. deterministic interior point lies inside GT mask;
2. box tightly encloses GT mask;
3. normalized geometry valid;
4. GT geometry only used in supervision/oracle;
5. free inference uses predicted geometry only;
6. grounding loss gradients reach `[SEG]`/LoRA;
7. visual tower frozen;
8. SAM image encoder frozen;
9. paired samples have different target geometry;
10. oracle uses official SAM2 prompt encoder;
11. predicted path uses official SAM2 prompt encoder;
12. old arbitrary language sparse token absent from candidate;
13. visual cache value-preserving;
14. no test split;
15. no 4B / `[REF]` / SRE / SCL;
16. strict determinism retained.

Run:
`python -m pytest tests/ -q`

# PART M — Git / Watt

## 19. Git hygiene

Do not stage:
- weights/checkpoints;
- local feature caches;
- `.conda`;
- raw large traces;
- dataset JSONL changes.

Recommended commit:
`feat: add spatial grounding bridge`

## 20. Watt

Ignore UU.

Use established Watt ownership rule for final push only.

# PART N — Handoff

## 21. FROM_DSH

Include:
1. Verdict
2. Frozen Baseline
3. Oracle Point
4. Oracle Box
5. Geometry Choice
6. SpatialGroundingHead
7. G0 Training
8. G0 Metrics
9. G1 Training (if run)
10. Strict E2E
11. Paired Geometry Probe
12. Paired Mask Probe
13. Representation Diagnostics
14. WHU/Pseudo-instance Error Analysis
15. Runtime / VRAM
16. Tests
17. Git / Watt
18. Recommendation for next architecture task

## 22. Final DSH UI — Chinese only

Report:
- verdict;
- oracle point/box mIoU and paired pass;
- selected geometry;
- G0 geometry paired pass;
- G1 strict e2e mIoU;
- G1 mask paired /20;
- own-minus-cross margin;
- whether same-image instructions now produce different geometry/masks;
- main remaining failure;
- whether WHU is visibly limiting;
- peak VRAM/runtime;
- tests;
- commit/push;
- Watt handling.

# 23. STOP

After Task 6D:

**STOP.**

Do not automatically:
- add `[REF]`;
- add Spatial Relation Encoder;
- add Spatial Consistency Loss;
- scale to 4B;
- change dataset;
- run full training;
- build GUI.

Wait for ChatGPT review.
