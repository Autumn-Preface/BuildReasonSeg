# TO_DSH — Task 6G: Dense Query–Visual Spatial Grounding Map v0.1

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: replace Task 6F's failed **global coordinate regression** with an explicit **query-to-spatial-feature localization map**.
>
> Accepted evidence:
> - Oracle BOX → frozen SAM2: strict mIoU `0.7506`, paired `20/20`.
> - Oracle POINT → frozen SAM2: strict mIoU `0.4876`, paired `18/20`.
> - Task 6F `[BOX]` query is causally clean and can overfit 20 records, but on 480 records direct box regression fails: val box IoU `0.025`, geometry paired `0/20`.
> - Task 6F query is no longer reasoning-template dominated, but is primarily image-dominated; hidden-distance vs GT-box-distance correlation is `0.476`.
>
> Key diagnosis:
>
> A single global hidden vector + SmoothL1 over four normalized box edges is a poor localization objective for WHU-scale tiny buildings. Task 6G preserves spatial structure and trains localization directly on a dense 2-D visual feature map.
>
> Candidate chain:
>
> **language query ↔ dense visual feature fusion → spatial grounding map → predicted point → frozen SAM2 segmentation**
>
> This is a capability-building architecture experiment. Do not claim generic query-to-map fusion as the final paper novelty.

## 0. User priority

Priority remains:
1. correct model capability;
2. train / predict / evaluate pipeline;
3. architecture suitable for later paper innovation;
4. UI/GUI last.

No GUI/Web UI.
Do not reopen generic performance tuning.

# PART A — Freeze Task 6F correctly

## 1. Accepted Task 6F interpretation

Do not rerun Task 6F unless a correctness issue is found.

Freeze:
- F0: train box IoU `0.9046`, paired `10/10`;
- F1 best epoch 8:
  - val box IoU `0.0250`;
  - center-inside `0.017`;
  - coordinate MAE `0.1543`;
  - geometry paired `0/20`;
- query representation:
  - same-image centered cosine `0.626`;
  - different-image centered cosine `0.018`;
  - hidden-pair-distance vs GT-box-distance correlation `0.476`;
- reasoning compatibility: `[SEG]` emission `1.0`, EOS `1.0`.

Do not freeze the stronger statement that “8 epochs were simply too few”.

Task 6F uses `SmoothL1(beta=1.0)` over normalized box coordinates. For small buildings, a modest coordinate error can yield a numerically small loss while destroying overlap. Task 6G therefore tests a spatially aligned objective instead of blindly extending training.

# PART B — Spatial-grid oracle audit

## 2. Candidate SAM2 spatial feature levels

Verify exact current tensors/shapes. Expected from prior measurement:
- main image embedding: approximately `C=256, H=W=64`;
- high-resolution feature: approximately `C=64, H=W=128`;
- high-resolution feature: approximately `C=32, H=W=256`.

Do not assume channel ordering.

## 3. Grid-snapped POINT oracle

For every validation target:

1. derive the frozen Task 6D deterministic interior point (distance-transform maximum);
2. snap it to the nearest cell centre of each candidate grid:
   - 64×64
   - 128×128
   - 256×256
3. feed the snapped point through the official frozen SAM2 positive-point prompt path.

Use:
- same 120 validation records;
- same 20 paired validation images.

Report for each grid:
- normalized displacement;
- displacement in original 512-pixel units;
- SAM strict mIoU;
- Dice;
- paired mask pass /20;
- own-target vs cross-target mask IoU.

Create:
`evaluation/task6g_grid_oracle.json`

### Grid selection rule

Continuous point oracle:
- mIoU `0.4876`
- paired `18/20`.

Choose the **smallest grid** satisfying:
- paired mask >= `18/20`;
- strict mIoU >= `0.4576`.

If none passes:
`DENSE_GRID_POINT_PATH_INADEQUATE`
and STOP before training.

# PART C — Reuse the causal query

## 4. Query token

Reuse Task 6F's pre-reasoning fixed `[BOX]` query token for controlled comparison.

Sequence remains:

```text
image + instruction
→ fixed [BOX] query
→ reasoning_zh [SEG] EOS
```

The token name may remain historical; in Task 6G it functions as a general spatial grounding query.

It must remain unable to attend to future reasoning/GT tokens.

Do not add:
- `<loc_*>`;
- a new query token;
- `[REF]`.

# PART D — Dense Spatial Grounding Head

## 5. Architecture

Implement:

`DenseSpatialGroundingHead`

Inputs:

```text
query_hidden: [B, 2048]
spatial_feature: [B, C, H, W]
```

Use:

```text
query_hidden
→ LayerNorm(2048)
→ Linear(2048,128)
→ q ∈ R^128

SAM2 spatial feature
→ Conv1x1(C,128)
→ K ∈ R^(128×H×W)

heatmap_logits[y,x]
= dot(q, K[:,y,x]) / sqrt(128)
```

No transformer block.
No relation module.
No large decoder.

Trainable in this head:
- query projection;
- 1×1 visual-key projection.

An optional scalar bias is allowed.

The experimental question is whether retaining the 2-D visual grid fixes global-regression failure.

## 6. Frozen visual backbone

SAM2 image encoder and source feature tensors remain frozen.

Gradients may flow through the new 1×1 projection, never into SAM2.

Reuse existing SAM feature computation/cache infrastructure where possible.

Do not cache anything downstream of the trainable projection.

# PART E — Dense supervision

## 7. GT spatial target

Use the **instruction-selected target building mask**, not the full building semantic mask.

For selected H×W:
- downsample binary target mask with area interpolation;
- retain soft occupancy values in `[0,1]`;
- assert target mass > 0 for every sample.

GT mask is supervision only.
No GT enters inference.

## 8. Heatmap loss

Use exactly:

```text
L_heatmap =
    1.0 * BCEWithLogits(heatmap_logits, soft_target)
  + 1.0 * SoftDice(sigmoid(heatmap_logits), soft_target)
```

Total:

```text
L_total =
    1.0 * L_reasoning
  + 2.0 * L_heatmap
```

`L_reasoning` remains CE over:
`reasoning_zh [SEG] EOS`.

Do not use:
- coordinate SmoothL1;
- GIoU/box loss;
- SAM-output mask loss;
- contrastive/pairwise loss;
- loss-weight sweep.

Record raw BCE, Dice, heatmap loss, reasoning CE.

# PART F — Point extraction

## 9. Predicted point

Primary inference rule:

```text
(y*,x*) = argmax(heatmap_logits)
point = centre of selected grid cell
```

Convert to normalized coordinates, then to official SAM2 point coordinates.

No threshold tuning.
No GT-guided component selection.

A soft-argmax/expected coordinate may be reported as diagnostic only.

# PART G — Trainables

## 10. Trainable set

Train:
- text-only Qwen LoRA;
- `[BOX]` query row;
- `[SEG]` row;
- DenseSpatialGroundingHead.

Freeze:
- Qwen base;
- Qwen visual tower;
- all SAM2;
- Task 6F TargetAwareBoxHead;
- Task 6D SpatialGroundingHead;
- old Projection MLP;
- Task 6E loc rows if present.

Verify optimizer coverage.

# PART H — Stage G0: overfit sanity

## 11. 20-record paired overfit

Use 20 records / 10 same-image different-target pairs.

Maximum:
- 1500 optimizer steps.

Use corrected scheduler horizon.

Evaluate through the real path:

```text
image + instruction
→ fixed [BOX]
→ query hidden
→ dense heatmap
→ argmax point
```

### G0 metrics

Report:
- low-res soft-mask Dice / IoU;
- predicted point inside own target;
- predicted point inside paired other target;
- paired point selection /10;
- same-image heatmap IoU(A,B);
- same-image point distance.

### G0 gate

Require:
- point inside own target >= `19/20`;
- paired point selection >= `9/10`;
- mean heatmap Dice >= `0.80`;
- paired instructions produce distinct points.

If unresolved after implementation audit:
`DENSE_GROUNDING_IMPLEMENTATION_FAILED`
and STOP.

# PART I — Stage G1: 480 paired mini-train

## 12. Training

Only if G0 passes.

Train on Task 6C paired `P`:
- 480 records / 240 images × 2 targets;
- maximum 8 epochs;
- corrected scheduler across actual max-step budget;
- validate each epoch.

No mid-run recipe changes.

Early stop if paired point metric and inside-target rate fail to improve for 3 consecutive epochs after epoch 3.

## 13. Model selection

Lexicographic:
1. paired point selection /20;
2. point-inside-target rate;
3. validation heatmap Dice.

Do not select by training loss.

# PART J — G1 evaluation

## 14. Metrics

On fixed 120 val + 20 paired report:
- heatmap BCE;
- heatmap Dice;
- binary heatmap IoU at fixed threshold 0.5 (diagnostic);
- point-inside-target rate;
- normalized point error to frozen deterministic interior point;
- original-512-pixel point error;
- paired point selection /20;
- same-image predicted-point distance;
- same-image heatmap IoU(A,B);
- L1/L2/L3 breakdown;
- query-family breakdown.

### G1 gate

Require:
- point-inside-target >= `0.70`;
- paired point selection >= `14/20`;
- val heatmap Dice >= `0.35`.

If best epoch fails:
`DENSE_SPATIAL_GROUNDING_FAILED`

Do not train SAM2 to compensate.

# PART K — Representation/fusion diagnosis

## 15. Diagnose what changed

At best G1 checkpoint report:
- `[BOX]` hidden same-image/different-instruction cosine;
- centered cosine;
- L2;
- effective rank;
- projected query `q` same-image distance;
- heatmap A/B IoU;
- correlation:
  - query-hidden distance vs GT-point distance;
  - heatmap distance vs GT-point distance.

Compare to Task 6F.

Question:

> Does the weak target-specific component become usable when matched against a dense spatial feature map?

Do not interpret cosine alone.

# PART L — Stage G2: frozen-SAM2 point segmentation

## 16. Run only if G1 passes

For every validation sample:

```text
image + instruction
→ fixed [BOX] query
→ dense heatmap
→ argmax predicted point
→ official frozen SAM2 positive-point prompt
→ frozen SAM2 mask decoder
→ mask
```

No GT geometry.
No SAM2 training.

## 17. G2 metrics

On 120 val + 20 paired:
- strict end-to-end mIoU;
- Dice;
- mask paired /20;
- own-target IoU;
- cross-target IoU;
- own-minus-cross margin;
- IoU(pred_A,pred_B);
- L1/L2/L3 breakdown;
- query-family breakdown.

Compare with:
- Task 6C P_C: `0.10604`, paired `0/20`;
- continuous Point Oracle: `0.4876`, paired `18/20`;
- continuous Box Oracle: `0.7506`, paired `20/20`.

# PART M — Verdicts

## 18. Use exactly one

### `DENSE_SPATIAL_GROUNDING_FIX_FOUND`
Require:
- G0 pass;
- G1 gate pass;
- G2 mask paired >= `14/20`;
- strict e2e mIoU >= `0.20`;
- own-minus-cross mask margin > `0.05`;
- no GT leakage.

### `DENSE_SPATIAL_GROUNDING_PARTIAL`
Heatmap/point becomes target-specific and materially improves localization, but full gate is not met.

### `DENSE_SPATIAL_GROUNDING_FAILED`
Implementation valid and G0 passes, but dense grounding does not generalize.

### `DENSE_GROUNDING_IMPLEMENTATION_FAILED`
G0 cannot pass after audit.

### `DENSE_GRID_POINT_PATH_INADEQUATE`
Spatial grid is too coarse even for oracle points.

### `INVALID_EXPERIMENT`
Leakage, wrong mask, split mismatch, backbone-freeze violation or correctness failure.

# PART N — Architecture interpretation

## 19. Not the final novelty claim

If Task 6G works, freeze only:

> Dense semantic–visual spatial grounding solves the functional target-selection bottleneck better than global `[SEG]` readout, coordinate-token generation, or global box regression.

Do not claim generic dot-product query-to-map fusion as final novelty.

A successful spatial map may later support:
- geometry-verifiable relation supervision;
- reference-object grounding;
- Spatial Relation Encoder;
- Spatial Consistency Loss.

Only after basic target selection works.

# PART O — Dataset policy

## 20. WHU remains provisional

Do not migrate data in Task 6G.

Classify errors:
- tiny target;
- border truncation;
- touching/merged pseudo-instance;
- visual ambiguity;
- relation complexity;
- heatmap on wrong building;
- diffuse/no-localization heatmap.

If simple L1 examples work but tiny/ambiguous cases dominate remaining errors, record that as evidence for later dataset migration.

# PART P — Reusable inference plumbing

## 21. If G1 passes

Expose reusable:
```python
predict_heatmap(image, instruction)
predict_point(image, instruction)
predict_mask(image, instruction)
```

No GUI.

# PART Q — Required artifacts

## 22. Create

```text
evaluation/task6g_grid_oracle.json
evaluation/task6g_token_setup.json
evaluation/task6g_g0_overfit.json
evaluation/task6g_g1_training.json
evaluation/task6g_spatial_eval.json
evaluation/task6g_representation.json
evaluation/task6g_segmentation_eval.json
evaluation/task6g_paired_probe.json
evaluation/task6g_error_analysis.json
evaluation/task6g_checkpoint_manifest.json
docs/task6g_dense_spatial_grounding.md
```

`task6g_segmentation_eval.json` only if G1 passes.

If G2 runs, create at least 6 paired diagnostic panels:
image, instruction A/B, target masks, heatmaps, predicted points, SAM masks.

No GUI.

# PART R — Tests

## 23. Required tests

Cover at least:
1. selected SAM feature level/shape verified;
2. grid snapping uses cell centres;
3. grid oracle selection obeys rule;
4. mask downsampling uses only selected target instance;
5. target soft mass > 0;
6. `[BOX]` remains pre-reasoning and causally clean;
7. DenseSpatialGroundingHead output shape;
8. query and visual projections receive gradients;
9. SAM2 encoder/features frozen;
10. GT heatmap absent at inference;
11. argmax point deterministic;
12. no GT-guided repair;
13. G0 same-image pairs have different targets;
14. scheduler horizon uses actual max steps;
15. model selection paired→inside→Dice;
16. G2 uses predicted point only;
17. SAM2 fully frozen in G2;
18. Task 6F box head unused;
19. Task 6E loc path unused;
20. no test split;
21. no 4B / `[REF]` / SRE / SCL;
22. strict determinism retained.

Run:
`python -m pytest tests/ -q`

# PART S — Git / Watt

## 24. Git hygiene

Do not stage:
- weights/checkpoints;
- cached feature tensors;
- local model/tokenizer snapshots;
- `.conda`;
- dataset JSONL edits.

Recommended commit:
`feat: add dense spatial grounding map`

Ignore UU completely.
Apply established Watt ownership rule only for final push.

# PART T — Handoff

## 25. `handoff/FROM_DSH.md`

Include:
1. Verdict
2. Frozen Task 6F Evidence
3. Grid Oracle
4. Selected Spatial Feature Level
5. DenseSpatialGroundingHead
6. Target Heatmap / Loss
7. Trainable/Frozen Parameters
8. G0 Overfit
9. G1 Training Curve
10. Spatial Localization Metrics
11. Paired Point Probe
12. Representation/Fusion Diagnostics
13. G2 Segmentation
14. Paired Mask Probe
15. L1/L2/L3 + Query Breakdown
16. Error / Data Adequacy Analysis
17. Reusable Inference API
18. Runtime / VRAM
19. Tests
20. Git / Watt
21. Recommended next architecture task

## 26. Final DSH UI — Chinese only

Report:
- verdict;
- 64/128/256 grid oracle and selected grid;
- G0 inside / paired / heatmap Dice;
- best G1 epoch;
- val inside rate;
- paired point /20;
- val heatmap Dice;
- same-image heatmap IoU / point distance;
- whether dense fusion improved target specificity vs Task 6F;
- if G2 ran: strict e2e mIoU, mask paired /20, own-cross margin;
- dominant remaining failure;
- whether WHU now appears practically limiting;
- tests;
- commit/push;
- Watt handling.

# 27. STOP

After Task 6G:

**STOP.**

Do not automatically:
- add `[REF]`;
- add Spatial Relation Encoder;
- add Spatial Consistency Loss;
- scale to 4B;
- migrate dataset;
- run full training;
- build GUI.

Wait for ChatGPT review.
