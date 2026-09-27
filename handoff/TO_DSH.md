# TO_DSH — Task 6F: Target-Aware `[BOX]` Query Grounding v0.1

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: replace Task 6E's failed 256-way autoregressive coordinate vocabulary with a dedicated **target-aware spatial query token** whose hidden state is directly supervised for box geometry.
>
> Accepted evidence:
> - Oracle BOX → SAM2: strict mIoU `0.7506`, paired `20/20`;
> - Task 6D.1: end-of-reasoning `[SEG]` hidden has no practically decodable target geometry under the audited readouts;
> - Task 6E: explicit coordinate tokens overfit 20 records but fail on the 480-record paired mini-train with runaway location-token sequences.
>
> Core hypothesis:
>
> > Spatial grounding should be read from a dedicated query token placed **before reasoning text**, so its state is conditioned by image + instruction instead of being dominated by the small reasoning-template vocabulary.
>
> The `[BOX]` query is a fixed learned control token, identical for every sample. It contains no GT information.
>
> No 4B. No `[REF]`. No Spatial Relation Encoder. No Spatial Consistency Loss. No dataset change. No GUI.

## 0. User-facing language

All narrative DSH UI/chat output must be **Chinese**.
Code, paths, metric names and raw logs may remain English.

# PART A — Freeze valid evidence

## 1. Accepted facts

Do not rerun unless a correctness issue is found:
- continuous Oracle BOX → SAM2: strict mIoU `0.7506`, paired `20/20`;
- Task 6E B=256 quantized Oracle BOX: strict mIoU `0.7343`, paired `20/20`;
- Task 6D.1: legacy end-of-reasoning `[SEG]` hidden is practically not decodable for target geometry under tested readouts;
- Task 6E: E0 20-record coordinate-token overfit succeeds, E1 480-record autoregressive coordinate-token training fails structurally `0/120`;
- current evidence does not indicate WHU or SAM2 as the binding constraint.

# PART B — Retire the failed coordinate-token path

## 2. Preserve history, do not reuse its failed representation

Keep Task 6E code/artifacts for ablation/history.

Task 6F must **not**:
- autoregressively emit `<loc_*>`;
- use location-token CE;
- parse location-token runs;
- initialize from failed Task 6E E1 weights.

If loc tokens remain in the tokenizer for code-compatibility, they must be unused and frozen.

# PART C — Clean initialization

## 3. Initialization source

Use the same clean base/seed convention as Task 6E E1:

- Qwen3-VL-2B-Instruct;
- existing `[SEG]`;
- text-only LoRA;
- add exactly one active Task 6F query token `[BOX]`.

Do not carry trained Task 6E `<loc_*>` rows into the candidate.

Record exact initialization provenance, token ids, and optimizer coverage.

# PART D — Query-token causal placement

## 4. `[BOX]` is a fixed input query

Construct the sequence as:

```text
USER:
image + instruction

ASSISTANT PREFIX:
[BOX]

ASSISTANT TARGET:
reasoning_zh [SEG] EOS
```

The `[BOX]` token is deterministically appended immediately after the assistant-generation prefix during both training and inference.

It is the same token for every sample and contains no target label.

### Critical causal property

The hidden state read at `[BOX]` may attend to:
- image tokens;
- user instruction;
- normal system/chat prefix;
- its own `[BOX]` embedding.

It must **not** attend to:
- `reasoning_zh`;
- GT target box;
- `[SEG]`;
- any future assistant token.

Add a direct causal-position/mask test.

This is the central difference from Task 6D's end-of-reasoning `[SEG]` readout.

# PART E — Target-aware box head

## 5. Architecture

Add:

`TargetAwareBoxHead`

Input:
`[BOX] hidden: 2048-d`

Use:

```text
LayerNorm(2048)
→ Linear(2048, 512)
→ GELU
→ Linear(512, 4)
→ sigmoid
→ differentiable box canonicalization
```

Output normalized `(x1,y1,x2,y2)` with canonical ordering.

This deliberately reuses a simple readout so the experimental change is localized to **where the supervised representation is formed**.

Do not use:
- coordinate tokens;
- a 256-way classifier;
- the legacy `[SEG]` geometry readout;
- SAM mask loss.

## 6. Box supervision

Use the same tight normalized GT box definition frozen in Task 6D.

GT geometry is used only for:
- SmoothL1 supervision;
- evaluation.

Inference must use only:
- image;
- instruction;
- constant `[BOX]`;
- learned model/head parameters.

# PART F — Loss and trainables

## 7. Objective

Use:

```text
L_total =
    1.0 * L_reasoning
  + 5.0 * L_box
```

where:
- `L_reasoning` = causal CE only for `reasoning_zh [SEG] EOS`;
- `L_box` = SmoothL1(pred_box, GT_box).

Do not predict `[BOX]`; it is an inserted query.

Do not use:
- location-token CE;
- mask BCE/Dice;
- pairwise/contrastive losses;
- hyperparameter sweeps.

Record both raw losses.

## 8. Trainable set

Train:
- text-only LoRA;
- `[BOX]` query row;
- existing `[SEG]` row;
- TargetAwareBoxHead.

Freeze:
- Qwen base;
- Qwen visual tower;
- all SAM2 in F0/F1;
- old Projection MLP;
- Task 6D SpatialGroundingHead;
- Task 6E loc rows, if present.

Verify optimizer coverage.

# PART G — Stage F0: implementation proof

## 9. 20-record paired overfit

Use the same deterministic 20-record / 10 paired-image style as Task 6E E0.

Maximum:
- 1500 optimizer steps.

Use corrected scheduler horizon over the real step budget.

Evaluate through the **inference-form query path**:

```text
image + instruction
→ append constant [BOX]
→ Qwen forward
→ read [BOX] hidden
→ TargetAwareBoxHead
→ predicted box
```

No future reasoning tokens may be teacher-forced into the query representation.

### F0 gate

Require:
- mean train box IoU >= `0.85`;
- geometry paired >= `9/10`;
- same-image two instructions produce non-identical predicted boxes;
- no GT leakage.

If implementation audit cannot make F0 pass:

`TARGET_QUERY_IMPLEMENTATION_FAILED`

and STOP.

# PART H — Stage F1: 480-paired mini-train

## 10. Training budget

Only if F0 passes.

Train on Task 6C paired `P`:
- 480 records / 240 images × 2 targets;
- deterministic training;
- maximum **8 epochs**.

Use:
- one corrected cosine scheduler over the full actual maximum step budget;
- checkpoint each epoch;
- validation each epoch.

Do not change LR/loss mid-run.

## 11. Early stopping / selection

Validate using the query path on:
- fixed 120 validation records;
- fixed 20 paired validation images.

Model-selection priority:
1. geometry paired /20;
2. mean val box IoU;
3. center-inside-target rate.

Early stop if geometry paired and val box IoU do not improve for 3 consecutive epochs after epoch 3.

Do not select by training loss.

# PART I — Geometry evaluation

## 12. Main metrics

For each epoch report:
- train box IoU;
- val mean/median box IoU;
- center-inside-target rate;
- coordinate MAE;
- geometry paired /20;
- own-target box IoU;
- cross-target box IoU;
- own-minus-cross box margin;
- same-image predicted-box L1 distance;
- per-coordinate prediction spread;
- L1/L2/L3 breakdown;
- query-family breakdown.

### F1 gate

Require:
- geometry paired >= `14/20`;
- val mean box IoU >= `0.35`;
- center-inside-target >= `0.70`.

If best epoch fails:

`TARGET_AWARE_QUERY_FAILED`

Do not train SAM to compensate.

# PART J — Representation diagnosis

## 13. Compare new `[BOX]` to legacy `[SEG]`

At best F1 checkpoint, report for `[BOX]`:
- same-image/different-instruction cosine;
- centered cosine;
- L2;
- effective rank;
- correlation between hidden-pair distance and GT-box-pair distance.

Use frozen Task 6D.1 legacy `[SEG]` values as comparison; do not rerun the full old audit unless necessary.

Question:

> Did moving the supervised query **before reasoning** create target-specific representation variation rather than template-dominated variation?

Do not infer causality from cosine alone.

# PART K — Reasoning compatibility

## 14. Secondary language compatibility

On fixed 20 validation samples, continue generation from:

`image + instruction + fixed [BOX]`

Report:
- exactly-one `[SEG]` rate;
- EOS termination rate;
- reasoning-template/operation-chain diagnostic.

These remain template diagnostics, not proof of reasoning.

Do not gate F1 geometry on reasoning exact match.

# PART L — Stage F2: frozen-SAM end-to-end test

## 15. Run only if F1 passes

For each validation sample:

```text
image + instruction
→ constant [BOX] query
→ predicted box
→ official frozen SAM2 box prompt encoder
→ frozen SAM2 mask decoder
→ mask
```

No SAM training.
No GT geometry.

## 16. F2 metrics

On 120 val + 20 paired:
- strict e2e mIoU;
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
- continuous Oracle BOX: `0.7506`, paired `20/20`;
- Task 6E quantized Oracle: `0.7343`, paired `20/20`.

# PART M — Verdicts

## 17. Use exactly one

### `TARGET_AWARE_QUERY_FIX_FOUND`

Require:
- F0 pass;
- F1 geometry gate pass;
- F2 mask paired >= `14/20`;
- strict e2e mIoU >= `0.20`;
- mask own-minus-cross > `0.05`;
- no GT leakage.

### `TARGET_AWARE_QUERY_PARTIAL`

Query becomes target-specific and geometry improves materially, but full geometry/mask gate is not met.

### `TARGET_AWARE_QUERY_FAILED`

Implementation is valid and F0 passes, but 480-record query grounding does not generalize.

### `TARGET_QUERY_IMPLEMENTATION_FAILED`

F0 cannot be made to work under verified implementation.

### `INVALID_EXPERIMENT`

Leakage, split/checkpoint mismatch, causal-position bug or other correctness failure.

# PART N — Dataset policy

## 18. WHU remains provisional

Do not migrate data in Task 6F.

Classify failures involving:
- tiny targets;
- border truncation;
- touching/merged pseudo-instances;
- visually ambiguous targets;
- L3/multi-hop queries.

If geometry works on simple cases but failures concentrate in data-quality categories, record that as evidence for a future dataset task.

# PART O — Product-facing core API

## 19. If F1 passes, expose reusable inference

Create/refactor:

```python
predict_box(image, instruction)
predict_mask(image, instruction)
generate_reasoning(image, instruction)
```

No GUI.
Do not spend time on CLI cosmetics yet.

# PART P — Required artifacts

## 20. Create

```text
evaluation/task6f_token_setup.json
evaluation/task6f_f0_overfit.json
evaluation/task6f_f1_training.json
evaluation/task6f_geometry_eval.json
evaluation/task6f_representation.json
evaluation/task6f_reasoning_compat.json
evaluation/task6f_segmentation_eval.json
evaluation/task6f_paired_probe.json
evaluation/task6f_error_analysis.json
evaluation/task6f_checkpoint_manifest.json
docs/task6f_target_aware_box_query.md
```

`task6f_segmentation_eval.json` only if F1 passes.

Weights/checkpoints stay local/gitignored.

# PART Q — Tests

## 21. Required tests

Cover at least:
1. `[BOX]` is one token and unique from `[SEG]`;
2. `[BOX]` is inserted as a fixed query, not predicted from GT;
3. `[BOX]` hidden cannot attend to future reasoning/GT tokens;
4. box target uses frozen Task 6D convention;
5. query-head output canonical/in `[0,1]`;
6. GT box is supervision only;
7. inference needs only image + instruction + constant `[BOX]`;
8. `[BOX]` row receives gradient;
9. `[SEG]` compatibility row remains valid;
10. LoRA remains text-only;
11. visual tower frozen;
12. SAM2 fully frozen in F0/F1/F2;
13. old `[SEG]` grounding head unused;
14. Task 6E loc-token loss/parser unused;
15. if loc tokens remain, their rows are frozen/excluded;
16. scheduler horizon equals actual max-step budget;
17. F0 evaluation does not teacher-force future reasoning;
18. F1 model selection follows paired→box-IoU→center rule;
19. F2 uses predicted box only;
20. no test split;
21. no 4B / `[REF]` / SRE / SCL;
22. strict determinism retained.

Run:
`python -m pytest tests/ -q`

# PART R — Git / Watt

## 22. Git hygiene

Do not stage:
- weights/checkpoints;
- hidden dumps;
- local model/tokenizer snapshots;
- `.conda`;
- feature caches;
- dataset JSONL edits.

Recommended commit:
`feat: add target-aware box query`

Ignore UU completely.
Apply established Watt ownership rule only for final push.

# PART S — Handoff

## 23. FROM_DSH

Include:
1. Verdict
2. Frozen Evidence
3. Initialization
4. Query-Token Causal Placement
5. TargetAwareBoxHead
6. Trainables / Loss
7. F0 Overfit
8. F1 Training Curve
9. Geometry Metrics
10. Paired Geometry Probe
11. `[BOX]` Representation Diagnostics
12. Reasoning Compatibility
13. F2 Segmentation
14. Paired Mask Probe
15. Error / Data Adequacy Analysis
16. Reusable Inference API
17. Runtime / VRAM
18. Tests
19. Git / Watt
20. Recommended next architecture step

## 24. Final DSH UI — Chinese only

Report:
- Task 6F verdict;
- F0 box IoU / paired;
- best F1 epoch;
- val box IoU;
- center-inside rate;
- geometry paired /20;
- same-image predicted-box distance;
- whether `[BOX]` is more target-specific than legacy `[SEG]`;
- reasoning `[SEG]` emission compatibility;
- if F2 ran: strict e2e mIoU, mask paired /20, own-cross margin;
- dominant remaining failure;
- whether WHU appears limiting;
- tests;
- commit/push;
- Watt handling.

# 25. STOP

After Task 6F:

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
