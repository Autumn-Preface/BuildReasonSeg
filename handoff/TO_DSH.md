# TO_DSH — Task 6H: Counterfactual Pair-Aligned Dense Grounding v0.1

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: strengthen Task 6G's weak instruction-dependent query signal by explicitly exploiting the existing paired supervision:
>
> **same image + different instruction + different target**.
>
> Task 6H keeps the verified Task 6G dense query–visual architecture and changes only the training objective / pair-step semantics.
>
> Core hypothesis:
>
> > For two instructions on the same image, heatmap A must score target A higher than target B, and heatmap B must score target B higher than target A.
>
> This within-image counterfactual ranking removes image identity as an easy shortcut and directly trains “which building does this instruction select?”
>
> No 4B. No `[REF]`. No Spatial Relation Encoder. No Spatial Consistency Loss. No dataset migration. No GUI.

## 0. User-facing language

All narrative DSH UI/chat output must be **Chinese**.

Code, paths, raw logs and metric names may remain English.

## 1. Execution intent

This task is intentionally focused.

Allowed scientific change:
- pair-aware optimizer steps;
- one own-vs-cross counterfactual region-ranking loss.

Do not redesign the architecture freely.

# PART A — Freeze Task 6G

## 2. Accepted Task 6G evidence

Do not rerun unless needed for correctness:

- selected SAM2 grid: `256×256`, 32 channels;
- point oracle: strict mIoU `0.4883`, paired `18/20`;
- DenseSpatialGroundingHead:
  - query `LayerNorm(2048) → Linear(128)`;
  - visual `Conv1x1(32,128)`;
  - dot-product heatmap;
- G0 after 1500 steps:
  - point-inside-own `0.10`;
  - paired point `0/10`;
  - heatmap Dice `0.125`;
- implementation audit found no train/eval, gradient, frozen-backbone or saturation defect.

Accepted conclusion:

> With only per-sample BCE+Dice supervision, the query-to-map system does not make the query instruction-specific enough even on the 20-record G0 subset.

# PART B — Documentation hygiene

## 3. Correct stale network-state wording

`handoff/PROJECT_STATE.md` still contains a static identity-row wording equivalent to:

`Watt Toolkit stopped during training`

while recent tasks correctly record cases where Watt was pre-existing and left running untouched while model execution stayed offline.

Replace the static identity row with neutral wording such as:

> model/data execution is offline from local assets; Watt is transport-only when needed, and ownership rules determine whether it is left running or closed.

Do not alter historical per-task Watt records.

# PART C — Canonical pair construction

## 4. Train pairs

Use Task 6C paired `P`:

- 480 records;
- 240 source images;
- exactly 2 different instructions/targets per image.

Build deterministic pair objects:

```text
(image_id, sample_A, sample_B, target_mask_A, target_mask_B)
```

Assertions:

- same source image;
- different sample ids;
- different target component ids;
- target masks not identical;
- deterministic pair order.

No test split.

Record pair manifest/hash.

# PART D — Architecture remains Task 6G

## 5. No new module

Reuse exactly:

```text
image + instruction
→ pre-reasoning fixed [BOX] query
→ query hidden
→ DenseSpatialGroundingHead
→ heatmap logits
→ argmax point
```

Keep:
- Qwen3-VL-2B;
- text-only LoRA;
- `[BOX]` row;
- `[SEG]` row;
- 256×256 SAM2 high-res feature;
- frozen SAM2;
- Task 6G head dimensions.

Do not add:
- cross-attention blocks;
- coordinate channels;
- extra query tokens;
- visual LoRA;
- larger decoder.

# PART E — Pair-aware optimizer step

## 6. One optimizer step = one pair

For pair `(A,B)` from the same image:

1. compute/reuse one frozen SAM2 spatial feature;
2. Qwen query path for A → `q_A`;
3. Qwen query path for B → `q_B`;
4. produce heatmaps `H_A`, `H_B` against the same visual feature;
5. compute both per-sample losses;
6. compute the pair loss;
7. one backward / optimizer step on the combined objective.

Do not start a separate batch=2 engineering project.

Sequential forwards with both graphs retained are acceptable.

Scheduler horizon:

```text
240 pair optimizer steps / epoch
```

not 480 record steps.

# PART F — Counterfactual region-ranking loss

## 7. Region scores

Use Task 6G's 256×256 soft target masks:

```text
M_A
M_B
```

Define:

```text
region_score(H, M)
= sum(H * M) / max(sum(M), eps)
```

Compute:

```text
s_AA = region_score(H_A, M_A)
s_AB = region_score(H_A, M_B)
s_BB = region_score(H_B, M_B)
s_BA = region_score(H_B, M_A)
```

Record downsampled mask overlap for each pair.

Do not silently discard overlapping pairs.

## 8. Fixed pair loss

Use:

```text
margin = 1.0

L_cf_A = softplus(margin - (s_AA - s_AB))
L_cf_B = softplus(margin - (s_BB - s_BA))

L_cf = 0.5 * (L_cf_A + L_cf_B)
```

No margin sweep.

This loss requires own target preference, not merely query-vector separation.

## 9. Total pair-step loss

Retain Task 6G per-sample terms:

```text
L_heatmap_X =
    BCEWithLogits(H_X, M_X)
  + SoftDiceLoss(H_X, M_X)
```

Pair objective:

```text
L_total =
    0.5 * (L_reasoning_A + L_reasoning_B)
  + 1.0 * (L_heatmap_A + L_heatmap_B)
  + 2.0 * L_cf
```

No weight sweep.

Record:
- reasoning CE;
- BCE;
- Dice loss;
- own/cross region scores;
- own-minus-cross margins;
- `L_cf`;
- total.

# PART G — Pair ranking metrics

## 10. Define success

Pair-ranking pass iff:

```text
s_AA > s_AB
and
s_BB > s_BA
```

Strict-margin pass iff:

```text
s_AA - s_AB >= 1.0
and
s_BB - s_BA >= 1.0
```

# PART H — H0: 10-pair overfit

## 11. H0 subset

Use the same 20-record / 10-pair material as Task 6G G0 if possible.

Initialize from the clean Task 6G initialization convention.

Do not initialize from failed Task 6G G0 final weights except as an explicitly separate diagnostic.

Maximum:
- 1500 **pair optimizer steps**.

## 12. H0 evaluation

At checkpoints report:

- pair ranking /10;
- strict-margin /10;
- point-inside-own /20;
- paired point-selection /10;
- mean own-minus-cross region margin;
- same-image heatmap IoU;
- same-image point distance;
- heatmap Dice diagnostic;
- loss components.

### H0 gate

Require:

- pair ranking >= `9/10`;
- point-inside-own >= `18/20`;
- paired point-selection >= `9/10`;
- mean own-minus-cross region margin > `0.5`.

Heatmap Dice is diagnostic, not a gate.

If H0 fails after focused audit:

`COUNTERFACTUAL_QUERY_SIGNAL_FAILED`

and STOP.

# PART I — H0 audit if needed

## 13. Focused audit only

If H0 fails, inspect:

- own/cross region-score gradients into:
  - query projection;
  - Qwen LoRA;
  - `[BOX]` row;
  - visual 1×1 projection;
- A/B query-hidden distance before/after training;
- projected-query distance;
- shared visual feature bit identity;
- target-mask identity / pair ordering;
- pair-loss sign;
- whether `L_cf` decreases.

Do not add architecture modules during audit.

# PART J — H1: 240-pair mini-train

## 14. Run only if H0 passes

Train:
- 240 counterfactual pairs / epoch;
- maximum 8 epochs;
- corrected cosine scheduler over actual maximum pair-step budget;
- deterministic pair order;
- validate every epoch.

No mid-run hyperparameter change.

Validation:
- fixed 20 paired validation images = primary target-selection probe;
- fixed independent 120 validation records = per-record localization metrics.

Do not invent pairs inside the independent 120-record set.

# PART K — H1 selection and gate

## 15. Model selection

Lexicographic:

1. paired point-selection /20;
2. validation pair-ranking /20;
3. 120-record point-inside-target rate.

## 16. H1 gate

Require:

- paired point-selection >= `14/20`;
- paired region-ranking >= `14/20`;
- 120-record point-inside-target >= `0.60`.

If best epoch fails:

`COUNTERFACTUAL_DENSE_GROUNDING_FAILED`

Do not train SAM2 to compensate.

# PART L — Representation diagnosis

## 17. Test the hypothesis

At best H1 checkpoint compare to frozen Task 6F/6G:

Same-image/different-instruction:

- `[BOX]` raw cosine;
- centered cosine;
- hidden L2;
- projected-query cosine/L2;
- own-minus-cross region margin;
- heatmap IoU(A,B);
- predicted-point distance;
- correlation between:
  - query distance and GT target-point distance;
  - own-minus-cross margin and localization correctness.

Question:

> Did explicit same-image counterfactual supervision make the query/output target-specific instead of merely image-specific?

# PART M — H2 frozen-SAM2 segmentation

## 18. Run only if H1 passes

Inference:

```text
image + instruction
→ [BOX] query
→ dense heatmap
→ argmax predicted point
→ official frozen SAM2 positive-point prompt
→ frozen SAM2 decoder
→ mask
```

No GT point/mask.
No SAM2 training.

## 19. H2 metrics

On fixed 120 val + 20 paired:

- strict e2e mIoU;
- Dice;
- mask paired /20;
- own-target mask IoU;
- cross-target mask IoU;
- own-minus-cross margin;
- IoU(pred_A,pred_B);
- L1/L2/L3;
- query-family breakdown.

Compare:
- Task 6C P_C: `0.10604`, paired `0/20`;
- Point Oracle: `0.4876`, paired `18/20`;
- Box Oracle: `0.7506`, paired `20/20`.

# PART N — Verdicts

## 20. Use exactly one

### `COUNTERFACTUAL_GROUNDING_FIX_FOUND`

Require:
- H0 pass;
- H1 gate pass;
- H2 mask paired >= `14/20`;
- strict e2e mIoU >= `0.20`;
- own-minus-cross mask margin > `0.05`;
- no GT leakage.

### `COUNTERFACTUAL_GROUNDING_PARTIAL`

Pair ranking/query separation improves strongly and localization improves materially, but full H1/H2 gate is not met.

### `COUNTERFACTUAL_DENSE_GROUNDING_FAILED`

H0 passes but pair-aware objective does not generalize.

### `COUNTERFACTUAL_QUERY_SIGNAL_FAILED`

Even H0 cannot learn own-vs-cross preference after audit.

### `INVALID_EXPERIMENT`

Pair construction/sign/leakage/split/freeze or other correctness failure.

# PART O — Interpretation discipline

## 21. Do not overclaim novelty

If Task 6H works, freeze only:

> Explicit same-image counterfactual supervision teaches the semantic query to select between competing buildings in the same image.

Do not yet claim paper novelty.

Do not call this Spatial Consistency Loss; that name remains reserved for the later relation-level method.

# PART P — Dataset policy

## 22. No dataset migration in Task 6H

Keep WHU/BuildSpatialReason for controlled comparison.

Classify failures:
- pair ranking correct but point outside;
- pair ranking wrong;
- diffuse heatmap;
- wrong building;
- tiny target;
- border truncation;
- pseudo-instance ambiguity;
- complex relation.

If H1 starts working on simple pairs but data-quality categories dominate residual errors, flag dataset migration for next review.

# PART Q — Reusable API

## 23. If H1 passes

Expose/refactor:

```python
predict_heatmap(image, instruction)
predict_point(image, instruction)
predict_mask(image, instruction)
```

No GUI.

# PART R — Required artifacts

## 24. Create

```text
evaluation/task6h_pair_manifest.json
evaluation/task6h_token_setup.json
evaluation/task6h_h0_overfit.json
evaluation/task6h_h0_audit.json
evaluation/task6h_h1_training.json
evaluation/task6h_spatial_eval.json
evaluation/task6h_representation.json
evaluation/task6h_segmentation_eval.json
evaluation/task6h_paired_probe.json
evaluation/task6h_error_analysis.json
evaluation/task6h_checkpoint_manifest.json
docs/task6h_counterfactual_pair_grounding.md
```

`task6h_h0_audit.json` only if needed.
`task6h_segmentation_eval.json` only if H1 passes.

# PART S — Tests

## 25. Required tests

Cover at least:

1. exactly 240 canonical train pairs;
2. each pair shares one source image;
3. paired targets differ;
4. pair ordering deterministic;
5. shared SAM feature identical for A/B;
6. own/cross region scores use correct masks;
7. pair-loss sign correct;
8. lowering own score increases loss;
9. raising cross score increases loss;
10. pair optimizer step includes both samples before backward;
11. scheduler horizon counts pair steps;
12. Task 6G architecture dimensions unchanged;
13. Qwen visual tower frozen;
14. SAM2 fully frozen;
15. `[BOX]` remains pre-reasoning/causal;
16. Task 6F box head unused;
17. Task 6E loc-token path unused;
18. no test split;
19. H2 uses predicted point only;
20. no `[REF]` / SRE / SCL / 4B;
21. strict determinism;
22. PROJECT_STATE neutral Watt wording updated without rewriting history.

Run:

`python -m pytest tests/ -q`

# PART T — Git / Watt

## 26. Git hygiene

Do not stage:
- weights/checkpoints;
- feature caches;
- hidden dumps;
- `.conda`;
- dataset JSONL changes.

Recommended commit:
`feat: add counterfactual pair grounding loss`

Apply established Watt ownership rule only for final push.

# PART U — Handoff

## 27. `handoff/FROM_DSH.md`

Include:

1. Verdict
2. Frozen Task 6G Evidence
3. Canonical Pair Construction
4. Pair-Step Semantics
5. Counterfactual Loss
6. Trainables/Frozen Parameters
7. H0 Overfit
8. H0 Audit (if any)
9. H1 Training Curve
10. Pair-Ranking Metrics
11. Point Localization Metrics
12. Representation Diagnostics
13. H2 Segmentation
14. Paired Mask Probe
15. L1/L2/L3 + Query Breakdown
16. Error / Data Adequacy Analysis
17. Reusable Inference API
18. Runtime / VRAM
19. Tests
20. Git / Watt
21. Recommended next architecture step

## 28. Final DSH UI — Chinese only

Report:
- verdict;
- H0 pair ranking /10;
- H0 point-inside /20;
- H0 paired point /10;
- best H1 epoch;
- H1 validation pair ranking /20;
- H1 paired point /20;
- 120-record point-inside rate;
- own-minus-cross region margin;
- query/heatmap target-specificity change vs Task 6G;
- if H2 ran: strict e2e mIoU, mask paired /20, own-cross mask margin;
- dominant remaining failure;
- whether WHU now appears practically limiting;
- tests;
- commit/push;
- Watt handling.

# 29. STOP

After Task 6H:

**STOP.**

Do not automatically:
- add extra query tokens;
- add cross-attention refinement;
- add `[REF]`;
- add Spatial Relation Encoder;
- add Spatial Consistency Loss;
- scale to 4B;
- migrate dataset;
- run full training;
- build GUI.

Wait for ChatGPT review.
