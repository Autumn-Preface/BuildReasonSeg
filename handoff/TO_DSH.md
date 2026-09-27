# TO_DSH — Task 6H.1: Spatial-Softmax Point Supervision + Bounded Counterfactual Grounding

> Status: **ACTIVE**
>
> Repository: `BuildReasonSeg`
>
> Purpose: correct Task 6H's scale-degenerate logit-ranking objective **without changing the Task 6G/6H architecture**.
>
> Accepted diagnosis from Task 6H:
>
> - same-image pair machinery is correct;
> - logit-space pair ranking reaches 10/10 and `L_cf` ≈ 0.02;
> - mean absolute logits grow from ~0.15 to ~16.3;
> - probability-space own-vs-cross margin remains only ~+0.112;
> - point-inside-own remains 3/20 and paired point 0/10.
>
> Therefore the historical Task 6H verdict is preserved, but the accepted causal conclusion is:
>
> **the Task 6H logit-space ranking objective is scale-degenerate; Task 6H does not establish that the query signal itself is unlearnable.**
>
> Task 6H.1 replaces the objective with one directly aligned to the deployed inference operation:
>
> **spatial softmax → target point-cell classification + bounded own-vs-cross target-region probability mass.**
>
> No architecture change. No 4B. No `[REF]`. No SRE. No SCL. No dataset migration. No GUI.

## 0. User-facing language

All DSH narrative/UI output must be **Chinese**.

Code, paths, metric keys and raw logs may remain English.

# PART A — Freeze architecture and evidence

## 1. Architecture is frozen

Reuse exactly Task 6G/6H:

```text
image + instruction
→ fixed pre-reasoning [BOX] query
→ Qwen query hidden
→ DenseSpatialGroundingHead
   query: LayerNorm(2048) → Linear(128)
   visual: Conv1x1(32,128) on frozen SAM2 256×256 feature
   dot-product heatmap
→ point from heatmap
```

Keep:
- Qwen3-VL-2B-Instruct;
- text-only LoRA;
- `[BOX]` query row;
- `[SEG]` row;
- SAM2 256×256 / 32-channel high-res feature;
- all SAM2 frozen;
- Task 6G head dimensions unchanged;
- same canonical 240 train pairs;
- same fixed validation material.

Do not add:
- extra query tokens;
- coordinate channels;
- cross-attention;
- visual LoRA;
- larger decoder;
- new dataset.

## 2. Frozen downstream ceiling

Do not rerun unless a correctness issue is found:

- 256-grid snapped point → frozen SAM2:
  - strict mIoU `0.4883`;
  - paired mask `18/20`.

# PART B — Correct Task 6H interpretation in docs

## 3. Preserve numbers, narrow the causal wording

Do not rewrite historical Task 6H measurements.

Where project-state/handoff summary implies:

> counterfactual query signal itself failed

supersede it with:

> Task 6H successfully learned the specified own-vs-cross logit ranking, but the loss was scale-degenerate: it could be minimized by magnifying logits without moving the heatmap peak onto the target. Query learnability remains unresolved until a bounded, localization-aligned objective is tested.

Historical verdict may remain `COUNTERFACTUAL_QUERY_SIGNAL_FAILED` as the task's original machine verdict, but the new accepted interpretation must be explicit.

# PART C — Deterministic target point-cell

## 4. Point target

For each instruction-selected target mask:

1. use the frozen Task 6D deterministic interior point;
2. snap it to the selected 256×256 grid cell using the Task 6G convention;
3. convert `(x_cell, y_cell)` to:

```text
target_index = y_cell * 256 + x_cell
```

This is the primary spatial training target.

GT point/cell is supervision only.

At inference the point remains:

```text
argmax(heatmap_logits)
```

with no GT repair.

# PART D — Spatial-softmax point classification

## 5. Primary localization loss

For:

```text
H ∈ R^(256×256)
```

flatten to 65,536 spatial classes and use:

```text
L_point = CrossEntropy(H.flatten(), target_index)
```

No class weighting.
No temperature sweep.
No label smoothing.

Report:
- point CE;
- top-1 target-cell accuracy;
- top-5 / top-25 cell accuracy;
- target-cell probability;
- argmax point-inside-target.

# PART E — Bounded counterfactual region mass

## 6. Spatial probability distribution

Define:

```text
P = softmax(H.flatten(), dim=-1).reshape(256,256)
```

For target soft mask `M`:

```text
mass(P, M) = sum(P * M)
```

Do **not** divide by target area.

For pair A/B:

```text
p_AA = mass(P_A, M_A)
p_AB = mass(P_A, M_B)
p_BB = mass(P_B, M_B)
p_BA = mass(P_B, M_A)
```

All are bounded in `[0,1]`.

## 7. Bounded pair-preference loss

Use:

```text
eps = 1e-8

L_cf_A = -log((p_AA + eps) / (p_AA + p_AB + 2*eps))
L_cf_B = -log((p_BB + eps) / (p_BB + p_BA + 2*eps))

L_cf = 0.5 * (L_cf_A + L_cf_B)
```

No margin hyperparameter.

Report:
- `p_AA`, `p_AB`, `p_BB`, `p_BA`;
- probability-mass own-minus-cross margins;
- own/(own+cross) ratios;
- `L_cf`.

Pair ranking passes iff:

```text
p_AA > p_AB
and
p_BB > p_BA
```

# PART F — Remove mismatched old objectives

## 8. Disable Task 6H raw-logit ranking

Task 6H's raw-logit region ranking must contribute **zero gradient** in Task 6H.1.

It may be logged only as historical diagnostic.

## 9. Disable BCE+Dice from the optimizer objective

Task 6G/6H heatmap BCE+Dice must also contribute **zero gradient** in Task 6H.1.

They may be logged as diagnostics.

Reason:
- inference uses one prompt point;
- point CE is directly aligned with argmax;
- Task 6G showed severe tiny-target imbalance / near-zero-map behavior.

# PART G — Pair-step objective

## 10. One optimizer step = one pair

Keep Task 6H pair semantics:
- one shared frozen SAM feature;
- query A forward;
- query B forward;
- both graphs retained;
- one backward;
- one optimizer step.

Scheduler horizon counts pair steps.

## 11. Fixed total loss

Use:

```text
L_total =
    0.5 * (L_reasoning_A + L_reasoning_B)
  + 1.0 * (L_point_A + L_point_B)
  + 1.0 * L_cf
```

No weight sweep.

# PART H — H0-R: corrected 10-pair overfit

## 12. Clean initialization

Use clean Task 6G/6H initialization.

Do not initialize from the scale-exploded Task 6H H0 checkpoint.

Use the same 10 canonical H0 pairs.

Maximum:
`1500 pair optimizer steps`.

## 13. H0-R metrics

At 250/500/750/1000/1500 pair steps report:

- target-cell top-1 /20;
- target-cell top-5 /20;
- point-inside-own /20;
- paired point-selection /10;
- bounded pair ranking /10;
- mean own target probability mass;
- mean cross target probability mass;
- mean own-minus-cross probability mass;
- same-image point distance;
- same-image probability-map overlap;
- mean absolute raw logit;
- logit std;
- loss components.

### H0-R gate

Require:

- point-inside-own >= `18/20`;
- paired point-selection >= `9/10`;
- bounded pair ranking >= `9/10`;
- mean own target probability mass > mean cross target probability mass;
- final mean absolute logit < `10.0`.

Target-cell top-1 is not a hard gate because any interior target cell can be a valid SAM prompt.

If H0-R fails, run the focused audit and STOP.

# PART I — Focused audit on failure

## 14. Audit only the corrected objective

If H0-R fails, inspect:

- point-CE gradient norms into query projection, visual projection, LoRA, `[BOX]` row;
- bounded `L_cf` gradients;
- target-cell probability evolution;
- maximum non-target probability;
- spatial-softmax entropy;
- whether `L_point` decreases;
- whether argmax approaches GT point;
- A/B query and projected-query distances;
- pair identities / target indices;
- shared-feature bit identity.

Do not change architecture or loss during audit.

Possible verdict:
`BOUNDED_POINT_OBJECTIVE_FAILED`.

# PART J — H1-R: 240-pair mini-train

## 15. Run only if H0-R passes

Train:
- 240 canonical pairs / epoch;
- maximum 8 epochs;
- deterministic order;
- corrected cosine scheduler over actual pair steps;
- validate every epoch.

No mid-run tuning.

## 16. Validation metrics

Primary fixed 20 paired validation images:
- bounded pair ranking /20;
- paired point-selection /20;
- own/cross target probability mass;
- point-inside-own.

Independent 120 validation records:
- point-inside-target rate;
- target-cell top-1/top-5;
- normalized point error;
- 512-pixel point error;
- spatial entropy;
- L1/L2/L3;
- query-family breakdown.

## 17. Model selection

Lexicographic:
1. paired point-selection /20;
2. pair ranking /20;
3. 120-record point-inside-target rate.

## 18. H1-R gate

Require:
- paired point-selection >= `14/20`;
- bounded pair ranking >= `14/20`;
- 120-record point-inside-target >= `0.60`.

If best epoch fails:
`BOUNDED_COUNTERFACTUAL_GROUNDING_FAILED`.

Do not train SAM2 to hide failure.

# PART K — Representation/output diagnosis

## 19. Compare against Task 6H

At best H1-R checkpoint report:
- `[BOX]` same-image centered cosine;
- projected-query distance;
- spatial-softmax entropy;
- own target probability mass;
- cross target probability mass;
- own-minus-cross probability margin;
- same-image probability-map overlap;
- predicted-point distance;
- correlation between probability-mass margin and localization correctness;
- correlation between projected-query distance and GT point distance.

Key question:

> Did the bounded point-aligned objective convert same-image instruction differences into correct spatial selection rather than magnitude-only separation?

# PART L — H2-R frozen-SAM2 segmentation

## 20. Run only if H1-R passes

Inference:

```text
image + instruction
→ [BOX] query
→ dense heatmap logits
→ argmax point
→ official frozen SAM2 positive-point prompt
→ frozen SAM2 decoder
→ mask
```

No GT.
No SAM2 training.

## 21. H2-R metrics

On fixed 120 val + 20 paired:
- strict e2e mIoU;
- Dice;
- mask paired /20;
- own-target mask IoU;
- cross-target mask IoU;
- own-minus-cross mask margin;
- IoU(pred_A,pred_B);
- L1/L2/L3;
- query-family breakdown.

Compare to:
- Task 6C P_C `0.10604 / 0/20`;
- point oracle `0.4876 / 18/20`;
- box oracle `0.7506 / 20/20`.

# PART M — Verdicts

## 22. Use exactly one

### `BOUNDED_COUNTERFACTUAL_FIX_FOUND`
Require:
- H0-R pass;
- H1-R gate pass;
- H2-R mask paired >= `14/20`;
- strict e2e mIoU >= `0.20`;
- mask own-minus-cross > `0.05`;
- no GT leakage.

### `BOUNDED_COUNTERFACTUAL_PARTIAL`
Corrected objective yields real localization improvement but misses full H1/H2 gate.

### `BOUNDED_COUNTERFACTUAL_GROUNDING_FAILED`
H0-R passes but generalization fails.

### `BOUNDED_POINT_OBJECTIVE_FAILED`
H0-R cannot pass after audit.

### `INVALID_EXPERIMENT`
Loss/sign/pair/target-index/leakage/freeze/split or correctness failure.

# PART N — Next-decision discipline

## 23. If H0-R still fails

Do **not** keep redesigning scalar losses indefinitely.

A clean failure of point CE + bounded pair mass means the existing single `[BOX]` query representation is inadequate even under a directly aligned spatial objective.

Then stop for review. Candidate next classes are:
- instruction-aware multi-query / iterative query refinement;
- stronger MLLM representation;
- architecture-level reference/relation grounding.

Do not implement them automatically.

## 24. If H0-R passes but H1-R fails

Then the problem becomes generalization/data/representation capacity, not basic optimization.

Stop for review and explicitly reconsider:
- training-set scale;
- dataset quality;
- 2B vs larger MLLM;
- stronger query architecture.

# PART O — Required artifacts

## 25. Create

```text
evaluation/task6h1_objective_setup.json
evaluation/task6h1_h0r_overfit.json
evaluation/task6h1_h0r_audit.json
evaluation/task6h1_h1r_training.json
evaluation/task6h1_spatial_eval.json
evaluation/task6h1_representation.json
evaluation/task6h1_segmentation_eval.json
evaluation/task6h1_paired_probe.json
evaluation/task6h1_error_analysis.json
evaluation/task6h1_checkpoint_manifest.json
docs/task6h1_bounded_point_counterfactual.md
```

Conditional files only when their stage runs.

# PART P — Tests

## 26. Required tests

Cover at least:

1. frozen 256-grid target-cell convention;
2. target cell corresponds to deterministic interior point;
3. point-CE target index correct;
4. spatial softmax sums to 1;
5. region probability mass bounded `[0,1]`;
6. region mass is not divided by target area;
7. pair preference is invariant to additive logit shift;
8. increasing own probability mass lowers `L_cf`;
9. increasing cross probability mass raises `L_cf`;
10. Task 6H raw-logit loss contributes zero gradient;
11. BCE/Dice contributes zero gradient;
12. one pair = one optimizer step;
13. scheduler uses pair-step horizon;
14. architecture identical to Task 6G/6H;
15. SAM2 fully frozen;
16. Qwen visual tower frozen;
17. `[BOX]` causally pre-reasoning;
18. no Task 6F box head;
19. no Task 6E loc path;
20. no test split;
21. H2-R uses predicted point only;
22. no 4B / `[REF]` / SRE / SCL;
23. strict determinism;
24. Task 6H causal wording corrected without changing historical metrics.

Run:
`python -m pytest tests/ -q`

# PART Q — Git / Watt

## 27. Git hygiene

Do not stage weights/checkpoints/caches/hidden dumps/`.conda`/dataset JSONL changes.

Recommended commit:
`fix: align counterfactual grounding with point localization`

Use established Watt ownership rules for final push only.

# PART R — Handoff

## 28. FROM_DSH

Include:
1. Verdict
2. Frozen Task 6H Evidence
3. Corrected Causal Interpretation
4. Point-Cell Target
5. Spatial-Softmax Objective
6. Bounded Counterfactual Mass Loss
7. Pair-Step Semantics
8. H0-R Overfit
9. H0-R Audit if needed
10. H1-R Training
11. Point Localization Metrics
12. Pair Preference Metrics
13. Representation/Entropy Diagnostics
14. H2-R Segmentation
15. Paired Mask Probe
16. Error/Data Adequacy Analysis
17. Runtime/VRAM
18. Tests
19. Git/Watt
20. Recommended next architecture decision

## 29. Final DSH UI — Chinese only

Report:
- verdict;
- H0-R inside-own /20;
- H0-R paired point /10;
- H0-R bounded pair ranking /10;
- final mean abs logit vs Task 6H 16.32;
- target/cross probability mass;
- if H1-R ran: best epoch, paired point /20, pair ranking /20, 120-val inside rate;
- if H2-R ran: strict mIoU, mask paired /20, mask own-cross margin;
- whether objective degeneracy is fixed;
- dominant remaining failure;
- tests;
- commit/push;
- Watt handling.

# 30. STOP

After Task 6H.1:

**STOP.**

Do not automatically:
- add multi-query refinement;
- add cross-attention;
- add `[REF]`;
- add SRE/SCL;
- scale to 4B;
- migrate dataset;
- run full training;
- build GUI.

Wait for ChatGPT review.
