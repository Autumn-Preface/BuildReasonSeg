# Task 6H.1 — Spatial-Softmax Point Supervision + Bounded Counterfactual Grounding

**Question.** Task 6H's own-vs-cross ranking was defined on **unbounded logits**, so it could be
minimized by magnifying the field (mean |logit| 0.148 → 16.32) without moving the heatmap peak onto
the target. Task 6H.1 keeps the Task 6G/6H architecture frozen and replaces only the spatial
objective with one directly aligned to the deployed inference operation (`argmax` of the heatmap):

```text
image + instruction -> [BOX] -> query hidden -> DenseSpatialGroundingHead -> heatmap -> argmax

target point-cell : frozen Task 6D interior point -> 256x256 cell -> index = y*256 + x
L_point           = CrossEntropy(H.flatten(), target_index)          (65,536 spatial classes)
P                 = softmax(H.flatten());   mass(P, M) = sum(P * M)  (bounded in [0,1], no area division)
L_cf_A            = -log((p_AA + eps) / (p_AA + p_AB + 2*eps))
L_total           = 0.5*(L_reasoning_A + L_reasoning_B) + 1.0*(L_point_A + L_point_B) + 1.0*L_cf
```

Task 6G's BCE+Dice and Task 6H's raw-logit ranking are computed **detached** and contribute exactly
zero gradient (sections 8–9). No architecture change: no extra query token, no coordinate channels,
no cross-attention, no visual LoRA, no larger decoder, no new dataset, no SAM2 training.

## 1. Objective setup and gradient audit (sections 4–11)

`evaluation/task6h1_objective_setup.json` (`passed: true`) verifies on the real model:

* architecture identical to Task 6G/6H (grid 256, level `[1,32,256,256]`, head 270 593 parameters);
* target point-cells for the first canonical pair: indices **29418 / 18943** (different, in range),
  built with the frozen interior point + Task 6G cell convention;
* softmax sums to 1; the four bounded masses and the pair preference are recorded;
* **the corrected objective reaches every trainable group** — point-CE gradient norms: query
  projection 0.881, query norm 0.0136, visual 1×1 0.109, Qwen LoRA 0.109, token rows 0.322;
  bounded-`L_cf` gradient norms: 0.059, 0.0010, 0.0071, 0.0134, 0.0531;
* **the retired terms carry no gradient**: `bce_dice_requires_grad: false`,
  `logit_ranking_requires_grad: false`, `gradient_contribution: 0.0`;
* SAM2 and the shared frozen feature stay **bit-identical** through the pair step; the `[BOX]` query
  stays causally clean; ordinary rows and the base table stay exactly unchanged; the visual tower
  carries zero LoRA.

## 2. H0-R — corrected 10-pair overfit (sections 12–13)

Clean Task 6G/6H initialization (never the scale-exploded Task 6H checkpoint), the same 10 canonical
pairs, 1500 pair steps.

| Pair step | inside-own | top-1 cell | paired point | bounded ranking | own mass | cross mass | \|logit\| |
|---|---|---|---|---|---|---|---|
| 250 | — | — | — | — | — | — | — |
| 500 | 4/20 | 1 | 1/10 | 5/10 | — | — | 5.65 |
| 750 | 2/20 | 1 | 0/10 | 6/10 | — | — | 9.73 |
| 1000 | 5/20 | 3 | 1/10 | 8/10 | 0.0030 | 0.0062 | 11.20 |
| 1250 | 7/20 | 4 | 2/10 | 8/10 | — | — | 12.95 |
| **1500** | **7/20** | **5** | **2/10** | **8/10** | **0.1835** | **0.0456** | **13.21** |

Gate (inside ≥ 18/20, paired point ≥ 9/10, bounded ranking ≥ 9/10, own mass > cross mass, final mean
|logit| < 10.0): **failed**. Own mass does exceed cross mass (margin **+0.1380**) — the bounded
objective is *not* the 6H degeneracy — and the point metrics improve steadily with steps
(inside 3/20 → 7/20 vs Task 6H, paired point 0/10 → 2/10, top-1 cell 25 %), but none of the three
point gates is close, and the logit magnitude drifts above the 10.0 limit near the end.

## 3. H0-R audit (section 14): the objective is healthy, the signal is not

`evaluation/task6h1_h0r_audit.json`, clean initialization → H0-R checkpoint:

| Quantity | Before | After |
|---|---|---|
| mean point CE (chance = ln 65536 = 11.090) | **11.0965** | **4.7427** |
| target-cell probability | 0.00002 | **0.0515** |
| max **non-target** probability | 0.00002 | **0.0532** |
| target-cell top-1 rate | 0.00 | **0.25** |
| point-inside rate | 0.00 | **0.35** |
| normalized point error | 0.4658 | **0.1673** |
| spatial-softmax entropy (nats) | 11.090 | **6.205** |
| mean own probability mass | 0.0072 | **0.1838** |
| mean cross probability mass | 0.0072 | **0.0453** |
| probability margin | −0.0000 | **+0.1385** |
| bounded `L_cf` | 0.854 | **0.243** |
| A/B hidden distance | 27.4 | 135.7 |
| A/B projected-query distance | 1.49 | 636.8 |

Reading: the corrected objective **works** — the spatial cross-entropy falls from exactly chance to
4.74, the distribution sharpens (entropy 11.09 → 6.21), the target cell gains ~2 500× probability,
the argmax moves from 0.466 to 0.167 normalized error, and the bounded preference margin becomes
clearly positive. But the mode is still not reliably the target: the **largest non-target cell has
0.0532 probability against a 0.0515 mean target probability**, so top-1 is 25 % and inside-target
35 % — far from the 90 % gate. The failure is therefore **not** objective degeneracy; it is that the
single `[BOX]` query representation does not carry enough target-specific spatial signal even under
a directly aligned, bounded objective.

## 4. Error classification

`evaluation/task6h1_error_analysis.json` (H0-R stage): pair preference 8/10, both points inside 2/10,
dominant failure class `tiny_target`; 120-record-style inside rate 0.350, top-1 0.250. The residual is
localization precision on WHU-scale targets, not pair-construction or objective correctness.

## 5. Verdict (sections 22–23)

`evaluation/task6h1_verdict.json`: **`BOUNDED_POINT_OBJECTIVE_FAILED`** — H0-R cannot pass after the
focused audit. Per section 13 the task **stops here**: H1-R and H2-R are not run, and
`evaluation/task6h1_h1r_training.json`, `task6h1_spatial_eval.json`,
`task6h1_representation.json`, `task6h1_segmentation_eval.json` are deliberately absent (the paired
probe carries the H0-R 10-pair stage).

Per section 23 this is a **clean** failure of point-CE + bounded pair mass: the existing single
`[BOX]` query representation is inadequate even under a directly aligned spatial objective, so the
candidate next classes are instruction-aware multi-query / iterative query refinement, a stronger MLLM
representation, or architecture-level reference/relation grounding — **none of which may be
implemented automatically** without review.

## 6. Historical correction (section 3)

Task 6H's machine verdict and all of its numbers are **unchanged**. The accepted causal
interpretation is narrowed: Task 6H *did* learn the specified own-vs-cross logit ranking, but the
loss was scale-degenerate, so Task 6H did **not** establish that the query signal itself is
unlearnable. Task 6H.1 supplies the bounded, localization-aligned test that the earlier task could
not: under that objective the query does produce real (but insufficient) spatial signal.
