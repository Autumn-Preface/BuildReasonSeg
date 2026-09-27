# Task 6H — Counterfactual Pair-Aligned Dense Grounding v0.1

**Question.** Task 6G's dense head localized nothing (G0 inside-own 0.10, Dice 0.125) even though
its implementation was audited clean. Task 6H keeps the Task 6G architecture **unchanged** and
changes only the training semantics: one optimizer step consumes one canonical counterfactual pair
from the Task 6C `P` subset (same image, different instruction, different target) and adds an
explicit own-vs-cross region-ranking loss, so heatmap A must score target A above target B and vice
versa.

```text
image + instruction -> [BOX] -> query hidden -> DenseSpatialGroundingHead -> heatmap -> argmax point

s_XY = sum(H_X * M_Y) / max(sum(M_Y), eps)                 (region score, section 7)
L_cf = 0.5*( softplus(1.0 - (s_AA - s_AB)) + softplus(1.0 - (s_BB - s_BA)) )
L_total = 0.5*(L_reasoning_A + L_reasoning_B)
        + 1.0*(L_heatmap_A + L_heatmap_B)
        + 2.0*L_cf
```

**Scope.** No extra query token, no cross-attention, no coordinate channels, no visual LoRA, no
`[REF]`, no SRE/SCL, no 4B, no dataset migration, no GUI, no SAM2 training. All Task 6E/6F machinery
stays retired.

## 1. Canonical pairs (section 4)

`evaluation/task6h_pair_manifest.json`: **240 pairs over 240 images**, every assertion enforced —
same source image, different sample ids, different target component ids, different masks,
deterministic (Task 6C `P` payload) order, no sample reused, no test split. Canonical identity hash
`c56f0507da41fd5506ffd0d3c37b056d017de7d48c6f4cc448941a9aaf997c3b`. The downsampled soft-target
overlap is recorded for every pair and **no pair was discarded**: 0/240 pairs overlap (max soft IoU
0.000), so the ranking task is unambiguous for all of them.

## 2. Pair-step semantics and loss (sections 6–9)

One pair step = one shared frozen SAM2 feature → query forward for A → query forward for B → both
heatmaps → per-sample BCE+Dice + `L_cf` → **one** backward and optimizer step. The scheduler horizon
counts **pair** steps (240/epoch), not record steps. Weights and the margin are constants
(`0.5/1.0/2.0`, margin 1.0); nothing was swept. `evaluation/task6h_token_setup.json` verifies on the
real model that the architecture is byte-for-byte the Task 6G one (grid 256, level `[1,32,256,256]`,
head 270 593 parameters), that SAM2 and the shared feature stay **bit-identical** through a pair
step, that the `[BOX]`/`[SEG]` rows and all head parameters move with gradient, that ordinary rows
and the base table stay exactly unchanged, and that the `[BOX]` query is still causally clean.

## 3. H0 — 10-pair counterfactual overfit (sections 11–12)

1500 pair steps on the same 10 pairs as Task 6G G0, from the clean Task 6G initialization
convention.

| Pair step | pair ranking | strict margin | inside-own | paired point | mean margin | heatmap Dice |
|---|---|---|---|---|---|---|
| 250 | 10/10 | 10/10 | 0/20 | 0/10 | +3.68 | 0.003 |
| 500 | 10/10 | 10/10 | 5/20 | 1/10 | +4.94 | 0.008 |
| 1000 | 10/10 | 10/10 | 4/20 | 1/10 | +5.37 | 0.127 |
| **1500** | **10/10** | **10/10** | **3/20** | **0/10** | **+5.58** | **0.134** |

Gate (ranking ≥ 9/10, inside-own ≥ 18/20, paired point ≥ 9/10, margin > 0.5): **failed** —
ranking and margin pass overwhelmingly, the two **point** gates fail by a wide margin.

## 4. H0 audit (section 13) — why the ranking gate is not enough

`evaluation/task6h_h0_audit.json`, clean initialization vs the H0 checkpoint:

| Quantity | Before | After |
|---|---|---|
| `L_cf` (mean over the 10 pairs) | 1.3133 | **0.0212** |
| own−cross margin (logit units) | −0.0001 | **+5.5788** |
| own−cross margin (probability units) | −0.0000 | **+0.1124** |
| mean abs logit | 0.148 | **16.32** |
| argmax peakiness | 1.6e-05 | 0.403 |
| point-inside rate | 0.00 | **0.15** |
| A/B hidden distance | 27.4 | 167.7 |
| A/B projected-query distance | 1.49 | 779.5 |

The pair loss has the **correct sign** (raising the own score lowers it, raising the cross score
raises it), it **decreases** (1.28 → 0.009 in the training history), and its gradient reaches
**every** trainable group (query projection 0.031, query norm 0.0004, visual 1×1 0.0041, Qwen LoRA
0.0164, token rows 0.0710). But the way it is satisfied is **logit magnification**: the field's
scale grows ~110×, which grows the *logit* margin to +5.6 while the *probability* margin stays at
0.11 and the argmax still lands inside the target only 15 % of the time. Per section 7 the region
score is defined on the unbounded heatmap logits, so the specified margin is scale-degenerate — an
objective that is satisfiable without localizing. The A/B query separation explodes by magnitude
(1.49 → 779.5 in projected space) rather than becoming target-specific.

Error classification (`evaluation/task6h_error_analysis.json`, H0 stage): 10/10 pairs are
`pair_ranking_correct_point_outside` — the dominant failure is exactly "ranking correct, point
outside", with the 120-record-style inside rate at 0.150. The same-image **heatmap A/B IoU is
0.0032**: the two instructions' thresholded maps barely overlap, i.e. they differ by scale/offset
rather than by *location*, which is the degeneracy seen from the output side.

## 5. Verdict (section 20)

**`COUNTERFACTUAL_QUERY_SIGNAL_FAILED`** — H0 cannot learn a *localizing* own-vs-cross preference
after the focused audit, even though it learns the specified ranking objective. Per section 12 the
task **stops here**: H1, the H1 representation diagnosis and H2 are not run, and
`evaluation/task6h_h1_training.json`, `task6h_spatial_eval.json`,
`task6h_segmentation_eval.json` are deliberately absent (the paired probe carries the H0 10-pair
stage). Section 21 applies: the same-image counterfactual supervision is not claimed as paper
novelty and is not called Spatial Consistency Loss.

## 6. What this adds to the project's evidence

Five readouts now fail, each with a verified implementation: the 6D.1 `[SEG]` probes, the 6E
coordinate vocabulary, the 6F global box regression, the 6G dense map, and now the 6H
counterfactually-ranked dense map. Task 6H adds the sharpest diagnosis yet — the failure is not
"the loss does not train" but "**the specified ranking objective is satisfiable by rescaling the
logit field**, and the task's point gates are the ones that expose it". Any future pair-ranking
objective for this project must be defined on a **bounded** (probability/softmax) scale or include
an explicit peak-sharpness/localization term, otherwise the ranking gate will keep passing while
localization stays at chance.
