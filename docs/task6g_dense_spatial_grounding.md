# Task 6G — Dense Query–Visual Spatial Grounding Map v0.1

**Question.** Task 6F's global box regression overfit 20 records but failed on 480 (val box IoU
0.025, paired 0/20), and its query representation was image-dominated (same-image centered cosine
0.626). Task 6G keeps the causally clean pre-reasoning `[BOX]` query but **stops regressing
coordinates**: the query hidden is matched against the frozen SAM2 dense spatial feature map to
produce a target-specific heatmap, and the predicted point is the heatmap argmax cell centre:

```text
image + instruction -> fixed [BOX] -> reasoning_zh [SEG] EOS
q = LayerNorm([BOX] hidden) -> Linear(2048,128)
K = Conv1x1(C,128)(frozen SAM2 spatial feature at the selected grid)
heatmap_logits[y,x] = dot(q, K[:,y,x]) / sqrt(128) (+ scalar bias)
predicted point = argmax cell centre
L_total = 1.0 * L_reasoning + 2.0 * (1.0 * BCEWithLogits + 1.0 * SoftDice)
```

**Scope.** Frozen stack unchanged; Task 6C paired `P` subset, the fixed 120/20 validation material,
the Task 6C.7 visual-feature cache, the Task 6D.1 corrected scheduler. Task 6E's coordinate path and
Task 6F's box head are **retired** (history preserved; no `<loc_*>`, no SmoothL1 box, no Task 6F
checkpoint initialization). Deliberately **not** used: 4B, `[REF]`, SRE, SCL, a new dataset, full
training, true batch > 1, GUI, or any SAM2 training.

## 1. Grid-snapped point oracle (sections 2–3)

The frozen Task 6D deterministic interior point is snapped to the nearest cell centre of each
candidate grid and fed through the official frozen SAM2 positive-point prompt (same 120 val + 20
paired material). `evaluation/task6g_grid_oracle.json`:

| Grid | strict mIoU | Dice | paired | mean displacement (512 px) |
|---|---|---|---|---|
| continuous | **0.4876** | — | **18/20** | — |
| 64 | 0.4950 | 0.6151 | 17/20 | 2.23 |
| 128 | 0.4894 | 0.6065 | 17/20 | 1.02 |
| **256 (selected)** | **0.4883** | 0.6061 | **18/20** | 0.50 |

Rule: smallest grid with paired ≥ 18/20 and mIoU ≥ `0.4876 − 0.03 = 0.4576` → **256** (the only
qualifying grid: 64/128 pass the mIoU bar but drop a paired image). The 256×256 level is the
32-channel SAM2 high-resolution feature, verified by spatial size, never by list order.

## 2. Query and head (sections 4–6)

The Task 6F `[BOX]` query token is reused unchanged (id 151 670, pre-reasoning, never predicted; its
hidden stays **bit-identical** under future-text mutation — re-verified in
`evaluation/task6g_token_setup.json`). `DenseSpatialGroundingHead`: LayerNorm(2048) →
Linear(2048,128) for the query, Conv1x1(32,128) for the frozen SAM2 feature, dot product scaled by
1/√128 plus a scalar bias (270 593 parameters). Trainable: query projection + visual-key projection;
the SAM2 encoder and source feature tensors stay frozen — the one-step smoke records
`sam2_max_abs_delta: 0.0` (bit-identical) while the `[BOX]`/`[SEG]` rows and all head parameters move
with non-zero gradient, and 16 ordinary rows plus the base embedding stay exactly unchanged.

## 3. Supervision and extraction (sections 7–9)

GT heatmap = the instruction-selected target mask, area-downsampled to 256×256 soft occupancy
(mass > 0 asserted). Loss: 1.0·BCEWithLogits + 1.0·SoftDice (fixed weights), total
`1.0·L_reasoning + 2.0·L_heatmap`. The predicted point is `argmax(heatmap_logits)` mapped to the cell
centre (torch argmax's first-maximum rule makes it deterministic); a soft-argmax expectation is
reported as a diagnostic only. No threshold tuning, no GT-guided repair.

## 4. G0 — implementation proof (section 11): FAILED

20 deterministic records / 10 pairs, 1500 steps, corrected horizon, evaluation through the real
query-to-heatmap path.

| Step | inside own target | heatmap Dice | paired selection | distinct points |
|---|---|---|---|---|
| 500 | 0.05 | 0.103 | 0/10 | 2/10 |
| 1000 | 0.10 | 0.121 | 0/10 | 6/10 |
| 1500 | 0.10 | 0.125 | 0/10 | 7/10 |

Gate (inside ≥ 19/20, paired ≥ 9/10, Dice ≥ 0.80, distinct points): **failed at budget exhaustion**.
Training trace: reasoning CE → 0.0000, BCE ≈ 0.02, but the **Dice term never moves** (0.96–0.99 the
whole run); the heatmap is a near-flat field (sigmoid mean 0.013 = the target occupancy, peakiness
0.086) whose argmax is image-content noise — errors of 44–225 px, and for one image both
instructions pinned the **identical** cell.

**Implementation audit** (`evaluation/task6g_g0_audit.json`, on the recorded G0 checkpoint): no
defect found —

* train/eval equivalence: the training-form and evaluation-form `[BOX]` hiddens are bit-identical,
  so the failure is not a train/eval discrepancy;
* heatmap logits are in a healthy range (mean [−25.5, 4.8], std 3.3) — no saturation bug;
* the BCE and Dice terms both flow comparable gradients into the head (0.395 vs 0.456) — the Dice
  supervision is not vanishing;
* token setup smoke passed before training (rows/head gradients, SAM2 bit-frozen, causal
  bit-identity, base table untouched).

So the frozen recipe genuinely cannot make the dot-product map localize even 20 memorized records:
the query's instruction-dependent component (measured at r ≈ 0.48 in Task 6F) is too weak to move
the dot-product peak onto the target against the image-dominated key field. Per section 11 the task
**stops here**: `DENSE_GROUNDING_IMPLEMENTATION_FAILED`; G1, the representation diagnosis and G2 are
not run.

## 5. Error classification (section 20)

`evaluation/task6g_error_analysis.json` (G0 stage, 20 records): 18/20 outside the target; dominant
class `diffuse_no_localization_heatmap` (18/20), with `tiny_target` on 17/18 failures and
`whu_data_quality_dominates` reported true. The honest reading: the failure is a **diffuse heatmap**
everywhere, so the WHU flags describe the material but cannot explain it — the heatmap does not
localize on simple L1 examples either (L1 inside rate 0.2).

## 6. Verdict (section 18)

`evaluation/task6g_verdict.json`: **`DENSE_GROUNDING_IMPLEMENTATION_FAILED`** — G0 cannot pass after
the implementation audit. G1-family artifacts (`task6g_g1_training.json`,
`task6g_spatial_eval.json`, `task6g_representation.json`, `task6g_segmentation_eval.json`) are
deliberately absent; the paired probe carries the G0 10-pair stage. Section 19 applies: no novelty
claim, and the dot-product fusion is not the paper's contribution.
