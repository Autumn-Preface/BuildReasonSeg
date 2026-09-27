# Task 6F — Target-Aware `[BOX]` Query Grounding v0.1

**Question.** Task 6E retired itself: the 256-way autoregressive coordinate vocabulary overfits 20
records but fails structurally (0/120) on the 480-record paired mini-train. Task 6F keeps the box an
explicit supervision target but changes **where the representation is formed** and **how it is
read**: a single fixed learned query token `[BOX]` is inserted immediately after the assistant
generation prefix — before any reasoning text — and the target box is regressed directly from its
hidden state:

```text
USER: image + instruction
ASSISTANT PREFIX: [BOX]                  <- fixed query, identical for every sample, never predicted
ASSISTANT TARGET: reasoning_zh [SEG] EOS

[BOX] hidden (2048-d) -> TargetAwareBoxHead -> (x1, y1, x2, y2) in [0,1], canonical
L_total = 1.0 * L_reasoning + 5.0 * L_box      (L_box = SmoothL1 vs the frozen Task 6D tight box)
```

**Scope.** Frozen stack: Qwen3-VL-2B-Instruct (text-only LoRA) + SAM2.1 Hiera Base+. Task 6C paired
`P` subset, the fixed 120-record validation set, the fixed 20 paired validation images, the Task 6C.7
visual-feature cache, the Task 6D.1 corrected scheduler. Task 6E's coordinate-token path is **retired**
(history preserved in git and `evaluation/task6e_*.json`): no `<loc_*>` tokens are added, no location
CE, no location-run parsing, no initialization from Task 6E weights. Deliberately **not** used: 4B,
`[REF]`, SRE, SCL, a new dataset, full training, true batch > 1, GUI, or any SAM2 training.

## 1. Initialization and vocabulary (sections 2–3)

Clean base/seed convention identical to Task 6E E1: fresh Qwen3-VL-2B-Instruct + existing `[SEG]`
(id 151 669) + exactly **one** new query token `[BOX]` (id 151 670; vocabulary 151 670 → 151 671).
No Task 6E `<loc_*>` rows are carried and none exist in the Task 6F tokenizer. Trainable token rows:
`{[SEG], [BOX]}` through PEFT `trainable_token_indices` (one 4096-element parameter, two rows), base
table frozen. Provenance, ids and optimizer coverage are recorded in
`evaluation/task6f_token_setup.json`.

## 2. Causal placement (section 4)

`[BOX]` is appended exactly at the position after the assistant-generation prefix, so its hidden
state attends to the image, the instruction, the chat prefix and its own embedding — and, under
causal masking, to **nothing** from the reasoning/GT/`[SEG]` future. Two proofs:

* structural: the query token sits at `box_position == prompt_length`, appears exactly once, precedes
  every target token, and the position that would predict it stays `-100` (the model never learns to
  emit `[BOX]`);
* functional: on the real model, replacing the entire reasoning tail with unrelated text leaves the
  `[BOX]` hidden **bit-identical** (`box_hidden_max_abs_delta: 0.0`).

This is the central difference from Task 6D's end-of-reasoning `[SEG]` readout.

## 3. TargetAwareBoxHead (section 5)

The deliberately simple readout Task 6D already used — `LayerNorm(2048) → Linear(2048, 512) → GELU →
Linear(512, 4) → sigmoid → min/max canonicalization` — fed by the pre-reasoning query hidden
(1 055 236 parameters, 6 tensors, in the decoder LR group at 3e-4). Output is a canonical normalized
box in `[0,1]`; GT geometry is used only for SmoothL1 supervision and evaluation (section 6).

## 4. One-step gradient smoke (section 8)

`evaluation/task6f_token_setup.json` records a real forward/backward: the `[BOX]` row and `[SEG]`
row both move (3.0e-4) with non-zero gradient; all six box-head parameters move with non-zero
gradient; 16 ordinary rows change by exactly **0.0** and the base embedding tensor is bit-identical;
SAM2 is fully frozen; the visual tower carries zero LoRA. Trainable totals: 399 tensors /
18 491 908 parameters (LoRA 17 432 576, box head 1 055 236, token rows 4 096).

## 5. F0 — implementation proof (section 9)

20 deterministic records (10 same-image/different-target pairs), ≤1500 steps, corrected horizon,
evaluation exclusively through the inference-form query path.

| Step | train box IoU | paired geometry | non-identical pairs |
|---|---|---|---|
| 250 | 0.0576 | 0/10 | 10/10 |
| 500 | 0.2464 | 0/10 | 10/10 |
| 750 | 0.7959 | 10/10 | 10/10 |
| **1000** | **0.9046** | **10/10** | **10/10** |

**`F0_PASS` at step 1000** (gate: train box IoU ≥ 0.85, paired ≥ 9/10, same-image non-identical
boxes, no GT leakage). Box loss converged to ~0.0 (per-step box IoU peaked at 0.978). F0's weights
are not carried into F1. Artifact: `evaluation/task6f_f0_overfit.json`.

## 6. F1 — 480-paired mini-train (sections 10–12)

480 paired `P` records, 8 epochs × 480 steps = 3840 optimizer steps, one corrected cosine horizon
(warmup 20), per-epoch checkpoint and query-path validation, selection by paired → val box IoU →
center-inside, early stop after epoch 3 on 3 consecutive non-improvements (never triggered: the
budget ran out at epoch 8).

| Epoch | train box IoU (mean) | val box IoU | median | center-inside | coord MAE | pred spread | paired geom |
|---|---|---|---|---|---|---|---|
| 1 | 0.0042 | 0.0073 | 0.0 | 0.000 | 0.2448 | 0.05 | 0/20 |
| 2 | 0.0057 | 0.0090 | 0.0 | 0.008 | 0.2211 | 0.17 | 0/20 |
| 3 | 0.0074 | 0.0071 | 0.0 | 0.025 | 0.2175 | 0.19 | 0/20 |
| 4 | 0.0089 | 0.0057 | 0.0 | 0.008 | 0.2173 | 0.20 | 0/20 |
| 5 | 0.0084 | 0.0115 | 0.0 | 0.033 | 0.2141 | 0.20 | 0/20 |
| 6 | 0.0136 | 0.0164 | 0.0 | 0.033 | 0.1868 | 0.23 | 0/20 |
| 7 | 0.0328 | 0.0178 | 0.0 | 0.017 | 0.1568 | 0.24 | 0/20 |
| **8** | **0.0504** | **0.0250** | **0.0** | **0.017** | **0.1543** | **0.25** | **0/20** |

**Verdict `TARGET_AWARE_QUERY_FAILED`** (gate needs paired ≥ 14/20, val box IoU ≥ 0.35, center ≥
0.70). What the curve shows, precisely:

* the box head **never fits even the training set** (train SmoothL1 plateaus at ≈0.003, which is an
  RMS coordinate error of ~8 % of the tile — not a near-zero fit like F0's 20-record run), so this
  is under-fitting under the fixed ≤8-epoch budget, not a generalization cliff;
* the predictions are **canonical and spread out** (per-coordinate spread grows 0.05 → 0.25 — the
  model does not collapse to a constant box, it just cannot localize), with predicted-box areas
  (~0.0078) close to the GT average (0.0120);
* the language side is unaffected (see section 8): the `[BOX]` prefix costs nothing.

Artifacts: `evaluation/task6f_f1_training.json`, `evaluation/task6f_geometry_eval.json`,
`evaluation/task6f_paired_probe.json` (geometry side, `mask_ran: false`).

## 7. Representation diagnosis (section 13)

`evaluation/task6f_representation.json` (best checkpoint, global-mean centring):

| Measure | Task 6F `[BOX]` | Legacy `[SEG]` (frozen 6D.1) |
|---|---|---|
| same-image/diff-instruction cosine | **0.969** | ~0.999 (raw) |
| same-image centered cosine | **0.626** | 0.077 |
| different-image centered cosine | **0.018** | 0.531 |
| effective rank (participation) | **7.07** | 8.33 |
| hidden-distance vs GT-box-distance correlation | **0.476** | ≈0 (real ≈ shuffled) |

The answer to the section 13 question is **nuanced and negative on the main hypothesis**: moving the
supervised query before reasoning removed the *template* domination (the query cannot see the
reasoning text at all) but replaced it with **image domination** — the two instructions of one image
stay much closer (centered 0.626) than two different images (0.018), because the query attends almost
exclusively to image + instruction and the 2B model's instruction conditioning is weak. The query does
carry more target-geometry variation than the legacy `[SEG]` (correlation 0.476 vs ≈0), but not enough
for a 6-tensor head to localize WHU-scale targets. No causal claim is made from cosine alone.

## 8. Reasoning compatibility (section 14)

Continuing generation from `image + instruction + fixed [BOX]` on the first 20 fixed validation
records: exactly-one `[SEG]` **1.000**, EOS termination **1.000**, known-reasoning-template rate
**1.000**. The query prefix is fully compatible with the assistant generation path — the F1 failure
is entirely on the geometry side. (Template diagnostics only; no reasoning claim.)

## 9. F2 — frozen-SAM end-to-end test (sections 15–16)

**Not run**: section 15 gates F2 on the F1 gate, which failed, so
`evaluation/task6f_segmentation_eval.json` is deliberately absent. The frozen ceilings stay:
continuous Oracle BOX 0.7506 / 20/20, Task 6E quantized oracle 0.7343 / 20/20, Task 6C `P_C`
0.10604 / 0/20.

## 10. Error classification (section 18)

`evaluation/task6f_error_analysis.json` (120 records, IoU threshold 0.5): **120/120 failures**;
dominant class `wrong_target_valid_box` (98.3 %); `tiny_target` on **93.3 %** of failures, border
truncation 29 %, L3 relation 33 %. L1 mean box IoU 0.011 vs L2/L3 0.032. `whu_data_quality_dominates`
is reported true (97.5 % of failures carry a WHU quality flag), with the honest reading: the failure
mode is *localization precision* on inherently tiny targets (mean GT area 1.2 % of the tile), where a
box needs sub-5 % placement accuracy to score IoU ≥ 0.5 — the dataset's difficulty is a real
confound for box-IoU-based gates at this model scale, but the F1 failure is primarily an
under-fitting failure (train box loss never converged), not a data-defect failure.

## 11. Verdict (section 17)

`evaluation/task6f_verdict.json`: **`TARGET_AWARE_QUERY_FAILED`** — implementation valid (token
setup smoke + causal bit-identity + F0 0.9046/10-10), F1 gate failed (paired 0/20, val box IoU
0.025, center 0.017), F2 not run. No SAM2 training, no `[REF]`/SRE/SCL/4B/dataset change/GUI, and
the Task 6E coordinate-token machinery stayed retired.

## 12. Reusable inference (section 19)

Gated on F1 passing, so no product-facing `predict_box` / `predict_mask` / `generate_reasoning`
facade was created; the verified query-path plumbing (`predict_box_for_sample`, `build_query_batch`,
`generate_with_box_prefix`) remains available for the next task. No GUI.
