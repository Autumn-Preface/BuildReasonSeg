# Task 6E — Explicit Spatial Token Grounding v0.1

**Question.** Task 6D and Task 6D.1 established that the frozen `[SEG]` hidden state of the Task 6C
`P_C` checkpoint contains **no practically decodable target geometry** with the tested readouts, while
the *downstream* machinery is fine: given the correct box, SAM2 reaches 0.7506 mIoU / paired 20/20.
Task 6E therefore stops asking a hidden state to *contain* the box and makes the box an **explicit,
sample-specific autoregressive target**:

```text
image + instruction -> reasoning -> [BOX] <loc_x1> <loc_y1> <loc_x2> <loc_y2> -> [SEG]
```

**Scope.** Frozen stack: Qwen3-VL-2B-Instruct (text-only LoRA) + SAM2.1 Hiera Base+. Text-only LoRA,
existing `[SEG]`, the Task 6C paired `P` subset, the fixed 120-record validation set, the fixed 20
paired validation images, the Task 6C.7 visual-feature cache, the Task 6D.1 corrected scheduler.
Deliberately **not** used: 4B, `[REF]`, Spatial Relation Encoder, Spatial Consistency Loss, a new
dataset, full training, true batch > 1, GUI, or any SAM2 training.

## 1. Quantization (sections 3–4)

One shared location family `<loc_000>…<loc_{B-1}>`; a box is four ordered tokens and the *position*
decides whether a token is x1, y1, x2 or y2. Quantization is a deterministic **enclosing** box:

* `q1 = floor(c1 * (B-1))` for the min edges, `q2 = ceil(c2 * (B-1))` for the max edges, clipped to
  `[0, B-1]`, dequantized with `c_hat = q / (B-1)`;
* a non-empty target keeps non-zero width/height: if a span collapses, the upper edge is expanded by
  one bin, or the lower edge is pulled back when the span already sits at the top.

The same `QuantizedBoxCodec` is used by target generation, the oracle audit, the parser and the tests.

## 2. Quantized-oracle ceiling and bin selection (section 5)

`scripts/task6e_quantized_oracle.py` replaces the continuous GT box with its quantized/dequantized
form on the official frozen SAM2 box prompt path, on the same 120 validation records and the same
20 paired validation images. Rule: the **smallest** `B` that keeps paired **20/20** *and* strict mIoU
`>= 0.7506 - 0.03 = 0.7206`; otherwise `QUANTIZED_BOX_REPRESENTATION_INADEQUATE` and stop.

| B | strict mIoU | Dice | paired | quantized-vs-GT box IoU | own−cross margin |
|---|---|---|---|---|---|
| continuous (Task 6D) | **0.7506** | 0.8507 | **20/20** | 1.0000 | — |
| 32 | 0.5400 | 0.6889 | 20/20 | 0.5608 | +0.5284 |
| 64 | 0.6473 | 0.7773 | 20/20 | 0.7193 | +0.6217 |
| 128 | 0.7129 | 0.8259 | 20/20 | 0.8413 | +0.6765 |
| **256 (selected)** | **0.7343** | 0.8397 | **20/20** | 0.9129 | +0.6919 |

`B = 256` is the smallest qualifying bin count: it recovers 97.8 % of the continuous-oracle mIoU and
keeps the paired probe at 20/20. Artifact: `evaluation/task6e_quantized_oracle.json`.

## 3. Vocabulary and trainable tokens (sections 6–7)

`buildreasonseg_mvp/spatial_tokens.py` adds `[BOX]` plus the 256 shared location tokens (`257` new
special tokens, each exactly one id, each round-tripping through
`decode(..., skip_special_tokens=False)`), and `qwen_seg.attach_lora(..., extra_token_ids=...)`
generalizes the trainable set from `{[SEG]}` to `{[SEG], [BOX], <loc_0>…<loc_255>}` (258 rows)
through PEFT's `trainable_token_indices`; the project-owned multi-row forward hook and the multi-row
output-row wrapper are the documented fallbacks. One-step smoke
(`evaluation/task6e_token_setup.json`, a real forward/backward):

* `[SEG]` row changed, `[BOX]` row changed, all four target `<loc_*>` rows changed;
* the 16 probed **ordinary** vocabulary rows changed by exactly **0.0** and the base embedding tensor
  is bit-identical (528 384 token-row parameters; base table frozen at 151 927 × 2048);
* every sampled new **output** row receives non-zero gradient from an output-head-only loss while the
  base output matrix receives none, so the new tokens can actually be *emitted*;
* zero LoRA modules under the visual tower.

## 4. Target sequence and objective (sections 8–9)

The assistant target is built from **explicit token ids**, never from a decoded string, so every span
position is known. `buildreasonseg_mvp/spatial_training.py` exposes the label positions of
`reasoning / [BOX] / loc / [SEG] / EOS` under the project's causal rule (`logits at position i predict
token i+1`), verifies `labels[i] == input_ids[i+1]` for every recorded position, and scores

```text
L_total = 1.0 * L_assistant + 5.0 * L_location
```

where `L_assistant` is the assistant-span causal CE and `L_location` is the causal CE at exactly the
four location-token prediction positions. There is no mask loss, no SmoothL1 on hidden states and no
weight sweep. Teacher-forced token accuracy is recorded as a **diagnostic only**; every gate uses free
generation parsed from token ids.

## 5. E0 — implementation sanity (section 10)

20 deterministic records (10 same-image/different-target pairs), ≤1500 optimizer steps, scheduler
horizon = the real step budget.

| Step | structural | exact 4-token | mean box IoU | paired geometry |
|---|---|---|---|---|
| 250 | 0/20 | 0/20 | — | 0/10 |
| **500** | **20/20** | **20/20** | **0.9166** | **10/10** |

Gate (≥19/20, ≥18/20, ≥0.70): **pass** — `E0_PASS`, stopped at step 500. Per-coordinate token
accuracy is 1.0 on all four coordinates: the overfit model reproduces the *quantized* GT box exactly,
and the residual 0.0834 IoU is the quantization ceiling, not model error. E0's weights are **not**
carried into E1 (they are fitted to 20 records); E1 starts from the same frozen base.
Artifact: `evaluation/task6e_e0_overfit.json`.

## 6. E1 — paired mini-train and geometry gate (sections 11–13)

480 paired `P` records (240 images × 2 instructions), 3 epochs × 480 steps = 1440 optimizer steps,
trainable = text LoRA + `[SEG]` + `[BOX]` + `<loc_*>` rows, SAM2 fully frozen, no mask loss, corrected
cosine horizon over the true total step count, free-generation evaluation after every epoch.
Model selection priority: geometry paired pass → validation mean box IoU → structural validity.

| Epoch | structural | exact 4-token | mean box IoU | paired geometry | TE `[BOX]` acc | TE loc acc | TE `[SEG]` acc |
|---|---|---|---|---|---|---|---|
| 1 | 0/120 | 0/120 | — | 0/20 | 1.000 | 0.021 | 0.000 |
| 2 | 0/120 | 0/120 | — | 0/20 | 1.000 | 0.058 | 0.000 |
| 3 | 0/120 | 0/120 | — | 0/20 | 1.000 | 0.075 | 0.000 |

**Verdict `EXPLICIT_SPATIAL_TOKENS_FAILED`** (the E1 geometry gate requires ≥90 % structural, ≥14/20
paired and ≥0.35 mean box IoU). The failure mode is specific and is *not* a formatting bug:

* teacher forcing reaches reasoning-token accuracy **1.000**, `[BOX]` accuracy **1.000**, EOS
  accuracy **1.000** — the format skeleton is learned;
* the four location tokens are **not** learned: location CE falls 9.64 → 5.15 on the *training* set
  (the 256-way uniform floor is `ln 256 = 5.545`), teacher-forced loc accuracy reaches only 0.075, and
  `[SEG]` is predicted 0 % of the time after the location run;
* free generation therefore emits exactly one `[BOX]` (120/120) and then a **runaway location run**
  (51–81 tokens, dominated by the attractor tokens `<loc_150>` / `<loc_255>`, absolute bin error
  61.5 → 118.25) with no `[SEG]` at all — a malformed sequence, not a valid-but-wrong box;
* the same code path, same loss and same schedule overfit the **first 20 of these 480 records** to
  20/20 structural and 0.9166 box IoU in 500 steps (E0), so this is an optimization/capacity limit of
  the fixed ≤3-epoch budget on the new 256-way vocabulary, not an implementation defect.

E1 was **not** extended (section 11 caps the budget at 3 epochs) and SAM2 was **not** trained to hide
the failure. Artifacts: `evaluation/task6e_e1_training.json`, `evaluation/task6e_geometry_eval.json`,
`evaluation/task6e_paired_probe.json`.

## 7. E2 — inference-only segmentation (sections 14–15)

**Not run**: section 14 gates E2 on the E1 geometry gate, which failed, so
`evaluation/task6e_segmentation_eval.json` is deliberately absent and the paired mask probe records
`mask_ran: false`. The frozen SAM2 box-prompt ceiling for this pathway is already measured by the
quantized-oracle audit (0.7343 mIoU / paired 20/20 at `B = 256`), so the missing piece is located
unambiguously *before* SAM2: the box itself.

## 8. Reusable inference plumbing (section 19)

`buildreasonseg_mvp/spatial_inference.py`:

```python
generate_spatial_tokens(runtime, image, instruction)     # token ids out
parse_box_tokens(runtime, token_ids)                     # token-id parse
predict_box(runtime, image, instruction)                 # dequantized box
predict_mask_from_generated_box(runtime, image, instruction)  # box -> official SAM2 mask
```

No GUI and no top-level `predict.py`; that waits for review.

## 9. Error classification and data adequacy (section 18)

`scripts/task6e_error_analysis.py` labels every validation failure with the WHU target-quality flags
Task 6D used (`tiny_target`, `border_truncation`, touching neighbours, ambiguous/insufficient
context) plus a **quantization ceiling** — the IoU of the quantized-and-dequantized GT box against the
continuous GT box at the selected `B`. Structural failures (malformed sequences) are separated from
localization failures, because a target's size, border or ambiguity cannot explain a sequence that
never emitted a valid box. Measured on the E1 model: 120 structural failures, **0 localization
failures**, 0 samples whose quantization ceiling is below 0.5, structural modes
`{no_seg_token_emitted: 120, runaway_location_run: 120, no_box_token_emitted: 0}` and
`whu_data_quality_dominates: null` (no localization evidence to attribute). **WHU target quality is
not the binding constraint of this result.** Artifact: `evaluation/task6e_error_analysis.json`.

## 10. Verdict (section 16)

`evaluation/task6e_verdict.json`: **`EXPLICIT_SPATIAL_TOKENS_FAILED`** — E0 and the tokenizer/loss
implementation are verified, the E1 geometry gate fails, and E2 is therefore not run. Section 17
applies: coordinate tokens are **not** the final novelty claim, and no architectural escalation
(`[REF]`, SRE, SCL, 4B, dataset change, full training, GUI) was attempted in this task.
