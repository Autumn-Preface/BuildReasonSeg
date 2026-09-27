<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task 6E Report: Explicit Spatial Token Grounding v0.1

_This file holds the Task 6E report. The Task 6D.1 report is preserved in git history and in
`docs/task6d1_corrective_grounding_audit.md`; Task 6D is in `docs/task6d_spatial_grounding_bridge.md`
and Task 6C.7 in `docs/task6c7_visual_cache_optimization.md`._

Full design notes: `docs/task6e_explicit_spatial_tokens.md`.

## 1. Verdict

**`EXPLICIT_SPATIAL_TOKENS_FAILED`.**

The explicit-spatial-token *implementation* is verified end to end: the quantized-box oracle keeps the
pathway's ceiling (0.7343 mIoU / paired 20/20 at `B = 256`), the tokenizer/row/loss/off-by-one
machinery passes a real one-step smoke, and E0 overfits 20 records to **20/20 structural, 20/20 exact
four-token sequences and 0.9166 box IoU** in 500 steps. The **E1 mini-train on 480 paired records
fails the geometry gate completely**: structural validity **0/120** at every one of the three epochs.

The failure mode is specific, and it is *not* a formatting or leakage defect: teacher forcing reaches
reasoning 1.000, `[BOX]` 1.000 and EOS 1.000, but location-token accuracy only 0.021 → 0.058 → 0.075
and `[SEG]` accuracy **0.000**. Free generation emits exactly one `[BOX]` (120/120) and then a
**runaway location run** of 51–81 tokens (attractor tokens `<loc_150>`/`<loc_255>`) with **no `[SEG]`
at all**, so every sequence is malformed. The location objective does not even fit its own training
data inside the fixed ≤3-epoch budget (train location CE 9.64 → 5.15, against the 256-way uniform floor
`ln 256 = 5.545`), while the same code path, loss and schedule overfit the first 20 of those same 480
records in E0. E2 (inference-only segmentation) was therefore **not run**, per section 14.

The honest reading: **making the box an explicit language target does not by itself make a 2B model
emit 256-way coordinate tokens under a 3-epoch, 480-record budget.** Nothing here says the *pathway* is
wrong — the oracle still recovers 0.7343 mIoU from a 256-bin box — it says the *learning signal and
budget* for the new spatial vocabulary were insufficient, and that the failure must be fixed there
rather than by training SAM2 or escalating the stack.

## 2. Frozen Evidence (section 1)

Not rerun, only quoted:

* Oracle Point (GT point → SAM2): mIoU **0.4876**, paired **18/20**.
* Oracle BOX (GT box → SAM2): mIoU **0.7506**, Dice 0.8507, paired **20/20**,
  IoU(pred_A, pred_B) **0.0000**.
* Task 6C `P_C` reference: strict e2e mIoU **0.10604**, paired mask **0/20**, same-image
  IoU(pred_A, pred_B) ≈ 0.999.
* Corrected Task 6D G0-R: emission 120/120, geometry paired **0/20**, box IoU **0.0088**.
* Task 6D.1: the frozen `[SEG]` representation contains **no practically decodable** target geometry
  with the tested readouts (linear overfit-20 0.903 / train-480 0.027 / val 0.004; raw MLP
  0.645/0.036/0.008; LayerNorm MLP 0.778/0.051/0.007; all paired 0/20; real ≈ label-shuffled).

## 3. Quantized Oracle by Bin Count (sections 3–5)

The continuous GT box was replaced by its deterministic enclosing quantization and fed to the same
official frozen SAM2 box prompt, on the same 120 validation records and 20 paired validation images:

| B | strict mIoU | Dice | paired | quantized-vs-GT box IoU | max coord error | own−cross margin |
|---|---|---|---|---|---|---|
| continuous | **0.7506** | 0.8507 | **20/20** | — | — | — |
| 32 | 0.5400 | 0.6889 | 20/20 | 0.5608 | 0.0323 | +0.5284 |
| 64 | 0.6473 | 0.7773 | 20/20 | 0.7193 | 0.0159 | +0.6217 |
| 128 | 0.7129 | 0.8259 | 20/20 | 0.8413 | 0.0079 | +0.6765 |
| **256** | **0.7343** | 0.8397 | **20/20** | 0.9129 | 0.0039 | +0.6919 |

Artifact: `evaluation/task6e_quantized_oracle.json`.

## 4. Selected Bin Count (section 5)

Rule: the smallest `B` with paired **20/20** and strict mIoU ≥ `0.7506 − 0.03 = 0.7206`.

**Selected `B = 256`** (the only qualifying candidate: 0.7343 ≥ 0.7206, paired 20/20). `B = 128`
reaches 0.7129 and misses the tolerance by 0.0077. Selecting `B = 256` costs 257 new tokens and
recovers 97.8 % of the continuous-oracle mIoU.

## 5. Vocabulary / Trainable Token Setup (sections 6–7)

* Added: `[BOX]` + `<loc_000>…<loc_255>` = **257** special tokens (vocabulary 151 670 → 151 927;
  embedding table resized to the tokenizer length, weight tying preserved).
* Every added token is exactly one id, has a unique id, and round-trips through
  `decode(..., skip_special_tokens=False)`.
* Trainable token rows: **258** = `{[SEG], [BOX], <loc_0> … <loc_255>}` through PEFT
  `trainable_token_indices` (528 384 row parameters); the project-owned multi-row forward hook and the
  multi-row output-row wrapper are the fallbacks. The base table (151 927 × 2048) stays frozen.
* One-step smoke (real forward/backward, `evaluation/task6e_token_setup.json`): `[SEG]` row changed,
  `[BOX]` row changed, all four target `<loc_*>` rows changed; **16 ordinary rows changed by exactly
  0.0** and the base embedding tensor is bit-identical; every sampled new **output** row received
  non-zero gradient from an output-head-only loss while the base output matrix received none; zero
  LoRA modules under the visual tower. `passed: true`.

## 6. Target Sequence and Loss (sections 8–9)

Assistant target, built from explicit token ids: `{reasoning_zh} [BOX] <loc_x1> <loc_y1> <loc_x2>
<loc_y2> [SEG]` + EOS. Label positions for `reasoning / [BOX] / loc / [SEG] / EOS` are computed from
the known spans under `logits at position i predict token i+1` and verified against
`input_ids[i+1]` for every supervised position. Objective **`1.0 · L_assistant + 5.0 · L_location`**,
`L_location` on exactly the four location-token prediction positions; no mask loss, no SmoothL1, no
weight sweep. Trainable set: text LoRA + the 258 token rows; Qwen base, visual tower, all SAM2, the
projection MLP and the Task 6D head frozen.

## 7. E0 Overfit (section 10)

20 deterministic records (10 same-image/different-target pairs), ≤1500 steps, scheduler horizon = the
real step budget.

| Step | structural | exact 4-token | mean box IoU | paired geometry |
|---|---|---|---|---|
| 250 | 0/20 | 0/20 | — | 0/10 |
| **500** | **20/20** | **20/20** | **0.9166** | **10/10** |

`E0_PASS` at step 500 (training plateaued: total loss 88.8 → 0.004; location-token accuracy 0.0 →
1.000). Per-coordinate token accuracy 1.0 on all four coordinates, so the residual 0.0834 is the
quantization ceiling rather than model error. E0's weights are **not** carried into E1.
Artifact: `evaluation/task6e_e0_overfit.json`.

## 8. E1 Training (section 11)

480 paired `P` records (240 images × 2 counterfactual instructions), 3 epochs × 480 steps = **1440**
optimizer steps with the corrected cosine horizon (warmup 20), LoRA 1e-4 / token rows 3e-4, gradient
clip 1.0, SAM2 fully frozen, no mask loss. Free generation on the fixed 120-record validation set and
the fixed 20 paired validation images after every epoch. Wall clock **7027 s** (117 min) for the whole
stage, including all three free-generation evaluations.

| Epoch | structural | exact 4-token | mean box IoU | paired geometry | TE `[BOX]` | TE loc | TE `[SEG]` | train loc CE |
|---|---|---|---|---|---|---|---|---|
| 1 | 0/120 | 0/120 | — | 0/20 | 1.000 | 0.021 | 0.000 | 7.45 |
| 2 | 0/120 | 0/120 | — | 0/20 | 1.000 | 0.058 | 0.000 | 6.42 |
| 3 | 0/120 | 0/120 | — | 0/20 | 1.000 | 0.075 | 0.000 | 5.15 |

Model selection (priority: geometry paired pass → mean box IoU → structural validity) is tied at
`[0, 0.0, 0.0]` for all three epochs and resolves to **epoch 1**. Training trace: total loss
49.7 → 26.3, assistant CE 1.53 → 0.51 (converged), location CE 9.64 → 5.15 (stalled at the 256-way
floor `ln 256 = 5.545`), location-token accuracy **0.000** at every logged step, absolute bin error
61.5 → 118.25 (the predictions drift to the vocabulary extremes). Token rows 393 trainable tensors /
17 960 960 parameters; `sam_frozen` and `projection_frozen` both true. Artifact:
`evaluation/task6e_e1_training.json`.

## 9. Free-Generation Geometry (sections 12–13)

Parsed from **token ids** (never text); no GT or prose repair. Selected epoch: **1**.

| Metric | Value |
|---|---|
| structural validity | **0.000** (0/120) |
| exact four-loc-token sequence | 0/120 |
| per-coordinate token accuracy (x1, y1, x2, y2) | 0.0, 0.0, 0.0, 0.0 |
| mean absolute bin error | undefined (no valid sequence) |
| normalized coordinate error | undefined (no valid sequence) |
| mean predicted-box IoU | undefined (no valid sequence) |
| center-inside-target rate | 0.000 |
| `[BOX]` emission | exactly one in **120/120** |
| `[SEG]` count histogram | `{0: 120}` |
| generated length | 96 tokens (the cap) for every sample, location runs 51–81 |
| teacher-forced loc-token accuracy (diagnostic only) | 0.075 (epoch 3) |

By level: 0/40 valid at each of L1, L2, L3. By query family: 0 valid in `extreme` (27), `direction`
(32), `nearest` (8), `multi_hop_direction_to_nearest` (40), `size` (13). Every failure is structural —
the sequence never emitted `[SEG]` — so no box IoU, no coordinate accuracy and no L1/L2/L3 *quality*
comparison exists. Teacher-forced accuracy is diagnostic only; the gate uses free generation.
Artifact: `evaluation/task6e_geometry_eval.json`.

## 10. Paired Geometry Probe (section 11 priority 1)

**0/20** paired images pass the geometry probe (both instructions must produce a valid box with IoU
≥ 0.5 against their own target). Because no sequence is structurally valid, the same-image
predicted-box L1 distance is undefined and the "different tokens for different instructions" question
cannot be answered from generated boxes: both instructions simply run away into the same location-token
attractor. This is **not** evidence that instruction conditioning improved — Task 6D's 0/20 paired
result stands unchanged for the mask path, and the geometry path never produced a comparable box.
Artifact: `evaluation/task6e_paired_probe.json` (`mask_ran: false`).

## 11. E2 End-to-End Segmentation (sections 14–15)

**Not run.** Section 14 makes E2 conditional on the E1 geometry gate, which failed. No GT geometry was
ever used as a prompt, and no SAM2 weight was updated anywhere in Task 6E. The frozen SAM2 box-prompt
ceiling for this pathway is already measured by the oracle audit (**0.7343 mIoU / paired 20/20** at
`B = 256`), which localizes the missing capability unambiguously *before* SAM2: the box.

## 12. Paired Mask Probe (section 15)

**Not run** (same reason). For reference, the frozen numbers Task 6E did not move: Task 6C `P_C`
mask mIoU **0.10604** / paired **0/20**; Task 6D Oracle BOX **0.7506** / paired **20/20**.

## 13. L1/L2/L3 + Query Breakdown (sections 13, 15)

Geometry breakdown by level (structural valid / count): L1 **0/40**, L2 **0/40**, L3 **0/40**.
By query family (structural valid / count): `extreme` 0/27, `direction` 0/32, `nearest` 0/8,
`multi_hop_direction_to_nearest` 0/40, `size` 0/13. The failure is uniform across levels and
families — consistent with a vocabulary-learning failure rather than a relation-specific one; there is
no level or family that partially worked, so no breakdown can be over-read.

## 14. Error / Data Adequacy Analysis (section 18)

`evaluation/task6e_error_analysis.json` (120 records, IoU threshold 0.5):

* **120 structural failures, 0 localization failures** — the classification separates them explicitly,
  because target size / border / ambiguity cannot explain a sequence with no valid box;
* structural modes: `runaway_location_run` 120, `no_seg_token_emitted` 120, `no_box_token_emitted` 0,
  `multiple_box_tokens` 0, `empty_generation` 0;
* quantization ceiling: **0/120** samples have a quantized GT box below 0.5 IoU against the continuous
  GT box (mean 0.913 box IoU at `B = 256`), so quantization is not the constraint either;
* `whu_data_quality_dominates: null` — with zero localization failures there is no evidence that WHU
  target quality is the binding constraint of this result. (WHU flags still describe the material:
  border truncation 35/120, ambiguous/insufficient context 6/120, touching neighbours 2/120 — but they
  cannot cause a malformed token sequence.)
* **Conclusion:** the dominant remaining error is the unlearned 256-way location vocabulary, i.e. the
  training signal/budget for the new spatial tokens, not the data and not SAM2.

## 15. Reusable Inference Plumbing (section 19)

`buildreasonseg_mvp/spatial_inference.py` provides exactly the four functions the spec asks for —
`generate_spatial_tokens`, `parse_box_tokens`, `predict_box`, `predict_mask_from_generated_box` —
with no GUI and no top-level `predict.py` (that waits for ChatGPT review). Ground truth cannot enter
any of them: the generation functions take only an image and an instruction, and the mask comes from
the generated box through the official frozen SAM2 box prompt. Section 19 gates the reusable path on
E1 passing; the plumbing is in place and unit-tested, but its end-to-end quality claim is open.

## 16. Runtime / VRAM

* E0: 610 s for model load + 500 steps + two 20-sample free-generation evaluations
  (0.9166 box IoU checkpoint, 100 MB).
* E1: 7027 s for model load + 1440 steps + three (120 val + 40 paired) free-generation evaluations
  plus three 120-record teacher-forced passes; ~13.6 GiB VRAM, batch 1, bf16 autocast, gradient
  checkpointing and the Task 6C.7 frozen visual-feature cache.
* The spatial objective is language-model-only: SAM2 runs in the oracle audit and (would run) in E2,
  never in a training step. E1's cost is dominated by free generation, not by the 1440 training steps.

## 17. Tests (section 21)

`tests/test_task6e_spatial_tokens.py` adds **38** tests covering the section 21 list: enclosing
quantizer determinism (4 bin counts), quantizer/dequantizer canonicality, non-empty-target
preservation, the oracle selection rule (smallest qualifying `B`, only `B = 256`, threshold
`0.7506 − 0.03`), `[BOX]`/`<loc_*>` single-token round-trips and id uniqueness against the real
tokenizer, base-table freezing, multi-row gradient flow on both the input and the **output** side with
ordinary rows bit-identical after a step, causal label-position off-by-one correctness (including a
negative test for a shifted layout), target format (one `[BOX]`, four loc, one `[SEG]`, in order), the
token-id parser rejecting nine malformed sequences, loss weighting `1.0 · assistant + 5.0 · location`,
no-GT-in-the-inference-path, the true-total-step scheduler horizon (source + numeric), the recorded
one-step-row smoke assertions, forbidden components (`[REF]`, SRE, SCL, 4B), no test-split reads,
artifact determinism, and the Task 6D.1 wording/boolean correction.

Full suite: **{{TESTS_RESULT}}**.

## 18. Git / Watt (section 22)

{{GIT_SECTION}}

## 19. Recommended Next Architecture Task (section 17)

Task 6E claims **no** novelty for coordinate tokens, and no escalation was attempted. The measured
result points at one narrow, testable next step and explicitly not at the ones the spec forbids:

1. **Fix the location-token learning signal before adding any new module (recommended).** Everything
   except the 256-way coordinate prediction is now verified: the oracle recovers 0.7343 mIoU / 20/20
   from a 256-bin box, the tokenizer/rows/loss/off-by-one machinery passes, and E0 proves the pathway
   can be driven to 20/20 exact tokens. E1 shows the *480-record* location objective stalls at the
   256-way uniform floor (train CE 5.15 vs `ln 256 = 5.545`) with `[SEG]` predicted 0 % of the time.
   Candidate fixes, in the order I would test them — **each one measured against the same E1 gate**:
   * **loss/optimization**: increase the location term's *effective* step (the global gradient clip
     scales a ~30–50 total loss to norm 1.0 every step, and the token rows share one LR with 256
     competing outputs); e.g. a warm-up that first teaches `[BOX] … [SEG]` with the four loc tokens
     masked out, then adds the coordinate values;
   * **target representation**: coarse-to-fine (emit 128 or 64 bins first, then refine), a separate
     head for the location logits over the 256 loc ids only (still no new *token* vocabulary), or
     ordering the coordinates so a wrong token cannot poison the `[SEG]` transition;
   * **budget**: the spec capped E1 at 3 epochs; whether the objective simply needs more steps is a
     question for the next task, and the corrected scheduler already handles an honest horizon.
2. **Only after geometry generalizes**: E2 (inference-only segmentation from the generated box) and
   then geometry-verifiable supervision — the emitted box is now a first-class output, so
   BuildSpatialReason's target-independent geometry can be checked at *token* level (relation
   satisfaction against reference/neighbour boxes) instead of only at mask level.
3. **Not indicated by anything measured here:** `[REF]`, Spatial Relation Encoder, Spatial Consistency
   Loss, 4B, a dataset change, full training, a GUI. The dataset is not the constraint (0/120
   localization failures, 0 quantization-limited samples) and SAM2 is not the constraint (0.7343 mIoU
   given the box).
