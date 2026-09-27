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

# FROM_DSH — Task 6F Report: Target-Aware `[BOX]` Query Grounding v0.1

_This file holds the Task 6F report. The Task 6E report is preserved in git history and in
`docs/task6e_explicit_spatial_tokens.md`; Task 6D.1 in `docs/task6d1_corrective_grounding_audit.md`._

Full design notes: `docs/task6f_target_aware_box_query.md`, ADR-017.

## 1. Verdict

**`TARGET_AWARE_QUERY_FAILED`.**

The target-aware query *implementation* is verified end to end: the `[BOX]` token is a fixed input
query with a **bit-identical** hidden under future-text mutation, the box head and both token rows
receive gradient in a real one-step smoke, and F0 overfits the same 20 records to **train box IoU
0.9046, paired 10/10 and clearly non-identical same-image boxes (L1 0.2285)** at step 1000. The
**F1 mini-train on 480 paired records fails the geometry gate completely**: geometry paired
**0/20**, val mean box IoU **0.025**, center-inside **0.017** at the selected (final) epoch.

The failure mode is specific: the box head **never fits even its own training set** — train
SmoothL1 plateaus at ≈0.003 (an RMS coordinate error of ~8 % of the tile) instead of converging the
way F0's 20-record run did — so this is under-fitting under the fixed ≤8-epoch budget, and the
predictions are canonical, spread over the image (spread 0.05 → 0.25) but unable to localize
WHU-scale targets (mean GT area 1.2 % of the tile; 93 % of failures carry `tiny_target`). The
language side is untouched: continuing generation from `image + instruction + [BOX]` reaches
exactly-one `[SEG]` **1.000** and EOS **1.000**. The representation diagnosis answers section 13
negatively on the main hypothesis: the query is no longer template-dominated, but it is
**image-dominated** (same-image centered cosine 0.626 vs 0.018 across images) with only moderate
target-geometry variation (hidden-distance vs GT-box-distance correlation 0.476, versus ≈0 for the
legacy `[SEG]`). F2 was therefore **not run** (section 15 gates it on the F1 gate).

The honest reading: **moving the supervised query before reasoning removes the template confound,
but at 2B / 8 epochs the query does not become target-specific enough to localize tiny targets — and
neither the autoregressive vocabulary (6E) nor the query head (6F) fits 480 records under the
spec's budget.** The bottleneck is now precisely measured: a geometry readout must localize to
~5 % of the tile at WHU target scales, and none of the three readouts tested (6D.1 `[SEG]` probes,
6E tokens, 6F query) achieves that at this scale/budget.

## 2. Frozen Evidence (section 1)

Not rerun, only quoted:

* continuous Oracle BOX → SAM2: strict mIoU **0.7506**, paired **20/20**;
* Task 6E `B = 256` quantized Oracle BOX: strict mIoU **0.7343**, paired **20/20**;
* Task 6D.1: legacy end-of-reasoning `[SEG]` hidden practically not decodable for target geometry
  (centered same-image cosine 0.077, effective rank 8.33, real ≈ label-shuffled at 480 samples);
* Task 6E: E0 20-record coordinate-token overfit succeeds, E1 480-record autoregressive training
  fails structurally **0/120**;
* current evidence does not indicate WHU or SAM2 as the binding constraint.

## 3. Initialization (section 3)

Clean base/seed convention identical to Task 6E E1: fresh Qwen3-VL-2B-Instruct, existing `[SEG]`
(id **151 669**), text-only LoRA (rank 16 / alpha 32 / dropout 0.05, q/k/v/o/gate/up/down), plus
exactly **one** new active query token `[BOX]` (id **151 670**; vocabulary 151 670 → 151 671). No
Task 6E `<loc_*>` rows exist in the Task 6F tokenizer and no Task 6E checkpoint is loaded. Trainable
token rows `{[SEG], [BOX]}` via PEFT `trainable_token_indices` (one 4096-element parameter); base
table frozen. Optimizer coverage: LoRA 392 tensors / 17 432 576 params @1e-4, box head 6 tensors /
1 055 236 params @3e-4, token rows 1 tensor / 4 096 params @3e-4 (weight decay 0).
`evaluation/task6f_token_setup.json`.

## 4. Query-Token Causal Placement (section 4)

Sequence: `image + instruction → [BOX] → reasoning_zh [SEG] EOS`. `[BOX]` is appended at
`box_position == prompt_length` (verified: appears exactly once, precedes every target token) and is
**never a prediction target** (the label at `prompt_length − 1` is `-100`). The direct causal test:
replacing the entire reasoning tail with unrelated text leaves the real model's `[BOX]` hidden
**bit-identical** (`max_abs_delta 0.0`). The query therefore attends to image + instruction + prefix +
its own embedding and to nothing from the future — the central difference from Task 6D.

## 5. TargetAwareBoxHead (section 5)

`LayerNorm(2048) → Linear(2048, 512) → GELU → Linear(512, 4) → sigmoid → min/max canonicalization`
— deliberately the same simple readout Task 6D used, so the only experimental change is *where the
supervised representation is formed*. Output: canonical normalized `(x1,y1,x2,y2)` in `[0,1]`.
One-step smoke (real forward/backward, `evaluation/task6f_token_setup.json`): `[BOX]` row, `[SEG]`
row and all six head parameters move with non-zero gradient; 16 ordinary rows change by exactly 0.0;
base embedding bit-identical; SAM2 fully frozen; visual tower zero LoRA; `passed: true`.

## 6. Trainables / Loss (sections 7–8)

`L_total = 1.0 · L_reasoning + 5.0 · L_box`; `L_reasoning` = causal CE over `reasoning_zh [SEG] EOS`
only, `L_box` = SmoothL1(pred_box, GT tight box). No mask BCE/Dice, no pairwise losses, no sweeps;
both raw losses recorded. Trainable: text LoRA + `[BOX]` row + `[SEG]` row + box head. Frozen: Qwen
base/visual tower, all SAM2, old Projection MLP, Task 6D head, and (asserted absent) any Task 6E loc
rows.

## 7. F0 Overfit (section 9)

20 deterministic records (10 same-image/different-target pairs), ≤1500 steps, corrected horizon,
evaluation through the inference-form query path only.

| Step | train box IoU | paired geometry | non-identical pairs |
|---|---|---|---|
| 250 | 0.0576 | 0/10 | 10/10 |
| 500 | 0.2464 | 0/10 | 10/10 |
| 750 | 0.7959 | 10/10 | 10/10 |
| **1000** | **0.9046** | **10/10** | **10/10** |

**`F0_PASS`** at step 1000 (train box IoU ≥ 0.85, paired ≥ 9/10, same-image non-identical boxes with
mean L1 **0.2285**, no GT leakage — the query path takes image + instruction + constant `[BOX]` only).
The run is bit-reproducible across processes (an earlier crashed write produced the identical
loss/IoU trace). F0's weights are not carried into F1. `evaluation/task6f_f0_overfit.json`.

## 8. F1 Training Curve (sections 10–11)

480 paired `P` records, 8 epochs × 480 steps = **3840** optimizer steps, one corrected cosine
scheduler over the true budget (warmup 20), per-epoch checkpoint + query-path validation, selection
by paired → val box IoU → center-inside, early stop after epoch 3 on 3 consecutive non-improvements
(never triggered; budget ran out). Wall clock **2448 s** (41 min) for all 8 epochs plus the final
diagnostics.

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

Selection is tied at `[0, val box IoU, center]` for all epochs and resolves to **epoch 8** (key
`[0, 0.0250, 0.017]`). Training trace: reasoning CE collapses to ≈0.0000 within epoch 1; box
SmoothL1 oscillates around 0.003–0.03 without converging; per-step train box IoU is 0.0 at 23 of 32
logged steps. Predicted boxes stay canonical (1.000) and in-range (1.000) with area ≈0.0078 against
GT mean 0.0120 — right size, wrong place. Artifact: `evaluation/task6f_f1_training.json`.

## 9. Geometry Metrics (section 12)

Selected epoch 8 (of 8): val mean box IoU **0.0250**, median **0.0**, center-inside **0.017**,
coordinate MAE **0.1543**, per-coordinate prediction spread **[0.25, 0.253, 0.248, 0.247]**. By
level: L1 **0.0109**, L2 **0.0182**, L3 **0.0459**. By query family: `size` 0.022, `extreme` 0.006,
`direction` 0.019, `nearest` 0.015, `multi_hop_direction_to_nearest` 0.046. The predictions are
spread across the image rather than collapsed, but no level or family localizes.
Artifact: `evaluation/task6f_geometry_eval.json`.

## 10. Paired Geometry Probe (section 11 priority 1)

**0/20** (both instructions must reach box IoU ≥ 0.5 against their own target). Mean same-image
predicted-box L1 **0.0979** (epoch 8; it varied 0.040 → 0.152 → 0.098 across epochs), mean
own-minus-cross box margin ≈ 0.0 — the two instructions of one image produce *different* boxes, but
neither box is near its own target, so no pair passes. This is not evidence of instruction
conditioning working; it is evidence the query produces image-driven variation without
localization. Artifact: `evaluation/task6f_paired_probe.json` (`mask_ran: false`).

## 11. `[BOX]` Representation Diagnostics (section 13)

Best checkpoint, global-mean centring (`evaluation/task6f_representation.json`):

| Measure | Task 6F `[BOX]` | Legacy `[SEG]` (frozen 6D.1) |
|---|---|---|
| same-image/diff-instruction cosine | **0.969** | ~0.999 raw |
| same-image centered cosine | **0.626** | 0.077 |
| different-image centered cosine | **0.018** | 0.531 |
| effective rank (participation ratio) | **7.07** | 8.33 |
| hidden-pair L2 vs GT-box L1 correlation | **0.476** | ≈0 (real ≈ shuffled) |

Answer to section 13's question: **partially, and negatively on the main hope.** The query is no
longer template-dominated (it cannot see reasoning at all) and it carries *measurable*
target-geometry variation (correlation 0.476 vs ≈0 for the legacy `[SEG]`), but it is
**image-dominated** — same-image pairs stay far closer than cross-image pairs (centered 0.626 vs
0.018) — so the query mostly encodes "which image", not "which target". No causal claim is made
from cosine alone. (Note: an earlier per-pair centring bug produced degenerate −1.0 values; the
recomputation uses the global mean and is recorded in the same artifact.)

## 12. Reasoning Compatibility (section 14)

Continuing generation from `image + instruction + fixed [BOX]` on the first 20 records of the fixed
validation set: exactly-one `[SEG]` **1.000**, EOS termination **1.000**, known-reasoning-template
**1.000**. The query prefix is fully compatible with generation — the F1 failure is entirely on the
geometry side. Template diagnostics only. Artifact: `evaluation/task6f_reasoning_compat.json`.

## 13. F2 Segmentation (sections 15–16)

**Not run.** Section 15 gates F2 on the F1 gate, which failed, so
`evaluation/task6f_segmentation_eval.json` is deliberately absent. No SAM2 weight was updated
anywhere in Task 6F and no GT geometry entered any prompt. Frozen ceilings for comparison: Task 6C
`P_C` 0.10604 / 0/20; continuous Oracle BOX 0.7506 / 20/20; Task 6E quantized oracle 0.7343 / 20/20.

## 14. Paired Mask Probe (section 16)

**Not run** (same reason). `evaluation/task6f_paired_probe.json` records `mask_ran: false` with the
geometry side only.

## 15. Error / Data Adequacy Analysis (section 18)

`evaluation/task6f_error_analysis.json` (120 records, IoU threshold 0.5): **120/120 failures**;
dominant class `wrong_target_valid_box` (98.3 %); `tiny_target` on **93.3 %**, `complex_l3_relation`
33 %, `border_truncation` 29 %, ambiguity 5 %, touching neighbours 1.7 %; 97.5 % of failures carry
at least one WHU quality flag, so the automatic `whu_data_quality_dominates` flag is true. The
honest reading: the failure is a **localization-precision** failure on inherently tiny targets (mean
GT area 1.2 % of the tile ⇒ a box needs ~5 % placement accuracy for IoU ≥ 0.5), combined with the
measured under-fitting (train box loss never converged). Simple cases (L1) do not work either
(0.011), so the dataset is not *the* cause — it is the sharpness of the metric at this scale plus a
budget-limited readout. Evidence for a future dataset task exists, but it is not the binding
constraint of this result.

## 16. Reusable Inference API (section 19)

Gated on F1 passing, so no product-facing `predict_box` / `predict_mask` / `generate_reasoning`
facade was created (section 19). The verified query-path plumbing — `predict_box_for_sample`,
`build_query_batch`, `generate_with_box_prefix` in `buildreasonseg_mvp/box_query.py` — is unit-tested
and remains available to the next task. No GUI.

## 17. Runtime / VRAM

F0: 20 records, 1000 steps, ~13 min wall clock. F1: 480 records, 3840 steps, **2448 s** wall clock
for 8 epochs plus per-epoch query-path validation (160 forwards each) and the final representation/
reasoning diagnostics; ~13–14 GiB VRAM, batch 1, bf16 autocast, gradient checkpointing, Task 6C.7
visual-feature cache. Training is language-model-only; SAM2 runs nowhere in F0/F1 and would run only
in the (gated-off) F2. F1's wall clock is dominated by the 480-step epochs (≈3.5 min each) plus the
8 query-path evaluations (≈1.5 min each) — no generation except the 20-sample compatibility check.

## 18. Tests (section 21)

`tests/test_task6f_box_query.py` adds **21** tests covering the section 21 list: `[BOX]` single-token
and uniqueness from `[SEG]` (real tokenizer), fixed-query insertion (never predicted, labels start at
the query position), causal non-attendance to future tokens (positional proof + a tiny causal
attention layer + the recorded real-model bit-identity smoke), the frozen Task 6D tight-box
convention, canonical/in-`[0,1]` head output and its determinism, GT-as-supervision-only (inspection
of the query path and generation), the fixed loss constants, box IoU/L1/center helpers, the recorded
one-step smoke (both rows moved with gradient, all six head parameters moved with gradient, 16
ordinary rows exactly unchanged, base bit-identical, SAM frozen, no visual LoRA, optimizer coverage
392+6+1 tensors), no forbidden components (`[REF]`, SRE, SCL, 4B), Task 6E coordinate machinery
retired (docstrings stripped before scanning), no loc tokens in the config, the true-total-step
scheduler horizon, the paired→IoU→center selection rule, the old `[SEG]` grounding head unused, box
head in checkpoints, deterministic JSON writing.

Full suite: **322 passed** in 712 s (`python -m pytest tests/ -q`, including the artifact-consistency
gate) — 301 pre-existing tests plus the 21 Task 6F tests, with no regressions.

## 19. Git / Watt (section 22)

**Commit: `4b4b4d2` — `feat: add target-aware box query`** (staged only code, tests, docs and
`evaluation/**` artifacts: no checkpoints, no weights, no `.conda`, no feature caches, no dataset
edits; `artifacts/**` stays gitignored and only `evaluation/task6f_checkpoint_manifest.json` records
the checkpoint hashes).

**Watt Toolkit handling:** the pre-existing Watt processes (`Steam++.exe`,
`Steam++.Accelerator.exe`) were left running untouched for the whole task and are used for the final
push only under the established ownership rules — never closed, never force-killed, no hosts-file,
certificate-store or TLS changes anywhere in this task.

## 20. Recommended Next Architecture Step (section 17)

Task 6F claims no novelty for the query token or the box head, and no escalation was attempted.
Three readouts have now been measured against the same frozen downstream oracle (0.7506 / 20/20):
the legacy `[SEG]` hidden (6D.1: no practically decodable geometry), the autoregressive coordinate
vocabulary (6E: overfits 20, fails 480 structurally), and the pre-reasoning query head (6F: overfits
20, underfits 480 — canonical boxes, wrong place). The next step should attack the measured
bottleneck — **localization at WHU target scale** — and none of the spec-forbidden escalations:

1. **Make the localization signal explicit and cheap to learn (recommended).** The box head is a
   6-tensor readout on a query hidden that is image-dominated (centered cosine 0.626 same-image,
   correlation 0.476). Candidates, each measurable against the F1 gate and the frozen oracle:
   (a) a **point-based** geometry target (Task 6D's point oracle reached 0.4876/18-20 with far fewer
   degrees of freedom) combined with box-style width/height heads; (b) a per-coordinate curriculum
   (center first, then extent); (c) longer honest training under the corrected scheduler — F0
   converges in ~1000 steps on 20 records, so 8 epochs on 480 records may simply be too little,
   which the spec capped; (d) supervision in SAM's prompt space rather than normalized [0,1].
2. **Only after geometry generalizes**: F2 (inference-only segmentation) and geometry-verifiable
   supervision — the query path is already the right interface for relation-level checks.
3. **Not indicated by anything measured here:** `[REF]`, SRE, SCL, 4B, a dataset change, full
   training, a GUI. The dataset contributes difficulty (93 % tiny targets) but is not the cause of
   the F1 failure (train loss never converged, L1 simple cases also fail).
