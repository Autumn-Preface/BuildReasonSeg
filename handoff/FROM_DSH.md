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

# FROM_DSH — Task 6R Report: Directional GRCL Feasibility

_This file holds the Task 6R report. The Task 6Q report is preserved in git history at commit
`7c19bec`; Task 6P at `b80f3cc`; Task 6O at `595e7bb`; Task 6N at `90f3735`; Task 6M.1 at `5e52d95`._

Full design notes and the literature-position note: `docs/task6r_grcl_directional_feasibility.md`.

## 1. Verdict

**`GRCL_NO_MEANINGFUL_RELATION_GAIN`** — section 22 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen artifacts unchanged, no test access, B3 not retrained, no
   target-as-input.
2. `GRCL_IMPLEMENTATION_INVALID` — no: the audit is `GRCL_VALID`.
3. `GRCL_OVERFIT_FAIL` — no: the R1 Overfit20 gate passes.
4. `GRCL_MASK_RELATION_TRADEOFF` — not applicable: mask retention passes.
5. **`GRCL_NO_MEANINGFUL_RELATION_GAIN`** — mask retention passes, but the relation-accuracy gain is
   below 0.08 **and** PairedVal is below 16/20. ← **verdict**
6. `GRCL_FIELD_REDUNDANCY_RISK` — not reached.
7. `GRCL_DIRECTIONAL_FEASIBLE` — no.

No criterion, gate or `lambda` was altered, and the test split was never read.

## 2. Literature-position note (fixed)

Recent RRSIS work already contains "consistency" losses/regularizers (e.g. text–vision structural
consistency, cross-modal alignment consistency), so the word "consistency" itself is not novel. The
candidate contribution tested here is narrower: **a mask-level reference–target geometric relation
constraint whose variables are the predicted target-mask centroid, the grounded reference-mask centroid
and the instruction-specified spatial relation**. No "first-ever" or final-novelty claim.

## 3. GRCL v0.1

`buildreasonseg_mvp/grcl_directional.py`: `alpha = 1.2`, `tau = 0.04`, `eps = 1e-6`,
`lambda_grcl = 0.5`; soft differentiable target centroid from `sigmoid(Z_t)` (never detached, never
thresholded inside the loss); signed primary displacement and orthogonal displacement per relation
(left_of `cx_r − cx_t`; right_of `cx_t − cx_r`; above `cy_r − cy_t`; below `cy_t − cy_r`);
`L_margin = relu(tau − signed)`, `L_axis = relu(alpha·orth − signed)`,
`L_GRCL = mean(L_margin + L_axis)`; `L_total = L_BCE + L_Dice + 0.5 · L_GRCL`. No `tau` normalisation,
no softplus, no extra sign/field/contrastive/pair loss.

`evaluation/task6r_grcl_audit.json` = **`GRCL_VALID`**: 8 deterministic non-binary soft masks, all four
relations, non-symmetric references, all hinges active, finite loss, gradient present/finite with
minimum L1 **0.026917**, and the directional sanity check passes on all four relations (correct-side
loss 0, wrong-side loss 0.5395).

## 4. R0 — frozen B3 relation baseline

`evaluation/task6r_b3_relation_baseline.json`: the Task 6O B3 checkpoint (SHA256 verified, not retrained)
reproduces MiniVal240 mIoU **0.4299680351479113** and Dice **0.5401551056992948** with **delta 0.0**.

| R0 | MiniVal240 | PairedVal20 members |
|---|---|---|
| mIoU / Dice / Pr@0.5 | 0.429968 / 0.540155 / 0.692450 | — |
| hard relation accuracy | **0.950000** | **0.950000** |
| mean positive signed margin | 0.230544 | 0.264902 |
| axis violation rate | 0.012987 | 0.000000 |
| mean GRCL (metric) | 0.018724 | — |
| per-direction accuracy (left/right/above/below) | 0.9333 / 0.9167 / 0.9833 / 0.9667 | — |

The frozen baseline already sits at 0.95 relation accuracy, so the predeclared `+0.08` gain has only
0.05 of headroom — reported as a measurement, not as a criterion change.

## 5. Training

**Overfit20** (`evaluation/task6r_overfit20.json`), AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 /
no scheduler / no augmentation / seed 20260929 / evaluate every 100 steps, identical for both variants:

| Variant | mIoU | Dice | Pr@0.5 | relation accuracy | signed margin | axis violation |
|---|---|---|---|---|---|---|
| **R1** (B3 + GRCL) | **0.917843** | **0.933565** | 0.990689 | **0.950000** | 0.191155 | 0.0 |
| R2 (no-field + GRCL) | 0.932792 | 0.957379 | 0.957196 | 1.000000 | 0.218146 | 0.0 |

**R1 overfit gate: PASS** (0.917843 ≥ 0.85, 0.933565 ≥ 0.90, 0.950000 ≥ 0.95); R2 has no gate.

**Mini stage** (`evaluation/task6r_training.json`): fresh initialisation, AdamW / lr 3e-4 / wd 1e-4 /
batch 8 / ≤ 25 epochs / patience 5 / selection = MiniVal240 mIoU / seed 20260929; R1 selected epoch 3,
R2 selected epoch 9.

## 6. MiniVal240 and PairedVal20 (R0 / R1 / R2)

| Metric | R0 (frozen B3) | **R1 (B3 + GRCL)** | R2 (no-field + GRCL) |
|---|---|---|---|
| mIoU | **0.429968** | **0.411857** | 0.233787 |
| Dice | 0.540155 | 0.536478 | 0.329888 |
| Pr@0.5 | 0.692450 | 0.626946 | 0.417037 |
| hard relation accuracy | 0.950000 | **0.954167** | 0.541667 |
| mean signed margin | 0.230544 | 0.238583 | 0.179077 |
| axis violation rate | 0.012987 | **0.000000** | 0.431034 |
| mean GRCL | 0.018724 | 0.047917 | 0.098439 |
| per relation mIoU (left/right/above/below) | — | 0.3741 / 0.3985 / 0.4307 / 0.4441 | 0.2167 / 0.2245 / 0.2372 / 0.2567 |
| per relation accuracy | 0.9333 / 0.9167 / 0.9833 / 0.9667 | 0.9333 / 0.9333 / 1.0000 / 0.9500 | — |
| largest / smallest family mIoU | — | 0.3694 / 0.4543 | 0.2022 / 0.2653 |
| parameters | 275,777 | 274,625 | 273,473 |
| selected epoch | 9 | 3 | 9 |
| wall time / peak VRAM | — | 57.3 s / 0.65 GB | 51.7 s / 0.65 GB |
| PairedVal20 pass | 14/20 | **12/20** | 10/20 |
| paired own / cross / margin | 0.398969 / 0.001773 / +0.397196 | 0.365532 / 0.001471 / **+0.364060** | 0.157666 / 0.039519 / +0.118147 |
| paired member relation accuracy (40) | 0.950000 | 0.925000 | 0.400000 |

## 7. Proposal-reference transfer diagnostic

`evaluation/task6r_proposal_reference_transfer.json` — trained R1 on the **frozen Task 6Q** proposal
resolver (exact frozen configuration, resolver cache reused, abstaining exactly when the resolver
abstains; diagnostic only, no training, no `lambda` selection):

| Metric | R1 under the 6Q resolver | Task 6Q B3 chain |
|---|---|---|
| target mIoU | 0.283782 | 0.304581 |
| PairedVal20 pass | 2/20 | 10/20 |
| own − cross margin | +0.079561 | +0.273700 |
| hard relation accuracy | 0.829100 | — |
| abstentions | 6 | 6 |

## 8. Predeclared criteria and the strong-signal flag

| # | Criterion | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | R1 overfit gate | pass | PASS | ✓ |
| 2 | mask retention `R1 ≥ R0 − 0.02` | ≥ 0.4099680351479113 | **0.411857** | ✓ |
| 3 | relation gain `R1 ≥ R0 + 0.08` | ≥ 1.03 (R0 = 0.95) | **+0.004167** | ✗ |
| 4 | R1 PairedVal | ≥ 16/20 | **12/20** | ✗ |
| 5 | R1 own − cross margin | ≥ 0.38 | **+0.364060** | ✗ |
| 6 | field remains useful `R1 − R2` | ≥ 0.10 | **+0.178069** | ✓ |

`strong_mask_gain = false` (R1 is 0.018111 mIoU below R0, not `R0 + 0.02`).

Measured context, without interpretation: GRCL keeps mask quality essentially intact (inside the 0.02
retention allowance) and slightly raises relation accuracy (0.950000 → 0.954167) while eliminating axis
violations (0.012987 → 0.000000), but it misses the predeclared gain and PairedVal falls 14/20 → 12/20
(margin +0.397196 → +0.364060). The field remains materially useful (`R1 − R2 = +0.178069` mIoU).

## 9. Tests, storage, git

`python -m pytest tests/ -q` → **733 passed, 1 skipped** (Task 6Q ended at 696 passed / 1 skipped; no
prior passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism
check that needs the proposal env). `tests/test_task6r_grcl_directional_feasibility.py` covers the 37
section-23 checks: Task 6Q artifacts unchanged, Task 6O B3 hash exact, field v0.2 unchanged, `alpha`
exactly 1.2, `tau` exactly 0.04, `lambda_grcl` exactly 0.5, differentiable target centroid, finite
nonzero target-logit GRCL gradient, no thresholding inside GRCL, hard-relation threshold 0.5 evaluation
only, exact left/right/above/below signs, exact axis and margin terms, R1 architecture equals B3, R1 has
the field, R1 has no direct reference channel, R2 architecture equals B0, R2 has no field input, R2 has
no reference-mask model input, oracle reference used by R2 only in loss/evaluation, exact frozen packs
reused, identical training settings for R1/R2, no test split, GT target never a model input, proposal
transfer uses the frozen Task 6Q resolver/config, proposal transfer does not tune, no YOLO retraining,
no `[REF]`, no nearest/L3, no MLLM hidden-state fusion, no graph transformer, no 4B, no new
dataset/download/install/GUI, previous suite preserved.

Not committed: checkpoints, proposal caches, feature caches, SAM2/YOLO weights, source imagery/vector
data, `.conda`, large caches. Committed: code, small JSON artifacts, docs, tests, handoff.

Watt was **not needed** in Task 6R: this task downloaded nothing (no new weights, packages or datasets)
and installed nothing. The pre-existing Watt instance is transport-only, is not owned by this project,
and was left running, per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 10. Interpretation boundary (section 24)

DSH reports measurements only: it does not claim final novelty, does not change `lambda`, does not design
another relation loss, and does not start nearest/L3, MLLM integration or reference-grounding redesign.

## 11. Recommended next step

等待 ChatGPT 根据 Task 6R 的 GRCL 因果实验结果决定 Task 6S，不自行修改 loss、reference 架构或开始 MLLM/nearest/L3。

## 12. STOP

Task 6R stops here: no MLLM integration, no nearest, no L3, no joint reference-target training, no new
reference architecture, no full-dataset training, no GUI. Waiting for the ChatGPT audit.
