# Task 6R — Directional Geometry-Relation Consistency Loss (GRCL) Feasibility

> Task: `handoff/TO_DSH.md` (Task 6R) · Base commit: `7c19bec` · Predecessor: Task 6Q →
> `REFERENCE_PROPOSAL_COVERAGE_INSUFFICIENT`
> **Verdict: `GRCL_NO_MEANINGFUL_RELATION_GAIN`**
> Tests: `tests/test_task6r_grcl_directional_feasibility.py` · Evidence: `evaluation/task6r_*.json`

Controlled loss ablation on the second candidate algorithm contribution: does **explicit supervision on
the geometric relation between the predicted target mask and the reference mask** improve relation
correctness and counterfactual target discrimination beyond field guidance alone, without materially
degrading mask quality? Four directional relations only (`left_of`, `right_of`, `above`, `below`) and
the same 8 directional L2 programs. Oracle reference throughout (intentional, to isolate the loss); GT
target is a label/score only; the test split was never read.

## 1. Literature-position note (fixed; no new search performed)

* Recent RRSIS work already contains "consistency" losses/regularizers, including text–vision structural
  consistency and cross-modal alignment consistency.
* Therefore the word "consistency" itself is **not** novel.
* The candidate contribution tested here is narrower: **a mask-level reference–target geometric relation
  constraint whose variables are the predicted target mask centroid, the grounded reference mask
  centroid, and the instruction-specified spatial relation**.
* No "first-ever" claim and no final-novelty claim is made from Task 6R alone.

## 2. GRCL v0.1 — exact definition

`buildreasonseg_mvp/grcl_directional.py`. Constants (no sweep): `alpha = 1.2`, `tau = 0.04`,
`eps = 1e-6`, `lambda_grcl = 0.5`; `alpha`/`tau` match the frozen Task 3B semantics.

```text
P_t    = sigmoid(Z_t)
mass_t = sum(P_t) + eps                        cx_t = sum(P_t·x)/mass_t     cy_t = sum(P_t·y)/mass_t
mass_r = sum(M_ref) + eps                      cx_r = sum(M_ref·x)/mass_r   cy_r = sum(M_ref·y)/mass_r

left_of:  signed = cx_r − cx_t   orth = |cy_t − cy_r|
right_of: signed = cx_t − cx_r   orth = |cy_t − cy_r|
above:    signed = cy_r − cy_t   orth = |cx_t − cx_r|
below:    signed = cy_t − cy_r   orth = |cx_t − cx_r|

L_margin = relu(tau − signed)          L_axis = relu(alpha·orth − signed)
L_GRCL   = mean(L_margin + L_axis)     L_total = L_BCE + L_Dice + 0.5 · L_GRCL
```

The target logits are never detached and probabilities are never thresholded inside the loss; the oracle
reference is treated as fixed. No `tau` normalisation, softplus, additional sign/field/contrastive/pair
loss is used. The **hard relation metric** (section 11, evaluation only, threshold 0.5) is
`relation correct iff signed >= tau AND signed >= alpha·orth`, with an empty predicted mask counted as
incorrect.

`evaluation/task6r_grcl_audit.json` = **`GRCL_VALID`**: 8 deterministic non-binary soft target-logit
tensors covering all four relations with non-symmetric reference masks, all hinges active, loss finite,
gradient present/finite with minimum L1 **0.026917** (> 1e-8); the directional sanity check passes for
all four relations (moving farther into the correct relation gives loss 0; crossing to the wrong side
gives 0.5395).

## 3. R0 — frozen B3 relation baseline

`evaluation/task6r_b3_relation_baseline.json`: the Task 6O B3 checkpoint (SHA256 verified, **not**
retrained) reproduces MiniVal240 mIoU `0.4299680351479113` and Dice `0.5401551056992948` with
**absolute delta 0.0**.

| R0 (frozen B3, BCE + Dice) | MiniVal240 | PairedVal20 members |
|---|---|---|
| mIoU / Dice / Pr@0.5 | 0.429968 / 0.540155 / 0.692450 | — |
| hard relation accuracy | **0.950000** | **0.950000** |
| mean positive signed margin | 0.230544 | 0.264902 |
| axis violation rate | 0.012987 | 0.000000 |
| mean GRCL (as a metric) | 0.018724 | — |
| per direction accuracy (left/right/above/below) | 0.9333 / 0.9167 / 0.9833 / 0.9667 | — |

The frozen baseline is already at **0.95** relation accuracy, i.e. the predeclared `+0.08` gain has only
0.05 of headroom (reported as a measurement, not as a criterion change).

## 4. Training (Parts E-G)

**Stage R1 (frozen Overfit20)** — AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler / no
augmentation / seed 20260929 / evaluate every 100 steps, identical for both variants:

| Variant | mIoU | Dice | Pr@0.5 | relation accuracy | mean GRCL | signed margin | axis violation |
|---|---|---|---|---|---|---|---|
| **R1** (B3 + GRCL) | **0.917843** | **0.933565** | 0.990689 | **0.950000** | 0.0 | 0.191155 | 0.0 |
| R2 (no-field + GRCL) | 0.932792 | 0.957379 | 0.957196 | 1.000000 | 0.0 | 0.218146 | 0.0 |

**R1 overfit gate: PASS** (mIoU 0.917843 ≥ 0.85, Dice 0.933565 ≥ 0.90, relation accuracy 0.950000 ≥
0.95); R2 has no gate. `evaluation/task6r_overfit20.json`.

**Stage R2 (MiniTrain1000 → MiniVal240)** — fresh initialisation, AdamW / lr 3e-4 / wd 1e-4 / batch 8 /
≤ 25 epochs / early stopping patience 5 / model selection = MiniVal240 mIoU / seed 20260929
(`evaluation/task6r_training.json`): R1 selected epoch 3, R2 selected epoch 9.

## 5. Main evaluation (Parts H)

`evaluation/task6r_mini_val.json` and `evaluation/task6r_paired_val.json`.

| Metric | R0 (frozen B3) | **R1 (B3 + GRCL)** | R2 (no-field + GRCL) |
|---|---|---|---|
| MiniVal240 mIoU | **0.429968** | **0.411857** | 0.233787 |
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
| paired own / cross / margin | 0.398969 / 0.001773 / **+0.397196** | 0.365532 / 0.001471 / **+0.364060** | 0.157666 / 0.039519 / +0.118147 |
| paired member relation accuracy (40 predictions) | 0.950000 | 0.925000 | 0.400000 |

## 6. Proposal-reference transfer diagnostic (Part I)

`evaluation/task6r_proposal_reference_transfer.json` — trained R1 onto the **frozen Task 6Q** proposal
resolver (exact frozen configuration, resolver cache reused, abstaining exactly when the resolver
abstains). Diagnostic only; it never selected `lambda` and trained nothing.

| Metric | R1 under the 6Q resolver | Task 6Q B3 chain |
|---|---|---|
| target mIoU | 0.283782 | 0.304581 |
| PairedVal20 pass | 2/20 | 10/20 |
| own − cross margin | +0.079561 | +0.273700 |
| hard relation accuracy | 0.829100 | — |
| abstentions | 6 | 6 |

## 7. Predeclared criteria, strong-signal flag and verdict

| # | Criterion | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | R1 overfit gate | pass | PASS | ✓ |
| 2 | mask retention `R1 ≥ R0 − 0.02` | ≥ `0.4099680351479113` | **0.411857** | ✓ |
| 3 | relation gain `R1 ≥ R0 + 0.08` | ≥ 1.03 with R0 = 0.95 | **+0.004167** | ✗ |
| 4 | R1 PairedVal | ≥ 16/20 | **12/20** | ✗ |
| 5 | R1 own − cross margin | ≥ 0.38 | **+0.364060** | ✗ |
| 6 | field remains useful `R1 − R2` | ≥ 0.10 | **+0.178069** | ✓ |

`strong_mask_gain = false` (R1 mIoU is 0.018111 below R0, so not `R0 + 0.02`).

`evaluation/task6r_verdict.json` — section 22 priority order applied literally:

1. `INVALID_EXPERIMENT` — no (frozen artifacts unchanged, no test access, no B3 retraining, no
   target-as-input).
2. `GRCL_IMPLEMENTATION_INVALID` — no (`GRCL_VALID`).
3. `GRCL_OVERFIT_FAIL` — no (gate passes).
4. `GRCL_MASK_RELATION_TRADEOFF` — not applicable (mask retention passes).
5. **`GRCL_NO_MEANINGFUL_RELATION_GAIN`** — mask retention passes but the relation-accuracy gain is
   below 0.08 and PairedVal is below 16/20. ← **verdict**
6. `GRCL_FIELD_REDUNDANCY_RISK` — not reached.
7. `GRCL_DIRECTIONAL_FEASIBLE` — no.

Measured context, reported without interpretation: GRCL leaves mask quality essentially intact
(−0.018111 mIoU, inside the 0.02 retention allowance) and slightly improves relation accuracy
(0.950000 → 0.954167) while removing axis violations (0.012987 → 0.000000), but it does not reach the
predeclared gain, and PairedVal drops 14/20 → 12/20 (margin +0.397196 → +0.364060). The field remains
materially useful: R1 − R2 = +0.178069 mIoU.

## 8. Interpretation boundary (section 24)

DSH reports measurements only. This document does **not** claim final novelty, does **not** change
`lambda`, does **not** design another relation loss, and does **not** start nearest/L3, MLLM integration
or reference-grounding redesign. No criterion or gate was altered.

## 9. Reproduce

```text
python scripts/task6r_grcl_audit.py            # Part C (gradient + directional audit)
python scripts/task6r_train.py --stage overfit # Part F (R1/R2 Overfit20 + gate)
python scripts/task6r_train.py --stage mini    # Part G (MiniTrain1000 -> MiniVal240)
python scripts/task6r_evaluate.py --stage baseline   # Part D (R0 reproduction + relation metric)
python scripts/task6r_evaluate.py --stage final      # Part H (R0/R1/R2 MiniVal + PairedVal)
python scripts/task6r_proposal_transfer.py     # Part I (diagnostic)
python scripts/task6r_report.py                # Parts J-K (criteria + verdict)
```

Run in `.conda/buildreasonseg-proposal` (frozen SAM2 + torch stack). Checkpoints live under the
gitignored `artifacts/task6r/checkpoints/`; the frozen Task 6N feature cache and the Task 6Q resolver
cache are reused unchanged.
