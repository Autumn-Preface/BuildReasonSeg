# Task 6P — Differentiable Relation Field + Predicted-Reference Substitution

> Task: `handoff/TO_DSH.md` (Task 6P) · Base commit: `595e7bb` · Predecessor: Task 6O →
> `FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED`
> **Verdict: `REFERENCE_HEAD_INSUFFICIENT`**
> Tests: `tests/test_task6p_differentiable_field_predicted_reference.py` · Evidence: `evaluation/task6p_*.json`

Task 6P asks whether the **oracle** reference mask can be replaced by a **learned predicted** reference
mask while preserving enough of the field-guided target-segmentation signal. It is not yet
language-to-mask end-to-end, not joint reference-target training and not a final MLLM architecture. The
reference family id (`largest` / `smallest`) comes from the canonical program record, which isolates
reference grounding. The test split was never read.

## 1. Implementation correction recorded (v0.1 is not differentiable)

`buildreasonseg_mvp/geometric_relation_field.py` (v0.1) is **frozen and unchanged**. Recorded facts:

* v0.1 calls `.detach()` on tensor masks;
* v0.1 converts the centroid to a Python `float`.

Therefore v0.1 is numerically smooth as a spatial function but is **not differentiable with respect to
the input reference mask**. This did **not** invalidate Task 6N/6O: there the reference mask was
oracle/frozen and no gradient to the reference source was required. Predicted-reference training does
require that gradient, hence v0.2.

## 2. GeometricRelationField v0.2

`buildreasonseg_mvp/geometric_relation_field_v02.py` — same formula, autograd preserved:
`mass = sum(M_ref) + eps` (`eps = 1e-6`), tensor soft centroid with pixel-centre normalized
coordinates (`x = (col+0.5)/W`, `y = (row+0.5)/H`), `alpha 1.2`, `tau 0.04`,
`s_axis = s_margin = 0.02`, `P_rel = clamp(sign · axis · margin, 0, 1)`, no learned parameters, and
**no** `.detach()`, Python-float centroid or NumPy in the forward path. Accepts `(H,W)`, `(B,H,W)` or
`(B,1,H,W)` masks and one relation id per batch element. The reduction order deliberately mirrors v0.1
so binary masks agree to float32 rounding.

`evaluation/task6p_field_v02_audit.json` — **`FIELD_V02_VALID`**:

| Gate | Requirement | Measured |
|---|---|---|
| equivalence masks | 64 deterministic binary oracle masks from MiniVal240, balanced 16 per direction | 16 / 16 / 16 / 16 |
| max abs error vs v0.1 at 64×64 | ≤ **1e-6** | **0.0** |
| mean abs error vs v0.1 | ≤ **1e-7** | **0.0** |
| gradient: soft masks with `requires_grad` | ≥ 8, all four directions | 8, four directions |
| gradient to soft `M_ref` | exists, finite, L1 > **1e-8** | finite, min L1 **11,833.33** |

## 3. Frozen B3 reproduced with the v0.2 oracle field

`evaluation/task6p_b3_reproduction.json` — the Task 6O B3 checkpoint (path + SHA256 verified, **not
retrained**) re-evaluated once on MiniVal240 with the oracle mask processed by **v0.2**:

| Metric | Frozen Task 6O B3 | Reproduced | Absolute delta |
|---|---|---|---|
| MiniVal240 mIoU | 0.4299680351479113 | 0.4299680351479113 | **0.0** |
| MiniVal240 Dice | 0.5401551056992948 | 0.5401551056992948 | **0.0** |

Verdict **`B3_REPRODUCED`** (tolerance 1e-6).

## 4. Reference packs (deduplicated)

`evaluation/task6p_reference_pack_manifest.json`, unique key
`(split, tile_id, reference_source_feature_id, reference_family)`:

| Pack | Size | Families |
|---|---|---|
| RefTrainUnique | **825** (from 1,000 MiniTrain1000 records, dedup 1.21×) | 491 largest / 334 smallest |
| RefValUnique | **219** (from 240 MiniVal240 records) | 110 largest / 109 smallest |
| Reference Overfit20 | **20** unique train references | 10 largest / 10 smallest |

Distinct `(tile, source_feature_id)`, no duplicate reference mask, and the packs carry **no** target
identity and **no** relation id (asserted by tests).

## 5. ReferenceMaskHead v0.1

`buildreasonseg_mvp/task6p_reference_head.py` — frozen SAM2 feature (256×64×64) + family id
(`largest`/`smallest`, embedding 2×16) → `Conv1x1(256→128)+GroupNorm(8,128)+GELU` →
concat(128 + 16 = **144**) → `Conv3x3(144→128)+GN+GELU` → `Conv3x3(128→64)+GN+GELU` → `Conv1x1(64→1)`
→ bilinear upsample to 512×512; loss exactly `BCEWithLogitsLoss + DiceLoss`. Parameters:
**273,441** (project 33,152 + family embedding 32 + trunk 240,257). No relation id, no target mask, no
relation field, no YOLO proposal, no candidate mask.

## 6. Reference-head training

**P1 (Reference Overfit20)** — AdamW / lr 1e-3 / wd 1e-4 / 1200 steps / batch 4 / no scheduler / no
augmentation / seed 20260929 / evaluated every 100 steps: best **mIoU 0.969504, Dice 0.984388** →
gate (≥ 0.85 / ≥ 0.90) **PASS**.

**P2 (RefTrainUnique → RefValUnique)** — fresh initialisation, AdamW / lr 3e-4 / wd 1e-4 / batch 8 /
max 30 epochs / early stopping patience 6 on RefValUnique mIoU / seed 20260929; selected epoch **12**:

| Metric | Value |
|---|---|
| RefValUnique mIoU | **0.220176** |
| Dice | **0.302794** |
| Precision@0.5 | 0.359514 |
| largest-reference mIoU / Dice (n=110) | 0.279714 / 0.376918 |
| smallest-reference mIoU / Dice (n=109) | **0.160091** / 0.227989 |
| centroid error (normalized) mean / median / p90 | 0.150767 / **0.124060** / **0.319618** |
| median `pred_area / gt_area` | 1.019795 |
| border-reference mIoU | no border references in RefValUnique (0 records) |
| non-border-reference mIoU (n=219) | 0.220176 |
| tiny-reference mIoU | no tiny references in RefValUnique (0 records) |
| best epoch / wall time / peak VRAM | 12 / 47.8 s / 0.65 GB |

The head fits 20 references almost perfectly (0.9695) but generalizes to only **0.220176** on 219 unique
validation references, with a median normalized centroid error of **0.124060**.

## 7. Predicted-reference chain (Parts G-H)

`evaluation/task6p_predicted_reference_target_val.json` — for every frozen MiniVal240 record: frozen
SAM2 feature → frozen ReferenceMaskHead with the canonical family → `M_ref_pred = sigmoid(logits@512)`
→ `P_rel_pred` via **v0.2** → frozen Task 6O **B3** → target mask. No oracle reference mask enters this
path; the GT reference is used only for reference-head metrics and the GT target only for scoring.

| Metric | Predicted reference (B3) | Oracle reference (frozen B3) |
|---|---|---|
| target mIoU | **0.240968** | 0.429968 |
| Dice | 0.316164 | 0.540155 |
| Precision@0.5 | 0.546548 | — |
| border target (n=114) | 0.231872 | 0.413228 |
| tiny target (n=4) | ≈0 | ≈0 |
| PairedVal20 pass | **0/20** | 14/20 |
| paired mean own / cross IoU | 0.064774 / 0.061155 | 0.398969 / 0.001773 |
| paired own − cross margin | **+0.003619** | +0.397196 |

Deltas: target mIoU **−0.189000**, paired pass **−14**, own−cross margin **−0.393577**. Per-family:
largest-reference mIoU 0.223331, smallest-reference mIoU 0.258818 (see artifact for the full tables).

`evaluation/task6p_field_propagation_diagnostics.json` — predicted vs oracle v0.2 field on MiniVal240:
**MAE 0.120730**, **RMSE 0.274298**, **mean per-record Pearson 0.653476**, predicted-vs-oracle
reference centroid error mean 0.150767 / median 0.124060 / p90 0.319618, with per-family and
per-relation breakdowns. Diagnostic only; no threshold was tuned.

## 8. Gates and verdict

**Section 17 reference-head adequacy — FAIL**: mIoU 0.220176 < 0.35; median normalized centroid error
0.124060 > 0.05; p90 0.319618 > 0.12.

**Section 18 predicted-reference chain retention — FAIL**: section 17 does not pass; predicted target
mIoU 0.240968 < 0.3009776246035379 (= 70 % of oracle B3); PairedVal 0/20 < 10; margin +0.003619 < 0.20.

`evaluation/task6p_verdict.json` — priority order applied literally:

1. `INVALID_EXPERIMENT` — no.
2. `DIFFERENTIABLE_FIELD_INVALID` — no (v0.2 audit passes).
3. `TASK6O_B3_REPRODUCTION_FAIL` — no (reproduced exactly).
4. `REFERENCE_HEAD_NOT_LEARNABLE` — no (P1 passes).
5. **`REFERENCE_HEAD_INSUFFICIENT`** — section 17 fails. ← **verdict**
6. `REFERENCE_ERROR_PROPAGATION_SEVERE` — not reached (only evaluated when section 17 passes).
7. `PREDICTED_REFERENCE_CHAIN_FEASIBLE` — no.

## 9. Interpretation boundary (section 21)

DSH reports measurements only. This document does **not** claim final novelty, does **not** choose
joint training, `[REF]`, GRCL, nearest/L3, another reference architecture or a different field, and no
gate or threshold was altered.

## 10. Reproduce

```text
python scripts/task6p_field_v02_audit.py                          # Part B
python scripts/task6p_freeze_reference_packs.py                   # Part D
python scripts/task6p_train_reference.py --stage p1               # P1 (gate)
python scripts/task6p_train_reference.py --stage p2               # P2
python scripts/task6p_evaluate_chain.py --stage b3-reproduction   # Part C
python scripts/task6p_evaluate_chain.py --stage chain             # Parts G-H
python scripts/task6p_report.py                                   # Parts I-J
```

Run in `.conda/buildreasonseg-proposal`. Checkpoints live under the gitignored
`artifacts/task6p/checkpoints/`; reference packs under `artifacts/task6p/reference_packs/`; the frozen
Task 6N feature cache is reused.
