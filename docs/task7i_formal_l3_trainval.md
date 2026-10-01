# Task 7I — Formal L3 Three-Seed Train/Validation: Z-B3 vs D-B1

> Task: `handoff/TO_DSH.md` (Task 7I) · Base commit: `511a7bd` · Predecessors: Task 7G →
> `LARGEST_SELECTOR_NOT_LEARNABLE`, Task 7H → `DEVELOPMENT_ARCHITECTURE_FROZEN`
> **Verdict: `DB1_FORMAL_VAL_NOT_CONFIRMED`** (7/8 section-25 gates) · **test lock stayed LOCKED**
> Six formal runs (Z-B3 × 3 seeds, D-B1 × 3 seeds) on the full 1344-record L3 train population, checkpoint
> selection and early stopping on all 936 oracle-reference L3 val records
> Tests: `tests/test_task7i_formal_trainval.py` · Evidence: `evaluation/task7i_*.json`

This is the formal train/validation confirmation stage for the frozen L3 target-decoder architecture. It is a
validation result, **not** a test result; no test image, mask, annotation, record id, metric or test-derived
statistic was read or produced.

## 1. Recorded Task 7G result

`LARGEST_SELECTOR_NOT_LEARNABLE`. Internal tile-disjoint selector holdout: G-I0 deterministic mean selected
reference IoU `0.5199913587`; G-I1 learned `0.6198877726` (gain `+0.0998964138`), oracle-best exact top-1
`0.6591928251`, mean best-minus-selected gap `0.1790755783`. Gate: gain ≥ +0.08 PASS · mean selected IoU ≥ 0.62
FAIL · top1 ≥ 0.55 PASS · gap ≤ 0.14 FAIL. Therefore the external E-Holdout stage was not executed, no
scene-disjoint claim exists for the Task 7G selector, the selector is not adopted and reference intervention
stops.

## 2. Frozen development roles (unchanged)

Dataset `WHU-EA-NativeVector v1.0 + BuildSpatialReason v0.2 + scene_disjoint_v1`; parser = Task 7C Qwen3-VL-2B
text-only ProgramHead, controlled/canonical interface only and **not** used for checkpoint selection; proposal =
Task 6M.1 YOLO26m-seg U-C1; practical reference = deterministic Task 6Q largest selector; directional field =
GeometricRelationField v0.2; nearest field = NearestBoundaryField v0.1; L3 baseline = Z-B3; preferred candidate
= D-B1; all rejected modules remain excluded.

## 3. Task 7I protocol corrections (Task 7H artifacts untouched)

* **I-01** — Task 7H's `formal_d_b1_training.schedule.checkpoint_selection = "MiniVal240 mIoU"` is superseded
  by **all 936 valid v0.2 L3 validation records, oracle-reference mIoU**.
* **I-02** — the formal comparison retrains **both Z-B3 and D-B1** with the identical formal train/val
  population, identical seeds, identical optimizer schedule and fresh trainable initialization. The historical
  Z-B3/D-B1 checkpoints remain development evidence only.

## 4. Formal populations (frozen before training)

`evaluation/task7i_formal_population_manifest.json`:

| | Records | left | right | above | below |
|---|---:|---:|---:|---:|---:|
| **train** (`G-FormalTrain`) | **1344** | 323 | 347 | 338 | 336 |
| **val** (`G-FormalVal`) | **936** | 250 | 249 | 224 | 213 |

Train/val sample-id overlap **0**, both sample-id hashes recorded, no test material read. `I-FormalValPairsAll`
(`evaluation/task7i_formal_val_pair_manifest.json`): every unique unordered formal-val pair with the same tile,
the same oracle largest reference, a different direction and a different target — **326 pairs** (0 duplicates),
sorted lexicographically, never subsampled, reporting-only (never used for selection).

## 5. Training protocol and runs

Fresh initialization per seed for both architectures (no historical checkpoint ever loaded as initialization),
frozen SAM2 `V ∈ R^(256×64×64)` features, frozen fields, oracle GT reference, loss exactly
`BCEWithLogitsLoss + DiceLoss`, AdamW lr 3e-4 / weight decay 1e-4 / batch 8 / max 25 epochs / patience 5 /
no scheduler / no augmentation / bfloat16 AMP, gradient updates on the formal train population only.

| Run | Selected epoch | Final epoch | Val mIoU (936) | Val Dice | Val Pr@0.5 | Params | Wall | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z-B3 / 20261001 | 4 | 9 | 0.3194 | — | — | 275,777 | 120.6 s | — |
| Z-B3 / 20261002 | 14 | 19 | 0.3394 | — | — | 275,777 | 75.8 s | — |
| Z-B3 / 20261003 | 5 | 10 | 0.3386 | — | — | 275,777 | 40.6 s | — |
| D-B1 / 20261001 | 11 | 16 | 0.3799 | — | — | 278,081 | 126.3 s | — |
| D-B1 / 20261002 | 9 | 14 | 0.3827 | — | — | 278,081 | 62.1 s | — |
| D-B1 / 20261003 | 8 | 13 | 0.3877 | — | — | 278,081 | 57.5 s | — |

(`best.pt` / `last.pt` per run live in the gitignored `artifacts/checkpoints/task7i/<model>/<seed>/`; exact
per-epoch histories, checkpoint hashes, bytes, peak VRAM and NaN/Inf status are in
`evaluation/task7i_training_zb3.json` / `..._db1.json`.)

## 6. Oracle-reference full-val evaluation

| Run | mIoU | Dice | Pr@0.5 | Pairs (326) | Margin |
|---|---:|---:|---:|---:|---:|
| Z-B3 / 20261001 | 0.3194 | 0.4360 | — | 274 | +0.3027 |
| Z-B3 / 20261002 | 0.3394 | 0.4613 | — | 283 | +0.3255 |
| Z-B3 / 20261003 | 0.3386 | 0.4604 | — | 293 | +0.3218 |
| **D-B1 / 20261001** | **0.3799** | 0.4965 | — | 260 | +0.3789 |
| **D-B1 / 20261002** | **0.3827** | 0.5079 | — | 289 | +0.3766 |
| **D-B1 / 20261003** | **0.3877** | 0.5117 | — | 277 | +0.3770 |

Aggregates (mean ± sample std, `ddof=1`): **Z-B3 0.3324 ± 0.0113** mIoU, pair pass rate 0.8691, margin +0.3167;
**D-B1 0.3834 ± 0.0039** mIoU, pair pass rate 0.8446, margin **+0.3775**. Matched-seed deltas D-B1 − Z-B3:
`+0.0606`, `+0.0433`, `+0.0491` → **3/3 seeds improve**.

## 7. Practical predicted-reference full-val evaluation

Frozen U-C1 proposals (imgsz 640, conf 0.05, max_det 300, default NMS, no TTA, no tiling) → deterministic
largest selector (max area → higher confidence → lower index); no Task 7G selector. The reference was computed
once and shared by all six runs: reference mIoU **0.4469**, Dice —, abstentions **5**, buckets
**REFERENCE_OK 491**, **SELECTION_WRONG 305**, **NOT_COVERED 135**, ABSTENTION 5, GEOMETRY_POOR 0,
best-eligible coverage@0.50 — (recorded in the artifact).

| Run | Strict mIoU | Answered-only | Reference-OK subset | Pairs (326) | Margin |
|---|---:|---:|---:|---:|---:|
| Z-B3 / 20261001 | 0.2203 | 0.2215 | 0.3440 | 163 | — |
| Z-B3 / 20261002 | 0.2267 | 0.2279 | 0.3544 | 172 | — |
| Z-B3 / 20261003 | 0.2340 | 0.2352 | 0.3657 | 169 | — |
| D-B1 / 20261001 | 0.2259 | 0.2271 | 0.3697 | 122 | — |
| D-B1 / 20261002 | 0.2356 | 0.2369 | 0.3843 | 149 | — |
| D-B1 / 20261003 | 0.2390 | 0.2403 | 0.3895 | 137 | — |

Aggregates: Z-B3 strict **0.2270**, pair pass rate 0.5153, margin +0.1952; D-B1 strict **0.2335**, pair pass
rate 0.4172, margin +0.2009. Oracle→predicted retention per D-B1 seed: 0.5946 / 0.6158 / 0.6164 (mean 0.6090).

## 8. Formal comparison and verdict

| # | Section-25 gate | Required | Measured | Pass |
|---|---|---|---:|---|
| 1 | D-B1 mean oracle full-val mIoU | ≥ 0.35 | 0.3834 | ✓ |
| 2 | D-B1 mean − Z-B3 mean (oracle) | ≥ +0.04 | +0.0510 | ✓ |
| 3 | matched seeds with D-B1 > Z-B3 | ≥ 2/3 | 3/3 | ✓ |
| 4 | lowest D-B1 seed oracle mIoU | ≥ 0.32 | 0.3799 | ✓ |
| 5 | D-B1 mean − Z-B3 mean (predicted strict) | ≥ +0.02 | **+0.0065** | **✗** |
| 6 | D-B1 mean predicted strict mIoU | ≥ 0.22 | 0.2335 | ✓ |
| 7 | D-B1 predicted pair margin not lower by > 0.02 | ≥ Z-B3 − 0.02 | +0.2009 (Z-B3 +0.1952) | ✓ |
| 8 | no protocol violation | — | — | ✓ |

Section 28 priority → **`DB1_FORMAL_VAL_NOT_CONFIRMED`** (gate 5 is the only failure).

Measured reading (reported only, no repair proposed): under the **oracle** reference the D-B1 advantage is
confirmed and markedly more seed-stable than Z-B3 (mean +0.0510 mIoU, 3/3 matched seeds, std 0.0039 vs 0.0113,
pair margin +0.3775 vs +0.3167). Under the **practical predicted** reference the same comparison loses most of
that advantage (+0.0065 strict mIoU, below the predeclared +0.02 gate) and its counterfactual pair *pass rate*
falls to 0.4172 versus Z-B3's 0.5153 even though its margin stays slightly higher (+0.2009 vs +0.1952). The
reference stage — 305/936 `SELECTION_WRONG` and 135/936 `NOT_COVERED` — dilutes the decoder-level gain, exactly
the limitation recorded in Tasks 7F/7H.

## 9. Test lock

`evaluation/task7i_test_lock_status.json`: `status = LOCKED`, `test_execution_authorized = false`,
`task7i_completed_train_val = true`, `db1_formal_val_confirmed = false`,
`unlock_requires = "ChatGPT audit of Task 7I"`. The Task 7H lock was read and re-verified as LOCKED before and
after the stage; DSH does not unlock it and does not authorize any final-test action.

## 10. Interpretation boundary

DSH reports measurements only. Validation results are not called test results; the test split is never called
untouched; no seed was chosen by test; no unfavourable seed was dropped; no architecture, loss, hyperparameter,
field, parser or reference policy was changed after seeing the formal val results; the test split stays LOCKED;
no final-test inference was started; parser/reference/YOLO were not retrained; no further selector was trained;
unrestricted natural-language capability, unseen-city generalization and novelty/"first" claims are not made.
Final recommendation exactly:

`等待 ChatGPT 审核 Task 7I 的三种子正式 train/val 结果；在 ChatGPT 明确解锁前，test 保持 LOCKED，不运行任何 final-test inference。`
