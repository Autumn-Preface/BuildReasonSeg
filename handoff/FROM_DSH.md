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

# FROM_DSH — Task 6Y Report: Oracle-Reference Nearest Boundary Field Feasibility

_This file holds the Task 6Y report. The Task 6X report is preserved in git history at commit `9318890`;
Task 6W at `3de2142`; Task 6V at `3e185fb`; Task 6U at `b8b5224`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task6y_oracle_nearest_boundary_field.md`.

## 1. Verdict

**`NEAREST_FIELD_NO_MEANINGFUL_GAIN`** — section 26 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths unchanged, no test use, exactly two nearest programs, four
   predeclared variants, no predicted reference, `sigma_diag` untouched and labels not regenerated.
2. `NEAREST_FIELD_DEPENDENCY_UNAVAILABLE` — no: SciPy 1.17.1 provides `distance_transform_edt`.
3. `NEAREST_PAIRED_SET_INSUFFICIENT` — no: **N_pair = 20** (≥ 12).
4. `NEAREST_BOUNDARY_FIELD_SEMANTIC_MISMATCH` — **no: the sanity gate passed perfectly** (top-1 1.0000,
   top-3 1.0000, mean Spearman 1.0000).
5. `NEAREST_FIELD_NOT_LEARNABLE` — no: B2 Overfit20 mIoU **0.9663** / Dice **0.9827** (≥ 0.85 / 0.90).
6. `NEAREST_FIELD_GEOMETRY_ONLY_CONFOUND` — no: B2−B3 = **+0.2396** ≥ 0.10 while B3's paired pass rate is
   0.30 < 0.60.
7. **`NEAREST_FIELD_NO_MEANINGFUL_GAIN`** — learnability passes but **B2−B1 = +0.0224 < +0.05** and
   **B2 MiniVal mIoU 0.2864 < 0.35**. ← **verdict**
8. `NEAREST_FIELD_COUNTERFACTUAL_WEAK` — not reached (a mask-gain condition already fails).
9. `NEAREST_BOUNDARY_FIELD_FEASIBLE` — no (6 of 9 section-25 criteria pass).

## 2. Recorded Task 6X result (reference hardening stops here)

`SAM2_REFINEMENT_NOT_HELPFUL`: train-only U-Calib200 X-C0 U-C1 baseline mIoU `0.4789276` vs X-C1 `0.4744`,
X-C2 `0.4284`, X-C3 `0.3741` → X-C0 won and Task 6X stopped before RefVal/MiniVal/Paired. **Frozen
practical reference resolver: `U-C1 + deterministic Task 6Q area semantics`** (Task 6U development
metrics: RefVal reference mIoU ≈ 0.4289355, MiniVal answered ≈ 0.3141364, strict ≈ 0.3089008, PairedVal
11/20, margin ≈ +0.3209). Known limitations: proposal coverage / extreme selection (especially smallest),
tiny buildings, scene-disjoint support-module generalization.

## 3. Field semantic sanity (parameter-free, section 7)

`evaluation/task6y_field_sanity.json` — Y-MiniVal240, `sigma_diag = 0.05`, no tuning:

| Metric | Measured | Gate |
|---|---|---|
| target top-1 | **1.0000** | ≥ 0.85 ✓ |
| target top-3 | **1.0000** | ≥ 0.97 ✓ |
| mean Spearman vs −boundary_distance | **1.0000** | ≥ 0.90 ✓ |
| mean target / best distractor score | 0.4622 / 0.1678 | — |
| mean target − distractor | **+0.3641** | — |
| mean eligible candidates | 4.40 | — |

By family: largest 1.0 / 1.0 / 1.0 with 5.01 candidates; smallest 1.0 / 1.0 / 1.0 with 3.80 candidates.

## 4. Frozen nearest packs (seed 20260930)

`evaluation/task6y_pack_manifest.json` (train/val v0.2 only, stable-hash selection):

| Pack | Records | Split | Unique tiles |
|---|---|---|---|
| Y-Overfit20 | 20 (10 + 10) | train | **20** |
| Y-MiniTrain1000 | 1000 (**650** largest + **350** smallest) | train | 786 |
| Y-MiniVal240 | 240 (**120 + 120**) | val | 221 |
| Y-PairedVal | **N_pair 20** (same tile, different reference ids, different target ids) | val | 20 |

Every pack contains only `largest_to_nearest` / `smallest_to_nearest`; no directional, L3 or test record.
Nearest semantics frozen: `boundary_distance`, `margin_px_floor 2.0`, `margin_diag_fraction 0.005`,
`margin_mode normalized_with_absolute_floor`; canonical labels were not regenerated.

## 5. Variants, training and evaluation

| Variant | Inputs | Fusion | Params | Overfit20 mIoU/Dice | MiniVal240 mIoU | Dice | Pr@0.5 | largest | smallest | epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| Y-B0 | visual_128 + embed16 | 144 | 273,425 | 0.9724 / 0.9860 | 0.1837 | 0.2735 | 0.2625 | 0.1374 | 0.2301 | 4 |
| Y-B1 | visual_128 + M_ref_64 + embed16 | 145 | 274,577 | 0.9705 / 0.9849 | 0.2640 | 0.3643 | 0.3525 | 0.2273 | 0.3008 | 6 |
| **Y-B2** | visual_128 + P_near_64 + embed16 | 145 | 274,577 | **0.9663 / 0.9827** | **0.2864** | **0.3863** | **0.4812** | **0.2336** | **0.3393** | 9 |
| Y-B3 | field_128 + embed16 (no visual) | 144 | 240,785 | 0.1457 / 0.2066 | 0.0468 | 0.0822 | 0.0713 | 0.0235 | 0.0701 | 8 |

Deltas: **B2−B0 = +0.1027**, **B2−B1 = +0.0224**, **B2−B3 = +0.2396**, B1−B0 = +0.0803. Loss exactly
`BCEWithLogitsLoss + DiceLoss`; Y1 lr 1e-3 / batch 4 / 1200 steps / eval every 100; Y2 lr 3e-4 / wd 1e-4 /
batch 8 / ≤25 epochs / patience 5 / selection by MiniVal240 mIoU; seed 20260930; bfloat16 AMP; wall time
30–135 s and peak VRAM 0.67–0.90 GB per variant.

Border-target and tiny-target mIoU are **not applicable** (0 records): the frozen nearest eligibility
rejects border-truncated and tiny components, so no nearest pack contains such a target.

B2 quartiles — target area (edges 1038.0 / 1636.5 / 2316.75 px): 0.2497 / 0.3904 / 0.2859 / 0.2198;
canonical target boundary distance (edges 8.35 / 25.66 / 82.06 px): **0.4094 / 0.3477 / 0.2356 / 0.1531** —
the field's advantage concentrates at small boundary distances, exactly as designed.

PairedVal (N_pair 20): Y-B0 **0/20** (+0.0000), Y-B1 **8/20** (+0.1840), **Y-B2 6/20** (**+0.1828**),
Y-B3 6/20 (+0.0363).

## 6. Criteria

| # | Criterion | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | field semantic sanity | 0.85 / 0.97 / 0.90 | 1.0000 / 1.0000 / 1.0000 | ✓ |
| 2 | B2 overfit | mIoU ≥ 0.85, Dice ≥ 0.90 | 0.9663 / 0.9827 | ✓ |
| 3 | B2 − B0 mIoU | ≥ +0.08 | **+0.1027** | ✓ |
| 4 | B2 − B1 mIoU | ≥ +0.05 | **+0.0224** | ✗ |
| 5 | B2 − B3 mIoU | ≥ +0.10 | **+0.2396** | ✓ |
| 6 | B2 mIoU | ≥ 0.35 | **0.2864** | ✗ |
| 7 | B2 paired pass rate | ≥ 0.70 | **0.30** (6/20) | ✗ |
| 8 | B2 own-cross margin | ≥ 0.15 | **+0.1828** | ✓ |
| 9 | no target GT model input | — | GT target = label/eval only | ✓ |

Measured reading (reported, not prescribed): the oracle-reference field is **semantically exact** and adds
clear signal over the visual baseline (+0.1027) and over geometry alone (+0.2396), with its advantage
concentrated at small boundary distances; but it does not beat the raw reference-mask control by the
required +0.05, absolute MiniVal quality stays below 0.35, and only 6/20 paired counterfactuals pass — so
the nearest relation is not demonstrated as a feasible dense-segmentation target from this causal setup.

## 7. Tests, storage, git

`python -m pytest tests/ -q` → **1011 passed, 1 skipped** (Task 6X ended at 965 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task6y_nearest_boundary_field.py` adds the 46 section-O checks.

Not committed: nearest pack JSONs, the four checkpoints, feature caches, source imagery/vectors, `.conda`.
Committed: field/decoder code, small manifests and evaluation JSON, scripts, tests, docs, handoff.

Task 6Y downloaded nothing and installed nothing (SciPy was already present in both environments). Watt was
**not needed** in Task 6Y: the pre-existing Watt instance is transport-only, is not owned by this project
and was left running per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 8. Interpretation boundary

DSH reports measurements only. No global novelty claim and no end-to-end nearest capability claim;
`sigma_diag` unchanged; no learned distance transform; no predicted reference; no combination of directional
and nearest fields; no L3; the reference resolver was not changed.

## 9. Recommended next step (exact wording required by Part M)

等待 ChatGPT 根据 Task 6Y 的 nearest boundary field 因果结果决定下一步，不自行进行 predicted-reference nearest 集成、direction+nearest 场组合或 L3 训练。

## 10. STOP

Task 6Y stops here: the frozen reference resolver is untouched, no predicted-reference nearest integration,
no directional+nearest combination, no L3, no GRCL, no test access, no GUI. Waiting for the ChatGPT audit.
