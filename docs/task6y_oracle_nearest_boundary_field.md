# Task 6Y — Oracle-Reference Nearest Boundary Field Feasibility

> Task: `handoff/TO_DSH.md` (Task 6Y) · Base commit: `9318890` · Predecessor: Task 6X →
> `SAM2_REFINEMENT_NOT_HELPFUL`
> **Verdict: `NEAREST_FIELD_NO_MEANINGFUL_GAIN`** · Field sanity **perfect** · B2 overfit **passed**
> Tests: `tests/test_task6y_nearest_boundary_field.py` · Evidence: `evaluation/task6y_*.json`

Task 6Y returns to the core spatial-reasoning method for the **nearest** relation family (canonical
semantics `boundary_distance`). It is an **oracle-reference causal experiment**: the GT reference mask is
used so reference errors cannot contaminate nearest-field feasibility, and no nearest end-to-end claim is
made. No model was trained other than the four predeclared nearest variants; the reference subsystem,
SAM2, GeometricRelationField v0.2 and all historical artifacts were left frozen.

## 1. Recorded Task 6X result (reference hardening stops here)

Verdict `SAM2_REFINEMENT_NOT_HELPFUL`. Train-only U-Calib200: X-C0 U-C1 baseline mIoU `0.4789276`, X-C1
exact-box single-mask `0.4744`, X-C2 exact-box multimask `0.4284`, X-C3 expanded-box multimask `0.3741`;
X-C0 won, so Task 6X correctly stopped before any RefVal/MiniVal/Paired evaluation.

**Frozen practical reference resolver for future integration: `U-C1 + deterministic Task 6Q area
semantics`.** Known development metrics from Task 6U: RefVal reference mIoU ≈ `0.4289355`, MiniVal answered
target mIoU ≈ `0.3141364`, strict target mIoU ≈ `0.3089008`, PairedVal `11/20`, own-cross margin
≈ `0.3209`. Known limitations remain: proposal coverage / extreme selection (especially smallest), tiny
buildings, and scene-disjoint support-module generalization.

## 2. Frozen nearest semantics and query scope

`configs/spatial_relations_v1.yaml` (read-only): metric **`boundary_distance`** (never centroid distance),
`margin_px_floor = 2.0`, `margin_diag_fraction = 0.005`, `margin_mode = normalized_with_absolute_floor`;
anchor and target eligibility unchanged (reject border-truncated, suspected-large-merge, tiny; require at
least two valid components). Canonical labels were **not** regenerated. Exactly two programs are handled:
`largest_to_nearest` and `smallest_to_nearest` (v0.2: train 672 / 358, val 458 / 249). No directional
program and no L3. Test split never read.

## 3. NearestBoundaryField v0.1

`buildreasonseg_mvp/nearest_boundary_field.py`, `sigma_diag = 0.05`, `eps = 1e-6`, source 512×512, decoder
field 64×64, no learned parameters, no sweep:

```text
R = M_ref > 0 ;  D_px = distance_transform_edt(~R)  (scipy.ndimage)
D_norm = D_px / sqrt(H^2 + W^2)
P_near_512 = exp(-D_norm / sigma_diag) ;  P_near_512[R] = 0 ;  clamp [0,1]
P_near_64  = bilinear(P_near_512, 64, 64, align_corners=False), clamped
M_ref_64   = area_resize(M_ref, 64, 64)          (Y-B1 only), clamped
```

No reference centroid, no bbox distance, no target mask, no candidate proposals and no GT candidate ids are
used. SciPy 1.17.1 is present, so `NEAREST_FIELD_DEPENDENCY_UNAVAILABLE` did not fire.

## 4. Field semantic sanity (section 7, parameter-free)

`evaluation/task6y_field_sanity.json` on the frozen Y-MiniVal240 (240 records):

| Metric | Measured | Gate |
|---|---|---|
| target top-1 rate | **1.0000** | ≥ 0.85 ✓ |
| target top-3 rate | **1.0000** | ≥ 0.97 ✓ |
| mean Spearman vs −boundary_distance | **1.0000** | ≥ 0.90 ✓ |
| mean target score | 0.4622 | — |
| mean best distractor score | 0.1678 | — |
| mean target − distractor | +0.3641 | — |
| mean eligible candidates | 4.40 | — |

By family: largest — top-1/top-3 1.0, mean Spearman 1.0, target 0.4045 vs distractor 0.1677, 5.01
candidates; smallest — top-1/top-3 1.0, mean Spearman 1.0, target 0.5200 vs distractor 0.1679, 3.80
candidates. **`sanity_passed = true`**; `sigma_diag` was not tuned.

## 5. Frozen nearest packs

`evaluation/task6y_pack_manifest.json` (seed 20260930, deterministic stable-hash selection, train/val v0.2
only):

| Pack | Records | Split | Unique tiles |
|---|---|---|---|
| Y-Overfit20 | 20 (10 `largest_to_nearest` + 10 `smallest_to_nearest`) | train | **20** (≥ 15 required) |
| Y-MiniTrain1000 | 1000 (650 + 350) | train | 786 |
| Y-MiniVal240 | 240 (120 + 120) | val | 221 |
| Y-PairedVal | **N_pair = 20** (40 records) | val | 20 |

Pair requirements verified: same tile, different reference ids, different target ids, both canonical. Only
the two nearest programs appear in every pack, with no directional, L3 or test record.

## 6. Variants and training

`buildreasonseg_mvp/task6y_nearest_decoder.py`: frozen SAM2 `V = 256×64×64`; visual projection
`Conv1x1(256→128) + GroupNorm(8,128) + GELU`; nearest embedding vocab 1 / dim 16 broadcast 64×64; target
trunk `Conv3x3(in→128)+GN+GELU → Conv3x3(128→64)+GN+GELU → Conv1x1(64→1)` upsampled bilinearly to 512;
loss exactly `BCEWithLogitsLoss + DiceLoss` (frozen `task6n_loss`), no GRCL, no ranking/auxiliary loss.

| Variant | Inputs | Fusion | Params |
|---|---|---|---|
| Y-B0 | visual_128 + nearest_embed_16 | 144 | 273,425 |
| Y-B1 | visual_128 + M_ref_64 + nearest_embed_16 | 145 | 274,577 |
| **Y-B2** | visual_128 + P_near_64 + nearest_embed_16 | 145 | 274,577 |
| Y-B3 | field_128 + nearest_embed_16 (no SAM2 visual) | 144 | 240,785 |

Y1 Overfit20 (AdamW lr 1e-3, wd 1e-4, batch 4, 1200 steps, eval every 100, bf16 AMP, seed 20260930):
B0 **0.9724**/0.9860, B1 0.9705/0.9849, **B2 0.9663/0.9827** (gate ≥ 0.85/0.90 ✓ → `NEAREST_OVERFIT_PASS`),
B3 0.1457/0.2066. Y2 then ran (lr 3e-4, batch 8, ≤25 epochs, patience 5, selection by MiniVal240 mIoU).

## 7. MiniVal240 and PairedVal (Parts J)

| Variant | mIoU | Dice | Pr@0.5 | largest-ref | smallest-ref | best epoch | wall | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| Y-B0 | 0.1837 | 0.2735 | 0.2625 | 0.1374 | 0.2301 | 4 | 135.2 s | 0.902 GB |
| Y-B1 | 0.2640 | 0.3643 | 0.3525 | 0.2273 | 0.3008 | 6 | 30.4 s | 0.690 GB |
| **Y-B2** | **0.2864** | **0.3863** | **0.4812** | **0.2336** | **0.3393** | 9 | 38.6 s | 0.690 GB |
| Y-B3 | 0.0468 | 0.0822 | 0.0713 | 0.0235 | 0.0701 | 8 | 35.6 s | 0.674 GB |

Deltas: **B2−B0 = +0.1027**, **B2−B1 = +0.0224**, **B2−B3 = +0.2396**, B1−B0 = +0.0803.

Border-target and tiny-target mIoU are **not applicable**: the frozen nearest eligibility rejects
border-truncated and tiny components, so the nearest packs contain 0 such targets (recorded, not omitted).

B2 quartiles — target area (edges 1038.0 / 1636.5 / 2316.75 px): q1 0.2497, q2 0.3904, q3 0.2859, q4
0.2198; canonical target boundary distance (edges 8.35 / 25.66 / 82.06 px): q1 **0.4094**, q2 0.3477, q3
0.2356, q4 **0.1531** — performance decays monotonically with distance, as the field design implies.

PairedVal (N_pair 20): B0 **0/20** (+0.0000), B1 **8/20** (+0.1840), **B2 6/20** (**+0.1828**), B3 6/20
(+0.0363).

## 8. Criteria and verdict (Parts K-L)

| # | Criterion | Required | Measured | Pass |
|---|---|---|---|---|
| 1 | field semantic sanity | top-1 ≥ 0.85, top-3 ≥ 0.97, Spearman ≥ 0.90 | 1.0000 / 1.0000 / 1.0000 | ✓ |
| 2 | B2 overfit | mIoU ≥ 0.85, Dice ≥ 0.90 | 0.9663 / 0.9827 | ✓ |
| 3 | B2 − B0 MiniVal mIoU | ≥ +0.08 | **+0.1027** | ✓ |
| 4 | B2 − B1 MiniVal mIoU | ≥ +0.05 | **+0.0224** | ✗ |
| 5 | B2 − B3 MiniVal mIoU | ≥ +0.10 | **+0.2396** | ✓ |
| 6 | B2 MiniVal mIoU | ≥ 0.35 | **0.2864** | ✗ |
| 7 | B2 paired pass rate | ≥ 0.70 | **0.30** (6/20) | ✗ |
| 8 | B2 own-cross margin | ≥ 0.15 | **+0.1828** | ✓ |
| 9 | no target GT model input | — | GT target = label/eval only | ✓ |

Section 26 priority: protocol clean, SciPy available, `N_pair = 20 ≥ 12`, sanity passed, learnability
passed; the geometry-only confound does **not** apply (B2−B3 = +0.2396 ≥ 0.10 while B3's paired rate is
0.30 < 0.60) → **`NEAREST_FIELD_NO_MEANINGFUL_GAIN`** (learnability passes but B2−B1 = +0.0224 < +0.05
and B2 mIoU 0.2864 < 0.35).

Measured reading (reported, not prescribed): the oracle-reference field is *semantically exact* and clearly
adds signal over the visual baseline (B2−B0 = +0.1027) and over geometry alone (B2−B3 = +0.2396), and its
advantage concentrates at small boundary distances (q1 0.4094 vs q4 0.1531); but it does not beat the raw
reference-mask control by the required +0.05 (B2−B1 = +0.0224), absolute MiniVal quality stays below 0.35,
and the paired counterfactual pass rate is only 6/20 — so the nearest relation is **not** demonstrated as a
feasible dense-segmentation target from this oracle-reference causal setup alone.

## 9. Interpretation boundary (Part M)

DSH reports measurements only. No global novelty claim, no end-to-end nearest capability claim; `sigma_diag`
unchanged; no learned distance transform; no predicted reference; no combination of directional and nearest
fields; no L3; the reference resolver was not changed. Final recommendation exactly:

`等待 ChatGPT 根据 Task 6Y 的 nearest boundary field 因果结果决定下一步，不自行进行 predicted-reference nearest 集成、direction+nearest 场组合或 L3 训练。`

## 10. Reproduce

```text
python scripts/task6y_freeze_packs.py      # Part E packs + manifest
python scripts/task6y_field_sanity.py      # Part D sanity gate
python scripts/task6y_train.py --stage o1  # Part H Overfit20 (+ B2 gate)
python scripts/task6y_train.py --stage o2  # Part I MiniTrain1000 -> MiniVal240
python scripts/task6y_evaluate.py          # Part J MiniVal240 + PairedVal
python scripts/task6y_report.py            # Parts K-L criteria + verdict
```

Packs and checkpoints live under the gitignored `artifacts/task6y/` and `artifacts/checkpoints/task6y/`; the
frozen SAM2 feature cache is reused from `artifacts/task6n/features`. Run the training/evaluation steps in
`.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`; the pack/sanity/report steps also run in
`.conda/buildreasonseg-mvp`.
