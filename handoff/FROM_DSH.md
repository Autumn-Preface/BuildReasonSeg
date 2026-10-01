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

# FROM_DSH — Task 7F Report: Reference Bottleneck Ceiling Decomposition for Frozen D-B1

_This file holds the Task 7F report. The Task 7E report is preserved in git history at commit `f9c6e87`;
Task 7D at `86e4f4c`; Task 7C at `632c9c0`; Task 7B at `b325585`; Task 7A at `6ee3d0d`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7f_reference_ceiling_decomposition.md`. **No model was trained in Task 7F.**

## 1. Verdict

**`REFERENCE_SELECTION_DOMINANT`** — section 20 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths unchanged, no training, no test use, no module mutation, GT only in
   the declared diagnostic modes.
2. `TASK7E_HOLDOUT_MISMATCH` — no: record-id hash and pair-id hash re-verified exact (669 records / 20 pairs),
   zero test, zero Task 6Z MiniVal/Paired overlap.
3. `TASK7E_NUMERIC_REPRODUCTION_FAIL` — no: F-R0 reproduces Task 7E predicted D-B1 with Δ **0.0** and F-R3
   reproduces Task 7E oracle D-B1 with Δ **2.27e-07** (tolerance 1e-6).
4. `REFERENCE_PROPOSAL_CEILING_INSUFFICIENT` — no: `CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING = true`
   (F-R2 strict 0.3312 ≥ 0.30, paired 16/20 ≥ 12, margin +0.3306 ≥ 0.22).
5. **`REFERENCE_SELECTION_DOMINANT`** — usable ceiling true, `SELECTION_IS_ACTIONABLE = true`, and
   `selection_gain (+0.119504) ≥ coverage_gain (+0.054316)`. ← **verdict**
6. `REFERENCE_COVERAGE_OR_GEOMETRY_DOMINANT` — not reached.
7. `REFERENCE_MIXED_BOTTLENECK` — not reached.

No threshold was changed after seeing results.

## 2. Frozen assets and population

D-B1 `artifacts/checkpoints/task7d/db1_minitrain1200.pt` SHA256 `6df31909…21a89c0` (not retrained); frozen
U-C1 (YOLO26m-seg Task 6M.1 `ef852b58…61f474`, imgsz 640, conf 0.05, max_det 300, default NMS, no TTA, no
tiling, 512×512 source; eligibility = not border-touching and bbox extent ratio ≤ 0.20, no other filter);
read-only GeometricRelationField v0.2, NearestBoundaryField v0.1, frozen SAM2.1 Hiera Base+ features. No test.

Population reused byte-for-byte: `E-HoldoutL3` **669** records and `E-PairedHoldout` **20** pairs from
`evaluation/task7e_holdout_manifest.json`, with both hashes re-verified, no new sampling.

## 3. Four exact reference modes (one frozen U-C1 proposal pass per tile, reused by every mode)

| Mode | Construction |
|---|---|
| **F-R0** `CURRENT_SELECTED_PRED_MASK` | eligible proposals → max predicted area → tie higher confidence → lower index → that predicted mask (abstain if none) |
| **F-R1** `ORACLE_SELECTED_PRED_MASK` | same eligible proposals → max IoU to the canonical GT reference → tie higher confidence → lower index → that **predicted** mask |
| **F-R2** `COVERAGE_CONDITIONAL_GT_MASK` | best eligible IoU ≥ 0.50 → canonical GT mask, else abstain |
| **F-R3** `FULL_ORACLE_GT_MASK` | canonical GT mask always |

## 4. Reference audit

| Mode | Answered | Abstentions | Ref mIoU | Dice | Pr@0.5 | Centroid median / p90 |
|---|---:|---:|---:|---:|---:|---:|
| F-R0 | 667/669 | 2 | 0.4406 | 0.5085 | 0.5016 | 19.114 / 247.748 |
| **F-R1** | 667/669 | 2 | **0.7119** | **0.8111** | **0.8710** | 4.195 / 25.916 |
| F-R2 | 566/669 | 103 | 1.0000 | 1.0000 | 1.0000 | 0.000 / 0.000 |
| F-R3 | 669/669 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.000 / 0.000 |

F-R0: mean selected confidence 0.3577, mean area 5350.76 px, mean selected-vs-best proposal IoU **0.5591**,
mean reference-IoU gap to the best eligible proposal **0.2713**. F-R1 best-eligible coverage at IoU ≥ 0.25 /
0.50 / 0.75 = **641** (0.9581) / **566** (0.8460) / **392** (0.5860). F-R2 coverage rate **0.8460** at the
frozen 0.50 threshold (566 covered, 103 uncovered).

## 5. Frozen D-B1 downstream (strict all-record mIoU) and reproductions

| Mode | Strict mIoU | Dice | Pr@0.5 | Answered | Answered-only |
|---|---:|---:|---:|---:|---:|
| F-R0 | **0.245405** | 0.324762 | 0.471130 | 667/669 | 0.246141 |
| F-R1 | **0.364909** | 0.484050 | 0.538483 | 667/669 | 0.366004 |
| F-R2 | **0.331180** | 0.436579 | 0.462272 | 566/669 | 0.391447 |
| F-R3 | **0.385496** | 0.510649 | 0.547313 | 669/669 | 0.385496 |

F-R0 Δ **0.0** vs Task 7E predicted D-B1 (`0.24540501038500215`, answered-only `0.24614085749260334`,
abstentions 2, all exact) and F-R3 Δ **2.27e-07** vs Task 7E oracle D-B1 (`0.38549570532647004`) →
`TASK7E_NUMERIC_REPRODUCTION_PASS`.

## 6. Covered-subset geometry, gaps and labels

COVERED50 = 566 records (at least one eligible proposal and best eligible IoU ≥ 0.50 — the non-abstaining
F-R2 population). G-PRED (F-R1 predicted mask) mIoU **0.3872** / Dice 0.5103 / Pr@0.5 0.5432; G-GT (GT mask)
mIoU **0.3914** / Dice 0.5160 / Pr@0.5 0.5464 → `geometry_gain_covered = **+0.004253**`. Paired restriction to
the 17 pairs with a COVERED50 shared reference: F-R1 16/17 (margin +0.3315), F-R3 16/17 (margin +0.3306).

```text
M0 = 0.245405   M1 = 0.364909   M2 = 0.331180   M3 = 0.385496
selection_gain        = M1 - M0 = +0.119504    selection_fraction = 0.8530
coverage_gain         = M3 - M2 = +0.054316    coverage_fraction  = 0.3877
geometry_gain_covered = G-GT - G-PRED = +0.004253   (different subset, reported separately)
total_reference_gap   = M3 - M0 = +0.140091
```

| Label | Measured conditions | Value |
|---|---|---|
| `SELECTION_IS_ACTIONABLE` | +0.1195 ≥ 0.05 ✓ · 0.3649 ≥ 0.29 ✓ · 18/20 ≥ 10 ✓ · +0.3005 ≥ 0.18 ✓ | **true** |
| `PROPOSAL_GEOMETRY_IS_MAJOR` | +0.0043 ≥ 0.05 ✗ | **false** |
| `PROPOSAL_COVERAGE_IS_MAJOR` | +0.0543 ≥ 0.04 ✓ or coverage 0.8460 < 0.85 ✓ | **true** |
| `CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING` | 0.3312 ≥ 0.30 ✓ · 16/20 ≥ 12 ✓ · +0.3306 ≥ 0.22 ✓ | **true** |

Paired ceilings (E-PairedHoldout20, reference built once per pair): F-R0 **6/20** (+0.1251) · **F-R1 18/20**
(+0.3005) · F-R2 **16/20** (17 answered, 3 abstention pairs, +0.3306) · F-R3 **18/20** (+0.3193).

## 7. D-B1 status

All Task 7E oracle holdout facts reproduce (Z-B3 0.3141113773, D-B1 0.3854957053, Δ +0.0713843280,
paired 18/20, margin +0.3193402994), so D-B1 is recorded as the
**`preferred oracle-reference L3 target decoder candidate`**. It is explicitly **not** end-to-end ready, **not**
the final model and **not** paper-final; Z-B3 remains the frozen baseline/ablation and the practical chain stays
blocked until ChatGPT decides what to do with the reference bottleneck.

Measured reading (reported only, no repair proposed): with the frozen U-C1 proposal set fixed, repairing
selection is worth about **2.2×** more strict mIoU than repairing coverage (+0.1195 vs +0.0543 out of a
+0.1401 total gap), and mask geometry refinement is worth almost nothing on already-covered records (+0.0043).
The candidate coverage itself has a usable ceiling (F-R2 0.3312 strict, 16/20 paired, margin +0.3306), and F-R0
agrees with the best eligible candidate at only 0.5591 IoU while leaving a mean 0.2713 reference-IoU unused —
selection, not candidate availability, is the dominant reference bottleneck, with coverage as a secondary
contributor (103/669 records have no eligible proposal at IoU ≥ 0.50).

## 8. Tests, storage, git

`python -m pytest tests/ -q` → **1325 passed, 1 skipped** (Task 7E ended at 1285 passed / 1 skipped; no prior
passing test was reduced). `tests/test_task7f_reference_ceiling.py` adds the 40 Part-O checks.

Not committed: model checkpoints, proposal/feature caches, source imagery/vectors, the per-record measurement
cache (`artifacts/task7f/`), `.conda`. Committed: small evaluation JSON, diagnostic scripts/helper, tests, docs,
handoff. No checkpoint was created.

Task 7F downloaded nothing and installed nothing. Watt was **not needed** in Task 7F: the pre-existing Watt
instance is transport-only, is not owned by this project and was left running per the ownership rule; no proxy,
host, certificate or TLS setting was read or modified.

## 9. Interpretation boundary

DSH reports measurements only. No selector was trained; YOLO was not retrained; U-C1 was not changed; D-B1 was
not retrained; no threshold change is proposed; no full training or test was started; no repair was chosen from
the diagnostic labels; Task 7G was not chosen. The verdict does **not** authorize an automatic repair.

## 10. Recommended next step (exact wording required by Part M)

等待 ChatGPT 根据 Task 7F 的 selection / proposal-geometry / coverage ceiling 分解决定是否值得进行最后一次 reference 干预；不自行训练 selector、重训 YOLO 或开始正式 test。

## 11. STOP

Task 7F stops here: nothing trained, no reference repair chosen or implemented, no test run, no formal full
training, no GUI. Waiting for the ChatGPT audit.
