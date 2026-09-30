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

# FROM_DSH — Task 6W Report: Proposal-Quality Filtering + Semantic Extreme Selection

_This file holds the Task 6W report. The Task 6V report is preserved in git history at commit `3e185fb`;
Task 6U at `b8b5224`; Task 6T at `e63f8c4`; Task 6S at `dc8544f`; Task 6R at `a3d59da`._

**Note on the legacy block above:** those `ARTIFACT-FACTS` numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task6w_proposal_quality_filter.md`.

## 1. Verdict

**`QUALITY_FILTER_NOT_HELPFUL`** — section 24 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: protocol clean (frozen modules unchanged, no test use, no GT in inference,
   no threshold change, no post-hoc gate change).
2. `FROZEN_ASSET_UNAVAILABLE` — no: YOLO26m-seg SHA256 verified exact, U-C1 config unchanged.
3. `QUALITY_FILTER_MECHANISM_INSUFFICIENT` — **no: the W0 oracle gate PASSED all four conditions.**
4. `QUALITY_ESTIMATOR_NOT_LEARNABLE` — **no: adequacy PASSED** (holdout AUROC 0.8438 ≥ 0.80, F1 0.8251 ≥
   0.65, no tile overlap).
5. **`QUALITY_FILTER_NOT_HELPFUL`** — estimator adequacy passes but `quality_reference_improved = false`
   and the section-23 downstream hardening gate fails on 8 of 12 conditions. ← **verdict**
6. `QUALITY_REFERENCE_HARDENING_PARTIAL` — not applicable (no improvement).
7. `PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS` — no.

No threshold was altered after seeing results; the quality threshold stayed frozen at 0.50.

## 2. Recorded Task 6V result (Task 6V artifacts not mutated)

`FAMILY_POLICY_NOT_BETTER`; frozen policy `largest → V-P2` (U-C1 + ranker), `smallest → V-P0` (U-C0 +
deterministic). RefValUnique mIoU 0.4383028 vs U-S1 0.4289355 (Δ +0.0093674 < +0.015),
`REFERENCE_OK` 114 < 117, `REFERENCE_SELECTION_WRONG` 41, `NOT_COVERED` 59; downstream answered mIoU
0.3069046, strict 0.3005107, paired 10/20, margin +0.287855, reference-fail 116. Family routing slightly
improves some reference metrics but does not solve the dominant bottleneck; U-C1 stays valuable for
candidate coverage; ProposalSetRanker is not accepted as a global selector.

## 3. W0 oracle mechanism diagnostic (U-Calib200 only)

`evaluation/task6w_oracle_quality_filter_calib.json` — `q_gt = max IoU(proposal, every native GT building
instance on the tile)`, oracle keeps `q_gt >= 0.50`, then deterministic largest/smallest with the Task 6Q
tie-break. GT is diagnostic metadata only.

| Metric (overall) | U-C1 deterministic baseline | Oracle quality filter | Gate |
|---|---|---|---|
| selected-reference mIoU | 0.4789 | **0.6575** (Δ **+0.1786**) | ≥ +0.08 ✓ |
| smallest mIoU | 0.3545 | **0.6707** (Δ **+0.3162**) | ≥ +0.10 ✓ |
| `REFERENCE_SELECTION_WRONG` | 76 | **23** | ≤ 45 ✓ |
| abstention rate | 0.0000 | 0.0000 | ≤ 0.10 ✓ |
| `REFERENCE_OK` | 116 | **169** | — |

**W0 gate PASSED → estimator training permitted.**

## 4. Quality dataset and estimator

`evaluation/task6w_quality_dataset_manifest.json`, `evaluation/task6w_quality_training.json`.

* source U-RankerTrain only, **deduplicated by tile id**; **no U-Calib200 and no RefValUnique tile** — the
  Task 6U split is reference-key based, so **36 tiles held both a U-Calib200 and a U-RankerTrain
  reference; those tiles were explicitly excluded** (plus 0 RefVal tiles) → **548** training tiles, both
  overlaps **0**;
* common structural eligibility only (no border touch, `bbox_extent_ratio ≤ 0.20`; the smallest
  `area ≥ 150` floor deliberately not applied);
* **5,246** proposals → **3,337 positive / 1,909 negative** at `q_gt ≥ 0.50`; 93 `feature_invalid_small`
  excluded, 0 empty rings; tile-disjoint split **420 train / 128 holdout** tiles (4,153 / 1,093 proposals,
  overlap 0);
* features: exactly 8 geometry/confidence scalars + frozen SAM2 `V ∈ R^(256×64×64)` pooled as inside-mean
  and one-cell-ring-mean (`Ring = clamp(max_pool2d(M64,3,1,1) − M64, 0, 1)`) → 512 visual dims; no family,
  no relation, no area rank, no x/y location;
* network `Linear(512→64) → LayerNorm(64) → GELU` ‖ `Linear(8→16) → GELU` → `Linear(80→32) → GELU →
  Linear(32→1)`, **35,729 parameters**, no attention/CNN/Transformer/GNN;
* `BCEWithLogitsLoss(pos_weight = N_neg/max(1,N_pos))` with **train-only** `pos_weight = 0.5666`; AdamW
  lr 5e-4 / wd 1e-4 / batch 256 / ≤40 epochs / patience 5 / seed 20260930 / AMP; selection by holdout
  AUROC → F1@0.50 → lower BCE; no sweep;
* selected epoch **19**: holdout **AUROC 0.8438**, AUPRC 0.8952, accuracy 0.7630, precision 0.7686,
  recall 0.8907, **F1 0.8251**, confusion `[[223, 184], [75, 611]]`;
* checkpoint `artifacts/checkpoints/task6w/proposal_quality_v01.pt` (local/gitignored).

## 5. RefValUnique diagnostics

| Metric | W-S0 (U-C1 + deterministic) | **W-SQ (learned filter)** | W-ORACLE (ceiling) |
|---|---|---|---|
| selected-reference mIoU | 0.4289 | **0.3793** | **0.5209** |
| Dice | 0.4933 | 0.4311 | 0.5901 |
| Pr@0.5 | 0.5209 | 0.4619 | **0.6872** |
| centroid mean / median / p90 | 0.1149 / 0.0159 / 0.3803 | 0.1301 / 0.0490 / 0.3963 | 0.0864 / **0.0065** / 0.3251 |
| area-ratio median | 1.0719 | 1.1062 | 1.1086 |
| abstention rate | 0.0183 | 0.0411 | 0.0365 |
| largest / smallest mIoU | 0.5183 / 0.3353 | 0.4574 / 0.2982 | **0.6043 / 0.4366** |

| Bucket | W-S0 | W-SQ | W-ORACLE |
|---|---|---|---|
| `NO_PROPOSALS` | 1 | 1 | 1 |
| `NO_ELIGIBLE_PROPOSALS` | 3 | 3 | 3 |
| `QUALITY_FILTER_ALL_REJECTED` | 0 | **5** | 4 |
| `REFERENCE_NOT_COVERED_IOU50` | 50 | 48 | 46 |
| `REFERENCE_SELECTION_WRONG` | 53 | **65** | **20** |
| `SELECTED_MASK_GEOMETRY_POOR` | 0 | 0 | 0 |
| `REFERENCE_OK` | 112 | 97 | **144** |

`quality_reference_improved = false`: mIoU 0.3793 < W-S0 + 0.04; `REFERENCE_OK` 97 < W-S0 + 8;
`SELECTION_WRONG` 65 > 0.75 × 53 = 39; abstention rate 0.0411 ≤ 0.10 ✓.

## 6. Downstream causal evaluation and integration

| Metric | W-S0 | W-SQ |
|---|---|---|
| strict all-240 mIoU | **0.3089** | 0.2631 |
| answered-only mIoU | **0.3141** | 0.2746 |
| abstentions | **4** | 10 |
| reference-fail count | **115** | 132 |
| target-fail-with-reference-ok | 69 | 60 |
| largest / smallest target mIoU | **0.3241 / 0.2937** | 0.2888 / 0.2375 |
| per direction (left/right/above/below) | **0.3583 / 0.2514 / 0.3307 / 0.2952** | 0.3146 / 0.2234 / 0.2619 / 0.2527 |
| border target (n=114) | **0.2896** | 0.2536 |
| tiny target (n=4) | ≈0 | ≈0 |
| PairedVal pass | **11/20** | 9/20 |
| own / cross / margin | 0.3249 / 0.0040 / **+0.3209** | 0.2576 / 0.0095 / +0.2482 |
| reference-abstention pairs | 0 | 1 |

W-S0 reproduces Task 6U U-S1 exactly (strict 0.3089 / answered 0.3141 / abstentions 4 / ref_fail 115 /
paired 11/20 / margin +0.3209), validating the causal isolation.

`evaluation/task6w_hardened_parser_integration.json` — Task 6T ProgramHead → W-SQ → field v0.2 → SAM2 →
B3: parser **240/240**, strict 0.2631, answered-only 0.2746, abstentions 10, reference-fail 132 (identical
to causal W-SQ). No nearest/L3 execution.

## 7. Gates and verdict

Section 21 estimator adequacy: **PASSED** (AUROC 0.8438 ≥ 0.80, F1 0.8251 ≥ 0.65, no tile overlap).
Section 22 `quality_reference_improved`: **false**.
Section 23 `PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS`: **4 of 12 pass** (W0 gate ✓, adequacy ✓, parser
240/240 ✓, no test/no GT ✓); failing: RefVal mIoU 0.3793 < 0.48, centroid median 0.0490 > 0.03, centroid
p90 0.3963 > 0.28, MiniVal answered 0.2746 < 0.33, strict 0.2631 < 0.31, paired 9 < 12, margin 0.2482 <
0.30, reference-fail 132 > 100.

Section 24 → **`QUALITY_FILTER_NOT_HELPFUL`**.

Measured context (reported, not interpreted as a remedy): the W0 oracle gate proves **the mechanism itself
is sound** (RefVal oracle ceiling mIoU 0.5209 with `REFERENCE_OK` 144 and only 20 selection errors), but
the learned estimator does not transfer well enough: with holdout AUROC 0.844 it rejects 5 RefVal records
outright, pushes `REFERENCE_SELECTION_WRONG` from 53 to 65 and raises abstentions from 4 to 10 downstream.

## 8. Tests, storage, git

`python -m pytest tests/ -q` → **929 passed, 1 skipped** (Task 6V ended at 881 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task6w_proposal_quality.py` adds the 48 section-O checks.

Not committed: the quality checkpoint, YOLO/SAM2/B3/parser/ranker weights, proposal/feature caches, the
generated quality rows (`artifacts/task6w/quality_dataset/`), source imagery/vectors, `.conda`. Committed:
quality model/resolver code, dataset manifest, small JSON eval artifacts, scripts, tests, docs, handoff.

Task 6W downloaded nothing and installed nothing; every frozen asset was loaded from local disk. Watt was
**not needed** in Task 6W: the pre-existing Watt instance is transport-only, is not owned by this project
and was left running per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 9. Interpretation boundary

DSH reports measurements only. The ProposalQualityEstimator is **not** claimed as a novelty; threshold 0.50
was not altered; no family-specific quality network, no ranker after filtering, no size estimator, no YOLO
retraining, no TTA/tiling, no U-C1 change, no field/B3 change and no nearest/L3 work was performed; Task 6X
was not chosen.

## 10. Recommended next step

等待 ChatGPT 根据 Task 6W 的 oracle-quality ceiling、learned quality filter 与 downstream 结果决定 reference 是否继续硬化，不自行增加 ranker、size estimator 或 detector 改动。

## 11. STOP

Task 6W stops here: no further reference model, no ranker after quality filtering, no size correction, no
detector/config/threshold change, no YOLO retraining, no field/B3/SAM2 change, no GRCL revisit, no
nearest/L3, no test access, no GUI. Waiting for the ChatGPT audit.
