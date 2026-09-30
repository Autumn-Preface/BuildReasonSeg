# Task 6W — Proposal-Quality Filtering + Semantic Extreme Selection

> Task: `handoff/TO_DSH.md` (Task 6W) · Base commit: `3e185fb` · Predecessor: Task 6V →
> `FAMILY_POLICY_NOT_BETTER`
> **Verdict: `QUALITY_FILTER_NOT_HELPFUL`**
> W0 oracle mechanism gate **PASSED**; estimator adequacy **PASSED** (AUROC 0.8438, F1 0.8251); the
> learned filter nevertheless **degrades** every reference and downstream metric
> Tests: `tests/test_task6w_proposal_quality.py` · Evidence: `evaluation/task6w_*.json`

Task 6W tests a specific hypothesis: *many selection errors are caused by fragmented / merged / incomplete
proposals whose predicted area corrupts the literal largest/smallest rule; if low-quality proposals can be
filtered first, deterministic largest/smallest semantics may become reliable again.* The proposal-quality
module is **support infrastructure, not a claimed algorithmic novelty**; the core method
(`ProgramHead → Reference → GeometricRelationField v0.2 → frozen SAM2 → B3`) is frozen.

## 1. Recorded Task 6V result (Task 6V artifacts not mutated)

Verdict `FAMILY_POLICY_NOT_BETTER`; frozen family policy `largest → V-P2` (U-C1 + frozen ranker),
`smallest → V-P0` (U-C0 + deterministic). RefValUnique: family-policy mIoU 0.4383028 vs U-S1 0.4289355
(Δ +0.0093674 < +0.015), `REFERENCE_OK` 114 < 117, `REFERENCE_SELECTION_WRONG` 41, `NOT_COVERED` 59;
downstream answered mIoU 0.3069046, strict 0.3005107, paired 10/20, margin +0.287855, reference-fail 116.
Interpretation boundary: family routing slightly improves some reference metrics; it does not solve the
dominant reference bottleneck; U-C1 remains valuable for candidate coverage; ProposalSetRanker is not
accepted as a global selector.

## 2. Frozen assets

Frozen and hash-verified: Task 6U/6V artifacts, U-C1 configuration and proposal caches, the Task 6M.1
YOLO26m-seg checkpoint `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474`, the Task 6T
hardened ProgramHead, Task 6Q family eligibility, the Task 6O B3 checkpoint, GeometricRelationField v0.2,
the frozen SAM2.1 Hiera Base+ feature path, U-Calib200, U-RankerTrain, RefValUnique, MiniVal240, PairedVal20,
WHU-EA-NativeVector v1.0 and BuildSpatialReason v0.2. No test split. U-C1 inference is fixed at
`imgsz 640 / conf 0.05 / max_det 300 / default NMS / no TTA / no tiling` with masks restored to exact
512×512 source coordinates.

## 3. W0 oracle mechanism diagnostic on U-Calib200 (Part C)

`evaluation/task6w_oracle_quality_filter_calib.json`. For every U-C1 proposal on a U-Calib200 tile,
`q_gt = max IoU(proposal_mask, every native GT building instance on the tile)`; the oracle filter keeps
`q_gt >= 0.50`, abstains if nothing remains, then applies the deterministic largest/smallest area rule with
the Task 6Q tie-break. GT is diagnostic metadata only.

| Metric (overall) | U-C1 deterministic baseline | Oracle quality filter |
|---|---|---|
| selected-reference mIoU | 0.4789 | **0.6575** (Δ **+0.1786**) |
| Pr@0.5 | 0.5800 | 0.8450 |
| `REFERENCE_SELECTION_WRONG` | 76 | **23** (≤ 45 ✓) |
| `REFERENCE_OK` | 116 | **169** |
| abstention rate | 0.0000 | 0.0000 ✓ |
| smallest mIoU | 0.3545 | **0.6707** (Δ **+0.3162**) |

**W0 mechanism gate: PASSED** (all four conditions) → estimator training is permitted.

## 4. Proposal-quality dataset (Part D)

`evaluation/task6w_quality_dataset_manifest.json` + gitignored `artifacts/task6w/quality_dataset/`.

* source: U-RankerTrain only, **deduplicated by tile id** (U-C1 run once per unique tile);
* **no U-Calib200 tile and no RefValUnique tile**: the Task 6U split is reference-key based, so 36 tiles
  held both a U-Calib200 and a U-RankerTrain reference; those tiles (and any RefVal tile) are explicitly
  excluded → 548 unique training tiles, overlaps 0/0;
* common structural eligibility only: no border touch and `bbox_extent_ratio <= 0.20`; the smallest-family
  `area >= 150` floor is deliberately **not** applied (quality is family-independent);
* labels: 5,246 proposals → **3,337 positive / 1,909 negative** at `q_gt >= 0.50`;
* 93 `feature_invalid_small` proposals excluded (no positive 64×64 cell), 0 empty rings;
* tile-disjoint split (seed 20260930): **420 train / 128 holdout tiles**, 4,153 / 1,093 proposals,
  tile overlap 0.

## 5. ProposalQualityEstimator v0.1 (Parts E-F)

`buildreasonseg_mvp/task6w_proposal_quality.py` + `evaluation/task6w_quality_training.json`.

* input: the exact 8 geometry/confidence scalars (`confidence`, `log_area`, `area_ratio`,
  `bbox_extent_ratio`, `width_ratio`, `height_ratio`, `fill_ratio`, `abs_log_aspect`) plus the frozen SAM2
  `V ∈ R^(256×64×64)` pooled as `inside_mean` (mask) and `ring_mean`
  (`Ring = clamp(max_pool2d(M64, 3, 1, 1) − M64, 0, 1)`) → 512 visual dims; no family, no relation, no area
  rank, no x/y location;
* network: `Linear(512→64) → LayerNorm(64) → GELU` ‖ `Linear(8→16) → GELU` → `Linear(80→32) → GELU →
  Linear(32→1)`; **35,729 parameters**; no attention/CNN/Transformer/GNN;
* loss `BCEWithLogitsLoss(pos_weight = N_negative / max(1, N_positive))` with train-only
  `pos_weight = 0.5666`; AdamW lr 5e-4 / wd 1e-4 / batch 256 proposals / ≤40 epochs / patience 5 / seed
  20260930 / bf16 AMP; selection by holdout AUROC → F1@0.50 → lower BCE; **no sweep**;
* selected epoch **19**: holdout **AUROC 0.8438**, AUPRC 0.8952, accuracy 0.7630, precision 0.7686,
  recall 0.8907, **F1 0.8251**, confusion `[[223, 184], [75, 611]]`;
* `artifacts/checkpoints/task6w/proposal_quality_v01.pt` (gitignored), threshold **0.50 frozen**.

Section 21 adequacy: AUROC 0.8438 ≥ 0.80 ✓, F1 0.8251 ≥ 0.65 ✓, no train/holdout tile overlap ✓ →
**PASSED**.

## 6. Resolver and RefValUnique diagnostics (Parts G-H)

`buildreasonseg_mvp/task6w_quality_reference_resolver.py`: U-C1 → Task 6Q family eligibility → quality
estimator → keep `quality_prob >= 0.50` → largest max area / smallest min area → Task 6Q tie-break;
abstention reason `no_quality_eligible_proposals` when everything is filtered, **no fallback**, no ranker.

`evaluation/task6w_refval_quality_filter.json` (219 references):

| Metric | W-S0 (U-C1 + deterministic) | **W-SQ (learned filter)** | W-ORACLE (ceiling) |
|---|---|---|---|
| selected-reference mIoU | 0.4289 | **0.3793** | **0.5209** |
| Dice | 0.4933 | 0.4311 | 0.5901 |
| Pr@0.5 | 0.5209 | 0.4619 | **0.6872** |
| centroid mean / median / p90 | 0.1149 / 0.0159 / 0.3803 | 0.1301 / 0.0490 / 0.3963 | 0.0864 / **0.0065** / 0.3251 |
| area-ratio median | 1.0719 | 1.1062 | 1.1086 |
| abstention rate | 0.0183 | 0.0411 | 0.0365 |
| largest mIoU | 0.5183 | 0.4574 | **0.6043** |
| smallest mIoU | 0.3353 | 0.2982 | **0.4366** |

| Bucket | W-S0 | W-SQ | W-ORACLE |
|---|---|---|---|
| `NO_PROPOSALS` | 1 | 1 | 1 |
| `NO_ELIGIBLE_PROPOSALS` | 3 | 3 | 3 |
| `QUALITY_FILTER_ALL_REJECTED` | 0 | **5** | 4 |
| `REFERENCE_NOT_COVERED_IOU50` | 50 | 48 | 46 |
| `REFERENCE_SELECTION_WRONG` | 53 | **65** | **20** |
| `SELECTED_MASK_GEOMETRY_POOR` | 0 | 0 | 0 |
| `REFERENCE_OK` | 112 | 97 | **144** |

`quality_reference_improved = false`: mIoU 0.3793 < W-S0 + 0.04, `REFERENCE_OK` 97 < W-S0 + 8,
`SELECTION_WRONG` 65 > 0.75 × 53 = 39; abstention rate 0.0411 ≤ 0.10 ✓.

## 7. Downstream causal evaluation and integration (Parts I-J)

`evaluation/task6w_downstream_minival240.json` (canonical program ids) and
`evaluation/task6w_downstream_pairedval20.json`:

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
| mean own / cross / margin | 0.3249 / 0.0040 / **+0.3209** | 0.2576 / 0.0095 / +0.2482 |
| reference-abstention pairs | 0 | 1 |

W-S0 reproduces Task 6U U-S1 exactly (strict 0.3089 / answered 0.3141 / abstentions 4 / ref_fail 115 /
paired 11/20 / margin +0.3209), which validates the causal isolation. Largest/smallest, per-direction,
border and tiny breakdowns are in the artifacts.

`evaluation/task6w_hardened_parser_integration.json` — Task 6T hardened ProgramHead → W-SQ → field v0.2 →
SAM2 → B3: parser **240/240**, strict 0.2631, answered-only 0.2746, abstentions 10, reference-fail 132
(identical to causal W-SQ). No nearest/L3 execution.

## 8. Predeclared gates and verdict (Parts K-L)

Section 23 `PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS`: **4 of 12 pass** (W0 gate ✓, estimator adequacy ✓,
parser 240/240 ✓, no test/no GT ✓). Failing: RefVal mIoU 0.3793 < 0.48, centroid median 0.0490 > 0.03,
centroid p90 0.3963 > 0.28, MiniVal answered 0.2746 < 0.33, strict 0.2631 < 0.31, paired 9 < 12, margin
0.2482 < 0.30, reference-fail 132 > 100.

Section 24 priority order: protocol clean, assets available, W0 passed, adequacy passed, but
`quality_reference_improved = false` **and** the downstream hardening gate fails →
**`QUALITY_FILTER_NOT_HELPFUL`**. No threshold was altered after seeing results.

Measured context (reported, not interpreted as a remedy): the oracle ceiling on RefValUnique is 0.5209 with
`REFERENCE_OK` 144 and only 20 selection errors, i.e. **the mechanism itself is sound**; the gap is the
estimator's *generalisation* — its holdout AUROC is 0.844, and on RefValUnique it rejects 5 records
outright and pushes `REFERENCE_SELECTION_WRONG` from 53 to 65, most visibly on the smallest family.

## 9. Interpretation boundary (Part M)

DSH reports measurements only. The ProposalQualityEstimator is **not** claimed as novelty; threshold 0.50
was not altered; no family-specific quality networks, no ranker after filtering, no size estimator, no YOLO
retraining, no TTA/tiling, no U-C1 change, no field/B3 change and no nearest/L3 work was performed.

## 10. Reproduce

```text
python scripts/task6w_oracle_quality_diagnostic.py --device 0   # Part C (W0 gate)
python scripts/task6w_build_quality_dataset.py --device 0       # Part D
python scripts/task6w_train_quality.py --device cuda            # Parts E-F
python scripts/task6w_evaluate_reference.py --device 0          # Parts G-H
python scripts/task6w_evaluate_downstream.py --stage causal     # Part I
python scripts/task6w_evaluate_downstream.py --stage parser     # Part J
python scripts/task6w_report.py                                 # Parts K-L
```

Quality rows and the checkpoint live under the gitignored `artifacts/task6w/`; proposal caches are reused
from `artifacts/task6u/proposals/U-C1/`. Run in `.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1`.
