# Task 6U — Reference Candidate Coverage + Proposal-Set Ranker Hardening

> Task: `handoff/TO_DSH.md` (Task 6U) · Base commit: `e63f8c4` · Predecessor: Task 6T →
> `PARSER_SEMANTIC_CONTRAST_FAIL`
> **Verdict: `REFERENCE_RANKER_NOT_HELPFUL`** · Selected proposal config: **U-C1** (frozen)
> Tests: `tests/test_task6u_reference_hardening.py` · Evidence: `evaluation/task6u_*.json`

Task 6U isolates the reference-side bottleneck that Task 6S measured (`REFERENCE` 117 vs target-decoder
67) into two subproblems: candidate-proposal **coverage** and extreme-instance **selection**. The core
method — GeometricRelationField v0.2, frozen SAM2 features and the frozen Task 6O B3 target decoder — is
untouched; the hardened ProgramHead, the YOLO26m-seg weights and the Task 6Q eligibility rules are frozen.

## 1. Recorded Task 6T audit notes and erratum

The Task 6T hardened ProgramHead is accepted as the current **directional-chain parser** (full v0.2 val
1.0000, MiniVal240 240/240, PairedVal members 40/40, fixed24 24/24, stress 0.99375 accuracy/macro-F1),
while its formal verdict remains `PARSER_SEMANTIC_CONTRAST_FAIL`: one `largest_to_nearest` stress/minimal
case and the two compact `direction + nearest` controls still fail. Task 6T artifacts and its verdict were
**not** modified.

**Reporting erratum (recorded, not mutated):** `handoff/FROM_DSH.md` states Task 6T peak VRAM as
`0.67 GB`, while the authoritative `evaluation/task6t_training_summary.json` records C1 peak VRAM as
**7.29 GB**. The training-summary value is authoritative; it has no effect on parser metrics.

## 2. Frozen assets and the train-only calibration split

Frozen and verified: Task 6M.1 YOLO26m-seg checkpoint
`ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` (exact, no retraining), the Task 6Q
resolver implementation, the Task 6O B3 checkpoint, field v0.2, the SAM2 feature path, WHU-EA-NativeVector
v1.0, BuildSpatialReason v0.2 and the Task 3B spatial config. No test-split access.

`evaluation/task6u_reference_train_split.json` — from the frozen Task 6P `RefTrainUnique` (825 unique
references), keyed by `(split, tile_id, reference_source_feature_id, reference_family)` and split
deterministically by `sha256("20260930:" + key)`:

| Group | Count | Families | Tiles |
|---|---|---|---|
| **U-Calib200** (config selection only) | **200** | 100 largest / 100 smallest | 195 |
| **U-RankerTrain** (ranker training only) | **625** | 391 largest / 234 smallest | — |
| RefValUnique (evaluation only) | 219 | 110 largest / 109 smallest | 212 |

Zero key overlap between the two groups, zero key overlap with RefValUnique, no test split.

## 3. Four declared proposal configurations and the frozen selection (Parts D-E)

`evaluation/task6u_calibration_candidate_coverage.json` — all four declared configurations of the same
frozen weights on U-Calib200, with the unchanged Task 6Q eligibility:

| Config | imgsz | conf | max_det | smallest eligible@0.50 | overall eligible@0.50 | largest eligible@0.50 | eligible/record |
|---|---|---|---|---|---|---|---|
| U-C0 | 640 | 0.10 | 100 | 0.9100 | 0.9450 | 0.9800 | 6.27 |
| **U-C1** | 640 | **0.05** | **300** | **0.9400** | **0.9600** | **0.9800** | 7.53 |
| U-C2 | 1024 | 0.10 | 300 | 0.8800 | 0.9200 | 0.9600 | 6.22 |
| U-C3 | 1024 | 0.05 | 300 | 0.9000 | 0.9300 | 0.9600 | 7.70 |

Section 10 priority → ranking `['U-C1', 'U-C0', 'U-C3', 'U-C2']` → **U-C1 frozen**
(`evaluation/task6u_selected_proposal_config.json`). RefValUnique and MiniVal240 never participated in
the choice. Source images remain the original 512×512 tiles, masks are restored to exact source
coordinates, and no TTA/tiling/super-resolution is used anywhere.

## 4. RefValUnique candidate audit (Part F)

`evaluation/task6u_refval_candidate_audit.json` — U-C0 versus the frozen U-C1 on the untouched 219-reference
RefValUnique:

| Metric (eligible@0.50) | U-C0 (baseline) | **U-C1 (selected)** | Δ |
|---|---|---|---|
| overall | 0.6895 | **0.7534** | **+0.0639** |
| largest | 0.8364 | **0.8727** | +0.0364 |
| smallest | 0.5413 | **0.6330** | **+0.0917** |

→ `candidate_coverage_improved = true` (requires ≥ +0.04 overall and ≥ +0.06 smallest).

**Oracle-selection ceiling** (diagnostic only; GT picks the best eligible proposal, never used in
inference): with U-C1, best-eligible mIoU **0.6203**, Dice 0.7092, Pr@0.5 0.7674, centroid median 0.0046,
p90 0.108 — i.e. the *candidate sets* contain a much better reference than either selector extracts.

## 5. ProposalSetRanker v0.1 (Part G)

`buildreasonseg_mvp/task6u_reference_ranker.py` + `evaluation/task6u_ranker_training.json`. Support
infrastructure addressing `REFERENCE_SELECTION_WRONG` only; it cannot create missing proposals and is
**not** a claimed novelty.

* exact 14-d feature vector (12 scalars + 2-d family one-hot) — no GT feature, no relation, no centroid,
  no image location, no SAM2 feature, no target mask, no source feature id;
* architecture `Linear(14→32) → GELU → Linear(32→16) → GELU → Linear(16→1)` shared per proposal, softmax
  over the set, cross-entropy on the labelled positive; **1,025 parameters**, no attention/Transformer/GNN;
* training examples from U-RankerTrain under the frozen U-C1: 625 records → **598 trainable**, **27
  `untrainable_not_covered`** (best eligible IoU < 0.50, excluded from the loss), 0 no-eligible;
* internal 90/10 by key hash (538/60), AdamW lr 1e-3 / wd 1e-4 / batch 64 sets / ≤50 epochs / patience 6;
  selected epoch **2**, internal-holdout **top-1 accuracy 0.45**, mean selected IoU 0.4510 (train split
  at the end: 0.4368 / 0.4229);
* checkpoint `artifacts/checkpoints/task6u/reference_ranker_v01.pt` (local/gitignored,
  SHA256 `c738fcf77419626f…`), 28.6 s wall.

## 6. Three reference selectors on RefValUnique (Part H)

`evaluation/task6u_refval_selector_comparison.json`.

| Metric | U-S0 (C0 + deterministic) | **U-S1 (C1 + deterministic)** | U-S2 (C1 + ranker) |
|---|---|---|---|
| selected-reference mIoU | 0.4248 | **0.4289** | 0.3107 |
| Dice | 0.4832 | **0.4933** | 0.3564 |
| Pr@0.5 | 0.5258 | 0.5209 | 0.3581 |
| centroid error median | 0.0156 | 0.0159 | 0.0890 |
| centroid error p90 | 0.3935 | 0.3803 | 0.3771 |
| area-ratio median | 1.0813 | 1.0719 | 1.4695 |
| abstention rate | 0.0274 | **0.0183** | 0.0183 |
| largest mIoU / Pr@0.5 | 0.5428 / 0.6514 | 0.5183 / 0.6000 | 0.5680 / 0.6636 |
| smallest mIoU / Pr@0.5 | 0.3011 / 0.3942 | **0.3353 / 0.4381** | 0.0411 / 0.0381 |

Failure buckets (Task 6Q definitions):

| Bucket | U-S0 | U-S1 | U-S2 |
|---|---|---|---|
| `NO_PROPOSALS` | 3 | 1 | 1 |
| `NO_ELIGIBLE_PROPOSALS` | 3 | 3 | 3 |
| `REFERENCE_NOT_COVERED_IOU50` | 62 | **50** | **50** |
| `REFERENCE_SELECTION_WRONG` | 39 | 53 | **88** |
| `SELECTED_MASK_GEOMETRY_POOR` | 1 | 0 | 0 |
| `REFERENCE_OK` | 111 | **112** | 77 |

Candidate hardening removes 14 uncovered records and 2 abstentions but *raises* `SELECTION_WRONG` from 39
to 53 (more, harder candidates for the fixed area rule). The learned ranker then degrades selection
further (88) and never exploits the 0.6203 oracle ceiling → `ranker_improved_selection = false`
(Δ mIoU −0.118 versus the required +0.05; `SELECTION_WRONG` ratio 1.66 versus the required ≤0.70;
abstention rate 0.0183 ≤ 0.10 ✓).

## 7. Downstream causal evaluation (Parts I-J)

`evaluation/task6u_downstream_minival240.json` — canonical program ids (no parser), only the
proposal/reference stage differs:

| Metric | U-S0 | **U-S1** | U-S2 |
|---|---|---|---|
| strict all-240 mIoU | 0.2970 | **0.3089** | 0.2425 |
| answered-only mIoU (236/236/236) | 0.3046 | **0.3141** | 0.2466 |
| abstentions | 6 | **4** | 4 |
| reference-fail count | 117 | **115** | 158 |
| parser bucket | not applicable | not applicable | not applicable |
| border target (n=114) | 0.2943 | 0.2896 | 0.2666 |
| tiny target (n=4) | ≈0 | ≈0 | ≈0 |

`evaluation/task6u_downstream_pairedval20.json`:

| Metric | U-S0 | U-S1 | U-S2 |
|---|---|---|---|
| pass | 10/20 | **11/20** | 11/20 |
| mean own IoU | 0.277946 | 0.324870 | 0.336264 |
| mean cross IoU | 0.004245 | 0.003988 | 0.003089 |
| own−cross margin | +0.2737 | **+0.3209** | +0.3332 |
| reference-abstention pairs | 0 | 0 | 0 |

(the margin gain of U-S2 comes with a much weaker own IoU absolute level; see the reference table above.)

**U-S0 reproduces Task 6S exactly** (strict 0.2969667, answered 0.3045813, abstentions 6, reference-fail
117, paired 10/20, margin +0.2737): the causal isolation is validated. Candidate hardening improves the
downstream chain modestly; the ranker degrades it.

`evaluation/task6u_hardened_parser_integration.json` — Task 6T hardened ProgramHead + U-S2 + frozen
field/B3 on MiniVal240 (integration check only): parser **240/240**, strict mIoU 0.2425, answered-only
0.2466, abstentions 4 — identical to the causal U-S2, which also confirms the hardened parser returns the
canonical program ids on all 240 records. No nearest/L3 execution was evaluated.

## 8. Predeclared flags, gates and verdict

* `candidate_coverage_improved` = **true** (overall +0.0639 ≥ +0.04; smallest +0.0917 ≥ +0.06).
* `ranker_improved_selection` = **false** (mIoU Δ −0.118 < +0.05; `SELECTION_WRONG` ratio 1.66 > 0.70).
* Section 24 U-S2 downstream hardening gates: **3 of 10 pass** (own-cross margin 0.3332 ≥ 0.30 ✓, no test
  split ✓, no GT in inference ✓). Failing: RefVal mIoU 0.3107 < 0.48; centroid median 0.0890 > 0.03;
  centroid p90 0.3771 > 0.25; MiniVal answered 0.2466 < 0.33; strict 0.2425 < 0.31; paired 11 < 12;
  reference-fail 158 > 93.

Section 25 priority order applied literally: protocol clean, checkpoint hash exact, coverage **does**
improve, the ranker is not helpful and U-S2 fails the downstream hardening gate →
**`REFERENCE_RANKER_NOT_HELPFUL`**. No threshold was altered.

## 9. Interpretation boundary

DSH reports measurements only. The ProposalSetRanker is **not** claimed as a project novelty; no decision
was taken to retrain YOLO, to switch detector, to keep 1024 permanently, to add tiling/TTA, to alter the
smallest threshold, to start nearest/L3, to retrain B3, to add GRCL, or to choose the next architecture.
The ranker's poor selection performance is reported as measured; DSH does not repair it and does not
prescribe a remedy.

## 10. Reproduce

```text
python scripts/task6u_freeze_reference_split.py            # Part C
python scripts/task6u_candidate_coverage.py --device 0     # Parts D-E (four configs + selection)
python scripts/task6u_train_ranker.py --device 0           # Part G
python scripts/task6u_evaluate_reference.py --device 0     # Parts F + H
python scripts/task6u_evaluate_downstream.py --stage causal   # Parts I
python scripts/task6u_evaluate_downstream.py --stage parser   # Part J
python scripts/task6u_report.py                            # Parts K-L
```

Proposal caches live under the gitignored `artifacts/task6u/proposals/<config>/`; the ranker checkpoint
under the gitignored `artifacts/checkpoints/task6u/`. Run in `.conda/buildreasonseg-proposal` with
`HF_HUB_OFFLINE=1`.
