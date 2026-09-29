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

# FROM_DSH — Task 6U Report: Reference Candidate Coverage + Proposal-Set Ranker

_This file holds the Task 6U report. The Task 6T report is preserved in git history at commit `e63f8c4`;
Task 6S at `dc8544f`; Task 6R at `a3d59da`; Task 6Q at `7c19bec`; Task 6P at `b80f3cc`._

**Note on the legacy block above:** those `ARTIFACT-FACTS` numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task6u_reference_hardening.md`.

## 1. Verdict

**`REFERENCE_RANKER_NOT_HELPFUL`** — section 25 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: no frozen-module mutation, no leakage, no test access, no GT in inference,
   proposal checkpoint hash exact.
2. `PROPOSAL_CHECKPOINT_UNAVAILABLE` — no: SHA256 `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` exact.
3. `REFERENCE_CANDIDATE_COVERAGE_STILL_LIMITING` — no: candidate coverage **does** improve
   (overall eligible@0.50 +0.0639, smallest +0.0917 on RefValUnique).
4. **`REFERENCE_RANKER_NOT_HELPFUL`** — coverage improves but `ranker_improved_selection = false` and U-S2
   fails the section-24 downstream hardening gate. ← **verdict**
5. `REFERENCE_HARDENING_PARTIAL` — not applicable (U-S2 does not improve materially; it degrades).
6. `REFERENCE_HARDENING_FEASIBLE` — no.

## 2. Recorded Task 6T notes and erratum (Task 6T artifacts not mutated)

The Task 6T hardened ProgramHead is accepted as the current directional-chain parser (full v0.2 val
1.0000, MiniVal240 240/240, PairedVal members 40/40, fixed24 24/24, stress 0.99375 accuracy/macro-F1);
its formal verdict remains `PARSER_SEMANTIC_CONTRAST_FAIL` with residual failures on one
`largest_to_nearest` case and the two compact `direction + nearest` controls.

**Reporting erratum:** Task 6T `FROM_DSH.md` states peak VRAM `0.67 GB`; the authoritative
`evaluation/task6t_training_summary.json` records C1 peak VRAM **7.29 GB**. The training-summary value is
authoritative and has no effect on parser metrics.

## 3. Frozen assets and the train-only calibration split

Frozen and verified: Task 6M.1 YOLO26m-seg checkpoint (exact SHA256, not retrained), the Task 6Q resolver
implementation, the Task 6O B3 checkpoint, GeometricRelationField v0.2, the SAM2 feature path, the native
vector dataset v1.0, BuildSpatialReason v0.2 and the Task 3B spatial config. No test-split access.

`evaluation/task6u_reference_train_split.json` — from frozen RefTrainUnique (825 references) by unique
reference key, split deterministically with seed 20260930:

| Group | Count | Families | Purpose |
|---|---|---|---|
| **U-Calib200** | **200** | 100 largest / 100 smallest | proposal-config selection only |
| **U-RankerTrain** | **625** | 391 largest / 234 smallest | ranker training only |
| RefValUnique (untouched) | 219 | 110 largest / 109 smallest | evaluation only |

Zero key overlap between the groups and zero key overlap with RefValUnique; no test split.

## 4. Four declared proposal configurations and the frozen selection (Parts D-E)

`evaluation/task6u_calibration_candidate_coverage.json` on U-Calib200 (unchanged Task 6Q eligibility):

| Config | imgsz | conf | max_det | smallest eligible@0.50 | overall | largest | eligible/record |
|---|---|---|---|---|---|---|---|
| U-C0 | 640 | 0.10 | 100 | 0.9100 | 0.9450 | 0.9800 | 6.27 |
| **U-C1** | 640 | **0.05** | **300** | **0.9400** | **0.9600** | **0.9800** | 7.53 |
| U-C2 | 1024 | 0.10 | 300 | 0.8800 | 0.9200 | 0.9600 | 6.22 |
| U-C3 | 1024 | 0.05 | 300 | 0.9000 | 0.9300 | 0.9600 | 7.70 |

Section 10 priority → ranking `['U-C1', 'U-C0', 'U-C3', 'U-C2']` → **U-C1 frozen** in
`evaluation/task6u_selected_proposal_config.json` (immutable afterwards). RefValUnique and MiniVal240 did
not participate. Source images stay the original 512×512 tiles, masks are restored to exact source
coordinates, and no TTA, tiling or super-resolution is used.

## 5. RefValUnique candidate audit and the oracle ceiling (Part F)

| eligible coverage@0.50 | U-C0 | **U-C1** | Δ |
|---|---|---|---|
| overall | 0.6895 | **0.7534** | **+0.0639** |
| largest | 0.8364 | **0.8727** | +0.0364 |
| smallest | 0.5413 | **0.6330** | **+0.0917** |

→ `candidate_coverage_improved = true` (requires ≥ +0.04 overall and ≥ +0.06 smallest).

Oracle-selection ceiling (diagnostic; GT picks the best eligible proposal, never used in inference) with
U-C1: mIoU **0.6203**, Dice 0.7092, Pr@0.5 0.7674, centroid median 0.0046, p90 0.108 — the candidate sets
contain far better references than either selector extracts.

## 6. ProposalSetRanker v0.1 (Part G)

* 14-d feature vector (12 mandated scalars + 2-d family one-hot); no GT feature, no relation, no centroid,
  no image location, no SAM2 feature, no target mask, no source feature id.
* `Linear(14→32) → GELU → Linear(32→16) → GELU → Linear(16→1)`, shared per proposal, softmax over the set,
  cross-entropy; **1,025 parameters**; no attention/Transformer/GNN.
* Training data: 625 U-RankerTrain references under frozen U-C1 → **598 trainable**, **27
  `untrainable_not_covered`** (best eligible IoU < 0.50 → excluded from the loss), 0 no-eligible; internal
  90/10 (538/60) used only for selection.
* AdamW lr 1e-3 / wd 1e-4 / batch 64 sets / ≤50 epochs / patience 6 / seed 20260930; selected epoch **2**;
  internal-holdout **top-1 accuracy 0.45**, mean selected IoU 0.4510.
* Checkpoint `artifacts/checkpoints/task6u/reference_ranker_v01.pt` (local/gitignored,
  `c738fcf77419626f…`), 28.6 s wall.

## 7. Three selectors on RefValUnique (Part H)

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

| Bucket | U-S0 | U-S1 | U-S2 |
|---|---|---|---|
| `NO_PROPOSALS` | 3 | 1 | 1 |
| `NO_ELIGIBLE_PROPOSALS` | 3 | 3 | 3 |
| `REFERENCE_NOT_COVERED_IOU50` | 62 | **50** | **50** |
| `REFERENCE_SELECTION_WRONG` | 39 | 53 | **88** |
| `SELECTED_MASK_GEOMETRY_POOR` | 1 | 0 | 0 |
| `REFERENCE_OK` | 111 | **112** | 77 |

Candidate hardening removes 14 uncovered records and 2 abstentions but raises `SELECTION_WRONG` from 39 to
53 (more, harder candidates for the fixed area rule); the ranker then degrades it to 88.
`ranker_improved_selection = false`: Δ mIoU **−0.118** (required ≥ +0.05), `SELECTION_WRONG` ratio
**1.66** (required ≤ 0.70), abstention rate 0.0183 ≤ 0.10 ✓.

## 8. Downstream causal evaluation and final integration (Parts I-J)

MiniVal240 with canonical program ids (no parser), so only the reference stage differs:

| Metric | U-S0 | **U-S1** | U-S2 |
|---|---|---|---|
| strict all-240 mIoU | 0.2970 | **0.3089** | 0.2425 |
| answered-only mIoU (236 each) | 0.3046 | **0.3141** | 0.2466 |
| abstentions | 6 | **4** | 4 |
| reference-fail count | 117 | **115** | 158 |
| border target (n=114) | 0.2943 | 0.2896 | 0.2666 |
| tiny target (n=4) | ≈0 | ≈0 | ≈0 |
| parser bucket | not applicable | not applicable | not applicable |

PairedVal20: pass **10/20 → 11/20 → 11/20**; own IoU 0.277946 / 0.324870 / 0.336264; cross IoU 0.004245 /
0.003988 / 0.003089; margin **+0.2737 / +0.3209 / +0.3332**; 0 reference-abstention pairs.

**U-S0 reproduces Task 6S exactly** (strict 0.2969667, answered 0.3045813, abstentions 6, reference-fail
117, paired 10/20, margin +0.2737), which validates the causal isolation.

`evaluation/task6u_hardened_parser_integration.json` — Task 6T hardened ProgramHead + U-S2 + frozen
field/B3 on MiniVal240: parser **240/240**, strict mIoU 0.2425, answered-only 0.2466, abstentions 4
(identical to causal U-S2). No nearest/L3 execution was evaluated.

## 9. Predeclared flags, gates and verdict

* `candidate_coverage_improved` = **true** (overall +0.0639 ≥ +0.04; smallest +0.0917 ≥ +0.06).
* `ranker_improved_selection` = **false**.
* Section 24 U-S2 downstream hardening gates: **3 of 10 pass** (margin 0.3332 ≥ 0.30 ✓, no test split ✓,
  no GT in inference ✓); failing: RefVal mIoU 0.3107 < 0.48, centroid median 0.0890 > 0.03, centroid p90
  0.3771 > 0.25, MiniVal answered 0.2466 < 0.33, strict 0.2425 < 0.31, paired 11 < 12, reference-fail
  158 > 93.

No gate or threshold was altered after seeing results.

## 10. Tests, storage, git

`python -m pytest tests/ -q` → **851 passed, 1 skipped** (Task 6T ended at 807 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task6u_reference_hardening.py` adds the 44 section-O checks.

Not committed: the ranker checkpoint, YOLO/SAM2/B3/parser weights, proposal caches
(`artifacts/task6u/proposals/<config>/`), feature caches, source imagery/vectors, `.conda`, large local
data. Committed: ranker code, small JSON artifacts, scripts, tests, docs, handoff.

Task 6U downloaded nothing and installed nothing; the frozen checkpoint loaded from local disk. Watt was
**not needed** in Task 6U: the pre-existing Watt instance is transport-only, is not owned by this project
and was left running per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 11. Interpretation boundary

DSH reports measurements only: the ProposalSetRanker is not claimed as a project novelty; no decision was
taken to retrain YOLO, switch detector, keep 1024 permanently, add tiling/TTA, alter the smallest
threshold, start nearest/L3, retrain B3, add GRCL, or choose the next architecture. The ranker's poor
selection performance is reported as measured and is not repaired.

## 12. Recommended next step

等待 ChatGPT 根据 Task 6U 的 candidate coverage、ranker 与 downstream 因果结果决定下一步，不自行修改 detector、reference 语义或 target 架构。

## 13. STOP

Task 6U stops here: no YOLO retraining, no tiling/TTA, no threshold changes outside the four declared
configurations, no ranker change, no additional reference model, no B3/field/SAM2 change, no GRCL, no
nearest/L3 target execution, no test access, no GUI. Waiting for the ChatGPT audit.
