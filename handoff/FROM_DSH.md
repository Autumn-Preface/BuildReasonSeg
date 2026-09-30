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

# FROM_DSH — Task 6X Report: Frozen SAM2.1 Proposal Refinement Audit

_This file holds the Task 6X report. The Task 6W report is preserved in git history at commit `3de2142`;
Task 6V at `3e185fb`; Task 6U at `b8b5224`; Task 6T at `e63f8c4`; Task 6S at `dc8544f`._

**Note on the legacy block above:** those `ARTIFACT-FACTS` numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task6x_sam2_proposal_refinement.md`.

## 1. Verdict

**`SAM2_REFINEMENT_NOT_HELPFUL`** — section 13 applied literally, and the **STOP rule triggered**:

1. `INVALID_EXPERIMENT` — no: frozen modules unchanged, exactly four options, no fifth option, no test use,
   no banned model (ranker / quality estimator), no threshold.
2. `FROZEN_ASSET_UNAVAILABLE` — no: SAM2.1 and U-C1 hashes verified exact.
3. **`SAM2_REFINEMENT_NOT_HELPFUL`** — **X-C0 (no refinement) wins the train-only U-Calib200 ranking**
   `['X-C0', 'X-C1', 'X-C2', 'X-C3']`; the best refinement option X-C1 reaches overall mIoU 0.4744 vs
   baseline 0.4789 (−0.0045), X-C2 −0.0505 and X-C3 −0.1048. ← **verdict**, and the task **stops here**.
4. `SAM2_REFINEMENT_PARTIAL` / `SAM2_REFINEMENT_USEFUL` — not applicable.

**No model was trained. Because the baseline won, RefValUnique, MiniVal240 and PairedVal20 were never
evaluated and the corresponding Task 6X artifacts were intentionally not produced.**

## 2. Recorded Task 6W findings (Task 6W artifacts not mutated)

Verdict `QUALITY_FILTER_NOT_HELPFUL`. W0 oracle-quality mechanism on train-only U-Calib200: deterministic
U-C1 reference mIoU `0.4789276` → oracle `q ≥ 0.50` filter mIoU `0.6575339`; smallest
`0.3544718 → 0.6707174`; selection wrong `76 → 23`; abstention `0` — so *"remove invalid proposals before
extreme-area selection"* is a valid mechanism. The learned estimator reached holdout AUROC `0.8438013` and
F1@0.50 `0.8251182`, but RefVal reference mIoU `0.3793127` (below W-S0 `0.4289355`), selection wrong `65`
(worse than W-S0 `53`) and downstream answered mIoU `0.2745576` (below W-S0 `0.3141364`); it is therefore
not part of the primary resolver.

## 3. Frozen assets and the four predeclared options

`evaluation/task6x_frozen_asset_audit.json` = **`FROZEN_ASSETS_VERIFIED`**:

* SAM2.1 Hiera Base+ `local_cache/models/sam2.1_hiera_base_plus.pt`, SHA256
  `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` **exact**, official
  `sam2.sam2_image_predictor.SAM2ImagePredictor`, `trainable_parameters = 0`, not retrained; a box-only
  probe `[10,10,200,200]` returns masks `(3, 512, 512)` with scores `[0.8870, 0.4559, 0.8629]` and
  `point_coords = point_labels = mask_input = None`, `return_logits = False`;
* U-C1 proposals `imgsz 640 / conf 0.05 / max_det 300 / default NMS / no TTA / no tiling`, YOLO SHA256
  `ef852b58…61f474` exact, not retrained;
* Task 6T hardened ProgramHead, Task 6O B3 and GeometricRelationField v0.2 hashes recorded and unchanged;
  ProposalSetRanker v0.1 and ProposalQualityEstimator v0.1 are explicitly excluded from this resolver.

Exactly four options: **X-C0** no refinement (U-C1 baseline), **X-C1** exact box + single-mask SAM2,
**X-C2** exact box + multimask (max predicted quality), **X-C3** 10 %-expanded box (clipped to [0,511]) +
multimask. The SAM2 predicted quality score is recorded and used **only** as a selection tie-break; it is
never thresholded.

## 4. U-Calib200 calibration (Parts D-E)

`evaluation/task6x_calibration_refinement.json` — 200 references over 195 tiles, train-only:

| Option | overall mIoU | smallest mIoU | largest mIoU | Pr@0.5 | centroid median | abstain | SAM2 calls/tile | wall time/tile |
|---|---|---|---|---|---|---|---|---|
| **X-C0** | **0.4789** | **0.3545** | **0.6034** | **0.5800** | 0.0078 | 0.0000 | 0.00 | ~0 s |
| X-C1 | 0.4744 | 0.3502 | 0.5986 | 0.5700 | **0.0061** | 0.0000 | 9.67 | 0.172 s |
| X-C2 | 0.4284 | 0.3277 | 0.5291 | 0.5250 | 0.0075 | 0.0000 | 9.67 | 0.126 s |
| X-C3 | 0.3741 | 0.3008 | 0.4481 | 0.4523 | 0.0098 | 0.0050 | 9.67 | 0.134 s |

No empty refined masks anywhere; X-C1/X-C2 abstain never, X-C3 once. Baseline buckets
`REFERENCE_SELECTION_WRONG` 84 / `REFERENCE_OK` 116 (the train-side split has no coverage failures).

Section 12 priority → ranking **`['X-C0', 'X-C1', 'X-C2', 'X-C3']`** →
`evaluation/task6x_frozen_refinement_option.json` freezes **X-C0** with `baseline_is_selected = true`,
`refinement_is_selected = false`, `fourth_option = null`, `immutable_after_creation = true` and
`chosen_without_refval = true`. Selection inputs: U-Calib200 only; RefValUnique / MiniVal240 / PairedVal20 /
test all `false`.

## 5. STOP rule — no downstream evaluation

`evaluation/task6x_verdict.json` records `stop_rule.triggered = true`,
`stop_rule.refval_minival_paired_evaluated = false` and `stop_rule.forbidden_files_absent = true`. The four
files that a non-stop path would have produced (`task6x_refval_refinement.json`,
`task6x_downstream_minival240.json`, `task6x_downstream_pairedval20.json`,
`task6x_hardened_parser_integration.json`) **do not exist**, exactly as section 13.1 requires.

Measured context (reported, not interpreted as a remedy): box-prompt refinement slightly improves the
*median centroid error* (X-C1 0.0078 → 0.0061, i.e. marginally better-centred masks) yet loses overall IoU,
because the refined **areas** shift the literal largest/smallest selection; multi-mask selection (X-C2) and
box expansion (X-C3) lose substantially more, consistent with SAM2 preferring object-like regions that are
not the literal area extremes.

## 6. Tests, storage, git

`python -m pytest tests/ -q` → **965 passed, 1 skipped** (Task 6W ended at 929 passed / 1 skipped; no prior
passing test was reduced — the single skip is still the Ultralytics-only eval-mode determinism check that
needs the proposal env). `tests/test_task6x_sam2_refinement.py` adds the 36 section-M checks, including the
STOP-path invariants (no downstream files, no scoring threshold, no training, four options only).

Not committed: SAM2/YOLO/B3/parser/ranker/quality checkpoints, proposal/feature caches, source
imagery/vectors, `.conda`. Committed: refiner code, small JSON artifacts, scripts, tests, docs, handoff.
**No new checkpoint or refinement cache file was created in Task 6X.**

Task 6X downloaded nothing and installed nothing; the frozen SAM2.1 checkpoint was loaded from local disk.
Watt was **not needed** in Task 6X: the pre-existing Watt instance is transport-only, is not owned by this
project and was left running per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 7. Interpretation boundary

DSH reports measurements only. The refiner is support infrastructure, **not** a claimed novelty; no SAM
quality threshold, no point or mask prompt, no box-expansion change, no TTA/tiling/super-resolution, no
U-C1 change, no YOLO/SAM2/ProgramHead/B3 retraining, no ranker or quality estimator in the resolver, no
nearest/L3 execution and no Task 6Y selection.

## 8. Recommended next step

等待 ChatGPT 根据 Task 6X 的 U-Calib200 refinement 结果决定 reference 子系统是否冻结并转向 nearest/L3，不自行启动 nearest/L3 或新增 reference 模块。

## 9. STOP

Task 6X stops here: no more reference modules, no SAM quality-score tuning, no SAM2/YOLO retraining, no U-C1
change, no field/B3 change, no GRCL revisit, no nearest/L3 without the ChatGPT audit, no test access, no
GUI. Waiting for the ChatGPT audit.
