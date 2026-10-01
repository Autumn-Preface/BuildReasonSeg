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

# FROM_DSH — Task 7G Report: Largest-Reference Set-Context Selector (Final Reference Intervention)

_This file holds the Task 7G report. The Task 7F report is preserved in git history at commit `c59c6d3`;
Task 7E at `f9c6e87`; Task 7D at `86e4f4c`; Task 7C at `632c9c0`; Task 7B at `b325585`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7g_largest_reference_set_context_selector.md`. Only the selector was trained;
YOLO, SAM2, the fields, the parser and D-B1 stayed frozen.

## 1. Verdict

**`LARGEST_SELECTOR_NOT_LEARNABLE`** — section 34 priority order applied literally:

1. `INVALID_EXPERIMENT` — no: frozen paths unchanged, train-split-only selector data, no test use, no external
   holdout in checkpoint selection, YOLO/D-B1 untouched.
2. `SELECTOR_DEPENDENCY_UNAVAILABLE` — no: SciPy present and used for the boundary feature.
3. `SELECTOR_TRAIN_DATA_INSUFFICIENT` — no: 1,028 trainable unique references ≥ 300 on 1,063 tiles.
4. **`LARGEST_SELECTOR_NOT_LEARNABLE`** — the section-23 internal tile-disjoint gate failed 2 of 4 conditions
   (`mean selected IoU 0.6199 < 0.62`; `mean best-minus-selected gap 0.1791 > 0.14`). ← **verdict**
5. `TASK7F_BASELINE_REPRODUCTION_FAIL` — not reached (external stage not executed).
6. `LARGEST_SELECTOR_SCENE_DISJOINT_FAIL` — not reached.
7. `LARGEST_SELECTOR_DEVELOPMENT_READY` — no.

No threshold, feature, model or loss was changed after seeing results.

## 2. Recorded Task 7F evidence (frozen, not modified)

`REFERENCE_SELECTION_DOMINANT`: M0 0.2454050104, M1 0.3649093893, M2 0.3311796697, M3 0.3854959324;
selection_gain +0.1195043789, coverage_gain +0.0543162627, geometry_gain_covered +0.0042531670,
total_reference_gap +0.1400909220. Labels: `SELECTION_IS_ACTIONABLE = true`,
`PROPOSAL_GEOMETRY_IS_MAJOR = false`, `PROPOSAL_COVERAGE_IS_MAJOR = true`,
`CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING = true`. Reference ceiling: F-R0 0.440647, F-R1 0.711942,
best eligible coverage@0.50 0.846039. Paired: F-R0 6/20 (+0.125063), F-R1 18/20 (+0.300548),
F-R2 16/20 (+0.330636), F-R3 18/20 (+0.319340).

**Task 7F reporting erratum (recorded, artifacts untouched):** the helper stores both
`selected_confidence/selected_area` (F-R0) and `oracle_selected_confidence/oracle_selected_area` (F-R1), but
`scripts/task7f_reference_modes.py::summarise_references` prints the generic `selected_confidence_mean/
selected_area_mean` fields for both modes, so the F-R1 proposal statistics shown in the Task 7F report are not
F-R1-specific. Reporting-only: the F-R1 mask, its reference mIoU, its downstream and paired metrics, the gap
decomposition and the verdict are unaffected. Frozen Task 7F artifacts were not rewritten.

## 3. Frozen assets and the train-only dataset

Frozen U-C1: YOLO26m-seg Task 6M.1 `ef852b58…61f474`, imgsz 640, conf 0.05, max_det 300, default NMS, no TTA,
no tiling, 512×512 tiles; largest eligibility = non-empty mask, not border-touching, bbox extent ratio ≤ 0.20
(no `area ≥ 150` rule). Frozen downstream: D-B1 `6df31909…21a89c0`, fields v0.2/v0.1, frozen SAM2.1 Hiera
Base+, Task 7E `E-HoldoutL3` (669) and `E-PairedHoldout20` (20).

`G-RefTrainUniqueLargest` (v0.2 **train** split only): 12,778 train rows → 4,119 rows with a `largest`
reference → **3,002** with an explicit canonical reference id (the 1,117 L1 `largest` rows have an empty
`references` list and were excluded); `smallest`-reference rows excluded 1,251; dedup by
`(split, tile_id, reference_source_feature_id)` → **1,063 unique references on 1,063 tiles** (1,939 duplicate
rows removed). Label = best eligible GT-IoU candidate (tie higher YOLO confidence, then lower index) with
coverage minimum IoU 0.50; `no_eligible` **2**, `untrainable_not_covered` **33**, **trainable 1,028** → section-9
gate passed. GT is label construction only and is never a selector input.

## 4. Features, architecture and training

Exact 18-D feature vector (proposal geometry/confidence + set overlap structure; boundary via
`scipy.ndimage.binary_erosion` 3×3, 1 iteration, `border_value=0`): log_area, area_ratio, area_over_set_max,
area_rank_desc, confidence, bbox_extent_ratio, width_ratio, height_ratio, fill_ratio, abs_log_aspect,
boundary_over_sqrt_area, max_iou_other, mean_iou_other, max_self_contained_by_other,
max_other_contained_in_self, self_contained_count_norm, contains_other_count_norm, proposal_count_norm. No GT
input, no centroid x/y, no direction/relation/program, no target, no RGB/SAM2 feature.

`SetContextLargestSelector v1`: shared `Linear(18→64) → LayerNorm(64) → GELU → Linear(64→32) → GELU`;
permutation-invariant `concat(mean, max) ∈ R^64`; head `concat(h_i, context) ∈ R^96 → Linear(96→32) → GELU →
Linear(32→1)`; softmax over valid candidates; exactly one listwise `CrossEntropyLoss`; **6,561 parameters**; no
Transformer/attention/GNN/positional encoding.

Training: tile-level 80/20 SHA256 split (seed 20261001) → **805 train / 223 holdout tiles** (tile overlap 0,
`(tile, reference)` overlap 0); AdamW lr 1e-3, wd 1e-4, batch 32, max 40 epochs, patience 6, seed 20261001,
FP32, no scheduler/augmentation/sweep; checkpoint selection on the internal holdout only (mean selected IoU →
oracle-best top-1 → lower gap → earlier epoch); early stop at epoch 25. Checkpoint
`artifacts/checkpoints/task7g/largest_set_context_selector_v1.pt` (gitignored).

## 5. Internal gate and the STOP

| Metric (223 holdout sets) | G-I0 deterministic | **G-I1 learned** | Gate |
|---|---:|---:|---|
| mean selected-reference GT IoU | 0.5200 | **0.6199** | ≥ 0.62 → **✗** |
| median selected IoU | 0.6609 | 0.7896 | — |
| Pr(selected IoU ≥ 0.50) | 0.6278 | 0.7309 | — |
| oracle-best exact top-1 | 0.4888 | **0.6592** | ≥ 0.55 → ✓ |
| mean `best − selected` gap | 0.2790 | 0.1791 | ≤ 0.14 → **✗** |
| mean selected IoU gain | — | **+0.0999** | ≥ +0.08 → ✓ |

Per the task file, no alteration followed and the **external scene-disjoint stage did not run**:
`evaluation/task7g_external_reference.json` and `evaluation/task7g_external_downstream.json` record
`executed: false` with the STOP reason, `task7g_external_paired.json` was not created, and both external
scripts are hard-guarded by the internal gate.

## 6. Decision

`TASK7G_SELECTOR_ADOPTED = false`: the practical chain keeps `U-C1 proposals → deterministic max-area selector →
predicted reference → P_dir + P_near → frozen D-B1`; D-B1 stays the development L3 decoder candidate, Z-B3 the
frozen decoder baseline/ablation, the deterministic selector the frozen reference baseline/ablation and
F-R1/R-F3 diagnostics only. **Practical reference selection is an unresolved limitation for this project
version** and reference intervention stops here — no further selector, no YOLO retraining, no D-B1 change.

Measured reading (no repair proposed): the learned selector does move the internal holdout the right way
(mean selected-reference IoU +0.0999, oracle-best top-1 +0.1704, mean gap −0.0999, Pr@0.5 +0.1031) but lands
0.0001 below the predeclared 0.62 mean-IoU bar and keeps a 0.1791 residual gap to the best eligible candidate
(bar 0.14). A purely geometric/structural listwise selector did not clear the learnability gate that would have
justified the scene-disjoint evaluation, even though Task 7F measured a usable U-C1 ceiling.

## 7. Tests, storage, git

`python -m pytest tests/ -q` → **1377 passed, 1 skipped** (Task 7F ended at 1325 passed / 1 skipped; no prior
passing test was reduced). `tests/test_task7g_largest_reference_selector.py` adds the 52 Part-P checks.

Not committed: the selector checkpoint, YOLO/SAM2/D-B1/Z-B3 weights, proposal caches, generated selector rows
(`artifacts/task7g/`), source imagery/vectors, `.conda`. Committed: selector architecture code, small
manifests/evaluation JSON, scripts, tests, docs, handoff. No new dataset was downloaded and nothing was
installed. Production CLI defaults were not changed.

Environment note: pushing needed the GitHub Desktop bundled Git because the local Git install is missing
`git-remote-https`; no Git installation or credential configuration was modified.

Watt was **not needed** in Task 7G: the pre-existing Watt instance is transport-only, is not owned by this
project and was left running per the ownership rule; no proxy, host, certificate or TLS setting was read or
modified.

## 8. Interpretation boundary

DSH reports measurements only. The selector is **not** claimed as a project novelty; no final end-to-end
readiness is claimed; no further selector was trained after the failure; YOLO was not retrained; D-B1 was not
modified; the test split was not accessed; no formal full-data training was started; the next
architecture/task was not chosen. The verdict authorizes no automatic repair.

## 9. Recommended next step (exact wording required by Part N)

等待 ChatGPT 根据 Task 7G 的 scene-disjoint selector 与 frozen D-B1 结果决定是否冻结开发版完整 L3 链；若 gate 失败，不自行继续 reference 干预。

## 10. STOP

Task 7G stops here: no Task 7G.1, no further selector, no YOLO retraining, no D-B1 modification, no test access,
no formal full training, no GUI. Waiting for the ChatGPT audit.
