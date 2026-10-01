# Task 7G — Largest-Reference Set-Context Selector (Final Reference Intervention)

> Task: `handoff/TO_DSH.md` (Task 7G) · Base commit: `c59c6d3` · Predecessor: Task 7F →
> `REFERENCE_SELECTION_DOMINANT`
> **Verdict: `LARGEST_SELECTOR_NOT_LEARNABLE`** (section-23 internal gate failed 2/4) · external
> scene-disjoint stage **not executed** (section 23 is a STOP gate)
> Tests: `tests/test_task7g_largest_reference_selector.py` · Evidence: `evaluation/task7g_*.json`

Task 7G is the project's final reference intervention: learn to pick the canonical `largest` reference out of the
**existing frozen U-C1 proposal set**, using only proposal geometry/confidence and proposal-set overlap
structure. The experiment stopped at the predeclared internal learnability gate, so no external artifact was
fabricated.

## 1. Recorded Task 7F evidence (frozen, not modified)

Verdict `REFERENCE_SELECTION_DOMINANT`.

```text
M0 F-R0 current selected predicted mask = 0.2454050104
M1 F-R1 oracle-selected predicted mask = 0.3649093893
M2 F-R2 coverage-conditional GT mask    = 0.3311796697
M3 F-R3 full oracle GT mask             = 0.3854959324

selection_gain        = +0.1195043789
coverage_gain         = +0.0543162627
geometry_gain_covered = +0.0042531670
total_reference_gap   = +0.1400909220
```

Labels: `SELECTION_IS_ACTIONABLE = true`, `PROPOSAL_GEOMETRY_IS_MAJOR = false`,
`PROPOSAL_COVERAGE_IS_MAJOR = true`, `CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING = true`.

Reference ceiling: F-R0 reference mIoU `0.440647`, F-R1 reference mIoU `0.711942`, best eligible coverage@0.50
`0.846039`. Paired: F-R0 `6/20` (margin `+0.125063`), F-R1 `18/20` (`+0.300548`), F-R2 `16/20` (`+0.330636`),
F-R3 `18/20` (`+0.319340`).

## 2. Task 7F reporting erratum (recorded, artifacts untouched)

`buildreasonseg_mvp/task7f_reference_ceiling.py` correctly stores both `selected_confidence/selected_area`
(F-R0) and `oracle_selected_confidence/oracle_selected_area` (F-R1), but
`scripts/task7f_reference_modes.py::summarise_references` reports the generic
`selected_confidence_mean/selected_area_mean` fields for **both** F-R0 and F-R1, so the F-R1 values printed in
the Task 7F report are not F-R1-specific proposal statistics. This is a reporting-only issue: it does not affect
the F-R1 mask, F-R1 reference mIoU, F-R1 downstream metrics, paired metrics, the gap decomposition or the Task 7F
verdict. Frozen Task 7F artifacts were not rewritten.

## 3. Frozen assets

Frozen U-C1 proposal generator: YOLO26m-seg Task 6M.1 SHA256 `ef852b58…61f474`, imgsz 640, conf 0.05,
max_det 300, default NMS, no TTA, no tiling, 512×512 source tile. Largest eligibility exactly: non-empty mask,
not border-touching, bbox extent ratio ≤ 0.20 (no `area ≥ 150` rule, no extra quality gate). Frozen downstream:
D-B1 `artifacts/checkpoints/task7d/db1_minitrain1200.pt` SHA256 `6df31909…21a89c0`, GeometricRelationField
v0.2, NearestBoundaryField v0.1, frozen SAM2.1 Hiera Base+ features, Task 7E `E-HoldoutL3` (669 records) and
`E-PairedHoldout20` (20 pairs). No D-B1 training.

## 4. `G-RefTrainUniqueLargest` (train split only)

`evaluation/task7g_training_dataset_manifest.json`:

| | Value |
|---|---|
| v0.2 train rows | 12,778 |
| rows whose program contains a `largest` reference | 4,119 |
| of those with an explicit canonical reference id | **3,002** (the 1,117 L1 `largest` rows carry an empty `references` list and were excluded) |
| `smallest`-reference rows excluded | 1,251 |
| dedup key | `(split, tile_id, reference_source_feature_id)` |
| **unique references / tiles** | **1,063 / 1,063** (0 duplicate rows remained) |
| per program | `largest_to_nearest` 672 · `largest_to_right_of_to_nearest` 347 · `largest_to_below_to_nearest` 336 · `largest_to_left_of_to_nearest` 323 · `largest_to_left_of` 260 · `largest_to_right_of` 248 · `largest_to_above` 246 · `largest_to_below` 232 |
| labels | best eligible GT-IoU candidate, tie higher YOLO confidence then lower index; coverage minimum IoU 0.50 |
| exclusions | `no_eligible` **2**, `untrainable_not_covered` **33**, **trainable 1,028** |
| section-9 gate | trainable 1,028 ≥ 300 ✓ and 1,063 unique train tiles ≥ 50 ✓ → `SELECTOR_TRAIN_DATA_READY` |

GT is label construction only; it is never a selector input.

## 5. Features and architecture (exact)

18-dimensional feature vector per eligible proposal (sections 11-14): `log_area`, `area_ratio`,
`area_over_set_max`, `area_rank_desc` (largest = 0, stable tie by higher confidence then lower index),
`confidence`, `bbox_extent_ratio`, `width_ratio`, `height_ratio`, `fill_ratio`, `abs_log_aspect`,
`boundary_over_sqrt_area` (`scipy.ndimage.binary_erosion`, 3×3 ones, 1 iteration, `border_value=0`),
`max_iou_other`, `mean_iou_other`, `max_self_contained_by_other`, `max_other_contained_in_self`,
`self_contained_count_norm`, `contains_other_count_norm`, `proposal_count_norm`. No GT-derived input, no
centroid x/y, no direction/relation/program, no target, no RGB/SAM2 feature.

`SetContextLargestSelector v1`: shared proposal encoder `Linear(18→64) → LayerNorm(64) → GELU → Linear(64→32) →
GELU`; permutation-invariant context `concat(mean_i h_i, max_i h_i) ∈ R^64`; candidate head
`concat(h_i, context) ∈ R^96 → Linear(96→32) → GELU → Linear(32→1)`; softmax over valid candidates; exactly one
listwise `CrossEntropyLoss`; **6,561 parameters**; no Transformer/attention/GNN, no positional encoding, no
proposal-order feature.

## 6. Internal tile-disjoint holdout and the STOP

Split at tile level by SHA256 with seed 20261001, 80/20: **805 train tiles / 805 trainable sets** and
**223 holdout tiles / 223 trainable sets**, tile overlap **0**, `(tile, reference)` overlap **0** (≥50 holdout
tiles ✓). Uncovered/no-eligible records stayed counted in their partition but out of the loss and accuracy
denominators. Training (section 21): fresh initialization, AdamW lr 1e-3, weight decay 1e-4, batch 32, max 40
epochs, patience 6, seed 20261001, no scheduler, no augmentation, FP32, no sweep; checkpoint selected on the
internal holdout only by (mean selected-reference GT IoU → oracle-best exact top-1 → lower mean
`best_iou - selected_iou` → earlier epoch). Early stopping fired at epoch 25.

| Metric (223 holdout sets) | G-I0 deterministic max-area | **G-I1 learned selector** | Gate |
|---|---:|---:|---|
| mean selected-reference GT IoU | 0.5200 | **0.6199** | ≥ 0.62 → **✗** (0.6199) |
| median selected IoU | 0.6609 | 0.7896 | — |
| Pr(selected IoU ≥ 0.50) | 0.6278 | 0.7309 | — |
| oracle-best exact top-1 | 0.4888 | **0.6592** | ≥ 0.55 → ✓ |
| mean `best - selected` IoU gap | 0.2790 | 0.1791 | ≤ 0.14 → **✗** (0.1791) |
| mean best eligible IoU | 0.7990 | 0.7990 | — |
| mean selected IoU gain over G-I0 | — | **+0.0999** | ≥ +0.08 → ✓ |

Section 23 gate result: **2 of 4 conditions fail** → `LARGEST_SELECTOR_NOT_LEARNABLE`. Per the task file the
features, model, loss and protocol were **not** altered, and the external scene-disjoint stage (Part H) was not
run: `evaluation/task7g_external_reference.json` and `evaluation/task7g_external_downstream.json` record
`executed: false` with the STOP reason, and `task7g_external_paired.json` was not created.

Checkpoint `artifacts/checkpoints/task7g/largest_set_context_selector_v1.pt` (gitignored, selected epoch 24).

## 7. Verdict and development decision

Section 34 priority: protocol clean, SciPy available, dataset gate passed, and the section-23 internal gate
failed → **`LARGEST_SELECTOR_NOT_LEARNABLE`**. The later conditions (`TASK7F_BASELINE_REPRODUCTION_FAIL`,
`LARGEST_SELECTOR_SCENE_DISJOINT_FAIL`, `LARGEST_SELECTOR_DEVELOPMENT_READY`) are not reached.

`TASK7G_SELECTOR_ADOPTED = false`: the practical chain keeps
`U-C1 proposals → deterministic max-area selector → predicted reference → P_dir + P_near → frozen D-B1`;
D-B1 stays the development L3 decoder candidate, Z-B3 the frozen decoder baseline/ablation, the deterministic
selector the frozen reference baseline/ablation and F-R1/F-R2/F-R3 diagnostics only. **Practical reference
selection is recorded as an unresolved limitation for this project version**, and reference intervention stops
here — no further selector, no YOLO retraining.

Measured reading (reported only, no repair proposed): the learned selector does move the internal holdout in the
right direction — mean selected-reference IoU +0.0999, top-1 +0.1704, gap −0.0999 and Pr@0.5 +0.1031 over the
deterministic baseline — but it lands at 0.6199 mean IoU, 0.0001 below the predeclared 0.62 bar, and its
0.1791 residual gap to the best eligible candidate stays well above the 0.14 bar. Even with the usable
U-C1 ceiling that Task 7F measured, a purely geometric/structural listwise selector did not clear the
learnability gate that would have justified the scene-disjoint evaluation.

## 8. Interpretation boundary

DSH reports measurements only. The selector is **not** claimed as a project novelty; no final end-to-end
readiness is claimed; no further selector was trained after the failure; YOLO was not retrained; D-B1 was not
modified; the test split was not accessed; no formal full-data training was started; the next architecture/task
was not chosen. Final recommendation exactly:

`等待 ChatGPT 根据 Task 7G 的 scene-disjoint selector 与 frozen D-B1 结果决定是否冻结开发版完整 L3 链；若 gate 失败，不自行继续 reference 干预。`

## 9. Reproduce

```text
python scripts/task7g_build_selector_dataset.py   # G-RefTrainUniqueLargest + manifest (train split only)
python scripts/task7g_train_selector.py           # training + section-23 internal gate (exit 4 on STOP)
python scripts/task7g_evaluate_selector.py        # section 24-26 external stage (guarded by section 23)
python scripts/task7g_evaluate_downstream.py      # section 27-28 frozen D-B1 (guarded)
python scripts/task7g_report.py                   # section 30-34 gate, decision, verdict
```

The dataset rows and the selector checkpoint live in the gitignored `artifacts/task7g/` and
`artifacts/checkpoints/task7g/`. Run in `.conda/buildreasonseg-proposal` with `HF_HUB_OFFLINE=1` (dataset
build); training needs no GPU (FP32, CPU) and the report step also runs in `.conda/buildreasonseg-mvp`.
