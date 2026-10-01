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

# FROM_DSH — Task 7J Report: Final Frozen-Architecture Test Evaluation

_This file holds the Task 7J report. The Task 7I report is preserved in git history at commit `cdd6a68`;
Task 7H at `511a7bd`; Task 7G at `22401fe`; Task 7F at `c59c6d3`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7j_final_test.md`. This was the single ChatGPT-authorized **final
frozen-architecture test evaluation** — measurement only, no training, no seed selection, no model change.

## 1. Verdict

**`FINAL_FROZEN_TEST_COMPLETE`** — the procedural priority order (no performance-based PASS/FAIL exists):
protocol clean, all six Task 7I `best.pt` checkpoints hash-verified, the final L3 test population frozen before
inference, all six checkpoints evaluated under both reference modes, no seed selected, no filtering.

Test consumption: **`FINAL_TEST_CONSUMED`** — `test_execution_authorized = false`,
`task7j_final_test_completed = true`, `architecture_changes_after_test_authorized = false`,
`retest_for_model_selection_authorized = false`.

## 2. Frozen Task 7I facts (recorded, not modified)

Oracle validation: Z-B3 `0.319367 / 0.339382 / 0.338579` → **0.332443 ± 0.011331**; D-B1
`0.379934 / 0.382651 / 0.387688` → **0.383424 ± 0.003934**; matched deltas `+0.060567 / +0.043269 / +0.049109`,
wins 3/3. Practical predicted-reference validation: Z-B3 `0.226984`, D-B1 `0.233519`, delta `+0.006535`
(`< +0.02` gate) → Task 7I keeps **`DB1_FORMAL_VAL_NOT_CONFIRMED`**. Reference on 936 val: mIoU `0.446949`,
`REFERENCE_OK 491`, `SELECTION_WRONG 305`, `NOT_COVERED 135`, abstentions `5`, coverage@0.50 `0.850427`.
Counterfactual validation: oracle pair pass Z-B3 `0.869121` / D-B1 `0.844581`, margin `+0.316684` / `+0.377485`;
predicted pair pass `0.515337` / `0.417178`, margin `+0.195162` / `+0.200931`.

**D-B1 does not universally improve the counterfactual pair pass rate**: it has the larger margin but the lower
pass rate under both references, on validation and on test. That is reported as measured.

## 3. Authorization and checkpoints

`evaluation/task7j_test_authorization.json` was written **before any test record was read**:
`authorized_by = "ChatGPT audit after Task 7I"`, scope "Task 7J final frozen-architecture evaluation only",
`training_allowed = false`, `checkpoint_selection_allowed = false`, `architecture_change_allowed = false`,
`test_access_authorized = true`, historical Task 6M test access disclosed, permitted wording
`final frozen-architecture test evaluation` ("untouched test" forbidden). The Task 7I lock was LOCKED.

All six `artifacts/checkpoints/task7i/<model>/<seed>/best.pt` files were verified against the authoritative
Task 7I training-artifact hashes/bytes (Z-B3 × 3 seeds, D-B1 × 3 seeds, 275,777 / 278,081 params). `last.pt`
files exist on disk but were never used. No retraining.

## 4. Frozen final L3 test population

**736** valid v0.2 test records of the four L3 programs — left 190 / right 178 / above 174 / below 194 — on
**462** unique tiles, frozen before inference, **no filtering** by model output/target size/reference
quality/success and no post-inference exclusion; sample-id SHA256 recorded.
`I-FinalTestPairsAll`: **274** pairs (same tile, same oracle reference, different direction, different target;
deduplicated by the exact `min||max` key, sorted, never subsampled) over 228 unique tiles; reporting only.

## 5. Oracle-reference final test

Z-B3 `0.3315 / 0.3416 / 0.3394` → **0.3375 ± 0.0053** mIoU, pair pass 0.8491, margin +0.3186.
**D-B1 `0.3985 / 0.3868 / 0.3917` → 0.3923 ± 0.0059**, pair pass 0.8236, margin **+0.3735**.
Mean delta **+0.0548**, matched-seed wins **3/3** (`+0.0670 / +0.0452 / +0.0523`). Dice Z-B3
0.4533/0.4651/0.4640, D-B1 0.5172/0.5131/0.5183. Inference ≈ 16–25 ms/record (excluding the first-run cache
warm-up outlier of 104.59 ms).

## 6. Practical predicted-reference final test

Reference computed once and shared by all six checkpoints: mIoU **0.3752**, Dice 0.4396, Pr@0.5 0.4152,
abstentions **2** (0.27 %), `REFERENCE_OK 319`, `SELECTION_WRONG 369`, `NOT_COVERED 46`, `ABSTENTION 2`,
`GEOMETRY_POOR 0`, best-eligible coverage@0.50 **0.9348**.

Z-B3 `0.2065 / 0.2076 / 0.2097` → **0.2079** strict mIoU (`REFERENCE_OK` subset 0.3683, pair pass 0.4136,
margin +0.1801). **D-B1 `0.2048 / 0.2119 / 0.2157` → 0.2108** (`REFERENCE_OK` subset **0.3884**, pair pass
0.3163, margin +0.1801). Mean delta **+0.0029**; oracle→predicted retention (D-B1) **0.5375**
(0.5139 / 0.5478 / 0.5507).

## 7. Final comparison (descriptive; no gate) and validation→test shift

* Oracle mean delta **+0.0548** with D-B1 winning **3/3** matched seeds.
* Practical mean delta **+0.0029**, retention **0.5375**.
* Val→test shift: oracle Z-B3 **+0.0051**, oracle D-B1 **+0.0089**; practical Z-B3 **−0.0191**, practical
  D-B1 **−0.0227** — the test split is slightly harder for the practical chain with the same ordering.

Safe interpretation (exactly the section-10 wording): **D-B1 retains a target-decoder advantage under correct
reference on the final frozen-architecture test**, and **the decoder-level gain does not translate
proportionally to the practical chain because reference selection/coverage remains limiting**.

## 8. Tests, storage, git

`python -m pytest tests/ -q` → **1511 passed, 1 skipped** (Task 7I ended at 1467 passed / 1 skipped; no prior
passing test was reduced). `tests/test_task7j_final_test.py` adds the 44 Part-14 checks.

Not committed: model checkpoints, feature/proposal caches, imagery/vectors, the expanded local test rows
(`artifacts/task7j/`), `.conda`. Committed: small manifests/results, scripts, tests, docs, handoff. No
checkpoint was created and no CLI default was changed. Nothing was downloaded or installed. Watt was **not
needed** in Task 7J (the pre-existing instance is transport-only, not owned by this project, left running).

## 9. Interpretation boundary

Task 7J has no performance-based PASS/FAIL verdict. Nothing was trained, continued, tuned, selected or
modified; only the six Task 7I `best.pt` checkpoints were evaluated, all three seeds of both models are
reported with mean ± sample std (`ddof=1`), no seed was chosen, no threshold/reference/architecture was changed
after the results, and no second test run with different settings was performed. **Not claimed**: unrestricted
natural-language understanding, unseen-city generalization, a solved reference stage, an untouched test split,
or novelty/"first". The historical Task 6M test access is disclosed, so the permitted wording is
`final frozen-architecture test evaluation`. Demo packaging was not started.

## 10. Recommended next step (exact wording required by section 17)

等待 ChatGPT 审核 Task 7J 的 final frozen-architecture test 结果；test 已消费，不自行据此修改模型、选择 seed、调整阈值或重新运行开发实验。

## 11. STOP

Task 7J stops here: no retraining, no rerun with changed settings, no best-seed selection, no
architecture/reference/parser change, no automatic demo packaging. Waiting for the ChatGPT audit.
