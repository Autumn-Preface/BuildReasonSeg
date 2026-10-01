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

# FROM_DSH — Task 7I Report: Formal L3 Three-Seed Train/Validation (Z-B3 vs D-B1)

_This file holds the Task 7I report. The Task 7H report is preserved in git history at commit `511a7bd`;
Task 7G at `22401fe`; Task 7F at `c59c6d3`; Task 7E at `f9c6e87`._

**Note on the legacy `ARTIFACT-FACTS` block above:** those numbers describe the superseded
BuildSpatialReason **v0.1.1** dataset (read-only legacy evidence). The active canonical reasoning dataset
for all Task 6L+ work is **BuildSpatialReason v0.2 over WHU-EA-NativeVector v1.0**.

Full design notes: `docs/task7i_formal_l3_trainval.md`. Validation stage only — **the test split was never
opened, hashed, enumerated or evaluated and stays LOCKED.**

## 1. Verdict

**`DB1_FORMAL_VAL_NOT_CONFIRMED`** — section 28 priority: protocol clean, test lock LOCKED before and after,
formal population exact, all six runs complete and fresh-initialized, but **one of the eight section-25 gates
fails** → item 5.

| # | Gate | Required | Measured | Pass |
|---|---|---|---:|---|
| 1 | D-B1 mean oracle full-val mIoU | ≥ 0.35 | 0.3834 | ✓ |
| 2 | D-B1 mean − Z-B3 mean (oracle) | ≥ +0.04 | +0.0510 | ✓ |
| 3 | matched seeds with D-B1 > Z-B3 | ≥ 2/3 | 3/3 | ✓ |
| 4 | lowest D-B1 seed oracle mIoU | ≥ 0.32 | 0.3799 | ✓ |
| 5 | D-B1 mean − Z-B3 mean (predicted strict) | ≥ +0.02 | **+0.0065** | **✗** |
| 6 | D-B1 mean predicted strict mIoU | ≥ 0.22 | 0.2335 | ✓ |
| 7 | D-B1 predicted pair margin not lower by > 0.02 | ≥ Z-B3 − 0.02 | +0.2009 vs +0.1952 | ✓ |
| 8 | no protocol violation | — | — | ✓ |

## 2. Recorded Task 7G result (unchanged)

`LARGEST_SELECTOR_NOT_LEARNABLE`: G-I0 mean selected reference IoU `0.5199913587`; G-I1 `0.6198877726`
(gain `+0.0998964138`), oracle-best top-1 `0.6591928251`, mean gap `0.1790755783`; gate gain PASS,
mean IoU ≥ 0.62 FAIL, top1 PASS, gap ≤ 0.14 FAIL → external stage never executed, no scene-disjoint claim,
selector not adopted, reference intervention stops.

## 3. Protocol corrections (Task 7H artifacts untouched)

* **I-01** — Task 7H's `checkpoint_selection = "MiniVal240 mIoU"` is superseded by **all 936 valid v0.2 L3
  validation records, oracle-reference mIoU**.
* **I-02** — the formal comparison retrains **both Z-B3 and D-B1** with the identical formal train/val
  population, identical seeds, identical optimizer schedule and fresh trainable initialization; the historical
  checkpoints remain development evidence only.

## 4. Frozen populations

`G-FormalTrain` **1344** = left 323 / right 347 / above 338 / below 336; `G-FormalVal` **936** = left 250 /
right 249 / above 224 / below 213; train∩val = **0**; both sample-id SHA256 recorded; no test material read.
`I-FormalValPairsAll` = **326** pairs (same tile, same oracle reference, different direction, different target;
dedup by the exact `min||max` key; sorted; never subsampled; reporting-only).

## 5. Six formal runs (frozen protocol)

Frozen SAM2 features, frozen fields, oracle GT reference, fresh initialization, AdamW lr 3e-4 / wd 1e-4 /
batch 8 / max 25 epochs / patience 5 / no scheduler / no augmentation / bf16 AMP / `BCE + Dice` only; per-epoch
evaluation of all 936 val records; selection by (val mIoU, val Dice, val Pr@0.5, earlier epoch); early stopping
on full-val mIoU.

| Run | Selected epoch | Final epoch | Val mIoU | Wall | Params |
|---|---:|---:|---:|---:|---:|
| Z-B3 / 20261001 | 4 | 9 | 0.3194 | 120.6 s | 275,777 |
| Z-B3 / 20261002 | 14 | 19 | 0.3394 | 75.8 s | 275,777 |
| Z-B3 / 20261003 | 5 | 10 | 0.3386 | 40.6 s | 275,777 |
| D-B1 / 20261001 | 11 | 16 | 0.3799 | 126.3 s | 278,081 |
| D-B1 / 20261002 | 9 | 14 | 0.3827 | 62.1 s | 278,081 |
| D-B1 / 20261003 | 8 | 13 | 0.3877 | 57.5 s | 278,081 |

## 6. Oracle-reference full-val results

Z-B3 mIoU 0.3194 / 0.3394 / 0.3386 (mean **0.3324** ± 0.0113), pair pass rate 0.8691, margin +0.3167.
**D-B1 0.3799 / 0.3827 / 0.3877 (mean 0.3834 ± 0.0039)**, pair pass rate 0.8446, margin **+0.3775**.
Matched-seed deltas D-B1 − Z-B3: **+0.0606 / +0.0433 / +0.0491 (3/3)**.

## 7. Practical predicted-reference full-val results

Reference computed once and shared by all six runs: mIoU **0.4469**, abstentions **5**, `REFERENCE_OK 491`,
`SELECTION_WRONG 305`, `NOT_COVERED 135`, `ABSTENTION 5`, `GEOMETRY_POOR 0`.
Z-B3 strict 0.2203 / 0.2267 / 0.2340 (mean **0.2270**, pair pass rate 0.5153, margin +0.1952);
**D-B1 strict 0.2259 / 0.2356 / 0.2390 (mean 0.2335**, pair pass rate 0.4172, margin **+0.2009)**.
Oracle→predicted retention (D-B1): 0.5946 / 0.6158 / 0.6164 (mean **0.6090**).

## 8. Measured reading (no repair proposed)

Under the **oracle** reference the D-B1 advantage is confirmed and clearly more seed-stable than Z-B3
(+0.0510 mIoU mean, 3/3 matched seeds, std 0.0039 vs 0.0113, pair margin +0.3775 vs +0.3167). Under the
**practical predicted** reference the same comparison loses most of that advantage (+0.0065 strict mIoU, below
the predeclared +0.02 gate) and its counterfactual pair **pass rate** drops to 0.4172 versus Z-B3's 0.5153,
although its margin stays slightly higher (+0.2009 vs +0.1952). The reference stage — 305/936
`SELECTION_WRONG` and 135/936 `NOT_COVERED` — dilutes the decoder-level gain, exactly the limitation recorded
in Tasks 7F/7H: **the architecture gain is real at the decoder level but not yet confirmed end-to-end.**

## 9. Test lock

`evaluation/task7i_test_lock_status.json`: `status = LOCKED`, `test_execution_authorized = false`,
`task7i_completed_train_val = true`, `db1_formal_val_confirmed = false`,
`unlock_requires = "ChatGPT audit of Task 7I"`. The Task 7H lock was re-verified as LOCKED before and after the
stage. DSH does not unlock it and authorizes no test action.

## 10. Tests, storage, git

`python -m pytest tests/ -q` → **1467 passed, 1 skipped** (Task 7H ended at 1420 passed / 1 skipped; no prior
passing test was reduced). `tests/test_task7i_formal_trainval.py` adds the 47 Part-P checks.

Not committed: the six Task 7I checkpoints (gitignored under `artifacts/checkpoints/task7i/`), SAM2/proposal
feature caches, expanded pack rows, source imagery/vectors, `.conda`. Committed: population/pair manifests,
small training/evaluation JSON, scripts, tests, docs, handoff. No production CLI default was changed. Nothing
was downloaded or installed. Watt was **not needed** in Task 7I (the pre-existing instance is transport-only,
not owned by this project, left running).

## 11. Interpretation boundary

DSH reports measurements only. Validation results are **not** called test results; the test split is never
called untouched; no seed was chosen by test; no unfavourable seed was dropped; no architecture, loss,
hyperparameter, field, parser or reference policy was changed after seeing the formal val results; the test
split stays LOCKED; no final-test inference was started; parser/reference/YOLO were not retrained; no further
selector was trained; unrestricted natural-language capability, unseen-city generalization and novelty/"first"
claims are not made.

## 12. Recommended next step (exact wording required by Part N)

等待 ChatGPT 审核 Task 7I 的三种子正式 train/val 结果；在 ChatGPT 明确解锁前，test 保持 LOCKED，不运行任何 final-test inference。

## 13. STOP

Task 7I stops here: the test lock is not unlocked, no final test is run, Task 7J is not started, the
architecture is not changed and no further selector/reference/parser training is started. Waiting for the
ChatGPT audit.
