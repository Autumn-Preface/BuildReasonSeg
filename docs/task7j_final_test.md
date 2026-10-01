# Task 7J — Final Frozen-Architecture Test Evaluation

> Task: `handoff/TO_DSH.md` (Task 7J) · Base commit: `cdd6a68` · Predecessor: Task 7I →
> `DB1_FORMAL_VAL_NOT_CONFIRMED`
> **Verdict: `FINAL_FROZEN_TEST_COMPLETE`** · test consumption status **`FINAL_TEST_CONSUMED`**
> This is the single ChatGPT-authorized `final frozen-architecture test evaluation`. It is measurement only —
> no training, no seed selection, no architecture/reference/threshold change, and no second run.
> Tests: `tests/test_task7j_final_test.py` · Evidence: `evaluation/task7j_*.json`

## 1. Frozen Task 7I facts (recorded, not modified)

Oracle validation (936 records): Z-B3 seeds `0.319367 / 0.339382 / 0.338579`, mean **0.332443 ± 0.011331**;
D-B1 seeds `0.379934 / 0.382651 / 0.387688`, mean **0.383424 ± 0.003934**; matched D-B1−Z-B3
`+0.060567 / +0.043269 / +0.049109`, wins **3/3**.

Practical predicted-reference validation: Z-B3 strict mean `0.226984`, D-B1 strict mean `0.233519`, delta
`+0.006535` → the predeclared `+0.02` gate failed, so Task 7I's verdict stays
**`DB1_FORMAL_VAL_NOT_CONFIRMED`**.

Reference on 936 val: mIoU `0.446949`, `REFERENCE_OK 491`, `SELECTION_WRONG 305`, `NOT_COVERED 135`,
abstentions `5`, coverage@0.50 `0.850427`.

Counterfactual validation: oracle pair pass Z-B3 `0.869121` / D-B1 `0.844581`, oracle margin Z-B3 `+0.316684` /
D-B1 `+0.377485`; predicted pair pass Z-B3 `0.515337` / D-B1 `0.417178`, predicted margin Z-B3 `+0.195162` /
D-B1 `+0.200931`. **D-B1 does not universally improve the pair pass rate** — it has the larger margin but the
lower pass rate under both references; that is reported as measured and is not claimed otherwise.

## 2. One-time authorization

`evaluation/task7j_test_authorization.json` was written **before any test record was read**:
`authorized_by = "ChatGPT audit after Task 7I"`, `scope = "Task 7J final frozen-architecture evaluation only"`,
`training_allowed = false`, `checkpoint_selection_allowed = false`, `architecture_change_allowed = false`,
`test_access_authorized = true`, `historical_test_access_disclosed = true`,
`permitted_test_description = "final frozen-architecture test evaluation"` (the wording "untouched test" is
forbidden). The Task 7I test lock was verified LOCKED before the authorization was consumed.

## 3. Checkpoints (six, `best.pt` only)

`evaluation/task7j_checkpoint_manifest.json` verifies all six Task 7I `best.pt` files against the authoritative
`evaluation/task7i_training_zb3.json` / `..._db1.json` hashes and byte counts:

| Checkpoint | Selected epoch | Params | SHA256 match |
|---|---:|---:|---|
| Z-B3 / 20261001 | 4 | 275,777 | ✓ |
| Z-B3 / 20261002 | 14 | 275,777 | ✓ |
| Z-B3 / 20261003 | 5 | 275,777 | ✓ |
| D-B1 / 20261001 | 11 | 278,081 | ✓ |
| D-B1 / 20261002 | 9 | 278,081 | ✓ |
| D-B1 / 20261003 | 8 | 278,081 | ✓ |

`last.pt` files exist on disk but were **never used**; no retraining happened.

## 4. Frozen final L3 test population

`evaluation/task7j_test_population_manifest.json` — every valid BuildSpatialReason v0.2 **test** record of the
four canonical L3 programs, frozen before inference:

| | Records |
|---|---:|
| total | **736** |
| `largest_to_left_of_to_nearest` | 190 |
| `largest_to_right_of_to_nearest` | 178 |
| `largest_to_above_to_nearest` | 174 |
| `largest_to_below_to_nearest` | 194 |
| unique tiles | 462 |

Filters applied: **none** (no filtering by model output, target size, reference quality or success); no
post-inference exclusion; sample-id SHA256 recorded. Historical Task 6M test access is disclosed.

`evaluation/task7j_test_pair_manifest.json` — `I-FinalTestPairsAll`: **274** pairs (same tile, same oracle
largest reference, different direction, different target; deduplicated by the exact `min||max` key, sorted
lexicographically, never subsampled), spanning 228 unique tiles, reporting only.

## 5. Oracle-reference final test

| Checkpoint | mIoU | Dice | Pairs (274) | Margin | ms/record |
|---|---:|---:|---:|---:|---:|
| Z-B3 / 20261001 | 0.3315 | 0.4533 | 224 | +0.3135 | 104.59 |
| Z-B3 / 20261002 | 0.3416 | 0.4651 | 233 | +0.3218 | 20.11 |
| Z-B3 / 20261003 | 0.3394 | 0.4640 | 241 | +0.3203 | 16.28 |
| **D-B1 / 20261001** | **0.3985** | 0.5172 | 216 | +0.3796 | 24.65 |
| **D-B1 / 20261002** | **0.3868** | 0.5131 | 227 | +0.3681 | 16.65 |
| **D-B1 / 20261003** | **0.3917** | 0.5183 | 234 | +0.3728 | 16.68 |

Aggregates: **Z-B3 0.3375 ± 0.0053**, **D-B1 0.3923 ± 0.0059**, mean delta **+0.0548**, matched-seed wins
**3/3** (+0.0670 / +0.0452 / +0.0523). Pair pass Z-B3 0.8491 / D-B1 0.8236; pair margin Z-B3 +0.3186 /
D-B1 **+0.3735**. (Per-relation, target-area and boundary-distance quartiles and exact efficiency values are in
`evaluation/task7j_oracle_test_results.json`.)

## 6. Practical predicted-reference final test

Reference computed once and reused by all six checkpoints: mIoU **0.3752**, abstentions **2** (0.27 %),
`REFERENCE_OK 319`, `SELECTION_WRONG 369`, `NOT_COVERED 46`, `ABSTENTION 2`, `GEOMETRY_POOR 0`.

| Checkpoint | Strict mIoU | Answered-only | `REFERENCE_OK` subset | Pairs (274) | Margin |
|---|---:|---:|---:|---:|---:|
| Z-B3 / 20261001 | 0.2065 | 0.2071 | 0.3624 | 109 | +0.1801 |
| Z-B3 / 20261002 | 0.2076 | 0.2082 | 0.3734 | 113 | +0.1781 |
| Z-B3 / 20261003 | 0.2097 | 0.2103 | 0.3690 | 118 | +0.1820 |
| D-B1 / 20261001 | 0.2048 | 0.2053 | 0.3834 | 79 | +0.1737 |
| D-B1 / 20261002 | 0.2119 | 0.2125 | 0.3861 | 89 | +0.1830 |
| D-B1 / 20261003 | 0.2157 | 0.2163 | 0.3956 | 92 | +0.1834 |

Aggregates: **Z-B3 strict 0.2079**, **D-B1 strict 0.2108**, mean delta **+0.0029**; `REFERENCE_OK` subset
Z-B3 0.3683 / D-B1 0.3884; pair pass Z-B3 0.4136 / D-B1 0.3163; pair margin Z-B3 +0.1801 / D-B1 +0.1801
(differences below 1e-4). Oracle→predicted retention (D-B1): 0.5139 / 0.5478 / 0.5507, mean **0.5375**.
Reference best-eligible coverage@0.50 on the test split is **0.9348**.

## 7. Final comparison (descriptive only, no gate)

* **Oracle**: Z-B3 0.3375 ± 0.0053 → **D-B1 0.3923 ± 0.0059**, delta **+0.0548**, D-B1 wins **3/3** matched
  seeds, pair margin +0.3735 vs +0.3185 (**D-B1 larger**), pair pass 0.8258 vs 0.8539 (**Z-B3 higher**).
* **Practical**: Z-B3 0.2079 → D-B1 0.2108, delta **+0.0029**, retention **0.5375**, `REFERENCE_OK` subset
  +0.0200 in favour of D-B1.
* **Validation→test shift** (descriptive): oracle Z-B3 +0.0051 and D-B1 +0.0089; practical Z-B3 −0.0191 and
  D-B1 −0.0227. The test split is slightly harder for the practical chain, with the same ordering.

Safe interpretation (exactly as allowed by section 10):
**D-B1 retains a target-decoder advantage under correct reference on the final frozen-architecture test**, and
**the decoder-level gain does not translate proportionally to the practical chain because reference
selection/coverage remains limiting** (369/736 `SELECTION_WRONG`, 46/736 `NOT_COVERED`; retention 0.5375).

## 8. Test consumption lock

`evaluation/task7j_test_consumption_status.json`: `status = FINAL_TEST_CONSUMED`,
`test_execution_authorized = false`, `task7j_final_test_completed = true`,
`architecture_changes_after_test_authorized = false`, `retest_for_model_selection_authorized = false`. The test
may later be reproduced or reported for the same frozen setup only, never for iterative development.

## 9. Interpretation boundary

No performance-based PASS/FAIL verdict exists for Task 7J. Nothing was trained, continued, tuned, selected or
modified: the six Task 7I `best.pt` checkpoints are the only models evaluated, all three seeds of both models
are reported with mean ± sample std (`ddof=1`), no seed was chosen, no threshold/reference/architecture was
changed after the results, and no second test run with different settings was performed. The following are
**not** claimed: unrestricted natural-language understanding, unseen-city generalization, a solved reference
stage, an untouched test split, or any novelty/"first" result. The historical Task 6M test access is disclosed,
so the permitted wording is `final frozen-architecture test evaluation`. Final recommendation exactly:

`等待 ChatGPT 审核 Task 7J 的 final frozen-architecture test 结果；test 已消费，不自行据此修改模型、选择 seed、调整阈值或重新运行开发实验。`
