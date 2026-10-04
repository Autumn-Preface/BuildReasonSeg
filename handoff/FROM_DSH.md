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

# FROM_DSH — Task 8B.3-M1C Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1C` |
| Status | **COMPLETE** (main integrated by fast-forward only) |
| Branch integrated | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `509ed5ea6c712da0d65d31c499fa40cec16e94cc` |
| Pre-integration origin/main | `c45ecbec7fd293c454ccced22310db32c1542be4` |
| merge-base | `c45ecbec7fd293c454ccced22310db32c1542be4` |
| Ahead/behind before doc commit | 25/0 |
| Ahead/behind after doc commit | 26/0 (re-proved before touching main) |
| Integration method | `git merge --ff-only` on local `main` after `git reset --hard origin/main` (no rebase/squash/cherry-pick/merge commit/force push) |
| Final origin/main | == origin/`fix/task8b3-mem01-compact-proposals` == INTEGRATION_TARGET (documentation commit HEAD) |
| `RC1-DEMO-MEM-01` | CLOSED |
| MEM gate verdict | MEM01_REAL_GATE_PASS |
| External manifest / regression | 135/135 PASS · 116 passed (single invocation) |
| Functional files modified | NONE |
| pytest / check_setup / predict / Demo / model | NOT RUN |
| PROP-01 / REF-01 / MASK-01 | UNCHANGED / UNCHANGED / UNCHANGED |
| Report | `docs/task8b3_m1c_mem01_main_integration.md` |
| Next action | Awaiting ChatGPT audit; PROP-01/REF-01/MASK-01 and Task 8B.4 / 8C not started. |

Watt was not needed for Task 8B.3-M1C (no downloads, no transfers).
