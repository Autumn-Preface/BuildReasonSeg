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

# FROM_DSH — Task MASK01_D1_R1E2_R2_REMAINING_FAILURE_DISAMBIGUATION Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1E2_R2_REMAINING_FAILURE_DISAMBIGUATION` |
| Status | **COMPLETE** |
| Starting branch / head | `audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation` / `7251604686da5846d5bbef28048b8a7454bc4181` |
| Task branch | `audit/task8b3-mask01-r1e2-r2-remaining-failure-disambiguation` |
| Nodes audited individually (`-vv`) | 13 |
| Classification counts | {"A": 7, "D": 2, "C": 1, "B": 3} |
| D-classified nodes | ['tests/test_paths_and_package.py::test_required_structure', 'tests/test_setup_checker.py::test_real_project_check_reports_ready'] |
| Pre-MASK base dependency diff | POSSIBLY_CHAIN_RELATED_REQUIRES_REVIEW (base `a98ccecce205`) |
| Required verdict 1 | **MASK01_FULL_SUITE_FAILURES_POSSIBLY_CAUSALLY_RELATED** |
| Required verdict 2 | **FULL_SUITE_GATE_PLACEMENT = EXTERNAL_COMPLETE_RC1** |
| merged commands / cross-node reuse | false / false |
| full suite / repairs / canonical writes / created assets / deps | not run / none / none / none / none |
| Evidence | `evaluation\task8b3_mask01_d1_r1e2_r2_remaining_failure_disambiguation.json` |
| Report | `docs\task8b3_mask01_d1_r1e2_r2_remaining_failure_disambiguation.md` |
| Next gate | `CHATGPT_R1E2_R2_REMOTE_AUDIT` |

Watt was not needed for Task MASK01_D1_R1E2_R2_REMAINING_FAILURE_DISAMBIGUATION (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
