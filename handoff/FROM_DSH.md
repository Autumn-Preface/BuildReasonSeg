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

# FROM_DSH — Task MASK01_D1_R1E2_R1_ASSET_REGRESSION_DISAMBIGUATION Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1E2_R1_ASSET_REGRESSION_DISAMBIGUATION` |
| Status | **COMPLETE** |
| Starting branch / head | `audit/task8b3-mask01-r1e2-full-suite-failure-triage` / `7a4e5454cdaf164c8d1706412c55b869074748e0` |
| Task branch | `audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation` |
| Nodes audited (individually, `-vv`) | 9 |
| Classification counts | {"D": 5, "E": 4} |
| True canonical code regressions | ['tests/test_language_contract.py::test_normal_path_is_qwen_first', 'tests/test_model_package.py::test_decoder_hash_and_bytes_exact', 'tests/test_model_package.py::test_detector_hash_and_bytes_exact', 'tests/test_model_package.py::test_metadata_hashes_match_copied_assets', 'tests/test_model_package.py::test_model_fallback_requires_user_confirmation'] |
| Verdict | **TRUE_CANONICAL_CODE_REGRESSION_PRESENT** |
| Recommendation | keep the complete-suite gate on the canonical source tree; triage the true regressions |
| merged commands / token-as-node / cross-node reuse | false / false / false |
| full suite / repairs / canonical writes / assets / deps | not run / none / none / none / none |
| Evidence | `evaluation\task8b3_mask01_d1_r1e2_r1_asset_regression_disambiguation.json` |
| Report | `docs\task8b3_mask01_d1_r1e2_r1_asset_regression_disambiguation.md` |
| Next gate | `CHATGPT_R1E2_R1_REMOTE_AUDIT` |

Watt was not needed for Task MASK01_D1_R1E2_R1_ASSET_REGRESSION_DISAMBIGUATION (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
