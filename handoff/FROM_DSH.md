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

# FROM_DSH — Task MASK01_D1_R1A_R4_TESTS_ONLY Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1A_R4_TESTS_ONLY` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1a-r3-product` / `37b9afe0bc674630c4f9f5040e0035ceeb809fc1` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1a-r4-tests` |
| Only file modified | `tests/test_cli_contract.py` |
| Real contracts used | `_report_single(result, args)` · `_run_batch(runtime, args, package_name, parsed, language_info, started)` · `_batch_files(input_dir)` |
| Gate A | exit 0 · 3 passed in 1.43s |
| Gate B | exit 0 · 4 passed in 6.61s |
| `-k` substitution / full suite / inspect node | NOT used / NOT run / NOT run |
| predict.py / pipeline / manifest / docs | UNCHANGED |
| Model inference / training / external sync | NONE |
| Evidence | `evaluation\task8b3_mask01_d1_r1a_r4_tests_only.json` |
| Report | `docs\task8b3_mask01_d1_r1a_r4_tests_only.md` |
| Next action | Awaiting ChatGPT remote review; next task not entered |

Watt was not needed for Task MASK01_D1_R1A_R4_TESTS_ONLY (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
