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

# FROM_DSH — Task MASK01_D1_R1A_R6_FINAL_TEST_FIX Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1A_R6_FINAL_TEST_FIX` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1a-r5-tests` / `3f7bfb0956de14284a77dbf1ae19ec9d7977a56f` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1a-r6-tests` |
| Only file modified | `tests/test_cli_contract.py` |
| batch input_dir is a Path | True |
| `Failed : 0` assertion added | True |
| synthetic artifact values unified | True (mask.png / overlay.png / diag / 7 / 123) |
| Helper counts | `{'_FakeResult': 1, '_success_payload': 1, '_args': 1}` |
| Gate A | exit 0 · 3 passed in 1.36s |
| Gate B | exit 0 · 4 passed in 6.33s |
| `-k` / full suite / inspect node | NOT used / NOT run / NOT run |
| predict.py / pipeline / manifest / docs | UNCHANGED |
| Model inference / training / external sync | NONE |
| Evidence | `evaluation\task8b3_mask01_d1_r1a_r6_final_test_fix.json` |
| Report | `docs\task8b3_mask01_d1_r1a_r6_final_test_fix.md` |
| Next action | Awaiting ChatGPT remote review; next task not entered |

Watt was not needed for Task MASK01_D1_R1A_R6_FINAL_TEST_FIX (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
