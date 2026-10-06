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

# FROM_DSH — Task MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE` |
| Status | **STOP** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1d2` / `78b749e7a5269d9d22e7b495193a79e011df87f0` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1e` |
| Gate exits | Gate A=1 · Gate B=1 · Gate C=1 · Gate D=1 |
| Gate D | complete `python -m pytest -q` executed, no `-k` / no `--lf` / nothing skipped |
| Failing nodes | {"A": ["at", "tests/test_language_contract.py::test_normal_path_is_qwen_first", "tests/test_model_package.py::test_components_complete", "tests/test_model_package.py::test_decoder_hash_and_bytes_exact", "tests/test_model_package.py::test_detector_hash_and_bytes_exact", "tests/test_model_package.py::test_hash_mismatch_raises"], "B": ["at", "tests/test_language_contract.py::test_normal_path_is_qwen_first", "tests/test_model_package.py::test_components_complete", "tests/test_model_package.py::test_decoder_hash_and_bytes_exact", "tests/test_model_package.py::test_detector_hash_and_bytes_exact", "tests/test_model_package.py::test_hash_mismatch_raises"], "C": ["at", "tests/test_language_contract.py::test_normal_path_is_qwen_first", "tests/test_model_package.py::test_components_complete", "tests/test_model_package.py::test_decoder_hash_and_bytes_exact", "tests/test_model_package.py::test_detector_hash_and_bytes_exact", "tests/test_model_package.py::test_hash_mismatch_raises"], "D": ["at", "tests/test_language_contract.py::test_normal_path_is_qwen_first", "tests/test_model_package.py::test_components_complete", "tests/test_model_package.py::test_decoder_hash_and_bytes_exact", "tests/test_model_package.py::test_detector_hash_and_bytes_exact", "tests/test_model_package.py::test_hash_mismatch_raises"]} |
| Canonical writes (product/tests/manifest/docs) | NONE |
| Repairs attempted | NONE |
| chain status | NOT_CLOSED |
| chain status | OPEN |
| Evidence | `evaluation\task8b3_mask01_d1_r1e_canonical_regression_closure.json` |
| Report | `docs\task8b3_mask01_d1_r1e_canonical_regression_closure.md` |
| Next gate | `CHATGPT_R1E_FAILURE_TRIAGE` (D2 not entered) |

Watt was not needed for Task MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
