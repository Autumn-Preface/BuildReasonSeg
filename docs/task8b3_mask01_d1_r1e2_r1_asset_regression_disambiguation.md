# Task MASK01_D1_R1E2_R1_ASSET_REGRESSION_DISAMBIGUATION

## 1. Task record

```text
task_id = MASK01_D1_R1E2_R1_ASSET_REGRESSION_DISAMBIGUATION
status  = COMPLETE
starting branch/head = audit/task8b3-mask01-r1e2-full-suite-failure-triage / 7a4e5454cdaf164c8d1706412c55b869074748e0
task branch = audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation
nodes audited = 9 · execution = one `python -m pytest <NODE_ID> -vv` command per node
merged commands = false · pytest tokens treated as nodes = false · cross-node reason reuse = false
full suite re-executed = false · repairs attempted = false · canonical files modified = false
assets created/downloaded = false · dependencies installed = false · model inference = false · external write = false
```

## 2. Per-node result (each from its own decisive failure)

| node | exit | classification | decisive failure |
|---|---:|---|---|
| `test_normal_path_is_qwen_first` | 1 | **D** | assert False is True |
| `test_components_complete` | 1 | **E** | FAILED tests/test_model_package.py::test_components_complete - buildreasonseg.errors.BuildReasonSegError: E302 |
| `test_decoder_hash_and_bytes_exact` | 1 | **D** | FAILED tests/test_model_package.py::test_decoder_hash_and_bytes_exact - AssertionError: assert False |
| `test_detector_hash_and_bytes_exact` | 1 | **D** | FAILED tests/test_model_package.py::test_detector_hash_and_bytes_exact - AssertionError: assert False |
| `test_hash_mismatch_raises` | 1 | **E** | FAILED tests/test_model_package.py::test_hash_mismatch_raises - buildreasonseg.errors.BuildReasonSegError: E302 |
| `test_metadata_hashes_match_copied_assets` | 1 | **D** | assert False |
| `test_model_fallback_requires_user_confirmation` | 1 | **D** | FAILED tests/test_model_package.py::test_model_fallback_requires_user_confirmation - assert False |
| `test_model_yaml_parses` | 1 | **E** | FAILED tests/test_model_package.py::test_model_yaml_parses - buildreasonseg.errors.BuildReasonSegError: E302 |
| `test_package_verification_matches` | 1 | **E** | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegError: E302 |

```text
classification_counts = {"D": 5, "E": 4}
PASS = TEST_PASSES_WHEN_RUN_INDIVIDUALLY · A = missing full-delivery asset · B = missing non-canonical fixture
C = runtime dependency/environment · D = true canonical code regression · E = other
true regressions = ['tests/test_language_contract.py::test_normal_path_is_qwen_first', 'tests/test_model_package.py::test_decoder_hash_and_bytes_exact', 'tests/test_model_package.py::test_detector_hash_and_bytes_exact', 'tests/test_model_package.py::test_metadata_hashes_match_copied_assets', 'tests/test_model_package.py::test_model_fallback_requires_user_confirmation']
```

## 3. Verdict on the previous D mislabelling

```text
verdict = TRUE_CANONICAL_CODE_REGRESSION_PRESENT
recommendation = keep the complete-suite gate on the canonical source tree; triage the true regressions
```

The previous triage labelled these nodes D by classifying whole-group output; with per-node execution their actual
causes are the classifications above, so the earlier `VALID_AND_HAS_TRUE_REGRESSION` decision is superseded by this
per-node evidence.

## 4. Persistence

```text
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_R1E2_R1_REMOTE_AUDIT
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
