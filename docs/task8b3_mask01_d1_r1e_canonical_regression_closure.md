# Task MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE

## 1. Task record

```text
task_id = MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE
status  = STOP
starting branch/head = fix/task8b3-mask01-success-semantics-r1d2 / 78b749e7a5269d9d22e7b495193a79e011df87f0
task branch = fix/task8b3-mask01-success-semantics-r1e
report-only task: no canonical product source, tests, manifest or docs were modified
```

## 2. Gates executed in order

| gate | exit | summary | failing nodes (first 6) |
|---|---:|---|---|
| A | 1 | 16 failed, 108 passed, 6 errors in 28.22s | `at`, `tests/test_language_contract.py::test_normal_path_is_qwen_first`, `tests/test_model_package.py::test_components_complete`, `tests/test_model_package.py::test_decoder_hash_and_bytes_exact`, `tests/test_model_package.py::test_detector_hash_and_bytes_exact`, `tests/test_model_package.py::test_hash_mismatch_raises` |
| B | 1 | 16 failed, 108 passed, 6 errors in 28.05s | `at`, `tests/test_language_contract.py::test_normal_path_is_qwen_first`, `tests/test_model_package.py::test_components_complete`, `tests/test_model_package.py::test_decoder_hash_and_bytes_exact`, `tests/test_model_package.py::test_detector_hash_and_bytes_exact`, `tests/test_model_package.py::test_hash_mismatch_raises` |
| C | 1 | 16 failed, 108 passed, 6 errors in 28.11s | `at`, `tests/test_language_contract.py::test_normal_path_is_qwen_first`, `tests/test_model_package.py::test_components_complete`, `tests/test_model_package.py::test_decoder_hash_and_bytes_exact`, `tests/test_model_package.py::test_detector_hash_and_bytes_exact`, `tests/test_model_package.py::test_hash_mismatch_raises` |
| D | 1 | 16 failed, 108 passed, 6 errors in 28.17s | `at`, `tests/test_language_contract.py::test_normal_path_is_qwen_first`, `tests/test_model_package.py::test_components_complete`, `tests/test_model_package.py::test_decoder_hash_and_bytes_exact`, `tests/test_model_package.py::test_detector_hash_and_bytes_exact`, `tests/test_model_package.py::test_hash_mismatch_raises` |

```text
Gate D executed the complete canonical suite: python -m pytest -q (no -k, no --lf, nothing skipped, no repair while running)
all_gates_passed = False
```

## 3. Outcome

```text
SUCCESS_SEMANTICS_HARDENING_V1 = NOT_CLOSED (gate failure recorded)
MASK01_ENGINEERING_HARDENING_CHAIN = OPEN (see failing nodes above)
repairs attempted = NONE (failures recorded only, as instructed)
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_R1E_FAILURE_TRIAGE
```
