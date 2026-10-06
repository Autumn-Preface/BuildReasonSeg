# Task MASK01_D1_R1E2_FULL_SUITE_FAILURE_TRIAGE

## 1. Task record

```text
task_id = MASK01_D1_R1E2_FULL_SUITE_FAILURE_TRIAGE
status  = COMPLETE
starting branch/head = fix/task8b3-mask01-success-semantics-r1e1 / 3320212a6992316196bc5b2cfacaeaf027fd070e
task branch = audit/task8b3-mask01-r1e2-full-suite-failure-triage
full suite re-executed = false · repairs attempted = false · canonical tracked files modified = false
assets created = false · dependencies installed = false · model inference = false · external write = false
```

## 2. Inventory check (existence in the canonical source tree + manifest listing)

| path | exists in canonical tree | manifest_listed |
|---|---|---|
| `model/buildreasonseg_advisor/detector.pt` | False | False |
| `model/buildreasonseg_advisor/decoder.pt` | False | False |
| `model/buildreasonseg_advisor/model.yaml` | True | True |
| `model/buildreasonseg_advisor/metadata.json` | True | True |
| `model/components/sam2/sam2.1_hiera_base_plus.pt` | False | False |
| `model/components/sam2/sam2.1_hiera_b+.yaml` | False | False |
| `model/components/program_head/program_parser_l3_rehearsal_v1.pt` | False | False |
| `model/components/program_head/Qwen3-VL-2B-Instruct` | False | False |
| `model/components/program_head/qwen_asset_manifest.json` | True | True |
| `check_setup.py` | True | True |
| `predict.py` | True | True |
| `configs/inference.yaml` | True | True |
| `requirements.txt` | True | True |
| `environment.yml` | True | True |
| `logs` | True | False |
| `inference/output` | True | False |
| `artifacts` | False | False |

```text
missing canonical assets = ['model/buildreasonseg_advisor/detector.pt', 'model/buildreasonseg_advisor/decoder.pt', 'model/components/sam2/sam2.1_hiera_base_plus.pt', 'model/components/sam2/sam2.1_hiera_b+.yaml', 'model/components/program_head/program_parser_l3_rehearsal_v1.pt', 'model/components/program_head/Qwen3-VL-2B-Instruct', 'artifacts']
```

## 3. The five previously failing groups

| group | title | exit | summary | failing nodes |
|---|---|---:|---|---:|
| 1 | 6. TRIAGE GROUP 1 — LANGUAGE AVAILABILITY | 1 | ============================== 1 failed in 0.06s ============================== | 2 |
| 2 | 7. TRIAGE GROUP 2 — MODEL PACKAGE | 1 | ============================== 8 failed in 0.24s ============================== | 10 |
| 3 | 8. TRIAGE GROUP 3 — PATHS / DELIVERY STRUCTURE | 1 | ============================== 2 failed in 0.17s ============================== | 4 |
| 4 | 9. TRIAGE GROUP 4 — SETUP CHECKER | 1 | ========================= 2 failed, 6 errors in 5.69s ========================= | 11 |
| 5 | 10. TRIAGE GROUP 5 — TASK 8B.1 FIXTURES | 1 | ============================== 3 failed in 1.40s ============================== | 5 |

## 4. Per-node classification

| group | node | classification | real error |
|---|---:|---|---|
| 1 | `[100%]` | D | assert False is True |
| 1 | `test_normal_path_is_qwen_first` | D | assert False is True |
| 2 | `[` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `[100%]` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `test_components_complete` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `test_decoder_hash_and_bytes_exact` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `test_detector_hash_and_bytes_exact` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `test_hash_mismatch_raises` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `test_metadata_hashes_match_copied_assets` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `test_model_fallback_requires_user_confirmation` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `test_model_yaml_parses` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 2 | `test_package_verification_matches` | D | FAILED tests/test_model_package.py::test_package_verification_matches - buildreasonseg.errors.BuildReasonSegEr |
| 3 | `[` | A | FAILED tests/test_paths_and_package.py::test_required_structure - AssertionError: assert ['inference/input'] = |
| 3 | `[100%]` | A | FAILED tests/test_paths_and_package.py::test_required_structure - AssertionError: assert ['inference/input'] = |
| 3 | `test_moving_root_keeps_config_and_model_paths` | A | FAILED tests/test_paths_and_package.py::test_required_structure - AssertionError: assert ['inference/input'] = |
| 3 | `test_required_structure` | A | FAILED tests/test_paths_and_package.py::test_required_structure - AssertionError: assert ['inference/input'] = |
| 4 | `[` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `[100%]` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `at` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `test_good_fixture_is_ready` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `test_hash_mismatch_nonzero` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `test_invalid_model_yaml_nonzero` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `test_missing_decoder_nonzero` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `test_missing_qwen_component_nonzero` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `test_missing_sam2_nonzero` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `test_real_project_check_reports_ready` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 4 | `test_real_runtime_reports_ultralytics_and_ready` | A | ========================= 2 failed, 6 errors in 5.69s ========================= |
| 5 | `[` | B | FAILED tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim - FileNotFoundError: [Errno 2]  |
| 5 | `[100%]` | B | FAILED tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim - FileNotFoundError: [Errno 2]  |
| 5 | `test_fixture_frozen_and_complete` | B | FAILED tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim - FileNotFoundError: [Errno 2]  |
| 5 | `test_no_prompt_to_program_table_in_delivery_code` | B | FAILED tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim - FileNotFoundError: [Errno 2]  |
| 5 | `test_task8b_paraphrases_are_verbatim` | B | FAILED tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim - FileNotFoundError: [Errno 2]  |

```text
classification_counts = {"D": 12, "A": 15, "B": 5}
A = MISSING_FULL_DELIVERY_ASSET · B = MISSING_NONCANONICAL_TEST_FIXTURE · C = RUNTIME_DEPENDENCY_ENVIRONMENT
D = TRUE_CANONICAL_CODE_REGRESSION · E = OTHER
```

## 5. Validity decision

```text
FULL_SUITE_CANONICAL_GATE = VALID_AND_HAS_TRUE_REGRESSION
recommendation = keep the complete-suite gate on the canonical source tree
```

See the classification counts above; the canonical tree retains the complete-suite gate.

## 6. Persistence

```text
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_R1E2_TRUE_REGRESSION_TRIAGE
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
