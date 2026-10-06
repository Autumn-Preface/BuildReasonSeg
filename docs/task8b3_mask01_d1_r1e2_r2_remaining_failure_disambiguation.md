# Task MASK01_D1_R1E2_R2_REMAINING_FAILURE_DISAMBIGUATION

## 1. Task record

```text
task_id = MASK01_D1_R1E2_R2_REMAINING_FAILURE_DISAMBIGUATION
status  = COMPLETE
starting branch/head = audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation / 7251604686da5846d5bbef28048b8a7454bc4181
task branch = audit/task8b3-mask01-r1e2-r2-remaining-failure-disambiguation
nodes audited = 13 · execution = one `python -m pytest <NODE_ID> -vv` command per node
merged commands = false · cross-node reason reuse = false
full suite re-executed = false · repairs attempted = false · canonical files modified = false
directories/fixtures/assets created = false · dependencies installed = false · inference = false · external write = false
```

## 2. Per-node result (each from its own traceback only)

| node | exit | classification | decisive failure (this node's own output) |
|---|---:|---|---|
| `test_moving_root_keeps_config_and_model_paths` | 1 | **A** | FAILED tests/test_paths_and_package.py::test_moving_root_keeps_config_and_model_paths - FileNotFoundError: [Errno 2] No such file  |
| `test_required_structure` | 1 | **D** | FAILED tests/test_paths_and_package.py::test_required_structure - AssertionError: assert ['inference/input'] == [] |
| `test_good_fixture_is_ready` | 1 | **A** | ERROR tests/test_setup_checker.py::test_good_fixture_is_ready - FileNotFoundError: [Errno 2] No such file or directory: 'C:\\D\\De |
| `test_hash_mismatch_nonzero` | 1 | **A** | ERROR tests/test_setup_checker.py::test_hash_mismatch_nonzero - FileNotFoundError: [Errno 2] No such file or directory: 'C:\\D\\De |
| `test_invalid_model_yaml_nonzero` | 1 | **A** | ERROR tests/test_setup_checker.py::test_invalid_model_yaml_nonzero - FileNotFoundError: [Errno 2] No such file or directory: 'C:\\ |
| `test_missing_decoder_nonzero` | 1 | **A** | ERROR tests/test_setup_checker.py::test_missing_decoder_nonzero - FileNotFoundError: [Errno 2] No such file or directory: 'C:\\D\\ |
| `test_missing_qwen_component_nonzero` | 1 | **A** | ERROR tests/test_setup_checker.py::test_missing_qwen_component_nonzero - FileNotFoundError: [Errno 2] No such file or directory: ' |
| `test_missing_sam2_nonzero` | 1 | **A** | ERROR tests/test_setup_checker.py::test_missing_sam2_nonzero - FileNotFoundError: [Errno 2] No such file or directory: 'C:\\D\\Dee |
| `test_real_project_check_reports_ready` | 1 | **D** | FAILED tests/test_setup_checker.py::test_real_project_check_reports_ready - AssertionError: [OK] Python �� 3.11.16 (C:\D\DeepSeekH |
| `test_real_runtime_reports_ultralytics_and_ready` | 1 | **C** | FAILED tests/test_setup_checker.py::test_real_runtime_reports_ultralytics_and_ready - AssertionError: [OK] Python �� 3.11.16 (C:\D |
| `test_fixture_frozen_and_complete` | 1 | **B** | FAILED tests/test_task8b1_fallback_ux.py::test_fixture_frozen_and_complete - FileNotFoundError: [Errno 2] No such file or director |
| `test_no_prompt_to_program_table_in_delivery_code` | 1 | **B** | FAILED tests/test_task8b1_fallback_ux.py::test_no_prompt_to_program_table_in_delivery_code - FileNotFoundError: [Errno 2] No such  |
| `test_task8b_paraphrases_are_verbatim` | 1 | **B** | FAILED tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim - FileNotFoundError: [Errno 2] No such file or dire |

```text
classification_counts = {"A": 7, "D": 2, "C": 1, "B": 3}
d_classified_nodes = ['tests/test_paths_and_package.py::test_required_structure', 'tests/test_setup_checker.py::test_real_project_check_reports_ready']
```

## 3. Pre-MASK base dependency-path diff (only if D appeared)

```text
base = a98ccecce20585dc37520523cd32b49b0a684248 · performed = True
changed paths a98ccec..HEAD = ['delivery_src/BuildReasonSeg_Advisor_RC1/README.md', 'delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py', 'delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md', 'delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md', 'delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md', 'delivery_src/BuildReasonSeg_Advisor_RC1/predict.py', 'delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json', 'delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py', 'delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py', 'docs/rc1_inspect01_preflight_order_fix.md', 'docs/task8b3_mask01_d1_r1a_r1_cli_contract_correction.md', 'docs/task8b3_mask01_d1_r1a_r2_executable_cli_fix.md', 'docs/task8b3_mask01_d1_r1a_r3_product_suffix_only.md', 'docs/task8b3_mask01_d1_r1a_r4_tests_only.md', 'docs/task8b3_mask01_d1_r1a_r5_test_contract_cleanup.md', 'docs/task8b3_mask01_d1_r1a_r6_final_test_fix.md', 'docs/task8b3_mask01_d1_r1a_state_persistence.md', 'docs/task8b3_mask01_d1_r1b_canonical_docs.md', 'docs/task8b3_mask01_d1_r1b_r1_canonical_docs_correction.md', 'docs/task8b3_mask01_d1_r1b_r2_exact_doc_replacement.md', 'docs/task8b3_mask01_d1_r1c_runtime_contract_test_cleanup.md', 'docs/task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.md', 'docs/task8b3_mask01_d1_r1d_source_manifest_canonicalization.md', 'docs/task8b3_mask01_d1_r1e1_targeted_canonical_gates.md', 'docs/task8b3_mask01_d1_r1e2_full_suite_failure_triage.md', 'docs/task8b3_mask01_d1_r1e2_r1_asset_regression_disambiguation.md', 'docs/task8b3_mask01_d1_r1e_canonical_regression_closure.md', 'docs/task8b3_mask01_d1_success_semantics_hardening.md', 'evaluation/rc1_inspect01_preflight_order_fix.json', 'evaluation/task8b3_mask01_d1_r1a_r1_cli_contract_correction.json', 'evaluation/task8b3_mask01_d1_r1a_r2_executable_cli_fix.json', 'evaluation/task8b3_mask01_d1_r1a_r3_product_suffix_only.json', 'evaluation/task8b3_mask01_d1_r1a_r4_tests_only.json', 'evaluation/task8b3_mask01_d1_r1a_r5_test_contract_cleanup.json', 'evaluation/task8b3_mask01_d1_r1a_r6_final_test_fix.json', 'evaluation/task8b3_mask01_d1_r1a_state_persistence.json', 'evaluation/task8b3_mask01_d1_r1b_canonical_docs.json', 'evaluation/task8b3_mask01_d1_r1b_r1_canonical_docs_correction.json', 'evaluation/task8b3_mask01_d1_r1b_r2_exact_doc_replacement.json', 'evaluation/task8b3_mask01_d1_r1c_runtime_contract_test_cleanup.json', 'evaluation/task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.json', 'evaluation/task8b3_mask01_d1_r1d_source_manifest_canonicalization.json', 'evaluation/task8b3_mask01_d1_r1e1_targeted_canonical_gates.json', 'evaluation/task8b3_mask01_d1_r1e2_full_suite_failure_triage.json', 'evaluation/task8b3_mask01_d1_r1e2_r1_asset_regression_disambiguation.json', 'evaluation/task8b3_mask01_d1_r1e_canonical_regression_closure.json', 'evaluation/task8b3_mask01_d1_success_semantics_hardening.json', 'handoff/FROM_DSH.md', 'handoff/TO_DSH.md']
chain-relevant paths = ['delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md']
verdict = POSSIBLY_CHAIN_RELATED_REQUIRES_REVIEW
```

## 4. Required verdicts

```text
MASK01_FULL_SUITE_FAILURES_POSSIBLY_CAUSALLY_RELATED
FULL_SUITE_GATE_PLACEMENT = EXTERNAL_COMPLETE_RC1
```

## 5. Persistence

```text
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_R1E2_R2_REMOTE_AUDIT
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
