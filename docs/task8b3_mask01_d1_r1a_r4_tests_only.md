# Task MASK01_D1_R1A_R4_TESTS_ONLY

## 1. Git identity

```text
starting branch = fix/task8b3-mask01-success-semantics-r1a-r3-product
starting head   = 37b9afe0bc674630c4f9f5040e0035ceeb809fc1
task branch     = fix/task8b3-mask01-success-semantics-r1a-r4-tests
final commit sha = POST_COMMIT_EXTERNAL_FACT
```

## 2. What was corrected (tests only)

```text
file = tests/test_cli_contract.py  (only file modified by this task)
_report_single(result, args)                    → fake result carries ok + full success payload
_run_batch(runtime, args, package_name, parsed, language_info, started) → six positional args supplied
_batch_files(input_dir)                         → monkeypatched for the batch behaviour test
fake payload keys = status, validity_scope, semantic_status, semantic_note, output_paths, reference_id, mask_area
failure case writes error lines to stderr and keeps only `Result       : FAILED` on stdout
```

## 3. Gate A (exactly as published, no -k)

```text
tests/test_cli_contract.py::test_r1a_single_success_semantics_block tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary
exit 0 · 3 passed in 1.43s
```

## 4. Gate B (exactly as published, no -k)

```text
tests/test_cli_contract.py::test_predict_help_lists_frozen_arguments tests/test_cli_contract.py::test_predict_prompt_required_for_normal_inference tests/test_cli_contract.py::test_predict_missing_image_reports_e201 tests/test_cli_contract.py::test_predict_unsupported_image_type_reports_e203
exit 0 · 4 passed in 6.61s
```

## 5. Untouched

```text
predict.py / pipeline.py / source_manifest.json / tests/test_task8b_runtime.py / canonical docs = unchanged (True)
inspect-proposals 30-vs-20 defect = NOT fixed and NOT run · full suite = NOT run
model inference / training / external sync = NOT executed
```

## 6. Disposition

```text
task_status = COMPLETE
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
