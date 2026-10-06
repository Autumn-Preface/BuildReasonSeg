# Task MASK01_D1_R1A_R2_EXECUTABLE_CLI_FIX

## 1. Git identity

```text
starting branch = fix/task8b3-mask01-success-semantics-r1a-r1
starting head   = 2fe6369d7def85b59a16c28af4340eb61f24437c
task branch     = fix/task8b3-mask01-success-semantics-r1a-r2
final commit sha = POST_COMMIT_EXTERNAL_FACT
```

## 2. Restored executable contract

```text
successes += 1 present           = True
batch success annotation present = False
batch summary                    = Runtime success: <successes>
single-image block               = Result / Validity / Semantic / Note (exactly one Note line)
batch function exercised         = _run_batch
```

## 3. Gate A (exactly as published, no -k)

```text
tests/test_cli_contract.py::test_r1a_single_success_semantics_block tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary
exit 1 · 2 failed, 1 passed in 1.55s
```

## 4. Gate B (exactly as published, no -k)

```text
tests/test_cli_contract.py::test_predict_help_lists_frozen_arguments tests/test_cli_contract.py::test_predict_prompt_required_for_normal_inference tests/test_cli_contract.py::test_predict_missing_image_reports_e201 tests/test_cli_contract.py::test_predict_unsupported_image_type_reports_e203
exit 0 · 4 passed in 6.62s
```

## 5. Untouched files

```text
pipeline.py / source_manifest.json / tests/test_task8b_runtime.py / canonical docs = unchanged (True)
inspect-proposals 30-vs-20 defect = NOT fixed and NOT run
full suite = NOT run · model inference / training / external sync = NOT executed
```

## 6. Disposition

```text
task_status = STOP
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
