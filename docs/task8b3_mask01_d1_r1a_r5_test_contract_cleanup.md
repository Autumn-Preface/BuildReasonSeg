# Task MASK01_D1_R1A_R5_TEST_CONTRACT_CLEANUP

## 1. Git identity

```text
starting branch = fix/task8b3-mask01-success-semantics-r1a-r4-tests
starting head   = 89c58e41d89770a02f0c7b8ff2271be41cd7c86a
task branch     = fix/task8b3-mask01-success-semantics-r1a-r5-tests
final commit sha = POST_COMMIT_EXTERNAL_FACT
```

## 2. Helper cleanup (tests/test_cli_contract.py only)

```text
before = {'_FakeResult': 3, '_success_payload': 2, '_args': 1}
after  = {'_FakeResult': 1, '_success_payload': 1, '_args': 1}
static helper-count sanity check passed = True
```

## 3. Test corrections

```text
single-image SUCCESS assertion = full expected stdout block, line by line (four semantic lines + five artifact lines)
batch test = uses the real _batch_files() discovery path over two real .png files; only predict_one() is monkeypatched
batch assertions = exactly two annotation lines and one `Runtime success: 2`
image suffixes accepted by the product = .tif,.tiff,.png,.jpg,.jpeg,.bmp,.webp
```

## 4. Gates (exactly as published, no -k)

```text
Gate A: tests/test_cli_contract.py::test_r1a_single_success_semantics_block tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary
  exit 1 · 1 failed, 2 passed in 1.35s
Gate B: tests/test_cli_contract.py::test_predict_help_lists_frozen_arguments tests/test_cli_contract.py::test_predict_prompt_required_for_normal_inference tests/test_cli_contract.py::test_predict_missing_image_reports_e201 tests/test_cli_contract.py::test_predict_unsupported_image_type_reports_e203
  exit 0 · 4 passed in 6.51s
```

## 5. Untouched

```text
predict.py / pipeline.py / source_manifest.json / tests/test_task8b_runtime.py / canonical docs = unchanged (True)
inspect-proposals 30-vs-20 = NOT fixed and NOT run · full suite = NOT run
model inference / training / external sync = NOT executed
```

## 6. Disposition

```text
task_status = STOP
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
