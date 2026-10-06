# Task MASK01_D1_R1A_R6_FINAL_TEST_FIX

## 1. Git identity

```text
starting branch = fix/task8b3-mask01-success-semantics-r1a-r5-tests
starting head   = 3f7bfb0956de14284a77dbf1ae19ec9d7977a56f
task branch     = fix/task8b3-mask01-success-semantics-r1a-r6-tests
final commit sha = POST_COMMIT_EXTERNAL_FACT
```

## 2. The three prescribed fixes (tests/test_cli_contract.py only)

```text
{
 "artifact_values_unified": true,
 "input_dir_is_path": true,
 "failed_zero_assertion": true
}
synthetic success values = mask.png / overlay.png / diag / mask_area 7 / reference_id 123
batch test input_dir = Path (tmp_path), only predict_one() monkeypatched
batch assertions = 2 x `SUCCESS [runtime-only; semantic=NOT_EVALUATED]` · `Runtime success: 2` · `Failed : 0`
helper counts = {'_FakeResult': 1, '_success_payload': 1, '_args': 1}
```

## 3. Gates (exactly as published, no -k)

```text
Gate A: tests/test_cli_contract.py::test_r1a_single_success_semantics_block tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary
  exit 0 · 3 passed in 1.36s
Gate B: tests/test_cli_contract.py::test_predict_help_lists_frozen_arguments tests/test_cli_contract.py::test_predict_prompt_required_for_normal_inference tests/test_cli_contract.py::test_predict_missing_image_reports_e201 tests/test_cli_contract.py::test_predict_unsupported_image_type_reports_e203
  exit 0 · 4 passed in 6.33s
```

## 4. Untouched

```text
predict.py / pipeline.py / source_manifest.json / tests/test_task8b_runtime.py / canonical docs = unchanged (True)
inspect-proposals 30-vs-20 = NOT fixed and NOT run · full suite = NOT run
model inference / training / external sync = NOT executed
```

## 5. Disposition

```text
task_status = COMPLETE
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
