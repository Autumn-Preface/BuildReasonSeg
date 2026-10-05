# Task MASK01_D1_R1A_STATE_PERSISTENCE — R1-A CLI work state

## 1. Purpose

Persist the real R1-A work state on `fix/task8b3-mask01-success-semantics-r1a` even though the R1-A gate ended in STOP. No product logic was changed and
the pre-existing `--inspect-proposals` failure was not touched.

## 2. Persisted implementation state

```text
predict.py
  single-image annotation present = True
  note line present               = True
  batch annotation present        = True
  sha256                          = e45e868ab0236960fd401a42692db0d494723999a36b678a7f1023210ecc1487
tests/test_cli_contract.py
  weak D1 test removed            = True
  tests present                   = ['test_cli_batch_success_prints_runtime_only_annotation', 'test_cli_placeholders_return_not_implemented', 'test_cli_single_image_success_prints_runtime_only_annotation', 'test_each_cli_help_works', 'test_error_registry_complete', 'test_evaluate_split_and_reference_defaults', 'test_evaluate_test_requires_explicit_split', 'test_predict_defaults_exact', 'test_predict_help_lists_frozen_arguments', 'test_predict_image_and_input_dir_are_exclusive', 'test_predict_inspect_proposals_does_not_require_prompt', 'test_predict_missing_image_reports_e201', 'test_predict_no_save_diagnostics_flag', 'test_predict_prompt_required_for_normal_inference', 'test_predict_unsupported_image_type_reports_e203', 'test_prepare_dataset_format_enum', 'test_success_status_and_ok_semantics_unchanged', 'test_train_enum_constraints', 'test_train_missing_dataset_reports_e101']
  sha256                          = 43729375e9f3631bad16af1b20607959ca57cb1ae97f055b5959df57f8ce2083
```

## 3. R1-A gate (unchanged, recorded verbatim)

```text
python -m pytest tests/test_cli_contract.py -q  →  exit 1
1 failed, 22 passed in 14.28s
FAILED tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt
  assert completed.returncode == 20 and 'E202' in completed.stderr -> assert (30 == 20)
attribution = PRE_EXISTING_INSPECT_PROPOSALS_EXIT_CODE_DEFECT_NOT_INTRODUCED_BY_R1A
```

## 4. Unchanged files verified

```text
pipeline.py / test_task8b_runtime.py / source_manifest.json / canonical docs = unchanged
model inference / training / external sync / full suite = NOT executed
next task entered = NO
```

## 5. Disposition

```text
task_status = STOP (R1-A gate not green; state persisted for independent ChatGPT review)
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
