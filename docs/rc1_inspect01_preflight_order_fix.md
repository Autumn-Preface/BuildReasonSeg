# Task RC1_INSPECT01_PREFLIGHT_ORDER_FIX

## 1. Task record

```text
task_id = RC1_INSPECT01_PREFLIGHT_ORDER_FIX
status  = COMPLETE
starting branch/head = fix/task8b3-mask01-success-semantics-r1d / 5808e64ed43fd383be9e08bf623a9bfdc27f62b4
task branch = fix/rc1-inspect-proposals-preflight-order
design_id = INSPECT_IMAGE_PREFLIGHT_BEFORE_MODEL_V1
```

## 2. Fix

```text
file = delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
added import : from buildreasonseg.runtime.imageio import load_image
added block  : if args.inspect_proposals: load_image(args.image)   (before _resolve_and_report / PredictRuntime)
normal inference ordering = unchanged
contract     : --inspect-proposals + unreadable image -> E202 IMAGE_UNREADABLE -> exit 20
```

## 3. Regression test

```text
tests/test_cli_contract.py::test_inspect_unreadable_image_precedes_model_resolution
asserts exit 20, E202 on stderr, and that no model/device resolution output precedes the image error
```

## 4. Gates (no full suite, no -k substitution)

```text
Gate A: tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt tests/test_cli_contract.py::test_inspect_unreadable_image_precedes_model_resolution
  exit 0 · 2 passed in 3.24s
Gate B: tests/test_cli_contract.py::test_predict_help_lists_frozen_arguments tests/test_cli_contract.py::test_predict_prompt_required_for_normal_inference tests/test_cli_contract.py::test_predict_missing_image_reports_e201 tests/test_cli_contract.py::test_predict_unsupported_image_type_reports_e203
  exit 0 · 4 passed in 6.12s
```

## 5. Untouched

```text
runtime pipeline / imageio / detector / manifest / canonical docs = UNCHANGED (True)
model inference / training / external sync = NOT executed
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_INSPECT01_REMOTE_AUDIT
```
