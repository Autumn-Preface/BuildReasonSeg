# TO_DSH — MASK01_D1_R1A_R5_TEST_CONTRACT_CLEANUP

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1a-r4-tests`
> Required starting HEAD: `89c58e41d89770a02f0c7b8ff2271be41cd7c86a`
> New task branch: `fix/task8b3-mask01-success-semantics-r1a-r5-tests`

## 0. PURPOSE

This is a tests-only cleanup/correction task.

ChatGPT independently audited R1-A-R4.

Accepted facts:

```text
Gate A = 3 passed
Gate B = 4 passed
predict.py unchanged
pipeline.py unchanged
inspect-proposals issue untouched
```

However, R1-A-R4 tests are NOT accepted yet because the committed test file still contains:

```text
3 x _FakeResult definitions
2 x _success_payload definitions
```

and the batch test still monkeypatches `_batch_files`, so it does not verify the real file enumeration path required by the task.

The single-success test also does not yet verify the existing Mask / Overlay / Diagnostics / Reference ID / Mask area output lines.

This task fixes only those test-contract issues.

Do NOT modify product code.

---

## 1. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1a-r4-tests

HEAD =
89c58e41d89770a02f0c7b8ff2271be41cd7c86a
```

Allowed initial worktree:

```text
clean
```

or only:

```text
M handoff/TO_DSH.md
```

Any other pre-existing mutation:
- record it;
- do not delete/reset/restore/clean;
- if it overlaps this task, STOP implementation.

Create:

```text
fix/task8b3-mask01-success-semantics-r1a-r5-tests
```

No reset/rebase/amend/stash/clean/force-push.

---

## 2. ALLOWED PATHS

Only modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
```

plus task records:

```text
docs/task8b3_mask01_d1_r1a_r5_test_contract_cleanup.md
evaluation/task8b3_mask01_d1_r1a_r5_test_contract_cleanup.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Explicitly forbidden:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
canonical README/docs
configs
models
weights
external RC1
```

---

## 3. REMOVE DUPLICATE TEST HELPERS

In:

```text
tests/test_cli_contract.py
```

after `test_error_registry_complete`, there must be exactly:

```text
1 x class _FakeResult
1 x def _success_payload
1 x def _args
```

Delete all redundant earlier duplicate definitions.

Keep one `_FakeResult` equivalent to:

```python
class _FakeResult:
    """Minimal stand-in for a PipelineResult as consumed by predict.py."""

    def __init__(self, payload, ok=True, error_code=None, error_reason=None):
        self.result_payload = payload
        self.ok = ok
        self.error_code = error_code
        self.error_reason = error_reason
```

Keep one `_success_payload()` returning a complete normal-success payload:

```python
def _success_payload():
    from buildreasonseg.runtime.pipeline import success_semantics

    return {
        "status": "SUCCESS",
        **success_semantics(),
        "output_paths": {
            "mask": "mask.png",
            "overlay": "overlay.png",
            "diagnostics": "diag",
        },
        "reference_id": 7,
        "mask_area": 123,
    }
```

Keep one small `_args(**overrides)` helper if useful.

Do not add source-code string scanning.

---

## 4. SINGLE SUCCESS TEST — COMPLETE ASSERTIONS

Test name:

```text
test_r1a_single_success_semantics_block
```

Call real:

```python
predict._report_single(_FakeResult(_success_payload()), _args())
```

Capture stdout.

Require exactly once:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

Also require presence of:

```text
Mask         : mask.png
Overlay      : overlay.png
Diagnostics  : diag
Reference ID : 7
Mask area    : 123
```

Assert:

```python
out.count("Result       : SUCCESS") == 1
out.count("Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.") == 1
```

No try/except.

---

## 5. SINGLE FAILURE TEST

Test name:

```text
test_r1a_single_failure_omits_success_semantics
```

Keep direct call to `_report_single()` with fake FAILED result.

No try/except.

Assert stdout contains exactly one:

```text
Result       : FAILED
```

and stdout does NOT contain:

```text
Validity     :
Semantic     :
Note         :
```

Do not overconstrain localized stderr wording.

---

## 6. BATCH TEST — USE REAL `_batch_files()`

Test name:

```text
test_r1a_batch_success_annotation_and_runtime_summary
```

This test must use the real `_batch_files()` function.

### 6.1 Files

Create:

```text
tmp_path / "one.png"
tmp_path / "two.png"
```

with arbitrary bytes.

Do NOT monkeypatch:

```text
_batch_files
_iter_images
_collect_images
```

The actual `_batch_files()` implementation must discover the two `.png` files.

### 6.2 Args

Pass:

```python
args = _args(input_dir=tmp_path)
```

Important:

```text
input_dir must be Path, not str
```

The remaining required `_run_batch()` fields must exist:

```text
prompt
device
alpha
save_diagnostics
```

### 6.3 Monkeypatch

Monkeypatch ONLY:

```python
predict.predict_one
```

to return:

```python
_FakeResult(_success_payload(), ok=True)
```

Do not instantiate real model/runtime.

### 6.4 Call real signature

Call exactly:

```python
exit_code = predict._run_batch(
    object(),
    args,
    "synthetic-model",
    object(),
    {"language_mode": "synthetic"},
    time.time(),
)
```

### 6.5 Assertions

Require:

```python
exit_code == 0
```

Require exact success suffix count:

```python
out.count("SUCCESS [runtime-only; semantic=NOT_EVALUATED]") == 2
```

Require both filenames appear on successful item lines:

```text
one.png ... SUCCESS [runtime-only; semantic=NOT_EVALUATED]
two.png ... SUCCESS [runtime-only; semantic=NOT_EVALUATED]
```

Require:

```text
Runtime success: 2
Failed : 0
```

This test must fail if:
- `_batch_files()` stops discovering valid files;
- success suffix is removed;
- `successes += 1` is removed;
- summary label changes.

---

## 7. INSPECT-PROPOSALS TEST — UNCHANGED

Do NOT modify:

```text
test_predict_inspect_proposals_does_not_require_prompt
```

Do NOT:
- skip;
- xfail;
- weaken;
- change expected return code/error code;
- modify product behavior.

Frozen classification:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

Separate task later.

---

## 8. TEST GATES — EXACT

From:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

run exactly:

### Gate A

```text
python -m pytest   tests/test_cli_contract.py::test_r1a_single_success_semantics_block   tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics   tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary   -q
```

Required:

```text
3 passed
exit 0
```

### Gate B

```text
python -m pytest   tests/test_cli_contract.py::test_predict_help_lists_frozen_arguments   tests/test_cli_contract.py::test_predict_prompt_required_for_normal_inference   tests/test_cli_contract.py::test_predict_missing_image_reports_e201   tests/test_cli_contract.py::test_predict_unsupported_image_type_reports_e203   -q
```

Required:

```text
4 passed
exit 0
```

Do not replace with `-k`.

Do NOT run:
- inspect-proposals node;
- full `test_cli_contract.py`;
- full canonical suite.

---

## 9. STATIC TEST-HELPER SANITY CHECK

Run a small deterministic source sanity check that verifies exact counts in the test file:

```text
_FakeResult definitions = 1
_success_payload definitions = 1
_args definitions = 1
```

This check is only for duplicate-helper cleanup.

Do NOT use source scanning as a substitute for the behavioral Gate A tests.

Record its result in the task report.

---

## 10. PRODUCT CODE MUST REMAIN UNCHANGED

Verify before commit:

```text
predict.py = unchanged from starting HEAD
pipeline.py = unchanged from starting HEAD
```

No product cleanup or formatting.

---

## 11. MANIFEST

Do NOT update:

```text
source_manifest.json
```

Record:

```text
manifest_update = DEFERRED_TO_R1_D
```

---

## 12. EVERY OUTCOME MUST BE PUSHED

Frozen project collaboration rule:

```text
ALL TASK OUTCOMES MUST BE PERSISTED TO GITHUB
```

Whether:

```text
COMPLETE
COMPLETE_WITH_EVIDENCE_GAPS
STOP
FAILED
```

you must:
- update `handoff/FROM_DSH.md`;
- create report/evidence;
- commit authorized state;
- push task branch.

---

## 13. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1a_r5_test_contract_cleanup.md
evaluation/task8b3_mask01_d1_r1a_r5_test_contract_cleanup.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report:

```text
task_id = MASK01_D1_R1A_R5_TEST_CONTRACT_CLEANUP
status

starting branch/head
task branch

fake_result_definition_count = 1
success_payload_definition_count = 1
args_definition_count = 1

single_success_checks_output_paths = true
batch_uses_real_batch_files = true
batch_input_dir_is_path = true
batch_only_monkeypatches_predict_one = true

Gate A command/result
Gate B command/result
static helper-count check/result

inspect_test_modified = false
predict_py_modified = false
pipeline_modified = false
manifest_modified = false
canonical_docs_modified = false

model_inference = false
training = false
external_write = false

manifest_update = DEFERRED_TO_R1_D
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1A_R5_REMOTE_AUDIT
```

---

## 14. DIFF GATE

Before commit:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Only these may be staged:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
docs/task8b3_mask01_d1_r1a_r5_test_contract_cleanup.md
evaluation/task8b3_mask01_d1_r1a_r5_test_contract_cleanup.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected paths:
- do not delete;
- do not stage;
- report them.

---

## 15. COMMIT / PUSH

Commit exactly once:

```text
git commit -m "test(rc1): clean cli success contract tests"
```

Push:

```text
fix/task8b3-mask01-success-semantics-r1a-r5-tests
```

No force push.

Reports use:

```text
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

---

## 16. ABSOLUTE PROHIBITIONS

Do NOT:

```text
modify predict.py
modify pipeline.py
modify manifest
modify canonical docs
modify test_task8b_runtime.py

modify/skip/xfail inspect-proposals test
fix inspect-proposals
change E202/E3xx behavior

run model inference
run training
run full test_cli_contract.py
run full canonical suite
write/sync external RC1

change algorithms
add thresholds

reset
rebase
amend
stash
clean
force-push

start R1-B
start R1-C
start R1-D
start R1-E
start D2
```

---

## 17. SUCCESS DEFINITION

Expected successful terminal state:

```text
MASK01_D1_R1A_R5_TEST_CONTRACT_CLEANUP = COMPLETE

helper definitions:
_FakeResult = 1
_success_payload = 1
_args = 1

single success test:
real _report_single = exercised
semantic lines = exact
output-path/reference/mask-area lines = asserted

single failure test:
real _report_single = exercised
no success semantics = asserted

batch test:
real _run_batch = exercised
real _batch_files = exercised
predict_one only = monkeypatched
two PNG files discovered
success suffix count = 2
Runtime success: 2
Failed : 0
exit = 0

Gate A = 3/3 PASS
Gate B = 4/4 PASS

product code = unchanged
inspect-proposals issue = unchanged/deferred
manifest = deferred to R1-D

NEXT = CHATGPT_R1A_R5_REMOTE_AUDIT
```

Then STOP.
