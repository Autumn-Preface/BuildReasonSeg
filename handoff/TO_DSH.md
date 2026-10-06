# TO_DSH — MASK01_D1_R1A_R4_TESTS_ONLY

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1a-r3-product`
> Required starting HEAD: `37b9afe0bc674630c4f9f5040e0035ceeb809fc1`
> New task branch: `fix/task8b3-mask01-success-semantics-r1a-r4-tests`

## 0. PURPOSE

This is a tests-only task.

ChatGPT has accepted the product-only R1-A-R3 state:

```text
batch successes += 1 = present
batch success suffix = restored
batch summary = Runtime success: <n>
single-image Note duplication = fixed
```

Do NOT modify `predict.py` in this task.

This task only repairs the three R1-A behavioral tests so they exercise the real existing functions correctly.

Do NOT fix the separate inspect-proposals `30 vs 20` issue.

---

## 1. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1a-r3-product

HEAD =
37b9afe0bc674630c4f9f5040e0035ceeb809fc1
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
fix/task8b3-mask01-success-semantics-r1a-r4-tests
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
docs/task8b3_mask01_d1_r1a_r4_tests_only.md
evaluation/task8b3_mask01_d1_r1a_r4_tests_only.json
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

## 3. CLEAN UP TEST FIXTURES

In:

```text
tests/test_cli_contract.py
```

there are currently duplicate `_FakeResult` class definitions.

Remove the redundant earlier definition.

Keep one fake result class equivalent to:

```python
class _FakeResult:
    def __init__(self, payload, ok=True, error_code=None, error_reason=None):
        self.result_payload = payload
        self.ok = ok
        self.error_code = error_code
        self.error_reason = error_reason
```

No source-text scanning tests.

No try/except that swallows failures.

---

## 4. EXACT TEST 1 — SINGLE SUCCESS

Test name must remain exactly:

```text
test_r1a_single_success_semantics_block
```

Use:

```python
import predict as predict_module
```

Construct a COMPLETE fake success payload:

```python
payload = {
    "status": "SUCCESS",
    "validity_scope": "RUNTIME_STRUCTURAL_ONLY",
    "semantic_status": "NOT_EVALUATED",
    "semantic_note": (
        "SUCCESS means the RC1 runtime completed and passed its current structural checks; "
        "semantic target correctness is not established."
    ),
    "output_paths": {
        "mask": "mask.png",
        "overlay": "overlay.png",
        "diagnostics": "diag",
    },
    "reference_id": 7,
    "mask_area": 123,
}
```

Use:

```python
fake = _FakeResult(payload, ok=True)
predict_module._report_single(fake, type("Args", (), {})())
```

Do NOT wrap in try/except.

Capture stdout with `capsys`.

Assert all of the following:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
Mask         : mask.png
Overlay      : overlay.png
Diagnostics  : diag
Reference ID : 7
Mask area    : 123
```

Also assert:

```text
count("Result       : SUCCESS") == 1
count("Note         : ...") == 1
```

This test must exercise `_report_single()` itself.

---

## 5. EXACT TEST 2 — SINGLE FAILURE

Test name must remain exactly:

```text
test_r1a_single_failure_omits_success_semantics
```

Construct:

```python
payload = {
    "status": "FAILED",
    "error_code": "E404",
    "detail": "synthetic failure",
}
fake = _FakeResult(
    payload,
    ok=False,
    error_code="E404",
    error_reason="synthetic failure",
)
```

Call:

```python
predict_module._report_single(fake, type("Args", (), {})())
```

Do NOT wrap in try/except.

Capture stdout + stderr.

Assert stdout contains:

```text
Result       : FAILED
```

Assert stdout does NOT contain:

```text
Validity     :
Semantic     :
Note         :
```

Do not assert exact localized error-message wording on stderr unless required by existing contract.

The purpose of this test is only to prove failed results do not receive SUCCESS semantic claims.

---

## 6. EXACT TEST 3 — REAL `_run_batch()` BEHAVIOR

Test name must remain exactly:

```text
test_r1a_batch_success_annotation_and_runtime_summary
```

This test MUST call the real current signature:

```python
predict_module._run_batch(
    runtime,
    args,
    package_name,
    parsed,
    language_info,
    started,
)
```

Do NOT use:
- `_iter_images`;
- `_collect_images`;
- fake alternate `_run_batch(args)` signatures;
- try/except TypeError fallback.

### 6.1 Input files

Create exactly two temporary supported image filenames, for example:

```text
one.png
two.png
```

Their bytes do not need to be valid images because `predict_one()` will be monkeypatched.

Use the real `_batch_files()` behavior by setting:

```python
args.input_dir = tmp_path
```

### 6.2 Args

Provide all fields read by current `_run_batch()`:

```python
args = type(
    "Args",
    (),
    {
        "input_dir": tmp_path,
        "prompt": "synthetic prompt",
        "device": "cpu",
        "alpha": 0.45,
        "save_diagnostics": False,
    },
)()
```

### 6.3 Other arguments

Use:

```python
runtime = object()
package_name = "synthetic-model"
parsed = object()
language_info = {"language_mode": "synthetic"}
started = time.time()
```

### 6.4 Monkeypatch

Monkeypatch only:

```python
predict_module.predict_one
```

to return a successful fake result:

```python
_FakeResult({"status": "SUCCESS"}, ok=True)
```

The test must NOT instantiate real `PredictRuntime`.

### 6.5 Assertions

Call `_run_batch(...)`.

Assert return code:

```text
0
```

Capture stdout and require:

```text
one.png ... SUCCESS [runtime-only; semantic=NOT_EVALUATED]
two.png ... SUCCESS [runtime-only; semantic=NOT_EVALUATED]
Runtime success: 2
Failed : 0
```

Also assert:

```python
out.count("SUCCESS [runtime-only; semantic=NOT_EVALUATED]") == 2
```

This test must fail if:
- batch suffix is removed;
- `successes += 1` is removed;
- summary label changes.

---

## 7. DO NOT TOUCH INSPECT-PROPOSALS TEST

Do NOT modify:

```text
test_predict_inspect_proposals_does_not_require_prompt
```

Do NOT:
- skip it;
- xfail it;
- weaken it;
- change expected 20/E202;
- modify inspect runtime behavior.

Frozen classification remains:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

It will be handled separately.

---

## 8. TEST GATES — EXACT COMMANDS

From:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

run exactly:

### Gate A — R1-A behavioral tests

```text
python -m pytest \
  tests/test_cli_contract.py::test_r1a_single_success_semantics_block \
  tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics \
  tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary \
  -q
```

Required:

```text
3 passed
exit 0
```

### Gate B — unaffected CLI smoke

```text
python -m pytest \
  tests/test_cli_contract.py::test_predict_help_lists_frozen_arguments \
  tests/test_cli_contract.py::test_predict_prompt_required_for_normal_inference \
  tests/test_cli_contract.py::test_predict_missing_image_reports_e201 \
  tests/test_cli_contract.py::test_predict_unsupported_image_type_reports_e203 \
  -q
```

Required:

```text
4 passed
exit 0
```

Do NOT replace these commands with `-k`.

Do NOT run:
- full `test_cli_contract.py`;
- inspect-proposals node;
- full canonical suite.

---

## 9. PRODUCT CODE MUST REMAIN BYTE-UNCHANGED

This task is tests-only.

Before commit, verify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py = unchanged from starting HEAD
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py = unchanged
```

Do not “clean up” product code while editing tests.

---

## 10. MANIFEST

Do NOT update:

```text
source_manifest.json
```

Record:

```text
manifest_update = DEFERRED_TO_R1_D
```

Temporary manifest mismatch remains expected.

---

## 11. EVERY OUTCOME MUST BE PUSHED

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

Do not leave the real state only locally.

---

## 12. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1a_r4_tests_only.md
evaluation/task8b3_mask01_d1_r1a_r4_tests_only.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report:

```text
task_id = MASK01_D1_R1A_R4_TESTS_ONLY
status

starting branch/head
task branch

product_files_changed = false
duplicate_fake_result_removed = true

test_1_calls_report_single = true
test_2_calls_report_single = true
test_3_calls_real_run_batch_signature = true

Gate A command/result
Gate B command/result

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

next_gate = CHATGPT_R1A_R4_REMOTE_AUDIT
```

---

## 13. DIFF GATE

Before commit:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Only these may be staged:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
docs/task8b3_mask01_d1_r1a_r4_tests_only.md
evaluation/task8b3_mask01_d1_r1a_r4_tests_only.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected files:
- do not delete;
- do not stage;
- report them.

---

## 14. COMMIT / PUSH

Commit exactly once:

```text
git commit -m "test(rc1): lock cli success semantics behavior"
```

Push:

```text
fix/task8b3-mask01-success-semantics-r1a-r4-tests
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

## 15. ABSOLUTE PROHIBITIONS

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

## 16. SUCCESS DEFINITION

Expected successful terminal state:

```text
MASK01_D1_R1A_R4_TESTS_ONLY = COMPLETE

product code = unchanged

behavioral tests:
test_r1a_single_success_semantics_block = PASS
test_r1a_single_failure_omits_success_semantics = PASS
test_r1a_batch_success_annotation_and_runtime_summary = PASS

Gate A = 3/3 PASS
Gate B = 4/4 PASS

inspect-proposals known issue = unchanged/deferred
manifest = deferred to R1-D
external RC1 = unchanged

NEXT = CHATGPT_R1A_R4_REMOTE_AUDIT
```

Then STOP.
