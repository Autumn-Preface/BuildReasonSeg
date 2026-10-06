# TO_DSH — MASK01_D1_R1A_R2_EXECUTABLE_CLI_FIX

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1a-r1`
> Required starting HEAD: `2fe6369d7def85b59a16c28af4340eb61f24437c`
> New task branch: `fix/task8b3-mask01-success-semantics-r1a-r2`

## 0. PURPOSE

This is a narrowly scoped corrective task for R1-A.

ChatGPT independently audited:

```text
2fe6369d7def85b59a16c28af4340eb61f24437c
fix(rc1): correct cli success semantics contract
```

The commit is preserved as historical state, but the implementation is NOT accepted.

Do not rewrite history.

This task fixes only the executable CLI presentation/counting regressions and the behavioral tests that must prove them.

Do NOT fix the separate inspect-proposals `30 vs 20` issue.

---

## 1. FROZEN AUDIT FINDINGS

### A. Batch success counter regression

Current code removed:

```python
successes += 1
```

from the successful batch branch.

This is a real regression.

It can make:

```text
Runtime success count = wrong
batch exit-code decision = wrong
```

because later logic still uses `successes`.

Required correction:

```python
if result.ok:
    successes += 1
    ...
```

must be restored.

### B. Batch success annotation was removed

Current successful item prints only:

```text
... SUCCESS
```

Required:

```text
... SUCCESS [runtime-only; semantic=NOT_EVALUATED]
```

Restore this exact suffix.

### C. `_report_single()` repeats the Note line

Current structure effectively does:

```python
for line in success_report_lines(payload):
    print(line)
    print(Note)
```

but `success_report_lines()` already returns the Note line.

Therefore the Note is repeated.

Required:

```python
for line in success_report_lines(payload):
    print(line)
```

and nothing else inside that loop.

A successful single result must print exactly four semantic lines, once each.

### D. Previous tests did not prove the required runtime behavior

The previous commit added helper-level tests but:
- did not use the exact required R1-A test names;
- used a `try/except Exception: pass` pattern around `_report_single()`;
- used a weak conditional assertion;
- did not exercise `_run_batch()` enough to catch the deleted `successes += 1`;
- did not prove the successful item suffix and summary count together.

This task must replace those weak tests with strict executable tests.

---

## 2. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1a-r1

HEAD =
2fe6369d7def85b59a16c28af4340eb61f24437c
```

Allowed initial worktree:

```text
clean
```

or only:

```text
M handoff/TO_DSH.md
```

from task-book replacement.

Any other pre-existing change:
- do not delete;
- do not reset;
- do not restore;
- do not clean;
- record and STOP overlapping implementation.

Create:

```text
fix/task8b3-mask01-success-semantics-r1a-r2
```

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED PATHS

Only these product/test files may change:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
```

And these task records:

```text
docs/task8b3_mask01_d1_r1a_r2_executable_cli_fix.md
evaluation/task8b3_mask01_d1_r1a_r2_executable_cli_fix.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Explicitly forbidden:

```text
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

## 4. SINGLE-IMAGE SUCCESS CONTRACT

Keep or minimally correct `success_report_lines(payload)`.

For this payload:

```python
{
    "status": "SUCCESS",
    "validity_scope": "RUNTIME_STRUCTURAL_ONLY",
    "semantic_status": "NOT_EVALUATED",
}
```

it must return exactly:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

`_report_single()` must print those four lines exactly once each.

Then existing successful output:

```text
Mask
Overlay
Diagnostics
Reference ID
Mask area
```

must still print.

There must be exactly one:

```text
Result       :
```

line and exactly one:

```text
Note         :
```

line.

FAILED output must not print Validity/Semantic/Note.

Do not change error handling.

---

## 5. BATCH CONTRACT — RESTORE EXECUTABLE BEHAVIOR

In `_run_batch()`:

### Successful result

Must perform:

```python
successes += 1
```

and print exactly:

```text
[<index>/<total>] <filename> ... SUCCESS [runtime-only; semantic=NOT_EVALUATED]
```

### Failed result

Existing failure count and presentation stay unchanged:

```python
failures += 1
```

### Summary

Must print:

```text
Batch complete
Runtime success: <successes>
Failed : <failures>
Total  : <...>s
```

### Exit semantics

Must remain:

```text
all successful -> 0
all failed -> EXIT_BATCH_ALL_FAILED
mixed -> EXIT_BATCH_PARTIAL
```

Do not change constants or status semantics.

---

## 6. REAL BEHAVIORAL TESTS — EXACT NAMES

In:

```text
tests/test_cli_contract.py
```

remove/replace the weak R1-A-R1 tests as needed.

Create exactly these tests:

```text
test_r1a_single_success_semantics_block
test_r1a_single_failure_omits_success_semantics
test_r1a_batch_success_annotation_and_runtime_summary
```

### Test 1 — single success

Use a fake result object with:

```python
result.result_payload = {
    "status": "SUCCESS",
    "validity_scope": "RUNTIME_STRUCTURAL_ONLY",
    "semantic_status": "NOT_EVALUATED",
    "semantic_note": "SUCCESS means the RC1 runtime completed and passed its current structural checks; semantic target correctness is not established.",
    "output_paths": {
        "mask": "mask.png",
        "overlay": "overlay.png",
        "diagnostics": "diag",
    },
    "reference_id": 7,
    "mask_area": 123,
}
result.ok = True
```

Call:

```python
predict_module._report_single(fake, fake_args)
```

directly.

Do NOT wrap it in `try/except`.

Capture stdout.

Assert:
- exact four semantic lines are present;
- each appears exactly once;
- exactly one `Result       :` line;
- exactly one `Note         :` line;
- mask/overlay/diagnostics/reference/mask area lines are present.

### Test 2 — single failure

Use a fake result with:

```text
status = FAILED
ok = False
error_code = E404
```

and enough payload fields for the existing error path.

Call `_report_single()` directly.

Capture stdout/stderr.

Assert stdout contains:

```text
Result       : FAILED
```

and does NOT contain:

```text
Validity     :
Semantic     :
Note         :
```

Do not suppress exceptions.

### Test 3 — batch success + summary + counting

This test must exercise the actual `_run_batch()` function without model execution.

Use monkeypatch/fakes for:
- `_batch_files` or a temporary directory with supported filenames;
- `predict_one()` returning fake successful results;
- runtime can be a dummy object because patched `predict_one()` must prevent model execution;
- args must contain the fields `_run_batch()` reads.

Use at least 2 fake input files, both successful.

Call `_run_batch()` directly.

Assert:
- return code is `0`;
- stdout contains 2 successful item lines with exact suffix:
  `SUCCESS [runtime-only; semantic=NOT_EVALUATED]`;
- stdout contains:
  `Runtime success: 2`;
- stdout contains:
  `Failed : 0`.

This test must fail if `successes += 1` is deleted.

Optional: if simple, add a separate mixed fake inside the same test or another existing helper test to preserve mixed exit semantics, but do not expand scope unnecessarily.

No source-code string scanning is allowed for these three tests.

---

## 7. TEST GATES — RUN EXACTLY

From canonical delivery root run exactly:

### Gate A

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

### Gate B

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

Do not replace these commands with `-k`.

Do not run the inspect-proposals failing node.

Do not run full `test_cli_contract.py`.

Do not run full canonical suite.

---

## 8. KNOWN INSPECT-PROPOSALS ISSUE

Do NOT modify:

```text
test_predict_inspect_proposals_does_not_require_prompt
```

Do NOT:
- xfail it;
- skip it;
- weaken it;
- change `--inspect-proposals`;
- change E202/E3xx behavior;
- change model verification/runtime initialization order.

Frozen current classification:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

Separate task later.

---

## 9. MANIFEST

Do not modify:

```text
source_manifest.json
```

Record:

```text
manifest_update = DEFERRED_TO_R1_D
```

Temporary manifest mismatch is expected until R1-D.

---

## 10. EVERY OUTCOME MUST BE PUSHED

Current project collaboration rule is frozen:

```text
ALL TASK OUTCOMES MUST BE PERSISTED TO GITHUB
```

Regardless of:

```text
COMPLETE
COMPLETE_WITH_EVIDENCE_GAPS
STOP
FAILED
```

you must:
- update FROM_DSH;
- write report/evidence;
- commit authorized current state;
- push task branch.

Do not leave facts only in the local workspace.

---

## 11. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1a_r2_executable_cli_fix.md
evaluation/task8b3_mask01_d1_r1a_r2_executable_cli_fix.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report:

```text
task_id = MASK01_D1_R1A_R2_EXECUTABLE_CLI_FIX
status

starting branch/head
task branch

batch_success_counter_restored
batch_success_suffix_restored
single_note_duplication_fixed

single SUCCESS exact block
batch success exact line format
batch summary exact label

Gate A command/result
Gate B command/result

known inspect-proposals issue modified = NO

pipeline modified = NO
manifest modified = NO
canonical docs modified = NO
model inference = NO
training = NO
external sync/write = NO

manifest_update = DEFERRED_TO_R1_D

github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1A_R2_REMOTE_AUDIT
```

---

## 12. DIFF GATE

Before commit:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Only Section 3 paths may be staged.

Unexpected files:
- do not delete;
- do not stage;
- report them.

---

## 13. COMMIT / PUSH

Commit exactly once:

```text
git commit -m "fix(rc1): restore executable cli success contract"
```

Push:

```text
fix/task8b3-mask01-success-semantics-r1a-r2
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

Do not enter any later task.

---

## 14. ABSOLUTE PROHIBITIONS

Do NOT:

```text
modify pipeline.py
modify source_manifest.json
modify canonical docs
modify test_task8b_runtime.py

fix inspect-proposals
skip/xfail inspect test
change E202/E3xx handling
change model initialization order

run model inference
run training
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

## 15. SUCCESS DEFINITION

Expected successful state:

```text
MASK01_D1_R1A_R2 = COMPLETE

single success:
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.

single Note count = 1
single Result count = 1

batch successful item:
SUCCESS [runtime-only; semantic=NOT_EVALUATED]

batch counting:
successes incremented correctly

batch summary:
Runtime success: <n>

Gate A = 3/3 PASS
Gate B = 4/4 PASS

inspect-proposals issue = unchanged/deferred
pipeline = unchanged
manifest = deferred to R1-D
external RC1 = unchanged

NEXT = CHATGPT_R1A_R2_REMOTE_AUDIT
```

Then STOP.
