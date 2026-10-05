# TO_DSH — MASK01_D1_R1A_R1_CLI_CONTRACT_CORRECTION

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1a`
> Required starting HEAD: `1e2282151aa8c0c3cbbdc63526d8e449dc7b655d`
> New task branch: `fix/task8b3-mask01-success-semantics-r1a-r1`

## 0. PURPOSE

This is a small corrective task for R1-A only.

ChatGPT independently audited commit:

```text
1e2282151aa8c0c3cbbdc63526d8e449dc7b655d
docs(rc1): persist r1a cli semantics state
```

The persisted R1-A state is useful and accepted as an audit snapshot, but the CLI implementation itself is NOT yet accepted.

This task corrects only the R1-A implementation contract.

Do not fix the separate inspect-proposals exit-code failure in this task.

---

## 1. CHATGPT AUDIT FINDINGS

Three R1-A defects are frozen for correction.

### Defect A — wrong single-image SUCCESS presentation

Current implementation prints:

```text
Result       : SUCCESS
Result       : SUCCESS [runtime-only; semantic=NOT_EVALUATED]
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

This is NOT the frozen contract.

Required exact block:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

There must be only one `Result` line.

### Defect B — batch summary label not changed

Current code still prints:

```text
Success: <n>
```

Required:

```text
Runtime success: <n>
```

### Defect C — contract tests are still source-string tests

Current new tests mainly read `predict.py` as text and assert strings exist.

That does not prove runtime presentation behavior.

R1-A requires real deterministic behavior tests using synthetic/fake result objects or a minimal pure formatting helper.

---

## 2. SEPARATE PRE-EXISTING FAILURE — DO NOT FIX HERE

Known current canonical test:

```text
test_predict_inspect_proposals_does_not_require_prompt
```

has been reported as:

```text
expected = returncode 20 / E202
observed = returncode 30
```

This issue existed before this corrective implementation and is classified for now as:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

Do NOT in this task:

```text
change inspect-proposals behavior
change E202/E3xx routing
change model package verification order
change PredictRuntime initialization order
edit or skip/xfail that existing inspect test
```

A separate ChatGPT task will handle it if needed.

---

## 3. GIT PREFLIGHT

Verify:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1a

HEAD =
1e2282151aa8c0c3cbbdc63526d8e449dc7b655d
```

Allowed initial working tree:

```text
clean
```

or only:

```text
M handoff/TO_DSH.md
```

from task-book replacement.

Any other pre-existing mutation:

```text
record it
do not delete it
do not reset/restore/clean it
STOP implementation if it overlaps this task
```

Create:

```text
fix/task8b3-mask01-success-semantics-r1a-r1
```

from the required HEAD.

No reset/rebase/amend/stash/clean/force-push.

---

## 4. EXACT ALLOWED PRODUCT CHANGES

Only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
```

plus task reporting:

```text
docs/task8b3_mask01_d1_r1a_r1_cli_contract_correction.md
evaluation/task8b3_mask01_d1_r1a_r1_cli_contract_correction.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Explicitly forbidden:

```text
buildreasonseg/runtime/pipeline.py
tests/test_task8b_runtime.py
source_manifest.json
canonical README/docs
configs
models
weights
external RC1
```

---

## 5. SINGLE-IMAGE NORMAL SUCCESS — EXACT CONTRACT

In `predict.py`, `_report_single()` normal SUCCESS must print exactly:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

Then preserve the existing blank line and existing output lines:

```text
Mask
Overlay
Diagnostics
Reference ID
Mask area
```

Do not print a second `Result` line.

For normal SUCCESS, obtain:

```text
validity_scope
semantic_status
```

from `result.result_payload`.

Expected machine values are already provided by the frozen `pipeline.py` contract.

The user-facing Note may be a stable CLI constant/string in `predict.py`.

Do not change machine semantics in `pipeline.py`.

---

## 6. SINGLE-IMAGE FAILED RESULT

For failed result:

```text
status != SUCCESS
```

must NOT print:

```text
Validity     :
Semantic     :
SUCCESS only confirms the current runtime structural checks
```

Existing failure/error rendering remains unchanged.

---

## 7. BATCH SUCCESS PRESENTATION

Keep exact success suffix:

```text
SUCCESS [runtime-only; semantic=NOT_EVALUATED]
```

Prefer reading the semantic status from the successful result payload if a minimal implementation permits it without unrelated refactoring.

Do not change:

```text
result.ok
success counting
failure counting
batch exit codes
iteration behavior
```

---

## 8. BATCH SUMMARY

Change only:

```text
Success: <n>
```

to:

```text
Runtime success: <n>
```

Keep:

```text
Failed : <n>
Total  : <time>s
```

and return-code behavior unchanged.

---

## 9. REAL CONTRACT TESTS — REQUIRED

Replace the R1-A source-string tests with behavioral tests.

Do not inspect source code text to prove CLI output.

Use:

```text
import predict as predict_module
```

and synthetic/fake result objects.

### Test A — successful single result

Use a small fake/synthetic result with:

```python
result.result_payload = {
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
result.ok = True
```

Call:

```text
predict_module._report_single(...)
```

directly.

Capture stdout.

Require exact lines:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

Also require:

```text
Mask         : mask.png
Overlay      : overlay.png
Diagnostics  : diag
Reference ID : 7
Mask area    : 123
```

Require only one line beginning with:

```text
Result       :
```

### Test B — failed single result

Use a fake FAILED result compatible with `_report_single()`.

Capture stdout/stderr.

Prove output does NOT contain:

```text
Validity     :
Semantic     :
SUCCESS only confirms
```

Do not change failure semantics.

### Test C — batch success annotation and summary

Do not run models.

Use monkeypatch/fakes around the smallest existing unit, or factor a tiny pure formatting/reporting helper in `predict.py` if needed.

Prove actual emitted output contains:

```text
SUCCESS [runtime-only; semantic=NOT_EVALUATED]
Runtime success:
```

This must exercise Python behavior, not source-text scanning.

Do not invoke real `PredictRuntime`.

---

## 10. R1-A-R1 TEST GATE

Because the unrelated inspect-proposals test is already a known separate candidate defect, do NOT use the full `test_cli_contract.py` file as this subtask's acceptance gate.

Give the new behavioral tests exact names:

```text
test_r1a_single_success_semantics_block
test_r1a_single_failure_omits_success_semantics
test_r1a_batch_success_annotation_and_runtime_summary
```

Run exactly:

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

Also run these existing unaffected smoke contracts:

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

Do NOT run the known failing inspect-proposals node in this task.

Do NOT run full `test_cli_contract.py`.

Do NOT run full canonical suite.

---

## 11. MANIFEST

Do NOT update:

```text
source_manifest.json
```

This remains intentionally deferred to R1-D.

Record:

```text
manifest_update = DEFERRED_TO_R1_D
```

---

## 12. EVERY OUTCOME MUST BE PUSHED

Project collaboration policy:

```text
ALL TASK OUTCOMES MUST BE PERSISTED TO GITHUB
```

Whether this task ends:

```text
COMPLETE
COMPLETE_WITH_EVIDENCE_GAPS
STOP
FAILED
```

DSH must update `FROM_DSH.md`, report/evidence, commit authorized state, and push.

Do not leave the true state only locally.

---

## 13. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1a_r1_cli_contract_correction.md
evaluation/task8b3_mask01_d1_r1a_r1_cli_contract_correction.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Include:

```text
task_id = MASK01_D1_R1A_R1_CLI_CONTRACT_CORRECTION
status

starting branch/head
task branch

audit defect A corrected?
audit defect B corrected?
audit defect C corrected?

exact single SUCCESS block
batch SUCCESS suffix
batch summary label

behavioral tests:
3-node R1A command/result
4-node unaffected smoke command/result

inspect-proposals defect modified = NO

pipeline modified = NO
manifest modified = NO
canonical docs modified = NO
model inference = NO
training = NO
external write = NO

manifest_update = DEFERRED_TO_R1_D

github_persistence_policy =
ALL_TASK_OUTCOMES_PUSHED

next_gate =
CHATGPT_R1A_R1_REMOTE_AUDIT
```

---

## 14. DIFF GATE

Before commit:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Only Section 4 paths may be staged.

Unexpected paths must not be deleted or staged.

---

## 15. COMMIT / PUSH

Commit exactly once:

```text
git commit -m "fix(rc1): correct cli success semantics contract"
```

Push:

```text
fix/task8b3-mask01-success-semantics-r1a-r1
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

Do not enter the inspect-proposals corrective task.
Do not enter R1-B.

---

## 16. ABSOLUTE PROHIBITIONS

Do NOT:

```text
modify pipeline.py
modify source_manifest.json
modify canonical docs
modify test_task8b_runtime.py
fix inspect-proposals
skip/xfail the inspect test
change E202/E3xx handling
change model initialization order
run model inference
run training
run full canonical suite
write external RC1
sync external RC1
add thresholds
change architecture
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
MASK01_D1_R1A_R1 = COMPLETE

single-image block:
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.

batch:
SUCCESS [runtime-only; semantic=NOT_EVALUATED]
Runtime success:

behavioral R1-A tests = 3/3 PASS
unaffected CLI smoke tests = 4/4 PASS

known inspect-proposals candidate defect = UNCHANGED / DEFERRED

pipeline = UNCHANGED
manifest = DEFERRED_TO_R1_D
external RC1 = UNCHANGED

NEXT = CHATGPT_R1A_R1_REMOTE_AUDIT
```

Then STOP.
