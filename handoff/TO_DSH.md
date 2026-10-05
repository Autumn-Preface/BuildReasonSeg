# TO_DSH — MASK01_D1_R1A_STATE_PERSISTENCE

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required current branch: `fix/task8b3-mask01-success-semantics-r1a`
> Required base commit / parent: `805f33eb1fbbf9f0903c67defbb130dd9124c339`

## 0. PURPOSE

This task does NOT continue implementation.

It only persists the current R1-A working state and its task report to GitHub so ChatGPT can independently audit it.

The project repository is now treated as the shared state bus / audit log between DSH and ChatGPT.

Therefore, from this task onward:

```text
EVERY TASK OUTCOME MUST BE PUSHED TO GITHUB
```

including:

```text
COMPLETE
COMPLETE_WITH_EVIDENCE_GAPS
STOP
FAILED
```

A STOP/FAILED state must be truthfully labelled, but its authorized code state, FROM_DSH report, evidence, and current TO_DSH must still be committed and pushed.

Do NOT hide task state only in the local working tree.

---

## 1. CURRENT FACTS TO PRESERVE

Expected current branch:

```text
fix/task8b3-mask01-success-semantics-r1a
```

Expected parent/base:

```text
805f33eb1fbbf9f0903c67defbb130dd9124c339
```

Expected existing uncommitted product/test changes from R1-A:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
```

The user will replace:

```text
handoff/TO_DSH.md
```

with this task book.

No other pre-existing working-tree mutation is authorized.

If any other path is already modified/untracked before this task begins:

```text
STOP
```

but still create/push a report-only state if doing so does not require touching or discarding the unknown file.

Never reset/rebase/amend/stash/clean/restore/discard unknown work.

---

## 2. DO NOT CHANGE R1-A PRODUCT LOGIC

Do NOT modify the current contents of:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
```

except only if needed to remove accidental trailing temporary/debug material that was not part of the R1-A implementation.

Preferred action:

```text
NO PRODUCT/TEST EDIT
```

This task is for persistence and reporting only.

Do NOT modify:

```text
pipeline.py
source_manifest.json
test_task8b_runtime.py
canonical README/docs
model code
runtime code
configs
```

---

## 3. RECORD CURRENT R1-A TEST FACTS

Do not rerun the full suite.

You may rerun exactly once:

```text
python -m pytest tests/test_cli_contract.py -q
```

from:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

only to capture the current exact result for the report.

Expected historical result supplied by DSH:

```text
1 failed, 22 passed
```

Expected failing test:

```text
tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt
```

Expected key mismatch:

```text
expected returncode = 20 / E202
observed returncode = 30
```

Do NOT fix this failure in this task.

Do NOT mark it xfail/skip.

Do NOT weaken the test.

Do NOT change inspect-proposals behavior.

This issue is classified only as:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

Final acceptance/classification belongs to ChatGPT after GitHub audit.

---

## 4. CREATE R1-A REPORT / EVIDENCE

Create:

```text
docs/task8b3_mask01_d1_r1a_state_persistence.md
evaluation/task8b3_mask01_d1_r1a_state_persistence.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

The report must include at minimum:

```text
task_id = MASK01_D1_R1A_STATE_PERSISTENCE

original_task_id = MASK01_D1_R1A_CLI_SEMANTICS

current_branch
base_commit
working_tree_paths_before_report

R1A product paths:
- predict.py
- tests/test_cli_contract.py

R1A implementation summary:
- single-image success presentation
- batch success annotation
- batch summary label
- failure presentation behavior
- inspect-proposals behavior intended unchanged

test command
test exit code
test summary
failing test name
expected vs actual return code
stderr/error code if available

classification =
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE

product_logic_changed_in_this_persistence_task = false

pipeline_modified = false
manifest_modified = false
canonical_docs_modified = false
model_inference_executed = false
training_executed = false
external_write_performed = false

github_persistence_policy =
ALL_TASK_OUTCOMES_PUSHED

next_gate =
CHATGPT_R1A_REMOTE_AUDIT
```

Do NOT claim R1-A accepted.

Use:

```text
original_r1a_gate_status = STOP
```

or equivalent truthful wording.

---

## 5. FROM_DSH IS MANDATORY FOR EVERY TASK

From this task onward, `handoff/FROM_DSH.md` is mandatory regardless of task outcome.

It must always record:

```text
Task
Status
starting branch/head
task branch/head
changed paths
tests run/results
STOP/FAILED reason if any
commit message
remote branch
next gate chosen by ChatGPT / awaiting ChatGPT review
```

DSH must never omit FROM_DSH merely because the task failed or stopped.

---

## 6. DIFF GATE

Before commit, allowed changed paths are exactly:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py

docs/task8b3_mask01_d1_r1a_state_persistence.md
evaluation/task8b3_mask01_d1_r1a_state_persistence.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may be staged.

If an unexpected path exists:

```text
do not delete it
do not clean it
do not stage it
record it in FROM_DSH
STOP implementation work
```

But if the authorized state can still be safely committed without the unexpected path, commit only the authorized paths and report the unexpected path explicitly.

---

## 7. COMMIT / PUSH — REQUIRED EVEN THOUGH ORIGINAL R1-A GATE STOPPED

This task intentionally overrides the previous "do not commit on STOP" policy.

The new collaboration rule is:

```text
AUTHORIZED TASK STATE MUST BE PERSISTED TO GITHUB EVEN WHEN STATUS = STOP/FAILED
```

Commit exactly once with:

```text
git commit -m "docs(rc1): persist r1a cli semantics state"
```

The commit may include the existing authorized R1-A product/test changes plus this persistence report/evidence/handoff.

Push:

```text
fix/task8b3-mask01-success-semantics-r1a
```

No force push.

Do not create a new branch.

Do not amend the D1 commit.

Do not rewrite history.

Committed reports must contain:

```text
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Verify local/remote HEAD equal.

---

## 8. ABSOLUTE PROHIBITIONS

Do NOT:

```text
fix inspect-proposals
change E202/E3xx behavior
change model initialization order
modify pipeline.py
modify source_manifest.json
modify canonical docs
modify test_task8b_runtime.py
run full test suite
run model inference
run training
write/sync external RC1
start R1-B
start R1-C
start R1-D
start R1-E
reset
rebase
amend
stash
clean
restore authorized R1-A changes
force-push
```

---

## 9. TERMINAL STATE

Required terminal disposition:

```text
R1-A LOCAL STATE = PERSISTED TO GITHUB
ORIGINAL R1-A GATE = STOP (known failing canonical test)
R1-A ACCEPTANCE = DEFER_TO_CHATGPT
FROM_DSH = PUSHED
EVIDENCE = PUSHED
PRODUCT/TEST WORKING STATE = PUSHED
NEXT = CHATGPT_R1A_REMOTE_AUDIT
```

Then STOP.
