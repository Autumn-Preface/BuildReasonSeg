# TO_DSH — MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1d2`
> Required starting HEAD: `78b749e7a5269d9d22e7b495193a79e011df87f0`
> New task branch: `fix/task8b3-mask01-success-semantics-r1e`

## 0. CHATGPT DECISION

R1-D2 is ACCEPTED / CLOSED.

Accepted remote state:

```text
branch = fix/task8b3-mask01-success-semantics-r1d2
HEAD   = 78b749e7a5269d9d22e7b495193a79e011df87f0
parent = efc48eb5ee8b0851a84e8a3d95533f1acd23cacf
commit = chore(rc1): recanonicalize source manifest after inspect fix
```

Accepted R1-D2 facts:

```text
manifest entries = 135
entries changed = 2
entries already canonical = 133

ordered_path_digest_before =
7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd

ordered_path_digest_after =
7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd

entry count preserved = true
path set/order preserved = true
all-entry Git canonical validation = PASS
manifest contract validation = PASS
```

The canonical tree and manifest are now ready for R1-E.

This is the FINAL canonical regression/closure gate for:

```text
SUCCESS_SEMANTICS_HARDENING_V1
MASK-01 engineering hardening chain
```

No product repair is authorized in this task.

---

## 1. PURPOSE

R1-E must answer only:

```text
Does the final canonical RC1 state pass:
1. the repaired SUCCESS-semantics contract,
2. inspect-proposals error-order contract,
3. source-manifest canonical identity,
4. the complete canonical test suite?
```

If YES:
- persist closure evidence;
- stop.

If ANY gate fails:
- do NOT fix code;
- persist truthful STOP/FAILED evidence;
- stop for ChatGPT review.

---

## 2. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1d2

HEAD =
78b749e7a5269d9d22e7b495193a79e011df87f0
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

Any other pre-existing mutation:
- do not delete/reset/restore/stash/clean;
- record it;
- STOP if it overlaps canonical files or tests.

Create:

```text
fix/task8b3-mask01-success-semantics-r1e
```

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED REPOSITORY CHANGES

NO canonical product/test/manifest/doc source file may be modified.

Only task records may change:

```text
docs/task8b3_mask01_d1_r1e_canonical_regression_closure.md
evaluation/task8b3_mask01_d1_r1e_canonical_regression_closure.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do NOT modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/**
scripts/sync_advisor_rc1_delivery.py
configs
models
weights
external RC1
```

Test/runtime-generated untracked files must not be staged.

Do not delete unknown/generated files with `clean`.

---

## 4. FROZEN SUCCESS CONTRACT TO VERIFY

Canonical machine contract:

```text
status = SUCCESS
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
semantic target correctness is not established
```

Canonical CLI contract:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
```

Batch successful item:

```text
SUCCESS [runtime-only; semantic=NOT_EVALUATED]
```

Batch summary:

```text
Runtime success: <n>
```

Compatibility:

```text
PipelineResult.ok == (status == "SUCCESS")
```

This task verifies these contracts only.
It does NOT claim semantic target correctness has been repaired.

---

## 5. GATE A — TARGETED SUCCESS SEMANTICS

From:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

run exactly:

```text
python -m pytest \
  tests/test_cli_contract.py::test_r1a_single_success_semantics_block \
  tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics \
  tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary \
  tests/test_task8b_runtime.py::test_success_semantics_contract_exact \
  tests/test_task8b_runtime.py::test_pipeline_result_ok_is_status_compatibility \
  -q
```

Required:

```text
5 passed
exit 0
```

If fail:
- record;
- do not edit code/tests;
- STOP.

---

## 6. GATE B — INSPECT-PROPOSALS CONTRACT

Run exactly:

```text
python -m pytest \
  tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt \
  tests/test_cli_contract.py::test_inspect_unreadable_image_precedes_model_resolution \
  -q
```

Required:

```text
2 passed
exit 0
```

Frozen result:

```text
unreadable supported-extension inspect image
=> E202
=> exit 20
=> no model setup first
```

If fail:
- record;
- do not repair;
- STOP.

---

## 7. GATE C — SOURCE MANIFEST CANONICAL IDENTITY

From repository root, validate the manifest using the existing helper semantics:

```python
from scripts.sync_advisor_rc1_delivery import (
    CANONICAL_ROOT,
    entry_source_bytes,
    load_manifest,
    manifest_identity_basis,
)

entries = load_manifest()
basis = manifest_identity_basis()

for entry in entries:
    entry_source_bytes(entry, basis, CANONICAL_ROOT)

print("validated", len(entries), "entries", "basis", basis)
```

Required:

```text
basis = GIT_CANONICAL_BLOB_BYTES
entries = 135
failures = 0
exit 0
```

Also verify:

```text
ordered path digest =
7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd
```

Do NOT rewrite the manifest in R1-E.

If mismatch:
- record;
- STOP.

---

## 8. GATE D — COMPLETE CANONICAL PYTEST SUITE

From:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

run exactly:

```text
python -m pytest -q
```

Required:

```text
exit 0
all collected tests PASS
0 failed
0 errors
```

Do NOT require a pre-guessed fixed test count.
Record:
- collected/passed count reported by pytest;
- skipped/xfailed count if any;
- duration;
- exit code.

If any failure/error occurs:
- do NOT repair it;
- capture failing node IDs and concise failure reasons;
- persist STOP state;
- await ChatGPT.

Do not use:
- `-k`;
- `--lf`;
- selective exclusion;
- xfail/skip edits;
- retries that hide a reproducible failure.

One re-run of the exact same full-suite command is allowed ONLY if the first run is clearly interrupted by a non-test environmental event (e.g. terminal interruption), and the reason must be recorded. Do not re-run merely to chase a green result.

---

## 9. NO MODEL / SCIENTIFIC CHANGES

Do NOT:
- run training;
- tune thresholds;
- change models;
- change detector/reference/mask algorithms;
- run locked-case inference as a repair experiment;
- modify scientific conclusions.

This is engineering regression only.

Semantic target correctness remains:

```text
NOT ESTABLISHED by runtime SUCCESS
```

MASK threshold-based semantic repair remains rejected.

---

## 10. NO EXTERNAL RC1 SYNC

Do NOT write/sync:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

External controlled sync is D2 and occurs only after ChatGPT accepts R1-E.

---

## 11. POST-TEST WORKTREE AUDIT

After all gates, run:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Canonical tracked files must have no task-induced modifications.

If tests generated untracked/cache/log files:
- do not stage;
- do not delete with `clean`;
- record them in the report.

Only authorized report/handoff files may be staged.

---

## 12. CLOSURE DECISION TO RECORD

If Gates A/B/C/D all PASS, report:

```text
MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE = COMPLETE

SUCCESS_SEMANTICS_HARDENING_V1 = IMPLEMENTED_CANONICAL

MASK01_ENGINEERING_HARDENING_CHAIN = CLOSED

runtime_SUCCESS_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED

inspect_proposals_contract = RESOLVED
source_manifest = CANONICAL_AND_VALIDATED

external_sync = NOT_YET_PERFORMED
next_gate = CHATGPT_R1E_REMOTE_AUDIT
```

Important:

```text
MASK01 engineering hardening CLOSED
!=
semantic target correctness solved
```

Do not overclaim scientific success.

---

## 13. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1e_canonical_regression_closure.md
evaluation/task8b3_mask01_d1_r1e_canonical_regression_closure.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report at least:

```text
task_id = MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE
status

starting branch/head
task branch

Gate A command
Gate A exit/result

Gate B command
Gate B exit/result

Gate C identity basis
Gate C entry count
Gate C ordered path digest
Gate C result

Gate D command
Gate D exit
Gate D collected/passed/skipped/xfailed/failed/errors
Gate D duration
Gate D failing nodes if any

canonical_tracked_files_modified_by_task = false
generated_untracked_files = [...]

model_inference = false
training = false
external_write = false

success_semantics_hardening_v1 =
IMPLEMENTED_CANONICAL if all gates pass, else NOT_CLOSED

mask01_engineering_hardening_chain =
CLOSED if all gates pass, else OPEN

semantic_target_correctness =
NOT_ESTABLISHED_BY_RUNTIME_SUCCESS

github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1E_REMOTE_AUDIT
```

---

## 14. EVERY OUTCOME MUST BE PUSHED

Whether COMPLETE / STOP / FAILED:
- update FROM_DSH;
- create report/evidence;
- commit authorized record state;
- push the task branch.

No result may remain local-only.

---

## 15. DIFF GATE

Only stage:

```text
docs/task8b3_mask01_d1_r1e_canonical_regression_closure.md
evaluation/task8b3_mask01_d1_r1e_canonical_regression_closure.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do NOT stage any canonical product/test/manifest file.

Unexpected tracked mutation:
- do not restore/delete automatically;
- record;
- STOP.

---

## 16. COMMIT / PUSH

If all four gates PASS, commit exactly:

```text
git commit -m "test(rc1): close success semantics canonical regression"
```

If STOP/FAILED, use a truthful status commit message and push.

Push:

```text
fix/task8b3-mask01-success-semantics-r1e
```

No force push.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

Do NOT enter D2.

---

## 17. SUCCESS DEFINITION

```text
MASK01_D1_R1E_CANONICAL_REGRESSION_CLOSURE = COMPLETE

Gate A targeted SUCCESS semantics = 5/5 PASS
Gate B inspect contract = 2/2 PASS
Gate C manifest Git canonical identity = PASS
Gate D complete canonical pytest = ALL PASS

canonical tracked product/test/manifest files = unchanged during R1-E

SUCCESS_SEMANTICS_HARDENING_V1 = IMPLEMENTED_CANONICAL
MASK01_ENGINEERING_HARDENING_CHAIN = CLOSED

semantic target correctness = NOT ESTABLISHED by SUCCESS
external RC1 sync = pending D2

NEXT = CHATGPT_R1E_REMOTE_AUDIT
```

Then STOP.
