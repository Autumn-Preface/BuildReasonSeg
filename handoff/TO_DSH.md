# TO_DSH — MASK01_D1_R1E1_TARGETED_CANONICAL_GATES

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1e`
> Required starting HEAD: `5c94163097605bb05d0373f12f7c6cb3d5681d9f`
> New task branch: `fix/task8b3-mask01-success-semantics-r1e1`

## 0. WHY THIS TASK EXISTS

Previous R1-E STOP is accepted as a truthful snapshot, but its gate execution was invalid.

Remote evidence shows Gate A/B/C/D were all executed as:

```text
python -m pytest -q
```

with the same result:

```text
16 failed, 108 passed, 6 errors
```

Therefore:
- Gate A targeted SUCCESS-semantics tests were NOT actually run as specified.
- Gate B targeted inspect tests were NOT actually run as specified.
- Gate C manifest validation was NOT actually run.
- The full-suite failures cannot yet be attributed to MASK-01.

This corrective task runs ONLY the three targeted canonical gates A/B/C.
It MUST NOT run the full suite.

No code/test/manifest/doc repair is authorized.

---

## 1. GIT PREFLIGHT

Verify exactly:

```text
current branch = fix/task8b3-mask01-success-semantics-r1e
HEAD = 5c94163097605bb05d0373f12f7c6cb3d5681d9f
```

Create:

```text
fix/task8b3-mask01-success-semantics-r1e1
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

No reset/rebase/amend/stash/clean/force-push.

---

## 2. ALLOWED CHANGES

Only task records:

```text
docs/task8b3_mask01_d1_r1e1_targeted_canonical_gates.md
evaluation/task8b3_mask01_d1_r1e1_targeted_canonical_gates.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

DO NOT MODIFY ANY FILE UNDER:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

Do not modify sync script or external RC1.

---

## 3. ABSOLUTE COMMAND RULE

DO NOT run:

```text
python -m pytest -q
```

in this task.

DO NOT substitute a full suite for any gate.

DO NOT use `-k`, `--lf`, xfail, skip, retries, or test edits.

Each command below is ONE SINGLE-LINE COMMAND. Execute exactly as written.

---

## 4. GATE A — SUCCESS SEMANTICS

Working directory:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

Execute exactly this ONE line:

```text
python -m pytest tests/test_cli_contract.py::test_r1a_single_success_semantics_block tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary tests/test_task8b_runtime.py::test_success_semantics_contract_exact tests/test_task8b_runtime.py::test_pipeline_result_ok_is_status_compatibility -q
```

Required:

```text
exit = 0
5 passed
```

Record exact stdout/stderr summary.

If Gate A fails:
- DO NOT repair;
- still proceed to Gate B and Gate C for diagnostic completeness;
- final status = STOP.

---

## 5. GATE B — INSPECT CONTRACT

Still in:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

Execute exactly this ONE line:

```text
python -m pytest tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt tests/test_cli_contract.py::test_inspect_unreadable_image_precedes_model_resolution -q
```

Required:

```text
exit = 0
2 passed
```

If Gate B fails:
- DO NOT repair;
- still execute Gate C;
- final status = STOP.

---

## 6. GATE C — MANIFEST GIT CANONICAL IDENTITY

Return to repository root.

Execute exactly this ONE line:

```text
python -c "from scripts.sync_advisor_rc1_delivery import CANONICAL_ROOT,entry_source_bytes,load_manifest,manifest_identity_basis; import hashlib; entries=load_manifest(); basis=manifest_identity_basis(); [entry_source_bytes(e,basis,CANONICAL_ROOT) for e in entries]; digest=hashlib.sha256('\n'.join(e['path'] for e in entries).encode('utf-8')).hexdigest(); print('basis='+str(basis)); print('entries='+str(len(entries))); print('ordered_path_digest='+digest)"
```

Required exact facts:

```text
exit = 0
basis=GIT_CANONICAL_BLOB_BYTES
entries=135
ordered_path_digest=7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd
```

Any `entry_source_bytes()` mismatch must make this gate fail.

DO NOT rewrite the manifest.

---

## 7. NO FULL SUITE

This task ends after Gate C.

Explicitly forbidden:

```text
python -m pytest -q
```

The full canonical suite is a separate next task after ChatGPT audits A/B/C.

---

## 8. WORKTREE AUDIT

After Gate C:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

No canonical tracked file may be modified.

Generated caches/untracked test files:
- do not stage;
- do not delete with clean;
- record them.

---

## 9. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1e1_targeted_canonical_gates.md
evaluation/task8b3_mask01_d1_r1e1_targeted_canonical_gates.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Record exactly:

```text
task_id = MASK01_D1_R1E1_TARGETED_CANONICAL_GATES
status

starting branch/head
task branch

Gate A exact command
Gate A exit
Gate A summary

Gate B exact command
Gate B exit
Gate B summary

Gate C exact command
Gate C exit
Gate C basis
Gate C entries
Gate C ordered_path_digest

full_suite_executed = false
canonical_tracked_files_modified = false
generated_untracked_files = [...]

model_inference = false
training = false
external_write = false
repairs_attempted = false

targeted_contracts_passed =
true only if A/B/C all pass

next_gate = CHATGPT_R1E1_REMOTE_AUDIT
```

---

## 10. EVERY OUTCOME MUST BE PUSHED

Whether COMPLETE / STOP / FAILED:
- update report/evidence/FROM_DSH;
- commit authorized record files;
- push branch.

Do not leave results local-only.

---

## 11. COMMIT / PUSH

If A/B/C all pass, commit exactly:

```text
git commit -m "test(rc1): verify targeted canonical contracts"
```

If any gate fails, use a truthful STOP commit message.

Push:

```text
fix/task8b3-mask01-success-semantics-r1e1
```

No force push.

Then print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

---

## 12. SUCCESS DEFINITION

```text
MASK01_D1_R1E1_TARGETED_CANONICAL_GATES = COMPLETE

Gate A = 5/5 PASS
Gate B = 2/2 PASS
Gate C = PASS

basis = GIT_CANONICAL_BLOB_BYTES
entries = 135
ordered_path_digest =
7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd

full suite = NOT RUN
canonical tracked files = unchanged

NEXT = CHATGPT_R1E1_REMOTE_AUDIT
```

Then STOP.
