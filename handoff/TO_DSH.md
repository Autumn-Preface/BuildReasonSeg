# TO_DSH — MASK01_D1_R1C_RUNTIME_CONTRACT_TEST_CLEANUP

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1b-r2`
> Required starting HEAD: `60cacc8a2d4460740ad9849fef00f506ad951549`
> New task branch: `fix/task8b3-mask01-success-semantics-r1c`

## 0. CHATGPT DECISION

R1-B-R2 is accepted and R1-B is CLOSED.

Accepted remote state:

```text
branch = fix/task8b3-mask01-success-semantics-r1b-r2
HEAD   = 60cacc8a2d4460740ad9849fef00f506ad951549
parent = 79fc7e859cb0acbce2e86658cb65630026a20d1b
commit = docs(rc1): finalize runtime success semantics docs
```

All four canonical docs passed:
- global marker validation;
- file-specific validation.

No product source, tests, manifest, metrics, or inspect-proposals behavior changed.

This task is now R1-C: runtime contract test cleanup only.

---

## 1. PURPOSE

Clean up the remaining weak D1 runtime SUCCESS contract test in:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
```

Current weak test contains a tautological assertion equivalent to:

```python
assert "semantic_status" in ("semantic_status", "runtime-only; semantic=")
```

That assertion proves nothing.

R1-C replaces it with:
1. exact `success_semantics()` dictionary behavior;
2. exact `PipelineResult.ok` compatibility behavior.

Do NOT modify production source.

---

## 2. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1b-r2

HEAD =
60cacc8a2d4460740ad9849fef00f506ad951549
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

Create:

```text
fix/task8b3-mask01-success-semantics-r1c
```

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED PATHS

Only modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
```

plus task records:

```text
docs/task8b3_mask01_d1_r1c_runtime_contract_test_cleanup.md
evaluation/task8b3_mask01_d1_r1c_runtime_contract_test_cleanup.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Explicitly forbidden:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
canonical docs
configs
models
weights
external RC1
```

---

## 4. TEST 1 — EXACT SUCCESS SEMANTICS DICT

Replace the existing weak:

```text
test_success_semantics_contract_present
```

with a strong behavioral test named exactly:

```text
test_success_semantics_contract_exact
```

Use the real helper:

```python
from buildreasonseg.runtime.pipeline import success_semantics
```

Assert:

```python
assert success_semantics() == {
    "validity_scope": "RUNTIME_STRUCTURAL_ONLY",
    "semantic_status": "NOT_EVALUATED",
    "semantic_note": (
        "SUCCESS means the RC1 runtime completed and passed its current structural checks; "
        "semantic target correctness is not established."
    ),
}
```

Do not inspect source text.
Do not assert tautologies.
Do not duplicate the implementation in a helper function.

This test must prove the exact public machine-readable contract returned by `success_semantics()`.

---

## 5. TEST 2 — PipelineResult.ok COMPATIBILITY

Add a separate test named exactly:

```text
test_pipeline_result_ok_is_status_compatibility
```

Use the real class:

```python
from buildreasonseg.runtime.pipeline import PipelineResult
```

Create at least these two synthetic instances:

```python
success = PipelineResult(
    status="SUCCESS",
    image=Path("synthetic.png"),
    result_payload={"status": "SUCCESS"},
)

failed = PipelineResult(
    status="FAILED",
    image=Path("synthetic.png"),
    result_payload={"status": "FAILED"},
)
```

Assert:

```python
assert success.ok is True
assert failed.ok is False
```

Also create one non-success arbitrary status, for example:

```python
other = PipelineResult(
    status="NOT_EVALUATED",
    image=Path("synthetic.png"),
    result_payload={},
)
```

Assert:

```python
assert other.ok is False
```

The purpose is to lock the compatibility rule:

```text
PipelineResult.ok == (status == "SUCCESS")
```

Do NOT change `PipelineResult.ok` implementation.

---

## 6. REMOVE WEAK / TAUTOLOGICAL ASSERTIONS

After editing, the old meaningless assertion must be gone:

```text
assert "semantic_status" in ("semantic_status", "runtime-only; semantic=")
```

No replacement tautology is allowed.

Do not add source-string scanning to prove runtime behavior.

---

## 7. TEST GATE — EXACT

From:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

run exactly:

```text
python -m pytest \
  tests/test_task8b_runtime.py::test_success_semantics_contract_exact \
  tests/test_task8b_runtime.py::test_pipeline_result_ok_is_status_compatibility \
  -q
```

Required:

```text
2 passed
exit 0
```

Do NOT run:
- full `test_task8b_runtime.py`;
- full canonical suite;
- `test_cli_contract.py`;
- inspect-proposals node.

---

## 8. PRODUCT / DOCS / MANIFEST FREEZE

Verify unchanged:

```text
predict.py
buildreasonseg/runtime/pipeline.py
tests/test_cli_contract.py
README.md
docs/model_card.md
docs/runtime_mapping.md
inference/README.md
source_manifest.json
```

R1-C is tests-only.

Manifest update remains:

```text
DEFERRED_TO_R1_D
```

---

## 9. INSPECT-PROPOSALS ISSUE — OUT OF SCOPE

Do NOT modify or run:

```text
test_predict_inspect_proposals_does_not_require_prompt
```

Frozen classification:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

No E202/E3xx changes.

---

## 10. EVERY OUTCOME MUST BE PUSHED

Frozen project rule:

```text
ALL TASK OUTCOMES MUST BE PERSISTED TO GITHUB
```

Whether COMPLETE / STOP / FAILED:
- update `handoff/FROM_DSH.md`;
- create report/evidence;
- commit authorized state;
- push branch.

---

## 11. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1c_runtime_contract_test_cleanup.md
evaluation/task8b3_mask01_d1_r1c_runtime_contract_test_cleanup.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report:

```text
task_id = MASK01_D1_R1C_RUNTIME_CONTRACT_TEST_CLEANUP
status

starting branch/head
task branch

old_tautological_assertion_removed = true
success_semantics_exact_dict_test = true
pipeline_result_ok_compatibility_test = true

gate_command
gate_result

product_source_changed = false
cli_tests_changed = false
canonical_docs_changed = false
manifest_changed = false
inspect_proposals_changed = false

model_inference = false
training = false
external_write = false

manifest_update = DEFERRED_TO_R1_D
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1C_REMOTE_AUDIT
```

---

## 12. DIFF GATE

Before commit, only these may be staged:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
docs/task8b3_mask01_d1_r1c_runtime_contract_test_cleanup.md
evaluation/task8b3_mask01_d1_r1c_runtime_contract_test_cleanup.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected paths:
- do not delete;
- do not stage;
- report them.

---

## 13. COMMIT / PUSH

If the 2-node gate passes, commit exactly:

```text
git commit -m "test(rc1): strengthen runtime success semantics contract"
```

If STOP/FAILED occurs, still commit the truthful authorized state with a truthful message and push.

Push:

```text
fix/task8b3-mask01-success-semantics-r1c
```

No force push.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

Do NOT enter R1-D.

---

## 14. ABSOLUTE PROHIBITIONS

Do NOT:
- modify product source;
- modify CLI tests;
- modify canonical docs;
- modify manifest;
- fix inspect-proposals;
- run model inference/training;
- run full suites;
- sync external RC1;
- reset/rebase/amend/stash/clean/force-push;
- start R1-D/R1-E/D2.

---

## 15. SUCCESS DEFINITION

```text
MASK01_D1_R1C_RUNTIME_CONTRACT_TEST_CLEANUP = COMPLETE

weak tautology = removed

success_semantics() exact dict = tested
PipelineResult.ok compatibility = tested

targeted gate = 2/2 PASS

product code = unchanged
CLI tests = unchanged
canonical docs = unchanged
manifest = deferred to R1-D
inspect-proposals issue = unchanged

NEXT = CHATGPT_R1C_REMOTE_AUDIT
```

Then STOP.
