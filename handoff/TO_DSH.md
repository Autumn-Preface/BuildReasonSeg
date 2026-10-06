# TO_DSH — MASK01_D1_R1B_CANONICAL_DOCS

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1a-r6-tests`
> Required starting HEAD: `385a3ba5d87c22712c75e8a22273d5720bd9cfe8`
> New task branch: `fix/task8b3-mask01-success-semantics-r1b`

## 0. CHATGPT AUDIT / FROZEN DECISION

R1-A is now accepted and closed for its intended engineering scope.

Accepted chain culminates at:

```text
branch = fix/task8b3-mask01-success-semantics-r1a-r6-tests
HEAD   = 385a3ba5d87c22712c75e8a22273d5720bd9cfe8
parent = 3f7bfb0956de14284a77dbf1ae19ec9d7977a56f
commit = test(rc1): finalize cli success contract tests
```

Accepted R1-A behavior:

```text
single-image SUCCESS:
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.

batch successful item:
... SUCCESS [runtime-only; semantic=NOT_EVALUATED]

batch summary:
Runtime success: <n>
```

Accepted machine-readable contract remains in `runtime/pipeline.py`:

```text
status = SUCCESS
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
semantic_note = SUCCESS means the RC1 runtime completed and passed its current structural checks; semantic target correctness is not established.
```

Frozen compatibility:

```text
PipelineResult.status == "SUCCESS"
PipelineResult.ok == True
successful exit-code behavior unchanged
```

The R1-A-R6 synthetic fixture used `reference_id=123` and `mask_area=7` instead of the task-book example `7/123`.
This is a non-functional synthetic-test value deviation only; it does not affect the accepted CLI semantics, counting, exit-code behavior, or product code.

Formal decision:

```text
MASK01_D1_R1A = ACCEPTED / CLOSED
CLI SUCCESS SEMANTICS = CLOSED
PIPELINE MACHINE CONTRACT = UNCHANGED / FROZEN
```

Known separate issue remains:

```text
test_predict_inspect_proposals_does_not_require_prompt
expected = E202 / exit 20
observed historical = exit 30
classification = PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

Do NOT touch that issue in R1-B.

---

## 1. PURPOSE

This task is **canonical documentation only**.

It documents the already-implemented `SUCCESS_SEMANTICS_HARDENING_V1` consistently in the four canonical delivery documents.

It does NOT change runtime behavior.

It does NOT change tests.

It does NOT update the manifest yet.

---

## 2. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1a-r6-tests

HEAD =
385a3ba5d87c22712c75e8a22273d5720bd9cfe8
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
fix/task8b3-mask01-success-semantics-r1b
```

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED CANONICAL DOCUMENT CHANGES

All four documents below MUST be updated:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
```

Task records:

```text
docs/task8b3_mask01_d1_r1b_canonical_docs.md
evaluation/task8b3_mask01_d1_r1b_canonical_docs.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Explicitly forbidden:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/*
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
configs
models
weights
generated inference outputs
external RC1
```

---

## 4. FROZEN DOCUMENTATION MEANING

Every one of the four canonical documents must clearly communicate the same distinction:

```text
SUCCESS = runtime structural success only
semantic target correctness = NOT_EVALUATED by RC1 runtime
```

The documentation must explicitly include the exact machine-readable values:

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

And communicate this exact meaning:

```text
semantic target correctness is not established
```

Chinese explanation is allowed around these exact machine tokens.

Do NOT claim or imply:

```text
SUCCESS means the intended building was correctly identified
SUCCESS means the final mask is semantically correct
SUCCESS is a semantic-quality acceptance result
```

---

## 5. README.md — REQUIRED EDIT

In canonical root README, add a concise subsection near the output/runtime behavior discussion.

Suggested heading:

```text
### SUCCESS 状态语义
```

It must state:

1. `status="SUCCESS"` remains for compatibility.
2. It means the RC1 runtime completed and passed its current structural checks.
3. Machine fields:
   - `validity_scope="RUNTIME_STRUCTURAL_ONLY"`
   - `semantic_status="NOT_EVALUATED"`
4. `semantic target correctness is not established`.
5. The CLI exposes the same distinction with:
   - `Validity : RUNTIME_STRUCTURAL_ONLY`
   - `Semantic : NOT_EVALUATED`
6. This is a status-contract clarification, not a model/mask quality repair.

Do not rewrite unrelated sections.

---

## 6. docs/model_card.md — REQUIRED EDIT

Add a concise subsection under the RC1 runtime / known limitations material.

Suggested heading:

```text
### Runtime SUCCESS 语义
```

It must make clear that:

```text
SUCCESS does not establish semantic target correctness.
```

Include:

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

Also explicitly distinguish:

```text
runtime structural validity
vs.
semantic target correctness
```

Do not alter any scientific metric, Task 7I/7J number, seed, model claim, or evaluation interpretation.

---

## 7. docs/runtime_mapping.md — REQUIRED EDIT

Add a short section describing the responsibility split:

```text
runtime/pipeline.py
    owns machine-readable SUCCESS semantics

predict.py
    owns user-facing CLI presentation
```

Document the frozen machine fields:

```text
status = SUCCESS
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

Document that:

```text
PipelineResult.ok remains compatible with status == SUCCESS
```

and that semantic correctness is not evaluated by this status.

Do not change the numerical/source mapping table except if a tiny wording addition is required.

---

## 8. inference/README.md — REQUIRED EDIT

Add a concise runtime result-semantics note.

It must state that a generated normal-inference `result.json` with:

```text
status = SUCCESS
```

also carries:

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

and that this means only current runtime structural checks passed.

Explicitly state:

```text
semantic target correctness is not established
```

Do not use this task to redesign the inference directory or fix unrelated historical wording.

---

## 9. DO NOT CHANGE SCIENTIFIC CLAIMS

Do not modify or reinterpret:

```text
Task 7I metrics
Task 7J metrics
oracle vs predicted-reference conclusions
reference-selection bottleneck
PROP-01
REF-01
A1/A2/A3/A4/B1/B2 history
fixed Demo policy
language robustness metrics
domain/generalization claims
```

Do not introduce new performance claims.

Do not claim MASK-01 semantic correctness is solved.

This task hardens documentation semantics only.

---

## 10. KNOWN INSPECT-PROPOSALS ISSUE — OUT OF SCOPE

Do NOT modify or document away:

```text
test_predict_inspect_proposals_does_not_require_prompt
```

Do not alter:
- E202/E3xx behavior;
- inspect initialization order;
- inspect docs for the purpose of hiding the known failure.

Keep classification:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

---

## 11. MANIFEST

Do NOT update:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Manifest canonicalization remains deferred to R1-D.

Record:

```text
manifest_update = DEFERRED_TO_R1_D
```

Temporary manifest mismatch after doc changes is expected.

---

## 12. VALIDATION — DOCS ONLY

Do NOT run pytest.

Run a deterministic text validation over the four canonical documents.

The validation must prove that **each of the four files** contains all three required semantic markers:

```text
RUNTIME_STRUCTURAL_ONLY
NOT_EVALUATED
semantic target correctness is not established
```

Case-sensitive check for the two machine tokens.

The English semantic phrase may appear as part of an English sentence; keep the exact phrase.

Record a per-file result:

```text
README.md = PASS
docs/model_card.md = PASS
docs/runtime_mapping.md = PASS
inference/README.md = PASS
```

Also verify no product/test source file changed.

---

## 13. EVERY OUTCOME MUST BE PUSHED

Frozen collaboration rule:

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

## 14. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1b_canonical_docs.md
evaluation/task8b3_mask01_d1_r1b_canonical_docs.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report:

```text
task_id = MASK01_D1_R1B_CANONICAL_DOCS
status

starting branch/head
task branch

documents_updated:
- README.md
- docs/model_card.md
- docs/runtime_mapping.md
- inference/README.md

documented_machine_contract:
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED

semantic_phrase =
semantic target correctness is not established

README marker check
model_card marker check
runtime_mapping marker check
inference_README marker check

scientific_metrics_changed = false
product_source_changed = false
tests_changed = false
inspect_proposals_changed = false
manifest_changed = false

model_inference = false
training = false
external_write = false

manifest_update = DEFERRED_TO_R1_D
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1B_REMOTE_AUDIT
```

---

## 15. DIFF GATE

Before commit:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Only these paths may be staged:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
docs/task8b3_mask01_d1_r1b_canonical_docs.md
evaluation/task8b3_mask01_d1_r1b_canonical_docs.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected paths:
- do not delete;
- do not stage;
- report them.

---

## 16. COMMIT / PUSH

Commit exactly once:

```text
git commit -m "docs(rc1): clarify runtime success semantics"
```

Push:

```text
fix/task8b3-mask01-success-semantics-r1b
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

Do not enter R1-C.

---

## 17. ABSOLUTE PROHIBITIONS

Do NOT:

```text
modify predict.py
modify pipeline.py
modify tests
modify source_manifest.json

change scientific metrics
change model claims
change architecture
add thresholds

fix inspect-proposals
change E202/E3xx behavior

run pytest
run model inference
run training
run full suite
write/sync external RC1

reset
rebase
amend
stash
clean
force-push

start R1-C
start R1-D
start R1-E
start D2
```

---

## 18. SUCCESS DEFINITION

Expected terminal state:

```text
MASK01_D1_R1B_CANONICAL_DOCS = COMPLETE

all four canonical docs explicitly state:
RUNTIME_STRUCTURAL_ONLY
NOT_EVALUATED
semantic target correctness is not established

runtime SUCCESS is documented as structural-only
semantic correctness remains not evaluated

scientific claims = unchanged
product code = unchanged
tests = unchanged
inspect-proposals known issue = unchanged
manifest = deferred to R1-D

NEXT = CHATGPT_R1B_REMOTE_AUDIT
```

Then STOP.
