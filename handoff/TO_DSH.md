# TO_DSH — MASK01_D1_R1B_R1_CANONICAL_DOCS_CORRECTION

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1b`
> Required starting HEAD: `ae08e405778672010168124196a4cadab90414c0`
> New task branch: `fix/task8b3-mask01-success-semantics-r1b-r1`

## 0. PURPOSE

Correct the incomplete R1-B documentation implementation only.

ChatGPT audit of `ae08e405778672010168124196a4cadab90414c0`:

```text
R1-B state persistence = accepted
R1-B implementation = NOT ACCEPTED
task status = STOP
```

All four canonical documents were updated, but:
- the exact phrase `semantic target correctness is not established` is not contiguous in any file;
- the four files do not yet contain their file-specific required semantics.

No runtime/product/test behavior may change.

---

## 1. GIT PREFLIGHT

Verify:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1b

HEAD =
ae08e405778672010168124196a4cadab90414c0
```

Create:

```text
fix/task8b3-mask01-success-semantics-r1b-r1
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

No reset/rebase/amend/stash/clean/force-push.

---

## 2. ALLOWED PATHS

Only these canonical documents:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
```

and task records:

```text
docs/task8b3_mask01_d1_r1b_r1_canonical_docs_correction.md
evaluation/task8b3_mask01_d1_r1b_r1_canonical_docs_correction.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do NOT modify:
- `predict.py`
- `runtime/pipeline.py`
- any tests
- `source_manifest.json`
- model/config/weights
- inference outputs
- external RC1.

---

## 3. GLOBAL FROZEN SEMANTICS

Every one of the four canonical docs must contain these exact contiguous strings:

```text
RUNTIME_STRUCTURAL_ONLY
NOT_EVALUATED
semantic target correctness is not established
```

Do not line-wrap the third phrase across lines in source text.

Frozen meaning:

```text
status = SUCCESS
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

`SUCCESS` means only that the current RC1 runtime completed and passed its structural checks.

It does NOT establish semantic target correctness.

Do not claim MASK-01 semantic correctness is solved.

---

## 4. README.md — FILE-SPECIFIC REQUIREMENTS

Keep the existing appended SUCCESS section, but correct/expand it so it explicitly states all of:

```text
status="SUCCESS" is retained for compatibility
validity_scope="RUNTIME_STRUCTURAL_ONLY"
semantic_status="NOT_EVALUATED"
semantic target correctness is not established
```

Also explicitly document the single-image CLI presentation:

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
```

And state that this change is:

```text
a status-contract clarification, not a model-quality or mask-quality repair
```

Do not rewrite unrelated README material.

---

## 5. docs/model_card.md — FILE-SPECIFIC REQUIREMENTS

Keep/add a concise `Runtime SUCCESS` subsection.

It must explicitly distinguish:

```text
runtime structural validity
```

from:

```text
semantic target correctness
```

Include exact fields:

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

Include exact contiguous phrase:

```text
semantic target correctness is not established
```

Do not change any Task 7I/7J metric, seed, model claim, language metric, or domain/generalization statement.

---

## 6. docs/runtime_mapping.md — FILE-SPECIFIC REQUIREMENTS

The SUCCESS section must explicitly record responsibility split:

```text
runtime/pipeline.py
  owns the machine-readable SUCCESS semantics

predict.py
  owns the user-facing CLI presentation
```

Document:

```text
status = SUCCESS
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

Also state explicitly:

```text
PipelineResult.ok remains compatible with status == SUCCESS
```

and:

```text
semantic target correctness is not established
```

Do not alter the source-mapping table except for this appended semantics section.

---

## 7. inference/README.md — FILE-SPECIFIC REQUIREMENTS

Add/correct the result-semantics note so it explicitly refers to normal-inference `result.json`.

It must state that normal inference `result.json` with:

```text
status = SUCCESS
```

also carries:

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

State:

```text
semantic target correctness is not established
```

Do not redesign the inference directory.

Do not fix unrelated historical Task 8A wording in this task.

---

## 8. SCIENTIFIC CLAIMS — FROZEN

Do not modify:
- Task 7I metrics
- Task 7J metrics
- oracle/predicted-reference conclusions
- REF-01 / PROP-01
- A1–B2 history
- Demo policy
- language robustness
- domain/generalization statements
- model/seed/threshold/architecture claims.

No new performance claims.

---

## 9. INSPECT-PROPOSALS ISSUE — OUT OF SCOPE

Do not touch or hide:

```text
test_predict_inspect_proposals_does_not_require_prompt
```

Frozen classification:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

No E202/E3xx or initialization-order changes.

---

## 10. MANIFEST

Do NOT update:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Record:

```text
manifest_update = DEFERRED_TO_R1_D
```

---

## 11. VALIDATION — EXACT

Do NOT run pytest.

Run a deterministic source-text validation.

For EACH of the four canonical files, require all three exact contiguous markers:

```text
RUNTIME_STRUCTURAL_ONLY
NOT_EVALUATED
semantic target correctness is not established
```

Required:

```text
README.md = PASS
docs/model_card.md = PASS
docs/runtime_mapping.md = PASS
inference/README.md = PASS
```

Also run file-specific marker checks:

### README
Require:

```text
status="SUCCESS"
Validity
Semantic
status-contract
mask-quality
```

Equivalent punctuation around `status="SUCCESS"` is acceptable only if the literal `status="SUCCESS"` appears.

### model_card
Require both contiguous concepts:

```text
runtime structural validity
semantic target correctness
```

### runtime_mapping
Require:

```text
runtime/pipeline.py
predict.py
PipelineResult.ok
```

### inference/README
Require:

```text
result.json
status = SUCCESS
```

Validation must PASS before COMPLETE status.

Also verify no product/test/manifest path changed.

---

## 12. EVERY OUTCOME MUST BE PUSHED

Frozen rule:

```text
ALL TASK OUTCOMES MUST BE PERSISTED TO GITHUB
```

Whether COMPLETE / STOP / FAILED:
- update `handoff/FROM_DSH.md`;
- create report/evidence;
- commit authorized state;
- push branch.

---

## 13. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1b_r1_canonical_docs_correction.md
evaluation/task8b3_mask01_d1_r1b_r1_canonical_docs_correction.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report at least:

```text
task_id = MASK01_D1_R1B_R1_CANONICAL_DOCS_CORRECTION
status
starting branch/head
task branch

global_marker_validation:
README = PASS/FAIL
model_card = PASS/FAIL
runtime_mapping = PASS/FAIL
inference_README = PASS/FAIL

README_specific_validation = PASS/FAIL
model_card_specific_validation = PASS/FAIL
runtime_mapping_specific_validation = PASS/FAIL
inference_README_specific_validation = PASS/FAIL

scientific_metrics_changed = false
product_source_changed = false
tests_changed = false
manifest_changed = false
inspect_proposals_changed = false

model_inference = false
training = false
external_write = false

manifest_update = DEFERRED_TO_R1_D
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1B_R1_REMOTE_AUDIT
```

---

## 14. DIFF GATE

Only stage:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
docs/task8b3_mask01_d1_r1b_r1_canonical_docs_correction.md
evaluation/task8b3_mask01_d1_r1b_r1_canonical_docs_correction.json
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
git commit -m "docs(rc1): complete runtime success semantics docs"
```

Push:

```text
fix/task8b3-mask01-success-semantics-r1b-r1
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

## 16. ABSOLUTE PROHIBITIONS

Do NOT:
- modify product code;
- modify tests;
- modify manifest;
- change scientific results;
- fix inspect-proposals;
- run pytest;
- run model inference/training;
- sync external RC1;
- reset/rebase/amend/stash/clean/force-push;
- start R1-C/R1-D/R1-E/D2.

---

## 17. SUCCESS DEFINITION

```text
MASK01_D1_R1B_R1_CANONICAL_DOCS_CORRECTION = COMPLETE

all four global marker checks = PASS
all four file-specific checks = PASS

README:
compatibility + CLI presentation + status-not-quality clarification documented

model_card:
runtime structural validity vs semantic target correctness documented

runtime_mapping:
pipeline/predict responsibility split + PipelineResult.ok compatibility documented

inference README:
result.json machine fields documented

scientific claims = unchanged
product/tests = unchanged
manifest = deferred to R1-D
inspect-proposals issue = unchanged

NEXT = CHATGPT_R1B_R1_REMOTE_AUDIT
```

Then STOP.
