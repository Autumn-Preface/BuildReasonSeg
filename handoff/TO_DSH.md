# TO_DSH — MASK01_D1_R1B_R2_EXACT_DOC_REPLACEMENT

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1b-r1`
> Required starting HEAD: `79fc7e859cb0acbce2e86658cb65630026a20d1b`
> New task branch: `fix/task8b3-mask01-success-semantics-r1b-r2`

## 0. EXECUTION RULE

READ THIS ENTIRE FILE BEFORE EDITING ANYTHING.

This task intentionally gives the exact replacement text for all four canonical documents.

Do NOT invent wording.
Do NOT summarize the instructions.
Do NOT stop because later sections were not read.
Do NOT restore/reset/rebase/amend history.

The four documents currently end with the same generic section:

```text
## SUCCESS semantics (frozen)
...
```

For each document, replace ONLY that final generic SUCCESS section with the exact document-specific block supplied below.

Do not edit earlier content.

---

## 1. CURRENT AUDITED STATE

Starting branch:

```text
fix/task8b3-mask01-success-semantics-r1b-r1
```

Starting HEAD:

```text
79fc7e859cb0acbce2e86658cb65630026a20d1b
```

Parent:

```text
ae08e405778672010168124196a4cadab90414c0
```

R1-B-R1 status:

```text
STOP
documents modified = NONE
```

Therefore the four canonical docs are still exactly the R1-B versions with the identical generic section appended at the end.

Create:

```text
fix/task8b3-mask01-success-semantics-r1b-r2
```

---

## 2. ALLOWED PATHS

Only modify these four canonical docs:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
```

And task records:

```text
docs/task8b3_mask01_d1_r1b_r2_exact_doc_replacement.md
evaluation/task8b3_mask01_d1_r1b_r2_exact_doc_replacement.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Forbidden:

```text
predict.py
runtime/pipeline.py
tests/*
source_manifest.json
configs
models
weights
inference outputs
external RC1
```

No pytest.
No model inference.
No training.
No external sync.

---

## 3. README.md — EXACT REPLACEMENT BLOCK

File:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
```

Find the FINAL section beginning:

```text
## SUCCESS semantics (frozen)
```

Replace that entire final section with exactly:

```markdown
## SUCCESS 状态语义

为保持既有接口兼容，正常推理仍保留 `status="SUCCESS"`。这里的 `SUCCESS` 只表示 RC1 runtime 已完成并通过当前的结构性检查，不表示目标建筑已经在语义上被正确识别或分割。

机器可读字段固定为：

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

单图 CLI 对应显示：

```text
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
```

因此，semantic target correctness is not established。

这属于 `status-contract` 的语义澄清，不是模型性能修复，也不是 `mask-quality` 修复。
```

Do not change any earlier README content.

---

## 4. docs/model_card.md — EXACT REPLACEMENT BLOCK

File:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Find the FINAL section beginning:

```text
## SUCCESS semantics (frozen)
```

Replace that entire final section with exactly:

```markdown
## Runtime SUCCESS 语义

RC1 必须区分 runtime structural validity 与 semantic target correctness。

机器可读字段固定为：

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

`SUCCESS` 只确认当前 runtime structural validity：即推理链完成并通过现有结构性检查。它不是语义质量判定，因此 semantic target correctness is not established。

这一状态契约不改变 Task 7I / Task 7J 的任何指标、模型选择、seed、reference-selection 结论或泛化边界。
```

Do not change any metric or earlier model-card content.

---

## 5. docs/runtime_mapping.md — EXACT REPLACEMENT BLOCK

File:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
```

Find the FINAL section beginning:

```text
## SUCCESS semantics (frozen)
```

Replace that entire final section with exactly:

```markdown
## SUCCESS 语义职责映射

`runtime/pipeline.py` 负责 machine-readable SUCCESS semantics；`predict.py` 负责 user-facing CLI presentation。

冻结的机器字段为：

```text
status = SUCCESS
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

兼容性保持不变：`PipelineResult.ok` remains compatible with status == SUCCESS。

这里的 `SUCCESS` 仅表示当前结构性 runtime checks 已通过；semantic target correctness is not established。
```

Do not alter the mapping table or earlier content.

---

## 6. inference/README.md — EXACT REPLACEMENT BLOCK

File:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
```

Find the FINAL section beginning:

```text
## SUCCESS semantics (frozen)
```

Replace that entire final section with exactly:

```markdown
## Runtime result semantics

正常推理生成的 `result.json` 若记录：

```text
status = SUCCESS
```

则同时携带：

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

这只表示当前 RC1 runtime 已完成并通过现有结构性检查；semantic target correctness is not established。

因此 `result.json` 中的 `SUCCESS` 不能被解释为目标建筑在语义上已经正确匹配，也不能被解释为 mask 质量验收通过。
```

Do not redesign the directory description or alter earlier content.

---

## 7. GLOBAL VALIDATION — REQUIRED

After editing, each of the four files MUST contain these exact contiguous strings:

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

The third phrase MUST exist as one contiguous source-text string.
Do not line-wrap inside that phrase.

---

## 8. FILE-SPECIFIC VALIDATION — REQUIRED

### README.md

Require exact source markers:

```text
status="SUCCESS"
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
status-contract
mask-quality
```

### docs/model_card.md

Require:

```text
runtime structural validity
semantic target correctness
RUNTIME_STRUCTURAL_ONLY
NOT_EVALUATED
```

### docs/runtime_mapping.md

Require:

```text
runtime/pipeline.py
predict.py
PipelineResult.ok
status = SUCCESS
RUNTIME_STRUCTURAL_ONLY
NOT_EVALUATED
```

### inference/README.md

Require:

```text
result.json
status = SUCCESS
RUNTIME_STRUCTURAL_ONLY
NOT_EVALUATED
```

All four file-specific validations must PASS.

---

## 9. SCIENTIFIC / ENGINEERING FREEZE

Do not change or reinterpret:

```text
Task 7I metrics
Task 7J metrics
oracle/predicted-reference conclusions
PROP-01
REF-01
A1/A2/A3/A4/B1/B2 history
Demo policy
language robustness
domain/generalization statements
model/seed/threshold/architecture
```

Do not claim MASK-01 semantic correctness is solved.

Do not modify inspect-proposals behavior or docs to hide its known issue.

Known issue remains:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

---

## 10. MANIFEST

Do NOT modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Record:

```text
manifest_update = DEFERRED_TO_R1_D
```

Manifest mismatch caused by doc changes is expected until R1-D.

---

## 11. VALIDATION COMMAND POLICY

Do NOT run pytest.

Run deterministic text checks only.

You may use Python or shell text inspection to verify Sections 7 and 8.

Also verify changed product/test paths are exactly:

```text
NONE
```

outside the four allowed docs.

---

## 12. EVERY OUTCOME MUST BE PUSHED

Frozen collaboration rule:

```text
ALL TASK OUTCOMES MUST BE PERSISTED TO GITHUB
```

Whether status is COMPLETE / STOP / FAILED:

- update `handoff/FROM_DSH.md`;
- create report/evidence;
- commit authorized state;
- push branch.

Do not leave task facts only locally.

---

## 13. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1b_r2_exact_doc_replacement.md
evaluation/task8b3_mask01_d1_r1b_r2_exact_doc_replacement.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report at least:

```text
task_id = MASK01_D1_R1B_R2_EXACT_DOC_REPLACEMENT
status

starting branch/head
task branch

documents_modified = 4

global_validation:
README = PASS/FAIL
model_card = PASS/FAIL
runtime_mapping = PASS/FAIL
inference_README = PASS/FAIL

file_specific_validation:
README = PASS/FAIL
model_card = PASS/FAIL
runtime_mapping = PASS/FAIL
inference_README = PASS/FAIL

scientific_metrics_changed = false
product_source_changed = false
tests_changed = false
manifest_changed = false
inspect_proposals_changed = false

pytest = not run
model_inference = false
training = false
external_write = false

manifest_update = DEFERRED_TO_R1_D
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1B_R2_REMOTE_AUDIT
```

---

## 14. DIFF GATE

Before commit, only these may be staged:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/runtime_mapping.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
docs/task8b3_mask01_d1_r1b_r2_exact_doc_replacement.md
evaluation/task8b3_mask01_d1_r1b_r2_exact_doc_replacement.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected files:
- do not delete;
- do not stage;
- report them.

---

## 15. COMMIT / PUSH

If the required doc validations all PASS, commit exactly:

```text
git commit -m "docs(rc1): finalize runtime success semantics docs"
```

If STOP/FAILED occurs, still commit the truthful authorized state with a truthful STOP/FAILED message and push.

Push:

```text
fix/task8b3-mask01-success-semantics-r1b-r2
```

No force push.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

Do NOT enter R1-C.

---

## 16. SUCCESS DEFINITION

Expected:

```text
MASK01_D1_R1B_R2_EXACT_DOC_REPLACEMENT = COMPLETE

four canonical docs modified = YES

all global marker checks = PASS
all file-specific checks = PASS

README:
compatibility + CLI presentation + status-contract-not-mask-quality meaning = documented

model_card:
runtime structural validity vs semantic target correctness = documented

runtime_mapping:
pipeline/predict responsibility split + PipelineResult.ok compatibility = documented

inference README:
result.json machine fields = documented

scientific claims = unchanged
product source = unchanged
tests = unchanged
manifest = deferred to R1-D
inspect-proposals issue = unchanged

NEXT = CHATGPT_R1B_R2_REMOTE_AUDIT
```

Then STOP.
