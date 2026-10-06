# TO_DSH — MASK01_D1_R1A_R3_PRODUCT_SUFFIX_ONLY

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1a-r2`
> Required starting HEAD: `aef9ff6fac3c9626130ddf96b3311eada601c9c3`
> New task branch: `fix/task8b3-mask01-success-semantics-r1a-r3-product`

## 0. PURPOSE

This is an intentionally tiny product-only correction.

ChatGPT independently audited R1-A-R2 and accepts its pushed state as a truthful STOP snapshot.

Current accepted partial facts:

```text
successes += 1 = restored
single Note duplication = fixed
batch summary = Runtime success: <n>
batch success suffix = still missing
```

This task fixes only the missing batch success suffix.

Do not edit tests in this task.

---

## 1. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1a-r2

HEAD =
aef9ff6fac3c9626130ddf96b3311eada601c9c3
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
fix/task8b3-mask01-success-semantics-r1a-r3-product
```

No reset/rebase/amend/stash/clean/force-push.

---

## 2. EXACT PRODUCT CHANGE

Only modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
```

In `_run_batch()`, current successful branch is:

```python
if result.ok:
    successes += 1
    print(f"[{index}/{len(files)}] {path.name} ... SUCCESS")
```

Change only the print line so the final block is exactly:

```python
if result.ok:
    successes += 1
    print(
        f"[{index}/{len(files)}] {path.name} ... "
        "SUCCESS [runtime-only; semantic=NOT_EVALUATED]"
    )
```

Equivalent one-line formatting is allowed, but emitted text must be exactly:

```text
[<index>/<total>] <filename> ... SUCCESS [runtime-only; semantic=NOT_EVALUATED]
```

Do not change any other executable logic.

---

## 3. FROZEN CODE — DO NOT TOUCH

Do NOT alter:

```text
successes += 1
batch_success_line()
Runtime success: <n>
success_report_lines()
_report_single()
failure presentation
batch exit-code logic
inspect-proposals
```

Do NOT modify any test file.

Do NOT modify:

```text
pipeline.py
source_manifest.json
canonical docs
test_task8b_runtime.py
configs
models
weights
external RC1
```

---

## 4. VALIDATION

Run only:

```text
python -m py_compile predict.py
```

from:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

Required:

```text
exit 0
```

Do NOT run pytest in this task.

The exact source diff will be independently audited by ChatGPT on GitHub.

---

## 5. REPORTING PATHS

In addition to `predict.py`, only these task records may change:

```text
docs/task8b3_mask01_d1_r1a_r3_product_suffix_only.md
evaluation/task8b3_mask01_d1_r1a_r3_product_suffix_only.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Report:

```text
task_id = MASK01_D1_R1A_R3_PRODUCT_SUFFIX_ONLY
status

starting branch/head
task branch

batch_success_counter_preserved = true
batch_success_suffix_restored = true
batch_summary_preserved = Runtime success: <n>
single_report_logic_changed = false
tests_changed = false

py_compile command/result

inspect_proposals_changed = false
pipeline_changed = false
manifest_changed = false
canonical_docs_changed = false
model_inference = false
training = false
external_write = false

github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1A_R3_PRODUCT_AUDIT
```

---

## 6. EVERY OUTCOME MUST BE PUSHED

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

update `FROM_DSH.md`, report/evidence, commit authorized state, and push.

---

## 7. DIFF GATE

Before commit:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Only these may be staged:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
docs/task8b3_mask01_d1_r1a_r3_product_suffix_only.md
evaluation/task8b3_mask01_d1_r1a_r3_product_suffix_only.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected paths:
- do not delete;
- do not stage;
- report them.

---

## 8. COMMIT / PUSH

Commit exactly once:

```text
git commit -m "fix(rc1): restore batch success annotation"
```

Push:

```text
fix/task8b3-mask01-success-semantics-r1a-r3-product
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

## 9. ABSOLUTE PROHIBITIONS

Do NOT:

```text
modify tests
fix test fixtures
run pytest

modify pipeline.py
modify manifest
modify canonical docs

fix inspect-proposals
change E202/E3xx behavior

run model inference
run training
run full suite
write/sync external RC1

change algorithms
add thresholds

reset
rebase
amend
stash
clean
force-push

start the next test-only task
start R1-B/R1-C/R1-D/R1-E/D2
```

---

## 10. SUCCESS DEFINITION

Expected terminal state:

```text
MASK01_D1_R1A_R3_PRODUCT_SUFFIX_ONLY = COMPLETE

predict.py successful batch branch:
successes += 1
SUCCESS [runtime-only; semantic=NOT_EVALUATED]

batch summary:
Runtime success: <n>

single-image contract = unchanged from R1-A-R2
tests = unchanged
py_compile = PASS

NEXT = CHATGPT_R1A_R3_PRODUCT_AUDIT
```

Then STOP.
