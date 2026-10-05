# Task MASK01_D1_R1A_R1_CLI_CONTRACT_CORRECTION

## 1. Git identity

```text
starting branch = fix/task8b3-mask01-success-semantics-r1a
starting head   = 1e2282151aa8c0c3cbbdc63526d8e449dc7b655d
task branch     = fix/task8b3-mask01-success-semantics-r1a-r1
final commit sha = POST_COMMIT_EXTERNAL_FACT
```

## 2. Corrected CLI contract

```text
single-image SUCCESS:
Result       : SUCCESS
Validity     : RUNTIME_STRUCTURAL_ONLY
Semantic     : NOT_EVALUATED
Note         : SUCCESS only confirms the current runtime structural checks; semantic target correctness is not established.
single-image failure: only `Result       : <status>` (no semantic claims)
batch summary: `Runtime success: <n>`
helpers: success_report_lines(payload), batch_success_line(count) in predict.py
```

## 3. Tests

```text
command = python -m pytest tests/test_cli_contract.py -q -k "not test_predict_inspect_proposals_does_not_require_prompt"
exit    = 0
summary = 23 passed, 1 deselected in 12.73s
source-string checks replaced by deterministic behaviour tests over fake result payloads
excluded known-broken test = test_predict_inspect_proposals_does_not_require_prompt (pre-existing 30-vs-20 defect, untouched)
```

## 4. Untouched files

```text
pipeline.py / source_manifest.json / tests/test_task8b_runtime.py / canonical docs = unchanged (True)
inspect-proposals 30-vs-20 defect = NOT fixed (out of scope)
model inference / training / external sync = NOT executed
```

## 5. Disposition

```text
task_status = COMPLETE
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
