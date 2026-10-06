# Task MASK01_D1_R1B_R2_EXACT_DOC_REPLACEMENT

## 1. Task record

```text
task_id        = MASK01_D1_R1B_R2_EXACT_DOC_REPLACEMENT
status         = COMPLETE
starting branch/head = fix/task8b3-mask01-success-semantics-r1b-r1 / 79fc7e859cb0acbce2e86658cb65630026a20d1b
task branch    = fix/task8b3-mask01-success-semantics-r1b-r2
documents_modified = 4
```

## 2. Global validation (three contiguous markers per document)

| document | RUNTIME_STRUCTURAL_ONLY | NOT_EVALUATED | semantic target correctness is not established | result |
|---|---|---|---|---|
| `README.md` | True | True | True | PASS |
| `docs/model_card.md` | True | True | True | PASS |
| `docs/runtime_mapping.md` | True | True | True | PASS |
| `inference/README.md` | True | True | True | PASS |

## 3. File-specific validation

| document | result | missing markers |
|---|---|---|
| `README.md` | PASS | NONE |
| `docs/model_card.md` | PASS | NONE |
| `docs/runtime_mapping.md` | PASS | NONE |
| `inference/README.md` | PASS | NONE |

## 4. Replacement provenance

```text
each document's final `## SUCCESS semantics (frozen)` section was replaced with the exact block published in the task
book (README §3, model_card §4, runtime_mapping §5, inference README §6); no wording was invented or summarized
```

## 5. Freeze and prohibitions

```text
scientific_metrics_changed = false · product_source_changed = false · tests_changed = false
manifest_changed = false · inspect_proposals_changed = false
pytest = not run · model_inference = false · training = false · external_write = false
manifest_update = DEFERRED_TO_R1_D · known issue = PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_R1B_R2_REMOTE_AUDIT
```
