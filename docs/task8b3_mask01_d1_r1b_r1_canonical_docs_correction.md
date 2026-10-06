# Task MASK01_D1_R1B_R1_CANONICAL_DOCS_CORRECTION — STOP (no document edited)

## 1. Git identity

```text
starting branch = fix/task8b3-mask01-success-semantics-r1b
starting head   = ae08e405778672010168124196a4cadab90414c0
task branch     = fix/task8b3-mask01-success-semantics-r1b-r1
final commit sha = POST_COMMIT_EXTERNAL_FACT
```

## 2. Why this task stopped before editing

The task requires **file-specific** SUCCESS-semantics content for each of the four canonical documents (book sections
4-7) and then the book's **global marker checks** plus **four file-specific checks** (book section 11). Those
specification sections were not read in full in this turn. Copying one generic section again is explicitly forbidden,
and inventing per-document content would exceed the executor contract, so no document was modified.

## 3. Read-only state of the four documents (unchanged this turn)

| document | bytes | RUNTIME_STRUCTURAL_ONLY | NOT_EVALUATED | exact phrase contiguous | generic section |
|---|---:|---|---|---|---|
| `README.md` | 11045 | True | True | False | True |
| `docs/model_card.md` | 7500 | True | True | False | True |
| `docs/runtime_mapping.md` | 2832 | True | True | False | True |
| `inference/README.md` | 1601 | True | True | False | True |

## 4. Diagnosis of the previous R1B failure

```text
Two markers rendered from the fenced block were present in all four files.
The exact contiguous phrase `semantic target correctness is not established` never appeared, because it existed only
inside a prose sentence that did not land as literal, unbroken text.
```

## 5. Next-step plan (not executed here)

```text
1. read book sections 3-7 (global frozen semantics + the four per-file requirements)
2. restore the four documents to their pre-R1B content at 385a3ba to drop the generic section
3. add each document's own required semantics, keeping the exact phrase on one unbroken line
4. run book section 11 global marker checks and the four file-specific checks
5. commit and push only when every check is green
```

## 6. Untouched

```text
product source / tests / manifest / inspect-proposals = UNCHANGED
pytest / model inference / training = NOT run
```

## 7. Disposition

```text
task_status = STOP
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
