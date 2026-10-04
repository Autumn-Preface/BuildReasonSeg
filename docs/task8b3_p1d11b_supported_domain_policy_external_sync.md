# Task 8B.3-P1D11B — Supported-Domain Policy External Sync (STOP)

## 1. Task and scope

Controlled canonical → external policy/source sync. The mandated pre-sync external gate expects exactly
`checked 135 · match 133 · missing 0 · mismatch 2 (README.md, docs/model_card.md)`. The measured state is
**129 match / 6 mismatch**, so per the task book **no sync was performed** and the task stops.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = f9bf6be13035445f45f956c2b6a5a64ccf7e6032
helper check invocations = 1 (pre-sync only)   sync invocations = 0
pytest / model inference / locked-candidate runs = NONE
```

## 2. Measured pre-sync external check

```text
exit = 1 · counts = {"checked": 135, "match": 129, "missing": 0, "mismatch": 6}
mismatched paths = README.md, buildreasonseg/runtime/core.py, buildreasonseg/runtime/detector.py, buildreasonseg/runtime/outputs.py, docs/model_card.md, tests/test_task8b_runtime.py
expected          = checked 135 · match 133 · missing 0 · mismatch 2 (README.md, docs/model_card.md)
```

## 3. Unexpected mismatches — byte detail

| path | canonical worktree bytes | external bytes | Git canonical bytes | external == Git canonical | worktree == Git canonical |
|---|---:|---:|---:|---|---|
| `README.md` | 10499 | 9484 | 10499 | False | True |
| `buildreasonseg/runtime/core.py` | 11310 | 11064 | 11064 | True | False |
| `buildreasonseg/runtime/detector.py` | 20772 | 20300 | 20300 | True | False |
| `buildreasonseg/runtime/outputs.py` | 9367 | 9166 | 9166 | True | False |
| `docs/model_card.md` | 6954 | 5997 | 6954 | False | True |
| `tests/test_task8b_runtime.py` | 25552 | 24983 | 24983 | True | False |

The two expected mismatches are the documents edited in P1D11A-R2/R3 (`README.md`, `docs/model_card.md`). The four
**unexpected** mismatch paths are the CRLF-affected runtime/test files
(`buildreasonseg/runtime/core.py`, `buildreasonseg/runtime/detector.py`, `buildreasonseg/runtime/outputs.py`,
`tests/test_task8b_runtime.py`): the external copies carry the **Git canonical (LF)** content while the canonical
working tree carries the CRLF-expanded checkout, so a raw working-tree byte comparison reports a mismatch even though
the canonical source identity is unchanged.

## 4. Actions not taken

```text
controlled 135-file helper sync              = NOT RUN
external source_manifest.json copy           = NOT RUN
external setup checker (pre/post)            = NOT RUN
```

No external delivery file was read-modified: the STOP happened at the pre-sync gate, before any write.

## 5. STOP reason and required decision

The pre-sync gate result differs from the task book's expectation (129/6 versus 133/2). Two semantics are now
conflicting by design:

1. `source_manifest.json` records **Git canonical (LF) blob** identities (`GIT_CANONICAL_BLOB_BYTES`);
2. the sync helper compares **working-tree bytes**, so CRLF-expanded canonical files can never match external copies
   that were themselves written from LF/Git-canonical content (or vice versa).

A new ChatGPT decision is needed to pick one of: (a) make the helper comparison CRLF-insensitive / Git-canonical;
(b) normalize the canonical working tree (`.gitattributes`, `git add --renormalize`) so working-tree bytes equal Git
canonical bytes for all 135 entries; or (c) accept and document a per-file convention split. Until then the controlled
policy/source sync cannot proceed.
