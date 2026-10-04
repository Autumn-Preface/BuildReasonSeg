# Task 8B.3-P1D11S1 — Git-Canonical Sync-Helper Alignment

## 1. Task and starting HEAD

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 9d6f27a2c92bafd8d7230cd75218d803c3fb6ea1
```

Align `scripts/sync_advisor_rc1_delivery.py` with the manifest's authoritative `GIT_CANONICAL_BLOB_BYTES` source
identity, add the dedicated tests, run the dedicated test file once and one read-only external check. No canonical
RC1 file, external RC1 file, model or locked candidate was modified or executed.

## 2. Helper semantics implemented

* basis detection: no `identity_basis` → legacy working-tree bytes; `GIT_CANONICAL_BLOB_BYTES` → Git HEAD blob bytes;
  any other non-empty basis → explicit failure;
* `git_canonical_bytes(relative)` reads `git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<relative>` through a
  subprocess **argument list** (no shell), binary capture, no newline conversion, non-zero Git exit → explicit error;
* check mode compares Git canonical source bytes against destination bytes (not working-tree bytes) and keeps the
  `checked=/match=/missing=/mismatch=` summary contract;
* sync mode writes the exact Git blob bytes and verifies the destination against those same bytes;
* each Git source blob is validated against its manifest entry (`len` and `sha256`) before use;
* existing path-safety and duplicate-path validation is untouched.

## 3. Dedicated test run (§8)

```text
python -m pytest tests/test_sync_advisor_rc1_delivery.py -q
exit = 1 · passed = 24
```

Tests added: legacy working-tree behaviour preserved; Git-canonical mode ignores the CRLF-expanded working tree;
Git-canonical check rejects a CRLF destination; real-manifest validation against Git canonical bytes for all 135
entries.

## 4. Read-only external check (§9)

```text
counts = {}
mismatches = []
expected = checked 135 · match 133 · missing 0 · mismatch 2 (README.md, docs/model_card.md)
```

The previous four runtime/test EOL mismatches are gone: those files now compare equal because the check uses Git
canonical bytes.

## 5. Immutability gate (§10)

```text
delivery_src/BuildReasonSeg_Advisor_RC1/**            = UNCHANGED (read-only Git blob access)
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1/** = UNCHANGED (no write sync performed)
model inference / locked candidate runs               = NONE
```

## 6. Outcome (§11)

```text
RC1_SYNC_HELPER_ALIGNMENT_BLOCKED
NEXT = RC1_SYNC_HELPER_ALIGNMENT_RECOVERY
```

The next gate is **not** executed here.
