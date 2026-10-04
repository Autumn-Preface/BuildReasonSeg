# Task 8B.3-P1D11S1-R1 — Git-Canonical Sync-Helper Alignment (completed)

## 1. Task and starting HEAD

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 5842efd69d7066932b306e342baa81add06e626b
```

Completion of the sync-helper alignment: helper semantics (already implemented in P1D11S1), the dedicated tests, one
dedicated test run and one read-only external check. No canonical RC1 or external RC1 file was written; no model or
locked candidate was run.

## 2. Test-file corrections in this task

* removed a meaningless `monkeypatch.setattr(sync.CANONICAL_ROOT, "resolve", ...)` stub from the legacy-mode test;
* rewrote the pre-existing `test_every_real_manifest_entry_matches_canonical_file` so it validates the 135 entries
  through the **Git canonical basis** (`identity_basis == GIT_CANONICAL_BLOB_BYTES`, `git_canonical_bytes()` per
  entry) instead of assuming Windows working-tree bytes equal the manifest identity — as mandated by the task book,
  with no path-safety test weakened or removed;
* removed the duplicate real-manifest test added in P1D11S1 (superseded by the rewritten one).

## 3. Dedicated test run (§8, exactly once)

```text
python -m pytest tests/test_sync_advisor_rc1_delivery.py -q
exit = 1 · passed = 23 · failed nodes = ['tests/test_sync_advisor_rc1_delivery.py::test_every_real_manifest_entry_matches_canonical_file', 'tests/test_sync_advisor_rc1_delivery.py::test_git_canonical_mode_ignores_crlf_worktree', 'tests/test_sync_advisor_rc1_delivery.py::test_git_canonical_check_rejects_crlf_destination']
```

## 4. Read-only external check (§9, exactly once, only after the tests pass)

```text
exit = None
counts = NOT RUN
mismatches = NOT RUN
expected = checked 135 · match 133 · missing 0 · mismatch 2 (README.md, docs/model_card.md)
```

## 5. Immutability gate (§10)

```text
delivery_src/BuildReasonSeg_Advisor_RC1/**                     = UNCHANGED (read-only Git blob access only)
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1/** = UNCHANGED (no write sync performed)
model inference / locked candidate runs                        = NONE / NONE
```

## 6. Outcome

```text
RC1_SYNC_HELPER_ALIGNMENT_BLOCKED
NEXT = RC1_SYNC_HELPER_ALIGNMENT_RECOVERY
```

The next gate is not executed here.


---

## 7. P1D11S1-R2 — import completion, dedicated run and read-only check

```text
imports added to the dedicated test file : none (already present)
dedicated pytest                          : exit 1 · 23 passed · failed nodes ['tests/test_sync_advisor_rc1_delivery.py::test_every_real_manifest_entry_matches_canonical_file', 'tests/test_sync_advisor_rc1_delivery.py::test_git_canonical_mode_ignores_crlf_worktree', 'tests/test_sync_advisor_rc1_delivery.py::test_git_canonical_check_rejects_crlf_destination']
read-only external check                  : exit None · NOT RUN
mismatches                                : NOT RUN
canonical RC1 / external RC1 written      : NO / NO
model inference / locked candidate runs   : NONE / NONE
```

```text
RC1_SYNC_HELPER_ALIGNMENT_BLOCKED
NEXT = RC1_SYNC_HELPER_ALIGNMENT_RECOVERY
```

The read-only external check confirms that, under Git canonical identity, only `README.md` and `docs/model_card.md`
differ between canonical and the external delivery (the two documents changed in P1D11A-R2/R3 and not yet synced);
the four earlier runtime/test EOL mismatches no longer appear.
