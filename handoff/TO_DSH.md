# TO_DSH — Task 8B.3-P1D11S1-R2: Finish Git-Canonical Sync-Helper Recovery

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `c49f342792dcc2a4d60a4a2e23dfa6f0854fdeda`

# 0. ChatGPT audit disposition

P1D11S1-R1 STOP is ACCEPTED.

The helper design remains frozen and correct:

```text
real RC1 manifest basis:
GIT_CANONICAL_BLOB_BYTES

legacy manifest without identity_basis:
working-tree source bytes
```

R1 failed because the required implementation/test cleanup was incomplete.

Confirmed remote defects at `c49f342...`:

```text
D1:
scripts/sync_advisor_rc1_delivery.py calls subprocess.run(...)
but still does not import subprocess.

D2:
tests/test_sync_advisor_rc1_delivery.py uses bare:
GIT_CANONICAL_BASIS
in two tests, but no such test-module symbol is defined.

D3:
required test proving manifest-identity mismatch blocks destination writes is absent.

D4:
required unsupported identity-basis failure test is absent.
```

Observed dedicated pytest:

```text
23 passed
3 failed

failed:
test_every_real_manifest_entry_matches_canonical_file
test_git_canonical_mode_ignores_crlf_worktree
test_git_canonical_check_rejects_crlf_destination
```

No external read-only check was authorized after the failed pytest.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = c49f342792dcc2a4d60a4a2e23dfa6f0854fdeda
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify anything under `delivery_src/BuildReasonSeg_Advisor_RC1/**`;
- modify external RC1;
- run write sync;
- run full project pytest;
- run model/detector/Qwen/SAM2/D-B1 inference;
- run locked candidates;
- modify `.gitattributes`;
- modify Git config;
- normalize line endings;
- redesign helper semantics;
- change PROP/REF/MASK behavior;
- update main;
- force push.

# 3. Allowed tracked changes

Only:

```text
scripts/sync_advisor_rc1_delivery.py
tests/test_sync_advisor_rc1_delivery.py
docs/task8b3_p1d11s1_git_canonical_sync_helper.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 4. Fix helper import — mandatory

In:

```text
scripts/sync_advisor_rc1_delivery.py
```

add exactly the required standard-library import:

```python
import subprocess
```

Do not alter the established source-basis design unless a test demonstrates another concrete defect.

Retain:

```text
identity_basis missing
-> working-tree bytes

identity_basis == GIT_CANONICAL_BLOB_BYTES
-> Git HEAD canonical blob bytes

unsupported non-empty identity_basis
-> explicit failure
```

Retain manifest length/SHA validation before Git-canonical source bytes are used.

# 5. Fix the test-module basis symbol

In:

```text
tests/test_sync_advisor_rc1_delivery.py
```

define one authoritative test symbol after importing the helper:

```python
GIT_CANONICAL_BASIS = sync.GIT_CANONICAL_BASIS
```

Do NOT hardcode a second independent string if this module symbol can be reused.

The following existing tests must then use that symbol successfully:

```text
test_git_canonical_mode_ignores_crlf_worktree
test_git_canonical_check_rejects_crlf_destination
```

# 6. Confirm obsolete real-manifest assertion remains corrected

Keep the rewritten:

```text
test_every_real_manifest_entry_matches_canonical_file
```

with Git-canonical semantics only.

It must require:

```text
payload["identity_basis"] == sync.GIT_CANONICAL_BASIS
len(entries) == 135

for every entry:
    blob = sync.git_canonical_bytes(entry["path"])
    len(blob) == entry["bytes"]
    sha256(blob) == entry["sha256"]
```

It must NOT compare the Windows working-tree file bytes against manifest identity.

# 7. Add mandatory identity-mismatch write-block test

Add exactly one deterministic test with a clear name such as:

```text
test_git_canonical_manifest_identity_mismatch_blocks_write
```

Required setup:

```text
temporary canonical root
temporary destination
manifest:
    identity_basis = GIT_CANONICAL_BLOB_BYTES
    entry identity = GOOD_BYTES
mock sync.git_canonical_bytes(...)
    returns BAD_BYTES
destination target:
    absent
```

Call:

```text
sync.run_sync(destination, entries, canonical)
```

Current helper contract may raise `SystemExit`.

Require:

```text
failure is raised/returned
destination/a.txt DOES NOT EXIST
BAD_BYTES are not written
```

The test must explicitly prove write prevention.

# 8. Add mandatory unsupported-basis test

Add one deterministic test with a clear name such as:

```text
test_unsupported_identity_basis_rejected
```

Manifest:

```json
"identity_basis": "UNKNOWN_BASIS"
```

Require an explicit failure from:

```text
sync.manifest_identity_basis(canonical)
```

or the normal sync/check call before any destination write.

Also require destination remains unmodified.

# 9. Dedicated pytest — exactly one run

Run exactly:

```text
python -m pytest tests/test_sync_advisor_rc1_delivery.py -q
```

Do NOT run any other pytest target.

Require:

```text
exit = 0
failed = 0
errors = 0
all tests passed
```

Record exact pass count.

If §9 fails:
- do NOT run the real external check;
- STOP;
- record exact failed nodes;
- commit/push safe evidence;
- wait for ChatGPT.

# 10. Real external read-only check — only after pytest PASS

Run exactly once:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

No write sync.

Require exactly:

```text
checked = 135
match = 133
missing = 0
mismatch = 2
```

Mismatches exactly:

```text
README.md
docs/model_card.md
```

Require these four old EOL-only mismatches are absent:

```text
buildreasonseg/runtime/core.py
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

If not exact:
- STOP;
- no external write.

# 11. Immutability gate

Require:

```text
canonical RC1 files = UNCHANGED
external RC1 files = UNCHANGED
model/inference = NONE
locked candidate execution = NONE
```

# 12. Outcome

Choose exactly one.

## A — `RC1_SYNC_HELPER_GIT_CANONICAL_ALIGNED`

Require:
- `import subprocess` present;
- test basis constant fixed;
- real-manifest test is Git-canonical only;
- identity-mismatch write-block test PASS;
- unsupported-basis test PASS;
- dedicated pytest all PASS;
- real external read-only check exactly 133/135 with only README/model_card mismatch;
- canonical/external RC1 unchanged.

Then:

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_RETRY
```

## B — `RC1_SYNC_HELPER_ALIGNMENT_BLOCKED`

For any failed gate.

Then:

```text
NEXT = RC1_SYNC_HELPER_ALIGNMENT_RECOVERY
```

Do not execute next gate.

# 13. Report

Append `P1D11S1-R2` to:

```text
docs/task8b3_p1d11s1_git_canonical_sync_helper.md
```

Required:

1. R1 STOP audit findings
2. subprocess import fix
3. test-basis symbol fix
4. Git-canonical real-manifest assertion
5. identity-mismatch write-block test
6. unsupported-basis test
7. exact dedicated pytest result
8. exact external read-only check
9. previous four EOL mismatches status
10. canonical/external RC1 immutability
11. model/inference/locked candidates = NONE
12. exact outcome
13. exact next gate.

Preserve prior STOP history.

# 14. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11S1-R2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: c49f342792dcc2a4d60a4a2e23dfa6f0854fdeda
subprocess import: PRESENT / MISSING
Test GIT_CANONICAL_BASIS symbol: FIXED / NOT FIXED
Real-manifest assertion basis: GIT_CANONICAL_BLOB_BYTES / other
Manifest identity mismatch blocks write test: PASS / FAIL / NOT RUN
Unsupported identity basis test: PASS / FAIL / NOT RUN
Dedicated pytest: <n> passed / FAILED
Real external read-only check: 133/135; mismatch README.md + docs/model_card.md / other / NOT RUN
Previous four EOL mismatches eliminated: YES / NO / NOT CHECKED
Canonical RC1 modified: NO
External RC1 modified: NO
Model/inference execution: NONE
Locked candidates run: NO
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Outcome: <exact enum>
Next gate: <exact enum>
Report: docs/task8b3_p1d11s1_git_canonical_sync_helper.md
Next action: Awaiting ChatGPT audit; do not sync external.
```

# 15. Commit / push

If COMPLETE:

```text
fix(rc1): finish git canonical sync helper
```

If STOP/FAILED:

```text
docs(rc1): record sync-helper recovery stop
```

Push normally.
No force push.

# 16. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- only allowed tracked files changed;
- helper imports subprocess;
- test basis symbol is valid;
- no real-manifest test assumes working-tree identity;
- identity-mismatch source is proven unable to write;
- unsupported basis is proven to fail;
- dedicated test file fully passes;
- real external read-only check is exactly 133/135 with only README/model_card mismatch;
- canonical/external RC1 untouched;
- no model/inference/locked candidate run;
- PROP-01 remains OPEN;
- next gate not executed;
- commit/push succeeds;
- STOP.
