# TO_DSH — Task 8B.3-P1D11S1: Align RC1 Sync Helper with Git-Canonical Source Identity

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `9d6f27a2c92bafd8d7230cd75218d803c3fb6ea1`

# 0. ChatGPT audit disposition

P1D11B STOP is ACCEPTED.

Measured pre-sync state:

```text
checked = 135
match = 129
missing = 0
mismatch = 6
```

Mismatches:

```text
expected policy drift:
README.md
docs/model_card.md

EOL-only helper drift:
buildreasonseg/runtime/core.py
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

For the four EOL-only paths:

```text
external bytes == Git canonical bytes = YES
canonical Windows working-tree bytes == Git canonical bytes = NO
```

Frozen manifest semantics:

```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

Root cause:

```text
source_manifest semantics = Git canonical bytes
sync helper source semantics = working-tree bytes
```

This task repairs the helper semantics only.

# 1. Design decision

Do NOT:
- normalize the repository with `.gitattributes`;
- run `git add --renormalize`;
- alter `core.autocrlf`;
- restore a mixed per-file convention.

Chosen repair:

```text
When source_manifest.identity_basis == GIT_CANONICAL_BLOB_BYTES,
the sync helper MUST compare and copy Git HEAD canonical blob bytes
for each manifest-listed source path.

When identity_basis is absent,
the helper MUST preserve its historical working-tree behavior.
```

This keeps old miniature/unit-test manifests backward compatible while making the real RC1 manifest cross-platform and consistent.

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 9d6f27a2c92bafd8d7230cd75218d803c3fb6ea1
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 3. Strict prohibitions

Do NOT:
- modify canonical RC1 README/model_card/runtime/tests/config/checkpoints;
- modify canonical `source_manifest.json`;
- modify external delivery;
- run write sync;
- run model/detector/Qwen/SAM2/D-B1 inference;
- run locked candidates;
- run full project pytest;
- modify `.gitattributes`;
- modify Git config;
- normalize repository line endings;
- fix PROP/REF/MASK behavior;
- enter Task 8B.4/8C;
- update main;
- force push.

# 4. Allowed tracked changes

Only:

```text
scripts/sync_advisor_rc1_delivery.py
tests/test_sync_advisor_rc1_delivery.py
docs/task8b3_p1d11s1_git_canonical_sync_helper.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 5. Required helper behavior

Modify:

```text
scripts/sync_advisor_rc1_delivery.py
```

The implementation may be structured differently, but externally observable behavior MUST satisfy all requirements below.

## 5.1 Manifest metadata awareness

The helper must read the complete manifest payload sufficiently to determine:

```text
identity_basis
files
```

Supported behavior:

```text
identity_basis == "GIT_CANONICAL_BLOB_BYTES"
    -> source basis = Git HEAD canonical blobs

identity_basis missing
    -> source basis = working-tree files (legacy compatibility)

any other non-empty identity_basis
    -> fail explicitly; do not silently choose a basis
```

Keep existing path-safety and duplicate-path validation.

## 5.2 Git canonical byte retrieval

For Git-canonical basis, each source relative path must be read as exact binary bytes equivalent to:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<relative_path>
```

Requirements:
- binary stdout capture;
- no text decoding/newline conversion;
- non-zero Git exit -> explicit failure;
- path comes only from already validated safe manifest entry;
- no shell string interpolation;
- use subprocess argument list, not `shell=True`.

The helper may define a dedicated function such as:

```python
git_canonical_bytes(relative: str) -> bytes
```

but naming is not mandated.

## 5.3 Check mode

When basis is Git canonical:

```text
--check
```

must compare:

```text
Git HEAD canonical source bytes
vs
destination bytes
```

NOT:

```text
Windows working-tree source bytes
vs
destination bytes
```

Output summary contract remains:

```text
checked=<n> match=<n> missing=<n> mismatch=<n>
```

## 5.4 Sync mode

When basis is Git canonical, normal sync must write exact Git blob bytes to destination.

Post-write verification must compare destination against those same Git canonical bytes.

Do NOT copy the CRLF-expanded working-tree source in Git-canonical mode.

## 5.5 Legacy behavior

For manifests without `identity_basis`, preserve existing behavior:

```text
source = canonical working-tree file
```

Existing temporary fixture tests should remain valid.

# 6. Manifest identity validation

For `GIT_CANONICAL_BLOB_BYTES` mode, before check/sync the helper must validate each Git source blob against its manifest entry:

```text
len(source_bytes) == entry["bytes"]
sha256(source_bytes) == entry["sha256"]
```

If a manifest identity mismatch occurs:
- report/fail before writing that file;
- sync result must be non-zero;
- do not silently copy unverified bytes.

This makes the helper consistent with the now-authoritative manifest semantics.

For legacy manifests without identity basis, do not impose a new behavior beyond existing contract unless required by the existing tests.

# 7. Tests

Modify only:

```text
tests/test_sync_advisor_rc1_delivery.py
```

Add deterministic temporary-directory / monkeypatch tests covering at least:

## A. Legacy working-tree behavior preserved

A manifest with no `identity_basis`:
- normal sync copies working-tree bytes;
- check passes afterwards.

Existing tests may satisfy this if still explicit.

## B. Git-canonical mode ignores CRLF-expanded working tree

Construct a temporary canonical fixture where:

```text
working-tree bytes = b"A\r\nB\r\n"
Git canonical bytes = b"A\nB\n"
manifest identity = LF Git bytes
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

Monkeypatch the Git-byte retrieval boundary or use an isolated temporary Git repo.

Require:
- sync writes LF Git bytes, not CRLF working-tree bytes;
- subsequent check passes;
- destination exact bytes == Git canonical bytes.

## C. Git-canonical check rejects CRLF destination

With same fixture:
- destination contains CRLF worktree bytes;
- `--check` returns non-zero.

## D. Manifest identity mismatch blocks Git-canonical sync

Provide Git canonical bytes that do not match manifest bytes/hash.

Require:
- non-zero/failure;
- destination target is not written with unverified source bytes.

## E. Real manifest policy

Update the real-manifest test so it validates the 135 entries according to:

```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

rather than directly assuming Windows working-tree bytes must equal manifest identity.

Require real manifest:
- schema v1;
- identity basis exact;
- 135 entries;
- all 135 Git canonical bytes match recorded bytes/SHA256.

Do NOT weaken/remove existing path-safety tests.

# 8. Dedicated test run

Run only:

```text
python -m pytest tests/test_sync_advisor_rc1_delivery.py -q
```

Use the normal repo test-capable environment already used for repository tests.

No full suite.

Require:

```text
all tests PASS
```

Record exact passed count.

If any test fails:
- do not modify external;
- record STOP;
- commit/push current evidence only if Git state is safe;
- STOP.

# 9. Real external read-only regression check

After dedicated tests PASS, run exactly one read-only check:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

No write sync.

Require exact result:

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

The previous four runtime/test EOL mismatches must disappear.

If not exact:
- do not write external;
- outcome = STOP.

# 10. Canonical and external immutability gate

Require this task does NOT change:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/**
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1/**
```

The only canonical-tree access is read-only through Git blobs/manifest.

No source manifest update is needed.

# 11. Outcome

Choose exactly one.

## A — `RC1_SYNC_HELPER_GIT_CANONICAL_ALIGNED`

Require:
- helper behavior implemented;
- dedicated tests all pass;
- real external read-only check = 133/135 with only README/model_card mismatch;
- canonical/external delivery unchanged.

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

# 12. Report

Create:

```text
docs/task8b3_p1d11s1_git_canonical_sync_helper.md
```

Required:

1. task / starting HEAD
2. accepted P1D11B STOP evidence
3. root cause
4. chosen design
5. helper behavior changes
6. backward-compatibility behavior
7. manifest identity validation behavior
8. test cases added/updated
9. dedicated pytest result
10. real external read-only check
11. canonical delivery unchanged
12. external delivery unchanged
13. model/inference/locked candidates = NONE
14. exact outcome
15. exact next gate.

# 13. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11S1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 9d6f27a2c92bafd8d7230cd75218d803c3fb6ea1
Sync helper modified: YES / NO
Sync helper source basis for real RC1: GIT_CANONICAL_BLOB_BYTES / other
Legacy no-identity-basis behavior preserved: YES / NO
Git-canonical manifest identity validation: ENABLED / NOT ENABLED
Dedicated test file: tests/test_sync_advisor_rc1_delivery.py
Dedicated pytest: <n> passed / FAILED
Real external check: 133/135; mismatch README.md + docs/model_card.md / other
Previous four EOL mismatches eliminated: YES / NO
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

# 14. Commit / push

If COMPLETE:

```text
fix(rc1): align sync helper with git canonical identity
```

If STOP/FAILED:

```text
docs(rc1): record sync-helper alignment stop
```

Push normally.
No force push.

# 15. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- only allowed tracked files changed;
- no canonical/external delivery changes;
- real manifest basis drives Git canonical source bytes;
- manifests without identity basis preserve working-tree behavior;
- manifest identity is verified before Git-canonical copy/check;
- dedicated sync-helper tests pass;
- real external read-only check becomes exactly 133/135 with only README/model_card mismatch;
- no model/inference/locked candidate run;
- PROP-01 remains OPEN;
- next gate not executed;
- commit/push succeeds;
- STOP.
