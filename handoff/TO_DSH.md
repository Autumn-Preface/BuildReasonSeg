# TO_DSH — Task 8B.3-P1D11S1-R1: Complete Git-Canonical Sync-Helper Alignment

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `5842efd69d7066932b306e342baa81add06e626b`

# 0. ChatGPT audit disposition

P1D11S1 STOP is ACCEPTED.

The design direction is retained:

```text
real RC1 manifest identity basis = GIT_CANONICAL_BLOB_BYTES
legacy manifest without identity_basis = working-tree source bytes
```

Three concrete defects must be repaired.

## Defect 1 — missing subprocess import

Current helper defines:

```python
subprocess.run(...)
```

but does not import `subprocess`.

This blocks every real Git-canonical read.

## Defect 2 — obsolete real-manifest test remains

The old test:

```text
test_every_real_manifest_entry_matches_canonical_file
```

still compares the Windows working-tree file bytes directly against the manifest.

That assertion is incompatible with:

```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

and must be replaced/re-written to validate Git canonical bytes.

## Defect 3 — required identity-mismatch blocking test is missing

P1D11S1 required an explicit test proving that a Git-canonical blob whose length/hash disagrees with the manifest:
- causes a non-zero/failure;
- is not written to destination.

Add it.

A fourth cleanup is authorized:
- remove the unnecessary attempt to monkeypatch `Path.resolve` on `sync.CANONICAL_ROOT` in the legacy fixture test.
  The explicit `canonical_root` argument already provides isolation.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 5842efd69d7066932b306e342baa81add06e626b
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify any file under `delivery_src/BuildReasonSeg_Advisor_RC1/**`;
- modify external RC1;
- run write sync;
- run full project pytest;
- run model/detector/Qwen/SAM2/D-B1 inference;
- run locked candidates;
- modify `.gitattributes`;
- modify Git config;
- normalize line endings;
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

# 4. Repair helper import

In:

```text
scripts/sync_advisor_rc1_delivery.py
```

add the required standard-library import:

```python
import subprocess
```

Do not otherwise redesign the already implemented source-basis logic unless a dedicated test proves another defect.

Retain all existing requirements:

```text
identity_basis missing
-> legacy working-tree bytes

identity_basis == GIT_CANONICAL_BLOB_BYTES
-> Git HEAD canonical bytes

unsupported non-empty identity_basis
-> explicit failure
```

Retain:
- argument-list subprocess invocation;
- `shell=False` behavior;
- binary stdout;
- manifest length/SHA validation;
- Git-canonical check;
- Git-canonical write/verification.

# 5. Repair dedicated tests

Modify only:

```text
tests/test_sync_advisor_rc1_delivery.py
```

## 5.1 Legacy test cleanup

In:

```text
test_legacy_manifest_without_basis_copies_worktree_bytes
```

remove the unnecessary instance-level monkeypatch of:

```text
sync.CANONICAL_ROOT.resolve
```

The test must use the explicitly supplied `canonical` root to `load_manifest`, `run_sync`, and `run_check`.

Require:
- destination gets exact working-tree bytes;
- subsequent check passes.

## 5.2 Replace obsolete real-manifest working-tree assertion

The test currently named:

```text
test_every_real_manifest_entry_matches_canonical_file
```

must no longer assert that:

```text
REAL_CANONICAL / path
```

working-tree size/hash equals the manifest.

Either delete it and rely on the new Git-canonical real-manifest test, or rewrite it.

Authoritative real-manifest test must establish:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
for every entry:
    git_canonical_bytes(path) length == manifest bytes
    sha256(git_canonical_bytes(path)) == manifest sha256
```

Do not normalize or reinterpret working-tree bytes.

## 5.3 Add required manifest-identity-mismatch blocking test

Add a deterministic temporary test.

Setup:

```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
manifest identity describes GOOD_BYTES
mocked git_canonical_bytes returns BAD_BYTES
destination target initially absent
```

Call the Git-canonical sync path.

Require:
- explicit failure (`SystemExit` is acceptable under current helper contract);
- destination target remains absent;
- BAD_BYTES are not written.

If using a pre-existing destination target instead, require it remains byte-identical to its pre-test content.

Test name should clearly describe the contract, e.g.:

```text
test_git_canonical_manifest_identity_mismatch_blocks_write
```

## 5.4 Unsupported basis test

Add one small deterministic test:

```text
identity_basis = "UNKNOWN_BASIS"
```

Require the helper to fail explicitly before copying/checking.

This freezes the already implemented behavior.

# 6. Dedicated pytest gate

Run exactly:

```text
python -m pytest tests/test_sync_advisor_rc1_delivery.py -q
```

Do not run any other pytest target.

Require:

```text
exit code = 0
all tests passed
failed = 0
error = 0
```

Record exact passed count.

If this fails:
- do NOT run the real external check;
- record STOP;
- commit/push safe evidence;
- STOP.

# 7. Real external read-only check

Only after §6 PASS, run exactly one:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

This is read-only.

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

Require previous four EOL-only mismatches absent:

```text
buildreasonseg/runtime/core.py
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

If result differs:
- no external write;
- STOP.

# 8. Immutability gate

Require:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/** = UNCHANGED
external BuildReasonSeg_Advisor_RC1/** = UNCHANGED
```

No model/inference/candidate execution.

# 9. Outcome

Choose exactly one.

## A — `RC1_SYNC_HELPER_GIT_CANONICAL_ALIGNED`

Require:
- missing import repaired;
- obsolete worktree-based real-manifest assertion removed/re-written;
- identity-mismatch write-block test present and passing;
- unsupported-basis test present and passing;
- dedicated pytest all pass;
- real external read-only check = 133/135 with only README/model_card mismatch;
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

# 10. Report

Update:

```text
docs/task8b3_p1d11s1_git_canonical_sync_helper.md
```

Append an R1 section with:

1. ChatGPT audit findings
2. missing `subprocess` import correction
3. legacy-test cleanup
4. obsolete working-tree real-manifest test correction
5. manifest-identity mismatch blocking test
6. unsupported-basis test
7. exact dedicated pytest result
8. exact real external read-only check
9. canonical RC1 unchanged
10. external RC1 unchanged
11. model/inference/locked candidates = NONE
12. exact outcome
13. exact next gate.

Do not erase the original STOP history.

# 11. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11S1-R1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 5842efd69d7066932b306e342baa81add06e626b
Missing subprocess import: FIXED / NOT FIXED
Legacy no-basis behavior preserved: YES / NO
Obsolete worktree real-manifest assertion: REMOVED_OR_REWRITTEN / PRESENT
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

# 12. Commit / push

If COMPLETE:

```text
fix(rc1): complete git canonical sync helper
```

If STOP/FAILED:

```text
docs(rc1): record sync-helper recovery stop
```

Push normally.
No force push.

# 13. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- only allowed tracked files changed;
- `subprocess` import present;
- legacy no-basis behavior passes;
- no real-manifest test assumes working-tree bytes equal Git identity;
- manifest identity mismatch is proven to block writing;
- unsupported basis is proven to fail explicitly;
- dedicated sync-helper pytest fully passes;
- real external check is exactly 133/135 with only README/model_card mismatch;
- canonical/external RC1 unchanged;
- no model/inference/locked candidate run;
- PROP-01 remains OPEN;
- next gate not executed;
- commit/push succeeds;
- STOP.
