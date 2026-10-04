# TO_DSH — Task 8B.3-P1D11S1-R3: Apply the Missing Sync-Helper Patch Exactly

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `3d491534a9dacb29e1ef51cc126e69013535c7e9`

# 0. Audit finding

P1D11S1-R2 STOP is accepted.

However, ChatGPT independently verified the remote files at `3d491534...` and found that the required code changes were NOT actually present:

```text
scripts/sync_advisor_rc1_delivery.py
  still uses subprocess.run(...)
  but still has NO "import subprocess"

tests/test_sync_advisor_rc1_delivery.py
  still has NO:
  GIT_CANONICAL_BASIS = sync.GIT_CANONICAL_BASIS

tests/test_sync_advisor_rc1_delivery.py
  still has NO:
  test_git_canonical_manifest_identity_mismatch_blocks_write

tests/test_sync_advisor_rc1_delivery.py
  still has NO:
  test_unsupported_identity_basis_rejected
```

This R3 contains no design decision.
Apply the exact missing patch.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 3d491534a9dacb29e1ef51cc126e69013535c7e9
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify `delivery_src/BuildReasonSeg_Advisor_RC1/**`;
- modify external RC1;
- run write sync;
- run full pytest;
- run model/inference;
- run locked candidates;
- modify `.gitattributes`;
- modify Git config;
- redesign helper behavior;
- change any existing test assertion except where explicitly authorized below;
- delete or rename any existing test;
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

# 4. Exact edit A — helper import

File:

```text
scripts/sync_advisor_rc1_delivery.py
```

Current import block contains:

```python
import argparse
import hashlib
import json
import sys
from pathlib import Path
```

Change it to:

```python
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
```

No other helper logic change is authorized in this task unless required by a failing dedicated test after this exact patch.

# 5. Exact edit B — test basis constant

File:

```text
tests/test_sync_advisor_rc1_delivery.py
```

Immediately after:

```python
import sync_advisor_rc1_delivery as sync  # noqa: E402
```

insert exactly:

```python
GIT_CANONICAL_BASIS = sync.GIT_CANONICAL_BASIS
```

Do not define a second hardcoded basis string.

# 6. Exact edit C — identity mismatch blocks write test

Append this test after the existing Git-canonical temporary tests.

Use this exact logical contract:

```python
def test_git_canonical_manifest_identity_mismatch_blocks_write(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    destination = tmp_path / "destination"
    canonical.mkdir()
    destination.mkdir()

    good_bytes = b"good\n"
    bad_bytes = b"bad\n"
    _write_manifest(canonical, [_entry(good_bytes)], basis=GIT_CANONICAL_BASIS)

    monkeypatch.setattr(sync, "git_canonical_bytes", lambda relative: bad_bytes)
    entries = sync.load_manifest(canonical)

    with pytest.raises(SystemExit):
        sync.run_sync(destination, entries, canonical)

    assert not (destination / "a.txt").exists()
```

Minor formatting/type-annotation differences are allowed.
The behavior is NOT negotiable.

# 7. Exact edit D — unsupported basis rejected test

Append this test after edit C.

Use this exact logical contract:

```python
def test_unsupported_identity_basis_rejected(tmp_path):
    canonical = tmp_path / "canonical"
    destination = tmp_path / "destination"
    canonical.mkdir()
    destination.mkdir()

    (canonical / "a.txt").write_bytes(b"value\n")
    _write_manifest(canonical, [_entry(b"value\n")], basis="UNKNOWN_BASIS")

    with pytest.raises(SystemExit):
        sync.manifest_identity_basis(canonical)

    assert list(destination.iterdir()) == []
```

Minor formatting/type-annotation differences are allowed.
The behavior is NOT negotiable.

# 8. Existing real-manifest test

Do NOT rewrite it again.

Keep the current Git-canonical version of:

```text
test_every_real_manifest_entry_matches_canonical_file
```

It already has the correct semantics.

# 9. Mandatory static patch verification BEFORE pytest

Before running pytest, run a read-only Python check that verifies the edited files actually contain the patch.

Equivalent required assertions:

```python
from pathlib import Path

helper = Path("scripts/sync_advisor_rc1_delivery.py").read_text(encoding="utf-8")
tests = Path("tests/test_sync_advisor_rc1_delivery.py").read_text(encoding="utf-8")

assert "import subprocess" in helper
assert "subprocess.run(" in helper

assert "GIT_CANONICAL_BASIS = sync.GIT_CANONICAL_BASIS" in tests
assert "def test_git_canonical_manifest_identity_mismatch_blocks_write" in tests
assert "def test_unsupported_identity_basis_rejected" in tests
```

Record:

```text
STATIC_PATCH_GATE = PASS
```

If any assertion fails:
- do NOT run pytest;
- STOP and report the missing assertion.

# 10. Mandatory Git diff gate BEFORE pytest

Run:

```text
git diff -- scripts/sync_advisor_rc1_delivery.py tests/test_sync_advisor_rc1_delivery.py
```

Require the diff visibly contains:
- one added `import subprocess`;
- one added `GIT_CANONICAL_BASIS = sync.GIT_CANONICAL_BASIS`;
- the two new test functions.

Record:

```text
PATCH_DIFF_GATE = PASS
```

If not:
- do NOT run pytest;
- STOP.

# 11. Dedicated pytest — exactly one run

Only after §9 and §10 PASS, run exactly:

```text
python -m pytest tests/test_sync_advisor_rc1_delivery.py -q
```

Do NOT run any other pytest target.

Because no existing test may be deleted and exactly two tests are added, expected result is:

```text
28 passed
exit code = 0
```

If pass count is not 28 or exit is non-zero:
- do NOT run external check;
- STOP;
- record exact result and failed nodes.

# 12. Real external read-only check

Only after:

```text
28 passed
```

run exactly once:

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

Only:

```text
README.md
docs/model_card.md
```

may mismatch.

These four must NOT mismatch:

```text
buildreasonseg/runtime/core.py
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

If exact result differs:
- STOP;
- no external write.

# 13. Final tracked-diff gate

Before commit require tracked changes only:

```text
scripts/sync_advisor_rc1_delivery.py
tests/test_sync_advisor_rc1_delivery.py
docs/task8b3_p1d11s1_git_canonical_sync_helper.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Explicitly record:

```text
canonical RC1 modified = NO
external RC1 modified = NO
model/inference run = NO
locked candidates run = NO
```

# 14. Outcome

If all gates pass:

```text
Outcome = RC1_SYNC_HELPER_GIT_CANONICAL_ALIGNED
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_RETRY
```

Otherwise:

```text
Outcome = RC1_SYNC_HELPER_ALIGNMENT_BLOCKED
NEXT = RC1_SYNC_HELPER_ALIGNMENT_RECOVERY
```

Do not execute NEXT.

# 15. Report

Append `P1D11S1-R3` to:

```text
docs/task8b3_p1d11s1_git_canonical_sync_helper.md
```

Required fields:

```text
Starting HEAD
STATIC_PATCH_GATE
PATCH_DIFF_GATE
subprocess import
test basis symbol
identity-mismatch write-block test
unsupported-basis test
dedicated pytest exact result
external read-only exact result
previous four EOL mismatches eliminated
canonical RC1 modified
external RC1 modified
model/inference
locked candidates
Outcome
NEXT
```

Preserve prior STOP history.

# 16. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11S1-R3
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 3d491534a9dacb29e1ef51cc126e69013535c7e9
STATIC_PATCH_GATE: PASS / FAIL
PATCH_DIFF_GATE: PASS / FAIL
subprocess import: PRESENT / MISSING
Test GIT_CANONICAL_BASIS symbol: PRESENT / MISSING
Manifest identity mismatch blocks write test: PASS / FAIL / NOT RUN
Unsupported identity basis test: PASS / FAIL / NOT RUN
Dedicated pytest: 28 passed / other
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

# 17. Commit / push

If COMPLETE:

```text
fix(rc1): apply missing git canonical sync patch
```

If STOP/FAILED:

```text
docs(rc1): record missing sync patch stop
```

Push normally.
No force push.

# 18. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- exact four edits A/B/C/D are actually present in Git diff;
- STATIC_PATCH_GATE = PASS;
- PATCH_DIFF_GATE = PASS;
- dedicated pytest = exactly 28 passed;
- external read-only check = exactly 133/135 with only README/model_card mismatch;
- canonical/external RC1 untouched;
- no model/inference/locked candidate run;
- only authorized tracked files committed;
- commit/push succeeds;
- STOP.
