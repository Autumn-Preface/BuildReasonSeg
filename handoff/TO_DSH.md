# TO_DSH — Task 8B.3-REF01-E3A-R2: Boundary-Based Mechanical Canonical Implementation

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-impl`
> Required starting HEAD: `fc2a33325eacf6ce5f366d6574abed3694431f5b`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Temporary patcher path: `C:\D\DeepSeekHarness\task8b3_ref01_e3a_r2_apply.py`

# 0. R1 AUDIT DISPOSITION

E3A-R1 STOP is accepted.

R1 correctly:
- verified the published patcher SHA256;
- executed the patcher exactly once;
- stopped on the first assertion failure;
- did not manually edit detector.py or tests;
- did not update source_manifest;
- did not sync external RC1;
- did not run detector/model inference;
- committed/pushed a STOP result.

R1 failed only because the patcher required an entire source block to be byte-for-byte identical.

R2 replaces that fragile condition with a fixed unique-boundary replacement:
- start boundary = `def eligible_proposals(`
- end boundary = `def proposal_by_id(`
- exact replacement contents remain fully prescribed by ChatGPT.

DSH does not choose any patch location or implementation.

# 1. EXECUTOR CONTRACT

DSH has ZERO technical discretion.

Allowed sequence only:
1. verify branch/head;
2. copy §5 patcher byte-for-byte to the exact temporary path;
3. verify patcher SHA256;
4. execute patcher exactly once;
5. delete temporary patcher after successful execution;
6. run fixed static/pytest gates;
7. write fixed evidence/report/FROM_DSH;
8. choose status and commit message using §14 only;
9. push current branch;
10. STOP.

Forbidden:
- manual edits to detector.py;
- manual edits to test_task8b_runtime.py;
- modifying the patcher;
- rerunning patcher after failure;
- modifying tests after pytest failure;
- changing algorithm/threshold/schema;
- updating source_manifest;
- syncing external RC1;
- detector/model inference;
- executing NEXT.

Any uncovered condition => STOP.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-impl

git rev-parse HEAD
= fc2a33325eacf6ce5f366d6574abed3694431f5b
```

Allowed initial repo status:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No checkout/reset/rebase/merge/stash/clean/cherry-pick.

# 3. ALLOWED TRACKED CHANGES

Only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
evaluation/task8b3_ref01_eligibility_repair_impl.json
docs/task8b3_ref01_eligibility_repair_impl.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Must remain unchanged:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

# 4. ABSOLUTE PROHIBITIONS

Do NOT:
- write external RC1;
- run sync helper;
- update source_manifest;
- instantiate DetectorRuntime;
- run detector/Qwen/SAM2/D-B1/target inference;
- change `MERGE_BBOX_EXTENT_RATIO_MAX`;
- change `SMALLEST_MIN_AREA_PX`;
- execute NEXT;
- update main;
- force push.

Detector/model calls = 0.

# 5. EXACT TEMPORARY PATCHER

Create exactly:

```text
C:\D\DeepSeekHarness\task8b3_ref01_e3a_r2_apply.py
```

with this content byte-for-byte:

```python
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
DETECTOR = REPO / "delivery_src" / "BuildReasonSeg_Advisor_RC1" / "buildreasonseg" / "runtime" / "detector.py"
TEST = REPO / "delivery_src" / "BuildReasonSeg_Advisor_RC1" / "tests" / "test_task8b_runtime.py"
EXTERNAL_DETECTOR = EXTERNAL / "buildreasonseg" / "runtime" / "detector.py"

EXPECTED_BRANCH = "fix/task8b3-ref01-eligibility-repair-impl"
EXPECTED_HEAD = "fc2a33325eacf6ce5f366d6574abed3694431f5b"
EXPECTED_EXTERNAL_SHA = "82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738"

NEW_BLOCK = r"""def eligible_proposals(proposals: list[GlobalProposal], *, family: str = "largest"
                       ) -> list[GlobalProposal]:
    return [proposal for proposal in proposals if eligible(proposal, family=family)]


def _reference_rank(proposal: GlobalProposal) -> tuple[int, float, int]:
    return (-proposal.mask_area, -proposal.confidence, proposal.proposal_id)


def _largest_reference_candidates_with_extent_exception(
        proposals: list[GlobalProposal]) -> list[GlobalProposal]:
    """Frozen base candidates plus the RC1 largest-only extent-dominance exception."""

    base_candidates = eligible_proposals(proposals, family="largest")
    if not base_candidates:
        return []

    baseline = sorted(base_candidates, key=_reference_rank)[0]
    exceptions = [
        proposal for proposal in proposals
        if proposal.mask_area > 0
        and proposal.mask_crop.any()
        and not proposal.touches_image_border
        and proposal.bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX
        and proposal.mask_area > baseline.mask_area
        and proposal.confidence > baseline.confidence
    ]
    return base_candidates + exceptions


def select_reference(proposals: list[GlobalProposal], *, family: str = "largest") -> GlobalProposal | None:
    """Select a reference with the frozen base rule and the largest-only RC1 exception."""

    if family == "largest":
        candidates = _largest_reference_candidates_with_extent_exception(proposals)
    else:
        candidates = eligible_proposals(proposals, family=family)
    if not candidates:
        return None
    return sorted(candidates, key=_reference_rank)[0]


"""

TEST_BLOCK = r"""def _reference_rect(height: int, width: int, confidence: float, proposal_id: int,
                    *, top: int = 100, left: int = 100) -> GlobalProposal:
    mask = np.zeros((512, 512), dtype=bool)
    mask[top:top + height, left:left + width] = True
    return _proposal(mask, confidence, f"r{proposal_id}", proposal_id, proposal_id,
                     proposal_id=proposal_id)


def test_largest_extent_dominance_exception_selects_strictly_dominant_candidate() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    dominant = _reference_rect(120, 120, 0.61, 2, top=250, left=250)

    assert eligible(baseline, family="largest") is True
    assert eligible(dominant, family="largest") is False
    assert detector.eligible_proposals([baseline, dominant], family="largest") == [baseline]
    assert select_reference([baseline, dominant], family="largest") is dominant


def test_largest_extent_exception_rejects_lower_confidence_candidate() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    candidate = _reference_rect(120, 120, 0.59, 2, top=250, left=250)
    assert select_reference([baseline, candidate], family="largest") is baseline


def test_largest_extent_exception_requires_strict_confidence_gain() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    candidate = _reference_rect(120, 120, 0.60, 2, top=250, left=250)
    assert select_reference([baseline, candidate], family="largest") is baseline


def test_largest_extent_exception_requires_strict_area_gain() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    same_area = _reference_rect(40, 160, 0.95, 2, top=250, left=250)

    assert baseline.mask_area == same_area.mask_area
    assert eligible(same_area, family="largest") is False
    assert select_reference([baseline, same_area], family="largest") is baseline


def test_largest_extent_exception_never_admits_border_proposal() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    border = _reference_rect(120, 120, 0.95, 2, top=0, left=250)

    assert border.touches_image_border is True
    assert select_reference([baseline, border], family="largest") is baseline


def test_largest_extent_exception_requires_frozen_baseline() -> None:
    only_extent_violation = _reference_rect(120, 120, 0.95, 1)

    assert eligible(only_extent_violation, family="largest") is False
    assert select_reference([only_extent_violation], family="largest") is None


def test_largest_extent_exception_keeps_production_area_order() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    exception_a = _reference_rect(120, 120, 0.70, 2, top=250, left=50)
    exception_b = _reference_rect(130, 130, 0.61, 3, top=250, left=250)

    assert select_reference([baseline, exception_a, exception_b], family="largest") is exception_b


def test_largest_extent_exception_does_not_change_smallest_family() -> None:
    baseline = _reference_rect(80, 80, 0.60, 1)
    extent_violation = _reference_rect(120, 120, 0.95, 2, top=250, left=250)

    assert eligible(extent_violation, family="smallest") is False
    assert select_reference([baseline, extent_violation], family="smallest") is baseline


"""

START_SENTINEL = b'def eligible_proposals('
END_SENTINEL = b'def proposal_by_id('
TEST_MARKER = b'# ---------------------------------------------------------------- reasoning context'
TEST_SENTINEL = b'def test_largest_extent_dominance_exception_selects_strictly_dominant_candidate'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_text(*args: str) -> str:
    p = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return p.stdout.strip()


def newline(raw: bytes) -> bytes:
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n")
    if crlf and crlf == lf:
        return b"\r\n"
    if not crlf and lf:
        return b"\n"
    raise AssertionError(f"unsupported mixed newline state: CRLF={crlf}, LF={lf}")


def adapt(text: str, nl: bytes) -> bytes:
    return text.encode("utf-8").replace(b"\n", nl)


def main() -> int:
    assert git_text("branch", "--show-current") == EXPECTED_BRANCH
    assert git_text("rev-parse", "HEAD") == EXPECTED_HEAD
    assert sha256(EXTERNAL_DETECTOR) == EXPECTED_EXTERNAL_SHA

    detector_raw = DETECTOR.read_bytes()
    test_raw = TEST.read_bytes()
    dnl = newline(detector_raw)
    tnl = newline(test_raw)

    assert detector_raw.count(START_SENTINEL) == 1
    assert detector_raw.count(END_SENTINEL) == 1
    assert detector_raw.count(b'def select_reference(') == 1
    assert detector_raw.count(b'def _reference_rank(') == 0
    assert detector_raw.count(b'def _largest_reference_candidates_with_extent_exception(') == 0

    start = detector_raw.index(START_SENTINEL)
    end = detector_raw.index(END_SENTINEL)
    assert start < end
    old_slice = detector_raw[start:end]
    assert b'candidates = eligible_proposals(proposals, family=family)' in old_slice
    assert b'proposal.bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX' not in old_slice

    new_slice = adapt(NEW_BLOCK, dnl)
    detector_after = detector_raw[:start] + new_slice + detector_raw[end:]

    assert detector_after.count(b'def _reference_rank(') == 1
    assert detector_after.count(b'def _largest_reference_candidates_with_extent_exception(') == 1
    assert detector_after.count(b'proposal.mask_area > baseline.mask_area') == 1
    assert detector_after.count(b'proposal.confidence > baseline.confidence') == 1
    assert detector_after.count(b'proposal.bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX') >= 2

    assert test_raw.count(TEST_MARKER) == 1
    assert test_raw.count(TEST_SENTINEL) == 0
    marker_pos = test_raw.index(TEST_MARKER)
    test_after = test_raw[:marker_pos] + adapt(TEST_BLOCK, tnl) + test_raw[marker_pos:]
    assert test_after.count(b'def test_largest_extent_') == 8

    DETECTOR.write_bytes(detector_after)
    TEST.write_bytes(test_after)

    print("PATCHER_BRANCH_HEAD: PASS")
    print("EXTERNAL_DETECTOR_IDENTITY: PASS")
    print("DETECTOR_BOUNDARY_REPLACEMENT: PASS")
    print("TEST_MARKER_INSERTION: PASS")
    print("NEW_TEST_COUNT: 8")
    print("PATCHER_RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

```

Required SHA256:

```text
2fa5f9991dc6e4782d668229430544572e9664f266852d40881a9c3a97a323d5
```

After writing the patcher:
- calculate SHA256;
- require exact equality;
- mismatch => STOP, do not run.

# 6. EXECUTE PATCHER EXACTLY ONCE

Run exactly:

```text
<REQUIRED_PYTHON> C:\D\DeepSeekHarness\task8b3_ref01_e3a_r2_apply.py
```

Require exit 0 and these stdout lines:

```text
PATCHER_BRANCH_HEAD: PASS
EXTERNAL_DETECTOR_IDENTITY: PASS
DETECTOR_BOUNDARY_REPLACEMENT: PASS
TEST_MARKER_INSERTION: PASS
NEW_TEST_COUNT: 8
PATCHER_RESULT: PASS
```

Any failure:
- STOP immediately;
- no manual edits;
- no rerun.

On success delete:

```text
C:\D\DeepSeekHarness\task8b3_ref01_e3a_r2_apply.py
```

# 7. STATIC GATE

Run exactly:

```text
<REQUIRED_PYTHON> -m py_compile delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
<REQUIRED_PYTHON> -m py_compile delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py
```

Require both exit 0.

Then require mechanically:

```text
def _reference_rank( count = 1
def _largest_reference_candidates_with_extent_exception( count = 1
proposal.mask_area > baseline.mask_area count = 1
proposal.confidence > baseline.confidence count = 1
new largest-extent test definitions = 8
source_manifest changed = NO
pipeline.py changed = NO
```

Any failure => STOP.

# 8. TARGETED PYTEST

Run exactly once:

```text
<REQUIRED_PYTHON> -m pytest delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py -q
```

Require exit 0.

Non-zero => STOP. Do not edit code/tests.

# 9. CANONICAL FULL PYTEST

Only if targeted PASS:

```text
<REQUIRED_PYTHON> -m pytest delivery_src\BuildReasonSeg_Advisor_RC1\tests -q
```

Require exit 0.

Non-zero => STOP.

# 10. EXTERNAL IMMUTABILITY

Hash actual external file after all test gates:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

Require exactly:

```text
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
```

No external sync/write is permitted.

# 11. MANIFEST POLICY

Canonical `source_manifest.json` intentionally remains unchanged in R2.

Record:

```text
manifest_status = INTENTIONALLY_STALE_PENDING_E3B
```

Do not run sync helper even in check mode.

# 12. FIXED EVIDENCE

If and only if all previous gates PASS, overwrite/create:

```text
evaluation/task8b3_ref01_eligibility_repair_impl.json
```

with exactly:

```json
{
  "task": "8B.3-REF01-E3A-R2",
  "starting_head": "fc2a33325eacf6ce5f366d6574abed3694431f5b",
  "branch": "fix/task8b3-ref01-eligibility-repair-impl",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "implementation_scope": "CANONICAL_ONLY_NO_SYNC",
  "patch_method": "UNIQUE_FUNCTION_BOUNDARY_PATCHER",
  "patcher_sha256": "2fa5f9991dc6e4782d668229430544572e9664f266852d40881a9c3a97a323d5",
  "detector_model_calls": 0,
  "eligible_function_changed": false,
  "eligible_proposals_function_changed": false,
  "largest_only_exception": true,
  "strict_area_operator": ">",
  "strict_confidence_operator": ">",
  "no_baseline_behavior": "NO_EXCEPTION_SAFE_FAILURE",
  "border_exception_allowed": false,
  "smallest_family_changed": false,
  "new_numeric_thresholds": [],
  "patcher_exit": 0,
  "py_compile_detector_exit": 0,
  "py_compile_test_exit": 0,
  "targeted_test_exit": 0,
  "canonical_full_test_exit": 0,
  "external_detector_after": "82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738",
  "external_detector_unchanged": true,
  "source_manifest_updated": false,
  "manifest_status": "INTENTIONALLY_STALE_PENDING_E3B",
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC"
}
```

Do not add/remove/rename keys.

# 13. REPORT / FROM_DSH

Update `docs/task8b3_ref01_eligibility_repair_impl.md` with a new authoritative E3A-R2 section.

Required COMPLETE facts:

```text
Task = 8B.3-REF01-E3A-R2
Status = COMPLETE
Design = LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
Patch method = UNIQUE_FUNCTION_BOUNDARY_PATCHER
Patcher SHA256 = 2fa5f9991dc6e4782d668229430544572e9664f266852d40881a9c3a97a323d5
Detector/model calls = 0
eligible() changed = NO
eligible_proposals() changed = NO
Largest-only exception = YES
Strict area operator = >
Strict confidence operator = >
Border bypass = NO
No-baseline exception = NO
Smallest family changed = NO
New numeric threshold = NO
Targeted pytest = PASS
Canonical full pytest = PASS
External RC1 modified = NO
Source manifest updated = NO
Manifest status = INTENTIONALLY_STALE_PENDING_E3B
Outcome = REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTED
NEXT = REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
```

Also state:

```text
PROP-01 remains PROP01_OPEN_ENGINEERING_DEFECT.
The left and below reference-selection defects remain unresolved.
No final Demo inference was executed.
NEXT was not executed.
```

Preserve the ARTIFACT-FACTS block in `handoff/FROM_DSH.md` exactly.

For COMPLETE, FROM_DSH must include:

```text
Task: 8B.3-REF01-E3A-R2
Status: COMPLETE
Branch: fix/task8b3-ref01-eligibility-repair-impl
Starting HEAD: fc2a33325eacf6ce5f366d6574abed3694431f5b
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Patch method: UNIQUE_FUNCTION_BOUNDARY_PATCHER
Patcher SHA256: 2fa5f9991dc6e4782d668229430544572e9664f266852d40881a9c3a97a323d5
Patcher runs: 1
Patcher exit: 0
Detector/model calls: 0
Canonical detector modified: YES
Canonical test file modified: YES
eligible() changed: NO
eligible_proposals() changed: NO
Largest-only private helper added: YES
Strict area operator: >
Strict confidence operator: >
Border bypass allowed: NO
No-baseline exception allowed: NO
Smallest family changed: NO
New numeric threshold: NO
Targeted test: PASS
Canonical full tests: PASS
External detector after SHA: 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
External detector unchanged: YES
Source manifest updated: NO
Manifest status: INTENTIONALLY_STALE_PENDING_E3B
Outcome: REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTED
Next gate: REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
External/canonical product files modified: NO / YES
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

For STOP, report only failed gate + observed facts. Do not invent PASS values.

# 14. FINAL STATUS / COMMIT MESSAGE GATE

Run:

```text
git status --porcelain
git diff --name-only fc2a33325eacf6ce5f366d6574abed3694431f5b
```

COMPLETE allowed only if:
- patcher ran once and passed;
- temporary patcher deleted;
- both py_compile pass;
- targeted pytest pass;
- full canonical pytest pass;
- external detector SHA exact;
- source_manifest unchanged;
- pipeline.py unchanged;
- only six allowed tracked paths differ;
- no untracked repo files remain.

If COMPLETE:

```text
STATUS = COMPLETE
COMMIT_MESSAGE = fix(rc1): implement largest extent dominance exception
```

Otherwise:

```text
STATUS = STOP
COMMIT_MESSAGE = fix(rc1): record extent dominance implementation stop
```

Write FROM_DSH Status first, then use exactly the matching message.

# 15. PUSH / STOP

Push only current branch:

```text
fix/task8b3-ref01-eligibility-repair-impl
```

No force push.
Do not update main.
Do not execute NEXT.

After push: STOP and wait for ChatGPT.

# 16. COMPLETE DEFINITION

COMPLETE only if every gate above passes exactly and no autonomous repair occurs.
