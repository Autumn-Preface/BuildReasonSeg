# TO_DSH — Task 8B.3-REF01-E3A: Implement Largest Extent-Dominance Repair in Canonical RC1

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required base branch: `fix/task8b3-ref01-eligibility-repair-design`
> Required base HEAD: `14f252e4cd70c8d62d0ceba9a389665dcb66ea68`
> New task branch: `fix/task8b3-ref01-eligibility-repair-impl`
> Canonical RC1: `delivery_src\BuildReasonSeg_Advisor_RC1`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. EXECUTOR CONTRACT

ChatGPT has already made the implementation decision. DSH MUST NOT redesign it.

DSH may only verify preconditions, apply the exact source/test patches below, run the exact tests, write the required evidence/report, commit/push with the status-matched message, and STOP.

Any unexpected condition => STOP. Do not self-repair.

# 1. APPROVED DESIGN

```text
Task 8B.3-REF01-E2: APPROVED
Design: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

right 1 -> 1
left 14 -> 14
above 4 -> 5
below 1 -> 1

Synthetic contract: 8/8 PASS
No new numeric threshold.
Border exclusion mandatory.
Smallest-family behavior unchanged.
```

# 2. FIXED IMPLEMENTATION ARCHITECTURE

`eligible()` and `eligible_proposals()` remain the frozen base eligibility API and MUST NOT change.

The exception is implemented only inside reference selection through two private helpers:

```text
_reference_rank()
_largest_reference_candidates_with_extent_exception()
```

`select_reference()` uses the new helper only for `family == "largest"`. Other families continue to use `eligible_proposals()`.

# 3. EXACT RULE

Base:
```text
eligible_proposals(proposals, family="largest")
```

No base candidate => no exception => `None`.

Exception requires ALL:
```text
mask_area > 0
mask_crop.any() == True
touches_image_border == False
bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX
mask_area > baseline.mask_area
confidence > baseline.confidence
```

Selection ranking:
```text
(-mask_area, -confidence, proposal_id)
```

Area/confidence operators are strict `>`.

# 4. GIT GATE

Require:
```text
branch = fix/task8b3-ref01-eligibility-repair-design
HEAD = 14f252e4cd70c8d62d0ceba9a389665dcb66ea68
```

Then create exactly:
```text
fix/task8b3-ref01-eligibility-repair-impl
```

No merge/rebase/reset/stash/clean/cherry-pick/force-push.

# 5. EXTERNAL IMMUTABILITY GATE

Before edits hash:
```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

Require exactly:
```text
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
```

Do not write external RC1 in E3A.

# 6. ALLOWED TRACKED CHANGES ONLY

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
evaluation/task8b3_ref01_eligibility_repair_impl.json
docs/task8b3_ref01_eligibility_repair_impl.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Must NOT change:
```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
scripts/sync_advisor_rc1_delivery.py
main
```

# 7. ABSOLUTE PROHIBITIONS

Do NOT:
- sync external RC1;
- edit source_manifest;
- edit pipeline.py;
- run detector/model inference;
- instantiate DetectorRuntime;
- run Qwen/SAM2/D-B1/target segmentation;
- change MERGE_BBOX_EXTENT_RATIO_MAX;
- change SMALLEST_MIN_AREA_PX;
- modify eligible();
- modify eligible_proposals();
- add public API;
- update __all__;
- execute NEXT.

Detector/model calls = 0.

# 8. EXACT DETECTOR PATCH

Target:
```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
```

Replace this exact OLD block:

```python
def eligible_proposals(proposals: list[GlobalProposal], *, family: str = "largest"
                       ) -> list[GlobalProposal]:
    return [proposal for proposal in proposals if eligible(proposal, family=family)]


def select_reference(proposals: list[GlobalProposal], *, family: str = "largest") -> GlobalProposal | None:
    """Largest eligible building: area desc → confidence desc → global id asc."""

    candidates = eligible_proposals(proposals, family=family)
    if not candidates:
        return None
    return sorted(candidates, key=lambda proposal: (-proposal.mask_area, -proposal.confidence,
                                                    proposal.proposal_id))[0]
```

with exactly:

```python
def eligible_proposals(proposals: list[GlobalProposal], *, family: str = "largest"
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
```

If OLD occurrence != 1 => STOP.

# 9. BASE ELIGIBILITY MUST REMAIN UNCHANGED

Require this exact fragment still exists after patch:

```python
def eligible(proposal: GlobalProposal, *, family: str = "largest") -> bool:
    """Frozen U-C1 eligibility: non-empty, not touching the original image border, extent <= 0.20."""

    if proposal.mask_area <= 0 or not proposal.mask_crop.any():
        return False
    if proposal.touches_image_border:
        return False
    if proposal.bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX:
        return False
    if family == "smallest" and proposal.mask_area < SMALLEST_MIN_AREA_PX:
        return False
    return True
```

And `eligible_proposals()` must still be exactly the original one-line filter.

Any difference => STOP.

# 10. EXACT TEST PATCH

Target:
```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
```

Insert immediately BEFORE the unique marker:
```python
# ---------------------------------------------------------------- reasoning context
```

Insert exactly:

```python
def _reference_rect(height: int, width: int, confidence: float, proposal_id: int,
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
```

Do not add any other test.

# 11. STATIC GATE

Compile both files with REQUIRED_PYTHON. Both exit 0.

Require:
```text
_reference_rank definition count = 1
_largest_reference_candidates_with_extent_exception definition count = 1
proposal.mask_area > baseline.mask_area count = 1
proposal.confidence > baseline.confidence count = 1
8 new tests present
eligible() unchanged
eligible_proposals() unchanged
```

Any failure => STOP.

# 12. TARGETED TEST

Run exactly:
```text
<REQUIRED_PYTHON> -m pytest delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py -q
```

Require exit 0. Non-zero => STOP, no self-repair.

# 13. CANONICAL FULL TEST

Only after targeted PASS:
```text
<REQUIRED_PYTHON> -m pytest delivery_src\BuildReasonSeg_Advisor_RC1\tests -q
```

Require exit 0. Non-zero => STOP.

# 14. MANIFEST / SYNC POLICY

Do NOT update:
```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Do NOT sync external RC1.

Record:
```text
manifest_status = INTENTIONALLY_STALE_PENDING_E3B
```

# 15. EXTERNAL POST-GATE

External detector SHA must still equal:
```text
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
```

Otherwise STOP.

# 16. EVIDENCE FILE

Create:
```text
evaluation/task8b3_ref01_eligibility_repair_impl.json
```

Required exact fields:

```text
task = 8B.3-REF01-E3A
starting_head = 14f252e4cd70c8d62d0ceba9a389665dcb66ea68
branch = fix/task8b3-ref01-eligibility-repair-impl
design_id = LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
implementation_scope = CANONICAL_ONLY_NO_SYNC
detector_model_calls = 0
eligible_function_changed = false
eligible_proposals_function_changed = false
largest_only_exception = true
strict_area_operator = >
strict_confidence_operator = >
no_baseline_behavior = NO_EXCEPTION_SAFE_FAILURE
border_exception_allowed = false
smallest_family_changed = false
new_numeric_thresholds = []
targeted_test_exit = 0
canonical_full_test_exit = 0
external_detector_before = 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
external_detector_after = 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
external_detector_unchanged = true
manifest_status = INTENTIONALLY_STALE_PENDING_E3B
overall_outcome = REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTED
next_gate = REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
```

# 17. REPORT

Create:
```text
docs/task8b3_ref01_eligibility_repair_impl.md
```

Required nonclaims:

```text
The external RC1 delivery was not modified.
The source manifest was not updated in E3A.
No detector/model inference was executed.
No new numeric threshold was introduced.
The repair applies only to largest-family automatic reference selection.
The smallest-family behavior remains unchanged.
PROP-01 remains PROP01_OPEN_ENGINEERING_DEFECT.
The left and below reference-selection defects remain unresolved.
This task does not claim final Demo success.
```

# 18. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required fields:

```text
Task: 8B.3-REF01-E3A
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-impl
Starting HEAD: 14f252e4cd70c8d62d0ceba9a389665dcb66ea68
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Canonical detector modified: YES / NO
Canonical test file modified: YES / NO
eligible() changed: NO / YES
eligible_proposals() changed: NO / YES
Largest-only private helper added: YES / NO
Strict area operator: >
Strict confidence operator: >
Border bypass allowed: NO
No-baseline exception allowed: NO
Smallest family changed: NO
New numeric threshold: NO
Targeted test: PASS / FAIL
Canonical full tests: PASS / FAIL
External detector before SHA: <full sha>
External detector after SHA: <full sha>
External detector unchanged: YES / NO
Source manifest updated: NO
Manifest status: INTENTIONALLY_STALE_PENDING_E3B
Outcome: REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTED / other
Next gate: REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC / other
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
External/canonical product files modified: NO / YES
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 19. FINAL DIFF GATE

Allowed ONLY the six paths in §6.

If source_manifest appears => STOP.

# 20. STATUS → COMMIT MESSAGE

If all gates pass:
```text
STATUS = COMPLETE
COMMIT_MESSAGE = fix(rc1): implement largest extent dominance exception
```

Otherwise:
```text
STATUS = STOP
COMMIT_MESSAGE = fix(rc1): record extent dominance implementation stop
```

Write Status first, then use the matching message exactly.

# 21. COMMIT / PUSH

Push only:
```text
fix/task8b3-ref01-eligibility-repair-impl
```

No force push. Do not update main. Then STOP.

# 22. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- only six allowed paths changed;
- external delivery unchanged;
- source_manifest unchanged;
- exact implementation architecture used;
- eligible() unchanged;
- eligible_proposals() unchanged;
- exception only largest family;
- no baseline => no exception;
- border cannot bypass;
- strict area/confidence dominance;
- smallest unchanged;
- exactly 8 new tests;
- targeted test PASS;
- canonical full test PASS;
- detector/model calls 0;
- manifest intentionally stale pending E3B;
- status/commit-message consistency PASS;
- evidence/report/FROM_DSH committed/pushed;
- NEXT not executed;
- STOP.
