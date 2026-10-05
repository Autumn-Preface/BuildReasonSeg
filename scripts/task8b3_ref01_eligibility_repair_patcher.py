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
EXPECTED_HEAD = "628e9dedefcce6e2cba8a832de882d1df2b714cf"
EXPECTED_EXTERNAL_SHA = "82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738"

OLD_LF = r"""def eligible_proposals(proposals: list[GlobalProposal], *, family: str = "largest"
                       ) -> list[GlobalProposal]:
    return [proposal for proposal in proposals if eligible(proposal, family=family)]


def select_reference(proposals: list[GlobalProposal], *, family: str = "largest") -> GlobalProposal | None:
    \"\"\"Largest eligible building: area desc → confidence desc → global id asc.\"\"\"

    candidates = eligible_proposals(proposals, family=family)
    if not candidates:
        return None
    return sorted(candidates, key=lambda proposal: (-proposal.mask_area, -proposal.confidence,
                                                    proposal.proposal_id))[0]
"""

NEW_LF = r"""def eligible_proposals(proposals: list[GlobalProposal], *, family: str = "largest"
                       ) -> list[GlobalProposal]:
    return [proposal for proposal in proposals if eligible(proposal, family=family)]


def _reference_rank(proposal: GlobalProposal) -> tuple[int, float, int]:
    return (-proposal.mask_area, -proposal.confidence, proposal.proposal_id)


def _largest_reference_candidates_with_extent_exception(
        proposals: list[GlobalProposal]) -> list[GlobalProposal]:
    \"\"\"Frozen base candidates plus the RC1 largest-only extent-dominance exception.\"\"\"

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
    \"\"\"Select a reference with the frozen base rule and the largest-only RC1 exception.\"\"\"

    if family == "largest":
        candidates = _largest_reference_candidates_with_extent_exception(proposals)
    else:
        candidates = eligible_proposals(proposals, family=family)
    if not candidates:
        return None
    return sorted(candidates, key=_reference_rank)[0]
"""

TEST_BLOCK_LF = r"""def _reference_rect(height: int, width: int, confidence: float, proposal_id: int,
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

MARKER_LF = "# ---------------------------------------------------------------- reasoning context"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def git_text(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()

def newline_style(raw: bytes) -> bytes:
    if b"\r\n" in raw:
        assert b"\n" not in raw.replace(b"\r\n", b"")
        return b"\r\n"
    assert b"\n" in raw
    return b"\n"

def adapt(text: str, nl: bytes) -> bytes:
    return text.encode("utf-8").replace(b"\n", nl)

def main() -> int:
    assert git_text("branch", "--show-current") == EXPECTED_BRANCH
    assert git_text("rev-parse", "HEAD") == EXPECTED_HEAD
    assert sha256(EXTERNAL_DETECTOR) == EXPECTED_EXTERNAL_SHA

    detector_raw = DETECTOR.read_bytes()
    test_raw = TEST.read_bytes()
    detector_nl = newline_style(detector_raw)
    test_nl = newline_style(test_raw)

    old = adapt(OLD_LF, detector_nl)
    new = adapt(NEW_LF, detector_nl)
    marker = adapt(MARKER_LF, test_nl)
    test_block = adapt(TEST_BLOCK_LF, test_nl)

    assert detector_raw.count(old) == 1
    assert detector_raw.count(b"_largest_reference_candidates_with_extent_exception") == 0
    assert test_raw.count(marker) == 1
    assert test_raw.count(b"test_largest_extent_dominance_exception_selects_strictly_dominant_candidate") == 0

    detector_after = detector_raw.replace(old, new, 1)
    test_after = test_raw.replace(marker, test_block + test_nl + test_nl + marker, 1)

    assert detector_after.count(b"def _reference_rank(") == 1
    assert detector_after.count(b"def _largest_reference_candidates_with_extent_exception(") == 1
    assert detector_after.count(b"proposal.mask_area > baseline.mask_area") == 1
    assert detector_after.count(b"proposal.confidence > baseline.confidence") == 1
    assert test_after.count(b"def test_largest_extent_") == 8

    DETECTOR.write_bytes(detector_after)
    TEST.write_bytes(test_after)

    print("PATCHER_BRANCH_HEAD: PASS")
    print("EXTERNAL_DETECTOR_IDENTITY: PASS")
    print("DETECTOR_EXACT_REPLACEMENT: PASS")
    print("TEST_EXACT_INSERTION: PASS")
    print("NEW_TEST_COUNT: 8")
    print("PATCHER_RESULT: PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
