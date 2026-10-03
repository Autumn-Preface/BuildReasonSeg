"""Task 6Q Part C — deterministic frozen-proposal reference resolver.

`largest` / `smallest` are **instance-set ranking operations**, so this module resolves the reference
building from the frozen Task 6M.1 YOLO26m-seg proposals with the exact deterministic rules of
Task 6Q section 6. It trains nothing, tunes nothing and never consults ground truth: eligibility,
ranking and tie-breaks depend only on predicted masks, predicted confidences and the family id.

Frozen proposal configuration (Task 6Q section 4): YOLO26m-seg, `imgsz = 640`, `conf = 0.10`,
`max_det = 100`, default NMS, no TTA, no tiling, no threshold sweep.

Proposal normalization (section 5): masks come back at the source 512x512 resolution via
`retina_masks=True`; binarization is the **canonical Task 6M segmentation threshold**, i.e.
Ultralytics' own `masks > 0` followed by `buildreasonseg.runtime._frozen.mvp.task6m_eval.normalize_mask` — no new
threshold is invented.

```text
touches_border  = any positive pixel on row 0, row 511, col 0 or col 511
largest  eligible: NOT touches_border AND bbox_extent_ratio <= 0.20            (tiny not rejected)
smallest eligible: NOT touches_border AND bbox_extent_ratio <= 0.20 AND area_px >= 150
tie-break (equal area): higher confidence, then lower original proposal index
abstention: zero proposals, or zero eligible proposals for the requested family
```
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

import numpy as np

TILE_SIZE = 512
MERGE_BBOX_EXTENT_RATIO_MAX = 0.20
TINY_COMPONENT_AREA_PX = 150
FAMILIES: tuple[str, ...] = ("largest", "smallest")

#: Frozen Task 6Q section 4 inference configuration.
PROPOSAL_CHECKPOINT = (
    Path("artifacts") / "checkpoints" / "task6m1" / "runs" / "m1_yolo26m_seg_continued"
    / "weights" / "best.pt"
)
PROPOSAL_CHECKPOINT_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
PROPOSAL_FAMILY = "YOLO26m-seg"
PROPOSAL_IMGSZ = 640
PROPOSAL_CONF = 0.10
PROPOSAL_MAX_DET = 100
PROPOSAL_TTA = False
PROPOSAL_TILING = False


def family_of_program(program_id: str) -> str:
    family = str(program_id).split("_", 1)[0]
    if family not in FAMILIES:
        raise ValueError(f"program {program_id!r} has no largest/smallest reference family")
    return family


@dataclass
class Proposal:
    """One predicted building instance, normalized to source 512x512 coordinates."""

    index: int
    confidence: float
    mask: np.ndarray
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    bbox_area: int
    bbox_extent_ratio: float
    touches_border: bool

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "confidence": self.confidence,
            "area_px": self.area_px,
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_area": self.bbox_area,
            "bbox_extent_ratio": self.bbox_extent_ratio,
            "touches_border": self.touches_border,
        }


@dataclass
class Selection:
    """The resolver's answer for one family on one tile."""

    family: str
    abstained: bool
    reason: str
    proposal: Proposal | None = None
    proposals_total: int = 0
    proposals_eligible: int = 0
    eligible_indices: list[int] = dataclass_field(default_factory=list)

    @property
    def mask(self) -> np.ndarray | None:
        return None if self.proposal is None else self.proposal.mask

    def as_dict(self) -> dict:
        payload = {
            "family": self.family,
            "abstained": self.abstained,
            "reason": self.reason,
            "proposals_total": self.proposals_total,
            "proposals_eligible": self.proposals_eligible,
            "eligible_indices": self.eligible_indices,
            "selected": None if self.proposal is None else self.proposal.as_dict(),
        }
        return payload


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive-exclusive `(x0, y0, x1, y1)` bounding box of a boolean mask."""

    rows = np.any(mask, axis=1)
    columns = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(columns)[0][[0, -1]]
    return (int(x0), int(y0), int(x1) + 1, int(y1) + 1)


def touches_border_exact(mask: np.ndarray) -> bool:
    """Task 6Q section 5: any positive pixel on row 0/511 or column 0/511."""

    if not mask.any():
        return False
    return bool(
        mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def build_proposal(index: int, confidence: float, mask: np.ndarray) -> Proposal:
    """Normalize one predicted mask into a `Proposal` (no GT involved)."""

    binary = np.asarray(mask).astype(bool)
    if binary.shape != (TILE_SIZE, TILE_SIZE):
        raise ValueError(f"proposal mask must be {TILE_SIZE}x{TILE_SIZE}, got {binary.shape}")
    area = int(binary.sum())
    bbox = bbox_from_mask(binary)
    bbox_area = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    extent = bbox_area / float(TILE_SIZE * TILE_SIZE)
    return Proposal(
        index=int(index),
        confidence=float(confidence),
        mask=binary,
        area_px=area,
        bbox_xyxy=bbox,
        bbox_area=bbox_area,
        bbox_extent_ratio=extent,
        touches_border=touches_border_exact(binary),
    )


def is_eligible(proposal: Proposal, family: str) -> bool:
    """Task 6Q section 6 eligibility for one family."""

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if proposal.touches_border:
        return False
    if proposal.bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX:
        return False
    if family == "smallest" and proposal.area_px < TINY_COMPONENT_AREA_PX:
        return False
    return True


def _sort_key(proposal: Proposal) -> tuple:
    """Tie-break (section 6): higher confidence first, then lower original index."""

    return (-proposal.confidence, proposal.index)


def select_reference(proposals: list[Proposal], family: str) -> Selection:
    """Deterministic largest/smallest selection with explicit abstention."""

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if not proposals:
        return Selection(family=family, abstained=True, reason="no_proposals", proposals_total=0)

    eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
    if not eligible:
        return Selection(family=family, abstained=True, reason="no_eligible_proposals",
                         proposals_total=len(proposals), proposals_eligible=0)

    if family == "largest":
        best_area = max(proposal.area_px for proposal in eligible)
    else:
        best_area = min(proposal.area_px for proposal in eligible)
    tied = [proposal for proposal in eligible if proposal.area_px == best_area]
    chosen = sorted(tied, key=_sort_key)[0]
    return Selection(
        family=family,
        abstained=False,
        reason="selected",
        proposal=chosen,
        proposals_total=len(proposals),
        proposals_eligible=len(eligible),
        eligible_indices=sorted(proposal.index for proposal in eligible),
    )


def proposals_from_results(results) -> list[Proposal]:
    """Convert one Ultralytics result into normalized proposals (canonical 6M binarization)."""

    from buildreasonseg.runtime._frozen.mvp.task6m_eval import normalize_mask

    if results.masks is None or results.boxes is None:
        return []
    masks = results.masks.data.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    proposals = []
    for index in range(masks.shape[0]):
        proposals.append(build_proposal(index, float(confidences[index]),
                                        normalize_mask(masks[index])))
    return proposals


def run_frozen_proposals(model, image_path: Path, device: str = "0") -> list[Proposal]:
    """Run the frozen proposal model exactly once on one tile with the frozen configuration."""

    results = model.predict(
        source=str(image_path), imgsz=PROPOSAL_IMGSZ, conf=PROPOSAL_CONF, max_det=PROPOSAL_MAX_DET,
        verbose=False, device=device, retina_masks=True,
    )[0]
    return proposals_from_results(results)


def config_report() -> dict:
    return {
        "checkpoint": str(PROPOSAL_CHECKPOINT),
        "checkpoint_sha256_expected": PROPOSAL_CHECKPOINT_SHA256,
        "family": PROPOSAL_FAMILY,
        "imgsz": PROPOSAL_IMGSZ,
        "conf": PROPOSAL_CONF,
        "max_det": PROPOSAL_MAX_DET,
        "nms": "default",
        "tta": PROPOSAL_TTA,
        "tiling": PROPOSAL_TILING,
        "threshold_sweep": False,
        "mask_threshold": "canonical Task 6M (Ultralytics retina mask > 0 via normalize_mask)",
        "merge_bbox_extent_ratio_max": MERGE_BBOX_EXTENT_RATIO_MAX,
        "tiny_component_area_px": TINY_COMPONENT_AREA_PX,
    }


def pack_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Pack boolean masks into `np.packbits` form for a compact, gitignored resolver cache."""

    if not masks:
        return np.zeros((0, TILE_SIZE * TILE_SIZE // 8), dtype=np.uint8)
    stacked = np.stack([np.asarray(mask).astype(bool) for mask in masks])
    return np.packbits(stacked.reshape(len(masks), -1), axis=1)


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(np.asarray(packed, dtype=np.uint8))[: TILE_SIZE * TILE_SIZE].reshape(
        TILE_SIZE, TILE_SIZE
    ).astype(bool)


__all__ = [
    "FAMILIES",
    "MERGE_BBOX_EXTENT_RATIO_MAX",
    "PROPOSAL_CHECKPOINT",
    "PROPOSAL_CHECKPOINT_SHA256",
    "PROPOSAL_CONF",
    "PROPOSAL_IMGSZ",
    "PROPOSAL_MAX_DET",
    "Proposal",
    "Selection",
    "TILE_SIZE",
    "TINY_COMPONENT_AREA_PX",
    "bbox_from_mask",
    "build_proposal",
    "config_report",
    "family_of_program",
    "is_eligible",
    "pack_masks",
    "proposals_from_results",
    "run_frozen_proposals",
    "select_reference",
    "touches_border_exact",
    "unpack_mask",
]
