"""Task 6X — frozen SAM2.1 box-prompt proposal refinement (support infrastructure).

Four predeclared refinement options over the frozen U-C1 YOLO proposals (section 6-9):

```text
X-C0  no refinement: original U-C1 YOLO masks + exact Task 6Q deterministic selection
X-C1  exact YOLO bbox prompt, multimask_output=False
X-C2  exact YOLO bbox prompt, multimask_output=True  -> mask with max SAM2 predicted quality
X-C3  10%-expanded YOLO bbox (clipped to [0,511]), multimask_output=True -> max SAM quality mask
```

For X-C1/X-C2/X-C3 the prompt is a **box only**: `point_coords=None`, `point_labels=None`,
`mask_input=None`, `return_logits=False`. The SAM2 predicted quality score is recorded but **never
thresholded**. Refined masks are converted to 512x512 boolean masks and the exact Task 6Q family
eligibility is applied again; an empty refined mask invalidates the candidate. No duplicate/IoU
suppression. Selection keeps the literal largest=max refined area / smallest=min refined area semantics
with the mandated tie-break: (1) higher SAM2 predicted quality score, (2) higher original YOLO
confidence, (3) lower original YOLO proposal index. X-C0 keeps the exact Task 6Q original tie-break.

No training, no SAM quality threshold, no points, no mask prompts, no TTA/tiling.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from buildreasonseg.runtime._frozen.mvp.task6q_reference_resolver import build_proposal, is_eligible

TILE = 512
BOX_EXPANSION_FRACTION = 0.10
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CHECKPOINT_SHA256 = "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM2_CHECKPOINT = PROJECT_ROOT / "local_cache" / "models" / "sam2.1_hiera_base_plus.pt"

#: Exactly four predeclared options (section 5-9). No fifth option may be introduced.
OPTIONS: dict[str, dict] = {
    "X-C0": {"id": "X-C0", "refinement": False, "description": "no refinement (U-C1 baseline)"},
    "X-C1": {"id": "X-C1", "refinement": True, "box": "exact", "multimask_output": False,
             "description": "exact box, single-mask SAM2"},
    "X-C2": {"id": "X-C2", "refinement": True, "box": "exact", "multimask_output": True,
             "description": "exact box, multimask SAM2 (max predicted quality)"},
    "X-C3": {"id": "X-C3", "refinement": True, "box": "expanded10", "multimask_output": True,
             "description": "10% expanded box, multimask SAM2 (max predicted quality)"},
}
OPTION_ORDER = ("X-C0", "X-C1", "X-C2", "X-C3")
SIMPLICITY_ORDER = {option: index for index, option in enumerate(OPTION_ORDER)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_box_10pct(bbox_xyxy, size: int = TILE) -> np.ndarray:
    """Section 9 expansion: 10% of width/height on each side, clipped to [0, size-1]."""

    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    width = x2 - x1
    height = y2 - y1
    expanded = np.asarray([
        x1 - BOX_EXPANSION_FRACTION * width,
        y1 - BOX_EXPANSION_FRACTION * height,
        x2 + BOX_EXPANSION_FRACTION * width,
        y2 + BOX_EXPANSION_FRACTION * height,
    ], dtype=np.float32)
    return np.clip(expanded, 0.0, float(size - 1))


def prompt_box(proposal, option: str) -> np.ndarray:
    """The exact box prompt for one option (XYXY source pixels)."""

    config = OPTIONS[option]
    if not config["refinement"]:
        raise ValueError(f"{option} has no refinement step")
    if config["box"] == "exact":
        return np.asarray([float(value) for value in proposal.bbox_xyxy], dtype=np.float32)
    if config["box"] == "expanded10":
        return expand_box_10pct(proposal.bbox_xyxy)
    raise ValueError(f"unknown box mode {config['box']!r}")


def load_sam2_image_predictor(device: str = "cuda"):
    """Official `sam2.sam2_image_predictor.SAM2ImagePredictor` over the frozen checkpoint."""

    os.environ.setdefault("SAM2_BUILD_CUDA", "0")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(SAM2_CHECKPOINT)
    sam = build_sam2(SAM2_CONFIG_NAME, str(SAM2_CHECKPOINT), device=device)
    sam.eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    predictor = SAM2ImagePredictor(sam)
    return predictor, {
        "config": SAM2_CONFIG_NAME,
        "checkpoint": str(SAM2_CHECKPOINT),
        "checkpoint_sha256": sha256_file(SAM2_CHECKPOINT),
        "expected_sha256": SAM2_CHECKPOINT_SHA256,
        "predictor_class": "sam2.sam2_image_predictor.SAM2ImagePredictor",
        "trainable_parameters": 0,
    }


@dataclass
class RefinedCandidate:
    original_index: int
    original_confidence: float
    sam_score: float
    mask: np.ndarray
    empty: bool


class Sam2ProposalRefiner:
    """Refine eligible U-C1 proposals with frozen SAM2 box prompts for one option."""

    def __init__(self, predictor, option: str, device: str = "cuda"):
        if option not in OPTIONS or not OPTIONS[option]["refinement"]:
            raise ValueError(f"{option} is not a refinement option")
        self.predictor = predictor
        self.option = option
        self.device = device
        self._image_key = None
        self.calls = 0

    def set_tile(self, image_rgb: np.ndarray, tile_key: str) -> None:
        """Cache the SAM2 image embedding once per tile."""

        if self._image_key != tile_key:
            self.predictor.set_image(image_rgb)
            self._image_key = tile_key

    def refine(self, proposal) -> RefinedCandidate:
        config = OPTIONS[self.option]
        box = prompt_box(proposal, self.option)
        masks, scores, _logits = self.predictor.predict(
            point_coords=None, point_labels=None, box=box[None, :], mask_input=None,
            multimask_output=bool(config["multimask_output"]), return_logits=False)
        self.calls += 1
        scores = np.asarray(scores).reshape(-1)
        if config["multimask_output"]:
            choice = int(np.argmax(scores))          # max SAM2 predicted quality, no threshold
        else:
            choice = 0
        mask = np.asarray(masks[choice]).astype(bool)
        if mask.shape != (TILE, TILE):
            mask = _resize_nearest(mask, TILE)
        return RefinedCandidate(original_index=int(proposal.index),
                                original_confidence=float(proposal.confidence),
                                sam_score=float(scores[choice]), mask=mask,
                                empty=bool(not mask.any()))


def _resize_nearest(mask: np.ndarray, size: int) -> np.ndarray:
    import torch.nn.functional as F

    tensor = torch.as_tensor(mask.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(size, size), mode="nearest")[0, 0].numpy()
    return resized > 0.5


def refined_proposal(candidate: RefinedCandidate):
    """Wrap a refined mask as a Task 6Q proposal (so the unchanged eligibility rules apply)."""

    return build_proposal(candidate.original_index, candidate.original_confidence, candidate.mask)


def eligible_original(proposals: list, family: str) -> list:
    """Section 5: the exact Task 6Q family eligibility on the ORIGINAL YOLO proposals."""

    return [proposal for proposal in proposals if is_eligible(proposal, family)]


def eligible_refined(candidates: list[RefinedCandidate], family: str) -> list[tuple]:
    """Section 10: exact Task 6Q family eligibility applied again on the refined masks.

    Empty refined masks are invalid and removed; no duplicate/IoU suppression and no SAM threshold.
    """

    kept = []
    for candidate in candidates:
        if candidate.empty:
            continue
        proposal = refined_proposal(candidate)
        if is_eligible(proposal, family):
            kept.append((proposal, candidate))
    return kept


def select_refined(kept: list[tuple], family: str):
    """Literal largest/smallest on refined area with the mandated tie-break."""

    if not kept:
        return None
    key = (lambda item: (item[0].area_px, item[1].sam_score, item[1].original_confidence,
                         -item[1].original_index)) if family == "largest" else \
          (lambda item: (-item[0].area_px, item[1].sam_score, item[1].original_confidence,
                         -item[1].original_index))
    return max(kept, key=key)


def option_report() -> dict:
    return {
        "options": {option: OPTIONS[option]["description"] for option in OPTION_ORDER},
        "option_count": len(OPTION_ORDER),
        "box_expansion_fraction": BOX_EXPANSION_FRACTION,
        "sam_quality_threshold": None,
        "sam_quality_score_used_as": "selection tie-break only (recorded, never thresholded)",
        "point_prompt": False,
        "mask_input": False,
        "tta": False,
        "tiling": False,
        "returns_logits": False,
        "training_performed": False,
    }


__all__ = [
    "BOX_EXPANSION_FRACTION",
    "OPTIONS",
    "OPTION_ORDER",
    "PROJECT_ROOT",
    "RefinedCandidate",
    "SAM2_CHECKPOINT",
    "SAM2_CHECKPOINT_SHA256",
    "SIMPLICITY_ORDER",
    "Sam2ProposalRefiner",
    "TILE",
    "eligible_original",
    "eligible_refined",
    "expand_box_10pct",
    "load_sam2_image_predictor",
    "option_report",
    "prompt_box",
    "refined_proposal",
    "select_refined",
    "sha256_file",
]
