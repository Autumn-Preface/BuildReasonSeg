"""Task 7G — `SetContextLargestSelector v1`: an 18-feature permutation-invariant largest-reference selector.

The selector chooses among the **frozen** U-C1 eligible `largest` proposals of one tile. Features come only from
the candidate's predicted mask/bbox/confidence and the other eligible predicted proposals of the same tile:
no GT, no SAM2/RGB, no absolute position, no direction/relation/program, no target information (Task 7G
sections 10-14). The architecture is exact (sections 16-18):

```text
proposal encoder (shared):  Linear(18->64) -> LayerNorm(64) -> GELU -> Linear(64->32) -> GELU   -> h_i in R^32
set context:                h_mean = mean_i h_i ; h_max = max_i h_i ; context = concat(h_mean, h_max) in R^64
candidate head:             z_i = concat(h_i, context) in R^96 -> Linear(96->32) -> GELU -> Linear(32->1)
```

Softmax over the valid candidates only; the loss is exactly one listwise `CrossEntropyLoss` on the oracle-best
candidate index (section 19). No Transformer/attention/GNN, no pairwise learned interaction beyond the frozen
overlap features and the pooled set context.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

FEATURE_DIM = 18
SOURCE_AREA = 512 * 512
EPS = 1.0e-6
PROPOSAL_CONFIG = {"imgsz": 640, "conf": 0.05, "max_det": 300, "nms": "default", "tta": False,
                   "tiling": False}
FEATURE_NAMES = (
    "log_area", "area_ratio", "area_over_set_max", "area_rank_desc", "confidence",
    "bbox_extent_ratio", "width_ratio", "height_ratio", "fill_ratio", "abs_log_aspect",
    "boundary_over_sqrt_area", "max_iou_other", "mean_iou_other", "max_self_contained_by_other",
    "max_other_contained_in_self", "self_contained_count_norm", "contains_other_count_norm",
    "proposal_count_norm",
)


class SelectorDependencyUnavailable(RuntimeError):
    """Raised when a required frozen dependency (SciPy) is missing; Task 7G must not install anything."""


def _check_scipy() -> None:
    try:  # pragma: no cover - the import is the check
        import scipy  # noqa: F401
    except ImportError as error:  # pragma: no cover - defensive
        raise SelectorDependencyUnavailable("SELECTOR_DEPENDENCY_UNAVAILABLE") from error


# ---------------------------------------------------------------- features


def proposal_feature_matrix(proposals: list) -> np.ndarray:
    """The exact 18-dimensional feature matrix for one tile's eligible proposals (section 11-14)."""

    _check_scipy()
    count = len(proposals)
    if count == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    masks = [np.asarray(proposal.mask, dtype=bool) for proposal in proposals]
    areas = np.asarray([float(mask.sum()) for mask in masks], dtype=np.float64)
    bboxes = [tuple(proposal.bbox_xyxy) for proposal in proposals]
    widths = np.asarray([max(0, bbox[2] - bbox[0]) for bbox in bboxes], dtype=np.float64)
    heights = np.asarray([max(0, bbox[3] - bbox[1]) for bbox in bboxes], dtype=np.float64)
    bbox_areas = widths * heights
    confidences = np.asarray([float(proposal.confidence) for proposal in proposals],
                             dtype=np.float64)
    indices = [int(getattr(proposal, "index", position))
               for position, proposal in enumerate(proposals)]

    # feature 4: rank by descending area, stable tie-break = higher confidence then lower index
    order = sorted(range(count), key=lambda position: (-areas[position], -confidences[position],
                                                       indices[position]))
    ranks = np.zeros(count, dtype=np.float64)
    for rank, position in enumerate(order):
        ranks[position] = rank

    overlap = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            intersection = float((masks[left] & masks[right]).sum())
            union = float((masks[left] | masks[right]).sum())
            overlap[left, right] = overlap[right, left] = intersection / max(union, EPS)

    features = np.zeros((count, FEATURE_DIM), dtype=np.float64)
    max_area = max(areas.max(), EPS)
    for position in range(count):
        mask = masks[position]
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool),
                                        iterations=1, border_value=0)
        boundary_px = float((mask & ~eroded).sum())
        area = areas[position]
        width = widths[position]
        height = heights[position]
        self_in_other, other_in_self = [], []
        for other in range(count):
            if other == position:
                continue
            intersection = float((mask & masks[other]).sum())
            self_in_other.append(intersection / max(1.0, area))
            other_in_self.append(intersection / max(1.0, areas[other]))
        features[position] = [
            math.log1p(area) / math.log1p(SOURCE_AREA),
            area / SOURCE_AREA,
            area / max_area,
            ranks[position] / max(1, count - 1),
            confidences[position],
            max(width / 512.0, height / 512.0),
            width / 512.0,
            height / 512.0,
            area / max(1.0, bbox_areas[position]),
            abs(math.log((width + 1.0) / (height + 1.0))),
            boundary_px / math.sqrt(max(1.0, area)),
            max(overlap[position][other] for other in range(count) if other != position)
            if count > 1 else 0.0,
            float(np.mean([overlap[position][other] for other in range(count)
                           if other != position])) if count > 1 else 0.0,
            max(self_in_other) if self_in_other else 0.0,
            max(other_in_self) if other_in_self else 0.0,
            sum(1 for value in self_in_other if value >= 0.50) / max(1, count - 1),
            sum(1 for value in other_in_self if value >= 0.50) / max(1, count - 1),
            min(count, 300) / 300.0,
        ]
    return features.astype(np.float32)


# ---------------------------------------------------------------- model


class SetContextLargestSelector(nn.Module):
    """Section 16-18: shared proposal encoder, permutation-invariant set context, candidate score head."""

    def __init__(self) -> None:
        super().__init__()
        self.proposal_encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.score_head = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.proposal_encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`features` is (N, 18) for one tile; returns (N,) candidate scores."""

        hidden = self.encode(features)
        context = torch.cat([hidden.mean(dim=0), hidden.max(dim=0).values], dim=-1)
        broadcast = context.unsqueeze(0).expand(hidden.shape[0], -1)
        return self.score_head(torch.cat([hidden, broadcast], dim=-1)).squeeze(-1)

    def batch_forward(self, feature_list: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.forward(features) for features in feature_list]

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


def select_with_scores(scores: np.ndarray, proposals: list) -> int:
    """Section 25 G-S1: highest score; exact-float tie -> higher confidence, then lower index."""

    order = sorted(range(len(proposals)),
                   key=lambda position: (-float(scores[position]),
                                         -float(proposals[position].confidence),
                                         int(getattr(proposals[position], "index", position))))
    return order[0]


def deterministic_max_area(proposals: list) -> int:
    """The frozen F-R0 / G-S0 rule: max predicted area, tie higher confidence, then lower index."""

    order = sorted(range(len(proposals)),
                   key=lambda position: (-int(np.asarray(proposals[position].mask).sum()),
                                         -float(proposals[position].confidence),
                                         int(getattr(proposals[position], "index", position))))
    return order[0]


def oracle_best(proposals: list, gt_reference: np.ndarray) -> int:
    """Section 8 label: maximum GT IoU, tie higher confidence, then lower original index."""

    scored = []
    for position, proposal in enumerate(proposals):
        mask = np.asarray(proposal.mask, dtype=bool)
        intersection = float((mask & gt_reference).sum())
        union = float((mask | gt_reference).sum())
        scored.append((intersection / max(union, EPS), float(proposal.confidence),
                       -int(getattr(proposal, "index", position)), position))
    return max(scored)[3]


def selector_report() -> dict:
    model = SetContextLargestSelector()
    return {
        "name": "SetContextLargestSelector", "version": "v1",
        "feature_dim": FEATURE_DIM, "feature_names": list(FEATURE_NAMES),
        "proposal_encoder": ["Linear(18,64)", "LayerNorm(64)", "GELU", "Linear(64,32)", "GELU"],
        "set_context": ["mean_i h_i", "max_i h_i", "concat -> 64"],
        "candidate_head": ["concat(h_i, context) -> 96", "Linear(96,32)", "GELU", "Linear(32,1)"],
        "softmax": "over valid candidates only", "loss": "CrossEntropyLoss (listwise, single loss)",
        "extra_losses": [], "parameters": model.parameter_count(),
        "transformer": False, "attention": False, "gnn": False,
        "positional_encoding": False, "proposal_order_feature": False,
        "uses_gt_at_inference": False, "uses_sam2_or_rgb": False, "uses_absolute_position": False,
        "uses_relation_or_program": False, "uses_target": False,
        "proposal_config": dict(PROPOSAL_CONFIG), "source_area": SOURCE_AREA, "eps": EPS,
        "boundary_operator": {"function": "scipy.ndimage.binary_erosion",
                              "structure": "3x3 ones", "iterations": 1, "border_value": 0},
        "overlap_features": ["max_iou_other", "mean_iou_other", "max_self_contained_by_other",
                             "max_other_contained_in_self", "self_contained_count_norm",
                             "contains_other_count_norm"],
        "select_rule": ["highest score", "exact-float tie -> higher confidence",
                        "then lower original index"],
    }


__all__ = ["FEATURE_DIM", "FEATURE_NAMES", "PROPOSAL_CONFIG", "SOURCE_AREA",
           "SelectorDependencyUnavailable", "SetContextLargestSelector", "deterministic_max_area",
           "oracle_best", "proposal_feature_matrix", "select_with_scores", "selector_report"]
