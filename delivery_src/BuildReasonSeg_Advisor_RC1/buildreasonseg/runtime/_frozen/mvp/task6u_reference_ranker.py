"""Task 6U Part G — ProposalSetRanker v0.1.

A small **support-infrastructure** ranker over the eligible proposal set of one reference record. It
addresses only `REFERENCE_SELECTION_WRONG`; it cannot create missing proposals and is not a claimed
project novelty.

Exact per-proposal feature vector (section 14) — 12 scalars plus a 2-d family one-hot = **14 dims**:

```text
 1 log_area            = log1p(area_px) / log1p(512*512)
 2 area_ratio          = area_px / (512*512)
 3 area_over_set_max   = area_px / max(area_px in set)
 4 set_min_over_area   = min(area_px in set) / area_px
 5 area_rank_desc      = rank by descending area / max(1, N-1)   (largest rank = 0)
 6 area_rank_asc       = rank by ascending area  / max(1, N-1)   (smallest rank = 0)
 7 confidence
 8 bbox_extent_ratio
 9 width_ratio         = bbox_width / 512
10 height_ratio        = bbox_height / 512
11 fill_ratio          = area_px / max(1, bbox_area)
12 proposal_count_norm = min(N, 300) / 300
13 family one-hot[0]   (largest = 1, smallest = 0)
14 family one-hot[1]   (smallest = 1, largest = 0)
```

Deliberately **not** used: any GT-derived feature, the target relation, centroid x/y, image location,
SAM2 features, the target mask or the source feature id.

Architecture (section 15): shared per-proposal MLP `Linear(14→32) → GELU → Linear(32→16) → GELU →
Linear(16→1)`, softmax over the eligible proposal scores, cross-entropy against the labelled positive.
No attention, Transformer or GNN.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

TILE = 512
FEATURE_DIM = 14
FEATURE_NAMES: tuple[str, ...] = (
    "log_area", "area_ratio", "area_over_set_max", "set_min_over_area", "area_rank_desc",
    "area_rank_asc", "confidence", "bbox_extent_ratio", "width_ratio", "height_ratio", "fill_ratio",
    "proposal_count_norm", "family_largest", "family_smallest",
)
MAX_CANDIDATES = 300
FAMILY_ONE_HOT = {"largest": (1.0, 0.0), "smallest": (0.0, 1.0)}


def proposal_features(proposals, family: str) -> np.ndarray:
    """(N, 14) feature matrix for one eligible proposal set (order preserved)."""

    if family not in FAMILY_ONE_HOT:
        raise ValueError(f"unsupported reference family {family!r}")
    count = len(proposals)
    if count == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    areas = np.asarray([float(proposal.area_px) for proposal in proposals], dtype=np.float64)
    max_area = float(areas.max()) if count else 0.0
    min_area = float(areas.min()) if count else 0.0
    desc_order = np.argsort(-areas, kind="stable")
    asc_order = np.argsort(areas, kind="stable")
    rank_desc = np.empty(count, dtype=np.float64)
    rank_asc = np.empty(count, dtype=np.float64)
    for rank, position in enumerate(desc_order):
        rank_desc[position] = rank
    for rank, position in enumerate(asc_order):
        rank_asc[position] = rank
    denominator = max(1, count - 1)
    one_hot = FAMILY_ONE_HOT[family]

    rows = []
    for index, proposal in enumerate(proposals):
        bbox = proposal.bbox_xyxy
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])
        bbox_area = float(proposal.bbox_area)
        rows.append([
            float(np.log1p(areas[index]) / np.log1p(TILE * TILE)),
            float(areas[index] / (TILE * TILE)),
            float(areas[index] / max_area) if max_area > 0 else 0.0,
            float(min_area / areas[index]) if areas[index] > 0 else 0.0,
            float(rank_desc[index] / denominator),
            float(rank_asc[index] / denominator),
            float(proposal.confidence),
            float(proposal.bbox_extent_ratio),
            float(width / TILE),
            float(height / TILE),
            float(areas[index] / bbox_area) if bbox_area > 0 else 0.0,
            float(min(count, MAX_CANDIDATES) / MAX_CANDIDATES),
            float(one_hot[0]),
            float(one_hot[1]),
        ])
    return np.asarray(rows, dtype=np.float32)


class ProposalSetRanker(nn.Module):
    """Shared per-proposal MLP scorer (14 -> 32 -> 16 -> 1). No attention/Transformer/GNN."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_1: int = 32, hidden_2: int = 16):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_1),
            nn.GELU(),
            nn.Linear(hidden_1, hidden_2),
            nn.GELU(),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, N, 14) -> (B, N) scores."""

        return self.network(features).squeeze(-1)

    def architecture_report(self) -> dict:
        return {
            "input_dim": self.feature_dim,
            "layers": ["Linear(14,32)", "GELU", "Linear(32,16)", "GELU", "Linear(16,1)"],
            "attention": False, "transformer": False, "gnn": False,
            "total_parameters": int(sum(parameter.numel() for parameter in self.parameters())),
            "feature_names": list(FEATURE_NAMES),
        }


def score_sets(model: ProposalSetRanker, feature_rows: list[np.ndarray], device: str = "cpu",
               requires_grad: bool = False):
    """Padded/masked scoring of a batch of candidate sets -> ((B,N) logits tensor, (B,N) mask tensor)."""

    if not feature_rows:
        empty = torch.zeros((0, 1), device=device)
        return empty, torch.zeros((0, 1), dtype=torch.bool, device=device)
    max_len = max(1, max(len(rows) for rows in feature_rows))
    batch = np.zeros((len(feature_rows), max_len, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((len(feature_rows), max_len), dtype=bool)
    for index, rows in enumerate(feature_rows):
        length = len(rows)
        if length:
            batch[index, :length] = rows
            mask[index, :length] = True
    features = torch.as_tensor(batch, device=device)
    mask_tensor = torch.as_tensor(mask, device=device)
    if requires_grad:
        logits = model(features)
    else:
        with torch.no_grad():
            logits = model(features)
    return logits, mask_tensor


def set_logits(model: ProposalSetRanker, feature_rows: list[np.ndarray], device: str = "cpu"):
    """Inference helper: (logits list, mask list) as numpy arrays."""

    logits, mask = score_sets(model, feature_rows, device=device, requires_grad=False)
    return logits.cpu().numpy(), mask.cpu().numpy()


def select_with_ranker(model: ProposalSetRanker, proposals, family: str, device: str = "cpu") -> dict:
    """Argmax of the ranker score over the eligible proposal set (inference path; no GT)."""

    if not proposals:
        return {"selected_index": None, "scores": [], "features": np.zeros((0, FEATURE_DIM))}
    features = proposal_features(proposals, family)
    logits, mask = set_logits(model, [features], device=device)
    scores = logits[0][mask[0]]
    best = int(np.argmax(scores))
    return {
        "selected_index": best,
        "selected_proposal_index": int(proposals[best].index),
        "scores": [float(value) for value in scores],
        "confidence_softmax": [float(value) for value in
                               np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()],
        "features": features,
    }


__all__ = [
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FAMILY_ONE_HOT",
    "MAX_CANDIDATES",
    "ProposalSetRanker",
    "proposal_features",
    "select_with_ranker",
    "set_logits",
]
