"""Task 6W Part E — ProposalQualityEstimator v0.1.

Support infrastructure (not a claimed novelty): predicts whether a proposal matches a real building well
enough (`q_gt >= 0.50`) so that a deterministic largest/smallest rule can operate on trustworthy
candidates. GT is never a model input.

Per-proposal input (section 11):

* **8 geometry/confidence scalars** — `confidence`, `log_area`, `area_ratio`, `bbox_extent_ratio`,
  `width_ratio`, `height_ratio`, `fill_ratio`, `abs_log_aspect`. No family, no relation, no area rank, no
  image x/y coordinate.
* **frozen SAM2 appearance/context** — the frozen `V ∈ R^(256×64×64)` feature, the proposal mask
  nearest-downsampled to `M64`, and `Ring = clamp(max_pool2d(M64, 3, 1, 1) − M64, 0, 1)`:
  `inside_mean` = channel-wise masked mean over `M64` (256) and `ring_mean` over `Ring` (256, zeros when
  the ring is empty) → 512 visual dims.

Network (section 12): `Linear(512→64) → LayerNorm(64) → GELU` for the visual branch, `Linear(8→16) → GELU`
for the geometry branch, concatenated to 80, then `Linear(80→32) → GELU → Linear(32→1)` producing a raw
quality logit. No attention, no CNN, no Transformer/GNN.

Inference: `quality_prob = sigmoid(logit)`; keep iff `quality_prob >= 0.50` (frozen threshold). A proposal
whose `M64` has no positive cell is `feature_invalid_small`: excluded from training and forced to quality 0
at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TILE = 512
FEATURE_SIZE = 64
SAM2_CHANNELS = 256
GEOMETRY_DIM = 8
VISUAL_DIM = 2 * SAM2_CHANNELS
QUALITY_THRESHOLD = 0.50
GEOMETRY_FEATURE_NAMES: tuple[str, ...] = (
    "confidence", "log_area", "area_ratio", "bbox_extent_ratio", "width_ratio", "height_ratio",
    "fill_ratio", "abs_log_aspect",
)


@dataclass
class ProposalQualityFeatures:
    geometry: np.ndarray          # (8,)
    visual: np.ndarray            # (512,)
    mask64: np.ndarray            # (64, 64) bool
    ring: np.ndarray              # (64, 64) bool
    valid: bool                   # False -> feature_invalid_small
    ring_empty: bool


def geometry_features(proposal) -> np.ndarray:
    """The exact eight geometry/confidence scalars of section 11.1."""

    bbox = proposal.bbox_xyxy
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    area = float(proposal.area_px)
    bbox_area = float(proposal.bbox_area)
    return np.asarray([
        float(proposal.confidence),
        float(np.log1p(area) / np.log1p(TILE * TILE)),
        float(area / (TILE * TILE)),
        float(proposal.bbox_extent_ratio),
        float(width / TILE),
        float(height / TILE),
        float(area / bbox_area) if bbox_area > 0 else 0.0,
        float(abs(np.log((width + 1.0) / (height + 1.0)))),
    ], dtype=np.float32)


def downsample_mask64(mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbour downsample of the 512x512 proposal mask to 64x64."""

    tensor = torch.as_tensor(np.asarray(mask, dtype=np.float32))[None, None]
    small = F.interpolate(tensor, size=(FEATURE_SIZE, FEATURE_SIZE), mode="nearest")[0, 0]
    return (small.numpy() > 0.5)


def one_cell_ring(mask64: np.ndarray) -> np.ndarray:
    """D = max_pool2d(M64, 3, 1, 1); Ring = clamp(D - M64, 0, 1)."""

    tensor = torch.as_tensor(np.asarray(mask64, dtype=np.float32))[None, None]
    dilated = F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    ring = torch.clamp(dilated - tensor, 0.0, 1.0)[0, 0]
    return (ring.numpy() > 0.5)


def pooled_means(feature: np.ndarray, mask64: np.ndarray) -> np.ndarray:
    """Channel-wise masked mean of V over the given 64x64 mask -> 256 dims."""

    tensor = np.asarray(feature, dtype=np.float32)
    weights = np.asarray(mask64, dtype=np.float32)
    total = float(weights.sum())
    if total <= 0:
        return np.zeros((tensor.shape[0],), dtype=np.float32)
    return (tensor * weights[None]).sum(axis=(1, 2)) / total


def proposal_quality_features(proposal, feature: np.ndarray | None) -> ProposalQualityFeatures:
    """Assemble the (geometry, visual) pair for one proposal."""

    geometry = geometry_features(proposal)
    mask64 = downsample_mask64(proposal.mask)
    valid = bool(mask64.any())
    ring = one_cell_ring(mask64) if valid else np.zeros((FEATURE_SIZE, FEATURE_SIZE), dtype=bool)
    ring_empty = bool(not ring.any())
    if feature is None:
        visual = np.zeros((VISUAL_DIM,), dtype=np.float32)
    else:
        inside = pooled_means(feature, mask64) if valid else np.zeros((SAM2_CHANNELS,),
                                                                     dtype=np.float32)
        ring_mean = (np.zeros((SAM2_CHANNELS,), dtype=np.float32) if ring_empty
                     else pooled_means(feature, ring))
        visual = np.concatenate([inside, ring_mean]).astype(np.float32)
    return ProposalQualityFeatures(geometry=geometry, visual=visual, mask64=mask64, ring=ring,
                                   valid=valid, ring_empty=ring_empty)


class ProposalQualityEstimator(nn.Module):
    """Exact section 12 architecture (visual 512 -> 64, geometry 8 -> 16, concat 80 -> 32 -> 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.visual_branch = nn.Sequential(
            nn.Linear(VISUAL_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.geometry_branch = nn.Sequential(
            nn.Linear(GEOMETRY_DIM, 16),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(80, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, visual: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        """(B, 512) and (B, 8) -> (B,) raw quality logits."""

        combined = torch.cat([self.visual_branch(visual), self.geometry_branch(geometry)], dim=1)
        return self.head(combined).squeeze(-1)

    def architecture_report(self) -> dict:
        return {
            "visual_branch": ["Linear(512,64)", "LayerNorm(64)", "GELU"],
            "geometry_branch": ["Linear(8,16)", "GELU"],
            "concat_dim": 80,
            "head": ["Linear(80,32)", "GELU", "Linear(32,1)"],
            "attention": False, "cnn": False, "transformer": False, "gnn": False,
            "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "visual_dim": VISUAL_DIM, "geometry_dim": GEOMETRY_DIM,
            "total_parameters": int(sum(parameter.numel() for parameter in self.parameters())),
        }


def pos_weight_from_labels(labels: np.ndarray) -> float:
    """`N_negative / max(1, N_positive)` computed from the training split only."""

    labels = np.asarray(labels, dtype=np.float64)
    positive = float((labels >= 0.5).sum())
    negative = float((labels < 0.5).sum())
    return negative / max(1.0, positive)


@torch.no_grad()
def quality_probabilities(model: ProposalQualityEstimator, visual: np.ndarray,
                          geometry: np.ndarray, device: str = "cpu",
                          batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(geometry), batch_size):
        chunk_v = torch.as_tensor(visual[start: start + batch_size], device=device)
        chunk_g = torch.as_tensor(geometry[start: start + batch_size], device=device)
        logits = model(chunk_v, chunk_g)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.zeros((0,), dtype=np.float32)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (no sklearn dependency)."""

    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels >= 0.5
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks for ties
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve) via the step rule."""

    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels >= 0.5
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = positive[order]
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_pos)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray,
                   threshold: float = QUALITY_THRESHOLD) -> dict:
    labels = np.asarray(labels) >= 0.5
    predicted = np.asarray(probabilities) >= threshold
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1, len(labels))
    return {
        "threshold": threshold, "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


__all__ = [
    "FEATURE_SIZE",
    "GEOMETRY_DIM",
    "GEOMETRY_FEATURE_NAMES",
    "ProposalQualityEstimator",
    "ProposalQualityFeatures",
    "QUALITY_THRESHOLD",
    "SAM2_CHANNELS",
    "TILE",
    "VISUAL_DIM",
    "auprc",
    "auroc",
    "binary_metrics",
    "downsample_mask64",
    "geometry_features",
    "one_cell_ring",
    "pooled_means",
    "pos_weight_from_labels",
    "proposal_quality_features",
    "quality_probabilities",
]
