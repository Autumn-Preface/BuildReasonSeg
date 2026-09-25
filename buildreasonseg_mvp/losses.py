"""Loss functions for the Task 6A smoke configuration.

    L_total = 1.0 * L_lm_ce + 2.0 * L_mask_bce + 0.5 * L_mask_dice

This weighting is a **provisional smoke/overfit configuration**, not a paper
hyperparameter (Task 6A section 12). There is deliberately no Spatial
Consistency Loss and no reference-mask loss in Task 6A.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lm_ce: float = 1.0
    mask_bce: float = 2.0
    mask_dice: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {"lm_ce": self.lm_ce, "mask_bce": self.mask_bce, "mask_dice": self.mask_dice}


def lm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on the assistant target tokens only.

    ``labels`` must already be ``-100`` outside the assistant target, so the
    user/image/prompt tokens contribute nothing.
    """

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
    )


def mask_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float())


def mask_soft_dice(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Soft Dice on probabilities; 1 - Dice so it is minimised."""

    probabilities = torch.sigmoid(logits.float())
    target = targets.float()
    intersection = (probabilities * target).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


@dataclass
class LossBreakdown:
    total: torch.Tensor
    lm_ce: torch.Tensor
    mask_bce: torch.Tensor
    mask_dice: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "lm_ce": float(self.lm_ce.detach()),
            "mask_bce": float(self.mask_bce.detach()),
            "mask_dice": float(self.mask_dice.detach()),
        }


def combined_loss(
    lm_logits: torch.Tensor,
    lm_labels: torch.Tensor,
    mask_logits: torch.Tensor,
    mask_targets: torch.Tensor,
    weights: LossWeights | None = None,
) -> LossBreakdown:
    """Compute the three components separately and sum them with the weights."""

    weights = weights or LossWeights()
    lm_ce = lm_cross_entropy(lm_logits, lm_labels)
    mask_bce = mask_bce_with_logits(mask_logits, mask_targets)
    mask_dice = mask_soft_dice(mask_logits, mask_targets)
    total = weights.lm_ce * lm_ce + weights.mask_bce * mask_bce + weights.mask_dice * mask_dice
    return LossBreakdown(total=total, lm_ce=lm_ce, mask_bce=mask_bce, mask_dice=mask_dice)
