"""Task 6R Part C — Directional Geometry-Relation Consistency Loss (GRCL v0.1).

A **mask-level reference–target geometric relation constraint** whose variables are the predicted
target-mask centroid, the grounded reference-mask centroid and the instruction-specified spatial
relation. Fixed constants (no sweep): `alpha = 1.2`, `tau = 0.04`, `eps = 1e-6`,
`lambda_grcl = 0.5`; `alpha`/`tau` match the frozen Task 3B relation semantics.

```text
P_t    = sigmoid(Z_t)
mass_t = sum(P_t) + eps                      cx_t = sum(P_t * x) / mass_t     cy_t = sum(P_t * y) / mass_t
mass_r = sum(M_ref) + eps                    cx_r = sum(M_ref * x) / mass_r   cy_r = sum(M_ref * y) / mass_r

left_of:  signed = cx_r - cx_t   orth = |cy_t - cy_r|
right_of: signed = cx_t - cx_r   orth = |cy_t - cy_r|
above:    signed = cy_r - cy_t   orth = |cx_t - cx_r|
below:    signed = cy_t - cy_r   orth = |cx_t - cx_r|

L_margin = relu(tau - signed)
L_axis   = relu(alpha * orth - signed)
L_GRCL   = mean(L_margin + L_axis)

L_total(R1) = L_BCE + L_Dice + 0.5 * L_GRCL
```

The target logits are **never** detached and target probabilities are **never** thresholded inside the
loss; the oracle reference is treated as fixed. Exactly these terms are used — no `tau` normalisation,
no softplus, no additional sign/field/contrastive/pair loss.

Section 11's hard relation-correctness metric is also defined here (evaluation only, threshold 0.5):

```text
relation correct  iff  signed >= tau   AND   signed >= alpha * orth
```
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

RELATIONS: tuple[str, ...] = ("left_of", "right_of", "above", "below")
RELATION_TO_INDEX: dict[str, int] = {relation: index for index, relation in enumerate(RELATIONS)}

#: Frozen Task 6R section 5 constants.
ALPHA = 1.2
TAU = 0.04
EPS = 1e-6
LAMBDA_GRCL = 0.5


def normalized_grid(size: tuple[int, int], *, device=None, dtype=torch.float32):
    """Pixel-centre normalized coordinate vectors `x` (w,) and `y` (h,) in ``[0, 1]``."""

    height, width = int(size[0]), int(size[1])
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    return x, y


def _as_batch_mask(mask, like: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("GRCL requires torch tensors")
    tensor = mask
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    return tensor.to(device=like.device, dtype=torch.float32)


def relation_indices(relation, batch: int, device=None) -> torch.Tensor:
    if isinstance(relation, str):
        if relation not in RELATION_TO_INDEX:
            raise ValueError(f"unsupported relation {relation!r}; expected one of {RELATIONS}")
        return torch.full((batch,), RELATION_TO_INDEX[relation], dtype=torch.long, device=device)
    if isinstance(relation, int):
        return torch.full((batch,), int(relation), dtype=torch.long, device=device)
    if isinstance(relation, torch.Tensor):
        indices = relation.to(device=device, dtype=torch.long).view(-1)
    else:
        indices = torch.as_tensor([RELATION_TO_INDEX[str(item)] for item in relation],
                                  dtype=torch.long, device=device)
    if indices.numel() == 1 and batch > 1:
        indices = indices.expand(batch)
    if indices.numel() != batch:
        raise ValueError(f"expected {batch} relation ids, got {indices.numel()}")
    return indices


def soft_centroids(logits: torch.Tensor, mask_ref: torch.Tensor) -> dict:
    """Differentiable soft centroids of the predicted target and of the reference mask."""

    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    probabilities = torch.sigmoid(logits)
    masks = _as_batch_mask(mask_ref, logits)
    if masks.shape[0] == 1 and probabilities.shape[0] > 1:
        masks = masks.expand(probabilities.shape[0], -1, -1, -1)
    if masks.shape[-2:] != probabilities.shape[-2:]:
        raise ValueError(
            f"reference mask {tuple(masks.shape[-2:])} must match the target logits "
            f"{tuple(probabilities.shape[-2:])}"
        )
    height, width = probabilities.shape[-2:]
    x, y = normalized_grid((height, width), device=logits.device, dtype=torch.float32)

    mass_t = probabilities.sum(dim=(1, 2, 3)) + EPS  # (B,)
    cx_t = (probabilities.sum(dim=2) * x.view(1, 1, width)).sum(dim=(1, 2)) / mass_t
    cy_t = (probabilities.sum(dim=3) * y.view(1, 1, height)).sum(dim=(1, 2)) / mass_t

    mass_r = masks.sum(dim=(1, 2, 3)) + EPS  # (B,)
    cx_r = (masks.sum(dim=2) * x.view(1, 1, width)).sum(dim=(1, 2)) / mass_r
    cy_r = (masks.sum(dim=3) * y.view(1, 1, height)).sum(dim=(1, 2)) / mass_r

    return {
        "cx_t": cx_t, "cy_t": cy_t, "cx_r": cx_r, "cy_r": cy_r,
        "probabilities": probabilities, "reference": masks,
    }


def signed_orthogonal(cx_t, cy_t, cx_r, cy_r, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Section 7 signed primary displacement and orthogonal displacement, per relation."""

    left = indices == 0
    right = indices == 1
    above = indices == 2
    below = indices == 3

    signed = torch.zeros_like(cx_t)
    orth = torch.zeros_like(cx_t)
    signed = torch.where(left, cx_r - cx_t, signed)
    orth = torch.where(left, (cy_t - cy_r).abs(), orth)
    signed = torch.where(right, cx_t - cx_r, signed)
    orth = torch.where(right, (cy_t - cy_r).abs(), orth)
    signed = torch.where(above, cy_r - cy_t, signed)
    orth = torch.where(above, (cx_t - cx_r).abs(), orth)
    signed = torch.where(below, cy_t - cy_r, signed)
    orth = torch.where(below, (cx_t - cx_r).abs(), orth)
    return signed, orth


@dataclass
class GRCLReport:
    loss: torch.Tensor
    margin: torch.Tensor
    axis: torch.Tensor
    signed: torch.Tensor
    orth: torch.Tensor

    def as_dict(self) -> dict:
        return {
            "loss": float(self.loss.detach().mean()),
            "margin": float(self.margin.detach().mean()),
            "axis": float(self.axis.detach().mean()),
            "mean_signed": float(self.signed.detach().mean()),
            "mean_orth": float(self.orth.detach().mean()),
        }


def grcl_directional(
    logits: torch.Tensor,
    mask_ref: torch.Tensor,
    relation,
    *,
    alpha: float = ALPHA,
    tau: float = TAU,
) -> GRCLReport:
    """`L_GRCL` (batch mean). The target logits keep their graph; no thresholding happens here."""

    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    centroids = soft_centroids(logits, mask_ref)
    indices = relation_indices(relation, logits.shape[0], device=logits.device)
    signed, orth = signed_orthogonal(
        centroids["cx_t"], centroids["cy_t"], centroids["cx_r"], centroids["cy_r"], indices
    )
    margin = torch.relu(tau - signed)
    axis = torch.relu(alpha * orth - signed)
    loss = (margin + axis).mean()
    return GRCLReport(loss=loss, margin=margin, axis=axis, signed=signed, orth=orth)


def total_loss(logits: torch.Tensor, target: torch.Tensor, mask_ref: torch.Tensor, relation,
               *, lambda_grcl: float = LAMBDA_GRCL, alpha: float = ALPHA, tau: float = TAU) -> dict:
    """`L_BCE + L_Dice + lambda * L_GRCL` — the R1/R2 training objective (section 8)."""

    from buildreasonseg.runtime._frozen.mvp.losses import mask_bce_with_logits, mask_soft_dice

    bce = mask_bce_with_logits(logits, target)
    dice = mask_soft_dice(logits, target)
    grcl = grcl_directional(logits, mask_ref, relation, alpha=alpha, tau=tau)
    return {
        "loss": bce + dice + lambda_grcl * grcl.loss,
        "bce": bce.detach(),
        "dice": dice.detach(),
        "grcl": grcl.loss.detach(),
        "grcl_detail": grcl,
    }


# --------------------------------------------------------------------------- evaluation-side metrics


@torch.no_grad()
def hard_relation_metrics(
    probabilities: torch.Tensor,
    mask_ref: torch.Tensor,
    relation,
    *,
    alpha: float = ALPHA,
    tau: float = TAU,
    threshold: float = 0.5,
) -> dict:
    """Section 11 hard relation-correctness metric — **evaluation only** (threshold 0.5)."""

    if probabilities.dim() == 3:
        probabilities = probabilities.unsqueeze(1)
    masks = _as_batch_mask(mask_ref, probabilities)
    if masks.shape[0] == 1 and probabilities.shape[0] > 1:
        masks = masks.expand(probabilities.shape[0], -1, -1, -1)
    height, width = probabilities.shape[-2:]
    x, y = normalized_grid((height, width), device=probabilities.device, dtype=torch.float32)

    binary = probabilities > threshold
    features = binary.float()
    mass_t = features.sum(dim=(1, 2, 3)) + EPS
    cx_t = (features.sum(dim=2) * x.view(1, 1, width)).sum(dim=(1, 2)) / mass_t
    cy_t = (features.sum(dim=3) * y.view(1, 1, height)).sum(dim=(1, 2)) / mass_t

    mass_r = masks.sum(dim=(1, 2, 3)) + EPS
    cx_r = (masks.sum(dim=2) * x.view(1, 1, width)).sum(dim=(1, 2)) / mass_r
    cy_r = (masks.sum(dim=3) * y.view(1, 1, height)).sum(dim=(1, 2)) / mass_r

    indices = relation_indices(relation, probabilities.shape[0], device=probabilities.device)
    signed, orth = signed_orthogonal(cx_t, cy_t, cx_r, cy_r, indices)
    empty = ~binary.any(dim=(1, 2, 3))
    correct = (signed >= tau) & (signed >= alpha * orth) & (~empty)
    violation = (signed < alpha * orth) & (~empty)
    return {
        "correct": correct,
        "empty": empty,
        "signed": signed,
        "orth": orth,
        "margin": signed - tau,
        "axis_violation": violation,
        "threshold": threshold,
    }


def summarise_relation_metrics(rows: list[dict]) -> dict:
    """Aggregate the per-sample hard-relation rows produced by `hard_relation_metrics`."""

    if not rows:
        return {"records": 0}
    correct = [row["correct"] for row in rows]
    margins = [row["margin"] for row in rows if not row["empty"]]
    violations = [row["axis_violation"] for row in rows if not row["empty"]]
    per_direction = {}
    for relation in RELATIONS:
        subset = [row for row in rows if row["relation"] == relation]
        per_direction[relation] = {
            "records": len(subset),
            "relation_accuracy": (sum(1 for row in subset if row["correct"]) / len(subset))
            if subset else None,
            "mean_signed_margin": (sum(row["margin"] for row in subset if not row["empty"])
                                   / max(sum(1 for row in subset if not row["empty"]), 1))
            if subset else None,
        }
    return {
        "records": len(rows),
        "relation_accuracy": sum(1 for value in correct if value) / len(correct),
        "empty_predictions": sum(1 for row in rows if row["empty"]),
        "mean_positive_signed_margin": (sum(margins) / len(margins)) if margins else None,
        "axis_violation_rate": (sum(1 for value in violations if value) / len(violations))
        if violations else None,
        "per_relation": per_direction,
    }


__all__ = [
    "ALPHA",
    "EPS",
    "GRCLReport",
    "LAMBDA_GRCL",
    "RELATIONS",
    "RELATION_TO_INDEX",
    "TAU",
    "grcl_directional",
    "hard_relation_metrics",
    "normalized_grid",
    "relation_indices",
    "signed_orthogonal",
    "soft_centroids",
    "summarise_relation_metrics",
    "total_loss",
]
