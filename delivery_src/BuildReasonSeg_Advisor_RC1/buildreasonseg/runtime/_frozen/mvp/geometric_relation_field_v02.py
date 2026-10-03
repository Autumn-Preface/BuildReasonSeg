"""Task 6P section 3 — `GeometricRelationField v0.2`.

A **gradient-preserving** re-implementation of the frozen `GeometricRelationField v0.1`
(`buildreasonseg.runtime._frozen.mvp/geometric_relation_field.py`).

Why v0.2 exists (Task 6P section 2, recorded as a correction, not a bug):

* v0.1 calls `.detach()` on tensor masks;
* v0.1 converts the centroid to a Python `float`.

Therefore v0.1 is numerically smooth as a spatial function but is **not differentiable with respect to
the input reference mask**. That did not invalidate Task 6N/6O, where the reference mask was oracle and
frozen and no gradient to the reference source was ever required. Predicted-reference training does
require that gradient, hence v0.2.

v0.2 is numerically equivalent to v0.1 for the same binary mask/relation/size and preserves autograd
from `P_rel` back to a soft `M_ref`:

```text
mass = sum(mask_ref) + eps          eps = 1e-6
cx   = sum(mask_ref * x_grid) / mass        x = (col + 0.5) / W
cy   = sum(mask_ref * y_grid) / mass        y = (row + 0.5) / H

alpha = 1.2   tau = 0.04   s_axis = 0.02   s_margin = 0.02
dx = x - cx   dy = y - cy   ax = |dx|   ay = |dy|

horizontal: axis = sigmoid((ax - alpha*ay)/s_axis)   margin = sigmoid((ax - tau)/s_margin)
vertical:   axis = sigmoid((ay - alpha*ax)/s_axis)   margin = sigmoid((ay - tau)/s_margin)
left_of: sign = sigmoid(-dx/s_margin)   right_of: sign = sigmoid(dx/s_margin)
above:   sign = sigmoid(-dy/s_margin)   below:    sign = sigmoid(dy/s_margin)

P_rel = clamp(sign * axis * margin, 0, 1)
```

No learned parameter, no `.detach()`, no Python-float centroid and no NumPy inside the forward path.
The reduction order deliberately mirrors v0.1 (per-column sums first, then the weighted sum) so the two
agree to float32 rounding.
"""

from __future__ import annotations

import torch

RELATIONS: tuple[str, ...] = ("left_of", "right_of", "above", "below")
RELATION_TO_INDEX: dict[str, int] = {relation: index for index, relation in enumerate(RELATIONS)}

#: Task 6N section 8.2 constants, reused unchanged.
S_AXIS = 0.02
S_MARGIN = 0.02
ALPHA = 1.2
TAU = 0.04
SOFTNESS = TAU / 2.0
EPS = 1e-6


def as_batch_mask(mask_ref) -> torch.Tensor:
    """Normalize `(H,W)`, `(B,H,W)` or `(B,1,H,W)` to a float32 `(B,1,H,W)` tensor.

    No `.detach()` and no NumPy: the tensor keeps its graph if it has one.
    """

    if not isinstance(mask_ref, torch.Tensor):
        raise TypeError("GeometricRelationField v0.2 requires a torch tensor mask (no NumPy in forward)")
    tensor = mask_ref
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.dim() != 4:
        raise ValueError(f"expected (H,W), (B,H,W) or (B,1,H,W), got {tuple(tensor.shape)}")
    if tensor.shape[1] != 1:
        raise ValueError(f"expected a single channel, got {tensor.shape[1]}")
    return tensor.to(torch.float32)


def normalized_grid(size: tuple[int, int], *, device=None, dtype=torch.float32):
    """Pixel-centre normalized coordinate vectors `x` (w,) and `y` (h,), exactly as in v0.1."""

    height, width = int(size[0]), int(size[1])
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    return x, y


def soft_centroid(mask_ref, eps: float = EPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable soft centroid ``(cx, cy)`` normalized to ``[0, 1]``, shape ``(B, 1, 1)``.

    Uses the mask's own resolution and the same reduction order as v0.1: per-column (and per-row)
    weighted sums first, then the final sum, so binary masks reproduce v0.1 to float32 rounding.
    """

    masks = as_batch_mask(mask_ref)
    _, _, height, width = masks.shape
    device, dtype = masks.device, torch.float32
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height

    mass = masks.sum() + eps
    # v0.1 computes sum over rows first, multiplies by the coordinate vector, then sums.
    column_weights = masks.sum(dim=-2) * x.view(1, 1, width)  # (B,1,W)
    row_weights = masks.sum(dim=-1) * y.view(1, 1, height)  # (B,1,H)
    cx = column_weights.sum().view(1, 1, 1) / mass
    cy = row_weights.sum().view(1, 1, 1) / mass
    return cx, cy


def relation_indices(relation, batch: int, device=None) -> torch.Tensor:
    """Accept one relation (str or int) for the whole batch, or one per batch element."""

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


def geometric_relation_field_v02(
    mask_ref,
    relation,
    size: tuple[int, int] | None = None,
    *,
    eps: float = EPS,
    alpha: float = ALPHA,
    tau: float = TAU,
    s_axis: float = S_AXIS,
    s_margin: float = S_MARGIN,
) -> torch.Tensor:
    """`P_rel` at ``(h, w)`` for a (possibly soft, possibly batched) reference mask.

    Returns ``(B, 1, h, w)`` in ``[0, 1]`` **with autograd intact** from the output to `mask_ref`.
    """

    masks = as_batch_mask(mask_ref)
    batch = masks.shape[0]
    if size is None:
        size = (int(masks.shape[-2]), int(masks.shape[-1]))
    height, width = int(size[0]), int(size[1])
    device = masks.device

    cx, cy = soft_centroid(masks, eps=eps)  # (1,1,1) each, differentiable
    x, y = normalized_grid((height, width), device=device, dtype=torch.float32)
    dx = (x.view(1, 1, 1, width) - cx.view(1, 1, 1)).expand(batch, 1, height, width)
    dy = (y.view(1, 1, height, 1) - cy.view(1, 1, 1)).expand(batch, 1, height, width)
    ax = dx.abs()
    ay = dy.abs()

    axis_horizontal = torch.sigmoid((ax - alpha * ay) / s_axis)
    margin_horizontal = torch.sigmoid((ax - tau) / s_margin)
    axis_vertical = torch.sigmoid((ay - alpha * ax) / s_axis)
    margin_vertical = torch.sigmoid((ay - tau) / s_margin)

    sign = torch.stack(
        [
            torch.sigmoid((-dx) / s_margin).squeeze(1),
            torch.sigmoid(dx / s_margin).squeeze(1),
            torch.sigmoid((-dy) / s_margin).squeeze(1),
            torch.sigmoid(dy / s_margin).squeeze(1),
        ],
        dim=1,
    )  # (B,4,h,w)

    indices = relation_indices(relation, batch, device=device)
    horizontal = (indices <= 1).view(batch, 1, 1, 1)
    axis_score = torch.where(horizontal, axis_horizontal, axis_vertical)
    margin_score = torch.where(horizontal, margin_horizontal, margin_vertical)
    sign_score = sign.gather(1, indices.view(batch, 1, 1, 1).expand(batch, 1, height, width))

    field = (sign_score * axis_score * margin_score).clamp(0.0, 1.0)
    return field


def field_for_program_v02(mask_ref, program_id: str | list[str], size=None, **kwargs) -> torch.Tensor:
    """`P_rel` for one of the 8 Task 6N directional program ids (single id or per batch element)."""

    from buildreasonseg.runtime._frozen.mvp.geometric_relation_field import PROGRAM_TO_RELATION

    if isinstance(program_id, str):
        if program_id not in PROGRAM_TO_RELATION:
            raise ValueError(f"program {program_id!r} is outside the directional L2 scope")
        relation = PROGRAM_TO_RELATION[program_id]
    else:
        relation = [PROGRAM_TO_RELATION[str(item)] for item in program_id]
    return geometric_relation_field_v02(mask_ref, relation, size, **kwargs)


__all__ = [
    "ALPHA",
    "EPS",
    "RELATIONS",
    "RELATION_TO_INDEX",
    "S_AXIS",
    "S_MARGIN",
    "SOFTNESS",
    "TAU",
    "as_batch_mask",
    "field_for_program_v02",
    "geometric_relation_field_v02",
    "normalized_grid",
    "relation_indices",
    "soft_centroid",
]
