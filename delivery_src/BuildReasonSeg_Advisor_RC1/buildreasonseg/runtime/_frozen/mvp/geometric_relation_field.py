"""Task 6N section 8 — `GeometricRelationField v0.1`.

A **parameter-free, differentiable, relation-conditioned geometric prior** built from an oracle
reference building mask. Given the reference mask ``M_ref`` (image resolution), a relation id in
``{left_of, right_of, above, below}`` and an output size ``(h, w)``, it returns ``P_rel`` in
``[0, 1]`` at ``(h, w)``.

Frozen convention (Task 3B / `configs/spatial_relations_v1.yaml`): ``relation(subject, object)``
means the subject satisfies the relation with respect to the object, so for a target ``T`` relative to
a reference ``R``:

* ``left_of(T, R)``  → ``cx_T < cx_R``
* ``right_of(T, R)`` → ``cx_T > cx_R``
* ``above(T, R)``    → ``cy_T < cy_R``   (image ``y`` increases downward)
* ``below(T, R)``    → ``cy_T > cy_R``

Definitions, exactly as specified:

```text
dx = x - cx_ref                 dy = y - cy_ref
ax = |dx|                       ay = |dy|
s_axis = 0.02   s_margin = 0.02   alpha = 1.2   tau = 0.04

horizontal:  axis_score = sigmoid((ax - alpha * ay) / s_axis)
             margin_score = sigmoid((ax - tau) / s_margin)
vertical:    axis_score = sigmoid((ay - alpha * ax) / s_axis)
             margin_score = sigmoid((ay - tau) / s_margin)

left_of: sign_score = sigmoid(-dx / s_margin)      right_of: sign_score = sigmoid( dx / s_margin)
above:   sign_score = sigmoid(-dy / s_margin)      below:    sign_score = sigmoid( dy / s_margin)

P_rel = sign_score * axis_score * margin_score      (clamped to [0, 1])
```

The softness is ``tau / 2 = 0.02``, i.e. exactly ``s_axis``/``s_margin``. There are **no learned
parameters** in field generation, and no distance transform, bounding box, candidate mask or extra
geometry channel is added in Task 6N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

RELATIONS: tuple[str, ...] = ("left_of", "right_of", "above", "below")
HORIZONTAL_RELATIONS = ("left_of", "right_of")
VERTICAL_RELATIONS = ("above", "below")

#: Task 6N sections 4/7: the only program ids in scope, mapped to their canonical direction.
PROGRAM_TO_RELATION: dict[str, str] = {
    "largest_to_left_of": "left_of",
    "largest_to_right_of": "right_of",
    "largest_to_above": "above",
    "largest_to_below": "below",
    "smallest_to_left_of": "left_of",
    "smallest_to_right_of": "right_of",
    "smallest_to_above": "above",
    "smallest_to_below": "below",
}
DIRECTIONAL_PROGRAMS: tuple[str, ...] = tuple(PROGRAM_TO_RELATION)
RELATION_TO_INDEX: dict[str, int] = {relation: index for index, relation in enumerate(RELATIONS)}


@dataclass(frozen=True)
class FieldConfig:
    """Frozen field constants (Task 6N section 8.2). No sweep is performed."""

    s_axis: float = 0.02
    s_margin: float = 0.02
    alpha: float = 1.2
    tau: float = 0.04

    @property
    def softness(self) -> float:
        """Softness is fixed at ``tau / 2`` (section 8.2)."""

        return self.tau / 2.0


DEFAULT_FIELD_CONFIG = FieldConfig()


def _as_mask_tensor(mask) -> torch.Tensor:
    """Accept numpy/torch masks in (H, W), (1, H, W) or (1, 1, H, W) form."""

    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().float()
    else:
        tensor = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
    while tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise ValueError(f"expected a 2-D mask, got shape {tuple(tensor.shape)}")
    return tensor.contiguous()


def reference_centroid(mask) -> tuple[float, float]:
    """Mask centroid ``(cx, cy)`` normalized to ``[0, 1]`` using pixel-centre coordinates.

    ``x = (col + 0.5) / W`` and ``y = (row + 0.5) / H`` so that the centroid and the field grid below
    use exactly the same coordinate convention.
    """

    tensor = _as_mask_tensor(mask)
    height, width = tensor.shape
    total = tensor.sum()
    if float(total) <= 0.0:
        raise ValueError("reference mask is empty; a centroid is undefined")
    columns = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    rows = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    cx = float((tensor.sum(dim=0) * columns).sum() / total)
    cy = float((tensor.sum(dim=1) * rows).sum() / total)
    return cx, cy


def resize_soft_mask(mask, size: tuple[int, int]) -> torch.Tensor:
    """Resize ``M_ref`` to ``(h, w)`` as a soft mask, clamped to ``[0, 1]``.

    Coverage-fraction (average) pooling is used: every output cell is the exact fraction of reference
    pixels it covers, which is the faithful soft downscale of a binary mask and is deterministic.
    """

    tensor = _as_mask_tensor(mask)
    height, width = int(size[0]), int(size[1])
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        tensor.unsqueeze(0).unsqueeze(0), (height, width)
    )
    return pooled.clamp(0.0, 1.0)


def normalized_grid(size: tuple[int, int], *, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centre normalized coordinate grids ``x`` (w,) and ``y`` (h,) in ``[0, 1]``."""

    height, width = int(size[0]), int(size[1])
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    return x, y


def geometric_relation_field(
    mask_ref,
    relation: str,
    size: tuple[int, int] | None = None,
    config: FieldConfig = DEFAULT_FIELD_CONFIG,
) -> torch.Tensor:
    """`P_rel` for one reference mask and one relation, at ``(h, w)``.

    Returns a ``(1, 1, h, w)`` float32 tensor in ``[0, 1]``.
    """

    if relation not in RELATION_TO_INDEX:
        raise ValueError(f"unsupported relation {relation!r}; expected one of {RELATIONS}")

    tensor = _as_mask_tensor(mask_ref)
    if size is None:
        size = (int(tensor.shape[0]), int(tensor.shape[1]))
    height, width = int(size[0]), int(size[1])

    cx_ref, cy_ref = reference_centroid(tensor)
    x, y = normalized_grid((height, width), device=tensor.device, dtype=torch.float32)
    dx = x.view(1, width) - cx_ref
    dy = y.view(height, 1) - cy_ref
    ax = dx.abs()
    ay = dy.abs()

    if relation in HORIZONTAL_RELATIONS:
        axis_score = torch.sigmoid((ax - config.alpha * ay) / config.s_axis)
        margin_score = torch.sigmoid((ax - config.tau) / config.s_margin)
    else:
        axis_score = torch.sigmoid((ay - config.alpha * ax) / config.s_axis)
        margin_score = torch.sigmoid((ay - config.tau) / config.s_margin)

    if relation == "left_of":
        sign_score = torch.sigmoid(-dx / config.s_margin)
    elif relation == "right_of":
        sign_score = torch.sigmoid(dx / config.s_margin)
    elif relation == "above":
        sign_score = torch.sigmoid(-dy / config.s_margin)
    else:  # below
        sign_score = torch.sigmoid(dy / config.s_margin)

    field = (sign_score * axis_score * margin_score).clamp(0.0, 1.0)
    return field.unsqueeze(0).unsqueeze(0)


def field_for_program(
    mask_ref,
    program_id: str,
    size: tuple[int, int] | None = None,
    config: FieldConfig = DEFAULT_FIELD_CONFIG,
) -> torch.Tensor:
    """`P_rel` for one of the 8 in-scope directional program ids."""

    if program_id not in PROGRAM_TO_RELATION:
        raise ValueError(f"program {program_id!r} is outside the Task 6N directional L2 scope")
    return geometric_relation_field(mask_ref, PROGRAM_TO_RELATION[program_id], size, config)


def instance_field_scores(
    field,
    instance_masks: list[np.ndarray],
    reference_index: int | None = None,
) -> dict:
    """Mean field score per native instance plus the target's rank among non-reference instances.

    ``instance_masks`` are full-resolution boolean masks (canonical native instances of one tile);
    ``field`` is the low-resolution ``P_rel``. Each instance score is the mean of the (bilinearly
    upsampled) field over that instance, which is the diagnostic Task 6N section 13 asks for.
    """

    tensor = field if isinstance(field, torch.Tensor) else torch.as_tensor(field)
    tensor = tensor.detach().float()
    while tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    height, width = instance_masks[0].shape if instance_masks else tensor.shape
    upsampled = torch.nn.functional.interpolate(
        tensor.view(1, 1, *tensor.shape), size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].numpy()

    scores = []
    for mask in instance_masks:
        selection = np.asarray(mask, dtype=bool)
        scores.append(float(upsampled[selection].mean()) if selection.any() else 0.0)

    distractors = [
        (index, score) for index, score in enumerate(scores) if index != reference_index
    ]
    distractors.sort(key=lambda item: -item[1])
    return {
        "scores": scores,
        "best_distractor_score": distractors[0][1] if distractors else None,
        "best_distractor_index": distractors[0][0] if distractors else None,
        "distractor_ranking": [index for index, _ in distractors],
    }


__all__ = [
    "DEFAULT_FIELD_CONFIG",
    "DIRECTIONAL_PROGRAMS",
    "FieldConfig",
    "HORIZONTAL_RELATIONS",
    "PROGRAM_TO_RELATION",
    "RELATIONS",
    "RELATION_TO_INDEX",
    "VERTICAL_RELATIONS",
    "field_for_program",
    "geometric_relation_field",
    "instance_field_scores",
    "normalized_grid",
    "reference_centroid",
    "resize_soft_mask",
]
