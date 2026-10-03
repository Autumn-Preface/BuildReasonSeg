"""Task 6Y — Oracle-Reference `NearestBoundaryField v0.1`.

Semantics: pixels closer to the **grounded reference-building boundary** receive a larger prior value.
The field never chooses a building candidate itself; it only supplies a boundary-proximity prior to the
dense target decoder.

Exact source-resolution definition (section 5), for a binary oracle reference mask `M_ref` (512x512,
non-empty), using `scipy.ndimage.distance_transform_edt`:

```text
R          = M_ref > 0
D_px       = distance_transform_edt(~R)
diag_px    = sqrt(H^2 + W^2)
D_norm     = D_px / diag_px
P_near_512 = exp(-D_norm / sigma_diag)
P_near_512[R] = 0
P_near_512 = clamp(P_near_512, 0, 1)
```

`sigma_diag = 0.05`, `eps = 1e-6`, source size 512x512, decoder field size 64x64. No learned parameters,
no sweep, no reference centroid, no bbox distance, no target mask and no candidate proposals are used.
"""

from __future__ import annotations

import numpy as np

SIGMA_DIAG = 0.05
EPS = 1e-6
SOURCE_SIZE = 512
DECODER_FIELD_SIZE = 64


def scipy_available() -> bool:
    try:
        from scipy.ndimage import distance_transform_edt  # noqa: F401
    except Exception:  # pragma: no cover - exercised only when SciPy is missing
        return False
    return True


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """`scipy.ndimage.distance_transform_edt` on the inverse binary reference mask."""

    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~np.asarray(mask, dtype=bool))


def nearest_boundary_field_512(reference_mask: np.ndarray) -> np.ndarray:
    """The exact `P_near_512` of section 5 (float32, zero inside the reference)."""

    reference = np.asarray(reference_mask, dtype=bool)
    if reference.size == 0:
        raise ValueError("reference mask is empty")
    if not reference.any():
        raise ValueError("reference mask has no positive pixel; the field is undefined")
    height, width = reference.shape
    distance_px = distance_transform(reference)
    diagonal_px = float(np.sqrt(float(height) ** 2 + float(width) ** 2))
    distance_normalized = distance_px / diagonal_px
    field = np.exp(-distance_normalized / SIGMA_DIAG)
    field[reference] = 0.0
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def field_to_decoder_resolution(field512: np.ndarray,
                                size: int = DECODER_FIELD_SIZE) -> np.ndarray:
    """Section 6: bilinear interpolation of `P_near_512` to 64x64, clamped to [0, 1]."""

    import torch
    import torch.nn.functional as F

    tensor = torch.as_tensor(np.asarray(field512, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=(int(size), int(size)), mode="bilinear",
                            align_corners=False)
    return resized.clamp(0.0, 1.0)[0, 0]


def reference_area_to_decoder_resolution(reference_mask: np.ndarray,
                                         size: int = DECODER_FIELD_SIZE) -> np.ndarray:
    """Section 6 (Y-B1 only): area-resize the reference mask to 64x64, clamped to [0, 1]."""

    import torch
    import torch.nn.functional as F

    mask = torch.as_tensor(np.asarray(reference_mask, dtype=np.float32))[None, None]
    resized = F.interpolate(mask, size=(int(size), int(size)), mode="area")
    return resized.clamp(0.0, 1.0)[0, 0]


def candidate_field_score(field512: np.ndarray, candidate_mask: np.ndarray) -> float:
    """Section 7 candidate score: `max(P_near_512[p] for p in C)`."""

    values = np.asarray(field512, dtype=np.float32)[np.asarray(candidate_mask, dtype=bool)]
    if values.size == 0:
        return float("nan")
    return float(values.max())


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rank correlation without SciPy (ties get average ranks)."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2:
        return float("nan")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.sqrt((left_centered ** 2).sum() * (right_centered ** 2).sum()))
    if denominator <= 0:
        return float("nan")
    return float((left_centered * right_centered).sum() / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(sorted_values) + 1):
        if index == len(sorted_values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


__all__ = [
    "DECODER_FIELD_SIZE",
    "EPS",
    "SIGMA_DIAG",
    "SOURCE_SIZE",
    "candidate_field_score",
    "distance_transform",
    "field_to_decoder_resolution",
    "nearest_boundary_field_512",
    "reference_area_to_decoder_resolution",
    "scipy_available",
    "spearman_correlation",
]
