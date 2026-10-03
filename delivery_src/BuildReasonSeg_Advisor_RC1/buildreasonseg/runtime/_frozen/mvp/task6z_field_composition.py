"""Task 6Z — L3 direction × nearest field composition (oracle reference).

Exactly four canonical L3 programs (section 2) and no others:

```text
largest_to_left_of_to_nearest
largest_to_right_of_to_nearest
largest_to_above_to_nearest
largest_to_below_to_nearest
```

Canonical semantics (section 3, not regenerated): step 1 = largest eligible building is the oracle
reference; step 2 = filter all buildings by one frozen directional relation with respect to the reference;
step 3 = among the direction-filtered candidates pick the minimum canonical `boundary_distance` subject to
the frozen nearest eligibility/margin policy.

Field composition (section 6) is the **exact deterministic product**, with no renormalization, no learned
scalar, no temperature, no exponent and no threshold:

```text
P_prod_512 = clamp(P_dir_512 * P_near_512, 0, 1)
P_prod_64  = clamp(P_dir_64  * P_near_64,  0, 1)
```

`P_dir` comes from the frozen `geometric_relation_field_v02` (alpha 1.2, tau 0.04 — untouched) and `P_near`
from the frozen Task 6Y `NearestBoundaryField v0.1` (sigma_diag 0.05 — untouched). Nothing here recomputes
or rewrites either formula.
"""

from __future__ import annotations

import numpy as np
import torch

from buildreasonseg.runtime._frozen.mvp import nearest_boundary_field as near_field

#: Exactly the four canonical L3 programs (section 2). No fifth program exists.
L3_PROGRAMS: tuple[str, ...] = (
    "largest_to_left_of_to_nearest",
    "largest_to_right_of_to_nearest",
    "largest_to_above_to_nearest",
    "largest_to_below_to_nearest",
)
PROGRAM_TO_RELATION: dict[str, str] = {
    "largest_to_left_of_to_nearest": "left_of",
    "largest_to_right_of_to_nearest": "right_of",
    "largest_to_above_to_nearest": "above",
    "largest_to_below_to_nearest": "below",
}
#: The frozen direction vocabulary and its exact embedding ids (section 15).
DIRECTION_IDS: tuple[str, ...] = ("left_of", "right_of", "above", "below")
RELATION_TO_INDEX: dict[str, int] = {relation: index for index, relation in enumerate(DIRECTION_IDS)}
SOURCE_SIZE = 512
DECODER_FIELD_SIZE = 64


def relation_of(program_id: str) -> str:
    if program_id not in PROGRAM_TO_RELATION:
        raise ValueError(f"{program_id!r} is outside the four canonical L3 programs")
    return PROGRAM_TO_RELATION[program_id]


def direction_index(program_id: str) -> int:
    return RELATION_TO_INDEX[relation_of(program_id)]


def directional_field(reference_mask: np.ndarray, program_id: str,
                      size: int = SOURCE_SIZE) -> torch.Tensor:
    """`P_dir` from the frozen v0.2 implementation (formula not rewritten)."""

    from buildreasonseg.runtime._frozen.mvp.geometric_relation_field_v02 import geometric_relation_field_v02

    mask = torch.as_tensor(np.asarray(reference_mask, dtype=np.float32))[None, None]
    field = geometric_relation_field_v02(mask, relation_of(program_id), (int(size), int(size)))
    return field[0, 0].clamp(0.0, 1.0)


def nearest_field(reference_mask: np.ndarray, size: int = SOURCE_SIZE) -> torch.Tensor:
    """`P_near` from the frozen Task 6Y module (sigma_diag 0.05, formula not rewritten)."""

    mask = np.asarray(reference_mask, dtype=bool)
    if int(size) == SOURCE_SIZE:
        values = near_field.nearest_boundary_field_512(mask)
        return torch.as_tensor(values)
    field = near_field.nearest_boundary_field_512(mask)
    return near_field.field_to_decoder_resolution(field, size=int(size))


def product_field(directional: torch.Tensor, nearest: torch.Tensor) -> torch.Tensor:
    """Section 6: the exact clamped product, no renormalization of any kind."""

    if tuple(directional.shape) != tuple(nearest.shape):
        raise ValueError(f"field shape mismatch: {tuple(directional.shape)} vs {tuple(nearest.shape)}")
    return (directional * nearest).clamp(0.0, 1.0)


def record_fields(reference_mask: np.ndarray, program_id: str) -> dict:
    """All four field tensors one L3 record needs, at both resolutions."""

    directional_512 = directional_field(reference_mask, program_id, size=SOURCE_SIZE)
    nearest_512 = nearest_field(reference_mask, size=SOURCE_SIZE)
    directional_64 = near_field.field_to_decoder_resolution(directional_512.numpy(),
                                                            size=DECODER_FIELD_SIZE)
    nearest_64 = near_field.field_to_decoder_resolution(nearest_512.numpy(),
                                                        size=DECODER_FIELD_SIZE)
    return {
        "P_dir_512": directional_512,
        "P_near_512": nearest_512,
        "P_prod_512": product_field(directional_512, nearest_512),
        "P_dir_64": directional_64,
        "P_near_64": nearest_64,
        "P_prod_64": product_field(directional_64, nearest_64),
    }


def field_report() -> dict:
    return {
        "programs": list(L3_PROGRAMS),
        "program_count": len(L3_PROGRAMS),
        "program_to_relation": dict(PROGRAM_TO_RELATION),
        "direction_ids": list(DIRECTION_IDS),
        "canonical_operation_order": ["argmax_area (largest eligible reference)", "filter_relation",
                                      "argmin_boundary_distance"],
        "directional_field": {"module": "buildreasonseg.runtime._frozen.mvp/geometric_relation_field_v02.py",
                              "alpha": 1.2, "tau": 0.04, "modified": False},
        "nearest_field": {"module": "buildreasonseg.runtime._frozen.mvp/nearest_boundary_field.py",
                          "sigma_diag": near_field.SIGMA_DIAG, "modified": False},
        "product_field": {"source": "clamp(P_dir_512 * P_near_512, 0, 1)",
                          "decoder": "clamp(P_dir_64 * P_near_64, 0, 1)",
                          "renormalization": False, "learned_scalar": False, "temperature": False,
                          "exponent": False, "threshold": False},
        "source_size": SOURCE_SIZE, "decoder_field_size": DECODER_FIELD_SIZE,
        "labels_regenerated": False,
        "test_split_used": False,
    }


__all__ = [
    "DECODER_FIELD_SIZE",
    "DIRECTION_IDS",
    "L3_PROGRAMS",
    "PROGRAM_TO_RELATION",
    "RELATION_TO_INDEX",
    "SOURCE_SIZE",
    "direction_index",
    "directional_field",
    "field_report",
    "nearest_field",
    "product_field",
    "record_fields",
    "relation_of",
]
