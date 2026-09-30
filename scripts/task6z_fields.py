"""Task 6Z field helpers shared by the sanity and training scripts (thin wrappers, no new formulas)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp.task6z_field_composition import (  # noqa: E402
    DECODER_FIELD_SIZE,
    directional_field,
    nearest_field,
    product_field,
    record_fields,
)


def compute_product_field(reference_mask: np.ndarray, program_id: str) -> np.ndarray:
    """`P_prod_512` as a numpy array (exact product of the frozen fields)."""

    fields = record_fields(reference_mask, program_id)
    return fields["P_prod_512"].numpy()


def field_bundle(reference_mask: np.ndarray, program_id: str) -> dict:
    """The 64x64 field tensors one training sample needs."""

    fields = record_fields(reference_mask, program_id)
    return {"P_dir_64": fields["P_dir_64"], "P_near_64": fields["P_near_64"],
            "P_prod_64": fields["P_prod_64"], "size": DECODER_FIELD_SIZE}


__all__ = ["compute_product_field", "field_bundle"]
