"""Training layer: frozen CLI defaults and training principles."""

from __future__ import annotations

from .contract import (  # noqa: F401
    DEFAULTS,
    INITS,
    PROGRAM_HEAD_TRAINED_WITH_NEW_DATASET,
    SAM2_FROZEN,
    STAGES,
    TEST_CONSUMED_AUTOMATICALLY,
    TrainingRequest,
)

__all__ = ["DEFAULTS", "INITS", "PROGRAM_HEAD_TRAINED_WITH_NEW_DATASET", "SAM2_FROZEN", "STAGES",
           "TEST_CONSUMED_AUTOMATICALLY", "TrainingRequest"]
