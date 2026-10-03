"""CLI package: shared parsing/error conventions for the RC1 entry points."""

from __future__ import annotations

from .common import (  # noqa: F401
    DATASET_FORMAT_CHOICES,
    DEVICE_CHOICES,
    NOT_IMPLEMENTED_STAGE,
    REFERENCE_MODE_CHOICES,
    SPLIT_CHOICES,
    TRAIN_INIT_CHOICES,
    TRAIN_STAGE_CHOICES,
    base_parser,
    not_implemented,
    print_boundary,
    resolve_config,
    run_cli,
)

__all__ = ["base_parser", "not_implemented", "print_boundary", "resolve_config", "run_cli"]
