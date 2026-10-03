"""Model layer: package contract + checkpoint verification."""

from __future__ import annotations

from .checkpoints import checkpoint_manifest, load_checkpoint_header, verify_checkpoint  # noqa: F401
from .package import (  # noqa: F401
    COMPONENTS,
    ModelPackage,
    ModelResolution,
    default_package_available,
    resolve_model,
)

__all__ = ["COMPONENTS", "ModelPackage", "ModelResolution", "checkpoint_manifest",
           "default_package_available", "load_checkpoint_header", "resolve_model",
           "verify_checkpoint"]
