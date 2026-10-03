"""Inference layer: frozen request/response contract."""

from __future__ import annotations

from .contract import (  # noqa: F401
    DEFAULT_ALPHA,
    IMAGE_SUFFIXES,
    UNSUPPORTED_MODALITIES,
    InferenceRequest,
    InferenceResult,
    execution_boundary,
)

__all__ = ["DEFAULT_ALPHA", "IMAGE_SUFFIXES", "InferenceRequest", "InferenceResult",
           "UNSUPPORTED_MODALITIES", "execution_boundary"]
