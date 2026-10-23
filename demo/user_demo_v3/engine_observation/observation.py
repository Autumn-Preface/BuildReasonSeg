"""V2-only observation transport. No scientific computation or model call lives here."""
from __future__ import annotations

import contextlib
import contextvars
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

_ACTIVE = contextvars.ContextVar("buildreasonseg_v2_observer", default=None)
_ERRORS = deque(maxlen=64)


def freeze(value):
    if isinstance(value, np.ndarray):
        # Backed by immutable bytes, rather than a reversible writeable=False flag.
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        detached = value.detach()
        # NumPy does not represent bfloat16; float32 preserves each bfloat16 value for display.
        if str(detached.dtype) == "torch.bfloat16":
            detached = detached.float()
        return freeze(detached.cpu().numpy())
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def observation_errors(observer):
    return [dict(record) for identity, record in _ERRORS if identity == id(observer)]


def emit(observer, stage, state, metadata=None, arrays=None):
    if observer is None:
        return
    try:
        packet = freeze({"stage_id": stage, "state": state, "metadata": metadata or {}, "arrays": arrays or {}})
        observer(packet)
    except Exception as error:
        # Observation failure never changes the mathematical result or its scientific error code.
        _ERRORS.append((id(observer), {"stage_id": stage, "error_type": type(error).__name__, "failure_scope": "OBSERVATION_ONLY"}))
        reporter = getattr(observer, "observation_failed", None)
        if callable(reporter):
            try:
                reporter(error, stage)
            except Exception:
                pass                   # the bounded error registry still retains this failure


def active_observer():
    return _ACTIVE.get()


def emit_current(stage, state, metadata=None, arrays=None):
    emit(_ACTIVE.get(), stage, state, metadata, arrays)


@contextlib.contextmanager
def observation_scope(observer):
    token = _ACTIVE.set(observer)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
