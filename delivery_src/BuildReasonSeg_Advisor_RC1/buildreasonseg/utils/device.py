"""Device selection for `--device {auto,cpu,cuda}`."""

from __future__ import annotations


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def gpu_name() -> str | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return str(torch.cuda.get_device_name(0))
    except Exception:
        return None


def resolve_device(requested: str = "auto") -> str:
    """`auto` prefers CUDA when available and otherwise falls back to CPU."""

    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unknown device request: {requested!r}")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if cuda_available() else "cpu"
    return "cuda" if cuda_available() else "cpu"


__all__ = ["cuda_available", "gpu_name", "resolve_device", "torch_available"]
