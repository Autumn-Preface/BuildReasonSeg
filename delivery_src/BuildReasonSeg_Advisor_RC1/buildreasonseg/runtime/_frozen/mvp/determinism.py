"""Determinism helpers and trainable-state fingerprinting (Task 6C section 4.1/8)."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch


def trainable_state_fingerprint(model: torch.nn.Module) -> dict:
    """SHA256 over every trainable tensor, independent of name ordering.

    Used for Task 6C section 8: all four arms must start from the same initial
    trainable state. Named tensors are sorted by name so the digest is stable.
    """

    digest = hashlib.sha256()
    entries: list[dict] = []
    total = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().to("cpu")
        raw = tensor.contiguous().view(-1).to(torch.float32).numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(raw)
        entries.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
        total += tensor.numel()
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": len(entries),
        "numel": total,
        "tensors": entries,
    }


def gradients_fingerprint(parameters: Iterable[torch.nn.Parameter]) -> dict:
    """SHA256 over the current gradients of the given parameters."""

    digest = hashlib.sha256()
    count = 0
    norm_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to("cpu").contiguous().view(-1).to(torch.float32)
        digest.update(grad.numpy().tobytes())
        norm_sq += float(grad.double().pow(2).sum())
        count += 1
    return {
        "sha256": digest.hexdigest(),
        "tensors_with_grad": count,
        "total_grad_norm": norm_sq**0.5,
    }
