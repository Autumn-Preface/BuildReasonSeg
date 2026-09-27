"""Adapter-only checkpointing for the MVP.

Only the state needed to reproduce the MVP is saved:

* the LoRA adapter weights (and the trainable `[SEG]` token state, which PEFT
  keeps inside the same state dict when `trainable_token_indices` is used);
* the project token-row parameter, when the project fallback mechanism is active;
* the projection MLP;
* the SAM2 mask decoder;
* optionally the optimizer state;
* the resolved config, the step, the metrics and the RNG state.

Base Qwen and base SAM2 weights are **never** duplicated.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_state(model) -> dict:
    """LoRA + trainable-token entries from the PEFT-wrapped language model."""

    state = {}
    for key, value in model.qwen.state_dict().items():
        if "lora_" in key or "trainable_tokens" in key or "token_row" in key:
            state[key] = value.detach().cpu()
    return state


@dataclass
class CheckpointContents:
    step: int
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def save_checkpoint(
    path: Path,
    model,
    step: int,
    metrics: dict | None = None,
    config: dict | None = None,
    optimizer=None,
    include_optimizer: bool = True,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "format": "buildreasonseg-mvp-adapter-v1",
        "step": int(step),
        "metrics": metrics or {},
        "config": config or {},
        "lora_and_token_state": _adapter_state(model),
        "projection": {k: v.detach().cpu() for k, v in model.projection.state_dict().items()},
        "sam_mask_decoder": {
            k: v.detach().cpu() for k, v in model.sam.sam_mask_decoder.state_dict().items()
        },
        # Task 6D: the Spatial Grounding Bridge head, when the model has one. It is absent
        # for the Task 6A-6C arms, so their checkpoints stay compatible with this change.
        "grounding_head": (
            {k: v.detach().cpu() for k, v in model.grounding_head.state_dict().items()}
            if getattr(model, "grounding_head", None) is not None
            else None
        ),
        "geometry_kind": getattr(model, "geometry_kind", None),
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if model.token_holder is not None:
        payload["project_token_row"] = model.token_holder.row.detach().cpu()
    if include_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "step": int(step),
    }


def load_checkpoint(path: Path, model, optimizer=None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)

    missing, unexpected = model.qwen.load_state_dict(payload["lora_and_token_state"], strict=False)
    model.projection.load_state_dict(payload["projection"])
    model.sam.sam_mask_decoder.load_state_dict(payload["sam_mask_decoder"])
    head_loaded = False
    if getattr(model, "grounding_head", None) is not None and payload.get("grounding_head"):
        model.grounding_head.load_state_dict(payload["grounding_head"])
        head_loaded = True
    if model.token_holder is not None and "project_token_row" in payload:
        with torch.no_grad():
            model.token_holder.row.copy_(payload["project_token_row"].to(model.token_holder.row.device))
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    return {
        "step": payload.get("step", 0),
        "metrics": payload.get("metrics", {}),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "grounding_head_loaded": head_loaded,
        "geometry_kind": payload.get("geometry_kind"),
    }


def build_manifest(entries: dict[str, dict], base_models: dict, extra: dict | None = None) -> dict:
    """The committed checkpoint manifest: paths, sizes, hashes -- never the bytes."""

    return {
        "_doc": (
            "Task 6A section 18. Checkpoint BYTES are local-only and gitignored; this "
            "manifest is committed so a reviewer can verify what was produced."
        ),
        "checkpoints": entries,
        "base_models": base_models,
        "note": "No base Qwen or SAM2 weights are duplicated in any checkpoint.",
        **(extra or {}),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
