"""Checkpoint verification helpers shared by the package contract and `check_setup.py`."""

from __future__ import annotations

import json
from pathlib import Path

from buildreasonseg.utils.hashing import sha256_file, verify_sha256


def verify_checkpoint(path: str | Path, expected_sha256: str | None,
                      expected_bytes: int | None = None) -> dict:
    return verify_sha256(path, expected_sha256, expected_bytes)


def load_checkpoint_header(path: str | Path, *, map_location: str = "cpu") -> dict:
    """Read a torch checkpoint header without instantiating any model (provenance checks only)."""

    import torch

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        return {"format": type(payload).__name__}
    header = {"keys": sorted(payload.keys())[:20]}
    for key in ("variant", "architecture", "version", "epoch", "seed", "metrics"):
        if key in payload:
            header[key] = payload[key]
    state = payload.get("state_dict")
    if isinstance(state, dict):
        header["parameter_tensors"] = len(state)
        header["parameter_elements"] = int(sum(int(value.numel()) for value in state.values()
                                               if hasattr(value, "numel")))
    return header


def checkpoint_manifest(path: str | Path) -> dict:
    target = Path(path)
    return {"path": str(target), "exists": target.is_file(),
            "bytes": target.stat().st_size if target.is_file() else None,
            "sha256": sha256_file(target) if target.is_file() else None}


def write_manifest(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


__all__ = ["checkpoint_manifest", "load_checkpoint_header", "verify_checkpoint", "write_manifest"]
