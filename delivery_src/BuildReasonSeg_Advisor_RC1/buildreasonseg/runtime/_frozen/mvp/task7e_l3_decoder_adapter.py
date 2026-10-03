"""Task 7E — read-only adapter that loads the two frozen L3 decoders behind one interface.

Nothing here changes either decoder: Z-B3 (frozen Task 6Z) is evaluated through the frozen
`scripts.task6z_evaluate` path and D-B1 (frozen Task 7D) through the frozen `scripts.task7d_evaluate` /
`buildreasonseg.runtime._frozen.mvp.task7d_global_competition_decoder` path. The adapter only verifies checkpoint hashes and
metadata (variant, selected epoch) and dispatches prediction, so both decoders keep their exact original
numerics.

No training, no parameter update, no architecture change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints"
Z_B3_CHECKPOINT = CHECKPOINT_ROOT / "task6z" / "zb3_minitrain1200.pt"
D_B1_CHECKPOINT = CHECKPOINT_ROOT / "task7d" / "db1_minitrain1200.pt"
Z_B3_SHA256 = "74f308e11e7f7f1098dd0d092ddcf6be1220d0b322b39df39c4ce05c9fc0f0bc"
D_B1_SHA256 = "6df31909cefdb54b9997b1e6ab76b8c5589771defa55666edbe75106221a89c0"
D_B1_SELECTED_EPOCH = 7
Z_B3_SHA256_KEY = "z_b3_sha256"
D_B1_SHA256_KEY = "d_b1_sha256"


class CheckpointUnavailable(RuntimeError):
    """Raised when a frozen Task 7E checkpoint is missing or has the wrong hash."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_metadata() -> dict:
    """Hash/metadata report for both frozen checkpoints (used by the scripts and the tests)."""

    z_sha = sha256_file(Z_B3_CHECKPOINT) if Z_B3_CHECKPOINT.is_file() else None
    d_sha = sha256_file(D_B1_CHECKPOINT) if D_B1_CHECKPOINT.is_file() else None
    metadata = {"z_b3": {"path": str(Z_B3_CHECKPOINT), "sha256": z_sha,
                         "expected_sha256": Z_B3_SHA256, "matches": z_sha == Z_B3_SHA256,
                         "variant": "Z-B3", "task": "6Z"},
                "d_b1": {"path": str(D_B1_CHECKPOINT), "sha256": d_sha,
                         "expected_sha256": D_B1_SHA256, "matches": d_sha == D_B1_SHA256,
                         "variant": "D-B1", "task": "7D"}}
    if d_sha == D_B1_SHA256:
        import torch

        payload = torch.load(D_B1_CHECKPOINT, map_location="cpu", weights_only=False)
        metadata["d_b1"].update({"checkpoint_variant": str(payload.get("variant")),
                                 "selected_epoch": int(payload.get("best_epoch", -1)),
                                 "selected_step": int(payload.get("best_step", -1)),
                                 "seed": int(payload.get("seed", -1)),
                                 "retrained": False})
        metadata["d_b1"]["epoch_matches"] = metadata["d_b1"]["selected_epoch"] == D_B1_SELECTED_EPOCH
    return metadata


class FrozenL3Decoder:
    """Unified read-only predictor for Z-B3 and D-B1."""

    def __init__(self, variant: str, device: str = "cuda") -> None:
        if variant not in ("Z-B3", "D-B1"):
            raise ValueError(f"{variant!r} is not a frozen Task 7E decoder")
        self.variant = variant
        self.device = device
        if variant == "Z-B3":
            if not Z_B3_CHECKPOINT.is_file() or sha256_file(Z_B3_CHECKPOINT) != Z_B3_SHA256:
                raise CheckpointUnavailable("TASK6Z_BASELINE_UNAVAILABLE")
            from scripts.task6z_evaluate import load_variant

            self.checkpoint = Z_B3_CHECKPOINT
            self.model = load_variant("Z-B3", device)
        else:
            if not D_B1_CHECKPOINT.is_file() or sha256_file(D_B1_CHECKPOINT) != D_B1_SHA256:
                raise CheckpointUnavailable("DB1_CHECKPOINT_UNAVAILABLE")
            from scripts.task7d_evaluate import load_variant_model

            self.checkpoint = D_B1_CHECKPOINT
            self.model = load_variant_model("D-B1", device)
        self.sha256 = sha256_file(self.checkpoint)
        self.params = int(sum(parameter.numel() for parameter in self.model.parameters()))

    # ------------------------------------------------------------------ prediction

    def predict(self, records: list[dict], store, masks, precomputed: dict,
                batch_size: int = 8) -> list[np.ndarray]:
        """Dispatch to the matching frozen prediction path (no shared code changes)."""

        if self.variant == "Z-B3":
            from scripts.task6z_evaluate import predict as predict_z_b3

            return predict_z_b3(self.model, records, store, masks, self.device, precomputed,
                                batch_size=batch_size)
        from scripts.task7d_evaluate import predict as predict_d_b1

        return predict_d_b1(self.model, records, store, masks, self.device, precomputed,
                            batch_size=batch_size)

    def report(self) -> dict:
        return {"variant": self.variant, "checkpoint": str(self.checkpoint), "sha256": self.sha256,
                "params": self.params, "retrained": False,
                "architecture": ("frozen Task 6Z Z-B3 L3 target decoder" if self.variant == "Z-B3"
                                 else "frozen Task 7D D-B1 deterministic field-weighted prototype"),
                "learned_competition": False, "score_head": False if self.variant == "D-B1" else None}


__all__ = ["CHECKPOINT_ROOT", "D_B1_CHECKPOINT", "D_B1_SELECTED_EPOCH", "D_B1_SHA256",
           "CheckpointUnavailable", "FrozenL3Decoder", "Z_B3_CHECKPOINT", "Z_B3_SHA256",
           "frozen_metadata", "sha256_file"]
