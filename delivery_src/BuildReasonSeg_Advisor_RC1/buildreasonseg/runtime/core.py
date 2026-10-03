"""Frozen core chain: SAM2 dense features → relation fields → D-B1 → target logits/mask.

Every numerical step here calls the **verbatim ported research implementation**
(`buildreasonseg/runtime/_frozen/mvp/`), so the delivery runtime cannot drift from Task 7I/7J:

* SAM2 dense feature: `task6n_relation_decoder.load_frozen_sam2_encoder` (frozen SAM2.1 Hiera Base+, 256×64×64);
* GeometricRelationField v0.2 + NearestBoundaryField v0.1: `task6z_field_composition.record_fields`;
* relation-conditioned target prototype `W`, normalised `A`, prototype `q`, cosine map `C` and the decoder
  logits: `task7d_global_competition_decoder.GlobalCompetitionDecoder`;
* threshold: the frozen research rule `upsampled_logits > 0.0` (no morphology, CRF, SAM refinement or
  component picking is applied anywhere).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from buildreasonseg import paths
from buildreasonseg.errors import BuildReasonSegError
from buildreasonseg.runtime import frozen_paths

TARGET_SIZE = 512
PROGRAM_TO_DIRECTION = {"largest_to_left_of_to_nearest": "left_of",
                        "largest_to_right_of_to_nearest": "right_of",
                        "largest_to_above_to_nearest": "above",
                        "largest_to_below_to_nearest": "below"}
CHECKPOINT_SHA256 = "9187b133ee4c71ca2750d421d6149bdba1812eabc8c9faacde193036c0db8586"


@dataclass
class CoreChainResult:
    logits: np.ndarray
    mask_context: np.ndarray
    probability_context: np.ndarray
    direction_field: np.ndarray
    nearest_field: np.ndarray
    relation_weight: np.ndarray
    prototype_similarity: np.ndarray
    attention: np.ndarray | None = None
    field_mass: float | None = None
    timings: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"logits_shape": list(self.logits.shape),
                "mask_area_context": int(self.mask_context.sum()),
                "field_mass": self.field_mass,
                "timings": dict(self.timings), "notes": list(self.notes)}


class Sam2Runtime:
    """Frozen SAM2.1 Hiera Base+ encoder loaded from the delivery component package."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._encoder = None
        self.report: dict = {}
        self.load_seconds: float | None = None

    def load(self):
        if self._encoder is not None:
            return self._encoder
        ok, detail = frozen_paths.sam2_assets_ok()
        if not ok:
            raise BuildReasonSegError("E302", detail=detail)
        frozen_paths.ensure()
        from buildreasonseg.runtime._frozen.mvp.task6n_relation_decoder import (
            load_frozen_sam2_encoder,
        )

        started = time.time()
        encoder, report = load_frozen_sam2_encoder(device=self.device)
        self._encoder = encoder
        self.report = report.as_dict() if hasattr(report, "as_dict") else dict(report or {})
        self.load_seconds = round(time.time() - started, 2)
        return encoder

    def encode(self, rgb: np.ndarray):
        """RGB uint8 512×512 → frozen dense feature tensor (256×64×64) on the device."""

        import torch

        encoder = self.load()
        with torch.no_grad():
            features = encoder.encode(rgb)
        tensor = features.image_embeddings[0].float()
        if tensor.dim() == 3:
            tensor = tensor[None]
        return tensor


class Db1Runtime:
    """D-B1 (`GlobalCompetitionDecoder("D-B1")`) loaded from the delivery model package."""

    def __init__(self, checkpoint: Path | None = None, device: str = "cuda") -> None:
        self.checkpoint = Path(checkpoint or (paths.default_model_dir() / "decoder.pt"))
        self.device = device
        self._model = None
        self.payload_step: int | None = None
        self.load_seconds: float | None = None

    def load(self):
        if self._model is not None:
            return self._model
        import torch

        frozen_paths.ensure()
        from buildreasonseg.runtime._frozen.mvp.task7d_global_competition_decoder import (
            GlobalCompetitionDecoder,
        )

        if not self.checkpoint.is_file():
            raise BuildReasonSegError("E301", detail=f"D-B1 权重缺失: {self.checkpoint}")
        started = time.time()
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise BuildReasonSegError("E304", detail="decoder.pt 不是 D-B1 state_dict 载荷")
        model = GlobalCompetitionDecoder("D-B1")
        model.load_state_dict(payload["state_dict"])
        model.to(self.device)
        model.eval()
        self._model = model
        self.payload_step = payload.get("step")
        self.load_seconds = round(time.time() - started, 2)
        return model

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.load().parameters()))

    def forward(self, visual, direction: str, directional, nearest) -> dict:
        """Run the frozen D-B1 chain and return logits plus every diagnostic map.

        `directional` / `nearest` are the frozen `P_dir_64` / `P_near_64` tensors `[1,1,64,64]`.
        `W = clamp(P_dir * P_near)`, `A` = global competition attention, `C` = prototype cosine similarity.
        """

        import torch

        model = self.load()
        relation = direction
        with torch.no_grad():
            visual_batch = visual.to(self.device).float()
            directional_batch = directional.to(self.device).float()
            nearest_batch = nearest.to(self.device).float()
            projected = model.visual_projection(visual_batch)
            state = model.competition(projected, [relation], directional_batch, nearest_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.device != "cpu"):
                logits = model(visual_batch, [relation], directional_batch, nearest_batch)
            upsampled = model.upsampled(logits, TARGET_SIZE)
            probability = torch.sigmoid(upsampled)
            relation_weight = (directional_batch * nearest_batch).clamp(0.0, 1.0)
        return {
            "logits_64": logits.detach().float().cpu().numpy()[0, 0],
            "logits": upsampled.detach().float().cpu().numpy()[0, 0],
            "probability": probability.detach().float().cpu().numpy()[0, 0],
            "mask": (upsampled > 0.0).detach().cpu().numpy()[0, 0],
            "direction_field": directional_batch.detach().float().cpu().numpy()[0, 0],
            "nearest_field": nearest_batch.detach().float().cpu().numpy()[0, 0],
            "relation_weight": relation_weight.detach().float().cpu().numpy()[0, 0],
            "prototype_similarity": state.similarity.detach().float().cpu().numpy()[0, 0],
            "attention": state.attention.detach().float().cpu().numpy()[0, 0],
            "field_mass": float(state.field_mass) if state.field_mass is not None else None,
        }


def record_fields(reference_mask: np.ndarray, program_id: str) -> dict:
    """Frozen field bundle (P_dir / P_near) for a 512 reference mask."""

    frozen_paths.ensure()
    from buildreasonseg.runtime._frozen.mvp.task6z_field_composition import record_fields as frozen

    return frozen(reference_mask, program_id)


def reference_mask_from_proposal(proposal, context) -> np.ndarray:
    """Extract the reference mask inside the reasoning context (512×512 boolean)."""

    from buildreasonseg.runtime.context import CONTEXT_SIZE

    reference = np.zeros((CONTEXT_SIZE, CONTEXT_SIZE), dtype=bool)
    top, left, bottom, right = proposal.global_bbox
    valid_top = max(0, context.top)
    valid_left = max(0, context.left)
    valid_bottom = min(proposal.global_mask.shape[0], context.top + CONTEXT_SIZE)
    valid_right = min(proposal.global_mask.shape[1], context.left + CONTEXT_SIZE)
    if valid_bottom <= valid_top or valid_right <= valid_left:
        return reference
    crop = proposal.global_mask[valid_top:valid_bottom, valid_left:valid_right]
    offset_top = valid_top - context.top
    offset_left = valid_left - context.left
    reference[offset_top:offset_top + crop.shape[0], offset_left:offset_left + crop.shape[1]] = crop
    return reference


def program_to_direction(program_id: str) -> str:
    if program_id not in PROGRAM_TO_DIRECTION:
        raise BuildReasonSegError("E102", detail=f"程序 {program_id!r} 不在 RC1 支持列表内")
    return PROGRAM_TO_DIRECTION[program_id]


def run_core_chain(rgb_context: np.ndarray, reference_mask: np.ndarray, program_id: str, *,
                   sam2: Sam2Runtime, db1: Db1Runtime, visual=None) -> CoreChainResult:
    """The frozen chain inside one 512 reasoning context.

    `visual` may inject an already-extracted frozen SAM2 feature tensor (used by the core-equivalence gate to
    isolate fields/decoder behaviour from feature extraction). The normal runtime path encodes the RGB itself.
    """

    import torch

    direction = program_to_direction(program_id)
    timings: dict[str, float] = {}
    started = time.time()
    features = sam2.encode(rgb_context) if visual is None else visual
    timings["sam2"] = round(time.time() - started, 3)
    started = time.time()
    fields = record_fields(reference_mask.astype(bool), program_id)
    timings["relation_fields"] = round(time.time() - started, 3)
    directional = torch.as_tensor(np.asarray(fields["P_dir_64"]))[None, None]
    nearest = torch.as_tensor(np.asarray(fields["P_near_64"]))[None, None]
    started = time.time()
    outputs = db1.forward(features, direction, directional, nearest)
    timings["db1"] = round(time.time() - started, 3)
    return CoreChainResult(
        logits=outputs["logits"], mask_context=outputs["mask"].astype(bool),
        probability_context=outputs["probability"], direction_field=outputs["direction_field"],
        nearest_field=outputs["nearest_field"], relation_weight=outputs["relation_weight"],
        prototype_similarity=outputs["prototype_similarity"], attention=outputs["attention"],
        field_mass=outputs["field_mass"], timings=timings,
        notes=[f"direction={direction}", f"field_size={np.asarray(fields['P_dir_64']).shape[-1]}",
               f"field_mass={outputs['field_mass']}",
               "threshold=upsampled_logits>0.0 (frozen research rule)"])


__all__ = ["CHECKPOINT_SHA256", "CoreChainResult", "Db1Runtime", "PROGRAM_TO_DIRECTION",
           "Sam2Runtime", "TARGET_SIZE", "program_to_direction", "record_fields",
           "reference_mask_from_proposal", "run_core_chain"]
