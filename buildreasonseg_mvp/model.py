"""The assembled Task 6A model: Qwen3-VL-2B -> `[SEG]` -> projection -> SAM2.1.

One forward pass produces both the language logits and the mask logits:

    image + instruction
        -> Qwen3-VL (teacher-forced with `reasoning_zh + " [SEG]"`)
        -> hidden state of the single `[SEG]` token
        -> ProjectionMLP
        -> one sparse prompt embedding
        -> frozen SAM2.1 features + trainable SAM2.1 mask decoder
        -> low-resolution mask logits

Trainable: text-only LoRA adapters, the `[SEG]` token representation, the
projection MLP, and the SAM2 mask decoder. Everything else is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .qwen_seg import TeacherForcedBatch, forward_qwen
from .sam2_bridge import (
    BRIDGE_CENTRE,
    BRIDGES,
    MaskDecodeResult,
    ProjectionMLP,
    Sam2Features,
    decode_mask,
)


@dataclass
class MvpForwardOutput:
    lm_logits: torch.Tensor
    seg_hidden: torch.Tensor
    projected: torch.Tensor
    mask_logits: torch.Tensor
    iou_prediction: torch.Tensor
    shapes: dict = field(default_factory=dict)
    sparse_prompt: torch.Tensor | None = None
    prompt_diagnostics: dict = field(default_factory=dict)
    #: Task 6D: the predicted target geometry when the grounded path is used.
    predicted_geometry: torch.Tensor | None = None

    def as_dict(self) -> dict:
        return {
            "lm_logits_shape": list(self.lm_logits.shape),
            "seg_hidden_shape": list(self.seg_hidden.shape),
            "projected_shape": list(self.projected.shape) if self.projected is not None else None,
            "mask_logits_shape": list(self.mask_logits.shape),
            "iou_prediction_shape": list(self.iou_prediction.shape),
            "predicted_geometry": (
                self.predicted_geometry.detach().float().cpu().tolist()
                if self.predicted_geometry is not None
                else None
            ),
            **self.shapes,
        }


class BuildReasonSegMvp(nn.Module):
    """Container that wires the language model, the projection and SAM2.1."""

    def __init__(
        self,
        qwen: nn.Module,
        sam: nn.Module,
        projection: ProjectionMLP,
        seg_token_id: int,
        token_holder: nn.Module | None = None,
        bridge: str = BRIDGE_CENTRE,
        grounding_head: nn.Module | None = None,
        geometry_kind: str | None = None,
    ) -> None:
        super().__init__()
        if bridge not in BRIDGES:
            raise ValueError(f"unknown SAM prompt bridge {bridge!r}; expected one of {BRIDGES}")
        self.qwen = qwen
        self.sam = sam
        self.projection = projection
        self.seg_token_id = int(seg_token_id)
        self.token_holder = token_holder
        #: Task 6C factor 2. Threaded to every `decode_mask` call so training,
        #: free-generation validation and the paired probe all use one bridge.
        self.bridge = bridge
        #: Task 6D: `[SEG]` hidden -> predicted target geometry -> official SAM2 prompt.
        #: Present only in the Spatial Grounding Bridge candidate; the Task 6C arms keep
        #: `grounding_head is None` and therefore their exact previous behaviour.
        self.grounding_head = grounding_head
        self.geometry_kind = geometry_kind

    # -- convenience -----------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.projection.parameters()).device

    def trainable_parameter_groups(
        self,
        lora_lr: float,
        head_lr: float,
        weight_decay: float,
        decoder_lr: float | None = None,
        token_lr: float | None = None,
    ):
        """Three AdamW groups.

        * token rows  -- zero weight decay; the `[SEG]` representation is a single
          embedding row and decaying it toward zero fights the objective. It has its
          own learning rate (`token_lr`), which defaults to `decoder_lr` so existing
          callers keep their behaviour.
        * decoder/projection -- the randomly-initialised projection and the
          repurposed SAM2 mask decoder need a higher learning rate than the
          pretrained LoRA adapters;
        * LoRA adapters -- pretrained-adjacent, so the lowest rate.

        Recorded defect (Task 6B): before `token_lr` existed this method always gave
        the token group `decoder_lr`, so a configured `optimizer.*.token_lr` was
        silently unused. In the Task 6B headline recipe `token_lr` and `decoder_lr`
        are both 3e-4, so the headline result is unaffected; in the adjusted recipe
        the `[SEG]` row therefore trained at 1e-3 rather than the configured 3e-4.
        """

        from .qwen_seg import is_token_parameter, output_row_param_ids, token_holder_parameters

        decoder_lr = decoder_lr if decoder_lr is not None else head_lr
        token_lr = token_lr if token_lr is not None else decoder_lr

        token_param_ids: set[int] = set(output_row_param_ids(self))
        for parameter in token_holder_parameters(self.token_holder):
            token_param_ids.add(id(parameter))
        for name, parameter in self.named_parameters():
            if is_token_parameter(name):
                token_param_ids.add(id(parameter))

        token: list[nn.Parameter] = []
        decoder: list[nn.Parameter] = []
        adapters: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in token_param_ids:
                token.append(parameter)
            elif name.startswith("projection.") or name.startswith("sam.sam_mask_decoder.") or name.startswith(
                "grounding_head."
            ):
                decoder.append(parameter)
            else:
                adapters.append(parameter)

        groups = []
        if adapters:
            groups.append({"params": adapters, "lr": lora_lr, "weight_decay": weight_decay, "name": "lora"})
        if decoder:
            groups.append(
                {"params": decoder, "lr": decoder_lr, "weight_decay": weight_decay, "name": "decoder"}
            )
        if token:
            groups.append({"params": token, "lr": token_lr, "weight_decay": 0.0, "name": "token"})
        return groups

    # -- forward ---------------------------------------------------------

    def forward(
        self,
        batch: TeacherForcedBatch,
        sam_features: Sam2Features,
        multimask_output: bool = False,
    ) -> MvpForwardOutput:
        lm_logits, seg_hidden = forward_qwen(self.qwen, batch)
        projected = self.projection(seg_hidden)
        decoded: MaskDecodeResult = decode_mask(
            self.sam,
            sam_features,
            projected,
            multimask_output=multimask_output,
            bridge=self.bridge,
        )
        return MvpForwardOutput(
            lm_logits=lm_logits,
            seg_hidden=seg_hidden,
            projected=projected,
            mask_logits=decoded.low_res_logits,
            iou_prediction=decoded.iou_prediction,
            shapes={
                "sparse_prompt_shape": list(decoded.sparse_prompt_shape),
                "visual_tokens": batch.visual_tokens,
                "sequence_length": batch.total_length,
                "prompt_length": batch.prompt_length,
                "bridge": self.bridge,
            },
            sparse_prompt=decoded.sparse_prompt,
            prompt_diagnostics=decoded.prompt_diagnostics,
        )

    def forward_grounded(
        self,
        batch: TeacherForcedBatch,
        sam_features: Sam2Features,
        multimask_output: bool = False,
    ) -> MvpForwardOutput:
        """Task 6D candidate path: `[SEG]` hidden -> predicted geometry -> SAM2.

        The projection MLP is not used and the old arbitrary projected language vector is
        never built, let alone fed to SAM. The only prompt is the geometry the head
        predicts; the language model contributes through the `[SEG]` hidden state.
        """

        if self.grounding_head is None or self.geometry_kind is None:
            raise RuntimeError("this model has no SpatialGroundingHead; call install_grounding_head first")

        from .grounding import decode_mask_from_geometry

        lm_logits, seg_hidden = forward_qwen(self.qwen, batch)
        geometry = self.grounding_head(seg_hidden)
        decoded = decode_mask_from_geometry(
            self.sam,
            sam_features,
            geometry,
            self.geometry_kind,
            multimask_output=multimask_output,
        )
        return MvpForwardOutput(
            lm_logits=lm_logits,
            seg_hidden=seg_hidden,
            projected=None,
            mask_logits=decoded.low_res_logits,
            iou_prediction=decoded.iou_prediction,
            shapes={
                "sparse_prompt_shape": list(decoded.sparse_prompt_shape),
                "visual_tokens": batch.visual_tokens,
                "sequence_length": batch.total_length,
                "prompt_length": batch.prompt_length,
                "bridge": "grounding",
                "geometry_kind": self.geometry_kind,
            },
            sparse_prompt=None,
            prompt_diagnostics=decoded.prompt_diagnostics,
            predicted_geometry=geometry,
        )
