"""Shared runtime for the Task 6A stage scripts.

Keeps model assembly, preprocessing, and one training step in one place so the
four stage scripts stay thin and cannot drift apart.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from . import data as data_mod
from .checkpointing import write_json
from .losses import LossWeights, combined_loss, lm_cross_entropy, mask_bce_with_logits, mask_soft_dice
from .model import BuildReasonSegMvp
from .qwen_seg import (
    TeacherForcedBatch,
    attach_lora,
    build_teacher_forcing_batch,
    load_qwen,
    parameter_report,
    setup_seg_token,
)
from .sam2_bridge import (
    ProjectionMLP,
    Sam2Encoder,
    Sam2FeatureCache,
    apply_mvp_freeze_policy,
    load_sam2,
    sam_source_revision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mvp" / "task6a_2b_seg.yaml"


def load_config(path: Path | str | None = None) -> dict:
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def enable_determinism(seed: int, strict: bool = True) -> dict:
    """Make `training.deterministic` real, and report exactly what was enforced.

    Task 6B declared `training.deterministic: true` but never consumed it, so its
    Phase B was not bit-reproducible. This records each setting individually rather
    than asserting determinism: `bit_reproducible_claimed` is only true when every
    requested measure was applied *and* strict deterministic algorithms were
    accepted. CuBLAS workspace configuration is also pinned because
    `torch.use_deterministic_algorithms` requires it for matmul backward.
    """

    set_seed(seed)
    report = {
        "requested": True,
        "seed": int(seed),
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_manual_seed": True,
        "torch_cuda_manual_seed_all": True,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "strict_requested": bool(strict),
        "use_deterministic_algorithms": None,
        "use_deterministic_algorithms_warn_only": False,
        "cublas_workspace_config": None,
        "cublas_error": None,
        "bit_reproducible_claimed": False,
    }

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if strict:
        try:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            report["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        except Exception as exc:  # noqa: BLE001 - pragma: no cover
            report["cublas_error"] = f"{type(exc).__name__}: {exc}"
        torch.use_deterministic_algorithms(True)
        report["use_deterministic_algorithms"] = True
        report["bit_reproducible_claimed"] = True
    else:
        torch.use_deterministic_algorithms(True, warn_only=True)
        report["use_deterministic_algorithms"] = True
        report["use_deterministic_algorithms_warn_only"] = True
        report["bit_reproducible_claimed"] = False

    return report


def disable_determinism() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def vram() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "free_gib": round(free / 1024**3, 3),
        "total_gib": round(total / 1024**3, 3),
    }


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def rss_gib() -> float:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1024**3, 3)
    except Exception:  # noqa: BLE001
        return 0.0


def apply_image_budget(processor, min_pixels: int | None, max_pixels: int | None) -> dict:
    """Set the Qwen image token budget explicitly and record what actually happened.

    Qwen3-VL's released preprocessor config carries no top-level
    `min_pixels`/`max_pixels`; the values live on `image_processor.size`. Both
    locations are tried and the result is reported rather than assumed.
    """

    image_processor = getattr(processor, "image_processor", None)
    applied: dict = {
        "supported": False,
        "processor_class": type(image_processor).__name__ if image_processor else None,
        "min_pixels": None,
        "max_pixels": None,
        "applied_via": None,
        "size_dict_before": None,
        "size_dict_after": None,
    }
    if image_processor is None:
        return applied

    size = getattr(image_processor, "size", None)
    if size is not None and hasattr(size, "min_pixels"):
        applied["size_dict_before"] = {
            "min_pixels": getattr(size, "min_pixels", None),
            "max_pixels": getattr(size, "max_pixels", None),
            "shortest_edge": getattr(size, "shortest_edge", None),
            "longest_edge": getattr(size, "longest_edge", None),
        }
        if min_pixels is not None:
            size.min_pixels = int(min_pixels)
            applied["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            size.max_pixels = int(max_pixels)
            applied["max_pixels"] = int(max_pixels)
        applied["supported"] = True
        applied["applied_via"] = "image_processor.size"
        applied["size_dict_after"] = {
            "min_pixels": getattr(size, "min_pixels", None),
            "max_pixels": getattr(size, "max_pixels", None),
        }
        return applied

    for name, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        if value is None:
            continue
        if hasattr(image_processor, name):
            setattr(image_processor, name, int(value))
            applied[name] = int(value)
            applied["supported"] = True
            applied["applied_via"] = "image_processor attribute"
        else:
            applied[name] = "UNSUPPORTED_BY_PROCESSOR"
    return applied


# --------------------------------------------------------------------------
# Runtime container
# --------------------------------------------------------------------------


@dataclass
class MvpRuntime:
    cfg: dict
    processor: object
    tokenizer: object
    qwen: torch.nn.Module
    sam: torch.nn.Module
    projection: ProjectionMLP
    model: BuildReasonSegMvp
    sam_encoder: Sam2Encoder
    feature_cache: Sam2FeatureCache
    device: str
    reports: dict = field(default_factory=dict)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    #: Task 6E: `[BOX]` + `<loc_*>` vocabulary added on top of `[SEG]` (None = Task 6C/6D runtime).
    spatial_setup: object | None = None
    #: Task 6F: the single `[BOX]` query token setup (None = every other task's runtime).
    box_setup: object | None = None
    #: Task 6G: the selected dense spatial grid (None = every other task's runtime).
    dense_grid: int | None = None

    @property
    def spatial_codec(self):
        """The deterministic quantized-box codec this runtime's spatial tokens use."""

        from .spatial_tokens import QuantizedBoxCodec

        if self.spatial_setup is None:
            raise RuntimeError("runtime has no spatial token setup")
        return QuantizedBoxCodec(int(self.spatial_setup.bins))

    @property
    def bridge(self) -> str:
        """The SAM sparse-prompt bridge this runtime's model was built with."""

        return getattr(self.model, "bridge", "centre")

    # -- batches ---------------------------------------------------------

    def prepare(self, sample: data_mod.Sample, image: np.ndarray | None = None) -> tuple[TeacherForcedBatch, np.ndarray]:
        """Build the teacher-forcing batch for one sample.

        `image` lets a caller supply an already-decoded frame (Task 6C.5 source cache)
        so the PNG is not decoded twice; the default decodes from disk exactly as before.
        """

        if image is None:
            image = sample.image_rgb()
        batch = build_teacher_forcing_batch(
            self.processor,
            self.tokenizer,
            image,
            sample.instruction_zh,
            sample.assistant_text,
            self.model.seg_token_id,
            append_eos=bool(
                self.cfg.get("stage2_overfit", {}).get(
                    "append_eos_to_target", self.cfg.get("training", {}).get("append_eos_to_target", True)
                )
            ),
        )
        return batch, image

    def features_for(self, sample: data_mod.Sample, image: np.ndarray):
        use_cache = bool(
            self.cfg.get("stage2_overfit", {}).get("use_sam_feature_cache", True)
            and self.cfg.get("training", {}).get("use_sam_feature_cache", True)
        )
        if use_cache:
            return self.feature_cache.get(sample.image_id, image)
        return self.sam_encoder.encode(image), False

    def features_for_image(self, image: np.ndarray, image_id: str | None = None):
        """SAM2 image features for a bare frame (Task 6E section 19 inference plumbing)."""

        use_cache = bool(
            image_id
            and self.cfg.get("stage2_overfit", {}).get("use_sam_feature_cache", True)
            and self.cfg.get("training", {}).get("use_sam_feature_cache", True)
        )
        if use_cache:
            return self.feature_cache.get(str(image_id), image)
        return self.sam_encoder.encode(image), False

    # -- Task 6C.7: frozen Qwen visual-feature cache ---------------------

    def set_visual_cache_key(self, key: str | None) -> None:
        """Identify the source image for the visual-feature cache.

        Task 6C.7 section 8 keys the cache by immutable source-image identity, so two
        instructions on one image share one entry. When no key is set the cache falls
        back to a content hash of the processed pixels, which is always correct and
        simply costs one hash instead of nothing.
        """

        self.visual_cache_key = key

    def install_visual_cache(self, enabled: bool = True, max_images: int = 512) -> dict:
        """Install the bounded CPU cache over the frozen Qwen visual tower.

        The wrapper is installed on this runtime's own Qwen module and stores only
        `pooler_output` and `deepstack_features` — the two things `Qwen3VLModel.forward`
        reads from the visual tower, both produced exclusively by frozen parameters. No
        language-model hidden state, `[SEG]` state or logit is cached.
        """

        from .visual_cache import VisualFeatureCache, install_visual_feature_cache

        self.visual_cache = VisualFeatureCache(max_images=max_images, enabled=enabled)
        self.visual_cache_key = None
        report = install_visual_feature_cache(
            self.model.qwen,
            self.visual_cache,
            key_provider=lambda: getattr(self, "visual_cache_key", None),
        )
        report["max_images"] = max_images
        report["enabled"] = enabled
        self.reports["visual_cache"] = report
        return report

    def visual_cache_stats(self) -> dict:
        cache = getattr(self, "visual_cache", None)
        if cache is None:
            return {"enabled": False, "installed": False}
        return {"installed": True, **cache.stats()}

    # -- Task 6D: Spatial Grounding Bridge -------------------------------

    def install_grounding_head(self, kind: str, hidden_dim: int | None = None, mid_dim: int = 512) -> dict:
        """Attach the `SpatialGroundingHead` to this runtime's model (Task 6D section 6)."""

        from .grounding import SpatialGroundingHead

        hidden = int(hidden_dim or self.reports.get("qwen_hidden_size") or 2048)
        head = SpatialGroundingHead(hidden_dim=hidden, mid_dim=mid_dim, kind=kind).to(self.device)
        self.model.grounding_head = head
        self.model.geometry_kind = kind
        self.grounding_kind = kind
        report = {
            "installed": True,
            "hidden_dim": hidden,
            "device": str(self.device),
            **head.as_dict(),
            "frozen_sam_image_encoder": True,
            "projection_used_by_candidate": False,
        }
        self.reports["grounding_head"] = report
        return report

    def set_grounding_trainables(self, stage: str) -> dict:
        """Task 6D section 9/10 trainable sets.

        * ``G0`` — grounding proof: text LoRA + `[SEG]` machinery + the head. The SAM2
          decoder and the projection MLP stay frozen, so no mask gradient flows at all.
        * ``G1`` — joint segmentation: additionally the SAM2 mask decoder. The prompt
          encoder, the SAM2 image encoder, the Qwen base and the visual tower stay frozen,
          and the projection MLP is never used by the candidate.
        """

        stage = stage.upper()
        if stage not in ("G0", "G1"):
            raise ValueError(f"stage must be 'G0' or 'G1', got {stage!r}")

        model = self.model
        for parameter in model.projection.parameters():
            parameter.requires_grad_(False)
        for parameter in model.sam.sam_mask_decoder.parameters():
            parameter.requires_grad_(stage == "G1")
        if model.grounding_head is not None:
            for parameter in model.grounding_head.parameters():
                parameter.requires_grad_(True)

        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        groups = {
            "lora_or_adapter": [n for n in trainable if "lora_" in n or "token_holder" in n],
            "grounding_head": [n for n in trainable if n.startswith("grounding_head.")],
            "sam_mask_decoder": [n for n in trainable if n.startswith("sam.sam_mask_decoder.")],
            "projection": [n for n in trainable if n.startswith("projection.")],
        }
        report = {
            "stage": stage,
            "trainable_tensors": len(trainable),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "grounding_head_trainable": any(p.requires_grad for p in model.grounding_head.parameters())
            if model.grounding_head is not None
            else False,
            "sam_mask_decoder_trainable": any(
                p.requires_grad for p in model.sam.sam_mask_decoder.parameters()
            ),
            "projection_trainable": any(p.requires_grad for p in model.projection.parameters()),
            "sam_prompt_encoder_trainable": any(
                p.requires_grad for p in model.sam.sam_prompt_encoder.parameters()
            ),
            "sam_image_encoder_trainable": any(
                p.requires_grad for p in model.sam.image_encoder.parameters()
            ),
            "qwen_visual_trainable": [
                n for n in trainable if "visual" in n or "vision_tower" in n or "vision_model" in n
            ],
            "group_counts": {key: len(value) for key, value in groups.items()},
        }
        self.reports["grounding_trainables"] = report
        return report

    def grounding_loss_weights(self, include_mask_loss: bool) -> "LossWeights":
        """Task 6D section 8: keep the Task 6C headline weights, add lambda_ground."""

        base = self.cfg.get("loss", {})
        return LossWeights(
            lm_ce=float(base.get("lm_ce", 2.0)),
            mask_bce=float(base.get("mask_bce", 2.0)) if include_mask_loss else 0.0,
            mask_dice=float(base.get("mask_dice", 1.0)) if include_mask_loss else 0.0,
        )

    def grounding_train_step(
        self,
        batch,
        gt_mask,
        features,
        optimizer=None,
        *,
        lambda_ground: float = 5.0,
        include_mask_loss: bool = False,
        geometry_kind: str | None = None,
        timer=None,
    ) -> dict:
        """One Spatial Grounding Bridge step (Task 6D sections 8-10).

        `gt_mask` is supervision only. GT geometry is derived from it here and is never an
        input to the model; the prompt path always uses the head's prediction.
        """

        from .grounding import geometry_smooth_l1, target_geometry

        kind = geometry_kind or getattr(self.model, "geometry_kind", None)
        if kind is None:
            raise RuntimeError("no geometry kind set; call install_grounding_head first")

        training_cfg = self.cfg["training"]
        batch = batch.to(self.device)
        use_autocast = bool(training_cfg.get("bf16_autocast", True))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast)

        def _stage(name):
            if timer is None:
                return contextlib.nullcontext()
            return timer.stage(name)

        weights = self.grounding_loss_weights(include_mask_loss)

        with _stage("grounding_forward"):
            with autocast:
                output = self.model.forward_grounded(batch, features)
                lm_ce = lm_cross_entropy(output.lm_logits, batch.labels)
                target_geometry_tensor = torch.as_tensor(
                    target_geometry(gt_mask, kind), dtype=torch.float32, device=self.device
                ).reshape(1, -1)
                ground = geometry_smooth_l1(output.predicted_geometry, target_geometry_tensor)

        total = weights.lm_ce * lm_ce + lambda_ground * ground
        breakdown = {
            "lm_ce": lm_ce,
            "mask_bce": torch.zeros((), device=self.device),
            "mask_dice": torch.zeros((), device=self.device),
            "ground": ground,
        }

        if include_mask_loss:
            with _stage("mask_loss"):
                with autocast:
                    target, supervision_logits, supervision_mode = self.build_mask_supervision(
                        gt_mask, output.mask_logits
                    )
                    mask_bce = mask_bce_with_logits(supervision_logits, target)
                    mask_dice = mask_soft_dice(supervision_logits, target)
            breakdown["mask_bce"] = mask_bce
            breakdown["mask_dice"] = mask_dice
            total = (
                weights.lm_ce * lm_ce
                + weights.mask_bce * mask_bce
                + weights.mask_dice * mask_dice
                + lambda_ground * ground
            )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with _stage("backward"):
            total.backward()

        clipped = None
        if optimizer is not None:
            with _stage("optimizer_step"):
                clipped = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.cfg["optimizer"]["grad_clip_norm"]),
                    )
                )
                optimizer.step()

        return {
            "losses": {
                "total": float(total.detach()),
                "lm_ce": float(lm_ce.detach()),
                "mask_bce": float(breakdown["mask_bce"].detach()),
                "mask_dice": float(breakdown["mask_dice"].detach()),
                "ground": float(ground.detach()),
                "lm_ce_raw": float(lm_ce.detach()),
                "mask_bce_raw": float(breakdown["mask_bce"].detach()),
                "mask_dice_raw": float(breakdown["mask_dice"].detach()),
                "ground_raw": float(ground.detach()),
            },
            "predicted_geometry": output.predicted_geometry.detach().float().cpu(),
            "gt_geometry": target_geometry_tensor.detach().float().cpu(),
            "grad_clip_total_norm": clipped,
            "geometry_kind": kind,
            "lambda_ground": lambda_ground,
            "include_mask_loss": include_mask_loss,
            "mask_supervision": None if not include_mask_loss else supervision_mode,
            "weights": weights.as_dict(),
        }

    # -- optimisation ----------------------------------------------------

    def freeze_for_spatial_tokens(self) -> dict:
        """Task 6E sections 10-11 trainable set.

        Train: text LoRA + `[SEG]` + `[BOX]` + every `<loc_*>` row.
        Freeze: Qwen base, the visual tower, all of SAM2, the old Projection MLP and the
        Task 6D `SpatialGroundingHead` (when one happens to be attached).
        """

        if self.spatial_setup is None:
            raise RuntimeError("freeze_for_spatial_tokens needs a spatial token setup")
        report = set_phase_trainables(self.model, "A")
        if self.model.grounding_head is not None:
            for parameter in self.model.grounding_head.parameters():
                parameter.requires_grad_(False)
            report["grounding_head_trainable"] = False
        report["spatial_bins"] = int(self.spatial_setup.bins)
        report["trainable_token_ids"] = list(self.reports.get("lora", {}).get("trainable_token_ids") or [])
        report["sam_frozen"] = not any(p.requires_grad for p in self.model.sam.parameters())
        report["projection_frozen"] = not any(p.requires_grad for p in self.model.projection.parameters())
        self.reports["spatial_trainables"] = report
        return report

    def spatial_train_step(self, batch, example, optimizer=None, timer=None) -> dict:
        """One Task 6E step: `1.0 * assistant CE + 5.0 * location CE` on the language model.

        No mask loss and no SAM forward: the Task 6E objective supervises the emitted
        coordinate tokens, and SAM2 stays completely frozen (sections 9 and 11).
        """

        from .qwen_seg import forward_qwen
        from .spatial_training import spatial_loss, spatial_teacher_forced_metrics

        batch = batch.to(self.device)
        use_autocast = bool(self.cfg["training"].get("bf16_autocast", True))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast)

        def _stage(name):
            if timer is None:
                return contextlib.nullcontext()
            return timer.stage(name)

        with _stage("qwen_forward"):
            with autocast:
                lm_logits, _hidden = forward_qwen(self.model.qwen, batch)
                losses = spatial_loss(lm_logits, batch, example, self.spatial_setup)

        with _stage("backward"):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()

        clipped = None
        if optimizer is not None:
            with _stage("optimizer_step"):
                clipped = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.cfg["optimizer"]["grad_clip_norm"]),
                    )
                )
                optimizer.step()

        with torch.no_grad():
            diagnostics = spatial_teacher_forced_metrics(lm_logits, batch, example)

        return {
            "losses": {
                "total": losses["total_raw"],
                "assistant_ce": losses["assistant_ce_raw"],
                "location_ce": losses["location_ce_raw"],
                "lm_ce": losses["assistant_ce_raw"],
                "mask_bce": 0.0,
                "mask_dice": 0.0,
                "ground": 0.0,
            },
            "location_token_accuracy": losses["location_token_accuracy"],
            "location_abs_bin_error": losses["location_abs_bin_error"],
            "teacher_forced": diagnostics,
            "grad_clip_total_norm": clipped,
            "weights": {
                "assistant": losses["assistant_weight"],
                "location": losses["location_weight"],
            },
            "example": example.as_dict(),
        }

    # -- Task 6F: Target-Aware [BOX] Query --------------------------------

    def install_box_head(self, hidden_dim: int | None = None, mid_dim: int = 512) -> dict:
        """Attach the `TargetAwareBoxHead` to this runtime's model (Task 6F section 5)."""

        from .box_query import TargetAwareBoxHead

        if self.box_setup is None:
            raise RuntimeError("install_box_head needs the Task 6F [BOX] query setup")
        hidden = int(hidden_dim or self.reports.get("qwen_hidden_size") or 2048)
        head = TargetAwareBoxHead(hidden_dim=hidden, mid_dim=int(mid_dim)).to(self.device)
        self.model.box_head = head
        self.box_head = head
        report = head.as_dict()
        report.update(
            {
                "installed": True,
                "box_token_id": int(self.box_setup.box_token_id),
                "trainable": all(parameter.requires_grad for parameter in head.parameters()),
            }
        )
        self.reports["box_head"] = report
        return report

    def freeze_for_box_query(self) -> dict:
        """Task 6F section 8 trainable set.

        Train: text LoRA, the `[BOX]` query row, the `[SEG]` row and the `TargetAwareBoxHead`.
        Freeze: Qwen base, the visual tower, all SAM2, the old Projection MLP, the Task 6D
        `SpatialGroundingHead`, and any Task 6E `<loc_*>` rows (none are added in Task 6F, and
        the freeze assertion below records that the location vocabulary is absent).
        """

        if self.box_setup is None:
            raise RuntimeError("freeze_for_box_query needs the Task 6F [BOX] query setup")
        report = set_phase_trainables(self.model, "A")
        if self.model.grounding_head is not None:
            for parameter in self.model.grounding_head.parameters():
                parameter.requires_grad_(False)
            report["grounding_head_trainable"] = False
        if self.model.box_head is not None:
            for parameter in self.model.box_head.parameters():
                parameter.requires_grad_(True)
        report["box_query_token_id"] = int(self.box_setup.box_token_id)
        report["trainable_token_ids"] = list(self.reports.get("lora", {}).get("trainable_token_ids") or [])
        report["box_head_trainable"] = (
            all(parameter.requires_grad for parameter in self.model.box_head.parameters())
            if self.model.box_head is not None
            else False
        )
        report["sam_frozen"] = not any(p.requires_grad for p in self.model.sam.parameters())
        report["projection_frozen"] = not any(p.requires_grad for p in self.model.projection.parameters())
        report["loc_tokens_present"] = any(
            name.startswith("<loc_") for name in self.tokenizer.get_added_vocab()
        )
        report["loc_rows_trainable"] = False  # Task 6F adds no location tokens at all
        self.reports["box_query_trainables"] = report
        return report

    def box_query_train_step(self, batch, gt_mask, optimizer=None, timer=None) -> dict:
        """One Task 6F step: `1.0 * L_reasoning + 5.0 * L_box` (section 7).

        `gt_mask` is supervision only: the GT box is derived from it and the query hidden is
        produced purely by the prompt + constant `[BOX]` under causal attention. No mask loss
        and no SAM forward; SAM2 stays completely frozen in F0/F1.
        """

        from .box_query import box_query_forward, box_query_loss
        from .grounding import target_geometry

        batch = batch.to(self.device)
        use_autocast = bool(self.cfg["training"].get("bf16_autocast", True))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast)

        def _stage(name):
            if timer is None:
                return contextlib.nullcontext()
            return timer.stage(name)

        with _stage("qwen_forward"):
            with autocast:
                lm_logits, box_hidden, _seg_hidden = box_query_forward(self.model.qwen, batch)
                predicted_box = self.model.box_head(box_hidden)
                gt_box = torch.as_tensor(
                    target_geometry(gt_mask, "box"), dtype=torch.float32, device=self.device
                ).reshape(1, -1)
                losses = box_query_loss(lm_logits, batch, predicted_box, gt_box)

        with _stage("backward"):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()

        clipped = None
        if optimizer is not None:
            with _stage("optimizer_step"):
                clipped = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.cfg["optimizer"]["grad_clip_norm"]),
                    )
                )
                optimizer.step()

        return {
            "losses": {
                "total": losses["total_raw"],
                "reasoning_ce": losses["reasoning_ce_raw"],
                "box_loss": losses["box_loss_raw"],
                "lm_ce": losses["reasoning_ce_raw"],
                "mask_bce": 0.0,
                "mask_dice": 0.0,
                "ground": losses["box_loss_raw"],
            },
            "box_iou": losses["box_iou"],
            "predicted_box": predicted_box.detach().float().cpu().reshape(-1).tolist(),
            "gt_box": gt_box.detach().float().cpu().reshape(-1).tolist(),
            "grad_clip_total_norm": clipped,
            "weights": {
                "reasoning": losses["reasoning_weight"],
                "box": losses["box_weight"],
            },
        }

    # -- Task 6G: Dense Query-Visual Spatial Grounding Map -------------------

    def install_dense_head(self, in_channels: int | None = None) -> dict:
        """Attach the `DenseSpatialGroundingHead` to this runtime's model (Task 6G section 5)."""

        from .dense_grounding import DenseSpatialGroundingHead

        if self.box_setup is None or self.dense_grid is None:
            raise RuntimeError("install_dense_head needs the [BOX] query setup and a selected grid")
        head = DenseSpatialGroundingHead(
            query_dim=int(self.reports.get("qwen_hidden_size") or 2048),
            key_dim=int(self.cfg.get("dense_grounding", {}).get("key_dim", 128)),
            in_channels=in_channels,
            use_bias=bool(self.cfg.get("dense_grounding", {}).get("scalar_bias", True)),
        ).to(self.device)
        self.model.dense_head = head
        self.dense_head = head
        report = head.as_dict()
        report.update(
            {
                "installed": True,
                "grid": int(self.dense_grid),
                "box_token_id": int(self.box_setup.box_token_id),
                "trainable": all(parameter.requires_grad for parameter in head.parameters()),
            }
        )
        self.reports["dense_head"] = report
        return report

    def freeze_for_dense_grounding(self) -> dict:
        """Task 6G section 10 trainable set.

        Train: text LoRA, the `[BOX]` query row, the `[SEG]` row and the
        `DenseSpatialGroundingHead`. Freeze: Qwen base, the visual tower, all SAM2, the Task 6F
        `TargetAwareBoxHead`, the Task 6D `SpatialGroundingHead`, the old Projection MLP, and any
        Task 6E `<loc_*>` rows (none are added in Task 6G).
        """

        if self.box_setup is None or self.dense_grid is None:
            raise RuntimeError("freeze_for_dense_grounding needs the [BOX] query and a grid")
        report = set_phase_trainables(self.model, "A")
        for head_name in ("grounding_head", "box_head"):
            head = getattr(self.model, head_name, None)
            if head is not None:
                for parameter in head.parameters():
                    parameter.requires_grad_(False)
            report[f"{head_name}_trainable"] = False
        if self.model.dense_head is not None:
            for parameter in self.model.dense_head.parameters():
                parameter.requires_grad_(True)
        report["box_query_token_id"] = int(self.box_setup.box_token_id)
        report["dense_grid"] = int(self.dense_grid)
        report["trainable_token_ids"] = list(self.reports.get("lora", {}).get("trainable_token_ids") or [])
        report["dense_head_trainable"] = (
            all(parameter.requires_grad for parameter in self.model.dense_head.parameters())
            if self.model.dense_head is not None
            else False
        )
        report["sam_frozen"] = not any(p.requires_grad for p in self.model.sam.parameters())
        report["projection_frozen"] = not any(p.requires_grad for p in self.model.projection.parameters())
        report["loc_tokens_present"] = any(
            name.startswith("<loc_") for name in self.tokenizer.get_added_vocab()
        )
        report["loc_rows_trainable"] = False  # Task 6G adds no location tokens at all
        self.reports["dense_trainables"] = report
        return report

    def dense_train_step(self, batch, gt_mask, features, optimizer=None, timer=None) -> dict:
        """One Task 6G step: `1.0 * L_reasoning + 2.0 * L_heatmap` (section 8).

        `gt_mask` is supervision only; the soft heatmap target is the instruction-selected target
        mask downsampled to the selected grid. The SAM2 spatial feature is a frozen encoder output
        (from the existing CPU cache); gradients flow only through the trainable 1x1 projection.
        """

        from .box_query import box_query_forward
        from .dense_grounding import (
            HEATMAP_LOSS_WEIGHT,
            REASONING_LOSS_WEIGHT,
            downsample_target_mask,
            feature_tensor_for_grid,
            heatmap_loss,
        )
        from .losses import lm_cross_entropy

        batch = batch.to(self.device)
        use_autocast = bool(self.cfg["training"].get("bf16_autocast", True))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast)

        def _stage(name):
            if timer is None:
                return contextlib.nullcontext()
            return timer.stage(name)

        with _stage("qwen_forward"):
            with autocast:
                lm_logits, box_hidden, _seg_hidden = box_query_forward(self.model.qwen, batch)
                spatial_feature = feature_tensor_for_grid(features, int(self.dense_grid)).to(self.device)
                heatmap_logits = self.model.dense_head(box_hidden, spatial_feature).unsqueeze(1)
                soft_target = downsample_target_mask(gt_mask, int(self.dense_grid)).to(self.device)
                soft_target = soft_target[None, None, :, :]
                losses = heatmap_loss(heatmap_logits, soft_target)
                reasoning = lm_cross_entropy(lm_logits, batch.labels)
                total = REASONING_LOSS_WEIGHT * reasoning + HEATMAP_LOSS_WEIGHT * losses["heatmap"]

        with _stage("backward"):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            total.backward()

        clipped = None
        if optimizer is not None:
            with _stage("optimizer_step"):
                clipped = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.cfg["optimizer"]["grad_clip_norm"]),
                    )
                )
                optimizer.step()

        return {
            "losses": {
                "total": float(total.detach()),
                "reasoning_ce": float(reasoning.detach()),
                "heatmap_bce": losses["bce_raw"],
                "heatmap_dice": losses["dice_raw"],
                "heatmap_loss": losses["heatmap_raw"],
                "lm_ce": float(reasoning.detach()),
                "mask_bce": losses["bce_raw"],
                "mask_dice": losses["dice_raw"],
                "ground": losses["heatmap_raw"],
            },
            "heatmap_dice_quality": float(1.0 - losses["dice_raw"]),
            "grad_clip_total_norm": clipped,
            "weights": {
                "reasoning": float(REASONING_LOSS_WEIGHT),
                "heatmap": float(HEATMAP_LOSS_WEIGHT),
                "heatmap_bce": 1.0,
                "heatmap_dice": 1.0,
            },
        }

    # -- Task 6H: counterfactual pair-aligned dense grounding ----------------

    def freeze_for_counterfactual(self) -> dict:
        """Task 6H section 5 trainable set: identical to Task 6G (no new module).

        Train: text LoRA, the `[BOX]` query row, the `[SEG]` row and the Task 6G
        `DenseSpatialGroundingHead`. Freeze: everything else, including the Task 6F box head and
        the Task 6D grounding head.
        """

        report = self.freeze_for_dense_grounding()
        report["task"] = "6H"
        report["pair_step"] = {
            "pair_steps_per_epoch": int(self.cfg.get("counterfactual", {}).get("pairs_per_epoch", 240)),
            "margin": float(self.cfg.get("counterfactual", {}).get("margin", 1.0)),
            "weights": {
                "reasoning": 0.5,
                "heatmap": 1.0,
                "cf": 2.0,
            },
        }
        self.reports["counterfactual_trainables"] = report
        return report

    def pair_train_step(self, batch_a, batch_b, mask_a, mask_b, features, optimizer=None,
                        timer=None) -> dict:
        """Task 6H sections 6-9: one optimizer step = one same-image counterfactual pair.

        Both query paths run sequentially with both graphs retained against **one shared frozen
        SAM2 feature**, and a single backward/optimizer step consumes the combined objective:

            L_total = 0.5 * (L_reasoning_A + L_reasoning_B)
                    + 1.0 * (L_heatmap_A + L_heatmap_B)
                    + 2.0 * L_cf
        """

        from .box_query import box_query_forward
        from .counterfactual import counterfactual_loss, region_scores
        from .dense_grounding import downsample_target_mask, feature_tensor_for_grid, heatmap_loss
        from .losses import lm_cross_entropy

        batch_a = batch_a.to(self.device)
        batch_b = batch_b.to(self.device)
        margin = float(self.cfg.get("counterfactual", {}).get("margin", 1.0))
        use_autocast = bool(self.cfg["training"].get("bf16_autocast", True))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast)

        def _stage(name):
            if timer is None:
                return contextlib.nullcontext()
            return timer.stage(name)

        with _stage("qwen_forward"):
            with autocast:
                # One frozen visual feature, shared by both instructions of the same image.
                spatial_feature = feature_tensor_for_grid(features, int(self.dense_grid)).to(self.device)
                logits_a, hidden_a, _seg_a = box_query_forward(self.model.qwen, batch_a)
                heatmap_a = self.model.dense_head(hidden_a, spatial_feature).unsqueeze(1)
                logits_b, hidden_b, _seg_b = box_query_forward(self.model.qwen, batch_b)
                heatmap_b = self.model.dense_head(hidden_b, spatial_feature).unsqueeze(1)

                target_a = downsample_target_mask(mask_a, int(self.dense_grid)).to(self.device)
                target_b = downsample_target_mask(mask_b, int(self.dense_grid)).to(self.device)
                target_a = target_a[None, None, :, :]
                target_b = target_b[None, None, :, :]

                losses_a = heatmap_loss(heatmap_a, target_a)
                losses_b = heatmap_loss(heatmap_b, target_b)
                reasoning = 0.5 * (
                    lm_cross_entropy(logits_a, batch_a.labels)
                    + lm_cross_entropy(logits_b, batch_b.labels)
                )
                scores = region_scores(heatmap_a[0, 0], heatmap_b[0, 0], target_a[0, 0], target_b[0, 0])
                cf = counterfactual_loss(scores, margin)
                total = (
                    0.5 * reasoning
                    + 1.0 * (losses_a["heatmap"] + losses_b["heatmap"])
                    + 2.0 * cf["loss"]
                )

        with _stage("backward"):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            total.backward()

        clipped = None
        if optimizer is not None:
            with _stage("optimizer_step"):
                clipped = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.cfg["optimizer"]["grad_clip_norm"]),
                    )
                )
                optimizer.step()

        return {
            "losses": {
                "total": float(total.detach()),
                "reasoning_ce": float(reasoning.detach()),
                "heatmap_a": losses_a["heatmap_raw"],
                "heatmap_b": losses_b["heatmap_raw"],
                "heatmap_sum": losses_a["heatmap_raw"] + losses_b["heatmap_raw"],
                "bce_a": losses_a["bce_raw"],
                "bce_b": losses_b["bce_raw"],
                "dice_a": losses_a["dice_raw"],
                "dice_b": losses_b["dice_raw"],
                "cf": cf["raw"],
                "cf_a": cf["raw_a"],
                "cf_b": cf["raw_b"],
                "lm_ce": float(reasoning.detach()),
                "mask_bce": losses_a["bce_raw"] + losses_b["bce_raw"],
                "mask_dice": losses_a["dice_raw"] + losses_b["dice_raw"],
                "ground": cf["raw"],
            },
            "region_scores": scores["raw"],
            "pair_ranking_pass": cf["pair_ranking_pass"],
            "strict_margin_pass": cf["strict_margin_pass"],
            "heatmap_dice_quality_a": float(1.0 - losses_a["dice_raw"]),
            "heatmap_dice_quality_b": float(1.0 - losses_b["dice_raw"]),
            "grad_clip_total_norm": clipped,
            "weights": {
                "reasoning": 0.5,
                "heatmap": 1.0,
                "cf": 2.0,
                "cf_margin": margin,
            },
        }

    # -- Task 6H.1: bounded spatial-softmax point objective ------------------

    def freeze_for_point_objective(self) -> dict:
        """Task 6H.1 section 1 trainable set: identical to Task 6G/6H (architecture frozen)."""

        report = self.freeze_for_counterfactual()
        report["task"] = "6H.1"
        report["objective"] = {
            "point_cross_entropy": True,
            "bounded_pair_mass": True,
            "legacy_bce_dice_gradient": False,
            "legacy_logit_ranking_gradient": False,
            "weights": {"reasoning": 0.5, "classification": 1.0, "cf": 1.0},
            "eps": 1e-8,
        }
        self.reports["point_objective_trainables"] = report
        return report

    def bounded_pair_train_step(self, batch_a, batch_b, mask_a, mask_b, features,
                                optimizer=None, timer=None) -> dict:
        """Task 6H.1 sections 5-11: one pair step with the point-aligned bounded objective.

        `L_total = 0.5*(L_reasoning_A+L_reasoning_B) + 1.0*(L_point_A+L_point_B) + 1.0*L_cf`, where
        `L_point` is the 65,536-class spatial cross-entropy of the target point-cell and `L_cf` is
        the bounded own-vs-cross target **probability mass** preference. Task 6G's BCE+Dice and
        Task 6H's raw-logit ranking are computed under `torch.no_grad()` for logging and therefore
        contribute **zero** gradient.
        """

        from .box_query import box_query_forward
        from .counterfactual import region_scores
        from .dense_grounding import downsample_target_mask, feature_tensor_for_grid, heatmap_loss
        from .losses import lm_cross_entropy
        from .spatial_objective import (
            POINT_CF_WEIGHT,
            POINT_CLASSIFICATION_WEIGHT,
            POINT_REASONING_WEIGHT,
            bounded_pair_loss,
            counterfactual_masses,
            max_non_target_probability,
            point_cross_entropy,
            spatial_entropy,
            spatial_probabilities,
            target_cell,
            target_cell_probability,
            topk_hit,
        )

        batch_a = batch_a.to(self.device)
        batch_b = batch_b.to(self.device)
        grid = int(self.dense_grid)
        use_autocast = bool(self.cfg["training"].get("bf16_autocast", True))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast)

        def _stage(name):
            if timer is None:
                return contextlib.nullcontext()
            return timer.stage(name)

        with _stage("qwen_forward"):
            with autocast:
                spatial_feature = feature_tensor_for_grid(features, grid).to(self.device)
                logits_a, hidden_a, _seg_a = box_query_forward(self.model.qwen, batch_a)
                heatmap_a = self.model.dense_head(hidden_a, spatial_feature)
                logits_b, hidden_b, _seg_b = box_query_forward(self.model.qwen, batch_b)
                heatmap_b = self.model.dense_head(hidden_b, spatial_feature)

                target_a = downsample_target_mask(mask_a, grid).to(self.device)[None, None, :, :]
                target_b = downsample_target_mask(mask_b, grid).to(self.device)[None, None, :, :]
                cell_a = target_cell(mask_a, grid)
                cell_b = target_cell(mask_b, grid)

                point_a = point_cross_entropy(heatmap_a[0], cell_a.index)
                point_b = point_cross_entropy(heatmap_b[0], cell_b.index)

                probabilities_a = spatial_probabilities(heatmap_a[0])
                probabilities_b = spatial_probabilities(heatmap_b[0])
                masses = counterfactual_masses(
                    probabilities_a, probabilities_b, target_a[0, 0], target_b[0, 0]
                )
                cf = bounded_pair_loss(masses["p_aa"], masses["p_ab"], masses["p_bb"], masses["p_ba"])
                reasoning = 0.5 * (
                    lm_cross_entropy(logits_a, batch_a.labels)
                    + lm_cross_entropy(logits_b, batch_b.labels)
                )
                total = (
                    POINT_REASONING_WEIGHT * reasoning
                    + POINT_CLASSIFICATION_WEIGHT * (point_a + point_b)
                    + POINT_CF_WEIGHT * cf["loss"]
                )

        with _stage("backward"):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            total.backward()

        clipped = None
        if optimizer is not None:
            with _stage("optimizer_step"):
                clipped = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.cfg["optimizer"]["grad_clip_norm"]),
                    )
                )
                optimizer.step()

        # ---- legacy diagnostics: detached, therefore gradient-free (sections 8-9) ----
        with torch.no_grad():
            heatmap_a_1 = heatmap_a.detach().unsqueeze(1) if heatmap_a.dim() == 3 else heatmap_a.detach()
            heatmap_b_1 = heatmap_b.detach().unsqueeze(1) if heatmap_b.dim() == 3 else heatmap_b.detach()
            legacy_a = heatmap_loss(heatmap_a_1, target_a)
            legacy_b = heatmap_loss(heatmap_b_1, target_b)
            legacy_ranking = region_scores(
                heatmap_a.detach()[0], heatmap_b.detach()[0], target_a[0, 0], target_b[0, 0]
            )
            diagnostics = {
                "target_cell_a": cell_a.as_dict(),
                "target_cell_b": cell_b.as_dict(),
                "target_cell_probability_a": target_cell_probability(probabilities_a, cell_a.index),
                "target_cell_probability_b": target_cell_probability(probabilities_b, cell_b.index),
                "target_cell_top1_a": topk_hit(heatmap_a[0], cell_a.index, 1),
                "target_cell_top5_a": topk_hit(heatmap_a[0], cell_a.index, 5),
                "target_cell_top25_a": topk_hit(heatmap_a[0], cell_a.index, 25),
                "target_cell_top1_b": topk_hit(heatmap_b[0], cell_b.index, 1),
                "spatial_entropy_a": spatial_entropy(probabilities_a),
                "spatial_entropy_b": spatial_entropy(probabilities_b),
                "max_non_target_probability_a": max_non_target_probability(
                    probabilities_a, target_a[0, 0]
                ),
                "mean_abs_logit_a": float(heatmap_a.detach().abs().mean()),
                "mean_abs_logit_b": float(heatmap_b.detach().abs().mean()),
                "logit_std_a": float(heatmap_a.detach().std()),
                "logit_std_b": float(heatmap_b.detach().std()),
                "legacy_bce_dice_a": legacy_a["heatmap_raw"],
                "legacy_bce_dice_b": legacy_b["heatmap_raw"],
                "legacy_logit_ranking_margin": legacy_ranking["raw"]["mean_margin"],
                "legacy_terms_require_grad": bool(
                    legacy_a["heatmap"].requires_grad
                    or legacy_b["heatmap"].requires_grad
                    or legacy_ranking["mean_margin"].requires_grad
                ),
            }

        mean_abs_logit = 0.5 * (
            diagnostics["mean_abs_logit_a"] + diagnostics["mean_abs_logit_b"]
        )
        return {
            "losses": {
                "total": float(total.detach()),
                "reasoning_ce": float(reasoning.detach()),
                "point_ce_a": float(point_a.detach()),
                "point_ce_b": float(point_b.detach()),
                "point_ce_sum": float((point_a + point_b).detach()),
                "cf": cf["raw"],
                "cf_a": cf["raw_a"],
                "cf_b": cf["raw_b"],
                "lm_ce": float(reasoning.detach()),
                "mask_bce": diagnostics["legacy_bce_dice_a"] + diagnostics["legacy_bce_dice_b"],
                "mask_dice": 0.0,
                "ground": cf["raw"],
            },
            "region_masses": masses["raw"],
            "pair_preference_pass": masses["pair_preference_pass"],
            "mean_abs_logit": mean_abs_logit,
            "logit_std": 0.5 * (diagnostics["logit_std_a"] + diagnostics["logit_std_b"]),
            "diagnostics": diagnostics,
            "grad_clip_total_norm": clipped,
            "weights": {
                "reasoning": POINT_REASONING_WEIGHT,
                "classification": POINT_CLASSIFICATION_WEIGHT,
                "cf": POINT_CF_WEIGHT,
                "eps": cf["eps"],
            },
        }

    def build_optimizer(self) -> torch.optim.AdamW:
        opt_cfg = self.cfg["optimizer"]
        groups = self.model.trainable_parameter_groups(
            lora_lr=float(opt_cfg["lora_lr"]),
            head_lr=float(opt_cfg["head_lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            decoder_lr=float(opt_cfg.get("decoder_lr", opt_cfg["head_lr"])),
        )
        return torch.optim.AdamW(groups, betas=tuple(opt_cfg["betas"]))

    def build_scheduler_for(self, optimizer, max_steps: int, schedule_cfg: dict):
        """Scheduler described by an explicit config block (Task 6B phases)."""

        schedule = str(schedule_cfg.get("lr_schedule", "none")).lower()
        warmup = int(schedule_cfg.get("warmup_steps", 0))

        def factor(step: int) -> float:
            if schedule == "none":
                return 1.0
            if warmup and step < warmup:
                return max(1e-3, (step + 1) / warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def build_scheduler(self, optimizer, max_steps: int):
        """Optional cosine decay with linear warmup.

        Added after the first two recorded Stage 2 runs, both of which oscillated
        late in training; Task 6A section 12 permits adjusting the recipe once the
        initial result has been recorded, and both initial results are kept in the
        smoke report history.
        """

        schedule = str(self.cfg["optimizer"].get("lr_schedule", "none")).lower()
        warmup = int(self.cfg["optimizer"].get("warmup_steps", 0))

        def factor(step: int) -> float:
            if schedule == "none":
                return 1.0
            if warmup and step < warmup:
                return max(1e-3, (step + 1) / warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def train_step(
        self,
        batch,
        gt_mask,
        features,
        optimizer=None,
        timer=None,
        collect_grad_norms: bool = True,
        clip_grad_foreach: bool | None = None,
    ) -> dict:
        """One forward/backward pass; optionally one optimizer step.

        `gt_mask` is supervision only and is never an input to either model.
        `timer` is an optional Task 6C.5 stage profiler; `None` (the default) adds no
        synchronization and does not change any value.
        `collect_grad_norms` defaults to `True`, preserving behaviour for every caller
        that consumes `grad_norms`. Task 6C's training loop discards that field, so the
        Task 6C.5 pipeline can switch the per-step instrumentation off; the equivalence
        gate proves losses, gradients and parameters are unaffected either way.
        `clip_grad_foreach` defaults to `None`, which is exactly what
        `clip_grad_norm_` does today. Task 6C.6 section 7 benchmarks the foreach
        reduction; the norm, the threshold and the parameter groups are unchanged, but
        the reduction order differs, so a caller must gate the change on equivalence.
        """

        training_cfg = self.cfg["training"]
        batch = batch.to(self.device)
        use_autocast = bool(training_cfg.get("bf16_autocast", True))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast)

        # Task 6C.5: `timer` is an optional stage profiler. When it is None (the
        # default, and the only path the training script uses) the code below is the
        # same computation with a few extra `is not None` checks and no synchronization.
        def _stage(name):
            if timer is None:
                return contextlib.nullcontext()
            return timer.stage(name)

        with _stage("qwen_forward"):
            with autocast:
                output = self.model(batch, features)
        with _stage("loss"):
            with autocast:
                target, supervision_logits, supervision_mode = self.build_mask_supervision(
                    gt_mask, output.mask_logits
                )
                breakdown = combined_loss(
                    output.lm_logits, batch.labels, supervision_logits, target, self.loss_weights
                )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with _stage("backward"):
            breakdown.total.backward()

        grad_norms = gradient_norms(self.model) if collect_grad_norms else {}
        clipped = None
        if optimizer is not None:
            with _stage("optimizer_step"):
                clipped = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.cfg["optimizer"]["grad_clip_norm"]),
                        foreach=clip_grad_foreach,
                    )
                )
                optimizer.step()

        return {
            "losses": breakdown.as_dict(),
            "grad_norms": grad_norms,
            "grad_clip_total_norm": clipped,
            "output": output,
            "mask_target": target,
            "mask_supervision": supervision_mode,
        }

    def build_mask_supervision(self, gt_mask, mask_logits):
        """Return (target, logits, mode) for the mask objective.

        Task 6A section 11.4 allows either resizing the ground truth to the logit
        resolution or post-processing the logits to the original size. This is a
        configuration switch because the choice matters for small objects: SAM2
        emits 256x256 logits, so a 148 px target occupies only ~37 px there,
        which caps the achievable original-resolution IoU. Supervising at the
        original 512x512 resolution removes that ceiling.

        `mode` is recorded so every run states which objective it used.
        """

        ground_truth = torch.as_tensor(np.asarray(gt_mask)).float()
        if ground_truth.dim() == 2:
            ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)
        original_size = (int(ground_truth.shape[-2]), int(ground_truth.shape[-1]))

        if bool(self.cfg["training"].get("train_at_original_resolution", False)):
            from .metrics import upsample_logits

            logits = upsample_logits(mask_logits, original_size, mode="bilinear")
            return ground_truth.to(self.device), logits, "original_resolution"

        target = downsample_target(ground_truth, mask_logits.shape[-2:]).to(self.device)
        return target, mask_logits, "logit_resolution"


def downsample_target(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    from .metrics import downsample_mask

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return downsample_mask(mask.float(), size)


def gradient_norms(model: BuildReasonSegMvp) -> dict:
    """Per-group gradient norms used by the Stage 1 success criteria."""

    def _norm(predicate) -> float:
        total = 0.0
        found = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if predicate(name):
                total += float(parameter.grad.detach().float().norm() ** 2)
                found += 1
        return round(total**0.5, 6), found

    lora, n_lora = _norm(lambda n: "lora_" in n)
    token, n_token = _norm(lambda n: "trainable_tokens" in n or "token_row" in n)
    projection, n_projection = _norm(lambda n: n.startswith("projection."))
    decoder, n_decoder = _norm(lambda n: n.startswith("sam.sam_mask_decoder."))
    encoder, n_encoder = _norm(lambda n: n.startswith("sam.image_encoder."))
    return {
        "lora": {"norm": lora, "tensors": n_lora},
        "seg_token": {"norm": token, "tensors": n_token},
        "projection": {"norm": projection, "tensors": n_projection},
        "sam_mask_decoder": {"norm": decoder, "tensors": n_decoder},
        "sam_image_encoder": {"norm": encoder, "tensors": n_encoder},
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def set_phase_trainables(model: BuildReasonSegMvp, phase: str) -> dict:
    """Two-phase training schedule from Task 6B section 12.

    * ``A`` -- language-format warm-up: only the text LoRA adapters and the
      trainable `[SEG]` row are optimised. The projection MLP and the SAM2 mask
      decoder are **frozen**, so the mask pathway is untouched during warm-up.
    * ``B`` -- joint segmentation training: the projection MLP and the SAM2 mask
      decoder join the optimiser.

    The Qwen vision tower, the Qwen base weights, the SAM2 image encoder, the
    prompt encoder and the memory modules stay frozen in both phases.
    """

    phase = phase.upper()
    if phase not in ("A", "B"):
        raise ValueError(f"phase must be 'A' or 'B', got {phase!r}")

    for parameter in model.projection.parameters():
        parameter.requires_grad_(phase == "B")
    for parameter in model.sam.sam_mask_decoder.parameters():
        parameter.requires_grad_(phase == "B")

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    return {
        "phase": phase,
        "trainable_tensors": len(trainable),
        "frozen_tensors": len(frozen),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "projection_trainable": any(p.requires_grad for p in model.projection.parameters()),
        "sam_mask_decoder_trainable": any(
            p.requires_grad for p in model.sam.sam_mask_decoder.parameters()
        ),
        "qwen_visual_trainable": [
            name
            for name in trainable
            if any(marker in name for marker in ("visual", "vision_tower", "vision_model"))
        ],
    }


def build_runtime(cfg: dict | None = None, device: str = "cuda", verbose: bool = True,
                  feature_cache_images: int | None = None,
                  deterministic_strict: bool | None = None) -> MvpRuntime:
    cfg = cfg or load_config()
    deterministic = bool(cfg.get("training", {}).get("deterministic", False))
    strict = (
        bool(cfg.get("training", {}).get("deterministic_strict", True))
        if deterministic_strict is None
        else bool(deterministic_strict)
    )
    if deterministic:
        determinism = enable_determinism(int(cfg["seed"]), strict=strict)
    else:
        set_seed(int(cfg["seed"]))
        determinism = {"requested": False, "seed": int(cfg["seed"])}
    determinism["strict_effective"] = bool(deterministic and strict)

    cache_root = REPO_ROOT / cfg["paths"]["hf_cache"]
    reports: dict = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if verbose:
        print("[runtime] loading Qwen ...", flush=True)
    rss_before = rss_gib()
    processor, qwen = load_qwen(
        cfg["models"]["qwen_model_id"],
        cache_dir=str(cache_root / "hub"),
        dtype=torch.bfloat16,
        device=device,
        attn_implementation=cfg["models"]["qwen_attn_implementation"],
    )
    rss_after_qwen = rss_gib()

    tokenizer = processor.tokenizer
    token_setup = setup_seg_token(qwen, tokenizer)
    reports["token"] = token_setup.as_dict()

    # Task 6E parts D/E: add `[BOX]` + `<loc_000>..<loc_{B-1}>` and make their rows trainable
    # alongside `[SEG]`. `cfg["spatial_tokens"]["bins"]` is absent (or null) for every
    # Task 6A-6D configuration, which keeps those paths bit-identical.
    spatial_cfg = cfg.get("spatial_tokens") or {}
    spatial_bins = spatial_cfg.get("bins")
    spatial_setup = None
    extra_token_ids: list[int] = []
    if spatial_bins:
        from .spatial_tokens import add_spatial_tokens, location_token_ids

        spatial_setup = add_spatial_tokens(qwen, tokenizer, int(spatial_bins))
        extra_token_ids = location_token_ids(spatial_setup)

    # Task 6F: add exactly ONE active query token `[BOX]` (no `<loc_*>` vocabulary) and make
    # its row trainable alongside `[SEG]`. Mutually exclusive with the Task 6E spatial block;
    # absent from every Task 6A-6E configuration. Task 6G reuses this exact setup.
    box_cfg = cfg.get("box_query") or {}
    box_setup = None
    if box_cfg.get("enabled"):
        if spatial_setup is not None:
            raise RuntimeError("cfg enables both spatial_tokens and box_query; Tasks 6F/6G forbid it")
        from .box_query import add_box_query_token

        box_setup = add_box_query_token(qwen, tokenizer, str(box_cfg.get("box_token", "[BOX]")))
        extra_token_ids = [int(box_setup.box_token_id)]

    # Task 6G: dense spatial grounding reuses the [BOX] query token and adds a selected grid.
    dense_cfg = cfg.get("dense_grounding") or {}
    dense_grid = None
    if dense_cfg.get("enabled"):
        if box_setup is None:
            raise RuntimeError("cfg enables dense_grounding without box_query; the [BOX] query is required")
        dense_grid = int(dense_cfg.get("grid"))
        if dense_grid not in (64, 128, 256):
            raise RuntimeError(f"dense_grounding grid must be one of 64/128/256, got {dense_grid}")

    if cfg["token"].get("prefer_peft_trainable_token_indices", True):
        qwen, token_holder, lora_report = attach_lora(
            qwen,
            token_setup.seg_token_id,
            rank=int(cfg["lora"]["rank"]),
            alpha=int(cfg["lora"]["alpha"]),
            dropout=float(cfg["lora"]["dropout"]),
            extra_token_ids=extra_token_ids,
        )
    else:
        qwen, token_holder, lora_report = attach_lora(
            qwen,
            token_setup.seg_token_id,
            rank=int(cfg["lora"]["rank"]),
            alpha=int(cfg["lora"]["alpha"]),
            dropout=float(cfg["lora"]["dropout"]),
            prefer_peft_token_indices=False,
            extra_token_ids=extra_token_ids,
        )
    reports["lora"] = lora_report.as_dict()
    if box_setup is not None:
        reports["box_query"] = {
            **box_setup.as_dict(),
            "trainable_token_ids": list(lora_report.trainable_token_ids),
            "token_mechanism": lora_report.token_mechanism,
        }
    if dense_grid is not None:
        reports["dense_grounding"] = {"enabled": True, "grid": int(dense_grid)}

    if cfg["training"].get("gradient_checkpointing", False):
        qwen.gradient_checkpointing_enable()
        if hasattr(qwen, "enable_input_require_grads"):
            qwen.enable_input_require_grads()
        reports["gradient_checkpointing"] = True

    if verbose:
        print("[runtime] loading SAM2.1 Base+ ...", flush=True)
    sam_config = cfg["models"]["sam2_config_name"]
    sam_ckpt = str(REPO_ROOT / cfg["paths"]["models_dir"] / cfg["models"]["sam2_checkpoint_file"])
    sam, sam_report = load_sam2(
        sam_config,
        sam_ckpt,
        device=device,
        source_revision=sam_source_revision(REPO_ROOT / cfg["paths"]["sam2_source"]),
    )
    freeze_report = apply_mvp_freeze_policy(sam)
    reports["sam2"] = sam_report.as_dict()
    reports["sam2_freeze"] = freeze_report.as_dict()

    hidden_size = int(qwen.config.text_config.hidden_size) if hasattr(qwen.config, "text_config") else int(qwen.config.hidden_size)
    projection = ProjectionMLP(
        in_dim=hidden_size,
        out_dim=int(sam_report.prompt_embed_dim),
        hidden_dim=cfg["projection"].get("hidden_dim"),
        dropout=float(cfg["projection"].get("dropout", 0.0)),
    ).to(device)

    bridge = str(cfg.get("bridge", {}).get("mode", "centre"))
    model = BuildReasonSegMvp(
        qwen=qwen,
        sam=sam,
        projection=projection,
        seg_token_id=token_setup.seg_token_id,
        token_holder=token_holder,
        bridge=bridge,
    ).to(device)

    reports["bridge"] = bridge
    reports["determinism"] = determinism
    reports["params"] = parameter_report(model, token_setup.seg_token_id, token_holder)
    reports["qwen_hidden_size"] = hidden_size
    reports["sam_prompt_embed_dim"] = int(sam_report.prompt_embed_dim)
    reports["projection"] = {
        "in_dim": projection.in_dim,
        "hidden_dim": projection.hidden_dim,
        "out_dim": projection.out_dim,
        "params": sum(p.numel() for p in projection.parameters()),
    }
    reports["image_budget"] = apply_image_budget(
        processor,
        cfg["preprocessing"]["image_budget"].get("min_pixels"),
        cfg["preprocessing"]["image_budget"].get("max_pixels"),
    )
    reports["ram"] = {"before_qwen_gib": rss_before, "after_qwen_gib": rss_after_qwen}
    reports["vram_after_load"] = vram()

    sam_encoder = Sam2Encoder(sam)
    cache_images = int(
        feature_cache_images
        if feature_cache_images is not None
        else cfg.get("training", {}).get(
            "feature_cache_images", cfg.get("stage2_overfit", {}).get("feature_cache_images", 160)
        )
    )
    return MvpRuntime(
        cfg=cfg,
        processor=processor,
        tokenizer=tokenizer,
        qwen=qwen,
        sam=sam,
        projection=projection,
        model=model,
        sam_encoder=sam_encoder,
        feature_cache=Sam2FeatureCache(
            sam_encoder,
            max_images=cache_images,
            share_constant_features=bool(
                cfg.get("training", {}).get("feature_cache_share_constants", True)
            ),
        ),
        device=device,
        reports=reports,
        loss_weights=LossWeights(**{k: float(v) for k, v in cfg["loss"].items() if isinstance(v, (int, float))}),
        spatial_setup=spatial_setup,
        box_setup=box_setup,
        dense_grid=dense_grid,
    )


def save_json(path: Path | str, payload: dict) -> None:
    write_json(Path(path), payload)


def timed(label: str):
    class _Timer:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *exc):
            self.seconds = round(time.time() - self.start, 2)
            print(f"[timing] {label}: {self.seconds}s", flush=True)
            return False

    return _Timer()
