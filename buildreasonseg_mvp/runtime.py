"""Shared runtime for the Task 6A stage scripts.

Keeps model assembly, preprocessing, and one training step in one place so the
four stage scripts stay thin and cannot drift apart.
"""

from __future__ import annotations

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
from .losses import LossWeights, combined_loss
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

    @property
    def bridge(self) -> str:
        """The SAM sparse-prompt bridge this runtime's model was built with."""

        return getattr(self.model, "bridge", "centre")

    # -- batches ---------------------------------------------------------

    def prepare(self, sample: data_mod.Sample) -> tuple[TeacherForcedBatch, np.ndarray]:
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

    # -- optimisation ----------------------------------------------------

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

    def train_step(self, batch, gt_mask, features, optimizer=None) -> dict:
        """One forward/backward pass; optionally one optimizer step.

        `gt_mask` is supervision only and is never an input to either model.
        """

        training_cfg = self.cfg["training"]
        batch = batch.to(self.device)
        use_autocast = bool(training_cfg.get("bf16_autocast", True))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast)

        with autocast:
            output = self.model(batch, features)
            target, supervision_logits, supervision_mode = self.build_mask_supervision(
                gt_mask, output.mask_logits
            )
            breakdown = combined_loss(
                output.lm_logits, batch.labels, supervision_logits, target, self.loss_weights
            )

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()

        grad_norms = gradient_norms(self.model)
        clipped = None
        if optimizer is not None:
            clipped = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    float(self.cfg["optimizer"]["grad_clip_norm"]),
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

    if cfg["token"].get("prefer_peft_trainable_token_indices", True):
        qwen, token_holder, lora_report = attach_lora(
            qwen,
            token_setup.seg_token_id,
            rank=int(cfg["lora"]["rank"]),
            alpha=int(cfg["lora"]["alpha"]),
            dropout=float(cfg["lora"]["dropout"]),
        )
    else:
        qwen, token_holder, lora_report = attach_lora(
            qwen,
            token_setup.seg_token_id,
            rank=int(cfg["lora"]["rank"]),
            alpha=int(cfg["lora"]["alpha"]),
            dropout=float(cfg["lora"]["dropout"]),
            prefer_peft_token_indices=False,
        )
    reports["lora"] = lora_report.as_dict()

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
