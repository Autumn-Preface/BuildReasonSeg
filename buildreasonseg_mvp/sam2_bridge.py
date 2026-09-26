"""Vanilla SAM2.1 bridge for the MVP.

No Sa2VA custom code is used anywhere. The bridge is deliberately small and
project-owned:

    Qwen [SEG] hidden
        -> Projection MLP
        -> SAM prompt-embedding dimension
        -> ONE sparse prompt embedding
        -> vanilla SAM2.1 mask decoder

Frozen: image encoder / backbone, memory encoder + memory attention, prompt
encoder. Trainable: the projection MLP and the SAM2 mask decoder.

The official `SAM2ImagePredictor` performs the image transform and the frozen
encoder forward, so the visual path is byte-for-byte official behaviour. The
official prompt encoder is used only for the positional encoding of the sparse
prompt slot and for the no-mask dense embedding. No ground-truth point, box,
mask, centroid, bounding box or component id is ever supplied.

Reference: SAM2 revision 2b90b9f5ceec907a1c18123530e92e794ad901a4,
`sam2/modeling/sam2_base.py::_forward_sam_heads` and
`sam2/sam2_image_predictor.py::set_image`.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn as nn

#: Centroid of the tile, used as a content-free positional anchor for the sparse
#: prompt slot. A constant, never derived from the annotation.
PROMPT_ANCHOR_XY = (0.5, 0.5)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class Sam2LoadReport:
    config_file: str = ""
    checkpoint: str = ""
    source_revision: str = ""
    image_size: int = 0
    backbone_stride: int = 0
    prompt_embed_dim: int = 0
    image_embedding_size: tuple[int, ...] = ()
    mask_input_size: tuple[int, ...] = ()
    use_high_res_features_in_sam: bool = False
    directly_add_no_mem_embed: bool = False
    total_params: int = 0
    mask_decoder_params: int = 0
    image_encoder_params: int = 0
    prompt_encoder_params: int = 0
    memory_attention_params: int = 0
    memory_encoder_params: int = 0
    loaded: bool = False

    def as_dict(self) -> dict:
        return {
            "config_file": self.config_file,
            "checkpoint": self.checkpoint,
            "source_revision": self.source_revision,
            "image_size": self.image_size,
            "backbone_stride": self.backbone_stride,
            "prompt_embed_dim": self.prompt_embed_dim,
            "image_embedding_size": list(self.image_embedding_size),
            "mask_input_size": list(self.mask_input_size),
            "use_high_res_features_in_sam": self.use_high_res_features_in_sam,
            "directly_add_no_mem_embed": self.directly_add_no_mem_embed,
            "total_params": self.total_params,
            "mask_decoder_params": self.mask_decoder_params,
            "image_encoder_params": self.image_encoder_params,
            "prompt_encoder_params": self.prompt_encoder_params,
            "memory_attention_params": self.memory_attention_params,
            "memory_encoder_params": self.memory_encoder_params,
            "loaded": self.loaded,
        }


def load_sam2(
    config_file: str,
    checkpoint: str,
    device: str = "cuda",
    source_revision: str = "",
) -> tuple[nn.Module, Sam2LoadReport]:
    """Build vanilla SAM2.1 from the official package.

    `SAM2_BUILD_CUDA=0` must already be set in the process environment: Task 6A
    intentionally skips the optional compiled CUDA extension, which only affects
    mask hole/sprinkle post-processing.
    """

    from sam2.build_sam import build_sam2

    sam = build_sam2(config_file, checkpoint, device=device)
    sam.eval()

    def _params(module) -> int:
        return sum(p.numel() for p in module.parameters()) if module is not None else 0

    report = Sam2LoadReport(
        config_file=config_file,
        checkpoint=checkpoint,
        source_revision=source_revision,
        image_size=int(getattr(sam, "image_size", 0)),
        backbone_stride=int(getattr(sam, "backbone_stride", 0)),
        prompt_embed_dim=int(sam.sam_prompt_encoder.embed_dim),
        image_embedding_size=tuple(sam.sam_prompt_encoder.image_embedding_size),
        mask_input_size=tuple(getattr(sam.sam_prompt_encoder, "mask_input_size", ()) or ()),
        use_high_res_features_in_sam=bool(getattr(sam, "use_high_res_features_in_sam", False)),
        directly_add_no_mem_embed=bool(getattr(sam, "directly_add_no_mem_embed", False)),
        total_params=_params(sam),
        mask_decoder_params=_params(sam.sam_mask_decoder),
        image_encoder_params=_params(sam.image_encoder),
        prompt_encoder_params=_params(sam.sam_prompt_encoder),
        memory_attention_params=_params(getattr(sam, "memory_attention", None)),
        memory_encoder_params=_params(getattr(sam, "memory_encoder", None)),
        loaded=True,
    )
    return sam, report


# --------------------------------------------------------------------------
# Freeze policy
# --------------------------------------------------------------------------


@dataclass
class FreezeReport:
    frozen_modules: list[str] = field(default_factory=list)
    trainable_modules: list[str] = field(default_factory=list)
    frozen_params: int = 0
    trainable_params: int = 0

    def as_dict(self) -> dict:
        return {
            "frozen_modules": self.frozen_modules,
            "trainable_modules": self.trainable_modules,
            "frozen_params": self.frozen_params,
            "trainable_params": self.trainable_params,
        }


def apply_mvp_freeze_policy(sam: nn.Module) -> FreezeReport:
    """Freeze **every** SAM2 parameter except the mask decoder.

    Note: `SAM2Base` also owns top-level `nn.Parameter`s that are not children
    modules (`no_mem_embed`, `no_mem_pos_enc`, `maskmem_tpos_enc`,
    `no_obj_ptr`, `no_obj_embed_spatial`). Iterating `named_children()` alone
    would leave those trainable, so the policy freezes the whole module first and
    then unfreezes exactly the mask decoder.
    """

    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    for parameter in sam.sam_mask_decoder.parameters():
        parameter.requires_grad_(True)

    trainable_names = {"sam_mask_decoder"}
    frozen_modules = sorted(name for name, _ in sam.named_children() if name not in trainable_names)
    trainable_modules = sorted(trainable_names)

    # top-level parameters that are not children modules
    child_param_ids = {
        id(p) for _, module in sam.named_children() for p in module.parameters()
    }
    top_level = sorted(
        name
        for name, parameter in sam.named_parameters()
        if id(parameter) not in child_param_ids
    )
    frozen_modules.extend(f"<top-level parameter> {name}" for name in top_level)

    return FreezeReport(
        frozen_modules=sorted(frozen_modules),
        trainable_modules=trainable_modules,
        frozen_params=sum(p.numel() for p in sam.parameters() if not p.requires_grad),
        trainable_params=sum(p.numel() for p in sam.parameters() if p.requires_grad),
    )


# --------------------------------------------------------------------------
# Frozen feature extraction (official predictor path)
# --------------------------------------------------------------------------


@dataclass
class Sam2Features:
    image_embeddings: torch.Tensor
    high_res_features: list[torch.Tensor] | None
    image_pe: torch.Tensor
    dense_no_mask_embedding: torch.Tensor
    source_shape: tuple[int, ...] = ()
    transformed_shape: tuple[int, ...] = ()
    embedding_shape: tuple[int, ...] = ()
    high_res_shapes: list[tuple[int, ...]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source_shape": list(self.source_shape),
            "transformed_shape": list(self.transformed_shape),
            "embedding_shape": list(self.embedding_shape),
            "high_res_shapes": [list(shape) for shape in self.high_res_shapes],
            "image_pe_shape": list(self.image_pe.shape),
            "dense_no_mask_shape": list(self.dense_no_mask_embedding.shape),
        }

    def detach(self) -> "Sam2Features":
        return Sam2Features(
            image_embeddings=self.image_embeddings.detach(),
            high_res_features=(
                [f.detach() for f in self.high_res_features] if self.high_res_features else None
            ),
            image_pe=self.image_pe.detach(),
            dense_no_mask_embedding=self.dense_no_mask_embedding.detach(),
            source_shape=self.source_shape,
            transformed_shape=self.transformed_shape,
            embedding_shape=self.embedding_shape,
            high_res_shapes=self.high_res_shapes,
        )


class Sam2Encoder(nn.Module):
    """Frozen official SAM2 image path, wrapped so it can be cached."""

    def __init__(self, sam: nn.Module) -> None:
        super().__init__()
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.sam = sam
        self.predictor = SAM2ImagePredictor(sam)

    @torch.no_grad()
    def encode(self, image_hwc_uint8) -> Sam2Features:
        """Official transform + frozen encoder, exactly as `set_image` does."""

        import numpy as np

        image = np.asarray(image_hwc_uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected HWC RGB uint8 image, got {image.shape}")

        source_shape = tuple(image.shape)
        transformed = self.predictor._transforms(image)[None, ...].to(self.predictor.device)
        self.predictor._orig_hw = [source_shape[:2]]

        backbone_out = self.sam.forward_image(transformed)
        _backbone_out, vision_feats, _vision_pos, feat_sizes = self.sam._prepare_backbone_features(
            backbone_out
        )
        if self.sam.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam.no_mem_embed

        batch = vision_feats[-1].shape[1]
        image_embeddings = (
            vision_feats[-1].permute(1, 2, 0).view(batch, self.sam.hidden_dim, *feat_sizes[-1])
        )

        if len(vision_feats) > 1:
            high_res = [
                level.permute(1, 2, 0).view(level.size(1), level.size(2), *size)
                for level, size in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res = None

        dense_no_mask = self.sam.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch, -1, image_embeddings.shape[-2], image_embeddings.shape[-1]
        )

        return Sam2Features(
            image_embeddings=image_embeddings,
            high_res_features=high_res,
            image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
            dense_no_mask_embedding=dense_no_mask,
            source_shape=source_shape,
            transformed_shape=tuple(transformed.shape),
            embedding_shape=tuple(image_embeddings.shape),
            high_res_shapes=[tuple(level.shape) for level in (high_res or [])],
        )


class Sam2FeatureCache:
    """Bounded cache of detached frozen SAM2 features, keyed by image id.

    Task 6A section 16.6 and Task 6B section 18 both allow this because the
    encoder is frozen. Two properties matter at Task 6B scale (480 training images
    rather than 10):

    * a **hard cap** on the number of cached images, with least-recently-used
      eviction, so the cache cannot grow without bound; and
    * tensors are held on **CPU** and moved to the device on use, so the cache
      does not consume VRAM alongside the model.

    Qwen hidden states are never cached.
    """

    def __init__(self, encoder: Sam2Encoder, max_images: int = 160) -> None:
        self.encoder = encoder
        self.max_images = int(max_images)
        self._store: "OrderedDict[str, Sam2Features]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.device = next(encoder.sam.parameters()).device

    def _to_cpu(self, features: Sam2Features) -> Sam2Features:
        def move(tensor):
            return tensor.detach().to("cpu")

        return Sam2Features(
            image_embeddings=move(features.image_embeddings),
            high_res_features=(
                [move(level) for level in features.high_res_features]
                if features.high_res_features
                else None
            ),
            image_pe=move(features.image_pe),
            dense_no_mask_embedding=move(features.dense_no_mask_embedding),
            source_shape=features.source_shape,
            transformed_shape=features.transformed_shape,
            embedding_shape=features.embedding_shape,
            high_res_shapes=features.high_res_shapes,
        )

    def _to_device(self, features: Sam2Features) -> Sam2Features:
        def move(tensor):
            return tensor.to(self.device, non_blocking=True)

        return Sam2Features(
            image_embeddings=move(features.image_embeddings),
            high_res_features=(
                [move(level) for level in features.high_res_features]
                if features.high_res_features
                else None
            ),
            image_pe=move(features.image_pe),
            dense_no_mask_embedding=move(features.dense_no_mask_embedding),
            source_shape=features.source_shape,
            transformed_shape=features.transformed_shape,
            embedding_shape=features.embedding_shape,
            high_res_shapes=features.high_res_shapes,
        )

    def get(self, image_id: str, image_hwc_uint8) -> tuple[Sam2Features, bool]:
        cached = self._store.get(image_id)
        if cached is not None:
            self._store.move_to_end(image_id)
            self.hits += 1
            return self._to_device(cached), True

        self.misses += 1
        features = self._to_cpu(self.encoder.encode(image_hwc_uint8))
        self._store[image_id] = features
        while len(self._store) > self.max_images:
            self._store.popitem(last=False)
            self.evictions += 1
        return self._to_device(features), False

    def stats(self) -> dict:
        return {
            "max_images": self.max_images,
            "cached_images": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "storage": "cpu_tensors",
        }

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# --------------------------------------------------------------------------
# Projection + sparse prompt decoding
# --------------------------------------------------------------------------


class ProjectionMLP(nn.Module):
    """`[SEG]` hidden state -> SAM prompt-embedding dimension.

    Kept as its own module so a later Spatial Relation Encoder can be inserted
    between this and the mask decoder without touching the language-model path.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(out_dim, in_dim // 2)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden.to(self.net[0].weight.dtype))


@dataclass
class MaskDecodeResult:
    low_res_logits: torch.Tensor
    iou_prediction: torch.Tensor
    sparse_prompt_shape: tuple[int, ...] = ()

    def as_dict(self) -> dict:
        return {
            "low_res_logits_shape": list(self.low_res_logits.shape),
            "iou_prediction_shape": list(self.iou_prediction.shape),
            "sparse_prompt_shape": list(self.sparse_prompt_shape),
        }


def build_sparse_prompt(sam: nn.Module, projected: torch.Tensor) -> torch.Tensor:
    """One sparse prompt embedding per sample, in SAM's prompt space.

    The official prompt encoder is consulted for a single real point prompt at the
    tile centre, which supplies (a) a positional encoding for the sparse slot and
    (b) the no-mask dense embedding. The projected language vector is added to the
    real point slot. The anchor coordinate is a constant; nothing is derived from
    the annotation.
    """

    batch = projected.shape[0]
    device = projected.device
    anchor = torch.tensor([[list(PROMPT_ANCHOR_XY)]], device=device, dtype=torch.float32).expand(
        batch, -1, -1
    )
    labels = torch.ones((batch, 1), device=device, dtype=torch.int32)

    sparse, _dense = sam.sam_prompt_encoder(points=(anchor, labels), boxes=None, masks=None)
    sparse = sparse.clone()
    sparse[:, 0, :] = sparse[:, 0, :] + projected.to(sparse.dtype)
    return sparse


def decode_mask(
    sam: nn.Module,
    features: Sam2Features,
    projected: torch.Tensor,
    multimask_output: bool = False,
) -> MaskDecodeResult:
    """Decode a mask from the projected `[SEG]` embedding."""

    sparse = build_sparse_prompt(sam, projected)
    low_res_logits, iou_prediction, _tokens, _obj_score = sam.sam_mask_decoder(
        image_embeddings=features.image_embeddings,
        image_pe=features.image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=features.dense_no_mask_embedding,
        multimask_output=multimask_output,
        repeat_image=False,
        high_res_features=features.high_res_features,
    )
    return MaskDecodeResult(
        low_res_logits=low_res_logits,
        iou_prediction=iou_prediction,
        sparse_prompt_shape=tuple(sparse.shape),
    )


def sam_source_revision(source_dir: str | os.PathLike) -> str:
    """Best-effort git revision of the installed SAM2 source tree."""

    import subprocess
    from pathlib import Path

    path = Path(source_dir)
    if not path.is_dir():
        return ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, timeout=20
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""
