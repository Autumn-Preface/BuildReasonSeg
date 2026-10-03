"""Task 6N sections 9-12 — the frozen-decoder family and its data path.

One common decoder family with exactly three controlled variants:

===========  ==============================================  ==================
variant      fusion input                                    first conv in-ch
===========  ==============================================  ==================
``N-B0``     ``visual_128`` + ``relation_embed``             144
``N-B1``     ``visual_128`` + ``M_ref_down`` + relation      145
``N-B2``     ``visual_128`` + ``M_ref_down`` + ``P_rel``     146
===========  ==============================================  ==================

Missing channels are **not** padded to equalise parameters; the exact parameter counts are reported.
The loss is exactly ``BCEWithLogitsLoss + DiceLoss`` (the project's canonical Dice). There is no
attention, transformer, graph block, extra MLP, relation loss, counterfactual loss or focal loss.

The visual branch is the **frozen SAM2.1 Hiera Base+ image-embedding path** reused from Task 6C.7 /
6I (`buildreasonseg.runtime._frozen.mvp.sam2_bridge.Sam2Encoder`), i.e. ``C = 256`` at ``64 x 64``. It is never
trained and never replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from buildreasonseg.runtime._frozen.mvp.geometric_relation_field import (
    DEFAULT_FIELD_CONFIG,
    DIRECTIONAL_PROGRAMS,
    FieldConfig,
    PROGRAM_TO_RELATION,
    RELATION_TO_INDEX,
    geometric_relation_field,
    reference_centroid,
    resize_soft_mask,
)
from buildreasonseg.runtime._frozen.mvp.losses import mask_bce_with_logits, mask_soft_dice

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Frozen SAM2 lineage reused from Task 6C.7 / 6I.
SAM2_CHECKPOINT = REPO_ROOT / "local_cache" / "models" / "sam2.1_hiera_base_plus.pt"
SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_REPO_ID = "facebook/sam2.1-hiera-base-plus"
SAM2_SOURCE_REVISION_EXPECTED = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
WHU_SOURCE_ROOT = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")

VARIANTS: tuple[str, ...] = ("B0", "B1", "B2")
#: Task 6O section 8-9: two further causal-decomposition variants of the same decoder family.
TASK6O_VARIANTS: tuple[str, ...] = ("B3", "B4")
ALL_VARIANTS: tuple[str, ...] = VARIANTS + TASK6O_VARIANTS
VARIANT_LABELS = {
    "B0": "N-B0", "B1": "N-B1", "B2": "N-B2", "B3": "N-B3", "B4": "N-B4",
}
VARIANT_USES_REFERENCE = {
    "B0": False, "B1": True, "B2": True,
    # Task 6O: neither B3 nor B4 receives the direct reference-mask channel.
    "B3": False, "B4": False,
}
VARIANT_USES_FIELD = {
    "B0": False, "B1": False, "B2": True,
    # B3 concatenates the raw field; B4 projects the field 1 -> 128 first (section 9).
    "B3": True, "B4": False,
}
#: B4 is the geometry-only control: it receives no RGB-derived visual feature at all.
VARIANT_USES_VISUAL = {"B0": True, "B1": True, "B2": True, "B3": True, "B4": False}


# --------------------------------------------------------------------------- decoder


@dataclass
class DecoderConfig:
    visual_channels: int = 256
    width: int = 128
    mid: int = 64
    relation_dim: int = 16
    relation_count: int = 4
    norm_groups: int = 8
    logit_size: tuple[int, int] = (64, 64)


def first_conv_in_channels(variant: str, config: DecoderConfig) -> int:
    """Fusion-trunk input width. B3 = 128 + 1 + 16 = 145; B4 = 128 + 16 = 144."""

    channels = config.width + config.relation_dim
    if VARIANT_USES_REFERENCE[variant]:
        channels += 1
    if VARIANT_USES_FIELD[variant]:
        channels += 1
    return channels


class RelationMaskDecoder(nn.Module):
    """Sections 9 (Task 6N) and 8-9 (Task 6O): visual/field projection → relation embedding → trunk.

    B0/B1/B2 are the frozen Task 6N variants and keep their exact structure and parameter names.
    B3 (visual + field + relation, no direct reference channel) and B4 (field + relation only, no
    visual) are the Task 6O causal-decomposition variants.
    """

    def __init__(self, variant: str, config: DecoderConfig | None = None) -> None:
        super().__init__()
        if variant not in ALL_VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {ALL_VARIANTS}")
        self.variant = variant
        self.config = config or DecoderConfig()

        if VARIANT_USES_VISUAL[variant]:
            self.project = nn.Sequential(
                nn.Conv2d(self.config.visual_channels, self.config.width, kernel_size=1),
                nn.GroupNorm(self.config.norm_groups, self.config.width),
                nn.GELU(),
            )
        else:
            # Task 6O section 9: B4 projects the single-channel field to 128 channels instead.
            self.field_project = nn.Sequential(
                nn.Conv2d(1, self.config.width, kernel_size=1),
                nn.GroupNorm(self.config.norm_groups, self.config.width),
                nn.GELU(),
            )
        self.relation_embedding = nn.Embedding(self.config.relation_count, self.config.relation_dim)
        nn.init.normal_(self.relation_embedding.weight, mean=0.0, std=0.02)

        channels = first_conv_in_channels(variant, self.config)
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, self.config.width, kernel_size=3, padding=1),
            nn.GroupNorm(self.config.norm_groups, self.config.width),
            nn.GELU(),
            nn.Conv2d(self.config.width, self.config.mid, kernel_size=3, padding=1),
            nn.GroupNorm(self.config.norm_groups, self.config.mid),
            nn.GELU(),
            nn.Conv2d(self.config.mid, 1, kernel_size=1),
        )

    def forward(
        self,
        visual: torch.Tensor | None,
        relation_index: torch.Tensor,
        mask_ref_down: torch.Tensor | None = None,
        field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits at the native feature resolution (``h, w``).

        Extra tensors that a variant does not use are simply ignored, so the "not used" property is
        directly testable. ``mask_ref_down`` is never read by B3 or B4, and B4 never reads ``visual``.
        """

        if VARIANT_USES_VISUAL[self.variant]:
            if visual is None:
                raise ValueError(f"{self.variant} requires the frozen visual feature")
            projected = self.project(visual)
            batch, _, height, width = projected.shape
        else:
            if field is None:
                raise ValueError(f"{self.variant} requires the relation field")
            projected = self.field_project(field)
            batch, _, height, width = projected.shape

        embedding = self.relation_embedding(relation_index).view(batch, -1, 1, 1)
        embedding = embedding.expand(batch, self.config.relation_dim, height, width)

        parts = [projected]
        if VARIANT_USES_REFERENCE[self.variant]:
            if mask_ref_down is None:
                raise ValueError(f"{self.variant} requires `mask_ref_down`")
            parts.append(mask_ref_down)
        if VARIANT_USES_FIELD[self.variant]:
            if field is None:
                raise ValueError(f"{self.variant} requires `field`")
            parts.append(field)
        parts.append(embedding)
        return self.trunk(torch.cat(parts, dim=1))

    def parameter_report(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        by_module = {
            "relation_embedding": int(sum(p.numel() for p in self.relation_embedding.parameters())),
            "trunk": int(sum(p.numel() for p in self.trunk.parameters())),
        }
        if VARIANT_USES_VISUAL[self.variant]:
            by_module["project"] = int(sum(p.numel() for p in self.project.parameters()))
        else:
            by_module["field_project"] = int(sum(p.numel() for p in self.field_project.parameters()))
        return {
            "variant": self.variant,
            "label": VARIANT_LABELS[self.variant],
            "first_conv_in_channels": first_conv_in_channels(self.variant, self.config),
            "uses_visual": VARIANT_USES_VISUAL[self.variant],
            "uses_reference_mask": VARIANT_USES_REFERENCE[self.variant],
            "uses_relation_field": VARIANT_USES_FIELD[self.variant],
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "by_module": by_module,
        }


# --------------------------------------------------------------------------- loss


def task6n_loss(logits: torch.Tensor, target: torch.Tensor) -> dict:
    """Exactly `BCEWithLogitsLoss + DiceLoss` (section 11), at the given resolution."""

    bce = mask_bce_with_logits(logits, target)
    dice = mask_soft_dice(logits, target)
    return {"loss": bce + dice, "bce": bce.detach(), "dice": dice.detach()}


def upsampled_logits(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear upsample to the canonical target-mask resolution (section 9.3)."""

    return F.interpolate(logits.float(), size=(int(size[0]), int(size[1])), mode="bilinear",
                         align_corners=False)


# --------------------------------------------------------------------------- samples and data


@dataclass
class Task6NSample:
    """One oracle-reference Task 6N sample."""

    sample_id: str
    tile_id: str
    program_id: str
    relation: str
    image_path: str
    reference_source_feature_id: int
    target_source_feature_id: int
    target_instance_id: int
    reference_instance_id: int
    level: int
    split: str
    pair_key: str = ""
    metadata: dict = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "tile_id": self.tile_id,
            "program_id": self.program_id,
            "relation": self.relation,
            "image_path": self.image_path,
            "reference_source_feature_id": self.reference_source_feature_id,
            "target_source_feature_id": self.target_source_feature_id,
            "target_instance_id": self.target_instance_id,
            "reference_instance_id": self.reference_instance_id,
            "level": self.level,
            "split": self.split,
            "pair_key": self.pair_key,
            "reference_source": "oracle_native_gt",
        }


def read_pack(path: Path) -> list[Task6NSample]:
    """Read a frozen pack. Records carry an explicit `reference_source = oracle_native_gt` marker."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = set(Task6NSample.__dataclass_fields__)
    records = []
    for record in payload["records"]:
        assert record.get("reference_source", "oracle_native_gt") == "oracle_native_gt", (
            "every Task 6N record must be an oracle-reference record"
        )
        records.append(Task6NSample(**{key: value for key, value in record.items() if key in allowed}))
    return records


def write_pack(path: Path, records: list[Task6NSample], extra: dict) -> dict:
    payload = {
        "records": [record.as_dict() for record in records],
        "count": len(records),
        "reference_source": "oracle_native_gt",
        **extra,
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                          encoding="utf-8")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "count": len(records)}


# --------------------------------------------------------------------------- frozen features


def sam2_checkpoint_sha256() -> str:
    digest = hashlib.sha256()
    with SAM2_CHECKPOINT.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_frozen_sam2_encoder(device: str = "cuda"):
    """Frozen SAM2.1 Hiera Base+ encoder, reused exactly as in Task 6C.7 / 6I."""

    os.environ.setdefault("SAM2_BUILD_CUDA", "0")
    from buildreasonseg.runtime._frozen.mvp.sam2_bridge import Sam2Encoder, load_sam2

    sam, report = load_sam2(SAM2_CONFIG_NAME, str(SAM2_CHECKPOINT), device=device)
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    sam.eval()
    encoder = Sam2Encoder(sam)
    encoder.eval()
    return encoder, report


def read_rgb_tile(image_path: Path) -> np.ndarray:
    """RGB uint8 tile. PIL is used because the WHU source path contains non-ASCII characters that
    `cv2.imread` cannot open on Windows; the project's derived export used the same pixels."""

    from PIL import Image

    with Image.open(image_path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


class FrozenFeatureStore:
    """Disk-memoised frozen SAM2 features (derived, regenerable, gitignored).

    The encoder is frozen and deterministic; memoising its outputs is value-preserving and matches
    the caching idea already used by `Sam2FeatureCache`, but survives across processes so the three
    controlled variants see bit-identical inputs.
    """

    def __init__(self, root: Path, encoder=None, device: str = "cuda") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.device = device
        self.hits = 0
        self.misses = 0
        self.memory: dict[str, torch.Tensor] = {}

    def path_for(self, tile_id: str) -> Path:
        return self.root / f"{tile_id}.npy"

    def get(self, tile_id: str, image_path: Path) -> torch.Tensor:
        if tile_id in self.memory:
            self.hits += 1
            return self.memory[tile_id]
        path = self.path_for(tile_id)
        if path.is_file():
            array = np.load(path)
            self.hits += 1
        else:
            if self.encoder is None:
                raise FileNotFoundError(f"missing frozen feature for {tile_id} and no encoder is loaded")
            image = read_rgb_tile(image_path)
            with torch.no_grad():
                features = self.encoder.encode(image)
            array = features.image_embeddings[0].float().cpu().numpy()
            np.save(path, array)
            self.misses += 1
        tensor = torch.from_numpy(array)
        self.memory[tile_id] = tensor
        return tensor

    def provenance(self) -> dict:
        return {
            "encoder": "frozen SAM2.1 Hiera Base+ image embedding (Sam2Encoder.encode)",
            "checkpoint": str(SAM2_CHECKPOINT),
            "checkpoint_sha256": sam2_checkpoint_sha256(),
            "config_name": SAM2_CONFIG_NAME,
            "repo_id": SAM2_REPO_ID,
            "source_revision_expected": SAM2_SOURCE_REVISION_EXPECTED,
            "cache_root": str(self.root),
            "cache_dtype": "float32",
            "cache_format": "npy (C, h, w)",
            "reused_from": "Task 6C.7 / Task 6I frozen visual path",
            "hits": self.hits,
            "misses": self.misses,
        }


# --------------------------------------------------------------------------- batch assembly


@dataclass
class Task6NBatch:
    visual: torch.Tensor
    mask_ref_down: torch.Tensor
    field: torch.Tensor
    relation_index: torch.Tensor
    target: torch.Tensor
    target_size: tuple[int, int]
    sample_ids: list[str]
    programs: list[str]


def collate_samples(
    samples: list[Task6NSample],
    store: FrozenFeatureStore,
    feature_size: tuple[int, int],
    target_masks: dict[str, np.ndarray],
    reference_masks: dict[str, np.ndarray],
    device: str,
    field_config: FieldConfig = DEFAULT_FIELD_CONFIG,
) -> Task6NBatch:
    """Build one batch. ``M_target`` is used only as the training label — never as an input."""

    visuals, refs, fields, targets, indices, ids, programs = [], [], [], [], [], [], []
    target_size = None
    for sample in samples:
        visual = store.get(sample.tile_id, Path(sample.image_path)).float()
        if tuple(visual.shape[-2:]) != tuple(feature_size):
            raise ValueError(
                f"cached feature for {sample.tile_id} has size {tuple(visual.shape[-2:])}, "
                f"expected {feature_size}"
            )
        mask_ref = reference_masks[sample.sample_id]
        mask_ref_down = resize_soft_mask(mask_ref, feature_size)
        relation_field = geometric_relation_field(mask_ref, sample.relation, feature_size, field_config)
        target = target_masks[sample.sample_id]
        target_tensor = torch.as_tensor(np.asarray(target, dtype=np.float32))
        target_size = tuple(target_tensor.shape)

        visuals.append(visual)
        refs.append(mask_ref_down[0])
        fields.append(relation_field[0])
        targets.append(target_tensor.unsqueeze(0))
        indices.append(RELATION_TO_INDEX[sample.relation])
        ids.append(sample.sample_id)
        programs.append(sample.program_id)

    return Task6NBatch(
        visual=torch.stack(visuals).to(device),
        mask_ref_down=torch.stack(refs).to(device),
        field=torch.stack(fields).to(device),
        relation_index=torch.as_tensor(indices, dtype=torch.long, device=device),
        target=torch.stack(targets).to(device),
        target_size=target_size,
        sample_ids=ids,
        programs=programs,
    )


__all__ = [
    "ALL_VARIANTS",
    "DIRECTIONAL_PROGRAMS",
    "DecoderConfig",
    "FrozenFeatureStore",
    "PROGRAM_TO_RELATION",
    "RelationMaskDecoder",
    "SAM2_CHECKPOINT",
    "SAM2_CONFIG_NAME",
    "TASK6O_VARIANTS",
    "Task6NBatch",
    "Task6NSample",
    "VARIANTS",
    "VARIANT_LABELS",
    "VARIANT_USES_FIELD",
    "VARIANT_USES_REFERENCE",
    "VARIANT_USES_VISUAL",
    "WHU_SOURCE_ROOT",
    "collate_samples",
    "first_conv_in_channels",
    "load_frozen_sam2_encoder",
    "read_pack",
    "read_rgb_tile",
    "reference_centroid",
    "sam2_checkpoint_sha256",
    "task6n_loss",
    "upsampled_logits",
    "write_pack",
]
