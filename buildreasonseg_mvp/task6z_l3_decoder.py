"""Task 6Z — L3 direction × nearest composition decoder with exactly six predeclared variants.

Frozen visual representation (section 14): the Task 6N/6O SAM2.1 Hiera Base+ image embedding
`V = 256x64x64`, no image-backbone training and no GT visual input. Common visual projection
`Conv1x1(256->128) + GroupNorm(8,128) + GELU`.

One trainable **direction embedding** (section 15): vocabulary size 4 with ids exactly
`left_of, right_of, above, below`, dim 16, broadcast 64x64. There is deliberately **no** separate trainable
`nearest` embedding because nearest is constant across every Task 6Z program.

Common target trunk (section 16): `Conv3x3(in->128,padding=1) + GroupNorm(8,128) + GELU ->
Conv3x3(128->64,padding=1) + GroupNorm(8,64) + GELU -> Conv1x1(64->1)`, logits bilinearly upsampled to
512x512. Loss exactly `BCEWithLogitsLoss + DiceLoss`; no auxiliary/ranking/GRCL/field supervision.

Six variants (section 17-22), no seventh:

```text
Z-B0  visual_128 + direction_embed_16                        (144)
Z-B1  visual_128 + P_dir_64 + direction_embed_16             (145)
Z-B2  visual_128 + P_near_64 + direction_embed_16            (145)
Z-B3  visual_128 + P_dir_64 + P_near_64 + direction_embed_16 (146)   <- primary hypothesis
Z-B4  visual_128 + P_prod_64 + direction_embed_16            (145)
Z-B5  field_128 + direction_embed_16 (no visual, field = P_prod_64 projected 1->128)  (144)
```
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from buildreasonseg_mvp.task6z_field_composition import DIRECTION_IDS, RELATION_TO_INDEX

VISUAL_CHANNELS = 256
VISUAL_PROJECTION_CHANNELS = 128
DIRECTION_VOCAB = 4
DIRECTION_EMBED_DIM = 16
DECODER_FIELD_SIZE = 64

#: Exactly six variants; no seventh variant may be introduced.
ALL_VARIANTS = ("Z-B0", "Z-B1", "Z-B2", "Z-B3", "Z-B4", "Z-B5")
VARIANT_USES_VISUAL = {"Z-B0": True, "Z-B1": True, "Z-B2": True, "Z-B3": True, "Z-B4": True,
                       "Z-B5": False}
VARIANT_USES_DIRECTIONAL_FIELD = {"Z-B0": False, "Z-B1": True, "Z-B2": False, "Z-B3": True,
                                  "Z-B4": False, "Z-B5": False}
VARIANT_USES_NEAREST_FIELD = {"Z-B0": False, "Z-B1": False, "Z-B2": True, "Z-B3": True,
                              "Z-B4": False, "Z-B5": False}
VARIANT_USES_PRODUCT_FIELD = {"Z-B0": False, "Z-B1": False, "Z-B2": False, "Z-B3": False,
                              "Z-B4": True, "Z-B5": True}
VARIANT_FUSION_CHANNELS = {"Z-B0": 144, "Z-B1": 145, "Z-B2": 145, "Z-B3": 146, "Z-B4": 145,
                           "Z-B5": 144}
VARIANT_LABELS = {
    "Z-B0": "semantic/visual baseline",
    "Z-B1": "directional field only",
    "Z-B2": "nearest field only",
    "Z-B3": "two-field learnable composition (primary hypothesis)",
    "Z-B4": "deterministic product-field comparator",
    "Z-B5": "deterministic composition, geometry-only control",
}
VARIANT_INPUT_TEXT = {
    "Z-B0": "visual_128 + direction_embed_16",
    "Z-B1": "visual_128 + P_dir_64 + direction_embed_16",
    "Z-B2": "visual_128 + P_near_64 + direction_embed_16",
    "Z-B3": "visual_128 + P_dir_64 + P_near_64 + direction_embed_16",
    "Z-B4": "visual_128 + P_prod_64 + direction_embed_16",
    "Z-B5": "field_128 + direction_embed_16",
}


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, channels)


class VisualProjection(nn.Module):
    """Section 14: Conv1x1(256->128) + GroupNorm(8,128) + GELU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(VISUAL_CHANNELS, VISUAL_PROJECTION_CHANNELS, kernel_size=1)
        self.norm = _group_norm(VISUAL_PROJECTION_CHANNELS)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(visual)))


class DirectionEmbedding(nn.Module):
    """Section 15: vocabulary size exactly 4 (left_of, right_of, above, below), dim 16."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(DIRECTION_VOCAB, DIRECTION_EMBED_DIM)
        self.ids = DIRECTION_IDS

    def forward(self, relations: list[str], height: int, width: int, device) -> torch.Tensor:
        indices = torch.as_tensor([RELATION_TO_INDEX[relation] for relation in relations],
                                  dtype=torch.long, device=device)
        vector = self.embedding(indices)
        return vector[:, :, None, None].expand(len(relations), DIRECTION_EMBED_DIM, height, width)


class TargetTrunk(nn.Module):
    """Section 16: shared dense target trunk (in -> 128 -> 64 -> 1)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.norm1 = _group_norm(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.norm2 = _group_norm(64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.norm1(self.conv1(features)))
        hidden = F.gelu(self.norm2(self.conv2(hidden)))
        return self.head(hidden)


class L3TargetDecoder(nn.Module):
    """One of the six predeclared L3 variants."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in ALL_VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {ALL_VARIANTS}")
        self.variant = variant
        self.uses_visual = VARIANT_USES_VISUAL[variant]
        self.uses_directional_field = VARIANT_USES_DIRECTIONAL_FIELD[variant]
        self.uses_nearest_field = VARIANT_USES_NEAREST_FIELD[variant]
        self.uses_product_field = VARIANT_USES_PRODUCT_FIELD[variant]
        if self.uses_visual:
            self.visual_projection = VisualProjection()
        if variant == "Z-B5":
            self.field_projection = nn.Sequential(
                nn.Conv2d(1, VISUAL_PROJECTION_CHANNELS, kernel_size=1),
                _group_norm(VISUAL_PROJECTION_CHANNELS),
                nn.GELU(),
            )
        self.direction_embedding = DirectionEmbedding()
        self.trunk = TargetTrunk(VARIANT_FUSION_CHANNELS[variant])
        self.logit_size = 512

    def forward(self, visual: torch.Tensor | None, relations: list[str],
                directional: torch.Tensor | None, nearest: torch.Tensor | None,
                product: torch.Tensor | None) -> torch.Tensor:
        """Return 64x64 logits; call `upsampled` for the 512x512 output."""

        if self.uses_visual:
            if visual is None:
                raise ValueError(f"{self.variant} requires the frozen SAM2 visual feature")
            features = self.visual_projection(visual)
        else:
            if product is None:
                raise ValueError(f"{self.variant} requires the deterministic product field")
            features = self.field_projection(product)
        batch_size, _, height, width = features.shape
        embedding = self.direction_embedding(relations, height, width, features.device)
        parts = [features]
        if self.uses_directional_field:
            if directional is None:
                raise ValueError(f"{self.variant} requires P_dir_64")
            parts.append(directional)
        if self.uses_nearest_field:
            if nearest is None:
                raise ValueError(f"{self.variant} requires P_near_64")
            parts.append(nearest)
        if self.uses_product_field and self.variant != "Z-B5":
            if product is None:
                raise ValueError(f"{self.variant} requires P_prod_64")
            parts.append(product)
        parts.append(embedding)
        return self.trunk(torch.cat(parts, dim=1))

    def upsampled(self, logits: torch.Tensor, size: int = 512) -> torch.Tensor:
        return F.interpolate(logits.float(), size=(int(size), int(size)), mode="bilinear",
                             align_corners=False)

    def input_spec(self) -> dict:
        return {
            "variant": self.variant,
            "label": VARIANT_LABELS[self.variant],
            "inputs": VARIANT_INPUT_TEXT[self.variant],
            "uses_visual": self.uses_visual,
            "uses_directional_field": self.uses_directional_field,
            "uses_nearest_field": self.uses_nearest_field,
            "uses_product_field": self.uses_product_field,
            "fusion_channels": VARIANT_FUSION_CHANNELS[self.variant],
            "total_parameters": int(sum(parameter.numel() for parameter in self.parameters())),
        }


def variant_report() -> dict:
    report = {}
    for variant in ALL_VARIANTS:
        model = L3TargetDecoder(variant)
        report[variant] = model.input_spec()
        del model
    return {
        "variants": report,
        "variant_order": list(ALL_VARIANTS),
        "variant_count": len(ALL_VARIANTS),
        "visual_representation": "frozen SAM2.1 Hiera Base+ 256x64x64 (no retraining, no GT visual)",
        "visual_projection": ["Conv1x1(256,128)", "GroupNorm(8,128)", "GELU"],
        "field_projection_z_b5": ["Conv1x1(1,128)", "GroupNorm(8,128)", "GELU"],
        "direction_embedding": {"vocab_size": DIRECTION_VOCAB, "ids": list(DIRECTION_IDS),
                                "dim": DIRECTION_EMBED_DIM},
        "separate_nearest_embedding": False,
        "target_trunk": ["Conv3x3(in,128)", "GroupNorm(8,128)", "GELU", "Conv3x3(128,64)",
                         "GroupNorm(8,64)", "GELU", "Conv1x1(64,1)"],
        "logit_resolution": 64, "output_resolution": 512,
        "loss": "BCEWithLogitsLoss + DiceLoss",
        "extra_losses": [], "grcl": False, "attention": False, "transformer": False, "gnn": False,
        "deterministic_multiplication_before_decoder_z_b3": False,
    }


__all__ = [
    "ALL_VARIANTS",
    "DIRECTION_EMBED_DIM",
    "DIRECTION_VOCAB",
    "DirectionEmbedding",
    "L3TargetDecoder",
    "TargetTrunk",
    "VARIANT_FUSION_CHANNELS",
    "VARIANT_LABELS",
    "VARIANT_USES_DIRECTIONAL_FIELD",
    "VARIANT_USES_NEAREST_FIELD",
    "VARIANT_USES_PRODUCT_FIELD",
    "VARIANT_USES_VISUAL",
    "VisualProjection",
    "variant_report",
]
