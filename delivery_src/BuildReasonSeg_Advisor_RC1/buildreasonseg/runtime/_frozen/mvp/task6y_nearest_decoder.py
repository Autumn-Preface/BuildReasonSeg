"""Task 6Y — nearest-only dense target decoder with four predeclared variants.

Frozen visual representation (section 13): the Task 6N/6O SAM2.1 Hiera Base+ image embedding
`V = 256x64x64`; no GT visual input and no retraining.

Common components:

* **visual projection** (section 14, Y-B0/B1/B2): `Conv1x1(256->128) + GroupNorm(8,128) + GELU`
* **nearest embedding** (section 15): one trainable embedding, vocab size 1, dim 16, semantic id
  `nearest`, broadcast to 64x64
* **target trunk** (section 16): `Conv3x3(in->128) + GN + GELU -> Conv3x3(128->64) + GN + GELU ->
  Conv1x1(64->1)`, logits bilinearly upsampled to 512x512
* **loss** (section 16): exactly `BCEWithLogitsLoss + DiceLoss` (the Task 6N `task6n_loss`); no GRCL, no
  ranking loss, no auxiliary field loss and no candidate loss.

Variants (section 17-20):

```text
Y-B0  visual_128 + nearest_embed_16                  (144 channels, no ref mask, no field)
Y-B1  visual_128 + M_ref_64 + nearest_embed_16       (145 channels, no nearest field)
Y-B2  visual_128 + P_near_64 + nearest_embed_16      (145 channels, the primary hypothesis)
Y-B3  field_128 + nearest_embed_16                   (144 channels, no SAM2 visual at all;
                                                      field projection Conv1x1(1->128)+GN+GELU)
```
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VISUAL_CHANNELS = 256
VISUAL_PROJECTION_CHANNELS = 128
EMBEDDING_DIM = 16
EMBEDDING_VOCAB = 1
SEMANTIC_ID = "nearest"
FIELD_CHANNELS = 1
DECODER_FIELD_SIZE = 64

#: The exact four variants; no fifth variant exists.
ALL_VARIANTS = ("Y-B0", "Y-B1", "Y-B2", "Y-B3")
VARIANT_USES_VISUAL = {"Y-B0": True, "Y-B1": True, "Y-B2": True, "Y-B3": False}
VARIANT_USES_REFERENCE_MASK = {"Y-B0": False, "Y-B1": True, "Y-B2": False, "Y-B3": False}
VARIANT_USES_NEAREST_FIELD = {"Y-B0": False, "Y-B1": False, "Y-B2": True, "Y-B3": True}
VARIANT_FUSION_CHANNELS = {"Y-B0": 144, "Y-B1": 145, "Y-B2": 145, "Y-B3": 144}
VARIANT_LABELS = {
    "Y-B0": "visual baseline",
    "Y-B1": "direct reference-mask control",
    "Y-B2": "nearest-boundary-field model",
    "Y-B3": "geometry-only control",
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


class NearestEmbedding(nn.Module):
    """Section 15: one trainable embedding, vocab size 1, dim 16, broadcast 64x64."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(EMBEDDING_VOCAB, EMBEDDING_DIM)
        self.semantic_id = SEMANTIC_ID
        self.vocabulary_index = 0

    def forward(self, batch_size: int, height: int, width: int, device) -> torch.Tensor:
        index = torch.zeros(batch_size, dtype=torch.long, device=device)
        vector = self.embedding(index)
        return vector[:, :, None, None].expand(batch_size, EMBEDDING_DIM, height, width)


class TargetTrunk(nn.Module):
    """Section 16: the shared dense target trunk (in -> 128 -> 64 -> 1)."""

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


class NearestTargetDecoder(nn.Module):
    """One of the four predeclared variants."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in ALL_VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; expected one of {ALL_VARIANTS}")
        self.variant = variant
        self.uses_visual = VARIANT_USES_VISUAL[variant]
        self.uses_reference_mask = VARIANT_USES_REFERENCE_MASK[variant]
        self.uses_nearest_field = VARIANT_USES_NEAREST_FIELD[variant]
        if self.uses_visual:
            self.visual_projection = VisualProjection()
        if variant == "Y-B3":
            self.field_projection = nn.Sequential(
                nn.Conv2d(FIELD_CHANNELS, VISUAL_PROJECTION_CHANNELS, kernel_size=1),
                _group_norm(VISUAL_PROJECTION_CHANNELS),
                nn.GELU(),
            )
        self.nearest_embedding = NearestEmbedding()
        self.trunk = TargetTrunk(VARIANT_FUSION_CHANNELS[variant])
        self.logit_size = 512

    def forward(self, visual: torch.Tensor | None, reference_mask: torch.Tensor | None,
                nearest_field: torch.Tensor | None) -> torch.Tensor:
        """Return logits at 64x64 (call `upsampled` for the 512x512 output)."""

        if self.uses_visual:
            if visual is None:
                raise ValueError(f"{self.variant} requires the frozen SAM2 visual feature")
            features = self.visual_projection(visual)
        else:
            if nearest_field is None:
                raise ValueError(f"{self.variant} requires the nearest boundary field")
            features = self.field_projection(nearest_field)
        batch_size, _, height, width = features.shape
        embedding = self.nearest_embedding(batch_size, height, width, features.device)
        if self.variant == "Y-B1":
            if reference_mask is None:
                raise ValueError("Y-B1 requires the direct reference-mask control")
            parts = [features, reference_mask, embedding]
        elif self.variant == "Y-B2":
            if nearest_field is None:
                raise ValueError("Y-B2 requires the nearest boundary field")
            parts = [features, nearest_field, embedding]
        else:
            # Y-B0: visual + embedding.  Y-B3: the field is already projected to 128 channels, so it
            # must NOT be appended a second time (that would make the fusion 145 instead of 144).
            parts = [features, embedding]
        return self.trunk(torch.cat(parts, dim=1))

    def upsampled(self, logits: torch.Tensor, size: int = 512) -> torch.Tensor:
        return F.interpolate(logits.float(), size=(int(size), int(size)), mode="bilinear",
                             align_corners=False)

    def input_spec(self) -> dict:
        return {
            "variant": self.variant,
            "label": VARIANT_LABELS[self.variant],
            "uses_visual": self.uses_visual,
            "uses_reference_mask": self.uses_reference_mask,
            "uses_nearest_field": self.uses_nearest_field,
            "fusion_channels": VARIANT_FUSION_CHANNELS[self.variant],
            "inputs": ("visual_128 + nearest_embed_16" if self.variant == "Y-B0" else
                       "visual_128 + M_ref_64 + nearest_embed_16" if self.variant == "Y-B1" else
                       "visual_128 + P_near_64 + nearest_embed_16" if self.variant == "Y-B2" else
                       "field_128 + nearest_embed_16"),
            "total_parameters": int(sum(parameter.numel() for parameter in self.parameters())),
        }


def variant_report() -> dict:
    report = {}
    for variant in ALL_VARIANTS:
        model = NearestTargetDecoder(variant)
        report[variant] = model.input_spec()
        del model
    return {
        "variants": report,
        "variant_order": list(ALL_VARIANTS),
        "visual_representation": "frozen SAM2.1 Hiera Base+ 256x64x64 (no retraining, no GT visual)",
        "visual_projection": ["Conv1x1(256,128)", "GroupNorm(8,128)", "GELU"],
        "field_projection_y_b3": ["Conv1x1(1,128)", "GroupNorm(8,128)", "GELU"],
        "nearest_embedding": {"vocab_size": EMBEDDING_VOCAB, "dim": EMBEDDING_DIM,
                              "semantic_id": SEMANTIC_ID},
        "target_trunk": ["Conv3x3(in,128)", "GroupNorm(8,128)", "GELU", "Conv3x3(128,64)",
                         "GroupNorm(8,64)", "GELU", "Conv1x1(64,1)"],
        "logit_resolution": 64,
        "output_resolution": 512,
        "loss": "BCEWithLogitsLoss + DiceLoss",
        "extra_losses": [],
        "grcl": False,
        "grcl_enabled": False,
        "ranking_loss": False,
        "auxiliary_field_loss": False,
        "candidate_loss": False,
    }


__all__ = [
    "ALL_VARIANTS",
    "DECODER_FIELD_SIZE",
    "EMBEDDING_DIM",
    "EMBEDDING_VOCAB",
    "NearestEmbedding",
    "NearestTargetDecoder",
    "SEMANTIC_ID",
    "TargetTrunk",
    "VARIANT_FUSION_CHANNELS",
    "VARIANT_LABELS",
    "VARIANT_USES_NEAREST_FIELD",
    "VARIANT_USES_REFERENCE_MASK",
    "VARIANT_USES_VISUAL",
    "VisualProjection",
    "variant_report",
]
