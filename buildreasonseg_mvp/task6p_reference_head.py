"""Task 6P section 9 — `ReferenceMaskHead v0.1`.

Given the frozen SAM2 image embedding and the **reference family** (`largest` / `smallest`) from the
canonical program record, predict the soft reference-building mask.

Input contract (enforced by construction and by tests):

* frozen SAM2 feature ``V``: 256 x 64 x 64;
* reference family id in ``{largest, smallest}``.

Explicitly **not** an input: the target relation id, the target mask, the relation field, any YOLO
proposal and any candidate mask.

```text
visual:  Conv1x1(256 -> 128), GroupNorm(8,128), GELU
family:  Embedding(2, 16) broadcast over (h, w)
fusion:  Conv3x3(144 -> 128, padding=1), GroupNorm(8,128), GELU
         Conv3x3(128 -> 64, padding=1),  GroupNorm(8,64),  GELU
         Conv1x1(64 -> 1)
```

Logits are upsampled bilinearly to 512 x 512; the loss is exactly
``BCEWithLogitsLoss + DiceLoss`` (the project's canonical Dice). No other modules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from buildreasonseg_mvp.losses import mask_bce_with_logits, mask_soft_dice
from buildreasonseg_mvp.metrics import upsample_logits

#: Fixed family vocabulary (Task 6P sections 6/9).
FAMILIES: tuple[str, ...] = ("largest", "smallest")
FAMILY_TO_INDEX: dict[str, int] = {family: index for index, family in enumerate(FAMILIES)}


def family_of_program(program_id: str) -> str:
    """`largest_to_*` -> `largest`, `smallest_to_*` -> `smallest` (canonical record)."""

    family = str(program_id).split("_", 1)[0]
    if family not in FAMILY_TO_INDEX:
        raise ValueError(f"program {program_id!r} has no largest/smallest reference family")
    return family


class ReferenceMaskHead(nn.Module):
    """One common head for both reference families; only the family embedding differs."""

    def __init__(self, visual_channels: int = 256, width: int = 128, mid: int = 64,
                 family_dim: int = 16, norm_groups: int = 8) -> None:
        super().__init__()
        self.visual_channels = int(visual_channels)
        self.width = int(width)
        self.mid = int(mid)
        self.family_dim = int(family_dim)
        self.norm_groups = int(norm_groups)

        self.project = nn.Sequential(
            nn.Conv2d(self.visual_channels, self.width, kernel_size=1),
            nn.GroupNorm(self.norm_groups, self.width),
            nn.GELU(),
        )
        self.family_embedding = nn.Embedding(len(FAMILIES), self.family_dim)
        nn.init.normal_(self.family_embedding.weight, mean=0.0, std=0.02)

        channels = self.width + self.family_dim
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, self.width, kernel_size=3, padding=1),
            nn.GroupNorm(self.norm_groups, self.width),
            nn.GELU(),
            nn.Conv2d(self.width, self.mid, kernel_size=3, padding=1),
            nn.GroupNorm(self.norm_groups, self.mid),
            nn.GELU(),
            nn.Conv2d(self.mid, 1, kernel_size=1),
        )

    @property
    def fusion_channels(self) -> int:
        return self.width + self.family_dim

    def forward(self, visual: torch.Tensor, family_index: torch.Tensor) -> torch.Tensor:
        """Reference logits at the frozen feature resolution (``h, w``)."""

        projected = self.project(visual)
        batch, _, height, width = projected.shape
        embedding = self.family_embedding(family_index).view(batch, -1, 1, 1)
        embedding = embedding.expand(batch, self.family_dim, height, width)
        return self.trunk(torch.cat([projected, embedding], dim=1))

    def reference_logits_512(self, visual: torch.Tensor, family_index: torch.Tensor,
                             size: tuple[int, int] = (512, 512)) -> torch.Tensor:
        return upsample_logits(self.forward(visual, family_index), size)

    def parameter_report(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "fusion_channels": self.fusion_channels,
            "by_module": {
                "project": int(sum(p.numel() for p in self.project.parameters())),
                "family_embedding": int(sum(p.numel() for p in self.family_embedding.parameters())),
                "trunk": int(sum(p.numel() for p in self.trunk.parameters())),
            },
            "inputs": ["frozen_visual_feature_256x64x64", "reference_family_id"],
            "not_inputs": ["target_relation_id", "target_mask", "relation_field", "yolo_proposal",
                           "candidate_mask"],
        }


def reference_loss(logits_512: torch.Tensor, target_512: torch.Tensor) -> dict:
    """Exactly `BCEWithLogitsLoss + DiceLoss` (Task 6P section 9)."""

    bce = mask_bce_with_logits(logits_512, target_512)
    dice = mask_soft_dice(logits_512, target_512)
    return {"loss": bce + dice, "bce": bce.detach(), "dice": dice.detach()}


__all__ = [
    "FAMILIES",
    "FAMILY_TO_INDEX",
    "ReferenceMaskHead",
    "family_of_program",
    "reference_loss",
]
