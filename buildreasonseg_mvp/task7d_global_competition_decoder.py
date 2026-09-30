"""Task 7D — relation-guided global competition decoder over dense frozen SAM2 tokens.

Architecture hypothesis (Task 7D Part B, an architecture hypothesis and **not** a novelty claim): explicit
geometric relation fields can parameterize a differentiable global competition distribution over dense
frozen visual tokens; the winning distribution can synthesize a target visual prototype, which then guides
pixel-level mask decoding without target proposals or graph construction.

Common pieces (sections 9-11):

* visual projection `F = Conv1x1(256->128) + GroupNorm(8,128) + GELU`;
* trainable direction embedding, vocab size 4 (`left_of/right_of/above/below`), dim 16, broadcast 64x64;
  there is deliberately no nearest embedding;
* frozen constants `competition_temperature = 1.0`, `eps = 1e-6`, spatial tokens `4096`.

Global competition (section 12, learned variants):

```text
S      = Conv3x3(in->64) + GroupNorm(8,64) + GELU + Conv1x1(64->1)
A_flat = softmax(S.flatten(2) / 1.0, dim=-1)          # global over all 4096 spatial tokens
A      = A_flat.reshape(B,1,64,64)
A_vis  = A * 4096
```

`A` is never detached and the spatial sum is `1 +- 1e-6`. The target visual prototype (section 13) is the
distribution-weighted mean `q = sum_i A_i F_i` with no stop-gradient, and the prototype similarity
(section 14) is the plain cosine `C_i = <F_norm_i, q_norm>` with **no** learned scale, sigmoid or threshold.

D-B1 instead uses the deterministic field competition `W = clamp(P_dir * P_near)` normalized to a
distribution (`A_fixed`); `INVALID_FIELD_MASS` is raised whenever the field mass is <= eps.

The four trainable variants (sections 17-20) and the shared decoder trunk (section 21) are exact:

```text
D-B1  decoder 148 = F + P_dir + P_near + embed + A_fixed_vis + C_fixed
D-B2  score   146 = F + P_dir + P_near + embed      decoder 148 = ... + A_vis + C
D-B3  score   144 = F + embed                        decoder 146 = F + embed + A_vis + C
D-B4  score   146 = F + P_dir + P_near + embed       decoder 147 = ... + A_vis (no q, no C)
```

Loss is exactly `BCEWithLogitsLoss + DiceLoss`; there is no competition supervision, no center/attention/
ranking/contrastive loss, no GRCL and no auxiliary decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VISUAL_CHANNELS = 256
VISUAL_PROJECTION_CHANNELS = 128
DIRECTION_VOCAB = 4
DIRECTION_EMBED_DIM = 16
SPATIAL_TOKENS = 4096
COMPETITION_TEMPERATURE = 1.0
EPS = 1e-6
DIRECTION_IDS = ("left_of", "right_of", "above", "below")
RELATION_TO_INDEX = {relation: index for index, relation in enumerate(DIRECTION_IDS)}

#: Exactly four trainable variants plus the frozen D-B0 baseline; no fifth variant exists.
TRAINABLE_VARIANTS = ("D-B1", "D-B2", "D-B3", "D-B4")
ALL_VARIANTS = ("D-B0", *TRAINABLE_VARIANTS)
PRIMARY_VARIANT = "D-B2"
VARIANT_LABELS = {
    "D-B0": "frozen Task 6Z Z-B3 baseline",
    "D-B1": "deterministic field-weighted visual prototype",
    "D-B2": "learned relation-guided global competition + prototype (primary)",
    "D-B3": "learned visual-only global competition",
    "D-B4": "relation-guided competition map, no prototype",
}
VARIANT_USES_RELATION_FIELDS = {"D-B1": True, "D-B2": True, "D-B3": False, "D-B4": True}
VARIANT_USES_LEARNED_SCORE_HEAD = {"D-B1": False, "D-B2": True, "D-B3": True, "D-B4": True}
VARIANT_USES_PROTOTYPE = {"D-B1": True, "D-B2": True, "D-B3": True, "D-B4": False}
SCORE_INPUT_CHANNELS = {"D-B1": None, "D-B2": 146, "D-B3": 144, "D-B4": 146}
DECODER_INPUT_CHANNELS = {"D-B1": 148, "D-B2": 148, "D-B3": 146, "D-B4": 147}


class InvalidFieldMass(RuntimeError):
    """Raised when the deterministic D-B1 field competition has no mass (section 15)."""


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, channels)


class VisualProjection(nn.Module):
    """Section 9: Conv1x1(256->128) + GroupNorm(8,128) + GELU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(VISUAL_CHANNELS, VISUAL_PROJECTION_CHANNELS, kernel_size=1)
        self.norm = _group_norm(VISUAL_PROJECTION_CHANNELS)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(visual)))


class DirectionEmbedding(nn.Module):
    """Section 10: vocab size 4, dim 16, ids left_of/right_of/above/below."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(DIRECTION_VOCAB, DIRECTION_EMBED_DIM)
        self.ids = DIRECTION_IDS

    def forward(self, relations: list[str], height: int, width: int, device) -> torch.Tensor:
        indices = torch.as_tensor([RELATION_TO_INDEX[relation] for relation in relations],
                                  dtype=torch.long, device=device)
        vector = self.embedding(indices)
        return vector[:, :, None, None].expand(len(relations), DIRECTION_EMBED_DIM, height, width)


class ScoreHead(nn.Module):
    """Section 12: Conv3x3(in->64) + GroupNorm(8,64) + GELU + Conv1x1(64->1) -> S."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.norm = _group_norm(64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(F.gelu(self.norm(self.conv1(features))))


class MaskDecoderTrunk(nn.Module):
    """Section 21: the shared mask decoder trunk (in -> 128 -> 64 -> 1)."""

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


@dataclass
class CompetitionState:
    """Competition tensors kept for diagnostics (never supervised)."""

    attention: torch.Tensor | None = None        # A, (B,1,64,64), sums to 1
    attention_vis: torch.Tensor | None = None    # A * 4096
    prototype: torch.Tensor | None = None        # q, (B,128)
    similarity: torch.Tensor | None = None       # C, (B,1,64,64)
    field_mass: float | None = None              # D-B1 only


def global_competition(score: torch.Tensor, temperature: float = COMPETITION_TEMPERATURE) -> torch.Tensor:
    """Section 12: global softmax over the 4096 spatial tokens (temperature frozen at 1.0)."""

    batch = score.shape[0]
    flat = score.flatten(2) / temperature
    attention = torch.softmax(flat, dim=-1)
    return attention.reshape(batch, 1, score.shape[2], score.shape[3])


def field_competition(directional: torch.Tensor, nearest: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Section 15: deterministic field competition `A_fixed = W / (sum W + eps)`."""

    weight = (directional * nearest).clamp(0.0, 1.0)
    mass = float(weight.sum().item())
    if mass <= EPS:
        raise InvalidFieldMass("INVALID_FIELD_MASS")
    attention = weight / (weight.sum(dim=(-2, -1), keepdim=True) + EPS)
    return attention, mass


def target_prototype(features: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """Section 13: `q = sum_i A_i * F_i` over flattened spatial tokens (no stop-gradient)."""

    batch, channels, height, width = features.shape
    flat = features.reshape(batch, channels, height * width)
    weights = attention.reshape(batch, 1, height * width)
    return (flat * weights).sum(dim=-1)


def prototype_similarity(features: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    """Section 14: plain cosine similarity with no learned scale, sigmoid or threshold."""

    batch, channels, height, width = features.shape
    flat = features.reshape(batch, channels, height * width)
    flat_norm = flat / (flat.norm(dim=1, keepdim=True) + EPS)
    prototype_norm = prototype / (prototype.norm(dim=1, keepdim=True) + EPS)
    similarity = (flat_norm * prototype_norm[:, :, None]).sum(dim=1)
    return similarity.reshape(batch, 1, height, width)


class GlobalCompetitionDecoder(nn.Module):
    """One of the four trainable Task 7D variants."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in TRAINABLE_VARIANTS:
            raise ValueError(f"{variant!r} is not a trainable Task 7D variant")
        self.variant = variant
        self.uses_relation_fields = VARIANT_USES_RELATION_FIELDS[variant]
        self.uses_learned_score_head = VARIANT_USES_LEARNED_SCORE_HEAD[variant]
        self.uses_prototype = VARIANT_USES_PROTOTYPE[variant]
        self.visual_projection = VisualProjection()
        self.direction_embedding = DirectionEmbedding()
        if self.uses_learned_score_head:
            self.score_head = ScoreHead(SCORE_INPUT_CHANNELS[variant])
        self.trunk = MaskDecoderTrunk(DECODER_INPUT_CHANNELS[variant])

    # ------------------------------------------------------------------ competition

    def competition(self, features: torch.Tensor, relations: list[str],
                    directional: torch.Tensor | None,
                    nearest: torch.Tensor | None) -> CompetitionState:
        batch, _, height, width = features.shape
        embedding = self.direction_embedding(relations, height, width, features.device)
        if self.uses_learned_score_head:
            parts = [features]
            if self.uses_relation_fields:
                if directional is None or nearest is None:
                    raise ValueError(f"{self.variant} requires P_dir_64 and P_near_64")
                parts.extend([directional, nearest])
            parts.append(embedding)
            score = self.score_head(torch.cat(parts, dim=1))
            attention = global_competition(score)
            field_mass = None
        else:
            if directional is None or nearest is None:
                raise ValueError("D-B1 requires P_dir_64 and P_near_64")
            attention, field_mass = field_competition(directional, nearest)
        attention_vis = attention * SPATIAL_TOKENS
        prototype = target_prototype(features, attention) if self.uses_prototype else None
        similarity = (prototype_similarity(features, prototype)
                      if prototype is not None else None)
        return CompetitionState(attention=attention, attention_vis=attention_vis, prototype=prototype,
                                similarity=similarity, field_mass=field_mass)

    # ------------------------------------------------------------------ forward

    def forward(self, visual: torch.Tensor, relations: list[str],
                directional: torch.Tensor | None = None, nearest: torch.Tensor | None = None,
                *, return_state: bool = False):
        features = self.visual_projection(visual)
        state = self.competition(features, relations, directional, nearest)
        embedding = self.direction_embedding(relations, features.shape[2], features.shape[3],
                                             features.device)
        parts = [features]
        if self.uses_relation_fields:
            parts.extend([directional, nearest])
        parts.extend([embedding, state.attention_vis])
        if state.similarity is not None:
            parts.append(state.similarity)
        logits = self.trunk(torch.cat(parts, dim=1))
        return (logits, state) if return_state else logits

    def upsampled(self, logits: torch.Tensor, size: int = 512) -> torch.Tensor:
        return F.interpolate(logits.float(), size=(int(size), int(size)), mode="bilinear",
                             align_corners=False)

    def input_spec(self) -> dict:
        return {
            "variant": self.variant,
            "label": VARIANT_LABELS[self.variant],
            "uses_relation_fields": self.uses_relation_fields,
            "uses_learned_score_head": self.uses_learned_score_head,
            "uses_prototype": self.uses_prototype,
            "score_input_channels": SCORE_INPUT_CHANNELS[self.variant],
            "decoder_input_channels": DECODER_INPUT_CHANNELS[self.variant],
            "total_parameters": int(sum(parameter.numel() for parameter in self.parameters())),
        }


def variant_report() -> dict:
    report = {}
    for variant in TRAINABLE_VARIANTS:
        model = GlobalCompetitionDecoder(variant)
        report[variant] = model.input_spec()
        del model
    return {
        "variants": report, "trainable_variants": list(TRAINABLE_VARIANTS),
        "baseline_variant": "D-B0", "primary_variant": PRIMARY_VARIANT,
        "variant_count": len(TRAINABLE_VARIANTS),
        "visual_projection": ["Conv1x1(256,128)", "GroupNorm(8,128)", "GELU"],
        "direction_embedding": {"vocab_size": DIRECTION_VOCAB, "ids": list(DIRECTION_IDS),
                               "dim": DIRECTION_EMBED_DIM},
        "nearest_embedding": False,
        "competition": {"temperature": COMPETITION_TEMPERATURE, "spatial_tokens": SPATIAL_TOKENS,
                        "eps": EPS, "global_softmax": True, "detached": False,
                        "attention_vis_scale": SPATIAL_TOKENS},
        "prototype": {"weighted_sum": True, "stop_gradient": False, "cosine_similarity": True,
                      "learned_scale": False, "sigmoid": False, "threshold": False},
        "decoder_trunk": ["Conv3x3(in,128)", "GroupNorm(8,128)", "GELU", "Conv3x3(128,64)",
                          "GroupNorm(8,64)", "GELU", "Conv1x1(64,1)"],
        "upsample": {"mode": "bilinear", "from": 64, "to": 512, "align_corners": False},
        "loss": "BCEWithLogitsLoss + DiceLoss",
        "extra_losses": [], "grcl": False, "attention_modules": False, "transformer": False,
        "gnn": False, "target_proposals": False, "graph_construction": False,
    }


__all__ = [
    "ALL_VARIANTS",
    "COMPETITION_TEMPERATURE",
    "DECODER_INPUT_CHANNELS",
    "DIRECTION_EMBED_DIM",
    "DIRECTION_IDS",
    "DIRECTION_VOCAB",
    "EPS",
    "CompetitionState",
    "DirectionEmbedding",
    "GlobalCompetitionDecoder",
    "InvalidFieldMass",
    "MaskDecoderTrunk",
    "PRIMARY_VARIANT",
    "RELATION_TO_INDEX",
    "SCORE_INPUT_CHANNELS",
    "SPATIAL_TOKENS",
    "ScoreHead",
    "TRAINABLE_VARIANTS",
    "VARIANT_LABELS",
    "VARIANT_USES_LEARNED_SCORE_HEAD",
    "VARIANT_USES_PROTOTYPE",
    "VARIANT_USES_RELATION_FIELDS",
    "VisualProjection",
    "field_competition",
    "global_competition",
    "prototype_similarity",
    "target_prototype",
    "variant_report",
]
