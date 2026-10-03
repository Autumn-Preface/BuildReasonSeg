"""Task 6I: Visual Query Refinement Block v0.1.

Task 6H.1's bounded point objective removed the scale degeneracy, but the single pre-reasoning
`[BOX]` query still lacks target-specific spatial signal (H0-R: inside 7/20, paired point 2/10).
Task 6I keeps the Task 6H.1 point-cell CE + bounded counterfactual pair objective and the paired
training protocol **exactly unchanged**, and gives the query exactly one explicit opportunity to
inspect frozen visual features before high-resolution localization:

    image + instruction
        -> Qwen pre-reasoning [BOX] query q0                      [B, 2048]  (causally clean)
        -> VisualQueryRefinementBlock (ONE cross-attention layer):
             q0  -> LayerNorm -> Linear(2048, 256) -> q           [B, 1, 256]
             F64 [B,256,64,64] -> Conv1x1(256,256) -> flatten -> LayerNorm   [B, 4096, 256]
             q_ref = q + CrossAttention(q, F64)     (embed 256, 4 heads, ONE layer)
             q1   = q_ref + FFN(LayerNorm(q_ref))   (256 -> 512 -> 256, GELU)
        -> q1 scores the frozen SAM2 256x256 / 32-channel feature:
             F256 -> Conv1x1(32,256) -> K256
             heatmap_logits[y,x] = dot(q1, K256[:,y,x]) / sqrt(256) (+ optional scalar bias)
        -> argmax point -> frozen SAM2 positive-point prompt

Frozen (re-imported, never modified): the Task 6H.1 point-cell CE, the bounded probability-mass
pair loss and their weights. The retired Task 6G BCE+Dice and Task 6H raw-logit ranking stay
detached diagnostics. SAM2 is frozen; only the block, the text LoRA, the `[BOX]` row and the
`[SEG]` row train. The old Task 6G `DenseSpatialGroundingHead` is **not** part of the candidate
forward (it stays installed only as frozen evidence).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .box_query import box_query_forward, build_query_batch
from .dense_grounding import (
    cell_centre,
    downsample_target_mask,
    feature_tensor_for_grid,
)
from .grounding import distance_transform_point

#: Section 3/5: the exact verified SAM2 feature levels (asserted, never assumed).
COARSE_GRID = 64
FINE_GRID = 256
COARSE_CHANNELS = 256
FINE_CHANNELS = 32

#: Section 4: the recommended block geometry, fixed (constants, not hyperparameters to sweep).
QUERY_DIM = 2048
EMBED_DIM = 256
NUM_HEADS = 4
FFN_DIM = 512


class VisualQueryRefinementBlock(nn.Module):
    """Section 4/5: exactly one cross-attention refinement layer + FFN + high-res scorer.

    `refine(q0, f64) -> (q1, attention_weights)`, `score(q1, f256) -> heatmap_logits`, and a
    combined `forward(q0, f64, f256) -> (q1, heatmap, attention_weights)`. The 1x4096
    cross-attention weights are returned for diagnostics only (section 7) and are never
    supervised.
    """

    def __init__(
        self,
        query_dim: int = QUERY_DIM,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        ff_dim: int = FFN_DIM,
        coarse_channels: int | None = None,
        fine_channels: int | None = None,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.query_dim = int(query_dim)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.coarse_channels = COARSE_CHANNELS if coarse_channels is None else int(coarse_channels)
        self.fine_channels = FINE_CHANNELS if fine_channels is None else int(fine_channels)
        if self.coarse_channels != COARSE_CHANNELS:
            raise ValueError(f"coarse feature must have {COARSE_CHANNELS} channels, got {self.coarse_channels}")
        if self.fine_channels != FINE_CHANNELS:
            raise ValueError(f"fine feature must have {FINE_CHANNELS} channels, got {self.fine_channels}")

        # q0 -> q [B, 1, embed_dim]
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.query_proj = nn.Linear(self.query_dim, self.embed_dim)

        # F64 -> [B, 4096, embed_dim]
        self.coarse_proj = nn.Conv2d(self.coarse_channels, self.embed_dim, 1)
        self.coarse_norm = nn.LayerNorm(self.embed_dim)

        # Exactly ONE cross-attention layer.
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        # FFN with pre-norm residual: q1 = q_ref + FFN(LayerNorm(q_ref)).
        self.ffn_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            nn.GELU(),
            nn.Linear(self.ff_dim, self.embed_dim),
        )

        # F256 scorer (section 5).
        self.fine_proj = nn.Conv2d(self.fine_channels, self.embed_dim, 1)
        self.bias = nn.Parameter(torch.zeros(())) if use_bias else None
        self.register_buffer("logit_scale", torch.tensor(1.0 / math.sqrt(self.embed_dim)), persistent=False)

    # -- structure --------------------------------------------------------

    @property
    def cross_attention_layer_count(self) -> int:
        """Must be exactly 1 (section 4); recorded and asserted everywhere."""
        return sum(1 for module in self.modules() if isinstance(module, nn.MultiheadAttention))

    def as_dict(self) -> dict:
        return {
            "class": "VisualQueryRefinementBlock",
            "query_dim": self.query_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "coarse_channels": self.coarse_channels,
            "fine_channels": self.fine_channels,
            "coarse_grid": COARSE_GRID,
            "fine_grid": FINE_GRID,
            "cross_attention_layers": self.cross_attention_layer_count,
            "scalar_bias": self.bias is not None,
            "structure": (
                "q0 -> LayerNorm -> Linear(2048,256) -> q [B,1,256]; "
                "F64 -> Conv1x1(256,256) -> flatten [B,4096,256] -> LayerNorm; "
                "CrossAttention(embed 256, 4 heads, ONE layer) -> q_ref = q + attn(q, F64); "
                "q1 = q_ref + FFN(LayerNorm(q_ref)) (256 -> 512 -> 256, GELU); "
                "heatmap = dot(q1, Conv1x1(32,256)(F256))/sqrt(256) + optional scalar bias"
            ),
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

    # -- forward ----------------------------------------------------------

    def _check_coarse(self, f64: torch.Tensor) -> None:
        expected = (self.coarse_channels, COARSE_GRID, COARSE_GRID)
        actual = tuple(f64.shape[1:])
        if f64.dim() != 4 or actual != expected:
            raise RuntimeError(f"F64 must be [B,{self.coarse_channels},{COARSE_GRID},{COARSE_GRID}], got {tuple(f64.shape)}")

    def _check_fine(self, f256: torch.Tensor) -> None:
        expected = (self.fine_channels, FINE_GRID, FINE_GRID)
        actual = tuple(f256.shape[1:])
        if f256.dim() != 4 or actual != expected:
            raise RuntimeError(f"F256 must be [B,{self.fine_channels},{FINE_GRID},{FINE_GRID}], got {tuple(f256.shape)}")

    def refine(self, q0: torch.Tensor, f64: torch.Tensor):
        """`q0 [B,2048] x frozen F64 [B,256,64,64] -> (q1 [B,256], attention [B,4096])`."""

        self._check_coarse(f64)
        if q0.dim() != 2 or q0.shape[-1] != self.query_dim:
            raise RuntimeError(f"q0 must be [B,{self.query_dim}], got {tuple(q0.shape)}")
        q = self.query_proj(self.query_norm(q0.float())).float().unsqueeze(1)   # [B, 1, E] fp32
        keys = self.coarse_proj(f64.float()).float()                            # [B, E, 64, 64] fp32
        keys = keys.flatten(2).transpose(1, 2)                                  # [B, 4096, E]
        keys = self.coarse_norm(keys)
        attended, weights = self.cross_attn(
            q, keys, keys, need_weights=True, average_attn_weights=True
        )                                                                       # [B,1,E], [B,1,4096]
        q_ref = q + attended.float()                                            # residual (section 4)
        q1 = q_ref + self.ffn(self.ffn_norm(q_ref)).float()                     # FFN residual
        return q1[:, 0, :].float(), weights[:, 0, :]                            # [B,E] fp32, [B,4096]

    def score(self, q1: torch.Tensor, f256: torch.Tensor) -> torch.Tensor:
        """`q1 [B,256] x frozen F256 [B,32,256,256] -> heatmap logits [B,256,256]` (section 5)."""

        self._check_fine(f256)
        if q1.dim() != 2 or q1.shape[-1] != self.embed_dim:
            raise RuntimeError(f"q1 must be [B,{self.embed_dim}], got {tuple(q1.shape)}")
        keys = self.fine_proj(f256.float()).float()                            # [B, E, 256, 256] fp32
        logits = torch.einsum("be,behw->bhw", q1.float(), keys) * self.logit_scale  # [B, 256, 256]
        if self.bias is not None:
            logits = logits + self.bias
        return logits

    def forward(self, q0: torch.Tensor, f64: torch.Tensor, f256: torch.Tensor):
        """`(q0, F64, F256) -> (q1 [B,256], heatmap [B,256,256], attention [B,4096])`."""

        q1, weights = self.refine(q0, f64)
        return q1, self.score(q1, f256), weights


# ------------------------------------------------------------------ attention diagnostics


def attention_diagnostics(weights: torch.Tensor, soft_target64: torch.Tensor) -> dict:
    """Section 7: 1x4096 cross-attention diagnostics. GT mass is scoring only, never input."""

    flat = torch.as_tensor(weights, dtype=torch.float32).reshape(-1).cpu()
    total = flat.numel()
    if total != COARSE_GRID * COARSE_GRID:
        raise RuntimeError(f"attention weights must be 1x4096, got {total}")
    clamped = flat.clamp(min=1e-12)
    entropy = float(-(clamped * clamped.log()).sum())
    index = int(torch.argmax(flat))
    y, x = divmod(index, COARSE_GRID)
    top10 = float(torch.topk(flat, min(10, total)).values.sum())
    soft = torch.as_tensor(soft_target64, dtype=torch.float32).reshape(-1).cpu()
    target_mass = float((flat * soft).sum())
    return {
        "attn_entropy": entropy,
        "attn_top1_index": index,
        "attn_top1_cell": [x, y],
        "attn_top1_centre": [cell_centre(x, COARSE_GRID), cell_centre(y, COARSE_GRID)],
        "attn_top10_mass": top10,
        "attn_target_mass": target_mass,
    }


# ------------------------------------------------------------------ inference path


@torch.no_grad()
def forward_refined(runtime, sample, image=None, features=None) -> dict:
    """The Task 6I inference path: `image + instruction -> [BOX] q0 -> refine -> heatmap`.

    GT is read for scoring only (`soft_target`, `gt_point`); the model sees exactly image +
    instruction + constant `[BOX]` + the two frozen SAM2 features. `box_hidden` is q0 and
    `query_projected` is q1, so the Task 6H.1 record builder and representation tooling work
    unchanged. The Qwen model runs in **eval mode** here: LoRA dropout (p=0.05) is a training-only
    regularizer and would otherwise inject noise into the recorded inference numbers.
    """

    if image is None:
        image = sample.image_rgb()
    if features is None:
        features, _cached = runtime.features_for(sample, image)
    runtime.set_visual_cache_key(str(sample.image_id))
    batch = build_query_batch(runtime, image, sample.instruction_zh).to(runtime.device)
    qwen = runtime.model.qwen
    was_training = bool(qwen.training)
    qwen.eval()
    try:
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=bool(runtime.cfg.get("training", {}).get("bf16_autocast", True))
        ):
            _logits, q0, _seg_hidden = box_query_forward(qwen, batch)
            f64 = feature_tensor_for_grid(features, COARSE_GRID).to(runtime.device)
            f256 = feature_tensor_for_grid(features, FINE_GRID).to(runtime.device)
            q1, heatmap, attention = runtime.model.refine_block(q0, f64, f256)
    finally:
        if was_training:
            qwen.train()
    target_mask = sample.target_mask()
    return {
        "sample_id": str(sample.sample_id),
        "grid": FINE_GRID,
        "logits": heatmap[0].detach().float().cpu(),
        "soft_target": downsample_target_mask(target_mask, FINE_GRID),
        "target_mask": target_mask,
        "gt_point": list(distance_transform_point(target_mask)),
        "box_hidden": q0[0].detach().float().cpu(),
        "query_projected": q1[0].detach().float().cpu(),
        "refined_query": q1[0].detach().float().cpu(),
        "attention_weights": attention[0].detach().float().cpu(),
        "soft_target_64": downsample_target_mask(target_mask, COARSE_GRID),
    }


def refined_record_from_raw(runtime, sample, raw: dict, *, with_hidden: bool = False) -> dict:
    """Task 6H.1 point record (unchanged builder) + section 7 attention diagnostics."""

    from .spatial_objective import point_record_from_raw

    record = point_record_from_raw(runtime, sample, raw, with_hidden=with_hidden)
    record.update(attention_diagnostics(raw["attention_weights"], raw["soft_target_64"]))
    if with_hidden:
        record["refined_query"] = raw["refined_query"].numpy().astype(np.float32).tolist()
    return record


@torch.no_grad()
def refined_point_record(runtime, sample, *, with_hidden: bool = False) -> dict:
    """One record through the real Task 6I inference path."""

    raw = forward_refined(runtime, sample)
    return refined_record_from_raw(runtime, sample, raw, with_hidden=with_hidden)


@torch.no_grad()
def evaluate_refined_pair(runtime, pair: dict, sample_a, sample_b, *,
                          with_hidden: bool = False) -> dict:
    """One canonical pair through the refined path, scored by the frozen bounded objective.

    Same contract as `spatial_objective.evaluate_point_pair`: one shared frozen SAM2 feature set
    for the pair, the Task 6H.1 probability masses, plus q0/q1 same-image separation and the
    section 7 attention masses (own and cross) as scalars — never the raw arrays.
    """

    from .spatial_objective import (
        bounded_pair_loss,
        counterfactual_masses,
        spatial_probabilities,
    )

    image = sample_a.image_rgb()
    if str(sample_b.image_id) != str(sample_a.image_id):
        raise RuntimeError("a canonical counterfactual pair must share one source image")
    features, _cached = runtime.features_for(sample_a, image)
    raw_a = forward_refined(runtime, sample_a, image=image, features=features)
    raw_b = forward_refined(runtime, sample_b, image=image, features=features)
    record_a = refined_record_from_raw(runtime, sample_a, raw_a, with_hidden=with_hidden)
    record_b = refined_record_from_raw(runtime, sample_b, raw_b, with_hidden=with_hidden)

    probabilities_a = spatial_probabilities(raw_a["logits"])
    probabilities_b = spatial_probabilities(raw_b["logits"])
    masses = counterfactual_masses(
        probabilities_a, probabilities_b, raw_a["soft_target"], raw_b["soft_target"]
    )
    masses["cf_loss"] = float(bounded_pair_loss(
        masses["p_aa"], masses["p_ab"], masses["p_bb"], masses["p_ba"]
    )["raw"])
    probability_overlap = float(torch.minimum(probabilities_a, probabilities_b).sum())
    binary_a = (probabilities_a > 0.5).numpy()
    binary_b = (probabilities_b > 0.5).numpy()
    union = float(np.logical_or(binary_a, binary_b).sum())
    binary_iou = float(np.logical_and(binary_a, binary_b).sum() / union) if union else 0.0

    q0_a = raw_a["box_hidden"].to(torch.float64)
    q0_b = raw_b["box_hidden"].to(torch.float64)
    q1_a = raw_a["refined_query"].to(torch.float64)
    q1_b = raw_b["refined_query"].to(torch.float64)

    def _separation(left: torch.Tensor, right: torch.Tensor) -> dict:
        left_n = left / (left.norm() + 1e-12)
        right_n = right / (right.norm() + 1e-12)
        return {
            "l2": float((left - right).norm()),
            "cosine": float((left_n * right_n).sum()),
        }

    attn_diag_a = attention_diagnostics(raw_a["attention_weights"], raw_a["soft_target_64"])
    attn_diag_b = attention_diagnostics(raw_b["attention_weights"], raw_b["soft_target_64"])
    soft_a = raw_a["soft_target_64"].reshape(-1)
    soft_b = raw_b["soft_target_64"].reshape(-1)
    attn_a = raw_a["attention_weights"].reshape(-1)
    attn_b = raw_b["attention_weights"].reshape(-1)
    attention_masses = {
        "attention_entropy_a": attn_diag_a["attn_entropy"],
        "attention_entropy_b": attn_diag_b["attn_entropy"],
        "attention_top10_mass_a": attn_diag_a["attn_top10_mass"],
        "attention_top10_mass_b": attn_diag_b["attn_top10_mass"],
        "attention_own_mass_a": float((attn_a * soft_a).sum()),
        "attention_own_mass_b": float((attn_b * soft_b).sum()),
        "attention_cross_mass_a": float((attn_a * soft_b).sum()),
        "attention_cross_mass_b": float((attn_b * soft_a).sum()),
    }
    attention_masses["attention_own_mass_mean"] = 0.5 * (
        attention_masses["attention_own_mass_a"] + attention_masses["attention_own_mass_b"]
    )
    attention_masses["attention_cross_mass_mean"] = 0.5 * (
        attention_masses["attention_cross_mass_a"] + attention_masses["attention_cross_mass_b"]
    )

    return {
        "image_id": str(pair["image_id"]),
        "a": str(pair["a"]),
        "b": str(pair["b"]),
        "target_a": int(pair.get("target_a", -1)),
        "target_b": int(pair.get("target_b", -1)),
        "a_level": int(pair.get("a_level", 0)),
        "b_level": int(pair.get("b_level", 0)),
        "a_query_type": str(pair.get("a_query_type", "")),
        "b_query_type": str(pair.get("b_query_type", "")),
        **{key: value for key, value in masses["raw"].items()},
        "pair_preference_pass": masses["pair_preference_pass"],
        "cf_loss": masses["cf_loss"],
        "point_inside_own_a": bool(record_a["point_inside_own"]),
        "point_inside_own_b": bool(record_b["point_inside_own"]),
        "point_selection_passed": bool(record_a["point_inside_own"] and record_b["point_inside_own"]),
        "target_cell_top1_a": bool(record_a["target_cell_top1"]),
        "target_cell_top1_b": bool(record_b["target_cell_top1"]),
        "same_image_point_distance": float(
            sum(
                abs(float(x) - float(y))
                for x, y in zip(record_a["predicted_point"], record_b["predicted_point"])
            )
        ),
        "same_image_probability_map_overlap": probability_overlap,
        "same_image_probability_binary_iou": binary_iou,
        "same_image_q0": _separation(q0_a, q0_b),
        "same_image_q1": _separation(q1_a, q1_b),
        "mean_abs_logit": 0.5 * (record_a["mean_abs_logit"] + record_b["mean_abs_logit"]),
        "spatial_entropy": 0.5 * (record_a["spatial_entropy"] + record_b["spatial_entropy"]),
        **attention_masses,
        "record_a": record_a,
        "record_b": record_b,
    }


# ------------------------------------------------------------------ representation


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def _pairwise_cosine(matrix: np.ndarray) -> float | None:
    """Mean cosine over `[N, 2, D]` pairs."""

    if matrix is None or matrix.shape[0] == 0 or matrix.shape[-1] < 2:
        return None
    left = matrix[:, 0, :].astype(np.float64)
    right = matrix[:, 1, :].astype(np.float64)
    left = left / (np.linalg.norm(left, axis=-1, keepdims=True) + 1e-12)
    right = right / (np.linalg.norm(right, axis=-1, keepdims=True) + 1e-12)
    return float((left * right).sum(axis=-1).mean())


def _pairwise_l2(matrix: np.ndarray) -> float | None:
    if matrix is None or matrix.shape[0] == 0:
        return None
    return float(np.linalg.norm(matrix[:, 0, :] - matrix[:, 1, :], axis=-1).mean())


def refinement_representation_report(paired_records: list[dict], val_records: list[dict],
                                     pairs: list[dict], *, rng_seed: int = 20260926) -> dict:
    """Section 11: q0 vs q1 on same-image/different-instruction pairs at the best checkpoint.

    `box_hidden` is q0 (2048-d) and `query_projected` is q1 (256-d); the record builder already
    stores both when `with_hidden=True`. Correlations use the same sampled different-image pool
    for q0 and q1 so the comparison is fair.
    """

    paired_by_id = {record["sample_id"]: record for record in paired_records}
    q0_val = np.stack([np.asarray(r["box_hidden"], np.float64) for r in val_records], axis=0)
    q1_val = np.stack([np.asarray(r["query_projected"], np.float64) for r in val_records], axis=0)
    points = np.asarray([r["gt_point"] for r in val_records], np.float64)

    def pair_stack(records, key) -> np.ndarray:
        rows = []
        for pair in pairs:
            a = records.get(str(pair["a"]))
            b = records.get(str(pair["b"]))
            if a is None or b is None:
                continue
            rows.append([key(a), key(b)])
        return np.stack(rows) if rows else np.zeros((0, 2, 1))

    q0_pairs = pair_stack(paired_by_id, lambda r: np.asarray(r["box_hidden"], np.float64))
    q1_pairs = pair_stack(paired_by_id, lambda r: np.asarray(r["query_projected"], np.float64))

    # global-mean centering over the analysed set (paired + val), matching Task 6G/6H.1.
    q0_mean = np.concatenate([q0_pairs.reshape(-1, q0_pairs.shape[-1]), q0_val], axis=0).mean(axis=0, keepdims=True)
    q1_mean = np.concatenate([q1_pairs.reshape(-1, q1_pairs.shape[-1]), q1_val], axis=0).mean(axis=0, keepdims=True)

    rng = np.random.default_rng(int(rng_seed))
    pool: list[tuple[int, int]] = []
    while len(pool) < min(200, len(val_records)):
        i = int(rng.integers(len(val_records)))
        j = int(rng.integers(len(val_records)))
        if i != j:
            pool.append((i, j))
    diff_q0 = np.stack([[q0_val[i], q0_val[j]] for i, j in pool]) if pool else np.zeros((0, 2, q0_val.shape[-1]))
    diff_q1 = np.stack([[q1_val[i], q1_val[j]] for i, j in pool]) if pool else np.zeros((0, 2, q1_val.shape[-1]))

    centred_q1 = q1_val - q1_mean
    _, singular_values, _ = np.linalg.svd(centred_q1, full_matrices=False)
    participation = float((singular_values**2).sum() ** 2 / max((singular_values**4).sum(), 1e-12))

    q0_dists = [float(np.linalg.norm(q0_val[i] - q0_val[j])) for i, j in pool]
    q1_dists = [float(np.linalg.norm(q1_val[i] - q1_val[j])) for i, j in pool]
    point_dists = [float(np.linalg.norm(points[i] - points[j])) for i, j in pool]

    def corr(left, right) -> float | None:
        if len(left) < 2:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "paired_sample_count": len(paired_records),
        "val_sample_count": len(val_records),
        "q0": {
            "raw_cosine": _pairwise_cosine(q0_pairs),
            "centered_cosine": _pairwise_cosine(q0_pairs - q0_mean[None, None, :]),
            "l2": _pairwise_l2(q0_pairs),
            "different_image_cosine": _pairwise_cosine(diff_q0),
        },
        "q1": {
            "raw_cosine": _pairwise_cosine(q1_pairs),
            "centered_cosine": _pairwise_cosine(q1_pairs - q1_mean[None, None, :]),
            "l2": _pairwise_l2(q1_pairs),
            "different_image_cosine": _pairwise_cosine(diff_q1),
            "effective_rank_participation_ratio": participation,
        },
        "correlations": {
            "q0_distance_vs_gt_point_distance": corr(q0_dists, point_dists),
            "q1_distance_vs_gt_point_distance": corr(q1_dists, point_dists),
        },
        "compared_pairs_sampled": len(pool),
        "centering": "global mean over the analysed set (paired + val)",
    }


def attention_vs_localization(pair_rows: list[dict]) -> dict:
    """Section 11: does more attention inside the own target mean correct selection?"""

    own = [row.get("attention_own_mass_mean") for row in pair_rows]
    localized = [1.0 if row.get("point_selection_passed") else 0.0 for row in pair_rows]
    correlation = None
    if len(own) > 1 and len(set(localized)) > 1:
        correlation = float(np.corrcoef(own, localized)[0, 1])
    passed = [row for row in pair_rows if row.get("point_selection_passed")]
    return {
        "pairs": len(pair_rows),
        "correlation_attention_own_mass_vs_paired_localization": correlation,
        "mean_attention_own_mass_when_both_inside": _mean(
            [row.get("attention_own_mass_mean") for row in passed]
        ),
        "mean_attention_own_mass_otherwise": _mean(
            [row.get("attention_own_mass_mean") for row in pair_rows if not row.get("point_selection_passed")]
        ),
    }
