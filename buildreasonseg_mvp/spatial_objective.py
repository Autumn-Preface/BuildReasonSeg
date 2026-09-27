"""Task 6H.1: spatial-softmax point supervision + bounded counterfactual grounding.

Task 6H's own-vs-cross ranking was defined on **unbounded logits**, so it could be minimized by
magnifying the field (mean |logit| 0.148 → 16.32) without moving the heatmap peak onto the target.
Task 6H.1 keeps the Task 6G/6H architecture frozen and replaces only the spatial objective with one
that is directly aligned to the deployed inference operation (`argmax` of the heatmap):

    target point-cell : frozen Task 6D interior point -> 256x256 cell -> index = y*256 + x
    L_point           = CrossEntropy(H.flatten(), target_index)          (65,536 spatial classes)
    P                 = softmax(H.flatten())                             (sums to 1)
    mass(P, M)        = sum(P * M)                                       (bounded in [0,1], no area division)
    L_cf              = 0.5*(-log((p_AA+eps)/(p_AA+p_AB+2eps)) - log((p_BB+eps)/(p_BB+p_BA+2eps)))

Task 6G's BCE+Dice and Task 6H's raw-logit ranking are **logged only** and contribute zero gradient
(sections 8-9). No architecture change: no extra query token, no coordinate channels, no
cross-attention, no visual LoRA, no larger decoder, no new dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .dense_grounding import cell_centre, snap_point_to_cell
from .grounding import distance_transform_point

#: Section 7 fixes the stabiliser; no margin hyperparameter exists.
BOUNDED_CF_EPS = 1e-8

#: Section 11 fixes the weights; no sweep.
POINT_REASONING_WEIGHT = 0.5
POINT_CLASSIFICATION_WEIGHT = 1.0
POINT_CF_WEIGHT = 1.0

#: Section 13 H0-R gate constants (also recorded in the config).
MAX_MEAN_ABS_LOGIT = 10.0


# ------------------------------------------------------------------ target point-cell


@dataclass(frozen=True)
class TargetCell:
    """The deterministic spatial training target of one sample."""

    index: int
    x_cell: int
    y_cell: int
    point: tuple[float, float]
    cell_centre: tuple[float, float]
    grid: int

    def as_dict(self) -> dict:
        return {
            "index": int(self.index),
            "x_cell": int(self.x_cell),
            "y_cell": int(self.y_cell),
            "point": [float(value) for value in self.point],
            "cell_centre": [float(value) for value in self.cell_centre],
            "grid": int(self.grid),
        }


def target_cell(mask, grid: int) -> TargetCell:
    """Frozen interior point -> Task 6G cell snap -> flat spatial index `y*grid + x`."""

    grid = int(grid)
    point = distance_transform_point(mask)
    x_cell, y_cell = snap_point_to_cell(point, grid)
    return TargetCell(
        index=int(y_cell) * grid + int(x_cell),
        x_cell=int(x_cell),
        y_cell=int(y_cell),
        point=(float(point[0]), float(point[1])),
        cell_centre=(cell_centre(x_cell, grid), cell_centre(y_cell, grid)),
        grid=grid,
    )


def index_to_cell(index: int, grid: int) -> tuple[int, int]:
    """Inverse of the `y*grid + x` convention (used by every metric here)."""

    y, x = divmod(int(index), int(grid))
    return int(x), int(y)


# ------------------------------------------------------------------ spatial softmax


def flatten_spatial(logits: torch.Tensor) -> torch.Tensor:
    """`[..., H, W]` (or `[N]`) -> `[N]` spatial logits."""

    return logits.reshape(-1)


def spatial_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """`P = softmax(H.flatten())`, returned with the original `[H, W]` shape (section 6)."""

    shape = tuple(logits.shape)
    flat = flatten_spatial(logits)
    probabilities = torch.softmax(flat.float(), dim=-1)
    return probabilities.reshape(shape)


def region_mass(probabilities: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
    """`mass(P, M) = sum(P * M)` — deliberately **not** divided by the target area (section 6)."""

    return (probabilities.float() * soft_mask.float()).sum()


def point_cross_entropy(logits: torch.Tensor, target_index: int) -> torch.Tensor:
    """Section 5: cross-entropy over the 65,536 flattened spatial classes."""

    flat = flatten_spatial(logits).float()
    target = torch.tensor([int(target_index)], dtype=torch.long, device=flat.device)
    return F.cross_entropy(flat.unsqueeze(0), target)


def spatial_entropy(probabilities: torch.Tensor) -> float:
    """Shannon entropy of the spatial distribution, in nats (diagnostic)."""

    flat = flatten_spatial(probabilities).float().clamp(min=1e-12)
    return float(-(flat * flat.log()).sum())


def target_cell_probability(probabilities: torch.Tensor, target_index: int) -> float:
    return float(flatten_spatial(probabilities)[int(target_index)])


def topk_hit(logits: torch.Tensor, target_index: int, k: int) -> bool:
    """Whether the GT cell is among the top-`k` spatial logits."""

    flat = flatten_spatial(logits).float()
    k = min(int(k), flat.numel())
    top = torch.topk(flat, k).indices
    return bool((top == int(target_index)).any())


def max_non_target_probability(probabilities: torch.Tensor, soft_mask: torch.Tensor) -> float:
    """Diagnostic: the largest probability outside the own target region."""

    flat = flatten_spatial(probabilities).float()
    mask = flatten_spatial(soft_mask).float() > 0.0
    outside = flat.clone()
    outside[mask] = 0.0
    return float(outside.max())


# ------------------------------------------------------------------ bounded pair loss


def bounded_pair_loss(p_aa, p_ab, p_bb, p_ba, eps: float = BOUNDED_CF_EPS) -> dict:
    """Section 7: bounded own-vs-cross preference, no margin."""

    p_aa = torch.as_tensor(p_aa, dtype=torch.float32)
    p_ab = torch.as_tensor(p_ab, dtype=torch.float32)
    p_bb = torch.as_tensor(p_bb, dtype=torch.float32)
    p_ba = torch.as_tensor(p_ba, dtype=torch.float32)
    loss_a = -torch.log((p_aa + eps) / (p_aa + p_ab + 2.0 * eps))
    loss_b = -torch.log((p_bb + eps) / (p_bb + p_ba + 2.0 * eps))
    loss = 0.5 * (loss_a + loss_b)
    return {
        "loss": loss,
        "loss_a": loss_a,
        "loss_b": loss_b,
        "raw": float(loss.detach()),
        "raw_a": float(loss_a.detach()),
        "raw_b": float(loss_b.detach()),
        "eps": float(eps),
    }


def pair_preference_pass(p_aa, p_ab, p_bb, p_ba) -> bool:
    """Section 7: pair ranking passes iff `p_AA > p_AB` and `p_BB > p_BA`."""

    return bool(float(p_aa) > float(p_ab) and float(p_bb) > float(p_ba))


def counterfactual_masses(probabilities_a: torch.Tensor, probabilities_b: torch.Tensor,
                          mask_a: torch.Tensor, mask_b: torch.Tensor) -> dict:
    """The four bounded own/cross masses plus margins and ratios (sections 6-7)."""

    p_aa = region_mass(probabilities_a, mask_a)
    p_ab = region_mass(probabilities_a, mask_b)
    p_bb = region_mass(probabilities_b, mask_b)
    p_ba = region_mass(probabilities_b, mask_a)
    margin_a = p_aa - p_ab
    margin_b = p_bb - p_ba
    return {
        "p_aa": p_aa,
        "p_ab": p_ab,
        "p_bb": p_bb,
        "p_ba": p_ba,
        "margin_a": margin_a,
        "margin_b": margin_b,
        "mean_margin": 0.5 * (margin_a + margin_b),
        "raw": {
            "p_aa": float(p_aa.detach()),
            "p_ab": float(p_ab.detach()),
            "p_bb": float(p_bb.detach()),
            "p_ba": float(p_ba.detach()),
            "margin_a": float(margin_a.detach()),
            "margin_b": float(margin_b.detach()),
            "mean_margin": float((0.5 * (margin_a + margin_b)).detach()),
            "ratio_a": float((p_aa / (p_aa + p_ab + 2 * BOUNDED_CF_EPS)).detach()),
            "ratio_b": float((p_bb / (p_bb + p_ba + 2 * BOUNDED_CF_EPS)).detach()),
            "bounded": bool(
                0.0 <= float(p_aa.detach()) <= 1.0
                and 0.0 <= float(p_ab.detach()) <= 1.0
                and 0.0 <= float(p_bb.detach()) <= 1.0
                and 0.0 <= float(p_ba.detach()) <= 1.0
            ),
        },
        "pair_preference_pass": pair_preference_pass(p_aa, p_ab, p_bb, p_ba),
    }


def point_pair_step_loss(reasoning_a: torch.Tensor, reasoning_b: torch.Tensor,
                         point_a: torch.Tensor, point_b: torch.Tensor,
                         cf: dict) -> dict:
    """Section 11: `0.5*(r_A+r_B) + 1.0*(L_point_A+L_point_B) + 1.0*L_cf`."""

    reasoning = 0.5 * (reasoning_a + reasoning_b)
    classification = point_a + point_b
    total = (
        POINT_REASONING_WEIGHT * reasoning
        + POINT_CLASSIFICATION_WEIGHT * classification
        + POINT_CF_WEIGHT * cf["loss"]
    )
    return {
        "total": total,
        "reasoning": reasoning,
        "classification": classification,
        "cf": cf,
        "raw": {
            "total": float(total.detach()),
            "reasoning_ce": float(reasoning.detach()),
            "point_ce_sum": float(classification.detach()),
            "cf": cf["raw"],
        },
        "weights": {
            "reasoning": POINT_REASONING_WEIGHT,
            "classification": POINT_CLASSIFICATION_WEIGHT,
            "cf": POINT_CF_WEIGHT,
        },
    }


# ------------------------------------------------------------------ pair summary


def pair_point_summary(rows: list[dict]) -> dict:
    """Sections 13/16: bounded pair ranking and probability-mass statistics."""

    ranked = [row for row in rows if row.get("pair_preference_pass")]
    margins = [row["mean_margin"] for row in rows if row.get("mean_margin") is not None]
    return {
        "pair_count": len(rows),
        "pair_ranking_pass": len(ranked),
        "pair_ranking_rate": (len(ranked) / len(rows)) if rows else None,
        "mean_p_aa": _mean([row.get("p_aa") for row in rows]),
        "mean_p_ab": _mean([row.get("p_ab") for row in rows]),
        "mean_p_bb": _mean([row.get("p_bb") for row in rows]),
        "mean_p_ba": _mean([row.get("p_ba") for row in rows]),
        "mean_own_mass": _mean(
            [
                0.5 * (row["p_aa"] + row["p_bb"])
                for row in rows
                if row.get("p_aa") is not None and row.get("p_bb") is not None
            ]
        ),
        "mean_cross_mass": _mean(
            [
                0.5 * (row["p_ab"] + row["p_ba"])
                for row in rows
                if row.get("p_ab") is not None and row.get("p_ba") is not None
            ]
        ),
        "mean_own_minus_cross_probability_mass": _mean(margins),
        "min_own_minus_cross_probability_mass": (float(min(margins)) if margins else None),
        "max_own_minus_cross_probability_mass": (float(max(margins)) if margins else None),
    }


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def point_record_summary(records: list[dict]) -> dict:
    """Sections 13/16: point-localization metrics over record dicts (counts and rates)."""

    return {
        "count": len(records),
        "point_inside_count": sum(1 for record in records if record.get("point_inside_own")),
        "point_inside_target_rate": _mean(
            [1.0 if record.get("point_inside_own") else 0.0 for record in records]
        ),
        "target_cell_top1_count": sum(1 for record in records if record.get("target_cell_top1")),
        "target_cell_top1_rate": _mean(
            [1.0 if record.get("target_cell_top1") else 0.0 for record in records]
        ),
        "target_cell_top5_count": sum(1 for record in records if record.get("target_cell_top5")),
        "target_cell_top5_rate": _mean(
            [1.0 if record.get("target_cell_top5") else 0.0 for record in records]
        ),
        "target_cell_top25_count": sum(1 for record in records if record.get("target_cell_top25")),
        "target_cell_top25_rate": _mean(
            [1.0 if record.get("target_cell_top25") else 0.0 for record in records]
        ),
        "mean_target_cell_probability": _mean(
            [record.get("target_cell_probability") for record in records]
        ),
        "mean_512px_point_error": _mean([record.get("error_512px") for record in records]),
        "mean_normalized_point_error": _mean([record.get("normalized_error") for record in records]),
        "mean_spatial_entropy": _mean([record.get("spatial_entropy") for record in records]),
        "by_level": _breakdown(records, "level"),
        "by_query_family": _breakdown(records, "query_family"),
    }


def _breakdown(records: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(str(record.get(key)), []).append(record)
    return {
        name: {
            "count": len(items),
            "point_inside_rate": _mean(
                [1.0 if item.get("point_inside_own") else 0.0 for item in items]
            ),
            "target_cell_top1_rate": _mean(
                [1.0 if item.get("target_cell_top1") else 0.0 for item in items]
            ),
            "mean_512px_point_error": _mean([item.get("error_512px") for item in items]),
        }
        for name, items in sorted(buckets.items())
    }


# ------------------------------------------------------------------ evaluation


@torch.no_grad()
def point_record(runtime, sample, *, with_hidden: bool = False) -> dict:
    """One record through the real inference path, scored for the point objective."""

    from .dense_eval import forward_heatmap

    raw = forward_heatmap(runtime, sample)
    return point_record_from_raw(runtime, sample, raw, with_hidden=with_hidden)


@torch.no_grad()
def evaluate_point_pair(runtime, pair: dict, sample_a, sample_b, *,
                        with_hidden: bool = False) -> dict:
    """One canonical pair scored by the bounded objective and by point localization."""

    from .dense_eval import forward_heatmap

    image = sample_a.image_rgb()
    if str(sample_b.image_id) != str(sample_a.image_id):
        raise RuntimeError("a canonical counterfactual pair must share one source image")
    features, _cached = runtime.features_for(sample_a, image)
    raw_a = forward_heatmap(runtime, sample_a, image=image, features=features)
    raw_b = forward_heatmap(runtime, sample_b, image=image, features=features)
    record_a = point_record_from_raw(runtime, sample_a, raw_a, with_hidden=with_hidden)
    record_b = point_record_from_raw(runtime, sample_b, raw_b, with_hidden=with_hidden)

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
        "mean_abs_logit": 0.5 * (record_a["mean_abs_logit"] + record_b["mean_abs_logit"]),
        "spatial_entropy": 0.5 * (record_a["spatial_entropy"] + record_b["spatial_entropy"]),
        "record_a": record_a,
        "record_b": record_b,
    }


def point_record_from_raw(runtime, sample, raw: dict, *, with_hidden: bool = False) -> dict:
    """`point_record` for an already-computed `forward_heatmap` result (shared feature reuse)."""

    from .dense_eval import record_from_raw

    record = record_from_raw(sample, raw, with_hidden=with_hidden)
    cell = target_cell(sample.target_mask(), int(raw["grid"]))
    logits = raw["logits"]
    probabilities = spatial_probabilities(logits)
    record.update(
        {
            "target_index": int(cell.index),
            "target_cell": [int(cell.x_cell), int(cell.y_cell)],
            "target_cell_centre": [float(value) for value in cell.cell_centre],
            "gt_point": [float(value) for value in cell.point],
            "target_cell_top1": topk_hit(logits, cell.index, 1),
            "target_cell_top5": topk_hit(logits, cell.index, 5),
            "target_cell_top25": topk_hit(logits, cell.index, 25),
            "target_cell_probability": target_cell_probability(probabilities, cell.index),
            "point_ce": float(point_cross_entropy(logits, cell.index)),
            "spatial_entropy": spatial_entropy(probabilities),
            "mean_abs_logit": float(logits.abs().mean()),
            "logit_std": float(logits.std()),
            "max_non_target_probability": max_non_target_probability(
                probabilities, raw["soft_target"]
            ),
            "argmax_point_inside_target": record["point_inside_own"],
        }
    )
    return record


def pair_probability_summary(rows: list[dict]) -> dict:
    """Sections 13/16: bounded pair-preference statistics over pair rows."""

    summary = pair_point_summary(rows)
    summary["mean_same_image_probability_map_overlap"] = _mean(
        [row.get("same_image_probability_map_overlap") for row in rows]
    )
    summary["mean_same_image_probability_binary_iou"] = _mean(
        [row.get("same_image_probability_binary_iou") for row in rows]
    )
    summary["mean_same_image_point_distance"] = _mean(
        [row.get("same_image_point_distance") for row in rows]
    )
    selected = [row for row in rows if row.get("point_selection_passed")]
    summary["paired_point_selection_pass"] = len(selected)
    summary["paired_point_selection_rate"] = (len(selected) / len(rows)) if rows else None
    summary["mean_cf_loss"] = _mean([row.get("cf_loss") for row in rows])
    summary["mean_abs_logit"] = _mean([row.get("mean_abs_logit") for row in rows])
    summary["mean_spatial_entropy"] = _mean([row.get("spatial_entropy") for row in rows])
    return summary


def margin_vs_localization(rows: list[dict]) -> dict:
    """Section 19: does a larger bounded own-minus-cross mass mean correct selection?"""

    margins = [float(row["mean_margin"]) for row in rows if row.get("mean_margin") is not None]
    localized = [
        1.0 if row.get("point_selection_passed") else 0.0
        for row in rows
        if row.get("mean_margin") is not None
    ]
    correlation = None
    if len(margins) > 1 and len(set(localized)) > 1:
        correlation = float(np.corrcoef(margins, localized)[0, 1])
    passed = [row for row in rows if row.get("point_selection_passed")]
    return {
        "pairs": len(margins),
        "correlation_probability_margin_vs_paired_localization": correlation,
        "mean_margin_when_both_inside": _mean([row["mean_margin"] for row in passed]),
        "mean_margin_otherwise": _mean(
            [row["mean_margin"] for row in rows if not row.get("point_selection_passed")]
        ),
    }
