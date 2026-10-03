"""Task 6H evaluation: counterfactual pair ranking, point localization, representation diagnosis.

The pair evaluator runs the real inference path for both instructions of one image against **one
shared frozen SAM2 feature**, computes the four own/cross region scores from the heatmap logits and
reports the pair-ranking and point-localization metrics of sections 10/12/17.
"""

from __future__ import annotations

import numpy as np
import torch

from . import data as data_mod
from .counterfactual import pair_ranking_summary, region_scores
from .dense_eval import forward_heatmap, record_from_raw
from .dense_grounding import downsample_target_mask


@torch.no_grad()
def evaluate_pair(runtime, pair: dict, sample_a, sample_b, *, with_heatmap: bool = True,
                  with_hidden: bool = False) -> dict:
    """One canonical pair through the real path, with the own-vs-cross region scores."""

    image = sample_a.image_rgb()
    if str(sample_b.image_id) != str(sample_a.image_id):
        raise RuntimeError("a canonical counterfactual pair must share one source image")
    features, _cached = runtime.features_for(sample_a, image)
    raw_a = forward_heatmap(runtime, sample_a, image=image, features=features)
    raw_b = forward_heatmap(runtime, sample_b, image=image, features=features)
    record_a = record_from_raw(sample_a, raw_a, with_hidden=with_hidden, with_heatmap=with_heatmap)
    record_b = record_from_raw(sample_b, raw_b, with_hidden=with_hidden, with_heatmap=with_heatmap)
    heatmap_iou = None
    if record_a.get("heatmap_binary") is not None and record_b.get("heatmap_binary") is not None:
        left = np.asarray(record_a["heatmap_binary"], dtype=bool)
        right = np.asarray(record_b["heatmap_binary"], dtype=bool)
        union = float(np.logical_or(left, right).sum())
        heatmap_iou = float(np.logical_and(left, right).sum() / union) if union else 0.0
    scores = region_scores(raw_a["logits"], raw_b["logits"], raw_a["soft_target"], raw_b["soft_target"])
    margin_a = float(scores["margin_a"].detach())
    margin_b = float(scores["margin_b"].detach())
    margin = float(((scores["margin_a"] + scores["margin_b"]) / 2.0).detach())
    return {
        "image_id": str(pair["image_id"]),
        "a": str(pair["a"]),
        "b": str(pair["b"]),
        "target_a": int(pair["target_a"]),
        "target_b": int(pair["target_b"]),
        "a_level": int(pair.get("a_level", 0)),
        "b_level": int(pair.get("b_level", 0)),
        "a_query_type": str(pair.get("a_query_type", "")),
        "b_query_type": str(pair.get("b_query_type", "")),
        "s_aa": scores["raw"]["s_aa"],
        "s_ab": scores["raw"]["s_ab"],
        "s_bb": scores["raw"]["s_bb"],
        "s_ba": scores["raw"]["s_ba"],
        "margin_a": margin_a,
        "margin_b": margin_b,
        "mean_margin": margin,
        "probability_margin_a": scores["raw"]["probability_margin_a"],
        "probability_margin_b": scores["raw"]["probability_margin_b"],
        "pair_ranking_pass": bool(margin_a > 0.0 and margin_b > 0.0),
        "strict_margin_pass": bool(margin_a >= 1.0 and margin_b >= 1.0),
        "point_inside_own_a": bool(record_a["point_inside_own"]),
        "point_inside_own_b": bool(record_b["point_inside_own"]),
        "point_selection_passed": bool(record_a["point_inside_own"] and record_b["point_inside_own"]),
        "same_image_heatmap_iou": heatmap_iou,
        "same_image_point_distance": float(
            sum(
                abs(float(x) - float(y))
                for x, y in zip(record_a["predicted_point"], record_b["predicted_point"])
            )
        ),
        "record_a": record_a,
        "record_b": record_b,
    }


def pair_summary(pair_rows: list[dict]) -> dict:
    """Sections 10/12: ranking, strict-margin and point-selection rates over a pair set."""

    ranking = pair_ranking_summary(pair_rows)
    selected = [row for row in pair_rows if row.get("point_selection_passed")]
    distances = [
        row["same_image_point_distance"]
        for row in pair_rows
        if row.get("same_image_point_distance") is not None
    ]
    heatmap_ious = [
        row["same_image_heatmap_iou"]
        for row in pair_rows
        if row.get("same_image_heatmap_iou") is not None
    ]
    return {
        "pair_count": len(pair_rows),
        **ranking,
        "paired_point_selection_pass": len(selected),
        "paired_point_selection_rate": (len(selected) / len(pair_rows)) if pair_rows else None,
        "mean_same_image_point_distance": (
            float(sum(distances) / len(distances)) if distances else None
        ),
        "mean_same_image_heatmap_iou": (
            float(sum(heatmap_ious) / len(heatmap_ious)) if heatmap_ious else None
        ),
    }


def pair_records_by_sample(pair_rows: list[dict]) -> dict:
    """Flatten the per-pair records so the record-level helpers can be reused."""

    records = {}
    for row in pair_rows:
        records[row["a"]] = row["record_a"]
        records[row["b"]] = row["record_b"]
    return records


def record_spatial_metrics(records: list[dict]) -> dict:
    """Section 12 record-level metrics over already-computed records."""

    from .dense_eval import spatial_metrics

    return spatial_metrics(records)


def gt_point_distance_correlation(pair_rows: list[dict]) -> dict:
    """Section 17: correlations between query/heatmap distances and GT target-point distances."""

    query_l2: list[float] = []
    projected_l2: list[float] = []
    heatmap_point_l2: list[float] = []
    gt_point_l2: list[float] = []
    for row in pair_rows:
        record_a, record_b = row["record_a"], row["record_b"]
        hidden_a = record_a.get("box_hidden")
        hidden_b = record_b.get("box_hidden")
        if hidden_a is None or hidden_b is None:
            continue
        hidden_a = np.asarray(hidden_a, dtype=np.float64)
        hidden_b = np.asarray(hidden_b, dtype=np.float64)
        query_l2.append(float(np.linalg.norm(hidden_a - hidden_b)))
        projected_l2.append(
            float(
                np.linalg.norm(
                    np.asarray(record_a["query_projected"], dtype=np.float64)
                    - np.asarray(record_b["query_projected"], dtype=np.float64)
                )
            )
        )
        heatmap_point_l2.append(float(row["same_image_point_distance"] or 0.0))
        gt_a = np.asarray(record_a["gt_point"], dtype=np.float64)
        gt_b = np.asarray(record_b["gt_point"], dtype=np.float64)
        gt_point_l2.append(float(np.linalg.norm(gt_a - gt_b)))

    def _corr(left: list[float], right: list[float]):
        if len(left) < 2:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "compared_pairs": len(gt_point_l2),
        "query_hidden_distance_vs_gt_point_distance_correlation": _corr(query_l2, gt_point_l2),
        "projected_query_distance_vs_gt_point_distance_correlation": _corr(projected_l2, gt_point_l2),
        "predicted_point_distance_vs_gt_point_distance_correlation": _corr(heatmap_point_l2, gt_point_l2),
        "mean_gt_point_distance": (float(np.mean(gt_point_l2)) if gt_point_l2 else None),
        "mean_query_hidden_distance": (float(np.mean(query_l2)) if query_l2 else None),
        "mean_projected_query_distance": (float(np.mean(projected_l2)) if projected_l2 else None),
    }


@torch.no_grad()
def mask_overlap_pairs(runtime, triples, grid: int) -> list[dict]:
    """Downsampled soft-target overlap for every canonical pair (never discarded)."""

    rows = []
    for pair, record_a, record_b in triples:
        sample_a = data_mod.to_sample(record_a)
        sample_b = data_mod.to_sample(record_b)
        soft_a = downsample_target_mask(sample_a.target_mask(), grid)
        soft_b = downsample_target_mask(sample_b.target_mask(), grid)
        intersection = float((soft_a * soft_b).sum())
        union = float(soft_a.sum() + soft_b.sum() - intersection)
        rows.append(
            {
                "image_id": pair.image_id,
                "a": pair.a,
                "b": pair.b,
                "soft_intersection": intersection,
                "soft_iou": (intersection / union) if union > 0 else 0.0,
                "overlapping": bool(intersection > 0.0),
                "mass_a": float(soft_a.sum()),
                "mass_b": float(soft_b.sum()),
            }
        )
    return rows
