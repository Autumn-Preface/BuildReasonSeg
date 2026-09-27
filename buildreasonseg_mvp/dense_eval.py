"""Task 6G evaluation: heatmap metrics, paired point probe, representation/fusion diagnosis, G2.

Every function here runs the real inference path — `image + instruction -> fixed [BOX] query ->
dense heatmap -> argmax point` — with GT used exclusively for scoring (sections 9, 14, 15).
"""

from __future__ import annotations

import numpy as np
import torch

from .box_query import box_query_forward
from .dense_grounding import (
    argmax_point_from_logits,
    binary_iou_from_logits,
    downsample_target_mask,
    feature_tensor_for_grid,
    normalized_point_error,
    point_inside_mask,
    soft_dice_from_logits,
)
from .grounding import decode_mask_from_geometry, distance_transform_point
from .metrics import dice_from_logits, mask_iou_from_logits, upsample_logits
from .validation import family_of


@torch.no_grad()
def predict_heatmap(runtime, sample, image=None, features=None, *,
                    with_hidden: bool = False, with_heatmap: bool = False) -> dict:
    """The section 9 inference path: query hidden x frozen spatial features -> heatmap + point.

    `with_hidden` / `with_heatmap` gate the storage of the 2048-d hiddens and the binary heatmap
    (used by the representation diagnosis and the paired probe); the per-epoch evaluations keep
    them off so the training artifact stays small. All stored arrays are plain lists for JSON.
    """

    from .box_query import build_query_batch

    if image is None:
        image = sample.image_rgb()
    if features is None:
        features, _cached = runtime.features_for(sample, image)
    runtime.set_visual_cache_key(str(sample.image_id))
    batch = build_query_batch(runtime, image, sample.instruction_zh).to(runtime.device)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=bool(runtime.cfg.get("training", {}).get("bf16_autocast", True))
    ):
        _logits, box_hidden, _seg_hidden = box_query_forward(runtime.model.qwen, batch)
        spatial_feature = feature_tensor_for_grid(features, int(runtime.dense_grid)).to(runtime.device)
        heatmap_logits = runtime.model.dense_head(box_hidden, spatial_feature)
    logits = heatmap_logits[0].detach().float().cpu()
    soft_target = downsample_target_mask(sample.target_mask(), int(runtime.dense_grid))
    extraction = argmax_point_from_logits(logits, int(runtime.dense_grid))
    gt_point = distance_transform_point(sample.target_mask())
    binary = (torch.sigmoid(logits.float()) > 0.5).detach().cpu().numpy()
    record = {
        "sample_id": str(sample.sample_id),
        "image_id": str(sample.image_id),
        "level": int(sample.level),
        "query_type": str(sample.query_type),
        "query_family": family_of(str(sample.query_type)),
        "grid": int(runtime.dense_grid),
        "predicted_point": extraction["point"],
        "soft_argmax_point": extraction["soft_argmax_point"],
        "peakiness": extraction["peakiness"],
        "gt_point": list(gt_point),
        "point_inside_own": point_inside_mask(sample.target_mask(), extraction["point"]),
        **normalized_point_error(extraction["point"], gt_point),
        "heatmap_soft_dice": soft_dice_from_logits(logits, soft_target),
        "heatmap_binary_iou_05": binary_iou_from_logits(logits, soft_target),
    }
    if with_hidden:
        record["box_hidden"] = box_hidden[0].detach().float().cpu().numpy().astype(np.float32).tolist()
        record["query_projected"] = (
            runtime.model.dense_head.query_proj(runtime.model.dense_head.query_norm(box_hidden.float()))[0]
            .detach().float().cpu().numpy().astype(np.float32).tolist()
        )
    if with_heatmap:
        record["heatmap_binary"] = binary.astype(bool).tolist()
    return record


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def _breakdown(records: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(str(record[key]), []).append(record)
    return {
        name: {
            "count": len(items),
            "inside_rate": _mean([1.0 if item["point_inside_own"] else 0.0 for item in items]),
            "mean_heatmap_dice": _mean([item["heatmap_soft_dice"] for item in items]),
            "mean_512px_error": _mean([item["error_512px"] for item in items]),
        }
        for name, items in sorted(buckets.items())
    }


def spatial_metrics(records: list[dict]) -> dict:
    """Section 14 metrics over query-path heatmap records."""

    return {
        "count": len(records),
        "mean_heatmap_binary_iou_05": _mean([item["heatmap_binary_iou_05"] for item in records]),
        "mean_heatmap_dice": _mean([item["heatmap_soft_dice"] for item in records]),
        "point_inside_target_rate": _mean([1.0 if item["point_inside_own"] else 0.0 for item in records]),
        "normalized_point_error": _mean([item["normalized_error"] for item in records]),
        "mean_512px_point_error": _mean([item["error_512px"] for item in records]),
        "mean_peakiness": _mean([item["peakiness"] for item in records]),
        "by_level": _breakdown(records, "level"),
        "by_query_family": _breakdown(records, "query_family"),
    }


def paired_point_report(records_by_id: dict[str, dict], pairs: list[dict]) -> dict:
    """Section 14 paired probe: point selection, same-image point distance, heatmap A/B IoU."""

    def _iou_binary(a: np.ndarray, b: np.ndarray) -> float:
        left = np.asarray(a).astype(bool)
        right = np.asarray(b).astype(bool)
        union = np.logical_or(left, right).sum()
        return float(np.logical_and(left, right).sum() / union) if union else 0.0

    rows = []
    for pair in pairs:
        a = records_by_id.get(str(pair["a"]))
        b = records_by_id.get(str(pair["b"]))
        if a is None or b is None:
            continue
        distance = float(
            sum(abs(float(x) - float(y)) for x, y in zip(a["predicted_point"], b["predicted_point"]))
        )
        rows.append(
            {
                "image_id": str(pair["image_id"]),
                "a": str(pair["a"]),
                "b": str(pair["b"]),
                "a_inside": a["point_inside_own"],
                "b_inside": b["point_inside_own"],
                "point_selection_passed": bool(a["point_inside_own"] and b["point_inside_own"]),
                "same_image_point_distance": distance,
                "points_distinct": bool(distance > 0.0),
                "same_image_heatmap_iou": _iou_binary(a["heatmap_binary"], b["heatmap_binary"]),
            }
        )
    return {
        "paired_total": len(rows),
        "paired_point_selection_pass": sum(1 for row in rows if row["point_selection_passed"]),
        "mean_same_image_point_distance": _mean([row["same_image_point_distance"] for row in rows]),
        "pairs_with_distinct_points": sum(1 for row in rows if row["points_distinct"]),
        "mean_same_image_heatmap_iou": _mean([row["same_image_heatmap_iou"] for row in rows]),
        "rows": rows,
    }


# ------------------------------------------------------------------ representation


def _pairwise_cosine(vectors: np.ndarray) -> float | None:
    if vectors is None or vectors.shape[0] == 0:
        return None
    left = vectors[:, 0, :].astype(np.float64)
    right = vectors[:, 1, :].astype(np.float64)
    left = left / (np.linalg.norm(left, axis=-1, keepdims=True) + 1e-12)
    right = right / (np.linalg.norm(right, axis=-1, keepdims=True) + 1e-12)
    return float((left * right).sum(axis=-1).mean())


def representation_report(paired_records: list[dict], val_records: list[dict], pairs: list[dict],
                          *, rng_seed: int = 20260926) -> dict:
    """Section 15: [BOX] hidden + projected q + heatmap diagnostics at the best G1 checkpoint."""

    paired_by_id = {record["sample_id"]: record for record in paired_records}
    hiddens = np.stack(
        [np.asarray(record["box_hidden"], dtype=np.float64) for record in val_records], axis=0
    )
    queries = np.stack(
        [np.asarray(record["query_projected"], dtype=np.float64) for record in val_records], axis=0
    )
    points = np.asarray([record["gt_point"] for record in val_records], dtype=np.float64)
    heatmap_points = np.asarray([record["predicted_point"] for record in val_records], dtype=np.float64)

    def _hidden(record) -> np.ndarray:
        return np.asarray(record["box_hidden"], dtype=np.float64)

    def _query(record) -> np.ndarray:
        return np.asarray(record["query_projected"], dtype=np.float64)

    def pair_stack(records, key) -> np.ndarray:
        rows = []
        for pair in pairs:
            a = records.get(str(pair["a"]))
            b = records.get(str(pair["b"]))
            if a is None or b is None:
                continue
            rows.append([key(a), key(b)])
        if rows:
            return np.stack(rows)
        return np.zeros((0, 2, 1))

    hidden_pairs = pair_stack(paired_by_id, _hidden)
    query_pairs = pair_stack(paired_by_id, _query)

    rng = np.random.default_rng(int(rng_seed))
    pool: list[tuple[int, int]] = []
    for _ in range(min(200, len(val_records) * max(len(val_records) - 1, 0) // 2)):
        i = int(rng.integers(len(val_records)))
        j = int(rng.integers(len(val_records)))
        if i == j:
            continue
        pool.append((i, j))
    diff_pairs = np.stack([[hiddens[i], hiddens[j]] for i, j in pool]) if pool else np.zeros((0, 2, hiddens.shape[-1]))
    diff_query_pairs = np.stack([[queries[i], queries[j]] for i, j in pool]) if pool else np.zeros((0, 2, queries.shape[-1]))

    all_hiddens = np.vstack([hidden_pairs.reshape(-1, hidden_pairs.shape[-1]), hiddens])
    global_mean = all_hiddens.mean(axis=0, keepdims=True)
    hidden_centered_pairs = (
        np.stack([[_hidden(paired_by_id[str(pair["a"])]) - global_mean[0],
                   _hidden(paired_by_id[str(pair["b"])]) - global_mean[0]]
                  for pair in pairs if str(pair["a"]) in paired_by_id and str(pair["b"]) in paired_by_id])
        if hidden_pairs.shape[0]
        else np.zeros((0, 2, hiddens.shape[-1]))
    )
    diff_centered = (
        np.stack([[hiddens[i] - global_mean[0], hiddens[j] - global_mean[0]] for i, j in pool])
        if pool
        else np.zeros((0, 2, hiddens.shape[-1]))
    )

    centered_matrix = hiddens - hiddens.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered_matrix, full_matrices=False)
    participation = float((singular_values**2).sum() ** 2 / max((singular_values**4).sum(), 1e-12))

    hidden_dists = []
    heatmap_dists = []
    point_dists = []
    for i, j in pool:
        hidden_dists.append(float(np.linalg.norm(hiddens[i] - hiddens[j])))
        heatmap_dists.append(float(np.linalg.norm(heatmap_points[i] - heatmap_points[j])))
        point_dists.append(float(np.linalg.norm(points[i] - points[j])))

    def corr(left, right):
        if len(left) < 2:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    same_image_query_l2 = []
    for pair in pairs:
        a = paired_by_id.get(str(pair["a"]))
        b = paired_by_id.get(str(pair["b"]))
        if a is None or b is None:
            continue
        same_image_query_l2.append(float(np.linalg.norm(_query(a) - _query(b))))

    return {
        "paired_sample_count": len(paired_records),
        "val_sample_count": len(val_records),
        "same_image_different_instruction_cosine": _pairwise_cosine(hidden_pairs),
        "same_image_different_instruction_centered_cosine": _pairwise_cosine(hidden_centered_pairs),
        "different_image_cosine": _pairwise_cosine(diff_pairs),
        "different_image_centered_cosine": _pairwise_cosine(diff_centered),
        "mean_hidden_l2_norm": float(np.linalg.norm(hiddens, axis=-1).mean()),
        "effective_rank_participation_ratio": participation,
        "projected_query_same_image_l2": _mean(same_image_query_l2),
        "projected_query_different_image_cosine": _pairwise_cosine(diff_query_pairs),
        "query_hidden_distance_vs_gt_point_distance_correlation": corr(hidden_dists, point_dists),
        "heatmap_point_distance_vs_gt_point_distance_correlation": corr(heatmap_dists, point_dists),
        "compared_pairs_sampled": len(pool),
        "centering": "global mean over the analysed hidden set (paired + val)",
        "task6f_reference": {
            "same_image_centered_cosine": 0.626,
            "different_image_centered_cosine": 0.018,
            "effective_rank": 7.07,
            "hidden_distance_vs_gt_box_distance_correlation": 0.476,
            "source": "evaluation/task6f_representation.json",
        },
    }


# ------------------------------------------------------------------ G2


@torch.no_grad()
def segment_from_predicted_point(runtime, sample, predicted_point, *, image=None) -> dict:
    """Official frozen SAM2 positive-point prompt from the heatmap argmax point (section 16)."""

    if image is None:
        image = sample.image_rgb()
    features, cached = runtime.features_for_image(image, image_id=getattr(sample, "image_id", None))
    geometry = torch.tensor([float(value) for value in predicted_point], dtype=torch.float32)
    decoded = decode_mask_from_geometry(runtime.model.sam, features, geometry, "point", multimask_output=False)
    logits = decoded.low_res_logits
    target = np.asarray(sample.target_mask()).astype(np.float32)
    logits = upsample_logits(logits, (int(target.shape[0]), int(target.shape[1])), mode="bilinear")
    target_tensor = torch.as_tensor(target)
    miou = float(mask_iou_from_logits(logits, target_tensor, tuple(target.shape), threshold=0.0))
    dice = float(dice_from_logits(logits, target_tensor, tuple(target.shape), threshold=0.0))
    mask = (logits[0, 0] > 0).detach().cpu().numpy().reshape(int(target.shape[0]), int(target.shape[1]))
    return {"miou": miou, "dice": dice, "cached_features": bool(cached), "mask": mask}


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@torch.no_grad()
def g2_report(runtime, val_samples, pairs, records_by_id: dict[str, dict]) -> dict:
    """Section 17 metrics: strict end-to-end segmentation from predicted points only."""

    from .data import to_sample

    rows = []
    for sample in val_samples:
        record = records_by_id.get(str(sample.sample_id))
        if record is None:
            continue
        segmentation = segment_from_predicted_point(runtime, sample, record["predicted_point"])
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                "image_id": str(sample.image_id),
                "level": int(sample.level),
                "query_family": family_of(str(sample.query_type)),
                "point_inside": record["point_inside_own"],
                "strict_miou": segmentation["miou"],
                "dice": segmentation["dice"],
            }
        )

    pair_rows = []
    for pair in pairs:
        a = records_by_id.get(str(pair["a"]))
        b = records_by_id.get(str(pair["b"]))
        if a is None or b is None:
            continue
        sample_a = to_sample(pair["record_a"]) if pair.get("record_a") is not None else None
        sample_b = to_sample(pair["record_b"]) if pair.get("record_b") is not None else None
        if sample_a is None or sample_b is None:
            continue
        mask_a = segment_from_predicted_point(runtime, sample_a, a["predicted_point"])["mask"]
        mask_b = segment_from_predicted_point(runtime, sample_b, b["predicted_point"])["mask"]
        target_a = np.asarray(sample_a.target_mask()).astype(bool)
        target_b = np.asarray(sample_b.target_mask()).astype(bool)
        own = 0.5 * (_mask_iou(mask_a, target_a) + _mask_iou(mask_b, target_b))
        cross = 0.5 * (_mask_iou(mask_a, target_b) + _mask_iou(mask_b, target_a))
        between = _mask_iou(mask_a, mask_b)
        pair_rows.append(
            {
                "image_id": str(pair["image_id"]),
                "a": str(pair["a"]),
                "b": str(pair["b"]),
                "mask_paired_pass": bool(own > cross),
                "own_iou": float(own),
                "cross_iou": float(cross),
                "own_minus_cross_margin": float(own - cross),
                "mask_iou_between_predictions": float(between),
            }
        )

    def by_key(key: str) -> dict:
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(str(row[key]), []).append(row)
        return {
            name: {"count": len(items), "strict_miou": _mean([item["strict_miou"] for item in items])}
            for name, items in sorted(buckets.items())
        }

    return {
        "count": len(rows),
        "strict_end_to_end_miou": _mean([row["strict_miou"] for row in rows]) or 0.0,
        "mean_dice": _mean([row["dice"] for row in rows]),
        "by_level": by_key("level"),
        "by_query_family": by_key("query_family"),
        "records": rows,
        "paired": {
            "paired_total": len(pair_rows),
            "mask_paired_pass": sum(1 for row in pair_rows if row["mask_paired_pass"]),
            "mean_own_iou": _mean([row["own_iou"] for row in pair_rows]),
            "mean_cross_iou": _mean([row["cross_iou"] for row in pair_rows]),
            "mean_own_minus_cross_margin": _mean([row["own_minus_cross_margin"] for row in pair_rows]),
            "mean_mask_iou_between_predictions": _mean(
                [row["mask_iou_between_predictions"] for row in pair_rows]
            ),
            "rows": pair_rows,
        },
    }
