"""Task 6F evaluation: geometry metrics, paired probe, representation diagnosis, F2.

Every function here runs the **inference-form query path** — `prompt + constant [BOX]` only,
then `TargetAwareBoxHead` — so no future reasoning token ever enters the query representation
(section 9), and GT geometry is used exclusively for scoring (section 6).
"""

from __future__ import annotations

import numpy as np
import torch

from .box_query import (
    box_l1,
    box_values_iou,
    center_inside_mask,
    generate_with_box_prefix,
    predict_box_for_sample,
)
from .grounding import decode_mask_from_geometry, target_geometry
from .metrics import dice_from_logits, mask_iou_from_logits, upsample_logits
from .validation import family_of


def box_record(runtime, sample, image=None, *, with_hidden: bool = False) -> dict:
    """One record through the query path, scored against its own GT box/mask.

    `with_hidden` stores the 2048-d query hidden (as a list) for the section 13 representation
    diagnosis; the per-epoch training evaluations keep it off so the training artifact stays small.
    """

    prediction = predict_box_for_sample(runtime, sample, image=image)
    gt_box = [float(value) for value in target_geometry(sample.target_mask(), "box")]
    box_iou = box_values_iou(prediction["predicted_box"], gt_box)
    center_inside = center_inside_mask(sample.target_mask(), prediction["predicted_box"])
    coordinate_mae = float(
        np.abs(np.asarray(prediction["predicted_box"]) - np.asarray(gt_box)).mean()
    )
    record = {
        "sample_id": str(sample.sample_id),
        "image_id": str(sample.image_id),
        "level": int(sample.level),
        "query_type": str(sample.query_type),
        "query_family": family_of(str(sample.query_type)),
        "predicted_box": prediction["predicted_box"],
        "gt_box": gt_box,
        "box_iou": float(box_iou),
        "center_inside": bool(center_inside),
        "coordinate_mae": coordinate_mae,
        "canonical": prediction["canonical"],
        "in_range": prediction["in_range"],
    }
    if with_hidden:
        record["box_hidden"] = prediction["box_hidden"].numpy().astype(np.float32).tolist()
    return record


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def _median(values) -> float | None:
    values = [value for value in values if value is not None]
    return float(np.median(values)) if values else None


def _breakdown(records: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(str(record[key]), []).append(record)
    return {
        name: {
            "count": len(items),
            "mean_box_iou": _mean([item["box_iou"] for item in items]),
            "center_inside_rate": _mean([1.0 if item["center_inside"] else 0.0 for item in items]),
            "coordinate_mae": _mean([item["coordinate_mae"] for item in items]),
        }
        for name, items in sorted(buckets.items())
    }


def geometry_report(records: list[dict]) -> dict:
    """Section 12 main metrics over query-path records."""

    boxes = np.asarray([record["predicted_box"] for record in records], dtype=np.float64)
    coordinate_spread = (
        [float(value) for value in boxes.std(axis=0).tolist()] if len(boxes) else None
    )
    return {
        "count": len(records),
        "mean_box_iou": _mean([record["box_iou"] for record in records]),
        "median_box_iou": _median([record["box_iou"] for record in records]),
        "center_inside_rate": _mean([1.0 if record["center_inside"] else 0.0 for record in records]),
        "coordinate_mae": _mean([record["coordinate_mae"] for record in records]),
        "per_coordinate_prediction_spread": coordinate_spread,
        "canonical_rate": _mean([1.0 if record["canonical"] else 0.0 for record in records]),
        "in_range_rate": _mean([1.0 if record["in_range"] else 0.0 for record in records]),
        "by_level": _breakdown(records, "level"),
        "by_query_family": _breakdown(records, "query_family"),
    }


def paired_geometry_report(records_by_id: dict[str, dict], pairs: list[dict]) -> dict:
    """Section 12 paired probe: own vs cross box IoU, margin, same-image box L1."""

    rows = []
    for pair in pairs:
        a = records_by_id.get(str(pair["a"]))
        b = records_by_id.get(str(pair["b"]))
        if a is None or b is None:
            continue
        own = 0.5 * (a["box_iou"] + b["box_iou"])
        cross = 0.5 * (
            box_values_iou(a["predicted_box"], b["gt_box"])
            + box_values_iou(b["predicted_box"], a["gt_box"])
        )
        same_image_l1 = box_l1(a["predicted_box"], b["predicted_box"])
        rows.append(
            {
                "image_id": str(pair["image_id"]),
                "a": str(pair["a"]),
                "b": str(pair["b"]),
                "a_box_iou": a["box_iou"],
                "b_box_iou": b["box_iou"],
                "own_box_iou": float(own),
                "cross_box_iou": float(cross),
                "own_minus_cross_margin": float(own - cross),
                "same_image_predicted_box_l1": float(same_image_l1),
                "boxes_non_identical": bool(same_image_l1 > 0.0),
                "geometry_passed": bool(a["box_iou"] >= 0.5 and b["box_iou"] >= 0.5),
            }
        )
    return {
        "paired_total": len(rows),
        "geometry_paired_pass": sum(1 for row in rows if row["geometry_passed"]),
        "mean_own_box_iou": _mean([row["own_box_iou"] for row in rows]),
        "mean_cross_box_iou": _mean([row["cross_box_iou"] for row in rows]),
        "mean_own_minus_cross_margin": _mean([row["own_minus_cross_margin"] for row in rows]),
        "mean_same_image_predicted_box_l1": _mean([row["same_image_predicted_box_l1"] for row in rows]),
        "pairs_with_non_identical_boxes": sum(1 for row in rows if row["boxes_non_identical"]),
        "rows": rows,
    }


# ------------------------------------------------------------------ representation


def _pairwise_cosine(vectors: np.ndarray) -> float | None:
    """Mean cosine over a stack of vector pairs."""

    if vectors is None or vectors.shape[0] == 0:
        return None
    left = vectors[:, 0, :].astype(np.float64)
    right = vectors[:, 1, :].astype(np.float64)
    left = left / (np.linalg.norm(left, axis=-1, keepdims=True) + 1e-12)
    right = right / (np.linalg.norm(right, axis=-1, keepdims=True) + 1e-12)
    return float((left * right).sum(axis=-1).mean())


def representation_report(
    paired_records: list[dict], val_records: list[dict], pairs: list[dict],
    *, rng_seed: int = 20260926,
) -> dict:
    """Section 13: `[BOX]` hidden diagnostics, comparable with the frozen Task 6D.1 values.

    Same-image/different-instruction cosines come from the 20 paired validation images (the 40
    paired records); the different-image pool, effective rank and the hidden-vs-GT-box distance
    correlation are computed on the fixed 120-record validation set, whose records all come from
    distinct images. "Centered" means centred by the **global** mean over the whole analysed set,
    so an antiparallel result is a real measurement and not an artefact of per-pair centring.
    """

    paired_by_id = {record["sample_id"]: record for record in paired_records}
    hiddens = np.stack(
        [np.asarray(record["box_hidden"], dtype=np.float64) for record in val_records], axis=0
    )
    boxes = np.asarray([record["gt_box"] for record in val_records], dtype=np.float64)

    def _hidden(record) -> np.ndarray:
        return np.asarray(record["box_hidden"], dtype=np.float64)

    def same_image_pair_stack() -> np.ndarray:
        rows = []
        for pair in pairs:
            a = paired_by_id.get(str(pair["a"]))
            b = paired_by_id.get(str(pair["b"]))
            if a is None or b is None:
                continue
            rows.append([_hidden(a), _hidden(b)])
        if rows:
            return np.stack(rows)
        return np.zeros((0, 2, hiddens.shape[-1]))

    paired_pairs = same_image_pair_stack()

    rng = np.random.default_rng(int(rng_seed))
    pairs_pool: list[tuple[int, int]] = []
    for _ in range(min(200, len(val_records) * max(len(val_records) - 1, 0) // 2)):
        i = int(rng.integers(len(val_records)))
        j = int(rng.integers(len(val_records)))
        if i == j:
            continue
        pairs_pool.append((i, j))
    diff_pairs = np.stack([[hiddens[i], hiddens[j]] for i, j in pairs_pool]) if pairs_pool else np.zeros((0, 2, hiddens.shape[-1]))

    all_hiddens = np.vstack([paired_pairs.reshape(-1, paired_pairs.shape[-1]), hiddens])
    global_mean = all_hiddens.mean(axis=0, keepdims=True)
    paired_centered = np.stack(
        [[_hidden(paired_by_id[str(pair["a"])]) - global_mean[0],
          _hidden(paired_by_id[str(pair["b"])]) - global_mean[0]]
         for pair in pairs if str(pair["a"]) in paired_by_id and str(pair["b"]) in paired_by_id]
    ) if paired_pairs.shape[0] else np.zeros((0, 2, hiddens.shape[-1]))
    diff_centered = (
        np.stack([[hiddens[i] - global_mean[0], hiddens[j] - global_mean[0]] for i, j in pairs_pool])
        if pairs_pool
        else np.zeros((0, 2, hiddens.shape[-1]))
    )

    centered_matrix = hiddens - hiddens.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered_matrix, full_matrices=False)
    participation = float((singular_values**2).sum() ** 2 / max((singular_values**4).sum(), 1e-12))

    hidden_dists = []
    box_dists = []
    for i, j in pairs_pool:
        hidden_dists.append(float(np.linalg.norm(hiddens[i] - hiddens[j])))
        box_dists.append(float(np.abs(boxes[i] - boxes[j]).mean()))
    correlation = None
    if len(hidden_dists) > 1:
        correlation = float(np.corrcoef(hidden_dists, box_dists)[0, 1])

    return {
        "paired_sample_count": len(paired_records),
        "val_sample_count": len(val_records),
        "same_image_different_instruction_cosine": _pairwise_cosine(paired_pairs),
        "same_image_different_instruction_centered_cosine": _pairwise_cosine(paired_centered),
        "different_image_cosine": _pairwise_cosine(diff_pairs),
        "different_image_centered_cosine": _pairwise_cosine(diff_centered),
        "mean_hidden_l2_norm": float(np.linalg.norm(hiddens, axis=-1).mean()),
        "effective_rank_participation_ratio": participation,
        "hidden_distance_vs_gt_box_distance_correlation": correlation,
        "compared_pairs_sampled": len(pairs_pool),
        "different_image_pairs_are_cross_image": True,
        "centering": "global mean over the analysed hidden set (paired + val)",
        "legacy_seg_reference": {
            "source": "evaluation/task6d1_representation_stats.json",
            "centered_cosine_same_image_diff_template": 0.077,
            "centered_cosine_diff_image_same_template": 0.531,
            "effective_rank_participation_ratio": 8.3329,
            "note": (
                "frozen Task 6D.1 values, not rerun; same-image cosine is the discriminative "
                "number — Task 6D.1 measured the template (0.077), Task 6F asks whether the "
                "pre-reasoning [BOX] query carries target-specific variation instead"
            ),
        },
    }


# ------------------------------------------------------------------ reasoning compatibility


def reasoning_compat_report(runtime, samples, lookup: dict, *, max_new_tokens: int = 96) -> dict:
    """Section 14: continue generation from `image + instruction + fixed [BOX]`."""

    from .language_metrics import normalise, reasoning_part

    rows = []
    for sample in samples:
        image = sample.image_rgb()
        runtime.set_visual_cache_key(str(sample.image_id))
        generation = generate_with_box_prefix(
            runtime.model.qwen,
            runtime.processor,
            runtime.tokenizer,
            image,
            sample.instruction_zh,
            runtime.box_setup.box_token_id,
            runtime.model.seg_token_id,
            max_new_tokens=max_new_tokens,
        )
        text = generation["text"]
        key = normalise(reasoning_part(text))
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                "seg_count": generation["seg_count"],
                "exactly_one_seg": generation["seg_count"] == 1,
                "terminated_with_eos": generation["terminated_with_eos"],
                "generated_token_count": generation["generated_token_count"],
                "reasoning_template_known": bool(key and key in lookup),
                "text": text,
            }
        )
    return {
        "count": len(rows),
        "exactly_one_seg_rate": float(sum(1 for row in rows if row["exactly_one_seg"]) / len(rows)),
        "eos_termination_rate": float(sum(1 for row in rows if row["terminated_with_eos"]) / len(rows)),
        "template_known_rate": float(
            sum(1 for row in rows if row["reasoning_template_known"]) / len(rows)
        ),
        "rows": rows,
    }


# ------------------------------------------------------------------ F2 segmentation


@torch.no_grad()
def segment_from_predicted_box(runtime, sample, predicted_box, *, image=None) -> dict:
    """Official frozen SAM2 box prompt from the query head's predicted box (section 15)."""

    if image is None:
        image = sample.image_rgb()
    features, cached = runtime.features_for_image(image, image_id=getattr(sample, "image_id", None))
    geometry = torch.tensor([float(value) for value in predicted_box], dtype=torch.float32)
    decoded = decode_mask_from_geometry(runtime.model.sam, features, geometry, "box", multimask_output=False)
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
def f2_report(runtime, val_samples, pairs, records_by_id: dict[str, dict]) -> dict:
    """Section 16 metrics: strict e2e segmentation from predicted boxes only."""

    from .data import to_sample

    rows = []
    for sample in val_samples:
        record = records_by_id.get(str(sample.sample_id))
        if record is None:
            continue
        segmentation = segment_from_predicted_box(runtime, sample, record["predicted_box"])
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                "image_id": str(sample.image_id),
                "level": int(sample.level),
                "query_family": family_of(str(sample.query_type)),
                "box_iou": record["box_iou"],
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
        mask_a = segment_from_predicted_box(runtime, sample_a, a["predicted_box"])["mask"]
        mask_b = segment_from_predicted_box(runtime, sample_b, b["predicted_box"])["mask"]
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
            name: {
                "count": len(items),
                "strict_miou": _mean([item["strict_miou"] for item in items]),
            }
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
