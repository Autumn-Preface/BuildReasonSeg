"""Task 6E parts H/J: free-generation geometry evaluation and inference-only segmentation.

Everything here reads the model's **generated token ids** and parses them with
`spatial_tokens.parse_generated_ids`; no GT geometry ever enters a prompt. The GT mask is
used only to score a prediction (and, for the oracle comparison, by
`scripts/task6e_quantized_oracle.py`, which is a separate audit).
"""

from __future__ import annotations

import numpy as np
import torch

from .data import to_sample
from .grounding import decode_mask_from_geometry, target_geometry
from .metrics import dice_from_logits, mask_iou_from_logits
from .qwen_seg import generate_with_seg
from .spatial_tokens import box_iou_values, parse_generated_ids
from .validation import family_of


def sample_key(record_or_sample) -> dict:
    """Level / query-family / instruction identity of a record or Sample."""

    query_type = getattr(record_or_sample, "query_type", None) or record_or_sample["query_type"]
    level = getattr(record_or_sample, "level", None)
    if level is None:
        level = record_or_sample["level"]
    return {"level": int(level), "query_family": family_of(str(query_type)), "query_type": str(query_type)}


@torch.no_grad()
def generate_spatial(runtime, sample, image=None, *, max_new_tokens: int | None = None) -> dict:
    """Free generation from image + instruction only, parsed as token ids."""

    setup = runtime.spatial_setup
    codec = runtime.spatial_codec
    if setup is None:
        raise RuntimeError("runtime has no spatial token setup")
    if image is None:
        image = sample.image_rgb()
    runtime.set_visual_cache_key(str(sample.image_id))
    limit = int(
        max_new_tokens
        if max_new_tokens is not None
        else runtime.cfg.get("inference", {}).get("validation_max_new_tokens", 96)
    )
    generation = generate_with_seg(
        runtime.model.qwen,
        runtime.processor,
        runtime.tokenizer,
        image,
        sample.instruction_zh,
        max_new_tokens=limit,
        seg_token_id=runtime.model.seg_token_id,
    )
    prompt_length = int(generation["prompt_length"])
    generated = [int(value) for value in generation["token_ids"][prompt_length:]]
    parsed = parse_generated_ids(generated, setup, codec, runtime.model.seg_token_id)
    gt_box = [float(value) for value in target_geometry(sample.target_mask(), "box")]
    gt_codes = codec.encode(gt_box)
    box_iou = None if parsed.box is None else float(box_iou_values(parsed.box, gt_box))
    center_inside = None
    if parsed.box is not None:
        mask = np.asarray(sample.target_mask()).astype(bool)
        height, width = mask.shape
        cx = int(round(0.5 * (parsed.box[0] + parsed.box[2]) * (width - 1)))
        cy = int(round(0.5 * (parsed.box[1] + parsed.box[3]) * (height - 1)))
        if 0 <= cx < width and 0 <= cy < height:
            center_inside = bool(mask[cy, cx])
        else:
            center_inside = False
    code_error = None
    if parsed.codes is not None:
        code_error = [abs(int(a) - int(b)) for a, b in zip(parsed.codes, gt_codes)]
    return {
        "sample_id": str(sample.sample_id),
        "image_id": str(sample.image_id),
        **sample_key(sample),
        "structural_valid": bool(parsed.structural_valid),
        "exact_four_token_sequence": bool(parsed.exact_four_token_sequence),
        "failure": parsed.failure,
        "box_token_count": parsed.box_token_count,
        "loc_token_count": parsed.loc_token_count,
        "seg_count": parsed.seg_count,
        "seg_after_box": parsed.seg_after_box,
        "predicted_codes": parsed.codes,
        "predicted_box": parsed.box,
        "gt_codes": [int(value) for value in gt_codes],
        "gt_box": gt_box,
        "box_iou": box_iou,
        "center_inside": center_inside,
        "code_error": code_error,
        "generated_token_count": len(generated),
        "generated_text": generation["text"],
        "prompt_length": prompt_length,
    }


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
            "structural_valid": sum(1 for item in items if item["structural_valid"]),
            "mean_box_iou": _mean([item["box_iou"] for item in items]),
        }
        for name, items in sorted(buckets.items())
    }


def geometry_report(records: list[dict], *, bins: int, strict_miou: list[float] | None = None) -> dict:
    """Section 13 metrics over already-generated records."""

    total = len(records)
    valid = [record for record in records if record["structural_valid"]]
    exact = [record for record in records if record["exact_four_token_sequence"] and record["structural_valid"]]
    coordinate_accuracy = []
    for index in range(4):
        correct = sum(
            1
            for record in valid
            if record["predicted_codes"] is not None
            and int(record["predicted_codes"][index]) == int(record["gt_codes"][index])
        )
        coordinate_accuracy.append(round(correct / max(len(valid), 1), 6))
    bin_errors = [
        abs(int(record["predicted_codes"][index]) - int(record["gt_codes"][index]))
        for record in valid
        for index in range(4)
    ]
    scale = float(max(int(bins) - 1, 1))
    result = {
        "count": total,
        "structural_valid": len(valid),
        "structural_valid_rate": round(len(valid) / max(total, 1), 6),
        "exact_four_token_sequence": len(exact),
        "exact_four_token_accuracy": round(len(exact) / max(total, 1), 6),
        "per_coordinate_token_accuracy": coordinate_accuracy,
        "mean_absolute_bin_error": _mean(bin_errors),
        "normalized_coordinate_error": (
            None if not bin_errors else float(_mean(bin_errors) / scale)
        ),
        "mean_predicted_box_iou": _mean([record["box_iou"] for record in records]),
        "mean_predicted_box_iou_valid_only": _mean([record["box_iou"] for record in valid]),
        "center_inside_rate": _mean([1.0 if record["center_inside"] else 0.0 for record in records]),
        "emit_box_token_count": {
            "one": sum(1 for record in records if record["box_token_count"] == 1),
            "zero": sum(1 for record in records if record["box_token_count"] == 0),
            "more": sum(1 for record in records if record["box_token_count"] > 1),
        },
        "seg_count_histogram": _histogram([record["seg_count"] for record in records]),
        "by_level": _breakdown(records, "level"),
        "by_query_family": _breakdown(records, "query_family"),
    }
    if strict_miou is not None:
        result["strict_end_to_end_miou"] = _mean(strict_miou)
    return result


def _histogram(values: list[int]) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(int(value))] = counts.get(str(int(value)), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def paired_geometry_report(parsed_by_id: dict[str, dict], pairs: list[dict]) -> dict:
    """Section 13 paired metrics: same-image predicted-box separation."""

    rows = []
    for pair in pairs:
        a = parsed_by_id.get(str(pair["a"]))
        b = parsed_by_id.get(str(pair["b"]))
        if a is None or b is None:
            continue
        both_valid = bool(a["structural_valid"] and b["structural_valid"])
        box_l1 = None
        if both_valid:
            box_l1 = float(
                sum(abs(float(x) - float(y)) for x, y in zip(a["predicted_box"], b["predicted_box"])) / 4.0
            )
        passed = bool(
            both_valid
            and a["box_iou"] is not None
            and b["box_iou"] is not None
            and a["box_iou"] >= 0.5
            and b["box_iou"] >= 0.5
        )
        rows.append(
            {
                "image_id": str(pair["image_id"]),
                "a": str(pair["a"]),
                "b": str(pair["b"]),
                "both_structural_valid": both_valid,
                "a_box_iou": a["box_iou"],
                "b_box_iou": b["box_iou"],
                "a_center_inside": a["center_inside"],
                "b_center_inside": b["center_inside"],
                "same_image_predicted_box_l1": box_l1,
                "same_image_predicted_box_identical": bool(box_l1 == 0.0) if box_l1 is not None else None,
                "geometry_passed": passed,
            }
        )
    return {
        "paired_total": len(rows),
        "geometry_paired_pass": sum(1 for row in rows if row["geometry_passed"]),
        "both_structural_valid": sum(1 for row in rows if row["both_structural_valid"]),
        "mean_same_image_predicted_box_l1": _mean(
            [row["same_image_predicted_box_l1"] for row in rows]
        ),
        "pairs_with_identical_predicted_box": sum(
            1 for row in rows if row["same_image_predicted_box_identical"]
        ),
        "rows": rows,
    }


# ------------------------------------------------------------------ teacher-forced diagnostics


@torch.no_grad()
def teacher_forced_report(runtime, samples, *, limit: int | None = None) -> dict:
    """Section 13: teacher-forced token accuracy is **diagnostic only**."""

    from .qwen_seg import forward_qwen
    from .spatial_training import spatial_loss, spatial_teacher_forced_metrics

    subset = samples[:limit] if limit else samples
    rows = []
    autocast = torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=bool(runtime.cfg.get("training", {}).get("bf16_autocast", True)),
    )
    for sample in subset:
        from .spatial_training import spatial_batch

        batch, example = spatial_batch(runtime, sample, image=sample.image_rgb())
        batch = batch.to(runtime.device)
        with autocast:
            lm_logits, _hidden = forward_qwen(runtime.model.qwen, batch)
            metrics = spatial_teacher_forced_metrics(lm_logits, batch, example)
            losses = spatial_loss(lm_logits, batch, example, runtime.spatial_setup)
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                **metrics,
                "assistant_ce": losses["assistant_ce_raw"],
                "location_ce": losses["location_ce_raw"],
            }
        )
    keys = [
        "reasoning_token_accuracy",
        "box_token_accuracy",
        "loc_token_accuracy",
        "seg_token_accuracy",
        "eos_token_accuracy",
    ]
    return {
        "count": len(rows),
        "mean": {key: _mean([row[key] for row in rows]) for key in keys},
        "mean_assistant_ce": _mean([row["assistant_ce"] for row in rows]),
        "mean_location_ce": _mean([row["location_ce"] for row in rows]),
        "rows": rows,
    }


# ------------------------------------------------------------------ E2 segmentation


@torch.no_grad()
def segment_from_predicted_box(runtime, sample, predicted_box, *, image=None) -> dict:
    """Official frozen SAM2 box prompt from a **predicted** box (section 14)."""

    if image is None:
        image = sample.image_rgb()
    features, cached = runtime.features_for_image(image, image_id=getattr(sample, "image_id", None))
    geometry = torch.tensor([float(value) for value in predicted_box], dtype=torch.float32)
    decoded = decode_mask_from_geometry(runtime.model.sam, features, geometry, "box", multimask_output=False)
    logits = decoded.low_res_logits
    target = np.asarray(sample.target_mask()).astype(np.float32)
    from .metrics import upsample_logits

    logits = upsample_logits(logits, (int(target.shape[0]), int(target.shape[1])), mode="bilinear")
    target_tensor = torch.as_tensor(target)
    miou = float(mask_iou_from_logits(logits, target_tensor, tuple(target.shape), threshold=0.0))
    dice = float(dice_from_logits(logits, target_tensor, tuple(target.shape), threshold=0.0))
    mask = (logits[0, 0] > 0).detach().cpu().numpy() if logits.dim() == 4 else (logits[0] > 0).numpy()
    mask = mask.reshape(int(target.shape[0]), int(target.shape[1]))
    return {"miou": miou, "dice": dice, "cached_features": bool(cached), "mask": mask}


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@torch.no_grad()
def e2_report(runtime, val_samples, pairs, parsed_by_id: dict[str, dict], *, mask_limit: int | None = None) -> dict:
    """Section 15: strict end-to-end segmentation from generated boxes only."""

    records = []
    samples = val_samples[:mask_limit] if mask_limit else val_samples
    for sample in samples:
        parsed = parsed_by_id.get(str(sample.sample_id))
        if parsed is None or not parsed["structural_valid"]:
            records.append(
                {
                    "sample_id": str(sample.sample_id),
                    "image_id": str(sample.image_id),
                    **sample_key(sample),
                    "structural_valid": False,
                    "strict_miou": 0.0,
                    "dice": 0.0,
                }
            )
            continue
        segmentation = segment_from_predicted_box(runtime, sample, parsed["predicted_box"])
        records.append(
            {
                "sample_id": str(sample.sample_id),
                "image_id": str(sample.image_id),
                **sample_key(sample),
                "structural_valid": True,
                "predicted_box": parsed["predicted_box"],
                "box_iou": parsed["box_iou"],
                "strict_miou": segmentation["miou"],
                "dice": segmentation["dice"],
            }
        )
    valid = [record for record in records if record["structural_valid"]]
    strict = _mean([record["strict_miou"] for record in records]) or 0.0
    conditional = _mean([record["strict_miou"] for record in valid])

    pair_rows = []
    for pair in pairs:
        a = parsed_by_id.get(str(pair["a"]))
        b = parsed_by_id.get(str(pair["b"]))
        if a is None or b is None:
            continue
        sample_a = to_sample(pair["record_a"]) if pair.get("record_a") is not None else to_sample(pair["a"])
        sample_b = to_sample(pair["record_b"]) if pair.get("record_b") is not None else to_sample(pair["b"])
        mask_a = mask_b = None
        if a["structural_valid"]:
            mask_a = segment_from_predicted_box(runtime, sample_a, a["predicted_box"])["mask"]
        if b["structural_valid"]:
            mask_b = segment_from_predicted_box(runtime, sample_b, b["predicted_box"])["mask"]
        if mask_a is None or mask_b is None:
            pair_rows.append(
                {
                    "image_id": str(pair["image_id"]),
                    "a": str(pair["a"]),
                    "b": str(pair["b"]),
                    "valid_pair": False,
                    "mask_paired_pass": False,
                    "own_iou": None,
                    "cross_iou": None,
                    "mask_iou_between_predictions": None,
                }
            )
            continue
        target_a = np.asarray(sample_a.target_mask()).astype(bool)
        target_b = np.asarray(sample_b.target_mask()).astype(bool)
        own = 0.5 * (mask_iou(mask_a, target_a) + mask_iou(mask_b, target_b))
        cross = 0.5 * (mask_iou(mask_a, target_b) + mask_iou(mask_b, target_a))
        between = mask_iou(mask_a, mask_b)
        pair_rows.append(
            {
                "image_id": str(pair["image_id"]),
                "a": str(pair["a"]),
                "b": str(pair["b"]),
                "valid_pair": True,
                "mask_paired_pass": bool(own > cross),
                "own_iou": float(own),
                "cross_iou": float(cross),
                "own_minus_cross_margin": float(own - cross),
                "mask_iou_between_predictions": float(between),
            }
        )

    return {
        "count": len(records),
        "structural_valid": len(valid),
        "strict_end_to_end_miou": float(strict),
        "conditional_end_to_end_miou": conditional,
        "mean_dice": _mean([record["dice"] for record in records]),
        "conditional_dice": _mean([record["dice"] for record in valid]),
        "by_level": {
            name: {
                "count": len(items),
                "strict_miou": _mean([item["strict_miou"] for item in items]),
            }
            for name, items in _group(records, "level").items()
        },
        "by_query_family": {
            name: {
                "count": len(items),
                "strict_miou": _mean([item["strict_miou"] for item in items]),
            }
            for name, items in _group(records, "query_family").items()
        },
        "records": records,
        "paired": {
            "paired_total": len(pair_rows),
            "mask_paired_pass": sum(1 for row in pair_rows if row["mask_paired_pass"]),
            "mean_own_iou": _mean([row["own_iou"] for row in pair_rows]),
            "mean_cross_iou": _mean([row["cross_iou"] for row in pair_rows]),
            "mean_own_minus_cross_margin": _mean(
                [row.get("own_minus_cross_margin") for row in pair_rows]
            ),
            "mean_mask_iou_between_predictions": _mean(
                [row["mask_iou_between_predictions"] for row in pair_rows]
            ),
            "rows": pair_rows,
        },
    }


def _group(records: list[dict], key: str) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for record in records:
        buckets.setdefault(str(record[key]), []).append(record)
    return {name: buckets[name] for name in sorted(buckets)}
