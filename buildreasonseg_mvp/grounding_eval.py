"""Task 6D evaluation for the Spatial Grounding Bridge.

Everything here runs the **candidate** path: `[SEG]` hidden -> SpatialGroundingHead ->
predicted geometry -> official SAM2 prompt encoder -> mask. Ground-truth geometry is
computed only to score the prediction; it is never placed into a prompt, which is what
`tests/test_task6d_grounding.py` asserts.

The module mirrors `validation.py`'s structure so the Task 6D numbers are produced by the
same aggregation and the same strict metric definitions as Task 6C.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from . import data as data_mod
from .grounding import (
    GEOMETRY_BOX,
    GEOMETRY_POINT,
    box_metrics,
    decode_mask_from_geometry,
    geometry_metrics,
    geometry_paired_decision,
    mask_contains_point,
    point_metrics,
    target_geometry,
)
from .metrics import collapse_flags, dice_from_logits, mask_iou_from_logits
from .qwen_seg import generate_with_seg, seg_hidden_from_full_sequence
from .validation import ValidationRun, _mean, level_key


def geometry_of_mask(mask: np.ndarray, kind: str) -> tuple[float, ...]:
    return target_geometry(mask, kind)


def predicted_geometry_for_hidden(runtime, hidden: torch.Tensor) -> torch.Tensor:
    """`[SEG]` hidden -> head -> normalized geometry (the only inference-time source)."""

    head = runtime.model.grounding_head
    if head is None:
        raise RuntimeError("no grounding head installed")
    return head(hidden)


def decode_from_hidden(runtime, hidden: torch.Tensor, features, kind: str):
    geometry = predicted_geometry_for_hidden(runtime, hidden)
    decoded = decode_mask_from_geometry(
        runtime.model.sam,
        features,
        geometry,
        kind,
        multimask_output=bool(runtime.cfg["inference"].get("multimask_output", False)),
    )
    return geometry, decoded


# ------------------------------------------------------------------ teacher forced


def teacher_forced_grounded(runtime, samples: Sequence[data_mod.Sample]) -> dict:
    """Teacher-forced geometry and mask metrics (diagnostic, not the headline)."""

    kind = runtime.model.geometry_kind
    runtime.model.eval()
    records = []
    with torch.no_grad():
        for sample in samples:
            image = sample.image_rgb()
            features, _ = runtime.features_for(sample, image)
            batch, _ = runtime.prepare(sample, image)
            output = runtime.model.forward_grounded(batch.to(runtime.device), features)
            predicted = output.predicted_geometry.detach().float().cpu()
            target = torch.as_tensor(target_geometry(sample.target_mask(), kind)).float().reshape(1, -1)
            metrics = geometry_metrics(predicted, target, kind)
            gt_mask = sample.target_mask()
            iou = mask_iou_from_logits(
                output.mask_logits, torch.as_tensor(gt_mask).float(), gt_mask.shape, threshold=0.0
            )
            dice = dice_from_logits(
                output.mask_logits, torch.as_tensor(gt_mask).float(), gt_mask.shape, threshold=0.0
            )
            record = {
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "level": sample.level,
                "query_type": sample.query_type,
                "predicted_geometry": predicted.reshape(-1).tolist(),
                "gt_geometry": target.reshape(-1).tolist(),
                "iou": iou,
                "dice": dice,
                **metrics,
            }
            record["center_inside_target"] = _center_inside(gt_mask, predicted, kind)
            records.append(record)
    runtime.model.train()
    return {
        "kind": kind,
        "records": records,
        "count": len(records),
        "mean_geometry_metric": (
            _mean([r["mean_box_iou"] for r in records])
            if kind == GEOMETRY_BOX
            else _mean([r["mean_abs_error"] for r in records])
        ),
        "mean_box_iou": _mean([r["mean_box_iou"] for r in records]) if kind == GEOMETRY_BOX else None,
        "mean_point_error": _mean([r["mean_abs_error"] for r in records]) if kind == GEOMETRY_POINT else None,
        "center_inside_rate": _mean([1.0 if r["center_inside_target"] else 0.0 for r in records]),
        "miou": _mean([r["iou"] for r in records]),
        "dice": _mean([r["dice"] for r in records]),
    }


def _center_inside(gt_mask: np.ndarray, geometry: torch.Tensor, kind: str) -> bool:
    values = geometry.detach().float().reshape(-1).tolist()
    if kind == GEOMETRY_POINT:
        return mask_contains_point(gt_mask, (values[0], values[1]))
    center = ((values[0] + values[2]) / 2.0, (values[1] + values[3]) / 2.0)
    return mask_contains_point(gt_mask, center)


# ------------------------------------------------------------------ free generation


def free_generation_grounded(
    runtime, samples: Sequence[data_mod.Sample], lookup: dict, *, include_component_overlap: bool = False
) -> dict:
    """The headline setting (section 11): generate, then ground the generated `[SEG]`."""

    from .language_metrics import char_similarity, exact_match, operation_chain_correct

    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    runtime.model.eval()
    config = runtime.cfg["inference"]
    max_new_tokens = int(config.get("validation_max_new_tokens", config["max_new_tokens"]))
    checkpointing_was_on = bool(getattr(runtime.qwen, "is_gradient_checkpointing", False))
    if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_disable"):
        runtime.qwen.gradient_checkpointing_disable()
    kind = runtime.model.geometry_kind

    out: list[dict] = []
    try:
        for sample in samples:
            image = sample.image_rgb()
            gt = sample.target_mask()
            generation = generate_with_seg(
                runtime.model.qwen,
                runtime.processor,
                runtime.tokenizer,
                image,
                sample.instruction_zh,
                max_new_tokens=max_new_tokens,
                seg_token_id=runtime.model.seg_token_id,
            )
            valid = generation["seg_count"] == 1
            entry = {
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "level": sample.level,
                "level_key": level_key(sample.raw),
                "query_type": sample.query_type,
                "target_component_id": sample.target_component_id,
                "target_area_px": int(gt.sum()),
                "generated_text": generation["text"],
                "seg_count": generation["seg_count"],
                "seg_valid": bool(valid),
                "generated_token_count": generation["generated_token_count"],
                "reasoning_exact_match": exact_match(generation["text"], sample.reasoning_zh),
                "reasoning_char_similarity": char_similarity(generation["text"], sample.reasoning_zh),
                "operation_chain_correct": operation_chain_correct(
                    generation["text"], sample.query_type, lookup
                ),
                "predicted_geometry": None,
                "gt_geometry": list(target_geometry(gt, kind)),
                "iou": 0.0,
                "dice": 0.0,
                "iou_conditional": None,
                "dice_conditional": None,
                "collapse": None,
                "geometry_valid": None,
                "center_inside_target": None,
                "geometry_metric": None,
                "mask_sha": None,
            }
            if valid:
                position = generation["seg_positions"][0]
                with torch.no_grad():
                    hidden = seg_hidden_from_full_sequence(
                        runtime.model.qwen,
                        runtime.processor,
                        image,
                        generation["token_ids"],
                        position,
                        instruction=sample.instruction_zh,
                    )
                    features, _ = runtime.features_for(sample, image)
                    geometry, decoded = decode_from_hidden(runtime, hidden, features, kind)
                    predicted = geometry.detach().float().cpu().reshape(1, -1)
                    target = torch.as_tensor(entry["gt_geometry"]).float().reshape(1, -1)
                    metrics = geometry_metrics(predicted, target, kind)
                    iou = mask_iou_from_logits(
                        decoded.low_res_logits, torch.as_tensor(gt).float(), gt.shape, threshold=0.0
                    )
                    dice = dice_from_logits(
                        decoded.low_res_logits, torch.as_tensor(gt).float(), gt.shape, threshold=0.0
                    )
                    entry.update(
                        {
                            "predicted_geometry": predicted.reshape(-1).tolist(),
                            "iou": iou,
                            "dice": dice,
                            "iou_conditional": iou,
                            "dice_conditional": dice,
                            "collapse": collapse_flags(decoded.low_res_logits.detach().float()),
                            "geometry_valid": bool(
                                torch.isfinite(predicted).all()
                                and float(predicted.min()) >= -1e-6
                                and float(predicted.max()) <= 1 + 1e-6
                            ),
                            "center_inside_target": _center_inside(gt, predicted, kind),
                            "geometry_metric": metrics,
                            "mask_sha": _mask_sha(decoded.low_res_logits, gt.shape),
                            "component_overlap": (
                                prediction_component_overlap(sample, _binarize(decoded.low_res_logits, gt.shape))
                                if include_component_overlap
                                else None
                            ),
                        }
                    )
            out.append(entry)
    finally:
        if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_enable"):
            runtime.qwen.gradient_checkpointing_enable()

    runtime.model.train()
    run = ValidationRun(records=out, seconds=time.time() - started)
    run.peak_vram = _vram()
    valid_records = [record for record in out if record["seg_valid"]]
    metric_key = "mean_box_iou" if kind == GEOMETRY_BOX else "mean_abs_error"
    return {
        "kind": kind,
        "run": {
            "records": out,
            "seconds": run.seconds,
            "peak_vram": run.peak_vram,
        },
        "count": len(out),
        "emission_valid": sum(1 for record in out if record["seg_valid"]),
        "emission_rate": _mean([1.0 if record["seg_valid"] else 0.0 for record in out]),
        "strict_end_to_end_miou": _mean([record["iou"] for record in out]),
        "strict_end_to_end_dice": _mean([record["dice"] for record in out]),
        "conditional_miou": _mean([record["iou_conditional"] for record in valid_records])
        if valid_records
        else None,
        "conditional_dice": _mean([record["dice_conditional"] for record in valid_records])
        if valid_records
        else None,
        "mean_geometry_metric": _mean(
            [record["geometry_metric"][metric_key] for record in valid_records]
        )
        if valid_records
        else None,
        "mean_box_iou": _mean([record["geometry_metric"]["mean_box_iou"] for record in valid_records])
        if valid_records and kind == GEOMETRY_BOX
        else None,
        "center_inside_rate": _mean(
            [1.0 if record["center_inside_target"] else 0.0 for record in valid_records]
        )
        if valid_records
        else None,
        "distinct_predicted_masks": len({record["mask_sha"] for record in valid_records if record["mask_sha"]}),
    }


def _mask_sha(logits: torch.Tensor, size: tuple[int, int]) -> str:
    import hashlib

    mask = (
        torch.nn.functional.interpolate(
            logits.detach().float(), size=tuple(size), mode="bilinear", align_corners=False
        )
        > 0.0
    )
    return hashlib.sha1(mask.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def target_component_stats(sample) -> dict:
    """GT-side pseudo-instance context (section 16): neighbours, border, connectivity."""

    from scipy import ndimage

    component_map = data_mod.load_component_map(sample.component_map_path)
    components = [int(value) for value in np.unique(component_map) if int(value) != 0]
    target_id = int(sample.target_component_id)
    target_mask = component_map == target_id
    if not target_mask.any():
        return {"components_in_tile": len(components), "target_component_present": False}
    dilated = ndimage.binary_dilation(target_mask, iterations=2)
    neighbours = sorted(
        {int(value) for value in np.unique(component_map[dilated]) if int(value) not in (0, target_id)}
    )
    _labelled, count = ndimage.label(target_mask)
    return {
        "components_in_tile": len(components),
        "target_component_present": True,
        "touching_neighbour_ids": neighbours,
        "touching_neighbour_count": len(neighbours),
        "target_connected_components": int(count),
        "target_area_px": int(target_mask.sum()),
        "target_area_fraction": float(target_mask.mean()),
        "touches_border": bool(
            target_mask[0, :].any()
            or target_mask[-1, :].any()
            or target_mask[:, 0].any()
            or target_mask[:, -1].any()
        ),
    }


def prediction_component_overlap(sample, predicted_mask: np.ndarray) -> dict:
    """How many GT components does a predicted mask cover? (merged-instance evidence)"""

    component_map = data_mod.load_component_map(sample.component_map_path)
    area = int(np.asarray(predicted_mask).sum())
    if area == 0:
        return {"predicted_area_px": 0, "components_covered": [], "significant_component_count": 0,
                "multi_component": False}
    values, counts = np.unique(component_map[np.asarray(predicted_mask)], return_counts=True)
    covered = [
        {"component_id": int(value), "share": float(count) / area}
        for value, count in zip(values, counts)
        if int(value) != 0
    ]
    covered.sort(key=lambda item: -item["share"])
    significant = [item for item in covered if item["share"] >= 0.05]
    return {
        "predicted_area_px": area,
        "components_covered": covered[:5],
        "significant_component_count": len(significant),
        "multi_component": len(significant) >= 2,
    }


# ------------------------------------------------------------------ paired probe


def paired_probe_grounded(
    runtime, pairs: Sequence[dict], lookup: dict, *, free_generation: bool = True
) -> dict:
    """Section 12: the same-image paired probe for masks **and** geometry."""

    kind = runtime.model.geometry_kind
    runtime.model.eval()
    config = runtime.cfg["inference"]
    max_new_tokens = int(config.get("validation_max_new_tokens", config["max_new_tokens"]))
    checkpointing_was_on = bool(getattr(runtime.qwen, "is_gradient_checkpointing", False))
    if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_disable"):
        runtime.qwen.gradient_checkpointing_disable()

    records = []
    try:
        for pair in pairs:
            sample_a = data_mod.to_sample(pair["a"])
            sample_b = data_mod.to_sample(pair["b"])
            info = {"image_id": sample_a.image_id, "sample_a": sample_a.sample_id, "sample_b": sample_b.sample_id}
            if free_generation:
                first = _generate_grounded(runtime, sample_a, max_new_tokens, kind)
                second = _generate_grounded(runtime, sample_b, max_new_tokens, kind)
                info["emission_valid_a"] = first["valid"]
                info["emission_valid_b"] = second["valid"]
                info["generated_text_a"] = first["text"]
                info["generated_text_b"] = second["text"]
            else:
                first = _teacher_forced_grounded(runtime, sample_a, kind)
                second = _teacher_forced_grounded(runtime, sample_b, kind)
            if not first["valid"] or not second["valid"]:
                info.update(
                    {
                        "paired_mask_pass": False,
                        "paired_geometry_pass": False,
                        "own_minus_cross_margin": None,
                        "iou_prediction_a_vs_prediction_b": None,
                        "geometry_a": None,
                        "geometry_b": None,
                    }
                )
                records.append(info)
                continue

            gt_a = sample_a.target_mask()
            gt_b = sample_b.target_mask()
            own_a = mask_iou_from_logits(first["logits"], torch.as_tensor(gt_a).float(), gt_a.shape, threshold=0.0)
            own_b = mask_iou_from_logits(second["logits"], torch.as_tensor(gt_b).float(), gt_b.shape, threshold=0.0)
            cross_a = mask_iou_from_logits(first["logits"], torch.as_tensor(gt_b).float(), gt_b.shape, threshold=0.0)
            cross_b = mask_iou_from_logits(second["logits"], torch.as_tensor(gt_a).float(), gt_a.shape, threshold=0.0)
            mask_a = _binarize(first["logits"], gt_a.shape)
            mask_b = _binarize(second["logits"], gt_b.shape)
            intersection = float((mask_a & mask_b).sum())
            union = float((mask_a | mask_b).sum())
            prediction_iou = intersection / union if union else 0.0

            geometry_a = first["geometry"]
            geometry_b = second["geometry"]
            decision = geometry_paired_decision(
                geometry_a, geometry_b, target_geometry(gt_a, kind), target_geometry(gt_b, kind), kind
            )
            info.update(
                {
                    "paired_mask_pass": bool(own_a > cross_a and own_b > cross_b),
                    "paired_geometry_pass": bool(decision["passed"]),
                    "own_iou_a": own_a,
                    "own_iou_b": own_b,
                    "cross_iou_a": cross_a,
                    "cross_iou_b": cross_b,
                    "mean_own_iou": (own_a + own_b) / 2.0,
                    "mean_cross_iou": (cross_a + cross_b) / 2.0,
                    "own_minus_cross_margin": ((own_a - cross_a) + (own_b - cross_b)) / 2.0,
                    "iou_prediction_a_vs_prediction_b": prediction_iou,
                    "geometry_a": list(geometry_a),
                    "geometry_b": list(geometry_b),
                    "gt_geometry_a": list(target_geometry(gt_a, kind)),
                    "gt_geometry_b": list(target_geometry(gt_b, kind)),
                    "geometry_decision": decision,
                    "geometry_distance": float(
                        np.linalg.norm(np.asarray(geometry_a) - np.asarray(geometry_b))
                    ),
                }
            )
            records.append(info)
    finally:
        if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_enable"):
            runtime.qwen.gradient_checkpointing_enable()
    runtime.model.train()

    passed = [record for record in records if record.get("paired_mask_pass")]
    geometry_passed = [record for record in records if record.get("paired_geometry_pass")]
    margins = [record["own_minus_cross_margin"] for record in records if record.get("own_minus_cross_margin") is not None]
    pred_ious = [
        record["iou_prediction_a_vs_prediction_b"]
        for record in records
        if record.get("iou_prediction_a_vs_prediction_b") is not None
    ]
    distances = [record["geometry_distance"] for record in records if record.get("geometry_distance") is not None]
    return {
        "kind": kind,
        "free_generation": bool(free_generation),
        "pairs": records,
        "paired_total": len(records),
        "mask_paired_pass": len(passed),
        "geometry_paired_pass": len(geometry_passed),
        "mean_own_iou": _mean(
            [record["mean_own_iou"] for record in records if record.get("mean_own_iou") is not None]
        ),
        "mean_cross_iou": _mean(
            [record["mean_cross_iou"] for record in records if record.get("mean_cross_iou") is not None]
        ),
        "mean_own_minus_cross_margin": _mean(margins),
        "mean_iou_pred_a_vs_b": _mean(pred_ious),
        "mean_geometry_distance": _mean(distances),
        "emission_valid_pairs": sum(
            1
            for record in records
            if record.get("emission_valid_a", True) and record.get("emission_valid_b", True)
        ),
    }


def _binarize(logits: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """(1,1,h,w) mask logits -> a 2-D boolean mask at the target resolution."""

    mask = (
        torch.nn.functional.interpolate(
            logits.detach().float().reshape(1, 1, *logits.shape[-2:]),
            size=tuple(size),
            mode="bilinear",
            align_corners=False,
        )
        .reshape(tuple(size))
        > 0.0
    )
    return mask.cpu().numpy().astype(bool)


def _generate_grounded(runtime, sample, max_new_tokens: int, kind: str) -> dict:
    image = sample.image_rgb()
    generation = generate_with_seg(
        runtime.model.qwen,
        runtime.processor,
        runtime.tokenizer,
        image,
        sample.instruction_zh,
        max_new_tokens=max_new_tokens,
        seg_token_id=runtime.model.seg_token_id,
    )
    valid = generation["seg_count"] == 1
    result = {"valid": bool(valid), "text": generation["text"], "logits": None, "geometry": None}
    if not valid:
        return result
    with torch.no_grad():
        hidden = seg_hidden_from_full_sequence(
            runtime.model.qwen,
            runtime.processor,
            image,
            generation["token_ids"],
            generation["seg_positions"][0],
            instruction=sample.instruction_zh,
        )
        features, _ = runtime.features_for(sample, image)
        geometry, decoded = decode_from_hidden(runtime, hidden, features, kind)
        result["logits"] = decoded.low_res_logits
        result["geometry"] = geometry.detach().float().cpu().reshape(-1).tolist()
    return result


def _teacher_forced_grounded(runtime, sample, kind: str) -> dict:
    image = sample.image_rgb()
    features, _ = runtime.features_for(sample, image)
    batch, _ = runtime.prepare(sample, image)
    with torch.no_grad():
        output = runtime.model.forward_grounded(batch.to(runtime.device), features)
    return {
        "valid": True,
        "text": None,
        "logits": output.mask_logits,
        "geometry": output.predicted_geometry.detach().float().cpu().reshape(-1).tolist(),
    }


# ------------------------------------------------------------------ representation


def representation_diagnostics(runtime, pairs: Sequence[dict]) -> dict:
    """Section 14: does instruction variation produce spatially different geometry?"""

    kind = runtime.model.geometry_kind
    runtime.model.eval()
    records = []
    with torch.no_grad():
        for pair in pairs:
            sample_a = data_mod.to_sample(pair["a"])
            sample_b = data_mod.to_sample(pair["b"])
            image = sample_a.image_rgb()
            features, _ = runtime.features_for(sample_a, image)
            batch_a, _ = runtime.prepare(sample_a, image)
            batch_b, _ = runtime.prepare(sample_b, image)
            output_a = runtime.model.forward_grounded(batch_a.to(runtime.device), features)
            output_b = runtime.model.forward_grounded(batch_b.to(runtime.device), features)
            hidden_a = output_a.seg_hidden.detach().float()
            hidden_b = output_b.seg_hidden.detach().float()
            cosine = float(torch.nn.functional.cosine_similarity(hidden_a, hidden_b).mean())
            l2 = float((hidden_a - hidden_b).norm(dim=-1).mean())
            norm_a = float(hidden_a.norm(dim=-1).mean())
            norm_b = float(hidden_b.norm(dim=-1).mean())
            penultimate_a = runtime.model.grounding_head.act(
                runtime.model.grounding_head.fc1(runtime.model.grounding_head.norm(hidden_a))
            )
            penultimate_b = runtime.model.grounding_head.act(
                runtime.model.grounding_head.fc1(runtime.model.grounding_head.norm(hidden_b))
            )
            penultimate_cosine = float(
                torch.nn.functional.cosine_similarity(penultimate_a, penultimate_b).mean()
            )
            geometry_a = output_a.predicted_geometry.detach().float().reshape(-1)
            geometry_b = output_b.predicted_geometry.detach().float().reshape(-1)
            mask_a = _binarize(output_a.mask_logits, sample_a.target_mask().shape)
            mask_b = _binarize(output_b.mask_logits, sample_b.target_mask().shape)
            union = float((mask_a | mask_b).sum())
            records.append(
                {
                    "image_id": sample_a.image_id,
                    "sample_a": sample_a.sample_id,
                    "sample_b": sample_b.sample_id,
                    "hidden_cosine": cosine,
                    "hidden_l2": l2,
                    "hidden_norm_a": norm_a,
                    "hidden_norm_b": norm_b,
                    "penultimate_cosine": penultimate_cosine,
                    "geometry_a": geometry_a.tolist(),
                    "geometry_b": geometry_b.tolist(),
                    "geometry_l1": float((geometry_a - geometry_b).abs().mean()),
                    "geometry_distance": float((geometry_a - geometry_b).norm()),
                    "mask_iou_a_vs_b": float((mask_a & mask_b).sum()) / union if union else 0.0,
                }
            )
    runtime.model.train()
    # Four-condition control (section 14): separate "same image" from "same template". If a
    # different-image/same-template pair is *more* similar than a same-image/different-template
    # pair, the read-out is dominated by the reasoning template rather than by the image or the
    # target -- which is a sharper statement than "instruction-independent".
    control = _hidden_control(runtime, pairs)
    return {
        "kind": kind,
        "pairs": records,
        "mean_hidden_cosine": _mean([record["hidden_cosine"] for record in records]),
        "mean_hidden_l2": _mean([record["hidden_l2"] for record in records]),
        "mean_penultimate_cosine": _mean([record["penultimate_cosine"] for record in records]),
        "mean_geometry_l1": _mean([record["geometry_l1"] for record in records]),
        "mean_geometry_distance": _mean([record["geometry_distance"] for record in records]),
        "mean_mask_iou_a_vs_b": _mean([record["mask_iou_a_vs_b"] for record in records]),
        "all_geometry_differs": bool(all(record["geometry_l1"] > 0 for record in records)),
        "hidden_control": control,
        "interpretation": _representation_interpretation(records, control),
    }


def _hidden_cosine(runtime, sample_left, sample_right) -> dict:
    image_left = sample_left.image_rgb()
    features_left, _ = runtime.features_for(sample_left, image_left)
    batch_left, _ = runtime.prepare(sample_left, image_left)
    image_right = sample_right.image_rgb()
    features_right, _ = runtime.features_for(sample_right, image_right)
    batch_right, _ = runtime.prepare(sample_right, image_right)
    with torch.no_grad():
        hidden_left = runtime.model.forward_grounded(
            batch_left.to(runtime.device), features_left
        ).seg_hidden.float()
        hidden_right = runtime.model.forward_grounded(
            batch_right.to(runtime.device), features_right
        ).seg_hidden.float()
    return {
        "sample_left": sample_left.sample_id,
        "sample_right": sample_right.sample_id,
        "image_left": sample_left.image_id,
        "image_right": sample_right.image_id,
        "template_left": sample_left.template_id,
        "template_right": sample_right.template_id,
        "same_image": sample_left.image_id == sample_right.image_id,
        "same_template": sample_left.template_id == sample_right.template_id,
        "hidden_cosine": float(torch.nn.functional.cosine_similarity(hidden_left, hidden_right).mean()),
        "hidden_l2": float((hidden_left - hidden_right).norm(dim=-1).mean()),
    }


def _hidden_control(runtime, pairs: Sequence[dict]) -> dict:
    """The four same/different image x same/different template conditions."""

    if len(pairs) < 2:
        return {"available": False, "reason": "needs at least two pairs"}
    pair_a, pair_b = pairs[0], pairs[1]
    same_image_diff_template = _hidden_cosine(
        runtime, data_mod.to_sample(pair_a["a"]), data_mod.to_sample(pair_a["b"])
    )
    diff_image_same_template = _hidden_cosine(
        runtime, data_mod.to_sample(pair_a["a"]), data_mod.to_sample(pair_b["a"])
    )
    diff_image_diff_template = _hidden_cosine(
        runtime, data_mod.to_sample(pair_a["a"]), data_mod.to_sample(pair_b["b"])
    )
    conditions = {
        "same_image_same_template": None,
        "same_image_different_template": same_image_diff_template,
        "different_image_same_template": diff_image_same_template,
        "different_image_different_template": diff_image_diff_template,
    }
    template_dominates = bool(
        diff_image_same_template["hidden_cosine"] > same_image_diff_template["hidden_cosine"]
        and diff_image_same_template["hidden_l2"] < same_image_diff_template["hidden_l2"]
    )
    return {
        "available": True,
        "conditions": conditions,
        "cosines": {
            key: (value["hidden_cosine"] if value else None) for key, value in conditions.items()
        },
        "l2": {key: (value["hidden_l2"] if value else None) for key, value in conditions.items()},
        "template_dominates_image": template_dominates,
        "statement": (
            "a different-image/same-template pair is closer than a same-image/different-template "
            "pair, so the read-out follows the reasoning template more than the image or the target"
            if template_dominates
            else "the read-out follows the image at least as much as the template"
        ),
    }


def _representation_interpretation(records: list, control: dict) -> dict:
    same_image = _mean([record["hidden_cosine"] for record in records])
    different_image = None
    if control.get("available"):
        different_image = (control["conditions"]["different_image_same_template"] or {}).get(
            "hidden_cosine"
        )
    geometry_l1 = _mean([record["geometry_l1"] for record in records])
    collapsed = bool(
        same_image is not None and same_image > 0.999 and geometry_l1 is not None and geometry_l1 < 0.01
    )
    return {
        "same_image_hidden_cosine": same_image,
        "different_image_same_template_hidden_cosine": different_image,
        "same_image_geometry_l1": geometry_l1,
        "hidden_not_more_similar_within_an_image": bool(
            same_image is not None
            and different_image is not None
            and different_image >= same_image
        ),
        "template_dominates_image": bool(control.get("template_dominates_image")),
        "statement": (
            "the [SEG] hidden state is effectively constant for grounding purposes: same-image and "
            "different-image cosines coincide, the predicted geometry does not move, and the "
            "different-image/same-template pair is the closest of all, so the read-out tracks the "
            "reasoning template rather than the target"
            if collapsed and control.get("template_dominates_image")
            else (
                "the [SEG] hidden state does not move with the instruction; see the numbers"
                if collapsed
                else "the [SEG] hidden state varies; see the numbers"
            )
        ),
    }


def _vram() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


@dataclass
class GroundedRun:
    records: list
    seconds: float
    peak_vram: dict
