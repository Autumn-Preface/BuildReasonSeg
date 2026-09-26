"""Task 6B validation: teacher-forced and free-generation paths, kept separate.

Task 6B section 13 requires the two paths to be reported separately and the
**strict end-to-end** free-generation result to be the headline metric:

* **teacher-forced**: the expected `reasoning_zh + " [SEG]"` is fed, and its
  `[SEG]` hidden state drives the mask. This measures the mask pathway given a
  perfect language prefix.
* **free generation (PRIMARY)**: only the image and the instruction are given.
  The model generates text, the generated `[SEG]` is located, the complete
  generated sequence is re-forwarded for its hidden state, and the mask is
  decoded. If `[SEG]` is missing or appears more than once the record is a format
  failure and scores **IoU = 0, Dice = 0** in the strict aggregate.

No ground-truth geometry, component id, reference mask or relation geometry enters
either path; ground truth is used only to score and to draw panels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import torch

from . import data as data_mod
from .language_metrics import (
    build_reasoning_lookup,
    char_similarity,
    exact_match,
    operation_chain_correct,
    reasoning_part,
    summarise_language,
)
from .metrics import collapse_flags, dice_from_logits, mask_iou_from_logits
from .qwen_seg import generate_with_seg, seg_hidden_from_full_sequence
from .sam2_bridge import decode_mask

#: Query-family buckets used by the Task 6B breakdowns.
FAMILY_BY_QUERY_TYPE = {
    "leftmost": "extreme",
    "rightmost": "extreme",
    "topmost": "extreme",
    "bottommost": "extreme",
    "largest": "size",
    "smallest": "size",
}
for _direction in ("left_of", "right_of", "above", "below"):
    FAMILY_BY_QUERY_TYPE[f"largest_to_{_direction}"] = "direction"
    FAMILY_BY_QUERY_TYPE[f"smallest_to_{_direction}"] = "direction"
FAMILY_BY_QUERY_TYPE["largest_to_nearest"] = "nearest"
FAMILY_BY_QUERY_TYPE["smallest_to_nearest"] = "nearest"


def family_of(query_type: str) -> str:
    if query_type in FAMILY_BY_QUERY_TYPE:
        return FAMILY_BY_QUERY_TYPE[query_type]
    if query_type.endswith("_to_nearest") and query_type.count("_to_") == 2:
        return "multi_hop_direction_to_nearest"
    return "other"


def level_key(record: dict) -> str:
    level = int(record["level"])
    if level == 3:
        return "L3_nontrivial" if not record.get("trivial_selection") else "L3_trivial"
    return f"L{level}"


@dataclass
class ValidationRun:
    records: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    peak_vram: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"seconds": round(self.seconds, 2), "peak_vram": self.peak_vram, "per_record": self.records}


# --------------------------------------------------------------------------
# Teacher-forced path
# --------------------------------------------------------------------------


def teacher_forced_validation(
    runtime,
    samples: Sequence[data_mod.Sample],
    reasoning_lookup: dict[str, set[str]] | None = None,
) -> ValidationRun:
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    runtime.model.eval()
    seg_id = runtime.model.seg_token_id
    eos_id = runtime.tokenizer.eos_token_id
    out: list[dict] = []

    with torch.no_grad():
        for sample in samples:
            batch, image = runtime.prepare(sample)
            features, _ = runtime.features_for(sample, image)
            gt = sample.target_mask()
            moved = batch.to(runtime.device)
            output = runtime.model(moved, features)

            iou = mask_iou_from_logits(output.mask_logits, torch.as_tensor(gt).float(), gt.shape)
            dice = dice_from_logits(output.mask_logits, torch.as_tensor(gt).float(), gt.shape)
            flags = collapse_flags(output.mask_logits.detach().float())

            lm_logits = output.lm_logits[0].float()
            labels = moved.labels[0]
            supervised = (labels != -100).nonzero(as_tuple=False).flatten().tolist()
            correct = 0
            reasoning_total = reasoning_correct = 0
            for position in supervised:
                target_id = int(labels[position])
                # `labels[i]` already holds token i+1 and `lm_logits[i]` is the
                # distribution over token i+1, so the matching index is `position`
                # itself. Using `position - 1` scores the previous token instead
                # and under-reports accuracy by roughly the whole sequence.
                hit = int(lm_logits[position].argmax()) == target_id
                correct += int(hit)
                if target_id not in (seg_id, eos_id):
                    reasoning_total += 1
                    reasoning_correct += int(hit)

            # Greedy decoding of the teacher-forced distribution: what the model
            # WOULD have emitted at every supervised position. This gives the
            # teacher-forced path real text metrics instead of the placeholders
            # `language_summary` used to substitute for them.
            if supervised:
                predicted_ids = lm_logits[supervised].argmax(dim=-1).tolist()
                decoded = runtime.tokenizer.decode(predicted_ids, skip_special_tokens=False)
            else:
                decoded = ""
            expected_reasoning = str(sample.raw.get("reasoning_zh", ""))
            text_metrics: dict = {
                "decoded": decoded,
                "reasoning_exact_match": exact_match(decoded, expected_reasoning),
                "reasoning_char_similarity": char_similarity(decoded, expected_reasoning),
                "operation_chain_correct": (
                    operation_chain_correct(decoded, sample.query_type, reasoning_lookup)
                    if reasoning_lookup is not None
                    else False
                ),
            }

            seg_position = moved.seg_position
            seg_logits = lm_logits[seg_position - 1]
            # Standard shifted-target CE over the whole sequence; every prompt
            # position is already -100 in `labels`, so only assistant tokens count.
            lm_ce = float(
                torch.nn.functional.cross_entropy(lm_logits, labels, ignore_index=-100)
            )

            out.append(
                {
                    "sample_id": sample.sample_id,
                    "image_id": sample.image_id,
                    "level": sample.level,
                    "level_key": level_key(sample.raw),
                    "query_type": sample.query_type,
                    "family": family_of(sample.query_type),
                    "target_component_id": sample.target_component_id,
                    "target_area_px": int(gt.sum()),
                    "iou": iou,
                    "dice": dice,
                    "collapse": flags,
                    "lm_ce": lm_ce,
                    "assistant_token_accuracy": correct / max(len(supervised), 1),
                    "reasoning_token_accuracy": (
                        reasoning_correct / reasoning_total if reasoning_total else None
                    ),
                    "seg_token_correct": bool(int(seg_logits.argmax()) == seg_id),
                    **text_metrics,
                }
            )

    runtime.model.train()
    run = ValidationRun(records=out, seconds=time.time() - started)
    run.peak_vram = _vram()
    return run


# --------------------------------------------------------------------------
# Free-generation path (PRIMARY)
# --------------------------------------------------------------------------


def free_generation_validation(runtime, samples: Sequence[data_mod.Sample], lookup: dict) -> ValidationRun:
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    runtime.model.eval()
    config = runtime.cfg["inference"]
    max_new_tokens = int(config.get("validation_max_new_tokens", config["max_new_tokens"]))
    checkpointing_was_on = bool(getattr(runtime.qwen, "is_gradient_checkpointing", False))
    if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_disable"):
        runtime.qwen.gradient_checkpointing_disable()

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
                "family": family_of(sample.query_type),
                "target_component_id": sample.target_component_id,
                "target_area_px": int(gt.sum()),
                "generated_text": generation["text"],
                "generated_reasoning": reasoning_part(generation["text"]),
                "seg_count": generation["seg_count"],
                "seg_valid": bool(valid),
                "generated_token_count": generation["generated_token_count"],
                "reasoning_exact_match": exact_match(generation["text"], sample.reasoning_zh),
                "reasoning_char_similarity": char_similarity(generation["text"], sample.reasoning_zh),
                "operation_chain_correct": operation_chain_correct(
                    generation["text"], sample.query_type, lookup
                ),
                "iou": 0.0,
                "dice": 0.0,
                "iou_conditional": None,
                "dice_conditional": None,
                "collapse": None,
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
                    projected = runtime.projection(hidden)
                    features, _ = runtime.features_for(sample, image)
                    decoded = decode_mask(
                        runtime.model.sam,
                        features,
                        projected,
                        multimask_output=bool(config["multimask_output"]),
                    )
                    iou = mask_iou_from_logits(decoded.low_res_logits, torch.as_tensor(gt).float(), gt.shape)
                    dice = dice_from_logits(decoded.low_res_logits, torch.as_tensor(gt).float(), gt.shape)
                    entry["iou"] = iou
                    entry["dice"] = dice
                    entry["iou_conditional"] = iou
                    entry["dice_conditional"] = dice
                    entry["collapse"] = collapse_flags(decoded.low_res_logits.detach().float())
            out.append(entry)
    finally:
        if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_enable"):
            runtime.qwen.gradient_checkpointing_enable()

    runtime.model.train()
    run = ValidationRun(records=out, seconds=time.time() - started)
    run.peak_vram = _vram()
    return run


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def aggregate(records: Sequence[dict]) -> dict:
    """Strict-aggregate metrics for one group of free-generation records."""

    if not records:
        return {"count": 0}
    valid = [r for r in records if r["seg_valid"]]
    strict_iou = [r["iou"] for r in records]
    strict_dice = [r["dice"] for r in records]
    collapsed = sum(
        1
        for r in records
        if r.get("collapse") and (r["collapse"]["empty"] or r["collapse"]["full"])
    )
    return {
        "count": len(records),
        "valid_seg_emission": len(valid),
        "valid_seg_emission_rate": len(valid) / len(records),
        "strict_end_to_end_miou": _mean(strict_iou),
        "strict_end_to_end_dice": _mean(strict_dice),
        "conditional_miou": _mean([r["iou_conditional"] for r in valid]) if valid else None,
        "conditional_dice": _mean([r["dice_conditional"] for r in valid]) if valid else None,
        "collapsed_samples": collapsed,
        "reasoning_exact_match": _mean([1.0 if r["reasoning_exact_match"] else 0.0 for r in records]),
        "reasoning_char_similarity": _mean([r["reasoning_char_similarity"] for r in records]),
        "operation_chain_accuracy": _mean(
            [1.0 if r["operation_chain_correct"] else 0.0 for r in records]
        ),
        "mean_generated_tokens": _mean([r["generated_token_count"] for r in records]),
    }


def aggregate_teacher_forced(records: Sequence[dict]) -> dict:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "miou": _mean([r["iou"] for r in records]),
        "dice": _mean([r["dice"] for r in records]),
        "lm_ce": _mean([r["lm_ce"] for r in records]),
        "assistant_token_accuracy": _mean([r["assistant_token_accuracy"] for r in records]),
        "reasoning_token_accuracy": _mean(
            [r["reasoning_token_accuracy"] for r in records if r["reasoning_token_accuracy"] is not None]
        ),
        "seg_token_accuracy": _mean([1.0 if r["seg_token_correct"] else 0.0 for r in records]),
    }


def breakdown(records: Sequence[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record[key]), []).append(record)
    return {name: aggregate(items) for name, items in sorted(groups.items())}


def teacher_forced_breakdown(records: Sequence[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record[key]), []).append(record)
    return {name: aggregate_teacher_forced(items) for name, items in sorted(groups.items())}


def language_summary(free_records: Sequence[dict], tf_records: Sequence[dict]) -> dict:
    """Language metrics from the teacher-forced path plus generation fidelity.

    The teacher-forced text metrics come from greedy-decoding the teacher-forced
    distribution (see `teacher_forced_validation`), so `reasoning_exact_match`,
    `operation_chain_accuracy` and `mean_reasoning_char_similarity` are real
    measurements on this path too, not placeholders.
    """

    tf_entries = []
    for record in tf_records:
        tf_entries.append(
            {
                "lm_ce": record["lm_ce"],
                "assistant_token_accuracy": record["assistant_token_accuracy"],
                "reasoning_token_accuracy": record.get("reasoning_token_accuracy"),
                "seg_token_correct": record["seg_token_correct"],
                "reasoning_exact_match": bool(record.get("reasoning_exact_match", False)),
                "operation_chain_correct": bool(record.get("operation_chain_correct", False)),
                "reasoning_char_similarity": float(record.get("reasoning_char_similarity", 0.0)),
            }
        )
    summary = summarise_language(tf_entries) if tf_entries else {"count": 0}
    if free_records:
        summary["generation_reasoning_exact_match"] = _mean(
            [1.0 if r["reasoning_exact_match"] else 0.0 for r in free_records]
        )
        summary["generation_operation_chain_accuracy"] = _mean(
            [1.0 if r["operation_chain_correct"] else 0.0 for r in free_records]
        )
        summary["generation_char_similarity"] = _mean(
            [r["reasoning_char_similarity"] for r in free_records]
        )
    return summary


# --------------------------------------------------------------------------
# Paired instruction probe (Task 6B section 15)
# --------------------------------------------------------------------------


def pair_pass_decision(
    a_iou_on_a: float, a_iou_on_b: float, b_iou_on_a: float, b_iou_on_b: float
) -> bool:
    """Task 6B section 15 pass rule, isolated so it can be unit-tested.

    Prediction A must overlap GT-A more than GT-B, **and** prediction B must
    overlap GT-B more than GT-A. Comparing the two predictions' own-target IoUs
    against each other would be a different (and unsatisfiable) condition.
    """

    return bool(a_iou_on_a > a_iou_on_b and b_iou_on_b > b_iou_on_a)


def paired_probe(runtime, pairs: Sequence[dict], lookup: dict) -> dict:
    """For 20 unseen val image pairs, does the instruction select the right target?

    Each prediction is scored against **both** ground truths of its image, and a
    pair fails if either instruction fails valid `[SEG]` emission.
    """

    config = runtime.cfg["inference"]
    max_new_tokens = int(config.get("validation_max_new_tokens", config["max_new_tokens"]))
    checkpointing_was_on = bool(getattr(runtime.qwen, "is_gradient_checkpointing", False))
    runtime.model.eval()
    if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_disable"):
        runtime.qwen.gradient_checkpointing_disable()

    started = time.time()
    results: list[dict] = []
    try:
        for pair in pairs:
            sample_a = data_mod.to_sample(pair["a"])
            sample_b = data_mod.to_sample(pair["b"])
            image = sample_a.image_rgb()
            gt_a = torch.as_tensor(sample_a.target_mask()).float()
            gt_b = torch.as_tensor(sample_b.target_mask()).float()

            entry = {
                "image_id": sample_a.image_id,
                "a": sample_a.sample_id,
                "b": sample_b.sample_id,
                "a_query_type": sample_a.query_type,
                "b_query_type": sample_b.query_type,
            }
            predictions: dict[str, dict | None] = {}
            for label, sample in (("a", sample_a), ("b", sample_b)):
                generation = generate_with_seg(
                    runtime.model.qwen,
                    runtime.processor,
                    runtime.tokenizer,
                    image,
                    sample.instruction_zh,
                    max_new_tokens=max_new_tokens,
                    seg_token_id=runtime.model.seg_token_id,
                )
                if generation["seg_count"] != 1:
                    predictions[label] = None
                    entry[f"{label}_seg_count"] = generation["seg_count"]
                    continue
                entry[f"{label}_seg_count"] = 1
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
                    projected = runtime.projection(hidden)
                    features, _ = runtime.features_for(sample, image)
                    decoded = decode_mask(runtime.model.sam, features, projected, multimask_output=False)
                    size = sample.target_mask().shape
                    predictions[label] = {
                        "iou_on_a": mask_iou_from_logits(decoded.low_res_logits, gt_a, size),
                        "iou_on_b": mask_iou_from_logits(decoded.low_res_logits, gt_b, size),
                        "generated_reasoning": reasoning_part(generation["text"]),
                    }
                    predictions[label]["operation_chain_correct"] = operation_chain_correct(
                        generation["text"], sample.query_type, lookup
                    )

            passed = False
            if predictions.get("a") and predictions.get("b"):
                a = predictions["a"]
                b = predictions["b"]
                passed = pair_pass_decision(
                    a["iou_on_a"], a["iou_on_b"], b["iou_on_a"], b["iou_on_b"]
                )
                entry["a_iou_on_a"] = a["iou_on_a"]
                entry["a_iou_on_b"] = a["iou_on_b"]
                entry["b_iou_on_a"] = b["iou_on_a"]
                entry["b_iou_on_b"] = b["iou_on_b"]
                entry["a_operation_chain_correct"] = a["operation_chain_correct"]
                entry["b_operation_chain_correct"] = b["operation_chain_correct"]
            entry["passed"] = passed
            results.append(entry)
    finally:
        if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_enable"):
            runtime.qwen.gradient_checkpointing_enable()
        runtime.model.train()

    own = [
        entry[key]
        for entry in results
        for key in ("a_iou_on_a", "b_iou_on_b")
        if key in entry
    ]
    cross = [
        entry[key]
        for entry in results
        for key in ("a_iou_on_b", "b_iou_on_a")
        if key in entry
    ]
    emission_valid = sum(
        1 for entry in results if entry.get("a_seg_count") == 1 and entry.get("b_seg_count") == 1
    )
    return {
        "n_pairs": len(results),
        "passed": sum(1 for entry in results if entry["passed"]),
        "requirement": ">= 14 of 20",
        "pairs_with_both_emissions_valid": emission_valid,
        "mean_own_target_iou": _mean(own),
        "mean_cross_target_iou": _mean(cross),
        "seconds": round(time.time() - started, 2),
        "pairs": results,
    }


def _vram() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "free_gib": round(free / 1024**3, 3),
        "total_gib": round(total / 1024**3, 3),
    }
