#!/usr/bin/env python
"""Task 6C.7 sections 5-8: is the frozen Qwen3-VL visual tower cache-safe?

    python scripts/task6c7_visual_cache.py [--pairs 8]

Writes `evaluation/task6c7_visual_cache_eligibility.json`. Nothing here changes training
semantics: it inspects the loaded runtime, calls the vision tower, and measures the cache
boundary and its footprint. If any eligibility condition fails, the artifact says exactly
which one and section 5 forbids caching.

Checks
------
1. every visual-tower parameter has requires_grad=False;
2. zero LoRA modules inside the visual tower;
3. the visual output for one image is bit-identical when recomputed;
4. no training-time state in the visual tower changes during a real optimizer step;
5. the visual output does not depend on the instruction text, the `[SEG]` adapter, LoRA
   parameters or optimizer state (same image, two instructions -> identical pixels and
   identical features);
6. bonus, and the reason bit-equivalence is plausible rather than lucky: no dropout with
   p > 0 and no buffers that could change, so the tower consumes no RNG.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.determinism import trainable_state_fingerprint  # noqa: E402
from buildreasonseg_mvp.visual_cache import (  # noqa: E402
    BYTES_PER_GIB,
    VisualFeatureCache,
    resolve_visual_host,
    visual_tower_trainable_report,
)
from task6c6_common import (  # noqa: E402
    EVAL,
    build_variant_runtime,
    load_benchmark_samples,
    make_optimizer,
    set_seed,
    write_json,
)

OUT = EVAL / "task6c7_visual_cache_eligibility.json"
#: Section 8 asks for both estimates.
TASK6C_IMAGES = 480
FULL_WHU_TRAIN_IMAGES = 2508


def _bytes_of(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if torch.is_tensor(t))


def _feature_signature(output) -> dict:
    pooler = tuple(output.pooler_output)
    deepstack = tuple(output.deepstack_features or ())
    return {
        "pooler_shapes": [list(t.shape) for t in pooler],
        "deepstack_shapes": [list(t.shape) for t in deepstack],
        "dtype": str(pooler[0].dtype),
        "device": str(pooler[0].device),
        "pooler_bytes": _bytes_of(pooler),
        "deepstack_bytes": _bytes_of(deepstack),
        "last_hidden_state_bytes": (
            output.last_hidden_state.numel() * output.last_hidden_state.element_size()
            if output.last_hidden_state is not None
            else 0
        ),
    }


def _equal(left, right) -> tuple[bool, float]:
    """Bitwise equality plus the max absolute difference, for the record."""

    left_pooler, right_pooler = tuple(left.pooler_output), tuple(right.pooler_output)
    left_deepstack, right_deepstack = tuple(left.deepstack_features or ()), tuple(right.deepstack_features or ())
    if len(left_pooler) != len(right_pooler) or len(left_deepstack) != len(right_deepstack):
        return False, float("inf")
    identical = True
    max_abs = 0.0
    for a, b in list(zip(left_pooler, right_pooler)) + list(zip(left_deepstack, right_deepstack)):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, float("inf")
        if not torch.equal(a, b):
            identical = False
            max_abs = max(max_abs, float((a.float() - b.float()).abs().max()))
    return identical, max_abs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=8, help="same-image instruction pairs to compare")
    parser.add_argument("--max-images", type=int, default=512)
    args = parser.parse_args(argv)

    samples = load_benchmark_samples()
    runtime = build_variant_runtime()
    report: dict = {
        "_doc": (
            "Task 6C.7 sections 5-8. Cache eligibility for the frozen Qwen3-VL visual tower, the "
            "narrowest safe boundary, the footprint of the cached tensors, and a measured "
            "recompute-determinism check. No training semantics are changed by this script."
        ),
        "task": "6C.7",
        "runtime": "formal Phase-B path with collect_grad_norms=false (integrated baseline B0.6)",
        "batch_size": 1,
        "strict_determinism": runtime.reports["determinism"].get("strict_effective"),
    }

    # -- 1/2/4/6: static + training-time state ---------------------------------
    report["visual_tower_static_check"] = visual_tower_trainable_report(runtime.model.qwen)

    qwen = runtime.model.qwen
    visual_host = resolve_visual_host(qwen)
    visual_model = visual_host.visual
    report["boundary"] = {
        "qwen_class": type(qwen).__name__,
        "language_model_class": type(getattr(qwen, "model", None)).__name__,
        "visual_class": type(visual_model).__name__,
        "interface": "Qwen3VLModel.get_image_features(pixel_values, image_grid_thw)",
        "consumed_by_forward": [
            "get_image_features(...).pooler_output  -> torch.cat -> masked_scatter into inputs_embeds",
            "get_image_features(...).deepstack_features -> deepstack injection at early LM layers",
        ],
        "not_cached": [
            "vision last_hidden_state (pre-merger, not read by the language model)",
            "language-model hidden states",
            "[SEG] hidden state",
            "logits",
            "anything downstream of a trainable parameter",
        ],
    }

    # -- 3: recompute determinism ----------------------------------------------
    sample = samples[0]
    batch, image = runtime.prepare(sample)
    pixel_values = batch.pixel_values.to(runtime.device)
    grid = batch.image_grid_thw.to(runtime.device) if batch.image_grid_thw is not None else None

    get_features = getattr(getattr(qwen, "model", qwen), "get_image_features")
    with torch.no_grad():
        first = get_features(pixel_values, grid)
        second = get_features(pixel_values, grid)
    identical, max_abs = _equal(first, second)
    report["recompute_determinism"] = {
        "identical": identical,
        "max_abs_difference": max_abs,
        "interpretation": "bit-identical recomputation means the tower is a pure function here",
    }
    report["feature_signature"] = _feature_signature(first)
    report["feature_signature"]["bytes_per_image_pooler_plus_deepstack"] = (
        report["feature_signature"]["pooler_bytes"] + report["feature_signature"]["deepstack_bytes"]
    )
    report["feature_signature"]["bytes_per_image_including_last_hidden_state"] = (
        report["feature_signature"]["pooler_bytes"]
        + report["feature_signature"]["deepstack_bytes"]
        + report["feature_signature"]["last_hidden_state_bytes"]
    )

    # -- 4: no training-time state changes across a real optimizer step ---------
    before_params = trainable_state_fingerprint(runtime.model)["sha256"]
    visual_before = {
        name: parameter.detach().clone()
        for name, parameter in visual_model.named_parameters()
    }
    buffer_before = {
        name: buffer.detach().clone()
        for name, buffer in visual_model.named_buffers()
    }
    optimizer = make_optimizer(runtime, kind="default")
    set_seed(int(runtime.cfg["seed"]))
    features, _cached = runtime.features_for(sample, image)
    runtime.train_step(
        batch,
        sample.target_mask(),
        features,
        optimizer=optimizer,
        collect_grad_norms=False,
    )
    visual_changed = [
        name
        for name, parameter in visual_model.named_parameters()
        if not torch.equal(parameter.detach(), visual_before[name])
    ]
    buffers_changed = [
        name
        for name, buf in visual_model.named_buffers()
        if name in buffer_before and not torch.equal(buf.detach(), buffer_before[name])
    ]
    report["training_time_state"] = {
        "visual_parameters_changed_by_step": visual_changed,
        "visual_buffers_changed_by_step": buffers_changed,
        "visual_state_stable": not visual_changed and not buffers_changed,
        "registered_buffers": sorted(buffer_before)[:12],
        "buffer_count": len(buffer_before),
        "trainable_fingerprint_before_step": before_params,
        "trainable_fingerprint_after_step": trainable_state_fingerprint(runtime.model)["sha256"],
        "note": (
            "the visual tower must not move; the trainable fingerprint is expected to change. "
            "Registered buffers are compared too, so a running statistic that moved would fail "
            "this check even though it is not a parameter."
        ),
    }

    # -- 5: independence of the instruction text -------------------------------
    by_image: dict[str, list] = {}
    for candidate in samples:
        by_image.setdefault(candidate.image_id, []).append(candidate)
    pairs = [group for group in by_image.values() if len(group) >= 2][: args.pairs]

    pair_records = []
    for group in pairs:
        left, right = group[0], group[1]
        left_batch, left_image = runtime.prepare(left)
        right_batch, right_image = runtime.prepare(right)
        same_pixels = bool(
            torch.equal(left_batch.pixel_values, right_batch.pixel_values)
            and (
                (left_batch.image_grid_thw is None and right_batch.image_grid_thw is None)
                or torch.equal(left_batch.image_grid_thw, right_batch.image_grid_thw)
            )
        )
        with torch.no_grad():
            left_features = get_features(
                left_batch.pixel_values.to(runtime.device),
                left_batch.image_grid_thw.to(runtime.device) if left_batch.image_grid_thw is not None else None,
            )
            right_features = get_features(
                right_batch.pixel_values.to(runtime.device),
                right_batch.image_grid_thw.to(runtime.device) if right_batch.image_grid_thw is not None else None,
            )
        features_identical, features_max_abs = _equal(left_features, right_features)
        instructions_differ = left.instruction_zh != right.instruction_zh
        pair_records.append(
            {
                "image_id": left.image_id,
                "sample_ids": [left.sample_id, right.sample_id],
                "instructions_differ": instructions_differ,
                "pixel_values_identical": same_pixels,
                "visual_features_identical": features_identical,
                "visual_features_max_abs_difference": features_max_abs,
            }
        )
    report["instruction_independence"] = {
        "pairs_checked": len(pair_records),
        "all_pixels_identical": all(record["pixel_values_identical"] for record in pair_records),
        "all_features_identical": all(record["visual_features_identical"] for record in pair_records),
        "instructions_differ_in_all_pairs": all(record["instructions_differ"] for record in pair_records),
        "pairs": pair_records,
    }

    # -- 8: cache footprint ----------------------------------------------------
    cache = VisualFeatureCache(max_images=args.max_images, enabled=True)
    started = time.perf_counter()
    for candidate in samples[: min(8, len(samples))]:
        candidate_batch, _image = runtime.prepare(candidate)
        with torch.no_grad():
            output = get_features(
                candidate_batch.pixel_values.to(runtime.device),
                candidate_batch.image_grid_thw.to(runtime.device)
                if candidate_batch.image_grid_thw is not None
                else None,
            )
        key, key_kind = cache.key_for(candidate_batch.pixel_values, candidate_batch.image_grid_thw, candidate.image_id)
        cache.put(key, output.pooler_output, output.deepstack_features or (), key_kind)
    build_seconds = time.perf_counter() - started
    per_image = cache.bytes_per_image()
    report["cache_footprint"] = {
        "stored_fields": ["pooler_output", "deepstack_features"],
        "dtype_kept": report["feature_signature"]["dtype"],
        "bytes_per_image": round(per_image, 1),
        "mib_per_image": round(per_image / 1024**2, 3),
        f"task6c_{TASK6C_IMAGES}_images_gib": round(per_image * TASK6C_IMAGES / BYTES_PER_GIB, 3),
        f"full_whu_train_{FULL_WHU_TRAIN_IMAGES}_images_gib": round(
            per_image * FULL_WHU_TRAIN_IMAGES / BYTES_PER_GIB, 3
        ),
        "measured_build_seconds_for_8_images": round(build_seconds, 3),
        "measured_build_ms_per_image": round(1000.0 * build_seconds / max(1, min(8, len(samples))), 2),
        "default_max_images": args.max_images,
        "default_max_gib": round(per_image * args.max_images / BYTES_PER_GIB, 3),
        "budget_gib": 8.0,
        "within_budget_at_default_bound": bool(per_image * args.max_images / BYTES_PER_GIB <= 8.0),
        "within_budget_for_full_whu_train": bool(
            per_image * FULL_WHU_TRAIN_IMAGES / BYTES_PER_GIB <= 8.0
        ),
    }

    # -- verdict ---------------------------------------------------------------
    static = report["visual_tower_static_check"]
    conditions = {
        "all_visual_parameters_frozen": bool(static["all_parameters_frozen"]),
        "zero_visual_lora": bool(static["zero_visual_lora"]),
        "visual_output_deterministic": bool(identical),
        "no_visual_state_moves_during_a_step": bool(report["training_time_state"]["visual_state_stable"]),
        "independent_of_instruction": bool(report["instruction_independence"]["all_features_identical"]),
        "no_visual_dropout": bool(not static["dropout_modules_with_p_gt_0"]),
    }
    report["eligibility"] = {
        "conditions": conditions,
        "eligible": all(conditions.values()),
        "failed_conditions": [name for name, ok in conditions.items() if not ok],
    }
    write_json(OUT, report)

    print(f"[task6c7:eligibility] eligible={report['eligibility']['eligible']} "
          f"failed={report['eligibility']['failed_conditions']}")
    print(f"[task6c7:eligibility] pooler {report['feature_signature']['pooler_shapes']} "
          f"deepstack {report['feature_signature']['deepstack_shapes']} "
          f"dtype {report['feature_signature']['dtype']}")
    print(f"[task6c7:eligibility] bytes/image {report['cache_footprint']['bytes_per_image']:.0f} "
          f"({report['cache_footprint']['mib_per_image']} MiB); 480 images "
          f"{report['cache_footprint'][f'task6c_{TASK6C_IMAGES}_images_gib']} GiB; "
          f"2508 images {report['cache_footprint'][f'full_whu_train_{FULL_WHU_TRAIN_IMAGES}_images_gib']} GiB")
    print(f"[task6c7:eligibility] recompute identical={identical} max_abs={max_abs}")
    print(f"[task6c7:eligibility] wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    del runtime
    torch.cuda.empty_cache()
    return 0 if report["eligibility"]["eligible"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
