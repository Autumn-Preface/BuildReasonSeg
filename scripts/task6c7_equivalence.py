#!/usr/bin/env python
"""Task 6C.7 section 7: exact-equivalence proof for the frozen visual-feature cache.

    python scripts/task6c7_equivalence.py [--samples 16] [--steps 12]

Writes `evaluation/task6c7_equivalence.json` in two parts.

1. **Feature equality.** For at least 16 representative samples, compare the features the
   runtime would compute (`original get_image_features`) against what the cache returns on
   a hit: same shape, same dtype, bitwise equality, and the max absolute difference.

2. **End-to-end gate.** Three 12-step sequences on one runtime, same initial trainable
   state, same samples in the same order, per-run re-seeding and strict determinism:

   * `uncached` — cache disabled, the vision tower recomputes every step;
   * `cached_image_key` — cache enabled, keyed by source-image identity;
   * `cached_content_key` — cache enabled, keyed by the pixel content hash;

   plus a control that runs `uncached` twice, because a gate is only meaningful if the
   reference reproduces itself. Losses, gradient fingerprints and post-step parameter
   fingerprints must be identical for adoption (`BIT_EQUIVALENT`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp.determinism import (  # noqa: E402
    gradients_fingerprint,
    trainable_state_fingerprint,
)
from buildreasonseg_mvp.visual_cache import resolve_visual_host  # noqa: E402
from task6c6_common import (  # noqa: E402
    EVAL,
    build_variant_runtime,
    load_benchmark_samples,
    make_optimizer,
    restore_trainable,
    set_seed,
    snapshot_trainable,
    write_json,
)

OUT = EVAL / "task6c7_equivalence.json"

#: Section 7: adoption prefers bit-equivalence; a numeric path needs these tolerances.
TOLERANCES = {"loss_max_abs": 1e-3, "loss_max_rel": 1e-4, "gradient_max_abs": 1e-5,
              "parameter_max_abs": 1e-6, "parameter_max_rel": 1e-6}


def compare_features(reference, candidate) -> dict:
    ref_pooler, cand_pooler = tuple(reference.pooler_output), tuple(candidate.pooler_output)
    ref_deep, cand_deep = tuple(reference.deepstack_features or ()), tuple(candidate.deepstack_features or ())
    shapes_match = [list(t.shape) for t in ref_pooler] == [list(t.shape) for t in cand_pooler] and [
        list(t.shape) for t in ref_deep
    ] == [list(t.shape) for t in cand_deep]
    dtype_match = (ref_pooler[0].dtype == cand_pooler[0].dtype) if ref_pooler and cand_pooler else False
    identical = bool(shapes_match and dtype_match)
    max_abs = 0.0
    if shapes_match:
        for a, b in list(zip(ref_pooler, cand_pooler)) + list(zip(ref_deep, cand_deep)):
            if not torch.equal(a, b):
                identical = False
                max_abs = max(max_abs, float((a.float() - b.float()).abs().max()))
    return {
        "shapes_match": shapes_match,
        "dtype_match": dtype_match,
        "identical": identical,
        "max_abs_difference": max_abs,
        "pooler_shape": [list(t.shape) for t in cand_pooler],
        "deepstack_shapes": [list(t.shape) for t in cand_deep],
        "dtype": str(cand_pooler[0].dtype) if cand_pooler else None,
    }


def feature_equality(runtime, samples) -> dict:
    host = resolve_visual_host(runtime.model.qwen)
    original = host._task6c7_original_get_image_features
    cache = runtime.visual_cache
    cache.clear()

    records = []
    for sample in samples:
        batch, _image = runtime.prepare(sample)
        pixel_values = batch.pixel_values.to(runtime.device)
        grid = batch.image_grid_thw.to(runtime.device) if batch.image_grid_thw is not None else None
        runtime.set_visual_cache_key(sample.image_id)
        with torch.no_grad():
            reference = original(pixel_values, grid)
            first = host.get_image_features(pixel_values, grid)   # miss -> stores
            second = host.get_image_features(pixel_values, grid)  # hit  -> cached
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "reference_vs_first_call": compare_features(reference, first),
                "reference_vs_cache_hit": compare_features(reference, second),
            }
        )
        del reference, first, second

    return {
        "samples_checked": len(records),
        "all_first_calls_identical": all(r["reference_vs_first_call"]["identical"] for r in records),
        "all_cache_hits_identical": all(r["reference_vs_cache_hit"]["identical"] for r in records),
        "max_abs_difference_over_samples": max(
            (r["reference_vs_cache_hit"]["max_abs_difference"] for r in records), default=0.0
        ),
        "cache_stats_after_equality_check": cache.stats(),
        "records": records,
    }


def run_sequence(runtime, samples, steps: int, initial: dict, *, use_cache: bool, key_kind: str) -> dict:
    set_seed(int(runtime.cfg["seed"]))
    restore_trainable(runtime.model, initial)
    runtime.visual_cache.enabled = use_cache
    runtime.visual_cache.clear()
    optimizer = make_optimizer(runtime, kind="default")
    losses: list[dict] = []
    grad_digests: list[str] = []
    param_digests: list[str] = []
    for index in range(steps):
        sample = samples[index % len(samples)]
        runtime.set_visual_cache_key(sample.image_id if key_kind == "image" else None)
        batch, image = runtime.prepare(sample)
        features, _cached = runtime.features_for(sample, image)
        result = runtime.train_step(
            batch,
            sample.target_mask(),
            features,
            optimizer=optimizer,
            collect_grad_norms=False,
        )
        losses.append({key: repr(float(value)) for key, value in result["losses"].items()})
        grad_digests.append(
            gradients_fingerprint([p for p in runtime.model.parameters() if p.requires_grad])["sha256"]
        )
        param_digests.append(trainable_state_fingerprint(runtime.model)["sha256"])
        del result, features, batch
    torch.cuda.synchronize()
    return {
        "losses": losses,
        "gradient_digests": grad_digests,
        "parameter_digests": param_digests,
        "cache_stats": runtime.visual_cache.stats(),
    }


def compare_sequences(reference: dict, candidate: dict) -> dict:
    losses_identical = reference["losses"] == candidate["losses"]
    gradients_identical = reference["gradient_digests"] == candidate["gradient_digests"]
    parameters_identical = reference["parameter_digests"] == candidate["parameter_digests"]
    loss_max_abs = 0.0
    loss_max_rel = 0.0
    for left, right in zip(reference["losses"], candidate["losses"]):
        for key in sorted(set(left) | set(right)):
            a, b = float(left.get(key, "nan")), float(right.get(key, "nan"))
            difference = abs(a - b)
            loss_max_abs = max(loss_max_abs, difference)
            if a:
                loss_max_rel = max(loss_max_rel, difference / abs(a))
    bit_equivalent = bool(losses_identical and gradients_identical and parameters_identical)
    within = bool(
        loss_max_abs <= TOLERANCES["loss_max_abs"] and loss_max_rel <= TOLERANCES["loss_max_rel"]
    )
    return {
        "losses_identical": losses_identical,
        "gradients_identical": gradients_identical,
        "post_step_parameters_identical": parameters_identical,
        "loss_max_abs_difference": loss_max_abs,
        "loss_max_rel_difference": loss_max_rel,
        "bit_equivalent": bit_equivalent,
        "within_declared_loss_tolerance": within,
        "category": "BIT_EQUIVALENT" if bit_equivalent else ("NUMERICALLY_EQUIVALENT" if within else "NOT_EQUIVALENT"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=64)
    args = parser.parse_args(argv)

    samples = load_benchmark_samples(limit=max(args.steps, args.samples, 16))
    runtime = build_variant_runtime()
    install = runtime.install_visual_cache(enabled=True, max_images=args.max_images)
    initial = snapshot_trainable(runtime.model)
    initial_fingerprint = trainable_state_fingerprint(runtime.model)["sha256"]

    equality = feature_equality(runtime, samples[: args.samples])

    uncached = run_sequence(runtime, samples, args.steps, initial, use_cache=False, key_kind="image")
    control = run_sequence(runtime, samples, args.steps, initial, use_cache=False, key_kind="image")
    cached_image = run_sequence(runtime, samples, args.steps, initial, use_cache=True, key_kind="image")
    cached_content = run_sequence(runtime, samples, args.steps, initial, use_cache=True, key_kind="content")

    control_comparison = compare_sequences(uncached, control)
    image_comparison = compare_sequences(uncached, cached_image)
    content_comparison = compare_sequences(uncached, cached_content)

    report = {
        "_doc": (
            "Task 6C.7 section 7. Exact-equivalence proof for the frozen visual-feature cache: "
            "16-sample feature comparison plus a 12-step end-to-end gate with per-run re-seeding, "
            "strict determinism, one uncached reference, an uncached control and two cache key "
            "strategies. Adoption requires BIT_EQUIVALENT."
        ),
        "task": "6C.7",
        "steps": args.steps,
        "install": install,
        "tolerances": TOLERANCES,
        "initial_trainable_sha256": initial_fingerprint,
        "strict_determinism": runtime.reports["determinism"].get("strict_effective"),
        "feature_equality": equality,
        "reference_run_to_run_control": control_comparison,
        "cached_image_key": {**image_comparison, "cache_stats": cached_image["cache_stats"]},
        "cached_content_key": {**content_comparison, "cache_stats": cached_content["cache_stats"]},
        "uncached_reference": {
            "losses": uncached["losses"],
            "cache_stats": uncached["cache_stats"],
        },
        "cached_image_losses": cached_image["losses"],
    }
    report["adoptable"] = bool(
        equality["all_cache_hits_identical"]
        and control_comparison["bit_equivalent"]
        and image_comparison["bit_equivalent"]
        and content_comparison["bit_equivalent"]
    )
    write_json(OUT, report)

    print(f"[task6c7:equivalence] feature equality: {equality['samples_checked']} samples, "
          f"all hits identical={equality['all_cache_hits_identical']} "
          f"max_abs={equality['max_abs_difference_over_samples']}")
    print(f"[task6c7:equivalence] control (uncached twice): {control_comparison['category']}")
    print(f"[task6c7:equivalence] cached image key: {image_comparison['category']} "
          f"(hits={cached_image['cache_stats']['hits']} misses={cached_image['cache_stats']['misses']})")
    print(f"[task6c7:equivalence] cached content key: {content_comparison['category']} "
          f"(hits={cached_content['cache_stats']['hits']} misses={cached_content['cache_stats']['misses']})")
    print(f"[task6c7:equivalence] adoptable={report['adoptable']}")
    print(f"[task6c7:equivalence] wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    del runtime
    torch.cuda.empty_cache()
    return 0 if report["adoptable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
