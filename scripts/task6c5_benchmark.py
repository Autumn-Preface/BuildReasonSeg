#!/usr/bin/env python
"""Task 6C.5: benchmark the batch-1 training pipeline variants.

    python scripts/task6c5_benchmark.py --runtime-mode default
    python scripts/task6c5_benchmark.py --runtime-mode no_checkpointing
    python scripts/task6c5_benchmark.py --runtime-mode det_off

Runs the fixed 64-record benchmark set (32 paired images x 2 instructions, taken from
the Task 6C `P` training subset) through a set of pipeline variants, one sample per
optimizer step, strict deterministic mode, and records warm steady-state throughput
plus system utilization. Every variant starts from the identical initial trainable
state, and results are merged into `evaluation/task6c5_variants.json`.

`--runtime-mode` exists because two candidate changes are runtime-build-time facts
rather than per-step switches: gradient checkpointing ON/OFF and strict deterministic
algorithms ON/OFF. They therefore need their own processes and are never mixed with
the headline pipeline numbers.
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

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.pipeline import PipelineFlags, TrainingPipeline, warm_caches  # noqa: E402
from buildreasonseg_mvp.perf import StageProfile, UtilizationSampler  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
IDS_JSON = EVAL / "task6c5_benchmark_ids.json"
VARIANTS_JSON = EVAL / "task6c5_variants.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"

BENCHMARK_PAIRS_PER_GROUP = 11  # 11 + 11 + 10 = 32 images -> 64 records


def build_benchmark_ids() -> dict:
    """Fixed 64-record benchmark set from the Task 6C P training subset."""

    payload = json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8"))
    pairs = payload["P"]["pairs"]
    groups: dict[str, list[dict]] = {}
    for pair in pairs:
        key = f"L{pair['a_level']}_L{pair['b_level']}"
        groups.setdefault(key, []).append(pair)

    order = [("L1_L2", 11), ("L1_L3", 11), ("L2_L3", 10)]
    chosen_pairs: list[dict] = []
    for key, quota in order:
        pool = sorted(groups.get(key, []), key=lambda p: p["a"])
        if len(pool) < quota:
            raise RuntimeError(f"benchmark group {key} has only {len(pool)} pairs, needs {quota}")
        chosen_pairs.extend(pool[:quota])

    record_ids: list[str] = []
    for pair in chosen_pairs:
        record_ids.extend([pair["a"], pair["b"]])

    levels: dict[str, int] = {}
    for pair in chosen_pairs:
        for level in (pair["a_level"], pair["b_level"]):
            levels[str(level)] = levels.get(str(level), 0) + 1

    return {
        "_doc": (
            "Task 6C.5 section 6 benchmark set: 64 records over 32 paired images, drawn deterministically "
            "from the Task 6C P training subset. Fixed ids and fixed order; no validation or test data."
        ),
        "task": "6C.5",
        "source": "evaluation/task6c_subset_ids.json -> P.pairs",
        "n_records": len(record_ids),
        "n_images": len({pair["image_id"] for pair in chosen_pairs}),
        "pairs_per_group": {key: quota for key, quota in order},
        "record_ids": record_ids,
        "pairs": [
            {
                "image_id": pair["image_id"],
                "a": pair["a"],
                "b": pair["b"],
                "a_level": pair["a_level"],
                "b_level": pair["b_level"],
            }
            for pair in chosen_pairs
        ],
        "level_counts_from_ids": levels,
        "test_split_used": False,
    }


def load_benchmark_samples() -> list:
    if not IDS_JSON.is_file():
        IDS_JSON.write_text(
            json.dumps(build_benchmark_ids(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    payload = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    train = {record["sample_id"]: record for record in data_mod.read_records("train")}
    return [data_mod.to_sample(train[sample_id]) for sample_id in payload["record_ids"]]


def variant_flags(runtime_mode: str) -> list[tuple[str, PipelineFlags, dict]]:
    if runtime_mode == "no_checkpointing":
        return [("B5_no_gradient_checkpointing", PipelineFlags(), {"gradient_checkpointing": False})]
    if runtime_mode == "det_off":
        return [("DET_algorithms_off", PipelineFlags(), {"deterministic_strict": False, "deterministic": False})]
    return [
        ("B0_current", PipelineFlags(), {}),
        ("B1_source_cache", PipelineFlags(source_cache=True), {}),
        ("B2_preprocessed_cache", PipelineFlags(source_cache=True, preprocessed_cache=True), {}),
        (
            "B3_pinned_nonblocking",
            PipelineFlags(source_cache=True, preprocessed_cache=True, pin_memory=True, non_blocking=True),
            {},
        ),
        (
            "B4_prefetch_threads_2",
            PipelineFlags(source_cache=True, preprocessed_cache=True, prefetch_threads=2),
            {},
        ),
        (
            "B4_prefetch_threads_4",
            PipelineFlags(source_cache=True, preprocessed_cache=True, prefetch_threads=4),
            {},
        ),
        (
            "B4_prefetch_threads_8",
            PipelineFlags(source_cache=True, preprocessed_cache=True, prefetch_threads=8),
            {},
        ),
        (
            "B6_skip_grad_norm_instrumentation",
            PipelineFlags(skip_grad_norm_instrumentation=True),
            {},
        ),
        (
            "B9_caches_plus_skip_grad_norms",
            PipelineFlags(
                source_cache=True, preprocessed_cache=True, skip_grad_norm_instrumentation=True
            ),
            {},
        ),
        (
            "B7_cache_plus_skip_grad_norms_plus_prefetch2",
            PipelineFlags(
                source_cache=True,
                preprocessed_cache=True,
                prefetch_threads=2,
                skip_grad_norm_instrumentation=True,
            ),
            {},
        ),
    ]


def snapshot_trainable(model) -> dict:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_trainable(model, snapshot: dict) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in snapshot:
                parameter.copy_(snapshot[name])


def make_optimizer(runtime):
    cfg = runtime.cfg["optimizer"]["phase_b"]
    groups = runtime.model.trainable_parameter_groups(
        lora_lr=float(cfg["lora_lr"]),
        head_lr=float(cfg["token_lr"]),
        weight_decay=float(runtime.cfg["optimizer"]["weight_decay"]),
        decoder_lr=float(cfg.get("decoder_lr", cfg["token_lr"])),
        token_lr=float(cfg["token_lr"]),
    )
    return torch.optim.AdamW(groups, betas=tuple(runtime.cfg["optimizer"]["betas"]))


def run_variant(runtime, samples, name, flags, steps, warmup, sampler_interval, cold_info) -> dict:
    pipeline = TrainingPipeline(runtime, flags)
    optimizer = make_optimizer(runtime)

    cold_started = time.perf_counter()
    cache_cold = warm_caches(pipeline, samples)
    cold_seconds = time.perf_counter() - cold_started

    if flags.prefetch_threads > 0:
        # the exact ordered request sequence, so the pool serves step i with sample i
        sequence = [samples[i % len(samples)] for i in range(warmup)] + [
            samples[i % len(samples)] for i in range(steps)
        ]
        pipeline.start_prefetch(sequence)

    try:
        # warmup: not measured
        for index in range(warmup):
            sample = samples[index % len(samples)]
            prepared = (
                pipeline.next_prepared(sample, index)
                if flags.prefetch_threads
                else pipeline.prepare(sample)
            )
            pipeline.step(sample, prepared, optimizer)

        torch.cuda.synchronize()
        timer = StageProfile()
        measured_started = time.perf_counter()
        with UtilizationSampler(interval_seconds=sampler_interval) as sampler:
            for index in range(steps):
                step_started = time.perf_counter()
                sample = samples[index % len(samples)]
                prepared = (
                    pipeline.next_prepared(sample, warmup + index)
                    if flags.prefetch_threads
                    else pipeline.prepare(sample)
                )
                pipeline.step(sample, prepared, optimizer, timer=timer)
                timer.step_walls.append(time.perf_counter() - step_started)
        torch.cuda.synchronize()
        measured_seconds = time.perf_counter() - measured_started
        utilization = sampler.summary()
    finally:
        pipeline.stop_prefetch()

    return {
        "variant": name,
        "flags": flags.as_dict(),
        "runtime_overrides": cold_info.get("runtime_overrides", {}),
        "runtime_mode": cold_info["runtime_mode"],
        "steps_measured": steps,
        "warmup_steps": warmup,
        "measured_seconds": round(measured_seconds, 4),
        "ms_per_sample": round(1000.0 * measured_seconds / steps, 4),
        "samples_per_sec": round(steps / measured_seconds, 4),
        "step_wall_p50_seconds": timer.summary()["step_wall_seconds"]["median"],
        "step_wall_p90_seconds": timer.summary()["step_wall_seconds"]["p90"],
        "stage_profile": timer.summary(),
        "cache_cold_build_seconds": round(cold_seconds, 4),
        "cache_stats": pipeline.stats(),
        "prefetch_queue_final": None,
        "vram": {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        },
        "utilization": utilization,
        "gradient_checkpointing": cold_info["gradient_checkpointing"],
        "deterministic_strict": cold_info["deterministic_strict"],
        "equivalence_status": "not evaluated here (see evaluation/task6c5_equivalence.json)",
        "adopted": False,
        "rejection_reason": None,
    }


def merge_variants(entries: list[dict]) -> None:
    existing = json.loads(VARIANTS_JSON.read_text(encoding="utf-8")) if VARIANTS_JSON.is_file() else {}
    if not isinstance(existing, dict):
        existing = {}
    record = existing.get("variants", {})
    for entry in entries:
        record[entry["variant"]] = entry
    existing.update(
        {
            "_doc": (
                "Task 6C.5 section 25: every pipeline variant that was tried, including slower and "
                "rejected ones. Warm steady-state, one sample per optimizer step, fixed sample order."
            ),
            "task": "6C.5",
            "benchmark": "evaluation/task6c5_benchmark_ids.json",
            "variants": record,
        }
    )
    VARIANTS_JSON.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-mode", default="default", choices=("default", "no_checkpointing", "det_off")
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--sampler-interval", type=float, default=0.5)
    parser.add_argument(
        "--only", default=None, help="comma-separated variant names to run (default: all for the mode)"
    )
    args = parser.parse_args(argv)

    samples = load_benchmark_samples()
    print(f"[task6c5] benchmark set: {len(samples)} records", flush=True)

    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": "centre"}
    overrides = {}
    if args.runtime_mode == "no_checkpointing":
        cfg["training"]["gradient_checkpointing"] = False
        overrides["gradient_checkpointing"] = False
    elif args.runtime_mode == "det_off":
        cfg["training"]["deterministic"] = False
        overrides["deterministic"] = False

    cold_started = time.perf_counter()
    runtime = build_runtime(cfg, device="cuda", verbose=False)
    runtime_load_seconds = time.perf_counter() - cold_started
    cold_info = {
        "runtime_mode": args.runtime_mode,
        "runtime_overrides": overrides,
        "gradient_checkpointing": bool(runtime.reports.get("gradient_checkpointing", False)),
        "deterministic_strict": bool(runtime.reports["determinism"].get("strict_effective", False)),
        "runtime_load_seconds": round(runtime_load_seconds, 3),
    }
    print(f"[task6c5] runtime loaded in {runtime_load_seconds:.1f}s: {cold_info}", flush=True)

    # Warm the shared SAM2 feature cache once; every variant then runs warm.
    warm_started = time.perf_counter()
    from buildreasonseg_mvp.pipeline import TrainingPipeline as _TP

    warmer = _TP(runtime, PipelineFlags())
    sam_warm = warm_caches(warmer, samples)
    cold_info["sam_feature_cache_warm_seconds"] = round(time.perf_counter() - warm_started, 3)
    cold_info["sam_feature_cache_stats"] = runtime.feature_cache.stats()

    initial = snapshot_trainable(runtime.model)
    entries = []
    selected = variant_flags(args.runtime_mode)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        selected = [item for item in selected if item[0] in wanted]
        missing = wanted - {item[0] for item in selected}
        if missing:
            raise SystemExit(f"unknown or unavailable variants for mode {args.runtime_mode}: {sorted(missing)}")
    for name, flags, variant_overrides in selected:
        restore_trainable(runtime.model, initial)
        torch.cuda.reset_peak_memory_stats()
        runtime.feature_cache.hits = 0
        runtime.feature_cache.misses = 0
        info = dict(cold_info)
        info["runtime_overrides"] = variant_overrides
        print(f"[task6c5] running {name} ...", flush=True)
        entry = run_variant(runtime, samples, name, flags, args.steps, args.warmup, args.sampler_interval, info)
        print(
            f"[task6c5] {name}: {entry['samples_per_sec']:.3f} samples/s "
            f"({entry['ms_per_sample']:.1f} ms/sample) "
            f"gpu={entry['utilization']['gpu_utilization_percent']['mean']} "
            f"cpu={entry['utilization']['cpu_total_percent']['mean']}",
            flush=True,
        )
        entries.append(entry)

    merge_variants(entries)
    print(f"[task6c5] merged {len(entries)} variants into {VARIANTS_JSON.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
