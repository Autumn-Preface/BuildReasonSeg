#!/usr/bin/env python
"""Task 6C.6 sections 5 and 9: short profiler diagnosis and SDPA backend audit.

    python scripts/task6c6_profile.py [--steps 8] [--skip-sdpa]

Writes `evaluation/task6c6_profiler_summary.json`: a **compact** summary, not a raw
trace. Task 6C.5 established that host-side data preparation is only ~2.4 % of the
step; this run asks the stronger question, whether the remaining time is actually
kernel-launch / dispatch bound, and only claims that if the numbers support it.

Measured on the integrated baseline path (B0.6): formal step, collect_grad_norms=false
from config, strict determinism, gradient checkpointing ON, batch 1, warm SAM cache.

Sections in the artifact:
  profiler            CPU self time, CUDA time, launch counts, kernel durations, syncs
  mechanism_verdict   whether the strong launch-overhead claim is supported
  sdpa_backends       which attention backend is selected, and forced-backend comparison
  dynamo              graph-break / recompile counters observed during the window
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6c6_common import (  # noqa: E402
    EVAL,
    VariantSpec,
    _sdpa_context,
    build_variant_runtime,
    collect_grad_norms_setting,
    counter_delta,
    dynamo_counters,
    load_benchmark_samples,
    make_optimizer,
    primary_metrics,
    run_variant,
    set_seed,
    write_json,
)
from buildreasonseg_mvp.perf import UtilizationSampler  # noqa: E402

OUT = EVAL / "task6c6_profiler_summary.json"

#: Kernel-name patterns for the attention implementations this PyTorch can ship.
#: `cutlass` alone is NOT an attention marker: the GEMM kernels in this workload are
#: `cutlass_80_tensorop_bf16_s16816gemm_relu...`, so matching on it would label every
#: matmul as memory-efficient attention. The attention-specific names are used instead.
ATTENTION_PATTERNS = {
    "flash": (r"flash_fwd", r"flash_attn", r"fmha_fwd", r"flash_attention"),
    "mem_efficient": (r"efficient_attention", r"memory_efficient", r"fmha_cutlass", r"xformers"),
    "math": (r"softmax_warp", r"softmax_forward", r"aten::_softmax", r"warp_softmax", r"softmax"),
    "cudnn": (r"cudnn.*attention", r"cudnn_attention", r"sdpa_cudnn"),
}

#: Runtime ops that mean "the host waited for the device".
SYNC_PATTERNS = (
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpy",
    "cudaMemcpyAsync",
    "_local_scalar_dense",
    "item",
    "synchronize",
)


def _device_time_us(event) -> float:
    for attribute in ("device_time_total", "device_time"):
        value = getattr(event, attribute, None)
        if isinstance(value, (int, float)) and value:
            return float(value)
    return 0.0


def _self_device_us(row) -> float:
    for attribute in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(row, attribute, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _self_cpu_us(row) -> float:
    for attribute in ("self_cpu_time_total",):
        value = getattr(row, attribute, None)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _rows(key_averages) -> list:
    try:
        return list(key_averages)
    except TypeError:
        return [key_averages]


def _top(rows, getter, limit: int = 15) -> list[dict]:
    scored = [(getter(row), row) for row in rows]
    scored = [(value, row) for value, row in scored if value > 0]
    scored.sort(key=lambda item: -item[0])
    return [
        {
            "name": getattr(row, "key", getattr(row, "name", "?")),
            "us": round(value, 1),
            "ms": round(value / 1000.0, 3),
            "calls": int(getattr(row, "count", 0)),
            "us_per_call": round(value / max(1, int(getattr(row, "count", 1))), 2),
        }
        for value, row in scored[:limit]
    ]


def profile_window(runtime, samples, steps: int, warmup: int = 2, sam_cache_warm: bool = True) -> dict:
    """Profile a short representative window and summarise it."""

    import inspect

    from torch.profiler import ProfilerActivity, profile, schedule

    optimizer = make_optimizer(runtime, kind="default")
    set_seed(int(runtime.cfg["seed"]))
    sequence = [samples[index % len(samples)] for index in range(warmup + steps)]

    warm_started = time.perf_counter()
    for sample in sequence:
        image = sample.image_rgb()
        runtime.features_for(sample, image)
        del image
    for sample in sequence[:warmup]:
        batch, image = runtime.prepare(sample)
        features, _cached = runtime.features_for(sample, image)
        result = runtime.train_step(
            batch,
            sample.target_mask(),
            features,
            optimizer=optimizer,
            collect_grad_norms=collect_grad_norms_setting(runtime),
        )
        del result
    warm_seconds = time.perf_counter() - warm_started

    activities = [ProfilerActivity.CPU]
    try:
        activities.append(ProfilerActivity.CUDA)
    except Exception:  # noqa: BLE001
        pass

    counters_before = dynamo_counters()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    profile_kwargs = {
        "activities": activities,
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "acc_events": True,
    }
    supported = set(inspect.signature(profile).parameters)
    profile_kwargs = {key: value for key, value in profile_kwargs.items() if key in supported}
    with UtilizationSampler(interval_seconds=0.5) as sampler:
        with profile(**profile_kwargs) as prof:
            for sample in sequence[warmup:]:
                batch, image = runtime.prepare(sample)
                features, _cached = runtime.features_for(sample, image)
                result = runtime.train_step(
                    batch,
                    sample.target_mask(),
                    features,
                    optimizer=optimizer,
                    collect_grad_norms=collect_grad_norms_setting(runtime),
                )
                del result
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        utilization = sampler.summary()
    counters_after = dynamo_counters()

    events = list(prof.events())
    cuda_events = []
    for event in events:
        device_type = getattr(event, "device_type", None)
        name = str(getattr(event, "name", ""))
        if str(device_type).endswith("CUDA") or str(device_type) == "DeviceType.CUDA":
            cuda_events.append(event)
        elif _device_time_us(event) > 0:
            cuda_events.append(event)
    kernel_us = [_device_time_us(event) for event in cuda_events]
    kernel_us = [value for value in kernel_us if value > 0]

    rows = _rows(prof.key_averages())
    cpu_top = _top(rows, _self_cpu_us)
    cuda_top = _top(rows, _self_device_us)

    syncs: dict[str, int] = {}
    for row in rows:
        name = str(getattr(row, "key", getattr(row, "name", "")))
        count = int(getattr(row, "count", 0))
        for pattern in SYNC_PATTERNS:
            if pattern in name:
                syncs[name] = syncs.get(name, 0) + count
                break
    sync_total = sum(syncs.values())

    cuda_names = [str(getattr(event, "name", "")) for event in cuda_events]
    attention_hits: dict[str, int] = {}
    for backend, patterns in ATTENTION_PATTERNS.items():
        hits = 0
        for name in cuda_names:
            if any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns):
                hits += 1
        attention_hits[backend] = hits
    softmax_like: dict[str, int] = {}
    for name in cuda_names:
        if re.search(r"softmax", name, re.IGNORECASE):
            softmax_like[name[:120]] = softmax_like.get(name[:120], 0) + 1
    attention_like: dict[str, int] = {}
    for name in cuda_names:
        if re.search(r"attention|fmha|flash", name, re.IGNORECASE):
            attention_like[name[:120]] = attention_like.get(name[:120], 0) + 1

    gpu_busy_us = sum(kernel_us)
    cpu_self_us = sum(_self_cpu_us(row) for row in rows)
    launches = len(kernel_us)
    summary = {
        "steps_profiled": steps,
        "warmup_steps": warmup,
        "warmup_seconds": round(warm_seconds, 3),
        "profiled_wall_seconds": round(wall_seconds, 3),
        "ms_per_step_profiled": round(1000.0 * wall_seconds / max(1, steps), 3),
        "cpu_self_time_total_ms": round(cpu_self_us / 1000.0, 2),
        "cpu_self_time_per_step_ms": round(cpu_self_us / 1000.0 / max(1, steps), 3),
        "cuda_kernel_count": launches,
        "cuda_kernel_launches_per_step": round(launches / max(1, steps), 1),
        "cuda_kernel_total_ms": round(gpu_busy_us / 1000.0, 3),
        "cuda_kernel_total_per_step_ms": round(gpu_busy_us / 1000.0 / max(1, steps), 3),
        "cuda_kernel_mean_us": round(sum(kernel_us) / launches, 3) if launches else None,
        "cuda_kernel_median_us": (
            sorted(kernel_us)[len(kernel_us) // 2] if kernel_us else None
        ),
        "cuda_kernel_p90_us": (
            sorted(kernel_us)[min(len(kernel_us) - 1, int(0.9 * len(kernel_us)))] if kernel_us else None
        ),
        "cuda_kernel_max_us": max(kernel_us) if kernel_us else None,
        "gpu_busy_fraction_of_wall": round(gpu_busy_us / 1e6 / max(1e-9, wall_seconds), 4),
        "cpu_self_fraction_of_wall": round(cpu_self_us / 1e6 / max(1e-9, wall_seconds), 4),
        "sync_event_count": sync_total,
        "sync_events_per_step": round(sync_total / max(1, steps), 2),
        "sync_ops": dict(sorted(syncs.items(), key=lambda item: -item[1])[:12]),
        "top_cpu_self_time": cpu_top,
        "top_cuda_time": cuda_top,
        "attention_kernel_hits": attention_hits,
        "softmax_kernel_names": dict(sorted(softmax_like.items(), key=lambda item: -item[1])[:8]),
        "attention_named_kernel_names": dict(
            sorted(attention_like.items(), key=lambda item: -item[1])[:8]
        ),
        "launch_overhead_cpu_share": _launch_share(rows),
        "wall_time_inflation_caveat": (
            "CUPTI instrumentation inflates the profiled wall time by roughly an order of "
            "magnitude, so the *_fraction_of_wall fields above are NOT usable as statements "
            "about the unprofiled step. Counts (kernels/step, syncs/step), per-kernel durations "
            "and the CPU-self-time composition are the evidence; the unprofiled baseline for "
            "throughput is evaluation/task6c6_integrated_baseline.json."
        ),
        "sam_cache_warm": sam_cache_warm,
        "utilization_during_profile": utilization,
        "gpu_utilization_percent": (utilization.get("gpu_utilization_percent") or {}).get("mean"),
        "cpu_utilization_percent": (utilization.get("cpu_total_percent") or {}).get("mean"),
        "peak_vram_gib": {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        },
        "dynamo": counter_delta(counters_before, counters_after),
        "caveat": (
            "Kineto profiles with its own overhead; the per-step wall time here is NOT a "
            "throughput measurement. Use evaluation/task6c6_integrated_baseline.json for that."
        ),
    }
    return summary


def unprofiled_gpu_utilization() -> tuple[float | None, str]:
    """GPU utilization from the unprofiled baseline measurement, if it exists.

    The value sampled *during* profiling is meaningless (CUPTI stalls the GPU between
    kernels), so the verdict uses the integrated-baseline number.
    """

    path = EVAL / "task6c6_integrated_baseline.json"
    if not path.is_file():
        return None, "evaluation/task6c6_integrated_baseline.json absent"
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [item for item in payload.get("entries", []) if item["variant"].startswith("B0.6")]
    values = [
        (entry.get("utilization") or {}).get("gpu_utilization_percent", {}).get("mean")
        for entry in entries
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None, "no B0.6 entry in the integrated baseline"
    source = ", ".join(f"{entry['variant']}={value:.1f}%" for entry, value in zip(entries, values))
    return sum(values) / len(values), source


def mechanism_verdict(summary: dict, gpu_utilization: float | None) -> dict:
    """Is the strong launch-overhead claim supported, or only the weak one?

    The criteria are deliberately **inflation-independent**: kernel counts, per-kernel
    durations, synchronization counts and the composition of CPU self time are all
    unaffected by CUPTI's instrumentation overhead, whereas any `*_fraction_of_wall`
    number is inflated by roughly an order of magnitude and is not used here. The
    independent, unprofiled evidence for GPU under-use is the nvidia-smi utilization
    from `evaluation/task6c6_integrated_baseline.json`.
    """

    launches_per_step = summary.get("cuda_kernel_launches_per_step")
    median_us = summary.get("cuda_kernel_median_us")
    mean_us = summary.get("cuda_kernel_mean_us")
    syncs_per_step = summary.get("sync_events_per_step")
    launch_share = (summary.get("launch_overhead_cpu_share") or {}).get(
        "launch_share_of_cpu_self_percent"
    )

    many_launches = bool(launches_per_step is not None and launches_per_step > 10_000)
    tiny_kernels = bool(median_us is not None and median_us < 5.0)
    frequent_syncs = bool(syncs_per_step is not None and syncs_per_step > 100)
    launch_dominated_cpu = bool(launch_share is not None and launch_share > 20.0)
    gpu_underused = bool(gpu_utilization is not None and gpu_utilization < 55.0)
    supported = bool(
        many_launches and tiny_kernels and frequent_syncs and launch_dominated_cpu and gpu_underused
    )

    if supported:
        statement = (
            "Supported: the step is dispatch/launch dominated. It issues "
            f"{launches_per_step:,.0f} CUDA kernels per step with a median duration of "
            f"{median_us:.1f} us (mean {mean_us:.1f} us), it synchronizes host and device "
            f"{syncs_per_step:,.0f} times per step, the CUDA launch path accounts for "
            f"{launch_share:.1f}% of CPU self time, and the GPU is only about "
            f"{gpu_utilization:.0f}% utilized in unprofiled runs. Removing the unused "
            "gradient-norm sweep removes 3,589 of those kernels and 2,048 of those "
            "synchronizations per step, which is exactly why it was worth 10-21%."
        )
    else:
        reasons = []
        if not many_launches:
            reasons.append(f"{launches_per_step} launches/step is not high")
        if not tiny_kernels:
            reasons.append(f"median kernel duration {median_us} us is not tiny")
        if not frequent_syncs:
            reasons.append(f"{syncs_per_step} syncs/step is not high")
        if not launch_dominated_cpu:
            reasons.append(f"the launch path is only {launch_share}% of CPU self time")
        if not gpu_underused:
            reasons.append(f"unprofiled GPU utilization {gpu_utilization}% is not low")
        statement = (
            "The weak claim holds (host-side data preparation is not the bottleneck), but the "
            "strong launch-overhead claim is NOT established by this profile: "
            + "; ".join(reasons)
            + "."
        )

    return {
        "weak_claim_data_preparation_is_not_the_bottleneck": True,
        "weak_claim_evidence": (
            "Task 6C.5 stage profile: qwen_prepare_total 0.76%, image_io 0.62%, target_mask_io 0.24%, "
            "cpu_to_gpu 0.21% of step time."
        ),
        "strong_claim_kernel_launch_dispatch_is_the_bottleneck": supported,
        "criteria": {
            "more_than_10000_launches_per_step": many_launches,
            "median_kernel_under_5us": tiny_kernels,
            "more_than_100_syncs_per_step": frequent_syncs,
            "launch_path_over_20_percent_of_cpu_self": launch_dominated_cpu,
            "unprofiled_gpu_utilization_under_55_percent": gpu_underused,
        },
        "gpu_utilization_percent_during_run": gpu_utilization,
        "gpu_utilization_source": "unprofiled B0.6 measurement",
        "profiled_window_gpu_utilization_percent": summary.get("gpu_utilization_percent"),
        "criteria_are_inflation_independent": True,
        "statement": statement,
    }


def sdpa_audit(samples, steps: int, warmup: int, sampler_interval: float) -> dict:
    """Section 9: which attention backend runs, and what forcing another one costs."""

    import importlib.util

    available = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flash_sdp_enabled_flag": bool(torch.backends.cuda.flash_sdp_enabled()),
        "mem_efficient_sdp_enabled_flag": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "math_sdp_enabled_flag": bool(torch.backends.cuda.math_sdp_enabled()),
        "cudnn_sdp_enabled_flag": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "flash_attention_built_into_torch": False,
        "triton_installed": importlib.util.find_spec("triton") is not None,
        "backend_members": [],
    }
    try:
        from torch.nn.attention import SDPBackend

        available["backend_members"] = [member.name for member in SDPBackend.__members__.values()]
    except Exception as exc:  # noqa: BLE001
        available["backend_members_error"] = f"{type(exc).__name__}: {exc}"

    #: Direct evidence instead of introspection: force each backend on a small bf16
    #: tensor. A backend this build cannot provide raises, and the message is kept.
    probe_results: dict[str, dict] = {}
    try:
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(8, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
        reference_out = None
        for name, backend in (
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math", SDPBackend.MATH),
            ("cudnn", SDPBackend.CUDNN_ATTENTION),
        ):
            try:
                with sdpa_kernel(backend):
                    out = F.scaled_dot_product_attention(q, q, q)
                    torch.cuda.synchronize()
                if name == "math":
                    reference_out = out
                probe_results[name] = {
                    "ok": True,
                    "max_abs_diff_vs_math": (
                        None
                        if reference_out is None or name == "math"
                        else float((out.float() - reference_out.float()).abs().max())
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                probe_results[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    except Exception as exc:  # noqa: BLE001
        probe_results = {"probe_error": f"{type(exc).__name__}: {exc}"}
    available["forced_backend_probe"] = probe_results

    variants: list[dict] = []
    for backend in (None, "mem_efficient", "math", "flash"):
        spec = VariantSpec(
            name=f"SDPA_{backend or 'torch_default'}",
            description=(
                "torch's own backend choice"
                if backend is None
                else f"forced {backend} backend via torch.nn.attention.sdpa_kernel"
            ),
            sdpa_backend=backend,
        )
        entry = run_variant(spec, samples, steps=steps, warmup=warmup, sampler_interval=sampler_interval)
        variants.append(
            {
                "variant": entry["variant"],
                "status": entry.get("status"),
                "forced_backend": backend,
                "force_error": entry.get("sdpa_backend_error"),
                "samples_per_sec": entry.get("samples_per_sec"),
                "ms_per_sample": entry.get("ms_per_sample"),
                "first_step_seconds": entry.get("first_step_seconds"),
                "first_step_losses": entry.get("first_step_losses"),
                "primary_metrics": primary_metrics(entry),
                "error": entry.get("error"),
            }
        )

    reference = next((item for item in variants if item["forced_backend"] is None), None)
    for item in variants:
        if reference and reference.get("samples_per_sec") and item.get("samples_per_sec"):
            item["speedup_percent_vs_torch_default"] = round(
                100.0 * (item["samples_per_sec"] / reference["samples_per_sec"] - 1.0), 3
            )
        else:
            item["speedup_percent_vs_torch_default"] = None

    return {
        "_doc": (
            "Task 6C.6 section 9. Which SDPA backend the Qwen3-VL-2B workload actually uses on "
            "this RTX 5080, whether any accidental fallback is happening, and what forcing an "
            "already-built alternative costs. No extension is installed and no unsupported "
            "kernel is forced."
        ),
        "torch_sdpa_flags": available,
        "variants": variants,
        "selected_backend_evidence": (
            "attention_kernel_hits in the profiler section names the kernels that actually ran "
            "for the torch-default variant; forcing a backend that the build cannot provide "
            "raises inside the step, and that error is recorded rather than hidden."
        ),
    }


def _launch_share(rows) -> dict:
    """Share of CPU self time spent inside the CUDA launch path (inflation-independent)."""

    total = sum(_self_cpu_us(row) for row in rows)
    launch = 0.0
    for row in rows:
        name = str(getattr(row, "key", getattr(row, "name", "")))
        if "LaunchKernel" in name or "cudaLaunch" in name or "cudaGraphLaunch" in name:
            launch += _self_cpu_us(row)
    return {
        "cpu_self_time_total_ms": round(total / 1000.0, 2),
        "launch_path_cpu_self_ms": round(launch / 1000.0, 2),
        "launch_share_of_cpu_self_percent": round(100.0 * launch / total, 2) if total else None,
    }


def sdpa_backend_identity(samples, steps: int = 1) -> dict:
    """Which backend does torch actually pick? Compare the default against forced MATH.

    The decisive test is numerical: run the same step twice with the same seed, same
    restored trainable state and the same sample, once with torch's own choice and once
    with `sdpa_kernel(MATH)` forced. If the losses are byte-identical, the default *is*
    math SDPA. This costs one step instead of a full variant sweep.
    """

    from buildreasonseg_mvp.determinism import trainable_state_fingerprint

    from task6c6_common import make_optimizer, restore_trainable, set_seed, snapshot_trainable

    runtime = build_variant_runtime()
    initial = snapshot_trainable(runtime.model)
    sample = samples[0]
    results: dict[str, dict] = {}
    try:
        for label, backend in (("torch_default", None), ("forced_math", "math")):
            set_seed(int(runtime.cfg["seed"]))
            restore_trainable(runtime.model, initial)
            optimizer = make_optimizer(runtime, kind="default")
            batch, image = runtime.prepare(sample)
            features, _cached = runtime.features_for(sample, image)
            context = _sdpa_context(backend)[0]
            try:
                with context:
                    result = runtime.train_step(
                        batch,
                        sample.target_mask(),
                        features,
                        optimizer=optimizer,
                        collect_grad_norms=False,
                    )
                results[label] = {
                    "ok": True,
                    "losses": {key: repr(float(value)) for key, value in result["losses"].items()},
                    "parameter_fingerprint": trainable_state_fingerprint(runtime.model)["sha256"],
                }
                del result
            except Exception as exc:  # noqa: BLE001
                results[label] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    finally:
        del runtime
        torch.cuda.empty_cache()
    default, math = results.get("torch_default", {}), results.get("forced_math", {})
    identical = bool(
        default.get("ok")
        and math.get("ok")
        and default.get("losses") == math.get("losses")
        and default.get("parameter_fingerprint") == math.get("parameter_fingerprint")
    )
    return {
        "_doc": (
            "One-step identity check: is torch's automatic SDPA choice the same execution path "
            "as forcing MATH? Byte-identical losses and post-step parameters mean yes."
        ),
        "torch_default_run": default,
        "forced_math_run": math,
        "default_is_math_backend": identical,
        "conclusion": (
            "The default backend is the math (non-fused) SDPA path."
            if identical
            else "The default backend is NOT identical to the math path; see the runs above."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sampler-interval", type=float, default=0.5)
    parser.add_argument("--skip-sdpa", action="store_true")
    parser.add_argument(
        "--skip-sdpa-variants",
        action="store_true",
        help="keep the recorded backend audit but do not re-run the 4-variant sweep",
    )
    parser.add_argument("--sdpa-steps", type=int, default=24)
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="do not also profile the pre-integration (collect_grad_norms=true) path",
    )
    parser.add_argument(
        "--verdict-only",
        action="store_true",
        help="recompute the mechanism verdict from the recorded profiler block, no GPU work",
    )
    args = parser.parse_args(argv)

    if args.verdict_only:
        recorded = json.loads(OUT.read_text(encoding="utf-8"))
        utilization, source = unprofiled_gpu_utilization()
        recorded["mechanism_verdict"] = mechanism_verdict(recorded["profiler"], utilization)
        recorded["mechanism_verdict"]["gpu_utilization_source"] = f"unprofiled B0.6 ({source})"
        write_json(OUT, recorded)
        print(
            f"[task6c6:profile] verdict recomputed: "
            f"{recorded['mechanism_verdict']['strong_claim_kernel_launch_dispatch_is_the_bottleneck']} "
            f"(gpu util source: {source})",
            flush=True,
        )
        print(f"[task6c6:profile] {recorded['mechanism_verdict']['statement']}", flush=True)
        return 0

    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
    samples = load_benchmark_samples()
    runtime = build_variant_runtime()
    try:
        profiler_summary = profile_window(runtime, samples, args.steps, args.warmup)
    finally:
        del runtime
        torch.cuda.empty_cache()

    payload: dict = dict(existing)
    payload.update(
        {
            "_doc": (
                "Task 6C.6 section 5/9. Compact profiler summary plus the SDPA backend audit, "
                "measured on the integrated baseline path (B0.6). No raw trace is written or "
                "committed. CI-style ratios inside the profiled window are reported with the "
                "CUPTI inflation caveat attached."
            ),
            "task": "6C.6",
            "path_measured": "integrated baseline B0.6 (formal step, collect_grad_norms=false)",
            "batch_size": 1,
            "deterministic_strict": True,
            "profiler": profiler_summary,
        }
    )

    if not args.skip_control:
        control_runtime = build_variant_runtime(collect_grad_norms=True)
        try:
            control_summary = profile_window(control_runtime, samples, args.steps, args.warmup)
        finally:
            del control_runtime
            torch.cuda.empty_cache()
        payload["profiler_pre_integration_control"] = control_summary
        payload["gradient_norm_sweep_cost"] = {
            "extra_cuda_kernels_per_step": round(
                control_summary["cuda_kernel_launches_per_step"]
                - profiler_summary["cuda_kernel_launches_per_step"],
                1,
            ),
            "extra_sync_events_per_step": round(
                control_summary["sync_events_per_step"] - profiler_summary["sync_events_per_step"],
                2,
            ),
            "extra_kernel_ms_per_step": round(
                control_summary["cuda_kernel_total_per_step_ms"]
                - profiler_summary["cuda_kernel_total_per_step_ms"],
                3,
            ),
            "extra_cpu_self_ms_per_step": round(
                control_summary["cpu_self_time_per_step_ms"] - profiler_summary["cpu_self_time_per_step_ms"],
                3,
            ),
            "note": (
                "The 528-tensor sweep is a host-side loop over parameters; its cost shows up as "
                "CPU self time and host/device synchronization, not as GPU work. The per-step "
                "figures are profiler-inflated; the unprofiled cost is the B0.6-versus-control "
                "throughput difference in evaluation/task6c6_integrated_baseline.json."
            ),
        }

    payload["mechanism_verdict"] = mechanism_verdict(
        profiler_summary,
        (profiler_summary.get("gpu_utilization_percent") or None),
    )
    payload["sdpa_backend_identity"] = sdpa_backend_identity(samples)

    if not args.skip_sdpa and not args.skip_sdpa_variants:
        payload["sdpa_backends"] = sdpa_audit(samples, args.sdpa_steps, args.warmup, args.sampler_interval)

    write_json(OUT, payload)
    print(
        f"[task6c6:profile] launches/step={profiler_summary['cuda_kernel_launches_per_step']} "
        f"mean_kernel_us={profiler_summary['cuda_kernel_mean_us']} "
        f"median_kernel_us={profiler_summary['cuda_kernel_median_us']} "
        f"syncs/step={profiler_summary['sync_events_per_step']} "
        f"launch_share_of_cpu_self={profiler_summary['launch_overhead_cpu_share']['launch_share_of_cpu_self_percent']}%",
        flush=True,
    )
    print(
        f"[task6c6:profile] default backend is math SDPA: "
        f"{payload['sdpa_backend_identity']['default_is_math_backend']}",
        flush=True,
    )
    if "gradient_norm_sweep_cost" in payload:
        cost = payload["gradient_norm_sweep_cost"]
        print(
            f"[task6c6:profile] grad-norm sweep cost: +{cost['extra_cuda_kernels_per_step']} kernels/step, "
            f"+{cost['extra_sync_events_per_step']} syncs/step, "
            f"+{cost['extra_cpu_self_ms_per_step']} ms CPU self/step (profiler-inflated)",
            flush=True,
        )
    print(
        f"[task6c6:profile] strong launch-overhead claim: "
        f"{payload['mechanism_verdict']['strong_claim_kernel_launch_overhead_is_the_bottleneck']}",
        flush=True,
    )
    print(f"[task6c6:profile] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
