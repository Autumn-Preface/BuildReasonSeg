#!/usr/bin/env python
"""Task 6C.6 sections 4, 6, 7 and 10: variant benchmark for the formal batch-1 step.

    python scripts/task6c6_benchmark.py --group baseline
    python scripts/task6c6_benchmark.py --group compile
    python scripts/task6c6_benchmark.py --group optimizer
    python scripts/task6c6_benchmark.py --group checkpointing

Groups write

    baseline        evaluation/task6c6_integrated_baseline.json
    compile         evaluation/task6c6_compile_variants.json
    optimizer       evaluation/task6c6_optimizer_variants.json
    checkpointing   evaluation/task6c6_checkpointing.json

through `scripts/task6c6_common.run_variant`, i.e. all variants share one measurement
implementation: the formal training path, the fixed Task 6C.5 benchmark ids, one sample
per optimizer step, 8 warmup + 64 measured steps, warm SAM2 feature cache, strict
determinism. Interleaved groups repeat the reference between candidates so thermal
drift shows up as a spread instead of as a gain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6c6_common import (  # noqa: E402
    EVAL,
    TASK6C5_ADOPTED_SAMPLES_PER_SEC,
    TASK6C5_REFERENCE_SAMPLES_PER_SEC,
    VariantSpec,
    annotate_speedups,
    bracket_reference,
    load_benchmark_samples,
    primary_metrics,
    run_variant,
    write_json,
)

OUTPUTS = {
    "baseline": EVAL / "task6c6_integrated_baseline.json",
    "compile": EVAL / "task6c6_compile_variants.json",
    "optimizer": EVAL / "task6c6_optimizer_variants.json",
    "checkpointing": EVAL / "task6c6_checkpointing.json",
    "final": EVAL / "task6c6_final_benchmark.json",
    "resource": EVAL / "task6c6_resource_usage.json",
}

#: Section 15 adoption thresholds.
ADOPT_GAIN_PERCENT = 8.0
ADOPT_GAIN_WITH_SECONDARY_PERCENT = 5.0
RESERVED_VRAM_LIMIT_GIB = 14.0
RSS_LIMIT_GIB = 24.0


def baseline_group(samples, steps: int, warmup: int, sampler_interval: float) -> dict:
    """Section 4: integrate first, then confirm the integrated baseline interleaved.

    The pre-integration path (`collect_grad_norms=true`) is interleaved *with* B0.6
    rather than run after it, because the whole point of the comparison is that the two
    differ by one diagnostic sweep. Thermal drift is the dominant error term on this
    laptop, and a run-after-run comparison cannot separate the two.
    """

    specs = [
        VariantSpec(
            name="B0.6_integrated_round1",
            description=(
                "Formal training path with training.collect_grad_norms=false from config."
            ),
        ),
        VariantSpec(
            name="CTRL_pre_integration_grad_norms_true_round1",
            description=(
                "Control: the same code with collect_grad_norms=true, i.e. the pre-integration "
                "diagnostic sweep."
            ),
            collect_grad_norms=True,
        ),
        VariantSpec(
            name="B0.6_integrated_round2",
            description="Second interleaved confirmation of the same integrated baseline.",
        ),
        VariantSpec(
            name="CTRL_pre_integration_grad_norms_true_round2",
            description="Second interleaved run of the pre-integration control.",
            collect_grad_norms=True,
        ),
    ]
    entries = [
        run_variant(spec, samples, steps=steps, warmup=warmup, sampler_interval=sampler_interval)
        for spec in specs
    ]

    def _mean_spread(prefix: str) -> tuple[float | None, float | None]:
        rates = [
            entry["samples_per_sec"]
            for entry in entries
            if entry["variant"].startswith(prefix) and entry.get("samples_per_sec")
        ]
        if not rates:
            return None, None
        mean = sum(rates) / len(rates)
        return mean, (max(rates) - min(rates)) / mean

    b0_mean, b0_spread = _mean_spread("B0.6")
    ctrl_mean, ctrl_spread = _mean_spread("CTRL")
    gain = 100.0 * (b0_mean / ctrl_mean - 1.0) if b0_mean and ctrl_mean else None
    reference_entry = next((entry for entry in entries if entry["variant"].startswith("B0.6")), None)
    return {
        "_doc": (
            "Task 6C.6 section 4. B0.6 is the Task 6C.5 adopted batch-1 path with "
            "collect_grad_norms=false wired into the formal loop by section 1. Two B0.6 rounds "
            "are interleaved with two rounds of the pre-integration path so the integration gain "
            "is measured under the same thermal conditions. Strict determinism ON, gradient "
            "checkpointing ON, batch 1, no caches, no pinning, no prefetch."
        ),
        "task": "6C.6",
        "benchmark_ids": "evaluation/task6c5_benchmark_ids.json",
        "steps_measured": steps,
        "warmup_steps": warmup,
        "batch_size": 1,
        "deterministic_strict": True,
        "reference_task6c5_adopted_samples_per_sec": TASK6C5_ADOPTED_SAMPLES_PER_SEC,
        "reference_target_samples_per_sec": TASK6C5_REFERENCE_SAMPLES_PER_SEC,
        "b0_6": {
            "samples_per_sec_per_round": [
                entry["samples_per_sec"] for entry in entries if entry["variant"].startswith("B0.6")
            ],
            "samples_per_sec_mean": round(b0_mean, 4) if b0_mean else None,
            "relative_spread": round(b0_spread, 5) if b0_spread else None,
            "primary_metrics": primary_metrics(reference_entry) if reference_entry else None,
        },
        "pre_integration_control": {
            "samples_per_sec_per_round": [
                entry["samples_per_sec"] for entry in entries if entry["variant"].startswith("CTRL")
            ],
            "samples_per_sec_mean": round(ctrl_mean, 4) if ctrl_mean else None,
            "relative_spread": round(ctrl_spread, 5) if ctrl_spread else None,
            "collect_grad_norms_effective": next(
                (entry.get("collect_grad_norms_effective") for entry in entries if entry["variant"].startswith("CTRL")),
                None,
            ),
        },
        "integration_gain_percent": round(gain, 3) if gain is not None else None,
        "ordering": [entry["variant"] for entry in entries],
        "entries": entries,
    }


def compile_group(
    samples, steps: int, warmup: int, sampler_interval: float, only: list[str] | None
) -> dict:
    """Section 6: C1a/C1b/C1c across the backends this install can actually run."""

    specs: list[VariantSpec] = [
        VariantSpec(name="uncompiled_reference_first", description="Integrated baseline reference (bracket start)."),
        VariantSpec(
            name="C1a_qwen_inductor_default",
            description="Qwen language model only, inductor default mode.",
            compile_scope="qwen",
            compile_backend="inductor",
        ),
        VariantSpec(
            name="C1a_qwen_inductor_reduce_overhead",
            description="Qwen language model only, mode=reduce-overhead (CUDA-graph trees).",
            compile_scope="qwen",
            compile_backend="inductor",
            compile_mode="reduce-overhead",
        ),
        VariantSpec(
            name="C1c_combined_inductor_default",
            description="Combined trainable model path, inductor default mode.",
            compile_scope="combined",
            compile_backend="inductor",
        ),
        VariantSpec(
            name="C3_cudagraphs_reduce_overhead",
            description=(
                "Section 8: backend=cudagraphs, the CUDA-graph path reduce-overhead would use. "
                "Feasibility probe for graph capture on this stack."
            ),
            compile_scope="combined",
            compile_backend="cudagraphs",
        ),
        VariantSpec(
            name="C1a_qwen_eager",
            description="Qwen language model only, backend=eager: pure dynamo trace, no codegen.",
            compile_scope="qwen",
            compile_backend="eager",
        ),
        VariantSpec(
            name="C1a_qwen_aot_eager",
            description=(
                "Qwen language model only, backend=aot_eager: dynamo graph capture without "
                "Triton codegen, so it runs on an install that has no Triton."
            ),
            compile_scope="qwen",
            compile_backend="aot_eager",
        ),
        VariantSpec(
            name="C1b_decoder_tail_aot_eager",
            description=(
                "Projection + SAM mask-decoder tail only, backend=aot_eager, function form so "
                "dynamo does not trace the parent module twice."
            ),
            compile_scope="decoder_tail",
            compile_backend="aot_eager",
        ),
        VariantSpec(
            name="C1c_combined_aot_eager",
            description="Combined trainable model path, backend=aot_eager.",
            compile_scope="combined",
            compile_backend="aot_eager",
        ),
        VariantSpec(name="uncompiled_reference_last", description="Integrated baseline reference (bracket end)."),
    ]
    if only:
        specs = [
            spec
            for spec in specs
            if spec.name in only or spec.name.startswith("uncompiled_reference")
        ]
    entries = [
        run_variant(spec, samples, steps=steps, warmup=warmup, sampler_interval=sampler_interval)
        for spec in specs
    ]
    bracket = bracket_reference(entries, "uncompiled_reference")
    annotate_speedups(entries, bracket, "uncompiled_reference")
    reference = next((entry for entry in entries if entry["variant"].startswith("uncompiled_reference")), None)
    return {
        "_doc": (
            "Task 6C.6 section 6/8. torch.compile scopes C1a (Qwen LM), C1b (projection + SAM "
            "mask decoder), C1c (combined) plus the CUDA-graph (cudagraphs) feasibility probe. "
            "Failures are recorded, not hidden: a wrapper that could not be installed has "
            "status=compile_failed and a graph that raises on its first call has "
            "status=runtime_failed, both with the exception text. Candidates are compared "
            "against an uncompiled reference run immediately before and after them, because "
            "this laptop drifts by ~17% in absolute throughput between identical runs."
        ),
        "task": "6C.6",
        "steps_measured": steps,
        "warmup_steps": warmup,
        "batch_size": 1,
        "reference_bracket": bracket,
        "reference_samples_per_sec": (reference or {}).get("samples_per_sec"),
        "entries": entries,
    }


def optimizer_group(samples, steps: int, warmup: int, sampler_interval: float) -> dict:
    """Section 7: AdamW implementation and clipping reduction, mathematics unchanged."""

    specs = [
        VariantSpec(name="O_reference_first", description="Current formal path (bracket start)."),
        VariantSpec(
            name="O1_adamw_foreach",
            description="AdamW(foreach=True): multi-tensor reductions instead of one launch per tensor.",
            optimizer_kind="foreach",
        ),
        VariantSpec(
            name="O2_adamw_fused",
            description="AdamW(fused=True): single fused CUDA kernel per parameter group.",
            optimizer_kind="fused",
        ),
        VariantSpec(
            name="O3_clip_foreach_true",
            description="clip_grad_norm_(foreach=True) on the same parameters and threshold.",
            clip_grad_foreach=True,
        ),
        VariantSpec(
            name="O4_clip_foreach_false",
            description="clip_grad_norm_(foreach=False), i.e. the per-tensor reduction path.",
            clip_grad_foreach=False,
        ),
        VariantSpec(name="O_reference_last", description="Current formal path (bracket end)."),
    ]
    entries = [
        run_variant(spec, samples, steps=steps, warmup=warmup, sampler_interval=sampler_interval)
        for spec in specs
    ]
    bracket = bracket_reference(entries, "O_reference")
    annotate_speedups(entries, bracket, "O_reference")
    return {
        "_doc": (
            "Task 6C.6 section 7. AdamW and gradient-clipping implementation audit. Learning "
            "rates, betas, weight decay, clipping threshold and parameter groups are identical "
            "across variants; only the reduction implementation differs, so every variant needs "
            "its own equivalence verdict (see evaluation/task6c6_equivalence.json). The reference "
            "path is run immediately before and after the candidates because this laptop drifts "
            "by ~17% in absolute throughput between identical runs."
        ),
        "task": "6C.6",
        "steps_measured": steps,
        "warmup_steps": warmup,
        "reference_bracket": bracket,
        "entries": entries,
    }


def checkpointing_group(
    samples,
    steps: int,
    warmup: int,
    sampler_interval: float,
    compile_scope: str | None,
    compile_backend: str,
) -> dict:
    """Section 10: checkpointing ON vs OFF, interleaved, on the best candidate path."""

    def _spec(name: str, enabled: bool) -> VariantSpec:
        return VariantSpec(
            name=name,
            description=f"Gradient checkpointing {'ON' if enabled else 'OFF'}.",
            gradient_checkpointing=enabled,
            compile_scope=compile_scope,
            compile_backend=compile_backend,
        )

    order = [
        _spec("CK_on_round1", True),
        _spec("CK_off_round1", False),
        _spec("CK_on_round2", True),
        _spec("CK_off_round2", False),
    ]
    entries = [
        run_variant(spec, samples, steps=steps, warmup=warmup, sampler_interval=sampler_interval)
        for spec in order
    ]

    def _mean(prefix: str) -> tuple[float | None, float | None]:
        rates = [
            entry["samples_per_sec"]
            for entry in entries
            if entry["variant"].startswith(prefix) and entry.get("samples_per_sec")
        ]
        if not rates:
            return None, None
        mean = sum(rates) / len(rates)
        return mean, (max(rates) - min(rates)) / mean

    on_mean, on_spread = _mean("CK_on")
    off_mean, off_spread = _mean("CK_off")
    change = 100.0 * (off_mean / on_mean - 1.0) if on_mean and off_mean else None
    return {
        "_doc": (
            "Task 6C.6 section 10. Gradient checkpointing ON vs OFF, interleaved, measured on the "
            "best compile candidate path when one exists and on the integrated baseline "
            "otherwise. Task 6C.5 measured OFF as 8.58% slower in a single non-interleaved "
            "sweep; this run decides whether that survives interleaving."
        ),
        "task": "6C.6",
        "path_measured": (
            f"compiled {compile_scope} ({compile_backend})" if compile_scope else "integrated baseline B0.6"
        ),
        "steps_measured": steps,
        "warmup_steps": warmup,
        "checkpointing_on": {
            "samples_per_sec_per_round": [
                entry["samples_per_sec"] for entry in entries if entry["variant"].startswith("CK_on")
            ],
            "samples_per_sec_mean": round(on_mean, 4) if on_mean else None,
            "relative_spread": round(on_spread, 5) if on_spread else None,
        },
        "checkpointing_off": {
            "samples_per_sec_per_round": [
                entry["samples_per_sec"] for entry in entries if entry["variant"].startswith("CK_off")
            ],
            "samples_per_sec_mean": round(off_mean, 4) if off_mean else None,
            "relative_spread": round(off_spread, 5) if off_spread else None,
        },
        "off_versus_on_percent": round(change, 3) if change is not None else None,
        "recommendation": (
            "keep checkpointing ON"
            if change is None or change < 8.0
            else "OFF is faster on this path; adoption still requires <14 GiB reserved and equivalence"
        ),
        "entries": entries,
    }


def build_winner_spec(args) -> VariantSpec:
    """The challenger runtime under test in the final head-to-head, from explicit CLI switches.

    With every switch left at its default the challenger *is* the integrated baseline, which
    is the honest "no candidate was adopted" case; `--winner-collect-grad-norms true`
    reconstructs the pre-integration path instead, which is how the section 1 integration
    is confirmed head-to-head.
    """

    return VariantSpec(
        name="CHALLENGER_candidate_runtime",
        description="The candidate runtime, measured head-to-head against the integrated baseline B0.6.",
        optimizer_kind=args.winner_optimizer,
        clip_grad_foreach=(
            None if args.winner_clip_foreach == "default" else args.winner_clip_foreach == "true"
        ),
        compile_scope=args.winner_compile_scope or None,
        compile_backend=args.winner_compile_backend,
        gradient_checkpointing=(
            None if args.winner_checkpointing == "default" else args.winner_checkpointing == "on"
        ),
        collect_grad_norms=(
            None if args.winner_collect_grad_norms == "default" else args.winner_collect_grad_norms == "true"
        ),
    )


def final_group(samples, args) -> dict:
    """Section 15: interleaved head-to-head between B0.6 and the adopted runtime."""

    baseline_spec = VariantSpec(
        name="B0.6_round1", description="Integrated baseline B0.6 (interleaved)."
    )
    baseline_spec_2 = VariantSpec(
        name="B0.6_round2", description="Integrated baseline B0.6 (interleaved, second round)."
    )
    winner_spec = build_winner_spec(args)
    winner_spec_2 = VariantSpec(
        name="CHALLENGER_candidate_runtime_round2",
        description=winner_spec.description,
        optimizer_kind=winner_spec.optimizer_kind,
        clip_grad_foreach=winner_spec.clip_grad_foreach,
        compile_scope=winner_spec.compile_scope,
        compile_backend=winner_spec.compile_backend,
        gradient_checkpointing=winner_spec.gradient_checkpointing,
        collect_grad_norms=winner_spec.collect_grad_norms,
    )
    order = [baseline_spec, winner_spec, baseline_spec_2, winner_spec_2]
    entries = [
        run_variant(spec, samples, steps=args.steps, warmup=args.warmup, sampler_interval=args.sampler_interval)
        for spec in order
    ]

    def _stats(prefix: str) -> dict:
        rates = [
            entry["samples_per_sec"]
            for entry in entries
            if entry["variant"].startswith(prefix) and entry.get("samples_per_sec")
        ]
        if not rates:
            return {"samples_per_sec_per_round": [], "mean": None, "spread": None}
        mean = sum(rates) / len(rates)
        return {
            "samples_per_sec_per_round": rates,
            "mean": round(mean, 4),
            "spread": round((max(rates) - min(rates)) / mean, 5),
        }

    baseline_stats = _stats("B0.6")
    winner_stats = _stats("CHALLENGER")
    speedup = (
        100.0 * (winner_stats["mean"] / baseline_stats["mean"] - 1.0)
        if baseline_stats["mean"] and winner_stats["mean"]
        else None
    )
    winner_entry = next((entry for entry in entries if entry["variant"].startswith("CHALLENGER")), None)
    reserved = ((winner_entry or {}).get("vram") or {}).get("peak_reserved_gib")
    rss = (
        (((winner_entry or {}).get("utilization") or {}).get("process_rss_gib") or {}).get("mean")
    )
    strict = ((winner_entry or {}).get("determinism") or {}).get("strict_effective")

    # Section 15: >=8% is enough on its own; >=5% is enough when it also removes
    # complexity (fewer graph breaks, lower CPU, lower variance).
    secondary = {
        "variance_reduction": bool(
            winner_stats.get("spread") is not None
            and baseline_stats.get("spread") is not None
            and winner_stats["spread"] < baseline_stats["spread"]
        ),
        "compile_graph_breaks_removed": bool(((winner_entry or {}).get("dynamo") or {}).get("graph_breaks", 0) == 0),
    }
    accepted = False
    reason = None
    if speedup is None:
        reason = "the candidate did not produce a throughput number"
    elif reserved is not None and reserved >= RESERVED_VRAM_LIMIT_GIB:
        reason = f"reserved VRAM {reserved} GiB is at or above the {RESERVED_VRAM_LIMIT_GIB} GiB limit"
    elif rss is not None and rss >= RSS_LIMIT_GIB:
        reason = f"RSS {rss} GiB is at or above the {RSS_LIMIT_GIB} GiB limit"
    elif strict is not True:
        reason = "strict deterministic mode is not effective for the candidate"
    elif speedup >= ADOPT_GAIN_PERCENT:
        accepted = True
        reason = f"+{speedup:.2f}% clears the {ADOPT_GAIN_PERCENT}% adoption gate"
    elif speedup >= ADOPT_GAIN_WITH_SECONDARY_PERCENT and any(secondary.values()):
        accepted = True
        reason = (
            f"+{speedup:.2f}% is below {ADOPT_GAIN_PERCENT}% but clears "
            f"{ADOPT_GAIN_WITH_SECONDARY_PERCENT}% and reduces variance or graph breaks"
        )
    else:
        reason = (
            f"+{speedup:.2f}% does not clear the {ADOPT_GAIN_PERCENT}% gate, and the "
            f"{ADOPT_GAIN_WITH_SECONDARY_PERCENT}% route needs a variance or graph-break reduction"
            if speedup is not None
            else "no throughput number"
        )

    baseline_entry = next((entry for entry in entries if entry["variant"].startswith("B0.6")), None)
    integration_gain = None
    baseline_artifact = EVAL / "task6c6_integrated_baseline.json"
    if baseline_artifact.is_file():
        recorded = json.loads(baseline_artifact.read_text(encoding="utf-8"))
        integration_gain = recorded.get("integration_gain_percent")
    return {
        "_doc": (
            "Task 6C.6 section 15. Interleaved head-to-head between the integrated baseline B0.6 "
            "and a challenger runtime, alternating A/B/A/B so both see the same thermal "
            "conditions. Adoption follows section 15: equivalence acceptable, strict determinism "
            "still effective, no OOM/paging, and either >=8% warm throughput or >=5% with a "
            "material variance / CPU / graph-break reduction. With the winner switches at their "
            "defaults the challenger is the pre-integration path "
            "(--winner-collect-grad-norms true), which is the head-to-head confirmation of the "
            "section 1 integration rather than of a new candidate."
        ),
        "task": "6C.6",
        "steps_measured": args.steps,
        "warmup_steps": args.warmup,
        "batch_size": 1,
        "ordering": [entry["variant"] for entry in entries],
        "adopted_runtime": {
            "name": "B0.6",
            "description": (
                "Task 6C.5 adopted batch-1 path with training.collect_grad_norms=false wired into "
                "the formal loop by section 1"
            ),
            "collect_grad_norms": False,
            "compilation": "none",
            "optimizer": "torch.optim.AdamW default resolution (foreach=True, measured)",
            "clipping": "clip_grad_norm_ default (foreach), threshold 1.0",
            "gradient_checkpointing": True,
            "strict_determinism": True,
            "integration_gain_percent_from_baseline_group": integration_gain,
        },
        "baseline": {**baseline_stats, "primary_metrics": primary_metrics(baseline_entry or {})},
        "challenger": {
            **winner_stats,
            "collect_grad_norms": (winner_entry or {}).get("collect_grad_norms_effective"),
            "optimizer_kind": (winner_entry or {}).get("optimizer_kind"),
            "clip_grad_foreach": (winner_entry or {}).get("clip_grad_foreach"),
            "compile_scope": (winner_entry or {}).get("compile_scope"),
            "primary_metrics": primary_metrics(winner_entry or {}),
        },
        "speedup_percent": round(speedup, 3) if speedup is not None else None,
        "secondary_criteria": secondary,
        "adoption_thresholds": {
            "gain_percent": ADOPT_GAIN_PERCENT,
            "gain_with_secondary_percent": ADOPT_GAIN_WITH_SECONDARY_PERCENT,
            "reserved_vram_limit_gib": RESERVED_VRAM_LIMIT_GIB,
            "rss_limit_gib": RSS_LIMIT_GIB,
        },
        "adopted": accepted,
        "adoption_reason": reason,
        "strict_determinism_effective": strict,
        "entries": entries,
    }


def resource_group() -> dict:
    """Section 16: aggregate resource accounting over every Task 6C.6 measurement."""

    sources = {
        "integrated_baseline": EVAL / "task6c6_integrated_baseline.json",
        "compile_variants": EVAL / "task6c6_compile_variants.json",
        "optimizer_variants": EVAL / "task6c6_optimizer_variants.json",
        "final_benchmark": EVAL / "task6c6_final_benchmark.json",
        "profiler": EVAL / "task6c6_profiler_summary.json",
    }
    per_variant: dict = {}
    peak_reserved = 0.0
    peak_allocated = 0.0
    peak_rss = 0.0
    peak_system_ram = 0.0
    hottest = 0.0
    highest_power = 0.0
    oom = False
    adopted_reserved = 0.0
    adopted_allocated = 0.0
    adopted_rss = 0.0
    for source, path in sources.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload.get("entries", []) + payload.get("sdpa_backends", {}).get("variants", []):
            metrics = primary_metrics(entry) if "utilization" in entry else entry.get("primary_metrics", {})
            if not metrics:
                continue
            per_variant[f"{source}:{entry.get('variant')}"] = {
                **metrics,
                "status": entry.get("status"),
                "collect_grad_norms": entry.get("collect_grad_norms_effective"),
                "gradient_checkpointing": entry.get("gradient_checkpointing_enabled"),
            }
            peak_reserved = max(peak_reserved, metrics.get("peak_reserved_vram_gib") or 0.0)
            peak_allocated = max(peak_allocated, metrics.get("peak_allocated_vram_gib") or 0.0)
            peak_rss = max(peak_rss, metrics.get("process_rss_gib_mean") or 0.0)
            hottest = max(hottest, metrics.get("gpu_temperature_c_mean") or 0.0)
            highest_power = max(highest_power, metrics.get("gpu_power_watts_mean") or 0.0)
            utilization = entry.get("utilization") or {}
            peak_system_ram = max(
                peak_system_ram, (utilization.get("system_ram_used_gib") or {}).get("max") or 0.0
            )
            oom = oom or ("out of memory" in str(entry.get("error", "")).lower())
            # The resource question that matters for adoption is the *adopted* runtime's
            # footprint, not the peak across rejected experiments: a CUDA-graph or dynamo
            # candidate can inflate reserved VRAM by 4 GiB without being adoptable.
            if str(entry.get("variant", "")).startswith("B0.6"):
                adopted_reserved = max(adopted_reserved, metrics.get("peak_reserved_vram_gib") or 0.0)
                adopted_allocated = max(adopted_allocated, metrics.get("peak_allocated_vram_gib") or 0.0)
                adopted_rss = max(adopted_rss, metrics.get("process_rss_gib_mean") or 0.0)
    worst = max(per_variant.items(), key=lambda item: item[1].get("peak_reserved_vram_gib") or 0.0, default=None)
    worst_rss = max(per_variant.items(), key=lambda item: item[1].get("process_rss_gib_mean") or 0.0, default=None)
    return {
        "_doc": (
            "Task 6C.6 section 16/14. Resource accounting for every measured runtime, aggregated "
            "from the Task 6C.6 artifacts. System RAM is 32 GiB and the RSS budget is 24 GiB; the "
            "reserved-VRAM limit is 14 GiB of 15.894 GiB. `peak_*` covers every experiment "
            "including rejected candidates; `adopted_runtime_*` is the B0.6 path that is actually "
            "kept, which is the figure the budget statement should use."
        ),
        "task": "6C.6",
        "system_ram_gib": 32.0,
        "adopted_runtime": {
            "name": "B0.6 (integrated baseline)",
            "peak_reserved_vram_gib": round(adopted_reserved, 3),
            "peak_allocated_vram_gib": round(adopted_allocated, 3),
            "process_rss_gib": round(adopted_rss, 3),
            "within_reserved_vram_limit": bool(adopted_reserved < RESERVED_VRAM_LIMIT_GIB),
            "within_rss_limit": bool(adopted_rss < RSS_LIMIT_GIB),
        },
        "peak_process_rss_gib": round(peak_rss, 3),
        "peak_process_rss_limit_gib": RSS_LIMIT_GIB,
        "peak_process_rss_variant": worst_rss[0] if worst_rss else None,
        "peak_system_ram_used_gib": round(peak_system_ram, 3),
        "peak_reserved_vram_gib": round(peak_reserved, 3),
        "peak_reserved_vram_variant": worst[0] if worst else None,
        "peak_allocated_vram_gib": round(peak_allocated, 3),
        "reserved_vram_limit_gib": RESERVED_VRAM_LIMIT_GIB,
        "max_gpu_temperature_c": round(hottest, 2),
        "max_gpu_power_watts": round(highest_power, 2),
        "paging_or_oom": bool(oom),
        "per_variant": per_variant,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=sorted(OUTPUTS))
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--sampler-interval", type=float, default=0.5)
    parser.add_argument("--only", default=None, help="comma-separated variant names (compile group)")
    parser.add_argument(
        "--checkpointing-compile-scope",
        default="",
        help="compile scope to measure the checkpointing comparison on (empty = none)",
    )
    parser.add_argument("--checkpointing-compile-backend", default="aot_eager")
    # Final head-to-head: the adopted runtime under test, all defaults preserving the
    # integrated baseline so an unset flag can never silently introduce a candidate.
    parser.add_argument("--winner-optimizer", default="default", choices=["default", "foreach", "fused"])
    parser.add_argument("--winner-clip-foreach", default="default", choices=["default", "true", "false"])
    parser.add_argument("--winner-compile-scope", default="", choices=["", "qwen", "decoder_path", "decoder_tail", "combined"])
    parser.add_argument("--winner-compile-backend", default="aot_eager")
    parser.add_argument("--winner-checkpointing", default="default", choices=["default", "on", "off"])
    parser.add_argument(
        "--winner-collect-grad-norms",
        default="default",
        choices=["default", "true", "false"],
        help="true reconstructs the pre-integration path, which is how section 1 is confirmed",
    )
    args = parser.parse_args(argv)

    if args.group == "resource":
        payload = resource_group()
        out = OUTPUTS[args.group]
        write_json(out, payload)
        print(
            f"[task6c6:resource] peak RSS {payload['peak_process_rss_gib']} GiB, "
            f"peak reserved VRAM {payload['peak_reserved_vram_gib']} GiB, "
            f"max temp {payload['max_gpu_temperature_c']} C, oom={payload['paging_or_oom']}",
            flush=True,
        )
        print(f"[task6c6:resource] wrote {out.relative_to(REPO_ROOT).as_posix()}", flush=True)
        return 0

    samples = load_benchmark_samples()
    if args.group == "baseline":
        payload = baseline_group(samples, args.steps, args.warmup, args.sampler_interval)
    elif args.group == "compile":
        only = [token.strip() for token in args.only.split(",") if token.strip()] if args.only else None
        payload = compile_group(samples, args.steps, args.warmup, args.sampler_interval, only)
    elif args.group == "optimizer":
        payload = optimizer_group(samples, args.steps, args.warmup, args.sampler_interval)
    elif args.group == "final":
        payload = final_group(samples, args)
    else:
        payload = checkpointing_group(
            samples,
            args.steps,
            args.warmup,
            args.sampler_interval,
            args.checkpointing_compile_scope or None,
            args.checkpointing_compile_backend,
        )

    out = OUTPUTS[args.group]
    write_json(out, payload)
    for entry in payload.get("entries", []):
        print(
            f"[task6c6:{args.group}] {entry['variant']}: status={entry.get('status')} "
            f"samples/s={entry.get('samples_per_sec')} "
            f"vs_ref={entry.get('speedup_percent_vs_interpolated_reference')} "
            f"resolvable={entry.get('gain_resolvable')} "
            f"uncert={entry.get('local_reference_uncertainty_percent')} "
            f"gpu={primary_metrics(entry).get('gpu_utilization_mean')} "
            f"err={str(entry.get('error'))[:110]}",
            flush=True,
        )
    if args.group == "final":
        print(
            f"[task6c6:final] speedup {payload['speedup_percent']}% adopted={payload['adopted']} "
            f"reason={payload['adoption_reason']}",
            flush=True,
        )
    print(f"[task6c6:{args.group}] wrote {out.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
