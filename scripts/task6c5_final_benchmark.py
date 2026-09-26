#!/usr/bin/env python
"""Task 6C.5 sections 22 and 25: interleaved final benchmark.

    python scripts/task6c5_final_benchmark.py

Runs the candidates in an interleaved order, twice each:

    B0 -> candidate(s) -> B0 -> candidate(s)

so a cold run is never compared against a thermally saturated one, and reports the
spread between repeats. Writes `evaluation/task6c5_final_benchmark.json`, which is the
throughput number the final report is allowed to quote.
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

from buildreasonseg_mvp.pipeline import PipelineFlags, TrainingPipeline, warm_caches  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config  # noqa: E402
from task6c5_benchmark import (  # noqa: E402
    CONFIG,
    load_benchmark_samples,
    restore_trainable,
    run_variant,
    snapshot_trainable,
)

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6c5_final_benchmark.json"

#: Candidates compared head to head. `B0_current` is the reference and is repeated in
#: every round so thermal drift is visible.
CANDIDATES: list[tuple[str, PipelineFlags]] = [
    ("B0_current", PipelineFlags()),
    ("B6_skip_grad_norm_instrumentation", PipelineFlags(skip_grad_norm_instrumentation=True)),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--sampler-interval", type=float, default=0.5)
    args = parser.parse_args(argv)

    samples = load_benchmark_samples()
    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": "centre"}
    runtime = build_runtime(cfg, device="cuda", verbose=False)

    sam_warm = warm_caches(TrainingPipeline(runtime, PipelineFlags()), samples)
    initial = snapshot_trainable(runtime.model)
    cold_info = {
        "runtime_mode": "default",
        "runtime_overrides": {},
        "gradient_checkpointing": bool(runtime.reports.get("gradient_checkpointing", False)),
        "deterministic_strict": bool(runtime.reports["determinism"].get("strict_effective", False)),
    }

    entries: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        for name, flags in CANDIDATES:
            label = f"{name}__round{round_index}"
            restore_trainable(runtime.model, initial)
            torch.cuda.reset_peak_memory_stats()
            print(f"[final] round {round_index}: {name} ...", flush=True)
            entry = run_variant(
                runtime,
                samples,
                label,
                flags,
                args.steps,
                args.warmup,
                args.sampler_interval,
                dict(cold_info),
            )
            entry["base_variant"] = name
            entry["round"] = round_index
            print(
                f"[final] {label}: {entry['samples_per_sec']:.3f} samples/s  "
                f"gpu={entry['utilization']['gpu_utilization_percent']['mean']:.1f}  "
                f"cpu={entry['utilization']['cpu_total_percent']['mean']:.1f}  "
                f"temp={entry['utilization']['gpu_temperature_c']['mean']}  "
                f"power={entry['utilization']['gpu_power_watts']['mean']}",
                flush=True,
            )
            entries.append(entry)

    summary: dict[str, dict] = {}
    for name, _flags in CANDIDATES:
        rates = [e["samples_per_sec"] for e in entries if e["base_variant"] == name]
        gpus = [
            e["utilization"]["gpu_utilization_percent"]["mean"]
            for e in entries
            if e["base_variant"] == name
        ]
        cpus = [
            e["utilization"]["cpu_total_percent"]["mean"]
            for e in entries
            if e["base_variant"] == name
        ]
        summary[name] = {
            "samples_per_sec_per_round": rates,
            "samples_per_sec_mean": sum(rates) / len(rates),
            "relative_spread": (
                (max(rates) - min(rates)) / (sum(rates) / len(rates)) if len(rates) > 1 else 0.0
            ),
            "gpu_utilization_mean": sum(gpus) / len(gpus),
            "cpu_utilization_mean": sum(cpus) / len(cpus),
            "flags": dict(next(flags.as_dict() for candidate, flags in CANDIDATES if candidate == name)),
        }

    baseline_rate = summary["B0_current"]["samples_per_sec_mean"]
    for name, data in summary.items():
        data["speedup_percent_vs_B0"] = round(100.0 * (data["samples_per_sec_mean"] / baseline_rate - 1.0), 3)

    best = max(
        (name for name in summary if name != "B0_current"),
        key=lambda name: summary[name]["samples_per_sec_mean"],
    )

    report = {
        "_doc": (
            "Task 6C.5 sections 22/25 interleaved final benchmark. Warm steady-state, one sample per "
            "optimizer step, strict deterministic mode, fixed sample order, every run restored to the "
            "identical initial trainable state. Candidates are interleaved with the B0 reference so "
            "thermal drift is visible."
        ),
        "task": "6C.5",
        "steps_measured": args.steps,
        "warmup_steps": args.warmup,
        "rounds": args.rounds,
        "sam_feature_cache_warm": sam_warm,
        "summary": summary,
        "entries": entries,
        "baseline_samples_per_sec": baseline_rate,
        "best_candidate": best,
        "best_samples_per_sec": summary[best]["samples_per_sec_mean"],
        "best_speedup_percent": summary[best]["speedup_percent_vs_B0"],
        "baseline_relative_spread": summary["B0_current"]["relative_spread"],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n[final] summary (mean over rounds):")
    for name, data in summary.items():
        print(
            f"  {name:38} {data['samples_per_sec_mean']:.3f} samples/s  "
            f"{data['speedup_percent_vs_B0']:+.2f}%  spread={data['relative_spread']*100:.2f}%  "
            f"gpu={data['gpu_utilization_mean']:.1f}  cpu={data['cpu_utilization_mean']:.1f}"
        )
    print(f"[final] wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
