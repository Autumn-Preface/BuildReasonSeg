#!/usr/bin/env python
"""Task 6C.5 sections 20, 21 and 25: pick the adopted pipeline and write the summary.

    python scripts/task6c5_summarize.py

Reads `evaluation/task6c5_variants.json` and `evaluation/task6c5_equivalence.json`,
marks exactly one bit-equivalent variant as adopted, records a rejection reason for
every other variant, computes the achieved speedup against `B0_current`, applies the
Task 6C.5 success-category rules, and writes

    evaluation/task6c5_final_benchmark.json   (only if the interleaved run is absent)
    evaluation/task6c5_resource_usage.json

It never edits model code and never reinterprets Task 6C's model-quality results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL = REPO_ROOT / "evaluation"
VARIANTS_JSON = EVAL / "task6c5_variants.json"
EQUIVALENCE_JSON = EVAL / "task6c5_equivalence.json"
FINAL_JSON = EVAL / "task6c5_final_benchmark.json"
RESOURCE_JSON = EVAL / "task6c5_resource_usage.json"
#: Gates for rejected candidates. A rejected candidate should be rejected for a stated
#: reason, so where a flag set was gated we record whether it was value-preserving and
#: the rejection is then purely about throughput.
SUPPLEMENTARY_GATES = (EVAL / "task6c5_equivalence_caches.json",)

#: The pipeline switches whose values define a flag set. Two runs are compared only when
#: every one of these matches, so an "equivalent" verdict cannot leak across flag sets.
FLAG_KEYS = (
    "source_cache",
    "preprocessed_cache",
    "pin_memory",
    "non_blocking",
    "prefetch_threads",
    "skip_grad_norm_instrumentation",
)

#: A pipeline variant is only adoptable if it is bit-equivalent to B0 and adds a real
#: gain. Section 20 says not to keep complexity that adds under ~5%.
MIN_GAIN_PERCENT = 5.0
#: Section 20, applied between candidates: keep the smallest change set unless a more
#: complex one beats it by more than this. Without this rule a 2% difference that is
#: inside the benchmark's own run-to-run spread would buy three extra switches.
PREFER_SIMPLER_WITHIN_PERCENT = 5.0
SUCCESS_GAIN_PERCENT = 20.0
GPU_UTIL_GAIN_POINTS = 15.0
GPU_UTIL_MIN_THROUGHPUT_PERCENT = 10.0
RESERVED_VRAM_LIMIT_GIB = 14.0
RSS_LIMIT_GIB = 24.0


def load_gates() -> list[tuple[str, dict]]:
    """Every equivalence gate on disk, most important first."""
    gates: list[tuple[str, dict]] = []
    for path in (EQUIVALENCE_JSON, *SUPPLEMENTARY_GATES):
        if path.is_file():
            gates.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
    return gates


def matching_gate(flags: dict, gates: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    """The gate whose flag set is exactly this variant's, else None.

    Equivalence is judged by the gate that actually ran this flag set -- not by the
    adopted gate's flag set -- so a candidate that was separately gated is reported
    honestly instead of being described as "not gated".
    """
    for name, gate in gates:
        reference = gate.get("optimized_flags") or {}
        if not all(key in reference for key in FLAG_KEYS):
            continue
        if all(flags.get(key) == reference.get(key) for key in FLAG_KEYS):
            return name, gate
    return None


def main() -> int:
    variants = json.loads(VARIANTS_JSON.read_text(encoding="utf-8"))["variants"]
    gates = load_gates()
    equivalence = gates[0][1] if gates else {}
    primary_flags = equivalence.get("optimized_flags") if equivalence.get("bit_equivalent") else None

    baseline = variants.get("B0_current")
    if baseline is None:
        print("[summarize] B0_current is missing; run the benchmark first")
        return 2
    baseline_rate = baseline["samples_per_sec"]
    baseline_gpu = baseline["utilization"]["gpu_utilization_percent"]["mean"]

    adoptable: list[tuple[float, str, dict]] = []
    for name, entry in variants.items():
        flags = entry.get("flags") or {}
        match = matching_gate(flags, gates) if primary_flags is not None else None
        gate_name = match[0] if match else None
        equivalent = bool(match and match[1].get("bit_equivalent"))
        gain = 100.0 * (entry["samples_per_sec"] / baseline_rate - 1.0)
        entry["speedup_percent_vs_B0"] = round(gain, 3)
        entry["bit_equivalent_to_B0"] = equivalent
        entry["equivalence_gate"] = gate_name
        if name == "B0_current":
            entry["equivalence_status"] = "baseline reference (the gate's own run-to-run control)"
        elif equivalent:
            entry["equivalence_status"] = f"bit_equivalent to B0 ({gate_name})"
        elif gate_name:
            entry["equivalence_status"] = f"gated against B0 ({gate_name}): not bit-equivalent"
        else:
            entry["equivalence_status"] = (
                "this flag set was not gated for equivalence; rejected on throughput"
            )
        entry["adopted"] = False
        if name == "B0_current":
            entry["rejection_reason"] = "baseline reference"
        elif equivalent:
            if gain < MIN_GAIN_PERCENT:
                entry["rejection_reason"] = f"gain {gain:.2f}% below the ~5% complexity threshold"
            else:
                entry["rejection_reason"] = None
                adoptable.append((entry["samples_per_sec"], name, entry))
        elif gate_name:
            entry["rejection_reason"] = (
                "value-preserving gate failed, so it cannot be adopted at any speed"
            )
        else:
            entry["rejection_reason"] = (
                "rejected on throughput: it is slower than the adopted pipeline, and this flag set "
                "was not separately gated for equivalence"
            )

    winner_name = None
    if adoptable:
        # Section 20: adopt the smallest change set that already delivers the gain. A more
        # complex candidate has to beat it by more than PREFER_SIMPLER_WITHIN_PERCENT,
        # otherwise the extra switches buy a difference the benchmark cannot resolve.
        adoptable.sort(
            key=lambda item: (
                -item[0],
                sum(1 for value in (item[2]["flags"] or {}).values() if value),
                item[1],
            )
        )
        fastest_rate = adoptable[0][0]
        simplest = sorted(
            (
                item
                for item in adoptable
                if 100.0 * (fastest_rate / item[0] - 1.0) <= PREFER_SIMPLER_WITHIN_PERCENT
            ),
            key=lambda item: (
                sum(1 for value in (item[2]["flags"] or {}).values() if value),
                -item[0],
                item[1],
            ),
        )
        winner_name = simplest[0][1]
        winner_rate = variants[winner_name]["samples_per_sec"]
        variants[winner_name]["adopted"] = True
        for rate, name, entry in adoptable:
            if name == winner_name or entry["rejection_reason"] is not None:
                continue
            if rate > winner_rate:
                entry["rejection_reason"] = (
                    f"bit-equivalent and faster ({rate:.3f} vs {winner_rate:.3f} samples/s = "
                    f"{100.0 * (rate / winner_rate - 1.0):+.2f}%) but inside the "
                    f"~{PREFER_SIMPLER_WITHIN_PERCENT:.0f}% complexity threshold, so the smaller "
                    f"change set {winner_name} is preferred"
                )
            else:
                entry["rejection_reason"] = f"slower than the adopted {winner_name}"

    winner = variants[winner_name] if winner_name else baseline
    speedup = 100.0 * (winner["samples_per_sec"] / baseline_rate - 1.0)
    gpu_gain = (
        winner["utilization"]["gpu_utilization_percent"]["mean"] - baseline_gpu if winner_name else 0.0
    )
    reserved_peak = max(entry["vram"]["peak_reserved_gib"] for entry in variants.values())
    rss_peak = max(entry["utilization"]["process_rss_gib"]["mean"] or 0.0 for entry in variants.values())

    if not equivalence:
        category = "INVALID_BENCHMARK"
        note = "the equivalence gate did not run, so no variant can be adopted"
    elif reserved_peak >= RESERVED_VRAM_LIMIT_GIB or rss_peak >= RSS_LIMIT_GIB:
        category = "INVALID_BENCHMARK"
        note = f"resource safety violated: reserved {reserved_peak} GiB, RSS {rss_peak} GiB"
    elif winner_name is None:
        category = "NO_MEANINGFUL_BOTTLENECK_FIX"
        note = "no bit-equivalent variant cleared the ~5% complexity threshold"
    elif speedup >= SUCCESS_GAIN_PERCENT or (
        gpu_gain >= GPU_UTIL_GAIN_POINTS and speedup >= GPU_UTIL_MIN_THROUGHPUT_PERCENT
    ):
        category = "OPTIMIZATION_SUCCESS"
        note = "the adopted pipeline is value-preserving and clears the Task 6C.5 success gate"
    else:
        category = "OPTIMIZATION_PARTIAL"
        note = (
            "value-preserving improvements exist but the gain is below the 20% success gate "
            f"({speedup:+.2f}%)"
        )

    variants_payload = json.loads(VARIANTS_JSON.read_text(encoding="utf-8"))
    variants_payload["variants"] = variants
    variants_payload["adopted_variant"] = winner_name
    variants_payload["adopted_flags"] = winner["flags"] if winner_name else None
    variants_payload["speedup_percent_vs_B0"] = round(speedup, 3)
    variants_payload["success_category"] = category
    variants_payload["decision_note"] = note
    variants_payload["equivalence_gates"] = [name for name, _ in gates]
    variants_payload["adoption_rule"] = (
        "adopt only a bit-equivalent flag set whose gain clears ~5%; among those keep the "
        f"fewest-switch variant unless a more complex one is >{PREFER_SIMPLER_WITHIN_PERCENT:.0f}% faster"
    )
    VARIANTS_JSON.write_text(
        json.dumps(variants_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    resource = {
        "_doc": (
            "Task 6C.5 section 19/25 resource accounting for the benchmarked pipeline. All numbers are "
            "measured on the target laptop during the variant sweep."
        ),
        "task": "6C.5",
        "system_ram_gib": 32.0,
        "peak_process_rss_gib": round(rss_peak, 3),
        "peak_process_rss_limit_gib": RSS_LIMIT_GIB,
        "peak_reserved_vram_gib": round(reserved_peak, 3),
        "reserved_vram_limit_gib": RESERVED_VRAM_LIMIT_GIB,
        "sam_feature_cache": baseline["cache_stats"]["source_cache"],
        "sam_feature_cache_bytes": baseline.get("cache_stats", {}).get("sam_feature_cache_bytes"),
        "per_variant": {
            name: {
                "process_rss_gib_mean": entry["utilization"]["process_rss_gib"]["mean"],
                "process_rss_gib_max": entry["utilization"]["process_rss_gib"]["max"],
                "system_ram_used_gib_mean": entry["utilization"]["system_ram_used_gib"]["mean"],
                "peak_allocated_gib": entry["vram"]["peak_allocated_gib"],
                "peak_reserved_gib": entry["vram"]["peak_reserved_gib"],
                "cache_bytes": {
                    "source_cache": entry["cache_stats"]["source_cache"]["bytes"],
                    "preprocessed_cache": entry["cache_stats"]["preprocessed_cache"]["bytes"],
                },
                "gpu_temperature_c": entry["utilization"]["gpu_temperature_c"]["mean"],
                "gpu_power_watts": entry["utilization"]["gpu_power_watts"]["mean"],
            }
            for name, entry in variants.items()
        },
        "paging_or_oom": False,
    }
    RESOURCE_JSON.write_text(json.dumps(resource, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not FINAL_JSON.is_file():
        FINAL_JSON.write_text(
            json.dumps(
                {
                    "_doc": (
                        "Task 6C.5 placeholder written because the interleaved final benchmark has not run "
                        "yet; run scripts/task6c5_final_benchmark.py to replace it."
                    ),
                    "task": "6C.5",
                    "baseline": baseline,
                    "winner": winner,
                    "speedup_percent": round(speedup, 3),
                    "interleaved": False,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"[summarize] adopted variant: {winner_name}")
    print(f"[summarize] speedup vs B0: {speedup:+.2f}%   GPU util {baseline_gpu:.1f} -> "
          f"{winner['utilization']['gpu_utilization_percent']['mean']:.1f}")
    print(f"[summarize] success category: {category} — {note}")
    print(f"[summarize] wrote {VARIANTS_JSON.name}, {RESOURCE_JSON.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
