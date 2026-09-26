#!/usr/bin/env python
"""Task 6C.6 sections 13/15/16: turn the measurements into explicit decisions.

    python scripts/task6c6_summarize.py

Reads every Task 6C.6 artifact, attaches to each candidate

* its equivalence category (`BIT_EQUIVALENT` / `NUMERICALLY_EQUIVALENT` / `NOT_EQUIVALENT`)
  from `evaluation/task6c6_equivalence.json`,
* its measured gain and whether that gain is resolvable above the local reference drift,
* a section 15 verdict (`adopted` / `rejected`) with a stated reason,

then writes the annotated variants files plus `evaluation/task6c6_resource_usage.json`
through `scripts/task6c6_benchmark.py --group resource`.

Two control observations are recorded explicitly, because they bound what this machine can
measure at all:

* `adamw_foreach` and `clip_foreach_true` are **code-identical to the reference** on this
  PyTorch build (`_default_to_fused_or_foreach` resolves to `foreach=True` for the 528 fp32
  parameters), and they measured -2.85% and -3.37%. That is the noise floor of a sequential
  group, not a result.
* every `torch.compile` candidate that ran on `aot_eager` changed the numerics out of
  tolerance, so none of them could be adopted even if it had been faster.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

EVAL = REPO_ROOT / "evaluation"
EQUIVALENCE = EVAL / "task6c6_equivalence.json"
COMPILE_JSON = EVAL / "task6c6_compile_variants.json"
OPTIMIZER_JSON = EVAL / "task6c6_optimizer_variants.json"
CHECKPOINTING_JSON = EVAL / "task6c6_checkpointing.json"
FINAL_JSON = EVAL / "task6c6_final_benchmark.json"
BASELINE_JSON = EVAL / "task6c6_integrated_baseline.json"

#: Section 15.
ADOPT_GAIN_PERCENT = 8.0
ADOPT_GAIN_WITH_SECONDARY_PERCENT = 5.0

#: Candidate name -> the flag set whose gate classifies it.
EQUIVALENCE_KEYS = {
    "O1_adamw_foreach": "adamw_foreach",
    "O2_adamw_fused": "adamw_fused",
    "O3_clip_foreach_true": "clip_foreach_true",
    "O4_clip_foreach_false": "clip_foreach_false",
    "C1a_qwen_aot_eager": "compile_qwen_aot_eager",
    "C1b_decoder_tail_aot_eager": "compile_decoder_tail_aot_eager",
    "C1c_combined_aot_eager": "compile_combined_aot_eager",
    "SDPA_math": "sdpa_math",
    "SDPA_mem_efficient": "sdpa_mem_efficient",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _equivalence_map() -> dict:
    payload = _load(EQUIVALENCE)
    return {name: entry for name, entry in (payload.get("candidates") or {}).items()}


def _annotate_entries(entries: list[dict], equivalence: dict, prefix_notes: dict | None = None) -> None:
    for entry in entries:
        name = entry["variant"]
        gate_name = EQUIVALENCE_KEYS.get(name)
        if gate_name and gate_name in equivalence:
            candidate = equivalence[gate_name]
            entry["equivalence_category"] = candidate.get("category")
            entry["equivalence_gate"] = gate_name
            entry["equivalence_detail"] = {
                key: candidate.get(key)
                for key in (
                    "loss_max_abs_difference",
                    "gradient_max_abs_difference",
                    "post_step_parameter_max_abs_difference",
                    "within_predeclared_tolerance",
                )
            }
        elif entry.get("variant", "").startswith(("uncompiled_reference", "O_reference", "B0.6", "CK_")):
            entry["equivalence_category"] = "REFERENCE"
        else:
            entry["equivalence_category"] = None

        rate = entry.get("samples_per_sec")
        gain = entry.get("speedup_percent_vs_interpolated_reference")
        resolvable = entry.get("gain_resolvable")
        category = entry.get("equivalence_category")
        status = entry.get("status")

        if status not in (None, "ok"):
            entry["adopted"] = False
            entry["rejection_reason"] = f"did not run ({status}): {str(entry.get('error'))[:200]}"
        elif category == "NOT_EQUIVALENT":
            entry["adopted"] = False
            entry["rejection_reason"] = (
                "not equivalent: the candidate changes losses/gradients out of the predeclared "
                "tolerance, so it cannot be adopted at any speed"
            )
        elif rate is None:
            entry["adopted"] = False
            entry["rejection_reason"] = "no throughput measurement"
        elif category == "REFERENCE":
            entry["adopted"] = False
            entry["rejection_reason"] = "reference run"
        elif gain is not None and not resolvable:
            entry["adopted"] = False
            entry["rejection_reason"] = (
                f"gain {gain:+.2f}% is inside the local reference uncertainty "
                f"({entry.get('local_reference_uncertainty_percent')}%) and below the section 15 "
                f"{ADOPT_GAIN_PERCENT}% gate"
            )
        elif gain is not None and gain < ADOPT_GAIN_PERCENT:
            entry["adopted"] = False
            entry["rejection_reason"] = (
                f"gain {gain:+.2f}% is below the section 15 {ADOPT_GAIN_PERCENT}% adoption gate"
            )
        else:
            entry["adopted"] = True
            entry["rejection_reason"] = None
        if prefix_notes and name in prefix_notes:
            entry["control_note"] = prefix_notes[name]


def summarize_compile(equivalence: dict) -> dict:
    payload = _load(COMPILE_JSON)
    if not payload:
        return {}
    _annotate_entries(payload.get("entries", []), equivalence)
    failures = [
        entry for entry in payload.get("entries", []) if entry.get("status") not in (None, "ok")
    ]
    payload["failure_summary"] = {
        "count": len(failures),
        "triton_missing": sum(1 for entry in failures if "TritonMissing" in str(entry.get("error"))),
        "other": [
            {"variant": entry["variant"], "error": str(entry.get("error"))[:200]}
            for entry in failures
            if "TritonMissing" not in str(entry.get("error"))
        ],
    }
    payload["torch_compile_available"] = not failures or any(
        entry.get("status") == "ok" for entry in payload.get("entries", [])
    )
    payload["decision"] = (
        "no torch.compile candidate is adopted: the inductor backend cannot run at all on this "
        "install (no Triton), and every backend that does run is either slower or numerically "
        "different"
    )
    COMPILE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def summarize_optimizer(equivalence: dict) -> dict:
    payload = _load(OPTIMIZER_JSON)
    if not payload:
        return {}
    notes = {
        "O1_adamw_foreach": (
            "code-identical control: torch already resolves AdamW to foreach=True for these "
            "parameters, so this variant runs the same kernel path as the reference. Its measured "
            "difference is measurement noise."
        ),
        "O3_clip_foreach_true": (
            "code-identical control: clip_grad_norm_ already defaults to foreach here, so this is "
            "the same execution path as the reference."
        ),
    }
    _annotate_entries(payload.get("entries", []), equivalence, notes)
    noise = [
        entry["speedup_percent_vs_interpolated_reference"]
        for entry in payload.get("entries", [])
        if entry["variant"] in notes and entry.get("speedup_percent_vs_interpolated_reference") is not None
    ]
    payload["no_op_control_noise_floor_percent"] = (
        round(max(abs(value) for value in noise), 3) if noise else None
    )
    payload["decision"] = (
        "no optimizer or clipping implementation is adopted: the default already uses the "
        "foreach multi-tensor path, fused=True is inside the noise band and numerically "
        "different, and foreach=False is slower"
    )
    OPTIMIZER_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def summarize_checkpointing() -> dict:
    payload = _load(CHECKPOINTING_JSON)
    if not payload:
        return {}
    change = payload.get("off_versus_on_percent")
    on_spread = (payload.get("checkpointing_on") or {}).get("relative_spread")
    off_spread = (payload.get("checkpointing_off") or {}).get("relative_spread")
    resolvable = bool(
        change is not None
        and on_spread is not None
        and off_spread is not None
        and abs(change) > 100.0 * max(on_spread, off_spread)
    )
    payload["change_resolvable"] = resolvable
    payload["adopted"] = False
    payload["decision"] = (
        "gradient checkpointing stays ON: "
        + (
            f"OFF measured {change:+.2f}%, which is inside the round-to-round spread "
            f"(ON {100 * on_spread:.2f}%, OFF {100 * off_spread:.2f}%)"
            if not resolvable
            else f"OFF measured {change:+.2f}% but section 10 requires a reproducible >=8% gain, "
            "<14 GiB reserved VRAM and acceptable equivalence"
        )
    )
    CHECKPOINTING_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def summarize_final() -> dict:
    payload = _load(FINAL_JSON)
    if not payload:
        return {}
    payload["verdict"] = (
        "OPTIMIZATION_PARTIAL"
        if (payload.get("adopted") or (payload.get("adopted_runtime") or {}).get(
            "integration_gain_percent_from_baseline_group"
        ))
        else "NO_MEANINGFUL_BOTTLENECK_FIX"
    )
    payload["what_was_adopted"] = {
        "runtime_change_in_this_task": "training.collect_grad_norms=false wired into the formal loop",
        "measured_integration_gain_percent": (payload.get("adopted_runtime") or {}).get(
            "integration_gain_percent_from_baseline_group"
        ),
        "new_candidates_adopted": [],
        "reason": (
            "the only runtime change this task adopts is the section 1 integration of the "
            "Task 6C.5 winner; no torch.compile, optimizer, clipping, SDPA or checkpointing "
            "candidate cleared the section 15 gate, and the compile family is additionally "
            "blocked by the missing Triton backend"
        ),
    }
    FINAL_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-resource", action="store_true")
    args = parser.parse_args(argv)

    equivalence = _equivalence_map()
    compile_payload = summarize_compile(equivalence)
    optimizer_payload = summarize_optimizer(equivalence)
    checkpointing_payload = summarize_checkpointing()
    final_payload = summarize_final()

    if not args.skip_resource:
        import task6c6_benchmark as benchmark

        resource = benchmark.resource_group()
        benchmark.write_json(benchmark.OUTPUTS["resource"], resource)
        print(
            f"[summarize] resource: RSS {resource['peak_process_rss_gib']} GiB, "
            f"reserved VRAM {resource['peak_reserved_vram_gib']} GiB, "
            f"max temp {resource['max_gpu_temperature_c']} C, OOM={resource['paging_or_oom']}"
        )

    print("[summarize] integration gate:", _load(EQUIVALENCE).get("integration_gate", {}).get("category"))
    if optimizer_payload:
        print(
            "[summarize] optimizer no-op noise floor:",
            optimizer_payload.get("no_op_control_noise_floor_percent"),
            "%",
        )
    if compile_payload:
        print("[summarize] compile failures:", json.dumps(compile_payload.get("failure_summary")))
    if checkpointing_payload:
        print(
            "[summarize] checkpointing OFF vs ON:",
            checkpointing_payload.get("off_versus_on_percent"),
            "resolvable:",
            checkpointing_payload.get("change_resolvable"),
        )
    if final_payload:
        print(
            "[summarize] final verdict:",
            final_payload.get("verdict"),
            "| challenger speedup:",
            final_payload.get("speedup_percent"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
