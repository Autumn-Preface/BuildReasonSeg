#!/usr/bin/env python
"""Task 6C.6 sections 12 and 13: equivalence gates and candidate classification.

    python scripts/task6c6_equivalence.py [--steps 12] [--candidates optimizer,compile]

Writes `evaluation/task6c6_equivalence.json` with three parts.

1. **Integration gate (section 12).** The formal path with
   `training.collect_grad_norms=false` must be bit-identical to the
   `collect_grad_norms=true` path: identical prepared tensors, identical 12-step losses,
   identical gradient fingerprints, identical post-step parameter fingerprints, with the
   seed re-applied per run (the Task 6C.5 finding: LoRA dropout consumes the global RNG,
   so "same seed" has to mean per run).

2. **Candidate classification (section 13).** Each candidate is classified as
   `BIT_EQUIVALENT`, `NUMERICALLY_EQUIVALENT` or `NOT_EQUIVALENT` against the same
   reference sequence, with the maximum absolute and relative loss difference, the
   gradient difference and the post-step parameter difference recorded. Tolerances are
   predeclared below and were fixed before the runs.

3. **Control.** The reference sequence is run twice. If the reference does not reproduce
   itself, every comparison in this file is meaningless, so the control result is stored
   next to them.

This script never edits model code and never reinterprets a model-quality result.
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
from task6c6_common import (  # noqa: E402
    EVAL,
    VariantSpec,
    apply_compile,
    build_variant_runtime,
    collect_grad_norms_setting,
    load_benchmark_samples,
    make_optimizer,
    restore_trainable,
    set_seed,
    snapshot_trainable,
    write_json,
)

OUT = EVAL / "task6c6_equivalence.json"
IDS_JSON = EVAL / "task6c5_benchmark_ids.json"

#: Tolerances for `NUMERICALLY_EQUIVALENT`, fixed before the runs (section 13).
#: A candidate must satisfy every one of them, and must not break strict determinism.
TOLERANCES = {
    "loss_max_abs": 1e-3,
    "loss_max_rel": 1e-4,
    "gradient_max_abs": 1e-5,
    "parameter_max_abs": 1e-6,
    "parameter_max_rel": 1e-6,
}


def _flatten(prefix: str, value, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)


def run_sequence(runtime, samples, steps: int, initial: dict, spec: VariantSpec) -> dict:
    """One fixed mini-run on the formal step, with per-run re-seeding."""

    set_seed(int(runtime.cfg["seed"]))
    restore_trainable(runtime.model, initial)
    optimizer = make_optimizer(runtime, kind=spec.optimizer_kind)
    losses: list[dict] = []
    grad_digests: list[str] = []
    param_digests: list[str] = []
    grad_vectors: list[dict[str, float]] = []
    param_vectors: list[dict[str, float]] = []
    for index in range(steps):
        sample = samples[index % len(samples)]
        batch, image = runtime.prepare(sample)
        features, _cached = runtime.features_for(sample, image)
        result = runtime.train_step(
            batch,
            sample.target_mask(),
            features,
            optimizer=optimizer,
            collect_grad_norms=collect_grad_norms_setting(runtime),
            clip_grad_foreach=spec.clip_grad_foreach,
        )
        losses.append({key: repr(float(value)) for key, value in result["losses"].items()})
        grad_digests.append(
            gradients_fingerprint([p for p in runtime.model.parameters() if p.requires_grad])["sha256"]
        )
        param_digests.append(trainable_state_fingerprint(runtime.model)["sha256"])
        if index == steps - 1:
            grad_vectors.append(
                {
                    name: float(parameter.grad.detach().float().abs().max())
                    for name, parameter in runtime.model.named_parameters()
                    if parameter.requires_grad and parameter.grad is not None
                }
            )
            param_vectors.append(
                {
                    name: parameter.detach().float().flatten()[:64].clone()
                    for name, parameter in runtime.model.named_parameters()
                    if parameter.requires_grad
                }
            )
        del result, features, batch
    torch.cuda.synchronize()
    return {
        "losses": losses,
        "gradient_digests": grad_digests,
        "parameter_digests": param_digests,
        "grad_absmax": grad_vectors[0] if grad_vectors else {},
        "parameter_sample": param_vectors[0] if param_vectors else {},
    }


def prepared_tensor_check(runtime, samples, spec: VariantSpec, reference: dict) -> dict:
    """Section 12: identical prepared tensors, elementwise."""

    mismatches: list[str] = []
    checked = 0
    for sample in samples:
        batch, image = runtime.prepare(sample)
        reference_batch = reference.get(sample.sample_id)
        if reference_batch is None:
            mismatches.append(f"{sample.sample_id}:missing_reference")
            continue
        checked += 1
        for attribute in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"):
            left = getattr(reference_batch["batch"], attribute, None)
            right = getattr(batch, attribute, None)
            if (left is None) != (right is None):
                mismatches.append(f"{sample.sample_id}:{attribute}_none")
            elif left is not None and not torch.equal(left, right):
                mismatches.append(f"{sample.sample_id}:{attribute}")
        for key in sorted(set(reference_batch["batch"].extra_inputs) | set(batch.extra_inputs)):
            left = reference_batch["batch"].extra_inputs.get(key)
            right = batch.extra_inputs.get(key)
            if torch.is_tensor(left) or torch.is_tensor(right):
                if left is None or right is None or not torch.equal(left, right):
                    mismatches.append(f"{sample.sample_id}:extra_inputs[{key}]")
        if not (reference_batch["image"] == image).all():
            mismatches.append(f"{sample.sample_id}:image")
        if not (reference_batch["mask"] == sample.target_mask()).all():
            mismatches.append(f"{sample.sample_id}:target_mask")
    return {
        "samples_checked": checked,
        "mismatches": mismatches[:20],
        "mismatch_count": len(mismatches),
        "identical": not mismatches,
    }


def compare(reference: dict, candidate: dict) -> dict:
    """Loss / gradient / parameter differences between two sequences."""

    loss_max_abs = 0.0
    loss_max_rel = 0.0
    first_difference = None
    for index, (left, right) in enumerate(zip(reference["losses"], candidate["losses"])):
        if left != right and first_difference is None:
            first_difference = {"step": index + 1, "reference": left, "candidate": right}
        for key in sorted(set(left) | set(right)):
            a = float(left.get(key, "nan"))
            b = float(right.get(key, "nan"))
            difference = abs(a - b)
            loss_max_abs = max(loss_max_abs, difference)
            if a:
                loss_max_rel = max(loss_max_rel, difference / abs(a))

    gradient_max_abs = 0.0
    for name, value in reference["grad_absmax"].items():
        other = candidate["grad_absmax"].get(name)
        if other is not None:
            gradient_max_abs = max(gradient_max_abs, abs(value - other))

    parameter_max_abs = 0.0
    parameter_max_rel = 0.0
    for name, tensor in reference["parameter_sample"].items():
        other = candidate["parameter_sample"].get(name)
        if other is None:
            continue
        difference = (tensor - other).abs()
        parameter_max_abs = max(parameter_max_abs, float(difference.max()))
        denominator = tensor.abs().clamp_min(1e-12)
        parameter_max_rel = max(parameter_max_rel, float((difference / denominator).max()))

    losses_identical = reference["losses"] == candidate["losses"]
    gradients_identical = reference["gradient_digests"] == candidate["gradient_digests"]
    parameters_identical = reference["parameter_digests"] == candidate["parameter_digests"]
    bit_equivalent = bool(losses_identical and gradients_identical and parameters_identical)
    within_tolerance = bool(
        loss_max_abs <= TOLERANCES["loss_max_abs"]
        and loss_max_rel <= TOLERANCES["loss_max_rel"]
        and gradient_max_abs <= TOLERANCES["gradient_max_abs"]
        and parameter_max_abs <= TOLERANCES["parameter_max_abs"]
        and parameter_max_rel <= TOLERANCES["parameter_max_rel"]
    )
    if bit_equivalent:
        category = "BIT_EQUIVALENT"
    elif within_tolerance:
        category = "NUMERICALLY_EQUIVALENT"
    else:
        category = "NOT_EQUIVALENT"
    return {
        "losses_identical": losses_identical,
        "gradients_identical": gradients_identical,
        "post_step_parameters_identical": parameters_identical,
        "bit_equivalent": bit_equivalent,
        "loss_max_abs_difference": loss_max_abs,
        "loss_max_rel_difference": loss_max_rel,
        "gradient_max_abs_difference": gradient_max_abs,
        "post_step_parameter_max_abs_difference": parameter_max_abs,
        "post_step_parameter_max_rel_difference": parameter_max_rel,
        "within_predeclared_tolerance": within_tolerance,
        "category": category,
        "first_loss_difference": first_difference,
    }


def candidate_specs(names: list[str]) -> list[VariantSpec]:
    catalogue = {
        "adamw_foreach": VariantSpec(name="adamw_foreach", optimizer_kind="foreach"),
        "adamw_fused": VariantSpec(name="adamw_fused", optimizer_kind="fused"),
        "clip_foreach_true": VariantSpec(name="clip_foreach_true", clip_grad_foreach=True),
        "clip_foreach_false": VariantSpec(name="clip_foreach_false", clip_grad_foreach=False),
        "det_algorithms_off": VariantSpec(
            name="det_algorithms_off",
            description="Strict determinism disabled (measured, not adopted).",
            deterministic_strict=False,
        ),
        "sdpa_math": VariantSpec(name="sdpa_math", sdpa_backend="math"),
        "sdpa_mem_efficient": VariantSpec(name="sdpa_mem_efficient", sdpa_backend="mem_efficient"),
    }
    return [catalogue[name] for name in names if name in catalogue]


def compile_candidate_specs() -> list[dict]:
    """Compile candidates are gated through `apply_compile`, which changes the module.

    Only the backends that can run on this install are gated: inductor cannot start
    without Triton, and the graph-replay backends (`eager`, `cudagraphs`) fail as soon
    as the sequence length changes.
    """

    return [
        {"name": "compile_qwen_aot_eager", "scope": "qwen", "backend": "aot_eager"},
        {"name": "compile_decoder_tail_aot_eager", "scope": "decoder_tail", "backend": "aot_eager"},
        {"name": "compile_combined_aot_eager", "scope": "combined", "backend": "aot_eager"},
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--candidates",
        default="adamw_foreach,adamw_fused,clip_foreach_true,det_algorithms_off",
        help="comma-separated candidate names, or 'none'",
    )
    parser.add_argument("--include-compile", action="store_true")
    args = parser.parse_args(argv)

    samples = load_benchmark_samples(limit=max(args.steps, 8))
    reference_runtime = build_variant_runtime(collect_grad_norms=True)

    # --- section 12: the integration gate -------------------------------------
    initial = snapshot_trainable(reference_runtime.model)
    initial_fingerprint = trainable_state_fingerprint(reference_runtime.model)["sha256"]

    reference_prepared: dict = {}
    prepared_runtime = reference_runtime
    for sample in samples:
        batch, image = prepared_runtime.prepare(sample)
        reference_prepared[sample.sample_id] = {"batch": batch, "image": image, "mask": sample.target_mask()}

    true_spec = VariantSpec(name="collect_grad_norms_true")
    false_spec = VariantSpec(name="collect_grad_norms_false", collect_grad_norms=False)

    reference_sequence = run_sequence(reference_runtime, samples, args.steps, initial, true_spec)
    control_sequence = run_sequence(reference_runtime, samples, args.steps, initial, true_spec)

    integration_runtime = build_variant_runtime(collect_grad_norms=False)
    integration_prepared = prepared_tensor_check(
        integration_runtime, samples, false_spec, reference_prepared
    )
    integrated_sequence = run_sequence(integration_runtime, samples, args.steps, initial, false_spec)
    integration_comparison = compare(reference_sequence, integrated_sequence)

    report: dict = {
        "_doc": (
            "Task 6C.6 sections 12/13. Integration gate for "
            "training.collect_grad_norms=false, plus candidate equivalence classification. "
            "Same initial trainable state, same samples in the same order, same optimizer "
            "construction, strict determinism, and the seed re-applied to every run."
        ),
        "task": "6C.6",
        "steps": args.steps,
        "tolerances": TOLERANCES,
        "initial_trainable_sha256": initial_fingerprint,
        "reference_run_to_run_control": {
            "losses_identical": reference_sequence["losses"] == control_sequence["losses"],
            "gradients_identical": reference_sequence["gradient_digests"] == control_sequence["gradient_digests"],
            "post_step_parameters_identical": (
                reference_sequence["parameter_digests"] == control_sequence["parameter_digests"]
            ),
        },
        "integration_gate": {
            "reference_flags": {"collect_grad_norms": True},
            "candidate_flags": {"collect_grad_norms": False},
            "prepared_tensors": integration_prepared,
            **integration_comparison,
        },
        "reference_losses": reference_sequence["losses"],
        "integrated_losses": integrated_sequence["losses"],
        "candidates": {},
    }
    del integration_runtime
    torch.cuda.empty_cache()

    # --- section 13: candidate classification ---------------------------------
    names = [] if args.candidates.strip().lower() == "none" else [
        token.strip() for token in args.candidates.split(",") if token.strip()
    ]
    for spec in candidate_specs(names):
        runtime = build_variant_runtime(deterministic_strict=spec.deterministic_strict)
        try:
            sequence = run_sequence(runtime, samples, args.steps, initial, spec)
            comparison = compare(reference_sequence, sequence)
            report["candidates"][spec.name] = {
                "spec": spec.as_dict(),
                "strict_effective": runtime.reports["determinism"].get("strict_effective"),
                **comparison,
            }
        except Exception as exc:  # noqa: BLE001
            report["candidates"][spec.name] = {
                "spec": spec.as_dict(),
                "category": "NOT_EQUIVALENT",
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
        finally:
            del runtime
            torch.cuda.empty_cache()

    if args.include_compile:
        for candidate in compile_candidate_specs():
            name = candidate["name"]
            runtime = build_variant_runtime()
            info = apply_compile(runtime, scope=candidate["scope"], backend=candidate["backend"])
            if not info.get("compiled"):
                report["candidates"][name] = {
                    "spec": candidate,
                    "category": "NOT_EVALUATED",
                    "compile": info,
                }
                del runtime
                torch.cuda.empty_cache()
                continue
            try:
                sequence = run_sequence(runtime, samples, args.steps, initial, VariantSpec(name=name))
                comparison = compare(reference_sequence, sequence)
                report["candidates"][name] = {
                    "spec": candidate,
                    "compile": info,
                    "strict_effective": runtime.reports["determinism"].get("strict_effective"),
                    **comparison,
                }
            except Exception as exc:  # noqa: BLE001
                report["candidates"][name] = {
                    "spec": candidate,
                    "compile": info,
                    "category": "NOT_EQUIVALENT",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            finally:
                del runtime
                torch.cuda.empty_cache()

    del reference_runtime
    torch.cuda.empty_cache()
    write_json(OUT, report)

    gate = report["integration_gate"]
    print(
        f"[task6c6:equivalence] integration prepared tensors identical: "
        f"{gate['prepared_tensors']['identical']} "
        f"({gate['prepared_tensors']['mismatch_count']} mismatches / "
        f"{gate['prepared_tensors']['samples_checked']}), losses identical: {gate['losses_identical']}, "
        f"gradients identical: {gate['gradients_identical']}, params identical: "
        f"{gate['post_step_parameters_identical']}, category: {gate['category']}",
        flush=True,
    )
    for name, payload in report["candidates"].items():
        print(
            f"[task6c6:equivalence] candidate {name}: {payload.get('category')} "
            f"loss_max_abs={payload.get('loss_max_abs_difference')} "
            f"params_max_abs={payload.get('post_step_parameter_max_abs_difference')}",
            flush=True,
        )
    print(f"[task6c6:equivalence] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
