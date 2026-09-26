#!/usr/bin/env python
"""Task 6C.5 section 18: value-preserving equivalence gate.

    python scripts/task6c5_equivalence.py --flags source_cache,preprocessed_cache

Runs the baseline (`B0`, every switch off) and one optimized flag set over the same
fixed samples, in the same order, from the same initial trainable state, with the same
seed, the same optimizer construction and strict deterministic mode. It then requires:

* prepared tensors identical (bit-exact, elementwise);
* per-step losses bit-identical;
* per-step gradient fingerprints identical;
* per-step post-step trainable-parameter fingerprints identical.

Anything that cannot meet bit identity (for example gradient checkpointing OFF) is
reported separately and is never merged into the bit-equivalent group.

Writes `evaluation/task6c5_equivalence.json`.
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

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.determinism import (  # noqa: E402
    gradients_fingerprint,
    trainable_state_fingerprint,
)
from buildreasonseg_mvp.pipeline import PipelineFlags, TrainingPipeline, warm_caches  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_seed  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6c5_equivalence.json"
IDS_JSON = EVAL / "task6c5_benchmark_ids.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"

FLAG_NAMES = (
    "source_cache",
    "preprocessed_cache",
    "pin_memory",
    "non_blocking",
    "prefetch_threads",
    "skip_grad_norm_instrumentation",
)


def parse_flags(text: str) -> PipelineFlags:
    flags = PipelineFlags()
    for token in [part.strip() for part in text.split(",") if part.strip()]:
        if token not in FLAG_NAMES:
            raise SystemExit(f"unknown flag {token!r}; expected a subset of {FLAG_NAMES}")
        if token == "prefetch_threads":
            flags.prefetch_threads = 2
        else:
            setattr(flags, token, True)
    return flags


def load_samples(limit: int) -> list:
    payload = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    train = {record["sample_id"]: record for record in data_mod.read_records("train")}
    return [data_mod.to_sample(train[sample_id]) for sample_id in payload["record_ids"][:limit]]


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


def prepared_tensor_check(runtime, samples, optimized: PipelineFlags) -> dict:
    """Elementwise comparison of the prepared tensors for every sample."""

    baseline_pipe = TrainingPipeline(runtime, PipelineFlags())
    optimized_pipe = TrainingPipeline(runtime, optimized)

    checked = 0
    mismatched: list[str] = []
    for sample in samples:
        base_batch, base_image, base_mask = baseline_pipe.prepare(sample)
        opt_batch, opt_image, opt_mask = optimized_pipe.prepare(sample)
        checked += 1
        if not torch.equal(base_batch.input_ids, opt_batch.input_ids):
            mismatched.append(f"{sample.sample_id}:input_ids")
        if not torch.equal(base_batch.attention_mask, opt_batch.attention_mask):
            mismatched.append(f"{sample.sample_id}:attention_mask")
        if not torch.equal(base_batch.labels, opt_batch.labels):
            mismatched.append(f"{sample.sample_id}:labels")
        if (base_batch.pixel_values is None) != (opt_batch.pixel_values is None):
            mismatched.append(f"{sample.sample_id}:pixel_values_none")
        elif base_batch.pixel_values is not None and not torch.equal(
            base_batch.pixel_values, opt_batch.pixel_values
        ):
            mismatched.append(f"{sample.sample_id}:pixel_values")
        if (base_batch.image_grid_thw is None) != (opt_batch.image_grid_thw is None):
            mismatched.append(f"{sample.sample_id}:image_grid_thw_none")
        elif base_batch.image_grid_thw is not None and not torch.equal(
            base_batch.image_grid_thw, opt_batch.image_grid_thw
        ):
            mismatched.append(f"{sample.sample_id}:image_grid_thw")
        for key in sorted(set(base_batch.extra_inputs) | set(opt_batch.extra_inputs)):
            left, right = base_batch.extra_inputs.get(key), opt_batch.extra_inputs.get(key)
            if torch.is_tensor(left) or torch.is_tensor(right):
                if left is None or right is None or not torch.equal(left, right):
                    mismatched.append(f"{sample.sample_id}:extra_inputs[{key}]")
        if (base_batch.prompt_length, base_batch.total_length, base_batch.seg_position, base_batch.visual_tokens) != (
            opt_batch.prompt_length,
            opt_batch.total_length,
            opt_batch.seg_position,
            opt_batch.visual_tokens,
        ):
            mismatched.append(f"{sample.sample_id}:metadata")
        if not (base_image == opt_image).all():
            mismatched.append(f"{sample.sample_id}:image")
        if not (base_mask == opt_mask).all():
            mismatched.append(f"{sample.sample_id}:target_mask")
    return {
        "samples_checked": checked,
        "mismatches": mismatched[:20],
        "mismatch_count": len(mismatched),
        "identical": not mismatched,
    }


def run_sequence(runtime, samples, flags: PipelineFlags, steps: int, initial: dict) -> dict:
    # Section 18 requires "same seed". The LoRA adapters carry dropout, so the forward
    # pass draws from the global RNG; without re-seeding, two runs in the same process
    # see different dropout masks and differ even when the pipeline is identical. That
    # is exactly what the control run detected.
    set_seed(int(runtime.cfg["seed"]))
    restore_trainable(runtime.model, initial)
    torch.cuda.reset_peak_memory_stats()
    pipeline = TrainingPipeline(runtime, flags)
    if flags.prefetch_threads > 0:
        warm_caches(pipeline, samples)
        pipeline.start_prefetch([samples[i % len(samples)] for i in range(steps)])
    optimizer = make_optimizer(runtime)
    losses: list[dict] = []
    grad_digests: list[str] = []
    param_digests: list[str] = []
    try:
        for index in range(steps):
            sample = samples[index % len(samples)]
            if flags.prefetch_threads > 0:
                prepared = pipeline.next_prepared(sample, index)
            else:
                prepared = pipeline.prepare(sample)
            result = pipeline.step(sample, prepared, optimizer)
            losses.append({key: repr(float(value)) for key, value in result["losses"].items()})
            grad_digests.append(
                gradients_fingerprint([p for p in runtime.model.parameters() if p.requires_grad])["sha256"]
            )
            param_digests.append(trainable_state_fingerprint(runtime.model)["sha256"])
            del result
    finally:
        pipeline.stop_prefetch()
    torch.cuda.synchronize()
    return {
        "steps": steps,
        "losses": losses,
        "gradient_digests": grad_digests,
        "parameter_digests": param_digests,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flags", default="source_cache,preprocessed_cache")
    parser.add_argument("--steps", type=int, default=12)
    # Supplementary flag sets (e.g. the rejected caches) may be gated without clobbering
    # the adopted pipeline's artifact, so a reviewer can see *why* a candidate was
    # rejected: on throughput, or on correctness.
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    optimized_flags = parse_flags(args.flags)
    samples = load_samples(max(args.steps, 16))
    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": "centre"}
    runtime = build_runtime(cfg, device="cuda", verbose=False)

    initial = snapshot_trainable(runtime.model)
    initial_fingerprint = trainable_state_fingerprint(runtime.model)["sha256"]

    tensor_check = prepared_tensor_check(runtime, samples[:16], optimized_flags)

    baseline = run_sequence(runtime, samples, PipelineFlags(), args.steps, initial)
    control = run_sequence(runtime, samples, PipelineFlags(), args.steps, initial)
    optimized = run_sequence(runtime, samples, optimized_flags, args.steps, initial)

    control_losses_identical = baseline["losses"] == control["losses"]
    control_gradients_identical = baseline["gradient_digests"] == control["gradient_digests"]
    control_parameters_identical = baseline["parameter_digests"] == control["parameter_digests"]
    run_to_run_reproducible = bool(
        control_losses_identical and control_gradients_identical and control_parameters_identical
    )

    losses_identical = baseline["losses"] == optimized["losses"]
    gradients_identical = baseline["gradient_digests"] == optimized["gradient_digests"]
    parameters_identical = baseline["parameter_digests"] == optimized["parameter_digests"]
    bit_equivalent = bool(
        tensor_check["identical"]
        and losses_identical
        and gradients_identical
        and parameters_identical
        and run_to_run_reproducible
    )

    first_difference = None
    if not losses_identical:
        for index, (left, right) in enumerate(zip(baseline["losses"], optimized["losses"])):
            if left != right:
                first_difference = {"step": index + 1, "baseline": left, "optimized": right}
                break

    report = {
        "_doc": (
            "Task 6C.5 section 18 value-preserving gate. Same clean initialization, same samples in the "
            "same order, same seed, same optimizer construction, strict deterministic mode."
        ),
        "task": "6C.5",
        "flag_set_label": args.label or args.flags,
        "optimized_flags": optimized_flags.as_dict(),
        "baseline_flags": PipelineFlags().as_dict(),
        "steps": args.steps,
        "initial_trainable_sha256": initial_fingerprint,
        "prepared_tensors": tensor_check,
        # Control: the baseline run twice. If the baseline does not reproduce itself,
        # a baseline-vs-optimized difference is run-to-run nondeterminism, not the flag.
        "baseline_run_to_run": {
            "losses_identical": control_losses_identical,
            "gradients_identical": control_gradients_identical,
            "post_step_parameters_identical": control_parameters_identical,
            "reproducible": run_to_run_reproducible,
        },
        "losses_identical": losses_identical,
        "gradients_identical": gradients_identical,
        "post_step_parameters_identical": parameters_identical,
        "bit_equivalent": bit_equivalent,
        "first_loss_difference": first_difference,
        "baseline": {
            "losses": baseline["losses"],
            "gradient_digests": baseline["gradient_digests"],
            "parameter_digests": baseline["parameter_digests"],
        },
        "optimized": {
            "losses": optimized["losses"],
            "gradient_digests": optimized["gradient_digests"],
            "parameter_digests": optimized["parameter_digests"],
        },
        "determinism": runtime.reports["determinism"],
        "peak_vram": {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        },
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[equivalence] prepared tensors identical: {tensor_check['identical']} "
          f"({tensor_check['mismatch_count']} mismatches over {tensor_check['samples_checked']} samples)")
    print(f"[equivalence] losses identical: {losses_identical}  gradients identical: {gradients_identical}  "
          f"post-step params identical: {parameters_identical}")
    print(f"[equivalence] BIT EQUIVALENT: {bit_equivalent}")
    print(f"[equivalence] wrote {out_path.relative_to(REPO_ROOT).as_posix()}")
    return 0 if bit_equivalent else 3


if __name__ == "__main__":
    raise SystemExit(main())
