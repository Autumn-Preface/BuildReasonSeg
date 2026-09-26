#!/usr/bin/env python
"""Task 6C.5 refactor control: prove the audit's code restructuring is value-neutral.

    python scripts/task6c5_refactor_control.py [--steps 4] [--output evaluation/task6c5_refactor_control.json]

Task 6C.5 touched `MvpRuntime.train_step` in three ways that are *not* the adopted
optimization:

1. it wrapped the existing stages in optional stage timers (`contextlib.nullcontext()`
   when no profiler is attached);
2. it split one `torch.autocast` region into two -- `qwen_forward` and `loss`;
3. it made the per-parameter gradient-norm sweep optional.

The equivalence gate proves (3) is value-preserving for the adopted flag, but its
baseline is *this* code with default flags, so it cannot by itself rule out (1) and (2).
This control does that: it reconstructs the pre-Task-6C.5 step body verbatim -- one
autocast region around forward + supervision + loss, unconditional gradient norms --
and compares it step by step, with the same initialization, the same samples in the
same order, the same seed re-applied per run and strict deterministic mode, against the
current default `train_step`.

Losses, gradient fingerprints and post-step parameter fingerprints must be identical.
If they were not, the audit would have changed training semantics and every Task 6C.5
number would be invalid.

This is a correctness control, not a benchmark: it measures nothing about throughput.
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

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.determinism import (  # noqa: E402
    gradients_fingerprint,
    trainable_state_fingerprint,
)
from buildreasonseg_mvp.losses import combined_loss  # noqa: E402
from buildreasonseg_mvp.pipeline import PipelineFlags, TrainingPipeline  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_seed  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6c5_refactor_control.json"
IDS_JSON = EVAL / "task6c5_benchmark_ids.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"


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


def previous_step(runtime, batch, gt_mask, features, optimizer) -> dict:
    """The pre-Task-6C.5 `train_step` body, reconstructed verbatim.

    Compare against `git show HEAD:buildreasonseg_mvp/runtime.py`: one autocast region
    wrapping the forward, the mask supervision and the combined loss; zero_grad;
    backward; unconditional `gradient_norms`; clip; step. No timers, no flags.
    """

    from buildreasonseg_mvp.runtime import gradient_norms  # same helper `train_step` uses

    training_cfg = runtime.cfg["training"]
    use_autocast = bool(training_cfg.get("bf16_autocast", True))

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
        output = runtime.model(batch, features)
        target, supervision_logits, supervision_mode = runtime.build_mask_supervision(
            gt_mask, output.mask_logits
        )
        breakdown = combined_loss(
            output.lm_logits, batch.labels, supervision_logits, target, runtime.loss_weights
        )

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    breakdown.total.backward()

    grad_norms = gradient_norms(runtime.model)
    clipped = None
    if optimizer is not None:
        clipped = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in runtime.model.parameters() if p.requires_grad],
                float(runtime.cfg["optimizer"]["grad_clip_norm"]),
            )
        )
        optimizer.step()

    return {
        "losses": breakdown.as_dict(),
        "grad_norms": grad_norms,
        "grad_clip": clipped,
        "supervision_mode": supervision_mode,
        "mask_logits_shape": tuple(output.mask_logits.shape),
    }


def run_sequence(runtime, samples, steps: int, initial: dict, mode: str) -> dict:
    # Same reason as the equivalence gate: LoRA dropout draws from the global RNG, so a
    # per-run re-seed is required for any same-process comparison to mean anything.
    set_seed(int(runtime.cfg["seed"]))
    restore_trainable(runtime.model, initial)
    pipeline = TrainingPipeline(runtime, PipelineFlags())
    optimizer = make_optimizer(runtime)
    losses: list[dict] = []
    grad_digests: list[str] = []
    param_digests: list[str] = []
    for index in range(steps):
        sample = samples[index % len(samples)]
        batch, image, gt_mask = pipeline.prepare(sample)
        moved = batch.to(runtime.device)
        features, _cached = runtime.features_for(sample, image)
        if mode == "previous":
            result = previous_step(runtime, moved, gt_mask, features, optimizer)
        else:
            result = runtime.train_step(moved, gt_mask, features, optimizer=optimizer)
        losses.append({key: repr(float(value)) for key, value in result["losses"].items()})
        grad_digests.append(
            gradients_fingerprint([p for p in runtime.model.parameters() if p.requires_grad])["sha256"]
        )
        param_digests.append(trainable_state_fingerprint(runtime.model)["sha256"])
        del result
    torch.cuda.synchronize()
    return {"losses": losses, "gradient_digests": grad_digests, "parameter_digests": param_digests}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    payload = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    train = {record["sample_id"]: record for record in data_mod.read_records("train")}
    samples = [data_mod.to_sample(train[sid]) for sid in payload["record_ids"][: args.steps]]

    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": "centre"}
    runtime = build_runtime(cfg, device="cuda", verbose=False)

    initial = snapshot_trainable(runtime.model)
    previous = run_sequence(runtime, samples, args.steps, initial, "previous")
    current = run_sequence(runtime, samples, args.steps, initial, "current")

    losses_identical = previous["losses"] == current["losses"]
    gradients_identical = previous["gradient_digests"] == current["gradient_digests"]
    parameters_identical = previous["parameter_digests"] == current["parameter_digests"]
    refactor_neutral = bool(losses_identical and gradients_identical and parameters_identical)

    report = {
        "_doc": (
            "Task 6C.5 refactor control. Reconstructs the pre-Task-6C.5 train_step body "
            "(one autocast region, unconditional gradient norms, no stage timers) and compares "
            "it against the current default path. Losses, gradient fingerprints and post-step "
            "parameter fingerprints must be identical, otherwise the audit itself changed "
            "training semantics. Correctness control only; it measures no throughput."
        ),
        "task": "6C.5",
        "steps": args.steps,
        "samples": [sample.sample_id for sample in samples],
        "initial_trainable_sha256": trainable_state_fingerprint(runtime.model)["sha256"],
        "previous_code_mode": "one autocast region + unconditional gradient norms (HEAD body)",
        "current_code_mode": "split autocast regions + optional timers + collect_grad_norms=True",
        "losses_identical": losses_identical,
        "gradients_identical": gradients_identical,
        "post_step_parameters_identical": parameters_identical,
        "refactor_value_neutral": refactor_neutral,
        "previous": previous,
        "current": current,
        "determinism": runtime.reports["determinism"],
        "initial_trainable_state_sha256_note": (
            "fingerprint of the restored initial state, identical for both runs"
        ),
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[refactor-control] steps: {args.steps}")
    print(f"[refactor-control] losses identical: {losses_identical}  gradients identical: {gradients_identical}  "
          f"post-step params identical: {parameters_identical}")
    print(f"[refactor-control] REFACTOR VALUE-NEUTRAL: {refactor_neutral}")
    print(f"[refactor-control] wrote {out_path.relative_to(REPO_ROOT).as_posix()}")
    return 0 if refactor_neutral else 3


if __name__ == "__main__":
    raise SystemExit(main())
