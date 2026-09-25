#!/usr/bin/env python
"""Task 6A Stage 1: a real 2-sample forward/backward pass and one optimizer step.

    python scripts/task6a_smoke2.py [--config ...] [--no-optimizer-step]

Uses the fixed same-image / different-target smoke pair and performs a genuine
forward/backward through

    image + instruction -> Qwen teacher-forced reasoning + [SEG]
                        -> [SEG] hidden -> projection -> SAM2 mask decoder
                        -> LM CE + BCE + Dice

Success criteria (Task 6A section 15) are all asserted, not merely printed:

* every loss finite;
* projected embedding shape equals the SAM prompt-embedding dimension;
* predicted mask logits finite and non-constant;
* non-zero finite gradients on LoRA, `[SEG]`, projection and the SAM2 mask decoder;
* **zero** trainable parameters in the Qwen visual tower;
* the frozen SAM2 image encoder receives no gradient;
* one optimizer step changes the `[SEG]` token state and leaves ordinary
  vocabulary rows untouched;
* no out-of-memory.

Merges its section into `evaluation/task6a_smoke_report.json`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.reporting import read_subset_ids, update_report  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, reset_peak, timed, vram  # noqa: E402

VISUAL_MARKERS = ("visual", "vision_tower", "vision_model")


def visual_tower_trainables(model) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(marker in name for marker in VISUAL_MARKERS)
    ]


def snapshot_embeddings(model, sample_ids: list[int]) -> dict[int, torch.Tensor]:
    """Read rows from the frozen base embedding table for the regression check."""

    embedding = model.qwen.get_input_embeddings()
    base_layer = getattr(getattr(embedding, "token_adapter", None), "base_layer", embedding)
    weight = base_layer.weight
    return {token_id: weight[token_id].detach().float().clone() for token_id in sample_ids}


def seg_row(model) -> torch.Tensor:
    """Current `[SEG]` representation, wherever the active mechanism stores it."""

    embedding = model.qwen.get_input_embeddings()
    adapter = getattr(embedding, "token_adapter", None)
    if adapter is not None and hasattr(adapter, "trainable_tokens_delta"):
        deltas = adapter.trainable_tokens_delta
        original = adapter.trainable_tokens_original[
            next(iter(deltas.keys()))
        ]
        return (original.float() + deltas[next(iter(deltas.keys()))].float()).detach().clone()
    if model.token_holder is not None:
        return model.token_holder.row.detach().float().clone()
    raise RuntimeError("no trainable [SEG] representation found")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-optimizer-step", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    subsets = read_subset_ids()
    pair_ids = subsets["smoke_pair"]["sample_ids"]
    records = {r["sample_id"]: r for r in data_mod.read_records("train")}
    samples = [data_mod.to_sample(records[sample_id]) for sample_id in pair_ids]

    started = time.time()
    reset_peak()
    runtime = build_runtime(cfg, device="cuda", verbose=True)
    optimizer = runtime.build_optimizer()

    visual_before = visual_tower_trainables(runtime.model)
    watch_tokens = [0, 1, 100, 1000, 50000, 151000]
    rows_before = snapshot_embeddings(runtime.model, watch_tokens)
    seg_before = seg_row(runtime.model)

    results = []
    for index, sample in enumerate(samples, start=1):
        batch, image = runtime.prepare(sample)
        gt_mask = sample.target_mask()
        features, cached = runtime.features_for(sample, image)

        do_step = (not args.no_optimizer_step) and index == len(samples)
        reset_peak()
        with timed(f"stage1_forward_backward_sample{index}"):
            step = runtime.train_step(batch, gt_mask, features, optimizer=optimizer if do_step else None)

        output = step["output"]
        logits = output.mask_logits.detach().float()
        results.append(
            {
                "sample_id": sample.sample_id,
                "query_type": sample.query_type,
                "level": sample.level,
                "target_component_id": sample.target_component_id,
                "losses": step["losses"],
                "grad_norms": step["grad_norms"],
                "grad_clip_total_norm": step["grad_clip_total_norm"],
                "optimizer_step_applied": bool(do_step),
                "sam_feature_cache_hit": bool(cached),
                "shapes": output.as_dict(),
                "mask_logits": {
                    "finite": bool(torch.isfinite(logits).all()),
                    "std": float(logits.std()),
                    "min": float(logits.min()),
                    "max": float(logits.max()),
                    "constant": bool(float(logits.std()) < 1e-9),
                    "positive_fraction": float((logits > 0).float().mean()),
                },
                "projection_matches_sam_dim": int(output.projected.shape[-1])
                == int(runtime.reports["sam_prompt_embed_dim"]),
                "vram": vram(),
            }
        )
        del step, output, logits

    visual_after = visual_tower_trainables(runtime.model)
    rows_after = snapshot_embeddings(runtime.model, watch_tokens)
    seg_after = seg_row(runtime.model)

    row_deltas = {
        str(token_id): float((rows_after[token_id] - rows_before[token_id]).abs().max())
        for token_id in watch_tokens
    }
    seg_delta = float((seg_after - seg_before).abs().max())

    sam_encoder_grads = [
        name
        for name, parameter in runtime.model.sam.named_parameters()
        if parameter.requires_grad and parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
        and "image_encoder" in name
    ]
    sam_frozen_with_grad = [
        name
        for name, parameter in runtime.model.sam.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]

    checks = {
        "all_losses_finite": all(
            np.isfinite(list(r["losses"].values())).all() for r in results
        ),
        "projection_shape_matches_sam_prompt_dim": all(r["projection_matches_sam_dim"] for r in results),
        "mask_logits_finite_and_non_constant": all(
            r["mask_logits"]["finite"] and not r["mask_logits"]["constant"] for r in results
        ),
        "grads_present_on_all_trained_modules": all(
            r["grad_norms"]["lora"]["norm"] > 0
            and r["grad_norms"]["seg_token"]["norm"] > 0
            and r["grad_norms"]["projection"]["norm"] > 0
            and r["grad_norms"]["sam_mask_decoder"]["norm"] > 0
            for r in results
        ),
        "zero_trainable_params_in_visual_tower": not visual_before and not visual_after,
        "frozen_sam_image_encoder_has_no_grad": not sam_encoder_grads and not sam_frozen_with_grad,
        "seg_token_changed_after_optimizer_step": seg_delta > 0 if not args.no_optimizer_step else None,
        "ordinary_vocab_rows_unchanged": all(delta == 0.0 for delta in row_deltas.values()),
        "no_oom": True,
    }
    success = all(value for value in checks.values() if value is not None)

    section = {
        "ok": bool(success),
        "checks": checks,
        "samples": results,
        "visual_tower_trainables": visual_before,
        "sam_image_encoder_grad_tensors": sam_encoder_grads,
        "frozen_sam_params_with_grad": sam_frozen_with_grad,
        "seg_row_max_abs_delta": seg_delta,
        "watched_token_rows_max_abs_delta": row_deltas,
        "optimizer_groups": [
            {"lr": group["lr"], "weight_decay": group["weight_decay"], "tensors": len(group["params"])}
            for group in optimizer.param_groups
        ],
        "vram_peak": vram(),
        "stage_seconds": round(time.time() - started, 2),
    }
    update_report("stage1", section)
    update_report("trainable_params", runtime.reports["params"])

    print(f"[stage1] checks: {checks}")
    print(f"[stage1] seg row delta: {seg_delta:.6g} | vocab rows delta: {row_deltas}")
    print(f"[stage1] peak VRAM: {section['vram_peak']}")
    print(f"[stage1] ok={success} in {section['stage_seconds']}s")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
