#!/usr/bin/env python
"""Task 6C.5 section 8: synchronized stage profile of the current pipeline.

    python scripts/task6c5_profile.py [--samples 16]

This is deliberately a **separate, synchronized** run: every stage is bracketed by
`torch.cuda.synchronize()` so the stage costs can be attributed. That makes the wall
time of this run unusable as a throughput number 鈥?that is what
`scripts/task6c5_benchmark.py` is for, and the two must not be confused.

It measures the stage list from Task 6C.5 section 8 and additionally splits
`qwen_forward` from `sam_projection_decode_forward` with a forward-only, CUDA-event
timed pass, because the joint backward makes an exact split impossible.

Writes `evaluation/task6c5_profile_baseline.json`.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.perf import StageProfile, UtilizationSampler, percentiles  # noqa: E402
from buildreasonseg_mvp.pipeline import PipelineFlags, TrainingPipeline  # noqa: E402
from buildreasonseg_mvp.qwen_seg import forward_qwen  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config  # noqa: E402
from buildreasonseg_mvp.sam2_bridge import decode_mask  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6c5_profile_baseline.json"
IDS_JSON = EVAL / "task6c5_benchmark_ids.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"


def load_samples(limit: int) -> list:
    payload = json.loads(IDS_JSON.read_text(encoding="utf-8"))
    train = {record["sample_id"]: record for record in data_mod.read_records("train")}
    return [data_mod.to_sample(train[sample_id]) for sample_id in payload["record_ids"][:limit]]


def processor_decomposition(runtime, samples, repeats: int = 8) -> dict:
    """Estimate the image-preprocessing vs chat/tokenizer split inside `prepare`.

    `build_teacher_forcing_batch` runs the processor once, so the two parts are not
    separable from a single call. This measures the image processor alone on the same
    frames and reports `total - image_processor` as the chat/template + tokenizer part,
    which is labelled as an estimate.
    """

    image_times: list[float] = []
    total_times: list[float] = []
    for index in range(repeats):
        sample = samples[index % len(samples)]
        image = sample.image_rgb()
        started = time.perf_counter()
        runtime.processor.image_processor(images=[image], return_tensors="pt")
        image_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        runtime.prepare(sample, image=image)
        total_times.append(time.perf_counter() - started)
    image_mean = float(np.mean(image_times))
    total_mean = float(np.mean(total_times))
    return {
        "repeats": repeats,
        "image_processor_seconds": percentiles(image_times),
        "prepare_total_seconds": percentiles(total_times),
        "chat_template_and_tokenizer_seconds_estimate": total_mean - image_mean,
        "estimate_note": (
            "the chat/template + tokenizer share is derived as prepare_total minus the image "
            "processor alone, because one processor call does both; treat it as an estimate"
        ),
    }


def forward_split(runtime, pipeline, samples, repeats: int = 8) -> dict:
    """Forward-only split of Qwen forward vs projection+SAM decode, with CUDA events."""

    qwen_times: list[float] = []
    sam_times: list[float] = []
    for index in range(repeats):
        sample = samples[index % len(samples)]
        batch, image, _gt = pipeline.prepare(sample)
        moved = batch.to(runtime.device)
        features, _cached = runtime.features_for(sample, image)

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            lm_logits, seg_hidden = forward_qwen(runtime.model.qwen, moved)
        end.record()
        torch.cuda.synchronize()
        qwen_times.append(start.elapsed_time(end) / 1000.0)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            projected = runtime.projection(seg_hidden)
            decode_mask(runtime.sam, features, projected, multimask_output=False, bridge=runtime.bridge)
        end.record()
        torch.cuda.synchronize()
        sam_times.append(start.elapsed_time(end) / 1000.0)
        del lm_logits, seg_hidden, projected
    return {
        "repeats": repeats,
        "qwen_forward_seconds": percentiles(qwen_times),
        "projection_plus_sam_decode_seconds": percentiles(sam_times),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sampler-interval", type=float, default=0.5)
    args = parser.parse_args(argv)

    samples = load_samples(args.samples)
    warm_samples = load_samples(64)

    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": "centre"}
    runtime = build_runtime(cfg, device="cuda", verbose=False)

    pipeline = TrainingPipeline(runtime, PipelineFlags())  # B0 path: no input caches
    from buildreasonseg_mvp.pipeline import warm_caches

    warm_caches(pipeline, warm_samples)
    optimizer = torch.optim.AdamW(
        runtime.model.trainable_parameter_groups(
            lora_lr=1e-4, head_lr=3e-4, weight_decay=0.01, decoder_lr=3e-4, token_lr=3e-4
        ),
        betas=(0.9, 0.999),
    )

    timer = StageProfile(cuda_events=True)
    with UtilizationSampler(interval_seconds=args.sampler_interval) as sampler:
        for sample in samples:
            step_started = time.perf_counter()
            with timer.stage("image_io"):
                image = sample.image_rgb()
            with timer.stage("qwen_prepare_total"):
                batch, _image = runtime.prepare(sample, image=image)
            with timer.stage("target_mask_io"):
                gt_mask = sample.target_mask()
            with timer.stage("cpu_to_gpu"):
                moved = batch.to(runtime.device)
            with timer.stage("sam_feature_lookup"):
                features, _cached = runtime.features_for(sample, image)
            runtime.train_step(moved, gt_mask, features, optimizer=optimizer, timer=timer)
            timer.step_walls.append(time.perf_counter() - step_started)
    utilization = sampler.summary()

    report = {
        "_doc": (
            "Task 6C.5 section 8 synchronized stage profile of the B0 (current) path. Because every "
            "stage is bracketed by torch.cuda.synchronize(), the wall times here are NOT throughput "
            "numbers; use evaluation/task6c5_variants.json for throughput."
        ),
        "task": "6C.5",
        "profile_kind": "synchronized_stage_profile",
        "samples": len(samples),
        "synchronized": True,
        "stage_profile": timer.summary(),
        "sam_feature_cache": runtime.feature_cache.stats(),
        "utilization_during_profile": utilization,
        "processor_decomposition": processor_decomposition(runtime, samples),
        "forward_split": forward_split(runtime, pipeline, samples),
        "gradient_checkpointing": bool(runtime.reports.get("gradient_checkpointing", False)),
        "deterministic": runtime.reports["determinism"],
        "vram": {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shares = report["stage_profile"]["stage_share_percent"]
    print(f"[profile] stage shares (% of synchronized wall): {shares}")
    print(f"[profile] wrote {OUT.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
