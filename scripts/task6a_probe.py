#!/usr/bin/env python
"""Task 6A Stage 0: environment / model-load proof, plus subset selection.

    python scripts/task6a_probe.py [--config configs/mvp/task6a_2b_seg.yaml]

Loads Qwen3-VL-2B-Instruct and SAM2.1 Hiera Base+ (no training), measures the
processor output on real WHU tiles, and fixes the deterministic 2-sample and
20-sample subsets. Writes:

    evaluation/task6a_smoke_report.json   (section: stage0, plus environment)
    evaluation/task6a_subset_ids.json
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
from buildreasonseg_mvp.reporting import update_report, write_subset_ids  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, reset_peak, rss_gib, timed, vram  # noqa: E402


def processor_measurements(runtime, samples) -> list[dict]:
    """Measure the Qwen branch on real images for one sample per level."""

    out: list[dict] = []
    for sample in samples:
        image = sample.image_rgb()
        batch = runtime.prepare(sample)[0]
        grid = batch.image_grid_thw
        out.append(
            {
                "sample_id": sample.sample_id,
                "level": sample.level,
                "query_type": sample.query_type,
                "source_image_shape": list(image.shape),
                "processor_pixel_values_shape": list(batch.pixel_values.shape)
                if batch.pixel_values is not None
                else None,
                "image_grid_thw": grid.tolist() if grid is not None else None,
                "measured_visual_tokens": batch.visual_tokens,
                "prompt_length": batch.prompt_length,
                "total_sequence_length": batch.total_length,
                "assistant_target_tokens": batch.total_length - batch.prompt_length,
                "seg_position": batch.seg_position,
                "seg_token_id": runtime.model.seg_token_id,
                "labels_supervised_tokens": int((batch.labels != -100).sum().item()),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    started = time.time()
    reset_peak()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with timed("stage0_build_runtime"):
        runtime = build_runtime(cfg, device=device)

    # ---- deterministic subsets ----------------------------------------
    train_records = data_mod.read_records("train")
    smoke_pair = data_mod.select_smoke_pair(train_records)
    overfit = data_mod.select_overfit_set(
        train_records,
        images=int(cfg["stage2_overfit"]["images"]),
        per_image=int(cfg["stage2_overfit"]["per_image"]),
    )
    subset_ids = {
        "_doc": "Task 6A section 8: deterministic subsets, chosen before any training and never revised by model results.",
        "smoke_pair": data_mod.subset_summary(smoke_pair),
        "overfit_set": data_mod.subset_summary(overfit),
        "smoke_pair_shares_image": len({r["image_id"] for r in smoke_pair}) == 1,
        "smoke_pair_targets": sorted({int(r["target_component_id"]) for r in smoke_pair}),
        "overfit_targets_per_image": {
            image_id: sorted(int(r["target_component_id"]) for r in overfit if r["image_id"] == image_id)
            for image_id in sorted({r["image_id"] for r in overfit})
        },
        "dataset_version": data_mod.DATASET_VERSION,
        "split_used": "train",
    }
    write_subset_ids(subset_ids)

    # ---- one record per level for preprocessing measurement ------------
    per_level: list[dict] = []
    for level in (1, 2, 3):
        candidate = next((r for r in train_records if int(r["level"]) == level), None)
        if candidate is not None:
            per_level.append(candidate)
    measurements = processor_measurements(runtime, [data_mod.to_sample(r) for r in per_level])

    # ---- SAM2 measurements --------------------------------------------
    probe_sample = data_mod.to_sample(smoke_pair[0])
    probe_image = probe_sample.image_rgb()
    reset_peak()
    with timed("stage0_sam2_encode"):
        features = runtime.sam_encoder.encode(probe_image)
    sam_vram = vram()
    sam_report = dict(runtime.reports["sam2"])
    sam_report.update(features.as_dict())

    combined = vram()
    stage0 = {
        "ok": True,
        "device": device,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "qwen": {
            "model_id": cfg["models"]["qwen_model_id"],
            "dtype": str(next(runtime.qwen.parameters()).dtype),
            "config_torch_dtype": str(getattr(runtime.qwen.config, "dtype", None)),
            "hidden_size": runtime.reports["qwen_hidden_size"],
            "tie_word_embeddings": bool(getattr(runtime.qwen.config, "tie_word_embeddings", False)),
            "vocab_size": int(runtime.qwen.config.text_config.vocab_size)
            if hasattr(runtime.qwen.config, "text_config")
            else int(runtime.qwen.config.vocab_size),
            "embedding_tied_now": runtime.reports["token"]["tied_after_resize"],
        },
        "sam2": sam_report,
        "token": runtime.reports["token"],
        "lora": runtime.reports["lora"],
        "params": runtime.reports["params"],
        "sam2_freeze": runtime.reports["sam2_freeze"],
        "projection": runtime.reports["projection"],
        "image_budget": runtime.reports["image_budget"],
        "ram_gib": runtime.reports["ram"],
        "vram_after_load": runtime.reports["vram_after_load"],
        "vram_after_sam2_encode": sam_vram,
        "vram_idle_combined": combined,
        "processor_measurements": measurements,
        "stage_seconds": round(time.time() - started, 2),
    }

    update_report(
        "stage0",
        stage0,
        extra={
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "vram_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                if torch.cuda.is_available()
                else None,
            },
            "model_revisions": {
                "qwen_model_id": cfg["models"]["qwen_model_id"],
                "qwen_snapshot_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
                "sam2_repo_id": cfg["models"]["sam2_repo_id"],
                "sam2_source_revision": sam_report.get("source_revision", ""),
            },
            "trainable_params": runtime.reports["params"],
        },
    )

    print(f"[stage0] visual tokens measured: {[m['measured_visual_tokens'] for m in measurements]}")
    print(f"[stage0] sequence lengths: {[m['total_sequence_length'] for m in measurements]}")
    print(f"[stage0] idle combined VRAM: {combined}")
    print(f"[stage0] done in {stage0['stage_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
