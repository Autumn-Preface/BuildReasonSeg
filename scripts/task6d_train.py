#!/usr/bin/env python
"""Task 6D sections 9-10: two-stage training of the Spatial Grounding Bridge.

    python scripts/task6d_train.py --stage G0 [--epochs 2]
    python scripts/task6d_train.py --stage G1 [--epochs 3]

* **G0 (grounding proof)** — trains text LoRA + the `[SEG]` machinery + the
  `SpatialGroundingHead` with `2.0 * LM CE + 5.0 * SmoothL1(geometry)`. The SAM2 decoder
  and the projection MLP stay frozen, so no mask gradient exists at all: the stage asks
  only whether the `[SEG]` hidden state can carry target geometry.
* **G1 (joint segmentation)** — additionally trains the SAM2 mask decoder with the full
  objective `2.0 LM CE + 2.0 mask BCE + 1.0 mask Dice + 5.0 grounding`. Qwen base, the
  visual tower, the SAM2 image encoder and the prompt encoder stay frozen; the old
  projection MLP is never used by the candidate.

Both stages use the Task 6C paired `P` subset (480 records / 240 images x 2 targets), the
Task 6C.7 accepted runtime (frozen visual-feature cache, `collect_grad_norms=false`, no
redundant Phase-B transfer), strict determinism and the fixed 120-record validation set.

Geometry comes from the oracle diagnostic's selection under section 5; GT geometry is
supervision only and is never fed to SAM.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.grounding import (  # noqa: E402
    DEFAULT_LAMBDA_GROUND,
    GEOMETRY_BOX,
    GEOMETRY_POINT,
    target_geometry,
)
from buildreasonseg_mvp.grounding_eval import (  # noqa: E402
    free_generation_grounded,
    paired_probe_grounded,
    teacher_forced_grounded,
)
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_seed, vram  # noqa: E402
from task6c_train import (  # noqa: E402
    checkpoint_dir,
    collect_grad_norms_setting,
    merge_json,
    subset_records,
    validation_material,
)

EVAL = REPO_ROOT / "evaluation"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"
ORACLE_JSON = EVAL / "task6d_oracle_prompt_diagnostic.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"

#: Section 9 G0 gate.
G0_EMISSION_GATE = 0.90
G0_PAIRED_GATE = 14
G0_BOX_IOU_GATE = 0.35
G0_POINT_INSIDE_GATE = 0.70


def chosen_geometry() -> str:
    if not ORACLE_JSON.is_file():
        raise SystemExit(
            "evaluation/task6d_oracle_prompt_diagnostic.json is missing; run "
            "scripts/task6d_oracle.py first (section 5 decides the geometry)"
        )
    payload = json.loads(ORACLE_JSON.read_text(encoding="utf-8"))
    kind = payload.get("selection", {}).get("chosen_geometry")
    if kind not in (GEOMETRY_POINT, GEOMETRY_BOX):
        raise SystemExit(f"oracle selection is {kind!r}: {payload.get('verdict')}")
    return kind


def make_optimizer(runtime, steps: int):
    cfg = runtime.cfg["optimizer"]["phase_b"]
    groups = runtime.model.trainable_parameter_groups(
        lora_lr=float(cfg["lora_lr"]),
        head_lr=float(cfg["token_lr"]),
        weight_decay=float(runtime.cfg["optimizer"]["weight_decay"]),
        decoder_lr=float(cfg.get("decoder_lr", cfg["token_lr"])),
        token_lr=float(cfg["token_lr"]),
    )
    optimizer = torch.optim.AdamW(groups, betas=tuple(runtime.cfg["optimizer"]["betas"]))
    scheduler = runtime.build_scheduler_for(optimizer, steps, cfg)
    return optimizer, scheduler, groups


def train_epoch(
    runtime,
    samples,
    optimizer,
    scheduler,
    *,
    stage: str,
    lambda_ground: float,
    log_every: int,
    tag: str,
) -> dict:
    include_mask_loss = stage.upper() == "G1"
    history: list[dict] = []
    started = time.time()
    for step, sample in enumerate(samples, start=1):
        batch, image = runtime.prepare(sample)
        runtime.set_visual_cache_key(sample.image_id)
        features, _cached = runtime.features_for(sample, image)
        result = runtime.grounding_train_step(
            batch,
            sample.target_mask(),
            features,
            optimizer=optimizer,
            lambda_ground=lambda_ground,
            include_mask_loss=include_mask_loss,
        )
        if scheduler is not None:
            scheduler.step()
        if step % log_every == 0 or step == len(samples):
            history.append(
                {
                    "step": step,
                    "losses": result["losses"],
                    "vram": vram(),
                    "predicted_geometry": result["predicted_geometry"].reshape(-1).tolist(),
                    "gt_geometry": result["gt_geometry"].reshape(-1).tolist(),
                }
            )
            losses = result["losses"]
            print(
                f"[{tag}] step {step}/{len(samples)} total {losses['total']:.4f} "
                f"lm {losses['lm_ce']:.4f} bce {losses['mask_bce']:.4f} dice {losses['mask_dice']:.4f} "
                f"ground {losses['ground']:.5f}",
                flush=True,
            )
        del result, features, batch
    return {
        "steps": len(samples),
        "seconds": round(time.time() - started, 2),
        "history": history,
        "stage": stage,
        "include_mask_loss": include_mask_loss,
    }


def evaluate(
    runtime,
    val_samples,
    pairs,
    lookup,
    *,
    free_limit: int | None,
    pairs_limit: int | None,
    skip_free: bool,
    tag: str,
) -> dict:
    tf = teacher_forced_grounded(runtime, val_samples)
    print(
        f"[{tag}] teacher-forced: metric={tf['mean_geometry_metric']:.4f} "
        f"center_inside={tf['center_inside_rate']} mask_mIoU={tf['miou']:.4f}",
        flush=True,
    )
    result = {"teacher_forced": tf}
    if not skip_free:
        subset = val_samples[:free_limit] if free_limit else val_samples
        free = free_generation_grounded(runtime, subset, lookup)
        print(
            f"[{tag}] free: emission={free['emission_valid']}/{free['count']} "
            f"strict_mIoU={free['strict_end_to_end_miou']:.4f} "
            f"center_inside={free['center_inside_rate']} geometry_metric={free['mean_geometry_metric']}",
            flush=True,
        )
        result["free_generation"] = free
        paired_subset = pairs[:pairs_limit] if pairs_limit else pairs
        paired = paired_probe_grounded(runtime, paired_subset, lookup, free_generation=True)
        print(
            f"[{tag}] free paired: mask {paired['mask_paired_pass']}/{paired['paired_total']} "
            f"geometry {paired['geometry_paired_pass']}/{paired['paired_total']} "
            f"own-cross={paired['mean_own_minus_cross_margin']}",
            flush=True,
        )
        result["paired_free"] = paired
    return result


def g0_gate(stage_report: dict, kind: str) -> dict:
    evaluation = stage_report["final_evaluation"]
    free = evaluation.get("free_generation") or {}
    paired = evaluation.get("paired_free") or {}
    emission = free.get("emission_rate")
    geometry_paired = paired.get("geometry_paired_pass")
    metric = free.get("mean_box_iou") if kind == GEOMETRY_BOX else free.get("center_inside_rate")
    checks = {
        "emission_rate_ge_0.90": bool(emission is not None and emission >= G0_EMISSION_GATE),
        "geometry_paired_ge_14": bool(geometry_paired is not None and geometry_paired >= G0_PAIRED_GATE),
        (f"box_iou_ge_{G0_BOX_IOU_GATE}" if kind == GEOMETRY_BOX else f"point_inside_ge_{G0_POINT_INSIDE_GATE}"): bool(
            metric is not None
            and metric >= (G0_BOX_IOU_GATE if kind == GEOMETRY_BOX else G0_POINT_INSIDE_GATE)
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "emission_rate": emission,
        "geometry_paired": geometry_paired,
        "metric": metric,
        "verdict_if_failed": "GROUNDING_REPRESENTATION_FAILED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["G0", "G1"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke only")
    parser.add_argument("--log-every", type=int, default=40)
    parser.add_argument("--free-limit", type=int, default=None, help="limit free-generation records")
    parser.add_argument("--pairs-limit", type=int, default=None)
    parser.add_argument("--skip-free", action="store_true", help="teacher-forced validation only")
    parser.add_argument("--smoke", action="store_true", help="no checkpoint, tiny run")
    parser.add_argument(
        "--init-from",
        default=None,
        help="checkpoint to continue from (G1 normally starts from the G0 checkpoint)",
    )
    args = parser.parse_args(argv)

    kind = chosen_geometry()
    stage = args.stage.upper()
    epochs = args.epochs if args.epochs is not None else (2 if stage == "G0" else 3)
    max_steps = args.max_steps if args.max_steps is not None else (2 if stage == "G0" else 3)

    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": "centre"}  # unused by the candidate; kept for report comparability
    train_records = subset_records(json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8")), "P")
    train_samples = [data_mod.to_sample(record) for record in train_records]
    val_samples, pairs, lookup, lookup_audit = validation_material()
    if args.max_steps:
        train_samples = train_samples[: args.max_steps]
    if args.free_limit:
        val_samples_free = val_samples[: args.free_limit]
    else:
        val_samples_free = val_samples

    print(
        f"[task6d] stage {stage}, geometry {kind}, train {len(train_samples)} (P subset), "
        f"val {len(val_samples)}, pairs {len(pairs)}",
        flush=True,
    )

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, device="cuda", verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    head_report = runtime.install_grounding_head(kind)
    trainables = runtime.set_grounding_trainables(stage)

    init_report = None
    if args.init_from:
        from buildreasonseg_mvp.checkpointing import load_checkpoint

        init_report = load_checkpoint(Path(args.init_from), runtime.model)
        print(f"[task6d] initialised from {args.init_from}: {json.dumps(init_report, default=str)}", flush=True)

    steps_per_epoch = len(train_samples)
    optimizer, scheduler, groups = make_optimizer(runtime, steps_per_epoch)

    report: dict = {
        "_doc": (
            f"Task 6D stage {stage} training of the Spatial Grounding Bridge v0.1 on the Task 6C "
            "paired P subset. Geometry is {kind}; GT geometry is supervision only and never enters a "
            "prompt. Runtime is the Task 6C.7 accepted configuration."
        ),
        "task": "6D",
        "stage": stage,
        "geometry_kind": kind,
        "geometry_selection_source": "evaluation/task6d_oracle_prompt_diagnostic.json",
        "config": cfg,
        "determinism": runtime.reports["determinism"],
        "visual_cache": runtime.visual_cache_stats(),
        "grounding_head": head_report,
        "trainables": trainables,
        "initialised_from": init_report,
        "lookup_audit": lookup_audit,
        "train_records": len(train_samples),
        "val_records": len(val_samples),
        "pairs": len(pairs),
        "lambda_ground": DEFAULT_LAMBDA_GROUND,
        "loss_weights": runtime.grounding_loss_weights(stage == "G1").as_dict(),
        "epochs_requested": epochs,
        "optimizer_groups": [
            {"lr": g["lr"], "weight_decay": g["weight_decay"], "name": g.get("name"), "tensors": len(g["params"])}
            for g in groups
        ],
        "epochs": [],
    }

    for epoch in range(1, epochs + 1):
        set_seed(int(cfg["seed"]) + epoch)
        result = train_epoch(
            runtime,
            train_samples,
            optimizer,
            scheduler,
            stage=stage,
            lambda_ground=DEFAULT_LAMBDA_GROUND,
            log_every=args.log_every,
            tag=f"task6d.{stage}.e{epoch}",
        )
        evaluation = evaluate(
            runtime,
            val_samples,
            pairs,
            lookup,
            free_limit=args.free_limit,
            pairs_limit=args.pairs_limit,
            skip_free=args.skip_free,
            tag=f"task6d.{stage}.e{epoch}",
        )
        report["epochs"].append({"epoch": epoch, **result, "validation": evaluation})
        if not args.smoke:
            directory = Path(cfg["paths"]["checkpoints"]) / f"task6d_{stage}"
            if not directory.is_absolute():
                directory = REPO_ROOT / directory
            directory.mkdir(parents=True, exist_ok=True)
            from buildreasonseg_mvp.checkpointing import save_checkpoint, sha256_file

            path = directory / f"{stage}_epoch{epoch}.pt"
            save_checkpoint(path, runtime.model, epoch * len(train_samples), metrics={}, config=cfg)
            merge_json(
                EVAL / "task6d_checkpoint_manifest.json",
                f"{stage}_epoch{epoch}",
                {
                    "stage": stage,
                    "epoch": epoch,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "geometry_kind": kind,
                    "trainable_params": trainables["trainable_params"],
                },
            )

    report["final_evaluation"] = report["epochs"][-1]["validation"]
    if stage == "G0":
        report["gate"] = g0_gate(report, kind)
        report["verdict"] = "G0_PASS" if report["gate"]["passed"] else "GROUNDING_REPRESENTATION_FAILED"
    else:
        report["verdict"] = "G1_COMPLETE"

    out = EVAL / f"task6d_{stage.lower()}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[task6d] verdict: {report['verdict']}", flush=True)
    if stage == "G0":
        print(f"[task6d] gate: {json.dumps(report['gate'], ensure_ascii=False)}", flush=True)
    print(f"[task6d] wrote {out.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if report.get("verdict") != "GROUNDING_REPRESENTATION_FAILED" else 5


if __name__ == "__main__":
    raise SystemExit(main())
