#!/usr/bin/env python
"""Task 6A Stage 2: deterministic 20-sample overfit + paired-instruction test.

    python scripts/task6a_overfit20.py [--config ...] [--max-steps N] [--out-tag TAG]

Trains the MVP on a fixed 10-image x 2-instruction set (two DIFFERENT targets per
image, so image-only memorisation is insufficient) and then measures:

* teacher-forced training mIoU on all 20;
* same-image pair sanity: prediction under instruction A must overlap GT-A more
  than GT-B, and vice versa;
* free-generation `[SEG]` emission for all 20 (image + instruction only, then the
  generated sequence is re-forwarded to obtain the hidden state).

Bounded by early stopping (max steps and max wall-clock minutes from the config).
Saves adapter-only `last` and `best` checkpoints and merges its section into
`evaluation/task6a_smoke_report.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import (  # noqa: E402
    build_manifest,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    write_json,
)
from buildreasonseg_mvp.metrics import collapse_flags, mask_iou_from_logits  # noqa: E402
from buildreasonseg_mvp.qwen_seg import generate_with_seg, seg_hidden_from_full_sequence  # noqa: E402
from buildreasonseg_mvp.reporting import (  # noqa: E402
    CHECKPOINT_MANIFEST,
    read_subset_ids,
    update_report,
)
from buildreasonseg_mvp.runtime import build_runtime, load_config, reset_peak, timed, vram  # noqa: E402
from buildreasonseg_mvp.sam2_bridge import decode_mask  # noqa: E402
from buildreasonseg_mvp.visualize import build_contact_sheet, render_panel  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "evaluation" / "task6a_samples"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"


def evaluate_teacher_forced(runtime, samples, original_sizes) -> list[dict]:
    """Teacher-forced mIoU for every sample, plus LM target diagnostics.

    The extra diagnostics matter: they separate "the mask pathway is wrong" from
    "the mask pathway is fine but the model cannot emit `[SEG]` in free
    generation", which are different failures with different fixes.
    """

    runtime.model.eval()
    results = []
    seg_id = runtime.model.seg_token_id
    with torch.no_grad():
        for sample in samples:
            batch, image = runtime.prepare(sample)
            features, _cached = runtime.features_for(sample, image)
            gt_mask = sample.target_mask()
            moved = batch.to(runtime.device)
            output = runtime.model(moved, features)
            logits = output.mask_logits
            # Primary metric = bilinear, matching SAM2's own postprocess_masks.
            iou = mask_iou_from_logits(logits, torch.as_tensor(gt_mask).float(), gt_mask.shape[-2:])
            iou_nearest = mask_iou_from_logits(
                logits, torch.as_tensor(gt_mask).float(), gt_mask.shape[-2:], mode="nearest"
            )
            flags = collapse_flags(logits.detach().float())

            # LM diagnostics on the teacher-forced assistant span
            lm_logits = output.lm_logits[0].float()
            seg_prediction_position = moved.seg_position - 1
            seg_logits = lm_logits[seg_prediction_position]
            seg_logit = float(seg_logits[seg_id])
            seg_prob = float(torch.softmax(seg_logits, dim=-1)[seg_id])
            seg_is_argmax = bool(int(seg_logits.argmax()) == seg_id)

            supervised = (moved.labels[0] != -100).nonzero(as_tuple=False).flatten()
            correct = 0
            for position in supervised.tolist():
                target_id = int(moved.labels[0, position])
                if int(lm_logits[position - 1].argmax()) == target_id:
                    correct += 1

            results.append(
                {
                    "sample_id": sample.sample_id,
                    "image_id": sample.image_id,
                    "level": sample.level,
                    "query_type": sample.query_type,
                    "target_component_id": sample.target_component_id,
                    "target_area_px": int(gt_mask.sum()),
                    "iou": iou,
                    "iou_nearest": iou_nearest,
                    "collapse": flags,
                    "seg_logit": seg_logit,
                    "seg_probability": seg_prob,
                    "seg_is_argmax": seg_is_argmax,
                    "assistant_token_accuracy": correct / max(len(supervised), 1),
                }
            )
    runtime.model.train()
    return results


def evaluate_pair_cross_iou(runtime, samples) -> tuple[list[dict], dict]:
    """Level-2 evaluation: score every prediction against BOTH targets of its image.

    Task 6A section 16.4 requires, for each same-image pair A/B:

        prediction under instruction A overlaps GT-A more than GT-B
        prediction under instruction B overlaps GT-B more than GT-A

    which is a statement about each prediction against both ground truths, not a
    comparison of the two predictions' own-target IoUs. Both quantities are
    recorded so the distinction is auditable.
    """

    runtime.model.eval()
    per_sample: list[dict] = []
    with torch.no_grad():
        for sample in samples:
            batch, image = runtime.prepare(sample)
            features, _ = runtime.features_for(sample, image)
            output = runtime.model(batch.to(runtime.device), features)
            logits = output.mask_logits.detach()
            cross = {}
            for other in samples:
                if other.image_id != sample.image_id:
                    continue
                gt = torch.as_tensor(other.target_mask()).float()
                cross[other.sample_id] = {
                    "target_component_id": other.target_component_id,
                    "query_type": other.query_type,
                    "iou": mask_iou_from_logits(logits, gt, other.target_mask().shape[-2:]),
                }
            per_sample.append(
                {
                    "sample_id": sample.sample_id,
                    "image_id": sample.image_id,
                    "query_type": sample.query_type,
                    "own_target": sample.target_component_id,
                    "iou_against_each_target_on_this_image": cross,
                }
            )

    by_image: dict[str, list[dict]] = {}
    for entry in per_sample:
        by_image.setdefault(entry["image_id"], []).append(entry)

    pairs: list[dict] = []
    successes = 0
    for image_id, entries in sorted(by_image.items()):
        if len(entries) != 2:
            continue
        a, b = sorted(entries, key=lambda e: e["sample_id"])
        a_own = a["iou_against_each_target_on_this_image"][a["sample_id"]]["iou"]
        a_other = a["iou_against_each_target_on_this_image"][b["sample_id"]]["iou"]
        b_own = b["iou_against_each_target_on_this_image"][b["sample_id"]]["iou"]
        b_other = b["iou_against_each_target_on_this_image"][a["sample_id"]]["iou"]
        a_correct = a_own > a_other
        b_correct = b_own > b_other
        both = bool(a_correct and b_correct)
        successes += int(both)
        pairs.append(
            {
                "image_id": image_id,
                "a": {
                    "sample_id": a["sample_id"],
                    "query_type": a["query_type"],
                    "target": a["own_target"],
                    "iou_on_own_target": a_own,
                    "iou_on_other_target": a_other,
                    "differs": a_correct,
                },
                "b": {
                    "sample_id": b["sample_id"],
                    "query_type": b["query_type"],
                    "target": b["own_target"],
                    "iou_on_own_target": b_own,
                    "iou_on_other_target": b_other,
                    "differs": b_correct,
                },
                "instruction_selects_correct_target": both,
                "both_predictions_identical": abs(a_own - b_other) < 1e-9 and abs(b_own - a_other) < 1e-9,
            }
        )
    runtime.model.train()
    return per_sample, {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "successes": successes,
        "requirement": "at least 9 of 10 pairs",
    }


def free_generation_probe(runtime, samples) -> list[dict]:
    """image + instruction only -> generate -> locate [SEG] -> decode a mask."""

    runtime.model.eval()
    # Autoregressive generation and gradient checkpointing do not mix: the
    # checkpointed path forces use_cache=False, which makes generation
    # pathologically slow. It is disabled for the probe and restored afterwards.
    checkpointing_was_on = bool(getattr(runtime.qwen, "is_gradient_checkpointing", False))
    if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_disable"):
        runtime.qwen.gradient_checkpointing_disable()

    out = []
    try:
        for sample in samples:
            image = sample.image_rgb()
            generation = generate_with_seg(
                runtime.model.qwen,
                runtime.processor,
                runtime.tokenizer,
                image,
                sample.instruction_zh,
                max_new_tokens=int(runtime.cfg["inference"]["max_new_tokens"]),
                seg_token_id=runtime.model.seg_token_id,
            )
            entry = {
                "sample_id": sample.sample_id,
                "seg_count": generation["seg_count"],
                "emitted_exactly_once": generation["seg_count"] == 1,
                "generated_token_count": generation["generated_token_count"],
                "generated_text_tail": generation["text"][-160:],
            }
            if generation["seg_count"] == 1:
                position = generation["seg_positions"][0]
                hidden = seg_hidden_from_full_sequence(
                    runtime.model.qwen,
                    runtime.processor,
                    image,
                    generation["token_ids"],
                    position,
                )
                with torch.no_grad():
                    projected = runtime.projection(hidden)
                    features = runtime.sam_encoder.encode(image)
                    decoded = decode_mask(
                        runtime.model.sam,
                        features,
                        projected,
                        multimask_output=bool(runtime.cfg["inference"]["multimask_output"]),
                    )
                    gt_mask = sample.target_mask()
                    entry["free_generation_iou"] = mask_iou_from_logits(
                        decoded.low_res_logits, torch.as_tensor(gt_mask).float(), gt_mask.shape[-2:]
                    )
            out.append(entry)
    finally:
        if checkpointing_was_on and hasattr(runtime.qwen, "gradient_checkpointing_enable"):
            runtime.qwen.gradient_checkpointing_enable()
    return out


def pair_sanity(evaluations: list[dict], samples: list[data_mod.Sample]) -> dict:
    """Deprecated own-target-only comparison, retained only for history.

    Superseded by `evaluate_pair_cross_iou`, which implements the Task 6A
    section 16.4 criterion correctly: a prediction must overlap its OWN ground
    truth more than the OTHER target on the same image. Comparing the two
    predictions' own-target IoUs (as this function does) can never satisfy a
    "both directions" requirement, because the two conditions are mutually
    exclusive.
    """

    by_image: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_image.setdefault(sample.image_id, []).append(index)

    iou_by_index = {index: evaluation["iou"] for index, evaluation in enumerate(evaluations)}
    per_pair = []
    successes = 0
    for image_id, indices in sorted(by_image.items()):
        if len(indices) != 2:
            continue
        a, b = indices
        a_correct = iou_by_index[a] > iou_by_index[b]
        b_correct = iou_by_index[b] > iou_by_index[a]
        both = a_correct and b_correct
        successes += int(both)
        per_pair.append({"image_id": image_id, "own_target_only": True, "both": both})
    return {"pairs": per_pair, "n_pairs": len(per_pair), "successes": successes, "deprecated": True}


def write_visuals(samples, evaluations, runtime=None) -> list[str]:
    written: list[str] = []
    by_id = {evaluation["sample_id"]: evaluation for evaluation in evaluations}
    for index, sample in enumerate(samples):
        evaluation = by_id.get(sample.sample_id)
        if evaluation is None:
            continue
        image = sample.image_rgb()
        gt = sample.target_mask()
        batch, _ = runtime.prepare(sample)
        features, _ = runtime.features_for(sample, image)
        with torch.no_grad():
            output = runtime.model(batch.to(runtime.device), features)
            from buildreasonseg_mvp.metrics import upsample_logits

            predicted = upsample_logits(output.mask_logits.detach().float(), gt.shape).squeeze() > float(
                runtime.cfg["inference"]["mask_threshold"]
            )
        name = f"{index:02d}_{sample.query_type}_L{sample.level}_{sample.sample_id[-8:]}.png"
        path = SAMPLES_DIR / name
        if render_panel(
            path,
            image,
            gt,
            predicted.cpu().numpy(),
            [sample.instruction_zh, sample.query_type, sample.sample_id],
            evaluation["iou"],
        ):
            written.append(name)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out-tag", default="")
    parser.add_argument(
        "--from-checkpoint",
        default=None,
        help="skip training and evaluate this checkpoint (resume/evaluate path)",
    )
    parser.add_argument(
        "--history-log",
        default=None,
        help="optional stage log whose lines are parsed into the training history",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    subsets = read_subset_ids()
    overfit_ids = subsets["overfit_set"]["sample_ids"]
    records = {r["sample_id"]: r for r in data_mod.read_records("train")}
    samples = [data_mod.to_sample(records[sample_id]) for sample_id in overfit_ids]

    stage_cfg = cfg["stage2_overfit"]
    max_steps = int(args.max_steps or stage_cfg["max_steps"])
    max_seconds = float(stage_cfg["max_minutes"]) * 60.0
    eval_every = int(stage_cfg["eval_every"])
    target_miou = float(stage_cfg["target_train_miou"])
    target_pairs = int(stage_cfg["target_pair_successes"])

    started = time.time()
    reset_peak()
    runtime = build_runtime(cfg, device="cuda", verbose=True)
    optimizer = runtime.build_optimizer()
    scheduler = runtime.build_scheduler(optimizer, max_steps)

    history: list[dict] = []
    if args.history_log:
        import re

        pattern = re.compile(
            r"step (\d+): mean mIoU ([\d.]+) min ([\d.]+) pairs (\S+) collapsed (\d+)"
        )
        for line in Path(args.history_log).read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.search(line)
            if match:
                history.append(
                    {
                        "step": int(match.group(1)),
                        "mean_train_miou": float(match.group(2)),
                        "min_train_miou": float(match.group(3)),
                        "pair_successes": None if match.group(4) == "None" else int(match.group(4)),
                        "collapsed_samples": int(match.group(5)),
                        "source": "parsed from stage log",
                    }
                )

    restored = None
    if args.from_checkpoint:
        restored = load_checkpoint(Path(args.from_checkpoint), runtime.model)
        stopped_reason = "evaluated_from_checkpoint"
        print(f"[stage2] restored {args.from_checkpoint}: {restored}", flush=True)

    # ---- warm the frozen SAM feature cache (never caches Qwen states) ----
    with timed("stage2_feature_cache_warmup"):
        for sample in samples:
            runtime.features_for(sample, sample.image_rgb())
    cache_hits = len(runtime.feature_cache)

    history_seed = history
    history: list[dict] = list(history_seed)
    best_miou = -1.0
    best_step = -1
    success_streak = 0
    stopped_reason = "evaluated_from_checkpoint" if args.from_checkpoint else "max_steps"
    nan_seen = False
    peak = vram()

    step = 0
    while not args.from_checkpoint and step < max_steps:
        sample = samples[step % len(samples)]
        batch, image = runtime.prepare(sample)
        gt_mask = sample.target_mask()
        features, _ = runtime.features_for(sample, image)

        step_result = runtime.train_step(batch, gt_mask, features, optimizer=optimizer)
        losses = step_result["losses"]
        if not np.isfinite(list(losses.values())).all():
            nan_seen = True
            stopped_reason = "nan_or_inf_loss"
            break
        if scheduler is not None:
            scheduler.step()
        step += 1
        del step_result, batch, features

        if step % eval_every == 0 or step == max_steps:
            reset_peak()
            evaluations = evaluate_teacher_forced(runtime, samples, None)
            mious = [evaluation["iou"] for evaluation in evaluations]
            mean_miou = float(np.mean(mious))
            min_miou = float(np.min(mious))
            collapsed = sum(1 for evaluation in evaluations if evaluation["collapse"]["empty"] or evaluation["collapse"]["full"])

            # The rigorous paired-instruction check costs an extra forward per
            # sample, so it is only run when the cheaper mIoU gate is already met
            # -- which is exactly when it can affect early stopping.
            pair_successes = None
            if min_miou >= target_miou and collapsed == 0:
                _cross, pair_result = evaluate_pair_cross_iou(runtime, samples)
                pair_successes = pair_result["successes"]

            entry = {
                "step": step,
                "elapsed_seconds": round(time.time() - started, 1),
                "last_losses": losses,
                "mean_train_miou": mean_miou,
                "min_train_miou": min_miou,
                "collapsed_samples": collapsed,
                "pair_successes": pair_successes,
            }
            history.append(entry)
            print(
                f"[stage2] step {step}: mean mIoU {mean_miou:.4f} min {min_miou:.4f} "
                f"pairs {pair_successes} collapsed {collapsed}",
                flush=True,
            )

            if min_miou > best_miou:
                best_miou = min_miou
                best_step = step
                save_checkpoint(
                    CHECKPOINT_DIR / f"best{args.out_tag}.pt",
                    runtime.model,
                    step,
                    metrics={"min_train_miou": min_miou, "mean_train_miou": mean_miou},
                    config=cfg,
                    optimizer=optimizer,
                )

            if (
                min_miou >= target_miou
                and collapsed == 0
                and pair_successes is not None
                and pair_successes >= target_pairs
            ):
                success_streak += 1
            else:
                success_streak = 0
            if success_streak >= 2:
                stopped_reason = "success_criteria_met_twice"
                break

        if time.time() - started > max_seconds:
            stopped_reason = "max_wall_clock"
            break

    if args.from_checkpoint:
        # Evaluation-only path: never overwrite the training checkpoints.
        last_checkpoint = {
            "path": str(Path(args.from_checkpoint)),
            "bytes": Path(args.from_checkpoint).stat().st_size,
            "sha256": sha256_file(Path(args.from_checkpoint)),
            "step": int(restored.get("step", 0)) if restored else 0,
            "note": "evaluated from an existing checkpoint; no checkpoint written in this run",
        }
    else:
        last_checkpoint = save_checkpoint(
            CHECKPOINT_DIR / f"last{args.out_tag}.pt",
            runtime.model,
            step,
            metrics={"best_min_miou": best_miou, "best_step": best_step},
            config=cfg,
            optimizer=optimizer,
        )
    best_path = CHECKPOINT_DIR / f"best{args.out_tag}.pt"
    best_checkpoint = None
    if best_path.is_file():
        best_checkpoint = {
            "path": str(best_path),
            "bytes": best_path.stat().st_size,
            "sha256": sha256_file(best_path),
            "step": best_step,
        }
    if args.from_checkpoint and best_checkpoint is None:
        # nothing was trained in this run, so the evaluated file IS the checkpoint
        best_checkpoint = last_checkpoint

    # ---- final measurements -------------------------------------------
    #
    # Two checkpoints are evaluated. Task 6A section 18 requires saving `best` by
    # training mIoU, and this is an overfit sanity test rather than a
    # generalisation test, so the reported number is the BEST checkpoint's -- but
    # the last checkpoint's number is reported alongside it so the choice is
    # visible rather than implicit.
    reset_peak()
    last_evaluations = evaluate_teacher_forced(runtime, samples, None)
    last_mious = [evaluation["iou"] for evaluation in last_evaluations]

    restored = None
    if best_path.is_file():
        restored = load_checkpoint(best_path, runtime.model)
        evaluations = evaluate_teacher_forced(runtime, samples, None)
    else:
        evaluations = last_evaluations
    mious = [evaluation["iou"] for evaluation in evaluations]

    cross_iou, pairs = evaluate_pair_cross_iou(runtime, samples)
    visuals = write_visuals(samples, evaluations, runtime=runtime)
    if visuals:
        build_contact_sheet(SAMPLES_DIR, SAMPLES_DIR / "contact_sheet.png")
    stage2_peak = vram()

    generation = free_generation_probe(runtime, samples)
    emission_count = sum(1 for entry in generation if entry["emitted_exactly_once"])
    free_ious = [entry["free_generation_iou"] for entry in generation if "free_generation_iou" in entry]

    all_miou_ok = bool(np.min(mious) >= target_miou)
    pair_ok = pairs["successes"] >= target_pairs
    emission_ok = emission_count == len(samples)
    no_collapse = all(
        not (evaluation["collapse"]["empty"] or evaluation["collapse"]["full"]) for evaluation in evaluations
    )

    if all_miou_ok and pair_ok and emission_ok and not nan_seen and no_collapse:
        verdict = "PASS"
    elif all_miou_ok and pair_ok and not nan_seen and no_collapse and not emission_ok:
        verdict = "PASS_WITH_WARNINGS"
    else:
        verdict = "FAIL_REQUIRES_DEBUG"

    section = {
        "ok": verdict != "FAIL_REQUIRES_DEBUG",
        "verdict": verdict,
        "stopped_reason": stopped_reason,
        "steps": step,
        "nan_seen": nan_seen,
        "feature_cache_images": cache_hits,
        "history": history,
        "final": {
            "checkpoint_used": "best" if restored is not None else "last",
            "best_checkpoint_restore": restored,
            "iou_mode_primary": "bilinear (matches SAM2 postprocess_masks)",
            "mean_train_miou": float(np.mean(mious)),
            "min_train_miou": float(np.min(mious)),
            "max_train_miou": float(np.max(mious)),
            "target_train_miou": target_miou,
            "all_samples_above_target": all_miou_ok,
            "no_collapse": no_collapse,
            "seg_argmax_count": sum(1 for e in evaluations if e["seg_is_argmax"]),
            "mean_assistant_token_accuracy": float(
                np.mean([e["assistant_token_accuracy"] for e in evaluations])
            ),
            "strict_nearest_view": {
                "mean": float(np.mean([e["iou_nearest"] for e in evaluations])),
                "min": float(np.min([e["iou_nearest"] for e in evaluations])),
                "above_target": int(sum(1 for e in evaluations if e["iou_nearest"] >= target_miou)),
                "note": (
                    "block-replication upsampling instead of SAM2's bilinear interpolation; reported so "
                    "the sensitivity of the result to the interpolation convention is visible"
                ),
            },
            "last_checkpoint_mean_train_miou": float(np.mean(last_mious)),
            "last_checkpoint_min_train_miou": float(np.min(last_mious)),
            "per_sample": evaluations,
        },
        "pair_sanity": pairs,
        "pair_cross_iou_per_sample": cross_iou,
        "free_generation": {
            "emission_count": emission_count,
            "n_samples": len(samples),
            "requirement": "20/20",
            "mean_free_generation_iou": float(np.mean(free_ious)) if free_ious else None,
        },
        "checkpoints": {
            "last": last_checkpoint,
            "best": best_checkpoint,
        },
        "vram_peak": stage2_peak,
        "stage_seconds": round(time.time() - started, 2),
        "visuals": visuals,
    }

    update_report("stage2", section)
    update_report(
        "pairwise_instruction_dependence",
        pairs,
    )
    update_report(
        "free_generation_seg_emission",
        {
            "emission_count": emission_count,
            "n_samples": len(samples),
            "requirement": "20/20",
            "per_sample": [
                {
                    "sample_id": entry["sample_id"],
                    "seg_count": entry["seg_count"],
                    "emitted_exactly_once": entry["emitted_exactly_once"],
                    "generated_token_count": entry["generated_token_count"],
                    "free_generation_iou": entry.get("free_generation_iou"),
                }
                for entry in generation
            ],
        },
    )
    update_report("vram", {"stage1": None, "stage2_peak": stage2_peak})
    update_report("timings", {"stage2_seconds": section["stage_seconds"]})
    update_report("checkpoint_hashes", {"last": last_checkpoint, "best": best_checkpoint})

    manifest = build_manifest(
        {"last": last_checkpoint, "best": best_checkpoint} if best_checkpoint else {"last": last_checkpoint},
        base_models={
            "qwen": cfg["models"]["qwen_model_id"],
            "qwen_snapshot_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "sam2": cfg["models"]["sam2_repo_id"],
        },
        extra={"stage": "2", "verdict": verdict, "steps": step},
    )
    write_json(CHECKPOINT_MANIFEST, manifest)

    print(f"[stage2] verdict={verdict} min_miou={min(mious):.4f} pairs={pairs['successes']}/{pairs['n_pairs']} emission={emission_count}/{len(samples)}")
    print(f"[stage2] peak VRAM: {stage2_peak}")
    return 0 if verdict != "FAIL_REQUIRES_DEBUG" else 1


if __name__ == "__main__":
    raise SystemExit(main())
