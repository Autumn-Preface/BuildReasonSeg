#!/usr/bin/env python
"""Task 6B: 2B real mini-train (Phase A language warm-up + Phase B joint) with validation.

    python scripts/task6b_train.py [--config ...] [--phase-a-steps N] [--phase-b-epochs N]
                                   [--max-phase-b-steps N] [--skip-baseline] [--tag TAG]

Starts from a **clean** Task 6A-equivalent architecture (fresh LoRA, fresh `[SEG]`
row, fresh Projection MLP, original SAM2 mask-decoder weights). Task 6A checkpoints
are never used as an initialisation.

Writes:
    evaluation/task6b_baseline.json          (pre-training, if not skipped)
    evaluation/task6b_training_report.json
    evaluation/task6b_validation.json        (final, from the best_joint checkpoint)
    evaluation/task6b_checkpoint_manifest.json
    artifacts/checkpoints/task6b/*.pt        (gitignored)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The run is offline: huggingface.co does not resolve on this network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp import language_metrics as LM  # noqa: E402
from buildreasonseg_mvp import validation as V  # noqa: E402
from buildreasonseg_mvp.checkpointing import (  # noqa: E402
    build_manifest,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
    write_json,
)
from buildreasonseg_mvp.runtime import (  # noqa: E402
    build_runtime,
    load_config,
    reset_peak,
    set_phase_trainables,
    vram,
)
from buildreasonseg_mvp.subsets import load_subset  # noqa: E402

SUBSETS = REPO_ROOT / "evaluation" / "task6b_subset_ids.json"
BASELINE_JSON = REPO_ROOT / "evaluation" / "task6b_baseline.json"
TRAINING_JSON = REPO_ROOT / "evaluation" / "task6b_training_report.json"
VALIDATION_JSON = REPO_ROOT / "evaluation" / "task6b_validation.json"
CHECKPOINT_MANIFEST = REPO_ROOT / "evaluation" / "task6b_checkpoint_manifest.json"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "task6b"

#: best_joint lexicographic rule, defined BEFORE Phase B and never changed after.
BEST_JOINT_RULE = ("valid_seg_emission_rate", "strict_end_to_end_miou", "operation_chain_accuracy", "lm_ce")


def samples_for(payload: dict, key: str) -> list[data_mod.Sample]:
    split = payload[key]["source_split"]
    records = {r["sample_id"]: r for r in data_mod.read_records(split)}
    return [data_mod.to_sample(records[sid]) for sid in payload[key]["sample_ids"]]


def full_validation(runtime, val_samples, lookup, do_free_generation: bool = True) -> dict:
    tf = V.teacher_forced_validation(runtime, val_samples, reasoning_lookup=lookup)
    result = {
        "teacher_forced": V.aggregate_teacher_forced(tf.records),
        "teacher_forced_breakdown_level": V.teacher_forced_breakdown(tf.records, "level_key"),
        "teacher_forced_breakdown_family": V.teacher_forced_breakdown(tf.records, "family"),
        "teacher_forced_seconds": round(tf.seconds, 2),
        "teacher_forced_peak_vram": tf.peak_vram,
    }
    if do_free_generation:
        fg = V.free_generation_validation(runtime, val_samples, lookup)
        result["free_generation"] = V.aggregate(fg.records)
        result["free_generation_breakdown_level"] = V.breakdown(fg.records, "level_key")
        result["free_generation_breakdown_family"] = V.breakdown(fg.records, "family")
        result["free_generation_seconds"] = round(fg.seconds, 2)
        result["free_generation_peak_vram"] = fg.peak_vram
        result["language"] = V.language_summary(fg.records, tf.records)
        result["_free_records"] = fg.records
        result["_tf_records"] = tf.records
    return result


def selection_metrics(result: dict) -> dict:
    fg = result.get("free_generation", {})
    tf = result.get("teacher_forced", {})
    return {
        "valid_seg_emission_rate": fg.get("valid_seg_emission_rate", 0.0) or 0.0,
        "strict_end_to_end_miou": fg.get("strict_end_to_end_miou", 0.0) or 0.0,
        "operation_chain_accuracy": fg.get("operation_chain_accuracy", 0.0) or 0.0,
        "lm_ce": tf.get("lm_ce", float("inf")),
        "teacher_forced_miou": tf.get("miou", 0.0) or 0.0,
        "conditional_miou": fg.get("conditional_miou"),
    }


def better_joint(candidate: dict, incumbent: dict | None) -> bool:
    """Lexicographic comparison using the pre-declared rule."""

    if incumbent is None:
        return True
    for key in BEST_JOINT_RULE:
        a, b = candidate.get(key), incumbent.get(key)
        if a is None or b is None:
            continue
        if key == "lm_ce":
            if a < b - 1e-9:
                return True
            if a > b + 1e-9:
                return False
        else:
            if a > b + 1e-9:
                return True
            if a < b - 1e-9:
                return False
    return False


def make_optimizer(runtime, phase: str, steps: int):
    cfg = runtime.cfg["optimizer"][f"phase_{phase.lower()}"]
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


def train_phase(runtime, samples, phase: str, steps: int, log_every: int, tag: str) -> dict:
    """Run `steps` optimisation steps over the fixed record list.

    Phase A optimises the language objective only and never executes the mask
    pathway; Phase B runs the joint objective through `runtime.train_step`.
    """

    optimizer, scheduler, groups = make_optimizer(runtime, phase, steps)
    history: list[dict] = []
    step = 0
    nan_seen = False
    started = time.time()
    reset_peak()
    is_language_only = phase.upper() == "A"

    while step < steps:
        sample = samples[step % len(samples)]
        batch, image = runtime.prepare(sample)
        moved = batch.to(runtime.device)

        if is_language_only:
            # No SAM2 work at all in Phase A: the mask pathway is not trained and
            # the frozen features are not even computed.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                output = runtime.model.qwen(
                    input_ids=moved.input_ids,
                    attention_mask=moved.attention_mask,
                    pixel_values=moved.pixel_values,
                    image_grid_thw=moved.image_grid_thw,
                    use_cache=False,
                    **moved.extra_inputs,
                )
                loss = torch.nn.functional.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]).float(),
                    moved.labels.reshape(-1),
                    ignore_index=-100,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clipped = float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in runtime.model.parameters() if p.requires_grad],
                    float(runtime.cfg["optimizer"]["grad_clip_norm"]),
                )
            )
            optimizer.step()
            value = float(loss.detach())
            losses = {"total": value, "lm_ce": value, "mask_bce": 0.0, "mask_dice": 0.0}
            grad_summary = {"grad_clip_total_norm": clipped}
            del output
        else:
            gt_mask = sample.target_mask()
            features, _cached = runtime.features_for(sample, image)
            step_result = runtime.train_step(batch, gt_mask, features, optimizer=optimizer)
            losses = step_result["losses"]
            grad_summary = {name: data["norm"] for name, data in step_result["grad_norms"].items()}
            grad_summary["grad_clip_total_norm"] = step_result["grad_clip_total_norm"]
            del step_result, features

        if not np.isfinite(list(losses.values())).all():
            nan_seen = True
            break

        if scheduler is not None:
            scheduler.step()
        step += 1
        del batch, moved

        if step % log_every == 0 or step == steps:
            entry = {
                "step": step,
                "phase": phase,
                "elapsed_seconds": round(time.time() - started, 1),
                "lr": [group["lr"] for group in optimizer.param_groups],
                "losses": losses,
                "grad": grad_summary,
                "vram": vram(),
            }
            history.append(entry)
            print(
                f"[{tag}] step {step}/{steps} total {losses['total']:.4f} "
                f"lm {losses['lm_ce']:.4f} bce {losses['mask_bce']:.4f} dice {losses['mask_dice']:.4f}",
                flush=True,
            )

    return {
        "phase": phase,
        "steps": step,
        "nan_seen": nan_seen,
        "seconds": round(time.time() - started, 2),
        "optimizer_groups": [
            {"lr": g["lr"], "weight_decay": g["weight_decay"], "name": g.get("name"), "tensors": len(g["params"])}
            for g in groups
        ],
        "history": history,
        "peak_vram": vram(),
        "feature_cache": runtime.feature_cache.stats(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--phase-a-steps", type=int, default=None)
    parser.add_argument("--phase-b-epochs", type=int, default=None)
    parser.add_argument("--max-phase-b-steps", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-per-epoch-freegen", action="store_true")
    parser.add_argument("--val-limit", type=int, default=None, help="smoke-test only: cap validation records")
    parser.add_argument("--tag", default="task6b")
    args = parser.parse_args(argv)

    cfg = load_config(args.config or (REPO_ROOT / "configs" / "mvp" / "task6b_2b_minitrain.yaml"))
    payload = json.loads(SUBSETS.read_text(encoding="utf-8"))
    train_samples = samples_for(payload, "train")
    val_samples = samples_for(payload, "val")
    paired_ids = payload["paired_probe"]["sample_ids"]
    paired_records = [r for r in data_mod.read_records("val") if r["sample_id"] in set(paired_ids)]
    by_id = {r["sample_id"]: r for r in paired_records}
    paired_records = [by_id[sid] for sid in paired_ids if sid in by_id]
    pairs = [{"a": paired_records[i], "b": paired_records[i + 1]} for i in range(0, len(paired_records), 2)]

    lookup = LM.build_reasoning_lookup(list(data_mod.read_records("train")) + list(data_mod.read_records("val")))

    if args.val_limit:
        val_samples = val_samples[: args.val_limit]
        pairs = pairs[: max(1, args.val_limit // 4)]

    print(f"[task6b] train {len(train_samples)} records | val {len(val_samples)} records | pairs {len(pairs)}")
    runtime = build_runtime(cfg, device="cuda", verbose=True)

    report: dict = {
        "_doc": "Task 6B training report: two-phase deterministic 2B mini-train.",
        "task": "6B",
        "config": cfg,
        "package_versions": _packages(),
        "model_revisions": {
            "qwen_model_id": cfg["models"]["qwen_model_id"],
            "qwen_snapshot_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "sam2_repo_id": cfg["models"]["sam2_repo_id"],
            "sam2_source_revision": runtime.reports["sam2"].get("source_revision", ""),
        },
        "subset_sizes": {"train": len(train_samples), "val": len(val_samples), "paired_pairs": len(pairs)},
        "started_from_clean_base": True,
        "trainable_params_initial": runtime.reports["params"],
        "best_joint_rule": list(BEST_JOINT_RULE),
        "phases": {},
    }

    # ---------------- fresh baseline (section 16) ----------------
    if not args.skip_baseline:
        print("[task6b] fresh baseline validation ...", flush=True)
        set_phase_trainables(runtime.model, "B")  # everything the final model can train
        baseline = full_validation(runtime, val_samples, lookup)
        baseline["selection"] = selection_metrics(baseline)
        baseline.pop("_free_records", None)
        baseline.pop("_tf_records", None)
        write_json(BASELINE_JSON, baseline)
        report["baseline"] = {k: v for k, v in baseline.items()}
        print(f"[task6b] baseline: {baseline['selection']}", flush=True)

    # ---------------- Phase A ----------------
    phase_a_trainables = set_phase_trainables(runtime.model, "A")
    report["phases"]["A_trainables"] = phase_a_trainables
    print(f"[task6b] Phase A trainables: {phase_a_trainables}", flush=True)
    phase_a_steps = int(args.phase_a_steps or (cfg["phase_a"]["epochs"] * len(train_samples)))
    report["phases"]["A"] = train_phase(
        runtime, train_samples, "A", phase_a_steps, int(cfg["phase_a"]["log_every_steps"]), "phaseA"
    )

    print("[task6b] validating after Phase A ...", flush=True)
    after_a = full_validation(runtime, val_samples, lookup)
    after_a["selection"] = selection_metrics(after_a)
    report["phases"]["A_validation"] = {k: v for k, v in after_a.items() if not k.startswith("_")}
    print(f"[task6b] after A: {after_a['selection']}", flush=True)
    save_checkpoint(
        CHECKPOINT_DIR / "phaseA_last.pt", runtime.model, phase_a_steps,
        metrics=after_a["selection"], config=cfg,
    )

    # ---------------- Phase B ----------------
    phase_b_trainables = set_phase_trainables(runtime.model, "B")
    report["phases"]["B_trainables"] = phase_b_trainables
    print(f"[task6b] Phase B trainables: {phase_b_trainables}", flush=True)

    epochs = int(args.phase_b_epochs or cfg["phase_b"]["epochs"])
    max_steps = int(args.max_phase_b_steps or cfg["phase_b"]["max_steps"])
    patience = int(cfg["phase_b"]["early_stop_patience"])
    steps_per_epoch = len(train_samples)

    best_joint_metrics: dict | None = None
    best_language_ce = float("inf")
    best_mask_miou = -1.0
    no_improve = 0
    total_b_steps = 0
    epochs_log: list[dict] = []
    final_result: dict | None = None

    for epoch in range(1, epochs + 1):
        if total_b_steps >= max_steps:
            print(f"[task6b] Phase B step budget reached ({max_steps})")
            break
        steps = min(steps_per_epoch, max_steps - total_b_steps)
        epoch_result = train_phase(runtime, train_samples, "B", steps, int(cfg["phase_b"]["log_every_steps"]), f"phaseB.e{epoch}")
        total_b_steps += epoch_result["steps"]

        do_fg = bool(cfg["validation"]["per_epoch_free_generation"]) and not args.skip_per_epoch_freegen
        validation = full_validation(runtime, val_samples, lookup, do_free_generation=do_fg)
        selection = selection_metrics(validation)
        epoch_entry = {
            "epoch": epoch,
            "steps_this_epoch": epoch_result["steps"],
            "total_phase_b_steps": total_b_steps,
            "seconds": epoch_result["seconds"],
            "peak_vram": epoch_result["peak_vram"],
            "history": epoch_result["history"],
            "nan_seen": epoch_result["nan_seen"],
            "selection": selection,
            "teacher_forced": validation["teacher_forced"],
            "free_generation": validation.get("free_generation"),
            "language": validation.get("language"),
        }
        epochs_log.append(epoch_entry)
        print(
            f"[task6b] epoch {epoch} done: emission {selection['valid_seg_emission_rate']:.3f} "
            f"strict mIoU {selection['strict_end_to_end_miou']:.4f} chain {selection['operation_chain_accuracy']:.3f} "
            f"lm_ce {selection['lm_ce']:.4f}",
            flush=True,
        )

        save_checkpoint(
            CHECKPOINT_DIR / "last.pt", runtime.model, total_b_steps,
            metrics=selection, config=cfg,
        )
        if selection["teacher_forced_miou"] > best_mask_miou:
            best_mask_miou = selection["teacher_forced_miou"]
            save_checkpoint(CHECKPOINT_DIR / "best_mask.pt", runtime.model, total_b_steps,
                            metrics=selection, config=cfg)
        if selection["lm_ce"] < best_language_ce:
            best_language_ce = selection["lm_ce"]
            save_checkpoint(CHECKPOINT_DIR / "best_language.pt", runtime.model, total_b_steps,
                            metrics=selection, config=cfg)

        if do_fg and better_joint(selection, best_joint_metrics):
            best_joint_metrics = selection
            no_improve = 0
            save_checkpoint(CHECKPOINT_DIR / "best_joint.pt", runtime.model, total_b_steps,
                            metrics=selection, config=cfg)
            final_result = validation
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[task6b] early stopping after {no_improve} epochs without best_joint improvement")
                break

        if epoch_result["nan_seen"]:
            print("[task6b] NaN/Inf detected; stopping Phase B")
            break

    report["phases"]["B_epochs"] = epochs_log
    report["phase_b_total_steps"] = total_b_steps
    report["best_joint_selection"] = best_joint_metrics

    # ---------------- final validation from best_joint ----------------
    best_joint_path = CHECKPOINT_DIR / "best_joint.pt"
    final_checkpoint = None
    if best_joint_path.is_file():
        restore = load_checkpoint(best_joint_path, runtime.model)
        final_checkpoint = {
            "path": str(best_joint_path),
            "bytes": best_joint_path.stat().st_size,
            "sha256": sha256_file(best_joint_path),
            "restore": restore,
            "metrics": best_joint_metrics,
        }
        print(f"[task6b] restored best_joint: {restore}")

    print("[task6b] final validation ...", flush=True)
    final = full_validation(runtime, val_samples, lookup)
    final_selection = selection_metrics(final)
    paired = V.paired_probe(runtime, pairs, lookup)

    validation_report = {
        "_doc": "Task 6B final validation. Headline metric = strict end-to-end free-generation mIoU.",
        "task": "6B",
        "checkpoint": final_checkpoint,
        "baseline": report.get("baseline", {}).get("selection"),
        "after_phase_a": report.get("phases", {}).get("A_validation", {}).get("selection"),
        "final_selection": final_selection,
        "teacher_forced": final["teacher_forced"],
        "teacher_forced_breakdown_level": final["teacher_forced_breakdown_level"],
        "teacher_forced_breakdown_family": final["teacher_forced_breakdown_family"],
        "free_generation": final["free_generation"],
        "free_generation_breakdown_level": final["free_generation_breakdown_level"],
        "free_generation_breakdown_family": final["free_generation_breakdown_family"],
        "language": final["language"],
        "paired_probe": paired,
        "per_record_free_generation": final["_free_records"],
        "per_record_teacher_forced": final["_tf_records"],
        "free_generation_seconds": final["free_generation_seconds"],
        "vram": final["free_generation_peak_vram"],
    }
    write_json(VALIDATION_JSON, validation_report)

    # ---------------- checkpoint manifest ----------------
    entries = {}
    for role, name in (
        ("phaseA_last", "phaseA_last.pt"),
        ("best_mask", "best_mask.pt"),
        ("best_language", "best_language.pt"),
        ("best_joint", "best_joint.pt"),
        ("last", "last.pt"),
    ):
        path = CHECKPOINT_DIR / name
        if path.is_file():
            entries[role] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    write_json(
        CHECKPOINT_MANIFEST,
        build_manifest(
            entries,
            base_models={
                "qwen": cfg["models"]["qwen_model_id"],
                "qwen_snapshot_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
                "sam2": cfg["models"]["sam2_repo_id"],
                "sam2_source_revision": runtime.reports["sam2"].get("source_revision", ""),
            },
            extra={
                "task": "6B",
                "phase_a_steps": phase_a_steps,
                "phase_b_steps": total_b_steps,
                "best_joint_rule": list(BEST_JOINT_RULE),
                "best_joint_selection": best_joint_metrics,
            },
        ),
    )

    report["final_validation"] = final_selection
    report["paired_probe"] = {
        "passed": paired["passed"],
        "n_pairs": paired["n_pairs"],
        "mean_own_target_iou": paired["mean_own_target_iou"],
        "mean_cross_target_iou": paired["mean_cross_target_iou"],
    }
    report["peak_vram_overall"] = vram()
    report["total_seconds"] = round(sum(e["seconds"] for e in epochs_log) + report["phases"]["A"]["seconds"], 2)
    write_json(TRAINING_JSON, report)

    print(f"[task6b] final: {final_selection}")
    print(f"[task6b] paired: {paired['passed']}/{paired['n_pairs']}")
    return 0


def _packages() -> dict:
    import importlib.metadata as metadata

    names = ["torch", "transformers", "peft", "accelerate", "safetensors", "huggingface_hub", "SAM-2", "numpy"]
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            out[name] = "NOT_INSTALLED"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
