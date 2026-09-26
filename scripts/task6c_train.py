#!/usr/bin/env python
"""Task 6C: train and evaluate one arm of the 2x2 ablation.

    python scripts/task6c_train.py --arm U_C [--max-steps N] [--epochs N] [--smoke]

Arms (Task 6C section 7):

    U_C  unpaired 480 unique images     centre-positive SAM bridge
    U_L  unpaired 480 unique images     language-only SAM bridge
    P_C  240 paired images x 2          centre-positive SAM bridge
    P_L  240 paired images x 2          language-only SAM bridge

Every arm shares the base snapshot, the seed, the recipe, the loss weights, the
validation set and the metric code; only the training subset and the SAM bridge
differ (Task 6C section 9: fixed recipe, no tuning).

Writes:
    evaluation/task6c_<ARM>.json                 primary + teacher-forced + paired
    evaluation/task6c_prompt_<ARM>.json          prompt-pathway diagnosis
    evaluation/task6c_initialization.json        merged initial-trainable fingerprints
    evaluation/task6c_checkpoint_manifest.json   merged checkpoint hashes
    artifacts/checkpoints/task6c/<ARM>/           per-arm checkpoints (gitignored)
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
from buildreasonseg_mvp import language_metrics as LM  # noqa: E402
from buildreasonseg_mvp import prompt_diagnostics as PD  # noqa: E402
from buildreasonseg_mvp import validation as V  # noqa: E402
from buildreasonseg_mvp.checkpointing import (  # noqa: E402
    save_checkpoint,
    sha256_file,
    write_json,
)
from buildreasonseg_mvp.determinism import trainable_state_fingerprint  # noqa: E402
from buildreasonseg_mvp.runtime import (  # noqa: E402
    build_runtime,
    load_config,
    reset_peak,
    set_phase_trainables,
    vram,
)
from buildreasonseg_mvp.subsets import _ids_sha256  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
TASK6C_SUBSETS = EVAL / "task6c_subset_ids.json"
TASK6B_SUBSETS = EVAL / "task6b_subset_ids.json"
INIT_JSON = EVAL / "task6c_initialization.json"
MANIFEST_JSON = EVAL / "task6c_checkpoint_manifest.json"

ARMS = {
    "U_C": {"subset": "U", "bridge": "centre"},
    "U_L": {"subset": "U", "bridge": "language"},
    "P_C": {"subset": "P", "bridge": "centre"},
    "P_L": {"subset": "P", "bridge": "language"},
}


def checkpoint_dir(cfg: dict, arm: str) -> Path:
    """Task 6C section 4.2: the arm directory comes from the config, never a constant."""

    root = Path(cfg["paths"]["checkpoints"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / arm


def samples_from_ids(records_by_id: dict[str, dict], sample_ids) -> list:
    return [data_mod.to_sample(records_by_id[sample_id]) for sample_id in sample_ids]


def subset_records(payload: dict, key: str) -> list[dict]:
    train = data_mod.read_records(payload["source_split"] if "source_split" in payload else "train")
    by_id = {record["sample_id"]: record for record in train}
    return [by_id[sample_id] for sample_id in payload[key]["sample_ids"]]


def validation_material() -> tuple[list, list[dict], dict, dict]:
    """Exactly Task 6B's validation material (section 10), plus a train-only lookup."""

    payload = json.loads(TASK6B_SUBSETS.read_text(encoding="utf-8"))
    val_records = {r["sample_id"]: r for r in data_mod.read_records("val")}
    val_samples = samples_from_ids(val_records, payload["val"]["sample_ids"])

    paired_ids = payload["paired_probe"]["sample_ids"]
    paired_records = [val_records[sid] for sid in paired_ids if sid in val_records]
    pairs = [
        {"a": paired_records[i], "b": paired_records[i + 1]}
        for i in range(0, len(paired_records), 2)
    ]

    # Task 6C section 4.3: the accepted-wording lookup must not be built from val
    # text. Only the train split contributes.
    train_records = list(data_mod.read_records("train"))
    lookup = LM.build_reasoning_lookup(train_records)
    lookup_audit = {
        "built_from": "train split only",
        "train_records": len(train_records),
        "distinct_accepted_reasonings": len(lookup),
        "val_text_used": False,
    }
    return val_samples, pairs, lookup, lookup_audit


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


def collect_grad_norms_setting(runtime) -> bool:
    """Task 6C.6 section 1: does the formal joint-training path collect gradient norms?

    Task 6C.5 measured the per-step, per-parameter gradient-norm sweep over 528 tensors
    at +10.24 % throughput when removed, and proved the removal bit-equivalent. This
    loop never reads `result["grad_norms"]`, so the sweep was pure diagnostic cost.

    The switch is configuration-controlled (`training.collect_grad_norms`) and defaults
    to `True`, so every other caller keeps the behaviour it had. Stage-1 / smoke
    diagnostics that genuinely consume the per-group dictionary can set it back to
    `true` in their own config, or call `runtime.train_step` without the argument.

    `clip_grad_norm_` is unaffected: it still runs inside `train_step` with the same
    threshold and the same parameter groups.
    """

    return bool(runtime.cfg.get("training", {}).get("collect_grad_norms", True))


def train_phase(runtime, samples, phase: str, steps: int, log_every: int, tag: str) -> dict:
    optimizer, scheduler, groups = make_optimizer(runtime, phase, steps)
    history: list[dict] = []
    step = 0
    nan_seen = False
    started = time.time()
    is_language_only = phase.upper() == "A"
    # Task 6C.6 section 1: the formal path passes the configured value explicitly.
    collect_grad_norms = collect_grad_norms_setting(runtime)

    while step < steps:
        sample = samples[step % len(samples)]
        batch, image = runtime.prepare(sample)
        moved = batch.to(runtime.device)
        if is_language_only:
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
            del output
        else:
            gt_mask = sample.target_mask()
            features, _cached = runtime.features_for(sample, image)
            result = runtime.train_step(
                batch,
                gt_mask,
                features,
                optimizer=optimizer,
                collect_grad_norms=collect_grad_norms,
            )
            losses = result["losses"]
            clipped = result["grad_clip_total_norm"]
            del result, features

        if not np.isfinite(list(losses.values())).all():
            nan_seen = True
            break
        if scheduler is not None:
            scheduler.step()
        step += 1
        del batch, moved

        if step % log_every == 0 or step == steps:
            history.append(
                {
                    "step": step,
                    "phase": phase,
                    "elapsed_seconds": round(time.time() - started, 1),
                    "losses": losses,
                    "grad_clip_total_norm": clipped,
                    "vram": vram(),
                }
            )
            print(
                f"[{tag}] step {step}/{steps} total {losses['total']:.4f} lm {losses['lm_ce']:.4f} "
                f"bce {losses['mask_bce']:.4f} dice {losses['mask_dice']:.4f}",
                flush=True,
            )

    return {
        "phase": phase,
        "steps": step,
        "nan_seen": nan_seen,
        "seconds": round(time.time() - started, 2),
        # Task 6C.6 section 1: recorded so a run's report states which diagnostic mode
        # it actually used.
        "collect_grad_norms": collect_grad_norms,
        "optimizer_groups": [
            {"lr": g["lr"], "weight_decay": g["weight_decay"], "name": g.get("name"), "tensors": len(g["params"])}
            for g in groups
        ],
        "history": history,
        "feature_cache": runtime.feature_cache.stats(),
        "peak_vram": vram(),
    }


def full_validation(runtime, val_samples, lookup) -> dict:
    tf = V.teacher_forced_validation(runtime, val_samples, reasoning_lookup=lookup)
    result = {
        "teacher_forced": V.aggregate_teacher_forced(tf.records),
        "teacher_forced_breakdown_level": V.teacher_forced_breakdown(tf.records, "level_key"),
        "teacher_forced_breakdown_family": V.teacher_forced_breakdown(tf.records, "family"),
        "teacher_forced_seconds": round(tf.seconds, 2),
    }
    fg = V.free_generation_validation(runtime, val_samples, lookup)
    result.update(
        {
            "free_generation": V.aggregate(fg.records),
            "free_generation_breakdown_level": V.breakdown(fg.records, "level_key"),
            "free_generation_breakdown_family": V.breakdown(fg.records, "family"),
            "free_generation_seconds": round(fg.seconds, 2),
            "language": V.language_summary(fg.records, tf.records),
            "per_record_free_generation": fg.records,
            "per_record_teacher_forced": tf.records,
        }
    )
    return result


def selection_metrics(result: dict) -> dict:
    fg = result.get("free_generation") or {}
    tf = result.get("teacher_forced") or {}
    return {
        "valid_seg_emission_rate": fg.get("valid_seg_emission_rate"),
        "strict_end_to_end_miou": fg.get("strict_end_to_end_miou"),
        "strict_end_to_end_dice": fg.get("strict_end_to_end_dice"),
        "conditional_miou": fg.get("conditional_miou"),
        "teacher_forced_miou": tf.get("miou"),
        "operation_chain_accuracy": fg.get("operation_chain_accuracy"),
        "lm_ce": tf.get("lm_ce"),
    }


def merge_json(path: Path, key: str, payload) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data[key] = payload
    write_json(path, data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Phase B step cap override (smoke only)")
    parser.add_argument("--phase-a-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-freegen", action="store_true")
    parser.add_argument("--skip-diagnosis", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="do not write checkpoints (smoke mode)")
    parser.add_argument("--val-limit", type=int, default=None, help="smoke only: cap validation records")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    arm = args.arm
    spec = ARMS[arm]
    cfg = load_config(args.config or (REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"))
    cfg["bridge"] = {"mode": spec["bridge"]}

    subset_payload = json.loads(TASK6C_SUBSETS.read_text(encoding="utf-8"))
    train_records = subset_records(subset_payload, spec["subset"])
    train_samples = [data_mod.to_sample(record) for record in train_records]
    val_samples, pairs, lookup, lookup_audit = validation_material()
    if args.val_limit:
        val_samples = val_samples[: args.val_limit]
        pairs = pairs[: max(1, args.val_limit // 4)]

    print(
        f"[task6c] arm {arm}: subset {spec['subset']} ({len(train_samples)} records), "
        f"bridge {spec['bridge']}, val {len(val_samples)}, pairs {len(pairs)}",
        flush=True,
    )

    runtime = build_runtime(cfg, device="cuda", verbose=True)
    determinism = runtime.reports["determinism"]
    arm_dir = checkpoint_dir(cfg, arm)
    arm_dir.mkdir(parents=True, exist_ok=True)
    print(f"[task6c] arm {arm}: checkpoint dir {arm_dir}", flush=True)

    # ---- section 8: initialization fingerprint, before the first optimizer step ----
    fingerprint = trainable_state_fingerprint(runtime.model)
    merge_json(
        INIT_JSON,
        arm,
        {
            "arm": arm,
            "subset": spec["subset"],
            "bridge": spec["bridge"],
            "seed": int(cfg["seed"]),
            "initial_trainable_sha256": fingerprint["sha256"],
            "tensor_count": fingerprint["tensor_count"],
            "numel": fingerprint["numel"],
            "from_clean_base": True,
        },
    )
    print(f"[task6c] initial trainable fingerprint {fingerprint['sha256'][:16]}… "
          f"({fingerprint['tensor_count']} tensors, {fingerprint['numel']} params)", flush=True)

    phase_a_steps = int(
        args.phase_a_steps
        if args.phase_a_steps is not None
        else cfg["phase_a"]["epochs"] * len(train_samples)
    )
    epochs = int(args.epochs if args.epochs is not None else cfg["phase_b"]["epochs"])
    max_steps = int(args.max_steps if args.max_steps is not None else cfg["phase_b"]["max_steps"])
    steps_per_epoch = len(train_samples)
    log_a = int(cfg["phase_a"]["log_every_steps"])
    log_b = int(cfg["phase_b"]["log_every_steps"])

    report: dict = {
        "_doc": (
            "Task 6C arm result. Fixed Task 6B headline recipe, no tuning; only the training subset "
            "(U/P) and the SAM sparse-prompt bridge (C/L) differ between arms."
        ),
        "task": "6C",
        "arm": arm,
        "subset": spec["subset"],
        "bridge": spec["bridge"],
        "config": cfg,
        "determinism": determinism,
        "lookup_audit": lookup_audit,
        "subset_sizes": {"train_records": len(train_samples), "val_records": len(val_samples), "pairs": len(pairs)},
        "trainable_params_initial": runtime.reports["params"],
        "initialization_fingerprint": {
            "sha256": fingerprint["sha256"],
            "tensor_count": fingerprint["tensor_count"],
            "numel": fingerprint["numel"],
        },
        "started_from_clean_base": True,
        "subset_sample_ids_sha256": _ids_sha256([s.sample_id for s in train_samples]),
        "phases": {},
    }

    report["phases"]["A_trainables"] = set_phase_trainables(runtime.model, "A")
    report["phases"]["A"] = train_phase(runtime, train_samples, "A", phase_a_steps, log_a, f"{arm}.A")
    after_a = full_validation(runtime, val_samples, lookup) if not args.skip_freegen else None
    if after_a is not None:
        report["phases"]["A_validation"] = {
            "teacher_forced": after_a["teacher_forced"],
            "free_generation": after_a["free_generation"],
        }
    save_checkpoint(arm_dir / "phaseA_last.pt", runtime.model, phase_a_steps, metrics={}, config=cfg)

    report["phases"]["B_trainables"] = set_phase_trainables(runtime.model, "B")
    total_b_steps = 0
    epoch_log: list[dict] = []
    for epoch in range(1, epochs + 1):
        if total_b_steps >= max_steps:
            break
        steps = min(steps_per_epoch, max_steps - total_b_steps)
        result = train_phase(runtime, train_samples, "B", steps, log_b, f"{arm}.B{epoch}")
        total_b_steps += result["steps"]
        tf = V.teacher_forced_validation(runtime, val_samples, reasoning_lookup=lookup)
        epoch_log.append(
            {
                "epoch": epoch,
                "steps_this_epoch": result["steps"],
                "total_phase_b_steps": total_b_steps,
                "seconds": result["seconds"],
                "nan_seen": result["nan_seen"],
                "teacher_forced": V.aggregate_teacher_forced(tf.records),
                "feature_cache": result["feature_cache"],
                "history": result["history"],
            }
        )
        print(
            f"[task6c] {arm} epoch {epoch}: tf mIoU {epoch_log[-1]['teacher_forced']['miou']:.5f} "
            f"lm_ce {epoch_log[-1]['teacher_forced']['lm_ce']:.5f}",
            flush=True,
        )
        if not args.smoke:
            save_checkpoint(arm_dir / "last.pt", runtime.model, total_b_steps, metrics={}, config=cfg)
    report["phases"]["B_epochs"] = epoch_log
    report["phase_b_total_steps"] = total_b_steps

    reset_peak()
    final = full_validation(runtime, val_samples, lookup)
    paired = V.paired_probe(runtime, pairs, lookup)
    report["final_selection"] = selection_metrics(final)
    report.update(
        {
            "teacher_forced": final["teacher_forced"],
            "teacher_forced_breakdown_level": final["teacher_forced_breakdown_level"],
            "teacher_forced_breakdown_family": final["teacher_forced_breakdown_family"],
            "free_generation": final["free_generation"],
            "free_generation_breakdown_level": final["free_generation_breakdown_level"],
            "free_generation_breakdown_family": final["free_generation_breakdown_family"],
            "language": final["language"],
            "paired_probe": paired,
            "per_record_free_generation": final["per_record_free_generation"],
            "per_record_teacher_forced": final["per_record_teacher_forced"],
            "peak_vram": vram(),
            "feature_cache_final": runtime.feature_cache.stats(),
        }
    )
    print(f"[task6c] {arm} final: {report['final_selection']}", flush=True)
    print(f"[task6c] {arm} paired: {paired['passed']}/{paired['n_pairs']} "
          f"margin {paired['mean_own_minus_cross_margin']}", flush=True)

    # ---- section 13: prompt-pathway diagnosis, identical for every arm ----
    diagnosis = None
    if not args.skip_diagnosis:
        diagnosis = PD.diagnose(runtime, pairs, n_same_image=10, n_cross_image=10)
        diagnosis.update(
            {
                "arm": arm,
                "subset": spec["subset"],
                "bridge": spec["bridge"],
                "same_image_metric_note": (
                    "cosine is on a different-image reference at the same stage; a representation is "
                    "only called collapsed together with the SVD/effective-rank numbers"
                ),
            }
        )
        write_json(EVAL / f"task6c_prompt_{arm}.json", diagnosis)
        print(f"[task6c] {arm} prompt diagnosis written", flush=True)

    final_checkpoint = arm_dir / "last.pt"
    checkpoint_entry = None
    if final_checkpoint.is_file():
        checkpoint_entry = {
            "path": str(final_checkpoint),
            "bytes": final_checkpoint.stat().st_size,
            "sha256": sha256_file(final_checkpoint),
            "phase_b_steps": total_b_steps,
        }
    report["checkpoint"] = checkpoint_entry

    write_json(EVAL / f"task6c_{arm}.json", report)
    merge_json(MANIFEST_JSON, arm, checkpoint_entry or {"note": "smoke run: no checkpoint written"})
    merge_json(
        MANIFEST_JSON,
        "_base_models",
        {
            "qwen": cfg["models"]["qwen_model_id"],
            "qwen_snapshot_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "sam2": cfg["models"]["sam2_repo_id"],
            "sam2_source_revision": runtime.reports["sam2"].get("source_revision", ""),
            "task": "6C",
            "arms": {name: {"subset": value["subset"], "bridge": value["bridge"]} for name, value in ARMS.items()},
        },
    )
    print(f"[task6c] {arm} done in {sum(e['seconds'] for e in epoch_log) + report['phases']['A']['seconds']:.0f}s "
          f"of optimisation", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
