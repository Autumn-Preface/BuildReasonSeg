"""Task 6F sections 10-14: Stage F1 -- 480-paired mini-train + geometry evaluation.

Trains the Task 6F trainable set (text LoRA + `[BOX]`/`[SEG]` rows + `TargetAwareBoxHead`) on the
Task 6C paired `P` subset (480 records / 240 images x 2), at most 8 epochs under one corrected
cosine scheduler over the full step budget, and validates every epoch through the inference-form
query path on the fixed 120-record validation set and the fixed 20 paired validation images.

Model selection priority (section 11): geometry paired /20, then mean val box IoU, then
center-inside-target rate. Early stop after epoch 3 when both selection metrics fail to improve
for 3 consecutive epochs. Failure of the F1 gate on the best epoch is `TARGET_AWARE_QUERY_FAILED`.

Writes `evaluation/task6f_f1_training.json`, `evaluation/task6f_geometry_eval.json`,
`evaluation/task6f_paired_probe.json` (geometry side), `evaluation/task6f_representation.json` and
`evaluation/task6f_reasoning_compat.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.box_query import build_box_query_batch  # noqa: E402
from buildreasonseg_mvp.box_query_eval import (  # noqa: E402
    box_record,
    geometry_report,
    paired_geometry_report,
    reasoning_compat_report,
    representation_report,
)
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6c_train import validation_material  # noqa: E402
from task6f_common import (  # noqa: E402
    EVAL,
    flat_paired_records,
    gate_report,
    load_box_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    save_stage_checkpoint,
    validation_pair_dicts,
    write_json,
)

OUT = EVAL / "task6f_f1_training.json"
GEOMETRY_OUT = EVAL / "task6f_geometry_eval.json"
PAIRED_OUT = EVAL / "task6f_paired_probe.json"
REPRESENTATION_OUT = EVAL / "task6f_representation.json"
REASONING_OUT = EVAL / "task6f_reasoning_compat.json"
REASONING_SAMPLES = 20


def _selection_key(evaluation: dict) -> tuple:
    """Section 11 priority: geometry paired /20, then mean val box IoU, then center-inside."""

    metrics = evaluation["metrics"]
    return (
        int(evaluation["paired"]["geometry_paired_pass"]),
        float(metrics["mean_box_iou"] or 0.0),
        float(metrics["center_inside_rate"] or 0.0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, payload = load_box_config()
    f1_cfg = cfg["f1"]
    epochs = int(args.epochs or f1_cfg["epochs"])
    gate_cfg = f1_cfg["gate"]

    records = flat_paired_records(payload)
    samples = [data_mod.to_sample(record) for record in records]
    steps_per_epoch = len(samples)
    total_steps = max(1, epochs * steps_per_epoch)

    val_samples, raw_pairs, lookup, lookup_audit = validation_material()
    pairs = validation_pair_dicts(raw_pairs)
    paired_samples = [
        data_mod.to_sample(pair[side]) for pair in pairs for side in ("record_a", "record_b")
    ]
    reasoning_samples = val_samples[:REASONING_SAMPLES]

    print(
        f"[task6f.f1] train {len(samples)} paired records ({len(samples) // 2} images), "
        f"val {len(val_samples)}, pairs {len(pairs)}, {epochs} epochs x {steps_per_epoch} "
        f"steps = {total_steps} optimizer steps",
        flush=True,
    )

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    runtime.install_box_head(
        hidden_dim=int(cfg["box_query"]["head_hidden_dim"]),
        mid_dim=int(cfg["box_query"]["head_mid_dim"]),
    )
    trainables = runtime.freeze_for_box_query()
    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime,
        total_steps=total_steps,
        warmup_steps=int(cfg["optimizer"]["warmup_steps"]),
    )

    report = {
        "_doc": (
            "Task 6F sections 10-14. Stage F1: 480-record paired mini-train (Task 6C `P` subset) "
            "with the box-query objective, at most 8 epochs, inference-form query-path validation "
            "every epoch, selection by paired -> box IoU -> center-inside. No mask loss, no SAM2 "
            "training, no Task 6E coordinate-token machinery."
        ),
        "task": "6F",
        "stage": "F1",
        "trainables": trainables,
        "head": runtime.reports["box_head"],
        "optimizer_groups": optimizer_coverage(groups),
        "loss": {
            "reasoning_weight": float(cfg["box_query"]["reasoning_loss_weight"]),
            "box_weight": float(cfg["box_query"]["box_loss_weight"]),
            "mask_loss": False,
        },
        "budget": {
            "epochs_requested": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "scheduler_horizon": total_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
            "early_stop_patience_epochs": int(f1_cfg["early_stop_patience_epochs"]),
            "early_stop_after_epoch": int(f1_cfg["early_stop_after_epoch"]),
        },
        "validation_material": {
            "val_records": len(val_samples),
            "paired_images": len(pairs),
            "lookup_audit": lookup_audit,
            "reasoning_compat_samples": REASONING_SAMPLES,
        },
        "epochs": [],
    }

    def evaluate(tag: str, epoch: int) -> dict:
        val_records = [box_record(runtime, sample, with_hidden=False) for sample in val_samples]
        paired_records = [box_record(runtime, sample, with_hidden=False) for sample in paired_samples]
        paired_by_id = {record["sample_id"]: record for record in paired_records}
        metrics = geometry_report(val_records)
        paired = paired_geometry_report(paired_by_id, pairs)
        gate = gate_report({"val": metrics, "paired": paired}, gate_cfg, "f1")
        print(
            f"[{tag}] epoch {epoch} query path: val boxIoU {metrics['mean_box_iou']} "
            f"median {metrics['median_box_iou']} center_inside {metrics['center_inside_rate']} "
            f"coord MAE {metrics['coordinate_mae']} paired {paired['geometry_paired_pass']}/"
            f"{paired['paired_total']} same-image L1 {paired['mean_same_image_predicted_box_l1']} "
            f"gate {gate['passed']}",
            flush=True,
        )
        return {
            "epoch": epoch,
            "metrics": metrics,
            "paired": paired,
            "gate": gate,
            "val_records": val_records,
            "paired_records": paired_records,
        }

    global_step = 0
    history: list[dict] = []
    best_key: tuple | None = None
    no_improvement_streak = 0
    stop_reason = "budget_exhausted"
    for epoch in range(1, epochs + 1):
        set_seed(int(cfg["seed"]) + epoch)
        epoch_box_ious: list[float] = []
        for step, sample in enumerate(samples, start=1):
            image = sample.image_rgb()
            batch = build_box_query_batch(
                runtime.processor,
                runtime.tokenizer,
                image,
                sample.instruction_zh,
                sample.reasoning_zh,
                int(runtime.box_setup.box_token_id),
                int(runtime.model.seg_token_id),
                append_eos=True,
            )
            runtime.set_visual_cache_key(str(sample.image_id))
            result = runtime.box_query_train_step(batch, sample.target_mask(), optimizer=optimizer)
            scheduler.step()
            global_step += 1
            epoch_box_ious.append(result["box_iou"])
            if step % int(args.log_every) == 0 or step == steps_per_epoch:
                entry = {
                    "epoch": epoch,
                    "epoch_step": step,
                    "global_step": global_step,
                    "losses": result["losses"],
                    "box_iou": result["box_iou"],
                    "lrs": lr_values(optimizer),
                }
                history.append(entry)
                print(
                    f"[task6f.f1] epoch {epoch} step {step}/{steps_per_epoch} total "
                    f"{entry['losses']['total']:.4f} reasoning {entry['losses']['reasoning_ce']:.4f} "
                    f"box {entry['losses']['box_loss']:.4f} boxIoU {entry['box_iou']:.4f} "
                    f"lr {entry['lrs'][0]:.3e}",
                    flush=True,
                )
            del result, batch, image

        evaluation = evaluate("task6f.f1", epoch)
        evaluation["train_box_iou_mean"] = float(sum(epoch_box_ious) / max(len(epoch_box_ious), 1))
        checkpoint = save_stage_checkpoint(
            cfg,
            "F1",
            f"f1_epoch{epoch}",
            runtime.model,
            global_step,
            metrics={
                "epoch": epoch,
                "mean_val_box_iou": evaluation["metrics"]["mean_box_iou"],
                "center_inside_rate": evaluation["metrics"]["center_inside_rate"],
                "geometry_paired_pass": evaluation["paired"]["geometry_paired_pass"],
                "train_box_iou_mean": evaluation["train_box_iou_mean"],
            },
        )
        evaluation["checkpoint"] = checkpoint
        report["epochs"].append(evaluation)

        key = _selection_key(evaluation)
        if best_key is None or key > best_key:
            best_key = key
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
        if (
            epoch >= int(f1_cfg["early_stop_after_epoch"])
            and no_improvement_streak >= int(f1_cfg["early_stop_patience_epochs"])
        ):
            stop_reason = f"early_stop_after_epoch_{epoch}"
            break

    report["history"] = history
    best = max(report["epochs"], key=_selection_key)
    report["selection"] = {
        "priority": ["geometry paired /20", "mean val box IoU", "center-inside-target rate"],
        "selected_epoch": int(best["epoch"]),
        "selected_checkpoint": best["checkpoint"],
        "key": list(_selection_key(best)),
        "per_epoch_keys": [
            {"epoch": int(entry["epoch"]), "key": list(_selection_key(entry))}
            for entry in report["epochs"]
        ],
        "stop_reason": stop_reason,
    }

    # ---- diagnostics on the BEST checkpoint (section 13-14) -------------------
    load_report = load_checkpoint(Path(best["checkpoint"]["path"]), runtime.model)
    # The representation diagnosis needs the 2048-d query hiddens, which are re-collected from
    # the loaded best checkpoint (the per-epoch records deliberately omit them to keep the
    # training artifact small).
    rep_val_records = [box_record(runtime, sample, with_hidden=True) for sample in val_samples]
    rep_paired_records = [box_record(runtime, sample, with_hidden=True) for sample in paired_samples]
    representation = representation_report(rep_paired_records, rep_val_records, pairs)
    reasoning = reasoning_compat_report(
        runtime, reasoning_samples, lookup,
        max_new_tokens=int(cfg["inference"]["validation_max_new_tokens"]),
    )

    report["final"] = {
        "epoch": int(best["epoch"]),
        "train_box_iou_mean": best["train_box_iou_mean"],
        "metrics": best["metrics"],
        "paired": best["paired"],
        "gate": best["gate"],
        "checkpoint": best["checkpoint"],
        "checkpoint_load": load_report,
    }
    report["verdict"] = "F1_GEOMETRY_GATE_PASS" if best["gate"]["passed"] else "TARGET_AWARE_QUERY_FAILED"
    report["seconds"] = round(time.time() - started, 2)

    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    write_json(
        GEOMETRY_OUT,
        {
            "_doc": (
                "Task 6F section 12. Free-generation-free geometry metrics for the selected F1 "
                "model through the inference-form query path: the predicted box comes from "
                "[BOX] hidden -> TargetAwareBoxHead only."
            ),
            "task": "6F",
            "stage": "F1_geometry",
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "metrics": best["metrics"],
            "paired": best["paired"],
            "val_records": best["val_records"],
            "paired_records": best["paired_records"],
        },
    )
    write_json(
        PAIRED_OUT,
        {
            "_doc": (
                "Task 6F paired probe on the fixed 20 paired validation images: the two "
                "instructions' predicted boxes, own vs cross box IoU and the same-image box L1. "
                "The mask side is added by the F2 stage (only when F1 passes)."
            ),
            "task": "6F",
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "geometry": best["paired"],
            "mask": None,
            "mask_ran": False,
            "mask_note": "F2 did not run (section 15 gates it on the F1 gate).",
        },
    )
    write_json(
        REPRESENTATION_OUT,
        {
            "_doc": (
                "Task 6F section 13. [BOX]-query hidden representation diagnostics at the best F1 "
                "checkpoint, compared against the frozen Task 6D.1 legacy [SEG] values. Do not "
                "infer causality from cosine alone."
            ),
            "task": "6F",
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "representation": representation,
        },
    )
    write_json(
        REASONING_OUT,
        {
            "_doc": (
                "Task 6F section 14. Secondary language compatibility: continuing generation from "
                "image + instruction + fixed [BOX] on the first 20 records of the fixed validation "
                "set. Template diagnostics, not proof of reasoning."
            ),
            "task": "6F",
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "reasoning_compat": reasoning,
        },
    )
    print(
        f"[task6f.f1] {report['verdict']}: selected epoch {best['epoch']} val box IoU "
        f"{best['metrics']['mean_box_iou']} center_inside {best['metrics']['center_inside_rate']} "
        f"paired {best['paired']['geometry_paired_pass']}/{best['paired']['paired_total']} "
        f"same-image L1 {best['paired']['mean_same_image_predicted_box_l1']} train box IoU "
        f"{best['train_box_iou_mean']}; reasoning compat: exactly-one-[SEG] "
        f"{reasoning['exactly_one_seg_rate']}, EOS {reasoning['eos_termination_rate']}",
        flush=True,
    )
    print(f"[task6f.f1] wrote {output} + geometry/paired/representation/reasoning artifacts", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if best["gate"]["passed"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
