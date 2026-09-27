"""Task 6G sections 12-15: Stage G1 -- 480-paired mini-train + spatial evaluation.

Trains the Task 6G trainable set (text LoRA + `[BOX]`/`[SEG]` rows + `DenseSpatialGroundingHead`)
on the Task 6C paired `P` subset, at most 8 epochs under one corrected cosine scheduler over the
full step budget, validating every epoch through the real query-to-heatmap path on the fixed
120-record validation set and the fixed 20 paired validation images.

Model selection (section 13, lexicographic): paired point selection /20, then point-inside-target
rate, then validation heatmap Dice. Early stop after epoch 3 when the paired metric and the inside
rate fail to improve for 3 consecutive epochs. Failure of the G1 gate on the best epoch is
`DENSE_SPATIAL_GROUNDING_FAILED`.

Writes `evaluation/task6g_g1_training.json`, `evaluation/task6g_spatial_eval.json`,
`evaluation/task6g_paired_probe.json` (point side) and `evaluation/task6g_representation.json`.
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
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.dense_eval import (  # noqa: E402
    paired_point_report,
    predict_heatmap,
    representation_report,
    spatial_metrics,
)
from buildreasonseg_mvp.dense_grounding import feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6c_train import validation_material  # noqa: E402
from task6g_common import (  # noqa: E402
    EVAL,
    flat_paired_records,
    gate_report,
    load_dense_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    save_stage_checkpoint,
    validation_pair_dicts,
    write_json,
)

OUT = EVAL / "task6g_g1_training.json"
SPATIAL_OUT = EVAL / "task6g_spatial_eval.json"
PAIRED_OUT = EVAL / "task6g_paired_probe.json"
REPRESENTATION_OUT = EVAL / "task6g_representation.json"


def _selection_key(evaluation: dict) -> tuple:
    """Section 13 lexicographic: paired point selection, inside rate, then heatmap Dice."""

    metrics = evaluation["metrics"]
    return (
        int(evaluation["paired"]["paired_point_selection_pass"]),
        float(metrics["point_inside_target_rate"] or 0.0),
        float(metrics["mean_heatmap_dice"] or 0.0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, payload = load_dense_config()
    grid = int(cfg["dense_grounding"]["grid"])
    g1_cfg = cfg["g1"]
    epochs = int(args.epochs or g1_cfg["epochs"])
    gate_cfg = g1_cfg["gate"]

    records = flat_paired_records(payload)
    samples = [data_mod.to_sample(record) for record in records]
    steps_per_epoch = len(samples)
    total_steps = max(1, epochs * steps_per_epoch)

    val_samples, raw_pairs, _lookup, _audit = validation_material()
    pairs = validation_pair_dicts(raw_pairs)
    paired_samples = [
        data_mod.to_sample(pair[side]) for pair in pairs for side in ("record_a", "record_b")
    ]

    print(
        f"[task6g.g1] grid {grid}: train {len(samples)} paired records ({len(samples) // 2} "
        f"images), val {len(val_samples)}, pairs {len(pairs)}, {epochs} epochs x "
        f"{steps_per_epoch} steps = {total_steps} optimizer steps",
        flush=True,
    )

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    probe_image = samples[0].image_rgb()
    probe_features, _cached = runtime.features_for(samples[0], probe_image)
    in_channels = int(feature_tensor_for_grid(probe_features, grid).shape[1])
    runtime.install_dense_head(in_channels=in_channels)
    trainables = runtime.freeze_for_dense_grounding()
    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime,
        total_steps=total_steps,
        warmup_steps=int(cfg["optimizer"]["warmup_steps"]),
    )

    report = {
        "_doc": (
            "Task 6G sections 12-15. Stage G1: 480-record paired mini-train (Task 6C `P` subset) "
            "with the dense heatmap objective, at most 8 epochs, real-path validation every "
            "epoch, selection by paired point selection -> inside rate -> heatmap Dice. No SAM2 "
            "training, no coordinate tokens, no global box regression."
        ),
        "task": "6G",
        "stage": "G1",
        "grid": grid,
        "oracle_selection": cfg["_grid_oracle_selection"],
        "trainables": trainables,
        "head": runtime.reports["dense_head"],
        "optimizer_groups": optimizer_coverage(groups),
        "loss": {
            "reasoning_weight": float(cfg["dense_grounding"]["reasoning_loss_weight"]),
            "heatmap_weight": float(cfg["dense_grounding"]["heatmap_loss_weight"]),
            "heatmap_bce_weight": float(cfg["dense_grounding"]["heatmap_bce_weight"]),
            "heatmap_dice_weight": float(cfg["dense_grounding"]["heatmap_dice_weight"]),
            "mask_loss": False,
        },
        "budget": {
            "epochs_requested": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "scheduler_horizon": total_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
            "early_stop_patience_epochs": int(g1_cfg["early_stop_patience_epochs"]),
            "early_stop_after_epoch": int(g1_cfg["early_stop_after_epoch"]),
        },
        "validation_material": {
            "val_records": len(val_samples),
            "paired_images": len(pairs),
        },
        "epochs": [],
    }

    def evaluate(tag: str, epoch: int) -> dict:
        val_records = [predict_heatmap(runtime, sample) for sample in val_samples]
        paired_records = [predict_heatmap(runtime, sample, with_heatmap=True) for sample in paired_samples]
        paired_by_id = {record["sample_id"]: record for record in paired_records}
        metrics = spatial_metrics(val_records)
        paired = paired_point_report(paired_by_id, pairs)
        # The binary heatmaps exist only in memory for the A/B IoU; they are stripped before
        # serialization (a 256x256 bool list per record would bloat the artifact).
        for record in paired_records:
            record.pop("heatmap_binary", None)
        gate = gate_report({"val": metrics, "paired": paired}, gate_cfg, "g1")
        print(
            f"[{tag}] epoch {epoch} query path: inside {metrics['point_inside_target_rate']} "
            f"dice {metrics['mean_heatmap_dice']} binIoU {metrics['mean_heatmap_binary_iou_05']} "
            f"err512 {metrics['mean_512px_point_error']} paired "
            f"{paired['paired_point_selection_pass']}/{paired['paired_total']} same-image dist "
            f"{paired['mean_same_image_point_distance']} heatmapIoU "
            f"{paired['mean_same_image_heatmap_iou']} gate {gate['passed']}",
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
        epoch_dice_quality: list[float] = []
        for step, sample in enumerate(samples, start=1):
            image = sample.image_rgb()
            features, _cached = runtime.features_for(sample, image)
            batch = build_box_query_batch(
                runtime.processor, runtime.tokenizer, image, sample.instruction_zh,
                sample.reasoning_zh, int(runtime.box_setup.box_token_id),
                int(runtime.model.seg_token_id), append_eos=True,
            )
            runtime.set_visual_cache_key(str(sample.image_id))
            result = runtime.dense_train_step(batch, sample.target_mask(), features, optimizer=optimizer)
            scheduler.step()
            global_step += 1
            epoch_dice_quality.append(result["heatmap_dice_quality"])
            if step % int(args.log_every) == 0 or step == steps_per_epoch:
                entry = {
                    "epoch": epoch,
                    "epoch_step": step,
                    "global_step": global_step,
                    "losses": result["losses"],
                    "heatmap_dice_quality": result["heatmap_dice_quality"],
                    "lrs": lr_values(optimizer),
                }
                history.append(entry)
                print(
                    f"[task6g.g1] epoch {epoch} step {step}/{steps_per_epoch} total "
                    f"{entry['losses']['total']:.4f} reasoning {entry['losses']['reasoning_ce']:.4f} "
                    f"bce {entry['losses']['heatmap_bce']:.4f} dice {entry['losses']['heatmap_dice']:.4f} "
                    f"diceQ {entry['heatmap_dice_quality']:.4f} lr {entry['lrs'][0]:.3e}",
                    flush=True,
                )
            del result, batch, image, features

        evaluation = evaluate("task6g.g1", epoch)
        evaluation["train_heatmap_dice_quality_mean"] = float(
            sum(epoch_dice_quality) / max(len(epoch_dice_quality), 1)
        )
        checkpoint = save_stage_checkpoint(
            cfg,
            "G1",
            f"g1_epoch{epoch}",
            runtime.model,
            global_step,
            metrics={
                "epoch": epoch,
                "point_inside_target_rate": evaluation["metrics"]["point_inside_target_rate"],
                "mean_heatmap_dice": evaluation["metrics"]["mean_heatmap_dice"],
                "paired_point_selection_pass": evaluation["paired"]["paired_point_selection_pass"],
                "train_heatmap_dice_quality_mean": evaluation["train_heatmap_dice_quality_mean"],
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
            epoch >= int(g1_cfg["early_stop_after_epoch"])
            and no_improvement_streak >= int(g1_cfg["early_stop_patience_epochs"])
        ):
            stop_reason = f"early_stop_after_epoch_{epoch}"
            break

    report["history"] = history
    best = max(report["epochs"], key=_selection_key)
    report["selection"] = {
        "priority": ["paired point selection /20", "point-inside-target rate", "validation heatmap Dice"],
        "selected_epoch": int(best["epoch"]),
        "selected_checkpoint": best["checkpoint"],
        "key": list(_selection_key(best)),
        "per_epoch_keys": [
            {"epoch": int(entry["epoch"]), "key": list(_selection_key(entry))}
            for entry in report["epochs"]
        ],
        "stop_reason": stop_reason,
    }

    # ---- representation diagnosis on the BEST checkpoint (section 15) ---------
    load_report = load_checkpoint(Path(best["checkpoint"]["path"]), runtime.model)
    rep_val_records = [predict_heatmap(runtime, sample, with_hidden=True) for sample in val_samples]
    rep_paired_records = [predict_heatmap(runtime, sample, with_hidden=True) for sample in paired_samples]
    representation = representation_report(rep_paired_records, rep_val_records, pairs)

    report["final"] = {
        "epoch": int(best["epoch"]),
        "train_heatmap_dice_quality_mean": best["train_heatmap_dice_quality_mean"],
        "metrics": best["metrics"],
        "paired": best["paired"],
        "gate": best["gate"],
        "checkpoint": best["checkpoint"],
        "checkpoint_load": load_report,
    }
    report["verdict"] = "G1_GATE_PASS" if best["gate"]["passed"] else "DENSE_SPATIAL_GROUNDING_FAILED"
    report["seconds"] = round(time.time() - started, 2)

    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    write_json(
        SPATIAL_OUT,
        {
            "_doc": (
                "Task 6G section 14. Spatial localization metrics for the selected G1 model: the "
                "predicted point comes from the dense heatmap argmax only; GT is scoring only."
            ),
            "task": "6G",
            "stage": "G1_spatial",
            "grid": grid,
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
                "Task 6G paired point probe on the fixed 20 paired validation images: the two "
                "instructions' argmax points, same-image point distance and heatmap A/B IoU. The "
                "mask side is added by the G2 stage (only when G1 passes)."
            ),
            "task": "6G",
            "grid": grid,
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "geometry": best["paired"],
            "mask": None,
            "mask_ran": False,
            "mask_note": "G2 did not run (section 16 gates it on the G1 gate).",
        },
    )
    write_json(
        REPRESENTATION_OUT,
        {
            "_doc": (
                "Task 6G section 15. Query hidden / projected query / heatmap diagnostics at the "
                "best G1 checkpoint, compared with the frozen Task 6F representation. Do not "
                "interpret cosine alone."
            ),
            "task": "6G",
            "grid": grid,
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "representation": representation,
        },
    )
    print(
        f"[task6g.g1] {report['verdict']}: selected epoch {best['epoch']} inside "
        f"{best['metrics']['point_inside_target_rate']} dice {best['metrics']['mean_heatmap_dice']} "
        f"paired {best['paired']['paired_point_selection_pass']}/{best['paired']['paired_total']} "
        f"same-image point dist {best['paired']['mean_same_image_point_distance']} heatmapIoU "
        f"{best['paired']['mean_same_image_heatmap_iou']} train diceQ "
        f"{best['train_heatmap_dice_quality_mean']}",
        flush=True,
    )
    print(f"[task6g.g1] wrote {output} + spatial/paired/representation artifacts", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if best["gate"]["passed"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
