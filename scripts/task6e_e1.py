"""Task 6E sections 11-13: Stage E1 -- 480-record paired mini-train + geometry gate.

Trains only the text LoRA and the `[SEG]` / `[BOX]` / `<loc_*>` rows on the Task 6C paired
`P` subset (240 images x 2 instructions), for at most 3 epochs with the corrected scheduler
horizon, and evaluates **free generation** after every epoch on the fixed 120-record
validation set and the fixed 20 paired validation images.

Model selection priority (section 11): geometry paired pass, then validation mean predicted
box IoU, then structural validity. Failure of the gate on the selected epoch is
`EXPLICIT_SPATIAL_TOKENS_FAILED`.

Writes `evaluation/task6e_e1_minitrain.json`.
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
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_eval import (  # noqa: E402
    generate_spatial,
    geometry_report,
    paired_geometry_report,
    teacher_forced_report,
)
from buildreasonseg_mvp.spatial_training import spatial_batch  # noqa: E402

from task6c_train import validation_material  # noqa: E402
from task6e_common import (  # noqa: E402
    EVAL,
    flat_paired_records,
    gate_report,
    load_spatial_config,
    lr_values,
    optimizer_and_scheduler,
    same_image_instruction_divergence,
    save_stage_checkpoint,
    write_json,
)

OUT = EVAL / "task6e_e1_training.json"


def _pair_dicts(pairs) -> list[dict]:
    """Task 6C's `{"a": record, "b": record}` pairs -> labelled pair dictionaries."""

    result = []
    for pair in pairs:
        record_a, record_b = pair["a"], pair["b"]
        result.append(
            {
                "image_id": str(record_a["image_id"]),
                "a": str(record_a["sample_id"]),
                "b": str(record_b["sample_id"]),
                "record_a": record_a,
                "record_b": record_b,
                "a_level": int(record_a["level"]),
                "b_level": int(record_b["level"]),
                "a_query_type": str(record_a["query_type"]),
                "b_query_type": str(record_b["query_type"]),
            }
        )
    return result


def _selection_key(evaluation: dict) -> tuple:
    """Section 11 priority: geometry paired pass, then mean box IoU, then structural validity."""

    metrics = evaluation["metrics"]
    return (
        1 if evaluation["gate"]["checks"]["geometry_paired_ge"] else 0,
        float(metrics["mean_predicted_box_iou"] or 0.0),
        float(metrics["structural_valid_rate"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=120)
    parser.add_argument("--free-limit", type=int, default=None)
    parser.add_argument("--skip-teacher-forced", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, payload = load_spatial_config()
    e1_cfg = cfg["e1"]
    bins = int(cfg["spatial_tokens"]["bins"])
    epochs = int(args.epochs or e1_cfg["epochs"])
    gate_cfg = e1_cfg["gate"]

    records = flat_paired_records(payload)
    samples = [data_mod.to_sample(record) for record in records]
    steps_per_epoch = len(samples)
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, int(args.max_steps_per_epoch))
        samples = samples[:steps_per_epoch]
    total_steps = max(1, epochs * steps_per_epoch)

    val_samples, raw_pairs, _lookup, lookup_audit = validation_material()
    pairs = _pair_dicts(raw_pairs)
    if args.free_limit:
        val_samples_for_free = val_samples[: int(args.free_limit)]
    else:
        val_samples_for_free = val_samples

    print(
        f"[task6e.e1] bins {bins}: train {len(samples)} paired records "
        f"({len(samples) // 2} images), val {len(val_samples)} (free on "
        f"{len(val_samples_for_free)}), pairs {len(pairs)}, {epochs} epochs x "
        f"{steps_per_epoch} steps = {total_steps} optimizer steps",
        flush=True,
    )

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, device="cuda", verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    trainables = runtime.freeze_for_spatial_tokens()
    optimizer, scheduler = optimizer_and_scheduler(
        runtime,
        total_steps=total_steps,
        warmup_steps=int(cfg["optimizer"]["warmup_steps"]),
    )

    report = {
        "_doc": (
            "Task 6E sections 11-13. Stage E1: 480-record paired mini-train (Task 6C `P` subset) "
            "with the explicit spatial targets, at most 3 epochs, free-generation geometry "
            "evaluation after every epoch. No mask loss and no SAM2 training."
        ),
        "task": "6E",
        "stage": "E1",
        "bins": bins,
        "oracle_selection": cfg["_oracle_selection"],
        "subset_source": cfg["_subset_source"],
        "trainables": trainables,
        "loss": {
            "assistant_weight": float(cfg["spatial_tokens"]["assistant_loss_weight"]),
            "location_weight": float(cfg["spatial_tokens"]["location_loss_weight"]),
            "mask_loss": False,
        },
        "budget": {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "scheduler_horizon": total_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
            "note": (
                "Task 6D.1 section 2 corrected horizon: the cosine schedule is built over the real "
                "total optimizer-step count, so no epoch trains at LR ~0."
            ),
        },
        "validation_material": {
            "val_records": len(val_samples),
            "paired_images": len(pairs),
            "free_generation_val_records": len(val_samples_for_free),
            "lookup_audit": lookup_audit,
        },
        "epochs": [],
    }

    def evaluate(tag: str, epoch: int) -> dict:
        free_records = [generate_spatial(runtime, sample) for sample in val_samples_for_free]
        # The 20 paired images are a *disjoint* validation subset, so their generations are
        # separate from the 120-record set.
        paired_free = [
            generate_spatial(runtime, data_mod.to_sample(pair[side]))
            for pair in pairs
            for side in ("record_a", "record_b")
        ]
        parsed_by_id = {record["sample_id"]: record for record in [*free_records, *paired_free]}
        metrics = geometry_report(free_records, bins=bins)
        paired = paired_geometry_report(parsed_by_id, pairs)
        gate = gate_report({**metrics, "paired": paired}, gate_cfg, "e1")
        print(
            f"[{tag}] epoch {epoch} free: structural {metrics['structural_valid']}/{metrics['count']} "
            f"({metrics['structural_valid_rate']:.3f}) exact {metrics['exact_four_token_sequence']} "
            f"boxIoU {metrics['mean_predicted_box_iou']} center_inside {metrics['center_inside_rate']} "
            f"paired geom {paired['geometry_paired_pass']}/{paired['paired_total']} "
            f"same-image L1 {paired['mean_same_image_predicted_box_l1']} gate {gate['passed']}",
            flush=True,
        )
        evaluation = {
            "epoch": epoch,
            "metrics": metrics,
            "paired": paired,
            "gate": gate,
            "free_generation_records": free_records,
            "paired_free_generation_records": paired_free,
        }
        if not args.skip_teacher_forced:
            evaluation["teacher_forced"] = teacher_forced_report(runtime, val_samples)
        return evaluation

    global_step = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        set_seed(int(cfg["seed"]) + epoch)
        for step, sample in enumerate(samples, start=1):
            batch, example = spatial_batch(runtime, sample, image=sample.image_rgb())
            runtime.set_visual_cache_key(str(sample.image_id))
            result = runtime.spatial_train_step(batch, example, optimizer=optimizer)
            scheduler.step()
            global_step += 1
            if step % int(args.log_every) == 0 or step == steps_per_epoch:
                entry = {
                    "epoch": epoch,
                    "epoch_step": step,
                    "global_step": global_step,
                    "losses": result["losses"],
                    "location_token_accuracy": result["location_token_accuracy"],
                    "location_abs_bin_error": result["location_abs_bin_error"],
                    "lrs": lr_values(optimizer),
                }
                history.append(entry)
                print(
                    f"[task6e.e1] epoch {epoch} step {step}/{steps_per_epoch} total "
                    f"{entry['losses']['total']:.4f} assistant {entry['losses']['assistant_ce']:.4f} "
                    f"location {entry['losses']['location_ce']:.4f} loc_acc "
                    f"{entry['location_token_accuracy']:.3f} lr {entry['lrs'][0]:.3e}",
                    flush=True,
                )
            del result, batch, example
        evaluation = evaluate("task6e.e1", epoch)
        evaluation["seconds"] = round(time.time() - started, 2)
        checkpoint = save_stage_checkpoint(
            cfg,
            "E1",
            f"e1_epoch{epoch}",
            runtime.model,
            global_step,
            metrics={
                "epoch": epoch,
                "structural_valid_rate": evaluation["metrics"]["structural_valid_rate"],
                "mean_predicted_box_iou": evaluation["metrics"]["mean_predicted_box_iou"],
                "geometry_paired_pass": evaluation["paired"]["geometry_paired_pass"],
            },
        )
        evaluation["checkpoint"] = checkpoint
        report["epochs"].append(evaluation)

    report["history"] = history
    best = max(report["epochs"], key=_selection_key)
    report["selection"] = {
        "priority": [
            "geometry paired pass",
            "validation mean predicted box IoU",
            "structural validity",
        ],
        "selected_epoch": int(best["epoch"]),
        "selected_checkpoint": best["checkpoint"],
        "key": list(_selection_key(best)),
        "per_epoch_keys": [
            {"epoch": int(entry["epoch"]), "key": list(_selection_key(entry))}
            for entry in report["epochs"]
        ],
    }
    report["final"] = {
        "epoch": int(best["epoch"]),
        "metrics": best["metrics"],
        "paired": best["paired"],
        "gate": best["gate"],
        "teacher_forced": best.get("teacher_forced"),
        "checkpoint": best["checkpoint"],
    }
    report["verdict"] = (
        "E1_GEOMETRY_GATE_PASS" if best["gate"]["passed"] else "EXPLICIT_SPATIAL_TOKENS_FAILED"
    )
    report["seconds"] = round(time.time() - started, 2)
    output = Path(args.output) if args.output else OUT
    write_json(output, report)

    divergence = same_image_instruction_divergence(best["paired_free_generation_records"], pairs)
    geometry_artifact = {
        "_doc": (
            "Task 6E sections 12-13. Free-generation geometry for the selected E1 model: token-id "
            "parse (never text), structural validity, exact four-token accuracy, per-coordinate token "
            "accuracy, box IoU, center-inside rate, and the paired same-image probe. Teacher-forced "
            "token accuracy is diagnostic only and is recorded separately."
        ),
        "task": "6E",
        "stage": "E1_geometry",
        "bins": bins,
        "selected_epoch": int(best["epoch"]),
        "checkpoint": best["checkpoint"],
        "oracle_selection": cfg["_oracle_selection"],
        "metrics": best["metrics"],
        "paired": best["paired"],
        "same_image_instruction_divergence": divergence,
        "teacher_forced_diagnostic": best.get("teacher_forced"),
        "free_generation_records": best["free_generation_records"],
        "paired_free_generation_records": best["paired_free_generation_records"],
        "comparisons": {
            "continuous_oracle_box_miou": 0.7506,
            "selected_quantized_oracle_miou": json.loads(
                (EVAL / "task6e_quantized_oracle.json").read_text(encoding="utf-8")
            )["candidates"][str(bins)]["strict_mask_miou"],
            "task6c_p_c_mask_miou": 0.10604,
            "task6d_g0_corrected_geometry_paired": 0,
        },
    }
    write_json(EVAL / "task6e_geometry_eval.json", geometry_artifact)
    print(
        f"[task6e.e1] {report['verdict']}: selected epoch {best['epoch']} structural "
        f"{best['metrics']['structural_valid_rate']:.3f} mean box IoU "
        f"{best['metrics']['mean_predicted_box_iou']} paired "
        f"{best['paired']['geometry_paired_pass']}/{best['paired']['paired_total']} "
        f"same-image L1 {best['paired']['mean_same_image_predicted_box_l1']} "
        f"different-token pairs {divergence['pairs_with_different_tokens']}/"
        f"{divergence['compared_pairs']}",
        flush=True,
    )
    print(f"[task6e.e1] wrote {output} and evaluation/task6e_geometry_eval.json", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if best["gate"]["passed"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
