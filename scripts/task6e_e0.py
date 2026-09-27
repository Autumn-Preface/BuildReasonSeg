"""Task 6E section 10: Stage E0 -- 20-sample implementation sanity overfit.

Trains on 20 deterministic records (10 same-image/different-target pairs) with the explicit
spatial targets, then measures the **free-generation** gate:

* structural validity >= 19/20;
* exact four-loc-token sequence >= 18/20;
* mean dequantized box IoU >= 0.70.

Failure after the implementation audit means `SPATIAL_TOKEN_IMPLEMENTATION_FAILED`.
The scheduler horizon is the *actual* optimizer-step budget (Task 6D.1 section 2).

Writes `evaluation/task6e_e0_overfit.json`.
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
)
from buildreasonseg_mvp.spatial_training import spatial_batch  # noqa: E402

from task6e_common import (  # noqa: E402
    EVAL,
    e0_selection,
    gate_report,
    load_spatial_config,
    lr_values,
    optimizer_and_scheduler,
    save_stage_checkpoint,
    write_json,
)

OUT = EVAL / "task6e_e0_overfit.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--min-steps-before-stop", type=int, default=250)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, payload = load_spatial_config()
    e0_cfg = cfg["e0"]
    max_steps = int(args.max_steps or e0_cfg["max_steps"])
    gate_cfg = e0_cfg["gate"]
    bins = int(cfg["spatial_tokens"]["bins"])

    records, pairs = e0_selection(payload, images=int(args.images))
    samples = [data_mod.to_sample(record) for record in records]
    steps_per_epoch = len(samples)
    total_steps = max(1, max_steps)
    warmup_steps = int(cfg["optimizer"]["warmup_steps"])

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, device="cuda", verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    trainables = runtime.freeze_for_spatial_tokens()

    print(
        f"[task6e.e0] bins {bins}, {len(samples)} records / {len(pairs)} pairs, "
        f"budget {total_steps} steps, warmup {warmup_steps}",
        flush=True,
    )

    prepared = []
    for sample in samples:
        image = sample.image_rgb()
        batch, example = spatial_batch(runtime, sample, image=image)
        prepared.append((sample, batch, example))

    optimizer, scheduler = optimizer_and_scheduler(
        runtime, total_steps=total_steps, warmup_steps=warmup_steps
    )

    report = {
        "_doc": (
            "Task 6E section 10. Stage E0: overfit 20 deterministic records (10 same-image/"
            "different-target pairs) with the explicit spatial targets and measure the "
            "free-generation gate. A pass is implementation sanity only; it is not a "
            "generalization claim."
        ),
        "task": "6E",
        "stage": "E0",
        "bins": bins,
        "oracle_selection": cfg["_oracle_selection"],
        "trainables": trainables,
        "loss": {
            "assistant_weight": float(cfg["spatial_tokens"]["assistant_loss_weight"]),
            "location_weight": float(cfg["spatial_tokens"]["location_loss_weight"]),
            "mask_loss": False,
        },
        "budget": {
            "steps_per_epoch": steps_per_epoch,
            "max_steps": total_steps,
            "epochs_equivalent": total_steps / max(steps_per_epoch, 1),
            "scheduler_horizon": total_steps,
            "warmup_steps": warmup_steps,
            "note": (
                "Task 6D.1 section 2: the cosine horizon is the real optimizer-step budget, so the "
                "terminal factor is reached only at the final step."
            ),
        },
        "sample_ids": [str(sample.sample_id) for sample in samples],
        "pair_images": [str(pair["image_id"]) for pair in pairs],
        "history": [],
        "evaluations": [],
    }

    def evaluate(step: int) -> dict:
        free_records = []
        for sample in samples:
            free_records.append(generate_spatial(runtime, sample))
        parsed_by_id = {record["sample_id"]: record for record in free_records}
        metrics = geometry_report(free_records, bins=bins)
        paired = paired_geometry_report(parsed_by_id, pairs)
        gate = gate_report({**metrics, "paired": paired}, gate_cfg, "e0")
        return {
            "step": step,
            "metrics": metrics,
            "paired": paired,
            "gate": gate,
            "records": free_records,
        }

    global_step = 0
    stop_reason = "budget_exhausted"
    while global_step < total_steps:
        sample, batch, example = prepared[global_step % steps_per_epoch]
        runtime.set_visual_cache_key(str(sample.image_id))
        result = runtime.spatial_train_step(batch, example, optimizer=optimizer)
        scheduler.step()
        global_step += 1
        if global_step % int(args.log_every) == 0 or global_step == total_steps:
            entry = {
                "step": global_step,
                "losses": result["losses"],
                "location_token_accuracy": result["location_token_accuracy"],
                "location_abs_bin_error": result["location_abs_bin_error"],
                "lrs": lr_values(optimizer),
                "vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            }
            report["history"].append(entry)
            print(
                f"[task6e.e0] step {global_step}/{total_steps} total {entry['losses']['total']:.4f} "
                f"assistant {entry['losses']['assistant_ce']:.4f} location "
                f"{entry['losses']['location_ce']:.4f} loc_acc "
                f"{entry['location_token_accuracy']:.3f} lr {entry['lrs'][0]:.3e}",
                flush=True,
            )
        if global_step % int(args.eval_every) == 0 or global_step == total_steps:
            evaluation = evaluate(global_step)
            report["evaluations"].append(evaluation)
            metrics = evaluation["metrics"]
            print(
                f"[task6e.e0] step {global_step} free: structural "
                f"{metrics['structural_valid']}/{metrics['count']} exact "
                f"{metrics['exact_four_token_sequence']}/{metrics['count']} boxIoU "
                f"{metrics['mean_predicted_box_iou']} paired geom "
                f"{evaluation['paired']['geometry_paired_pass']}/{evaluation['paired']['paired_total']} "
                f"gate {evaluation['gate']['passed']}",
                flush=True,
            )
            if evaluation["gate"]["passed"] and global_step >= int(args.min_steps_before_stop):
                stop_reason = "gate_passed"
                break

    final = report["evaluations"][-1]
    report["final"] = {
        "step": global_step,
        "metrics": final["metrics"],
        "paired": final["paired"],
        "gate": final["gate"],
        "records": final["records"],
        "stop_reason": stop_reason,
    }
    report["verdict"] = (
        "E0_PASS" if final["gate"]["passed"] else "SPATIAL_TOKEN_IMPLEMENTATION_FAILED"
    )
    checkpoint = save_stage_checkpoint(
        cfg,
        "E0",
        "e0_final",
        runtime.model,
        global_step,
        metrics={
            "gate_passed": bool(final["gate"]["passed"]),
            "structural_valid": int(final["metrics"]["structural_valid"]),
            "mean_predicted_box_iou": final["metrics"]["mean_predicted_box_iou"],
        },
    )
    report["checkpoint"] = checkpoint
    report["selection"] = {
        "carried_into_e1": False,
        "reason": (
            "E0 is a 20-sample implementation sanity check. Its weights are fitted to those 20 "
            "records, so E1 starts from the same frozen base with fresh LoRA and token rows "
            "instead of continuing from an overfit checkpoint."
        ),
    }
    report["seconds"] = round(time.time() - started, 2)
    write_json(OUT, report)
    print(
        f"[task6e.e0] {report['verdict']} at step {global_step}: structural "
        f"{final['metrics']['structural_valid']}/{final['metrics']['count']}, exact four-token "
        f"{final['metrics']['exact_four_token_sequence']}/{final['metrics']['count']}, mean box IoU "
        f"{final['metrics']['mean_predicted_box_iou']}, paired geometry "
        f"{final['paired']['geometry_paired_pass']}/{final['paired']['paired_total']}",
        flush=True,
    )
    print(f"[task6e.e0] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if final["gate"]["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
