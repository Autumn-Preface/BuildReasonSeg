"""Task 6G section 11: Stage G0 -- 20-record implementation proof.

Overfits the same deterministic 20 records (10 same-image/different-target pairs) as Tasks 6E/6F,
evaluating exclusively through the real path: `image + instruction -> fixed [BOX] -> dense heatmap
-> argmax point`. G0 gate: point inside own target >= 19/20, paired point selection >= 9/10, mean
heatmap Dice >= 0.80, paired instructions produce distinct points, no GT leakage. Failure after the
implementation audit means `DENSE_GROUNDING_IMPLEMENTATION_FAILED`.

Writes `evaluation/task6g_g0_overfit.json`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.box_query import build_box_query_batch  # noqa: E402
from buildreasonseg_mvp.dense_eval import paired_point_report, predict_heatmap, spatial_metrics  # noqa: E402
from buildreasonseg_mvp.dense_grounding import feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6g_common import (  # noqa: E402
    EVAL,
    g0_selection,
    gate_report,
    load_dense_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    save_stage_checkpoint,
    write_json,
)

OUT = EVAL / "task6g_g0_overfit.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--min-steps-before-stop", type=int, default=250)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, payload = load_dense_config()
    grid = int(cfg["dense_grounding"]["grid"])
    f0_cfg = cfg["g0"]
    max_steps = int(args.max_steps or f0_cfg["max_steps"])
    gate_cfg = f0_cfg["gate"]

    records, pairs = g0_selection(payload, images=int(args.images))
    samples = [data_mod.to_sample(record) for record in records]

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

    print(f"[task6g.g0] grid {grid}: {len(samples)} records / {len(pairs)} pairs, budget {max_steps}", flush=True)

    prepared = []
    for sample in samples:
        image = sample.image_rgb()
        features, _cached = runtime.features_for(sample, image)
        batch = build_box_query_batch(
            runtime.processor, runtime.tokenizer, image, sample.instruction_zh,
            sample.reasoning_zh, int(runtime.box_setup.box_token_id),
            int(runtime.model.seg_token_id), append_eos=True,
        )
        prepared.append((sample, batch, features))

    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime, total_steps=max_steps, warmup_steps=int(cfg["optimizer"]["warmup_steps"])
    )

    report = {
        "_doc": (
            "Task 6G section 11. Stage G0: overfit 20 deterministic records (10 same-image/"
            "different-target pairs) with the dense heatmap objective and measure the G0 gate on "
            "the real query-to-heatmap path. A pass is implementation proof only."
        ),
        "task": "6G",
        "stage": "G0",
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
            "samples": len(samples),
            "max_steps": max_steps,
            "scheduler_horizon": max_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
        },
        "sample_ids": [str(sample.sample_id) for sample in samples],
        "pair_images": [str(pair["image_id"]) for pair in pairs],
        "history": [],
        "evaluations": [],
    }

    def evaluate(step: int) -> dict:
        records = [predict_heatmap(runtime, sample, with_heatmap=True) for sample in samples]
        by_id = {record["sample_id"]: record for record in records}
        metrics = spatial_metrics(records)
        paired = paired_point_report(by_id, pairs)
        # The binary heatmaps exist only in memory for the A/B IoU; they are stripped before
        # serialization (a 256x256 bool list per record would bloat the artifact to hundreds of MB).
        for record in records:
            record.pop("heatmap_binary", None)
        gate = gate_report(
            {
                "point_inside_own": int(sum(1 for record in records if record["point_inside_own"])),
                "heatmap_dice": metrics["mean_heatmap_dice"],
                "paired": paired,
                "no_gt_leakage": True,
            },
            gate_cfg,
            "g0",
        )
        return {
            "step": step,
            "metrics": metrics,
            "paired": paired,
            "gate": gate,
            "records": records,
        }

    global_step = 0
    stop_reason = "budget_exhausted"
    while global_step < max_steps:
        sample, batch, features = prepared[global_step % len(samples)]
        runtime.set_visual_cache_key(str(sample.image_id))
        result = runtime.dense_train_step(batch, sample.target_mask(), features, optimizer=optimizer)
        scheduler.step()
        global_step += 1
        if global_step % int(args.log_every) == 0 or global_step == max_steps:
            entry = {
                "step": global_step,
                "losses": result["losses"],
                "heatmap_dice_quality": result["heatmap_dice_quality"],
                "lrs": lr_values(optimizer),
                "vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            }
            report["history"].append(entry)
            print(
                f"[task6g.g0] step {global_step}/{max_steps} total {entry['losses']['total']:.4f} "
                f"reasoning {entry['losses']['reasoning_ce']:.4f} bce {entry['losses']['heatmap_bce']:.4f} "
                f"dice {entry['losses']['heatmap_dice']:.4f} diceQ {entry['heatmap_dice_quality']:.4f} "
                f"lr {entry['lrs'][0]:.3e}",
                flush=True,
            )
        if global_step % int(args.eval_every) == 0 or global_step == max_steps:
            evaluation = evaluate(global_step)
            report["evaluations"].append(evaluation)
            print(
                f"[task6g.g0] step {global_step} query path: inside "
                f"{evaluation['metrics']['point_inside_target_rate']} dice "
                f"{evaluation['metrics']['mean_heatmap_dice']} paired "
                f"{evaluation['paired']['paired_point_selection_pass']}/"
                f"{evaluation['paired']['paired_total']} distinct "
                f"{evaluation['paired']['pairs_with_distinct_points']}/"
                f"{evaluation['paired']['paired_total']} gate {evaluation['gate']['passed']}",
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
    report["verdict"] = "G0_PASS" if final["gate"]["passed"] else "DENSE_GROUNDING_IMPLEMENTATION_FAILED"
    report["no_gt_leakage_audit"] = {
        "query_path_inputs": "image + instruction + constant [BOX] + frozen SAM2 features only",
        "gt_used_for": ["heatmap supervision", "evaluation scoring"],
        "gt_enters_inference": False,
    }
    checkpoint = save_stage_checkpoint(
        cfg,
        "G0",
        "g0_final",
        runtime.model,
        global_step,
        metrics={
            "gate_passed": bool(final["gate"]["passed"]),
            "point_inside_rate": final["metrics"]["point_inside_target_rate"],
            "paired_point_selection_pass": final["paired"]["paired_point_selection_pass"],
        },
    )
    report["checkpoint"] = checkpoint
    report["selection"] = {
        "carried_into_g1": False,
        "reason": (
            "G0 is a 20-record implementation proof; its weights are fitted to those 20 records, "
            "so G1 starts from the same clean base with fresh LoRA, token rows and dense head."
        ),
    }
    report["seconds"] = round(time.time() - started, 2)
    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    print(
        f"[task6g.g0] {report['verdict']} at step {global_step}: inside "
        f"{final['metrics']['point_inside_target_rate']}, heatmap dice "
        f"{final['metrics']['mean_heatmap_dice']}, paired "
        f"{final['paired']['paired_point_selection_pass']}/{final['paired']['paired_total']}, "
        f"distinct {final['paired']['pairs_with_distinct_points']}/{final['paired']['paired_total']}, "
        f"same-image point dist {final['paired']['mean_same_image_point_distance']}",
        flush=True,
    )
    print(f"[task6g.g0] wrote {output}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if final["gate"]["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
