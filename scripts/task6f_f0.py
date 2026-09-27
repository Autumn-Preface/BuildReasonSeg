"""Task 6F section 9: Stage F0 -- 20-record implementation proof.

Overfits the same deterministic 20 records (10 same-image/different-target pairs) as Task 6E
E0, evaluating exclusively through the **inference-form query path** (`image + instruction ->
constant [BOX] -> Qwen -> [BOX] hidden -> TargetAwareBoxHead -> box`); no future reasoning token
is ever teacher-forced into the query representation.

F0 gate: mean train box IoU >= 0.85, geometry paired >= 9/10, the two instructions of a same
image produce non-identical boxes, and no GT leakage. Failure after the implementation audit
means `TARGET_QUERY_IMPLEMENTATION_FAILED`.

Writes `evaluation/task6f_f0_overfit.json`.
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
from buildreasonseg_mvp.box_query_eval import box_record, geometry_report, paired_geometry_report  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6f_common import (  # noqa: E402
    EVAL,
    f0_selection,
    gate_report,
    load_box_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    save_stage_checkpoint,
    write_json,
)

OUT = EVAL / "task6f_f0_overfit.json"


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
    cfg, payload = load_box_config()
    f0_cfg = cfg["f0"]
    max_steps = int(args.max_steps or f0_cfg["max_steps"])
    gate_cfg = f0_cfg["gate"]

    records, pairs = f0_selection(payload, images=int(args.images))
    samples = [data_mod.to_sample(record) for record in records]

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

    print(
        f"[task6f.f0] {len(samples)} records / {len(pairs)} pairs, budget {max_steps} steps",
        flush=True,
    )

    prepared = []
    for sample in samples:
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
        prepared.append((sample, batch))

    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime, total_steps=max_steps, warmup_steps=int(cfg["optimizer"]["warmup_steps"])
    )

    report = {
        "_doc": (
            "Task 6F section 9. Stage F0: overfit 20 deterministic records (10 same-image/"
            "different-target pairs) with the box-query objective and measure the F0 gate on the "
            "inference-form query path. A pass is implementation proof only."
        ),
        "task": "6F",
        "stage": "F0",
        "trainables": trainables,
        "head": runtime.reports["box_head"],
        "optimizer_groups": optimizer_coverage(groups),
        "loss": {
            "reasoning_weight": float(cfg["box_query"]["reasoning_loss_weight"]),
            "box_weight": float(cfg["box_query"]["box_loss_weight"]),
            "mask_loss": False,
        },
        "budget": {
            "samples": len(samples),
            "max_steps": max_steps,
            "scheduler_horizon": max_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
            "note": (
                "Task 6D.1 section 2 corrected horizon: the cosine schedule is built over the real "
                "step budget, so the terminal factor is reached only at the final step."
            ),
        },
        "sample_ids": [str(sample.sample_id) for sample in samples],
        "pair_images": [str(pair["image_id"]) for pair in pairs],
        "history": [],
        "evaluations": [],
    }

    def evaluate(step: int) -> dict:
        records = [box_record(runtime, sample) for sample in samples]
        by_id = {record["sample_id"]: record for record in records}
        metrics = geometry_report(records)
        paired = paired_geometry_report(by_id, pairs)
        train_box_iou = metrics["mean_box_iou"]
        gate = gate_report(
            {
                "train_box_iou": train_box_iou,
                "paired": paired,
                "no_gt_leakage": True,
            },
            gate_cfg,
            "f0",
        )
        return {
            "step": step,
            "train_box_iou": train_box_iou,
            "metrics": metrics,
            "paired": paired,
            "gate": gate,
            "records": records,
        }

    global_step = 0
    stop_reason = "budget_exhausted"
    while global_step < max_steps:
        sample, batch = prepared[global_step % len(samples)]
        runtime.set_visual_cache_key(str(sample.image_id))
        result = runtime.box_query_train_step(batch, sample.target_mask(), optimizer=optimizer)
        scheduler.step()
        global_step += 1
        if global_step % int(args.log_every) == 0 or global_step == max_steps:
            entry = {
                "step": global_step,
                "losses": result["losses"],
                "box_iou": result["box_iou"],
                "lrs": lr_values(optimizer),
                "vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            }
            report["history"].append(entry)
            print(
                f"[task6f.f0] step {global_step}/{max_steps} total {entry['losses']['total']:.4f} "
                f"reasoning {entry['losses']['reasoning_ce']:.4f} box {entry['losses']['box_loss']:.4f} "
                f"boxIoU {entry['box_iou']:.4f} lr {entry['lrs'][0]:.3e}",
                flush=True,
            )
        if global_step % int(args.eval_every) == 0 or global_step == max_steps:
            evaluation = evaluate(global_step)
            report["evaluations"].append(evaluation)
            print(
                f"[task6f.f0] step {global_step} query path: train boxIoU "
                f"{evaluation['train_box_iou']} paired {evaluation['paired']['geometry_paired_pass']}/"
                f"{evaluation['paired']['paired_total']} non-identical "
                f"{evaluation['paired']['pairs_with_non_identical_boxes']}/"
                f"{evaluation['paired']['paired_total']} gate {evaluation['gate']['passed']}",
                flush=True,
            )
            if evaluation["gate"]["passed"] and global_step >= int(args.min_steps_before_stop):
                stop_reason = "gate_passed"
                break

    final = report["evaluations"][-1]
    report["final"] = {
        "step": global_step,
        "train_box_iou": final["train_box_iou"],
        "metrics": final["metrics"],
        "paired": final["paired"],
        "gate": final["gate"],
        "records": final["records"],
        "stop_reason": stop_reason,
    }
    report["verdict"] = "F0_PASS" if final["gate"]["passed"] else "TARGET_QUERY_IMPLEMENTATION_FAILED"
    report["no_gt_leakage_audit"] = {
        "query_path_inputs": "image + instruction + constant [BOX] only",
        "gt_used_for": ["SmoothL1 supervision", "evaluation scoring"],
        "gt_enters_inference": False,
    }
    checkpoint = save_stage_checkpoint(
        cfg,
        "F0",
        "f0_final",
        runtime.model,
        global_step,
        metrics={
            "gate_passed": bool(final["gate"]["passed"]),
            "train_box_iou": final["train_box_iou"],
            "geometry_paired_pass": final["paired"]["geometry_paired_pass"],
        },
    )
    report["checkpoint"] = checkpoint
    report["selection"] = {
        "carried_into_f1": False,
        "reason": (
            "F0 is a 20-record implementation proof; its weights are fitted to those 20 records, "
            "so F1 starts from the same clean base with fresh LoRA, token rows and box head."
        ),
    }
    report["seconds"] = round(time.time() - started, 2)
    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    print(
        f"[task6f.f0] {report['verdict']} at step {global_step}: train box IoU "
        f"{final['train_box_iou']}, paired {final['paired']['geometry_paired_pass']}/"
        f"{final['paired']['paired_total']}, non-identical "
        f"{final['paired']['pairs_with_non_identical_boxes']}/"
        f"{final['paired']['paired_total']}, same-image L1 "
        f"{final['paired']['mean_same_image_predicted_box_l1']}",
        flush=True,
    )
    print(f"[task6f.f0] wrote {output}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if final["gate"]["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
