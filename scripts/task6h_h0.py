"""Task 6H sections 11-12: Stage H0 -- 10-pair counterfactual overfit.

One optimizer step = one canonical pair (same image, different instructions, different targets),
with the fixed pair objective `0.5*(reasoning_A+reasoning_B) + 1.0*(heatmap_A+heatmap_B) + 2.0*L_cf`.
Evaluation runs the real path for both instructions of every H0 pair and reports pair ranking,
strict-margin, point-inside-own, paired point selection and the own-minus-cross region margin.

H0 gate: pair ranking >= 9/10, point-inside-own >= 18/20, paired point selection >= 9/10, mean
own-minus-cross region margin > 0.5 (heatmap Dice is diagnostic only). Failure after the focused
audit is `COUNTERFACTUAL_QUERY_SIGNAL_FAILED`.

Writes `evaluation/task6h_h0_overfit.json`.
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
from buildreasonseg_mvp.dense_eval import spatial_metrics  # noqa: E402
from buildreasonseg_mvp.dense_grounding import feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.pair_eval import evaluate_pair, pair_summary  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6h_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    gate_report,
    h0_pairs,
    load_pair_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    pair_dicts_for_eval,
    save_stage_checkpoint,
    write_json,
)

OUT = EVAL / "task6h_h0_overfit.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pair-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--min-steps-before-stop", type=int, default=250)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, payload = load_pair_config()
    grid = int(cfg["dense_grounding"]["grid"])
    h0_cfg = cfg["h0"]
    max_steps = int(args.max_pair_steps or h0_cfg["max_pair_steps"])
    gate_cfg = h0_cfg["gate"]

    triples = h0_pairs(canonical_train_pairs(payload), int(args.pairs))
    pair_dicts = pair_dicts_for_eval(triples)
    samples = {
        pair.a: data_mod.to_sample(record_a) for pair, record_a, _b in triples
    } | {pair.b: data_mod.to_sample(record_b) for pair, _a, record_b in triples}

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    first_sample = next(iter(samples.values()))
    probe_image = first_sample.image_rgb()
    probe_features, _cached = runtime.features_for(first_sample, probe_image)
    in_channels = int(feature_tensor_for_grid(probe_features, grid).shape[1])
    runtime.install_dense_head(in_channels=in_channels)
    trainables = runtime.freeze_for_counterfactual()

    print(
        f"[task6h.h0] grid {grid}: {len(triples)} pairs / {len(samples)} records, "
        f"budget {max_steps} PAIR steps, margin {cfg['counterfactual']['margin']}",
        flush=True,
    )

    prepared = []
    for pair, record_a, record_b in triples:
        sample_a, sample_b = samples[pair.a], samples[pair.b]
        image = sample_a.image_rgb()
        features, _cached = runtime.features_for(sample_a, image)
        batch_a = build_box_query_batch(
            runtime.processor, runtime.tokenizer, image, sample_a.instruction_zh,
            sample_a.reasoning_zh, int(runtime.box_setup.box_token_id),
            int(runtime.model.seg_token_id), append_eos=True,
        )
        batch_b = build_box_query_batch(
            runtime.processor, runtime.tokenizer, image, sample_b.instruction_zh,
            sample_b.reasoning_zh, int(runtime.box_setup.box_token_id),
            int(runtime.model.seg_token_id), append_eos=True,
        )
        prepared.append((pair, sample_a, sample_b, batch_a, batch_b, features))

    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime, total_steps=max_steps, warmup_steps=int(cfg["optimizer"]["warmup_steps"])
    )

    report = {
        "_doc": (
            "Task 6H sections 11-12. Stage H0: overfit 10 canonical same-image counterfactual pairs "
            "with the pair objective (per-sample BCE+Dice plus the own-vs-cross region-ranking loss) "
            "and measure the H0 gate on the real query-to-heatmap path. A pass is implementation "
            "proof for the pair-step semantics only."
        ),
        "task": "6H",
        "stage": "H0",
        "grid": grid,
        "pair_manifest_hash": cfg.get("_pair_manifest_hash"),
        "trainables": trainables,
        "head": runtime.reports["dense_head"],
        "optimizer_groups": optimizer_coverage(groups),
        "loss": {
            "reasoning_weight": 0.5,
            "heatmap_weight": 1.0,
            "cf_weight": 2.0,
            "cf_margin": float(cfg["counterfactual"]["margin"]),
        },
        "budget": {
            "pairs": len(triples),
            "max_pair_steps": max_steps,
            "scheduler_horizon": max_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
            "note": "the scheduler horizon counts PAIR optimizer steps, not record steps",
        },
        "pair_ids": [f"{pair.a}|{pair.b}" for pair, _a, _b in triples],
        "history": [],
        "evaluations": [],
    }

    def evaluate(step: int) -> dict:
        rows = []
        for pair, sample_a, sample_b, _ba, _bb, features in prepared:
            rows.append(
                evaluate_pair(
                    runtime,
                    next(d for d in pair_dicts if d["a"] == pair.a and d["b"] == pair.b),
                    sample_a,
                    sample_b,
                    with_heatmap=True,
                )
            )
        pair_metrics = pair_summary(rows)
        records = [row["record_a"] for row in rows] + [row["record_b"] for row in rows]
        # The binary heatmaps exist only in memory for the A/B IoU (kept as the pair scalar
        # `same_image_heatmap_iou`); storing 256x256 bool lists per record would bloat the artifact.
        for record in records:
            record.pop("heatmap_binary", None)
        for row in rows:
            row.pop("record_a", None)
            row.pop("record_b", None)
        metrics = spatial_metrics(records)
        inside_own = int(sum(1 for record in records if record["point_inside_own"]))
        gate = gate_report(
            {
                "pair": pair_metrics,
                "point_inside_own": inside_own,
                "no_gt_leakage": True,
            },
            gate_cfg,
            "h0",
        )
        return {
            "step": step,
            "pair": pair_metrics,
            "point_inside_own": inside_own,
            "heatmap_dice_diagnostic": metrics["mean_heatmap_dice"],
            "records": records,
            "pairs": rows,
            "gate": gate,
        }

    global_step = 0
    stop_reason = "budget_exhausted"
    while global_step < max_steps:
        pair, sample_a, sample_b, batch_a, batch_b, features = prepared[global_step % len(prepared)]
        runtime.set_visual_cache_key(str(pair.image_id))
        result = runtime.pair_train_step(
            batch_a, batch_b, sample_a.target_mask(), sample_b.target_mask(), features,
            optimizer=optimizer,
        )
        scheduler.step()
        global_step += 1
        if global_step % int(args.log_every) == 0 or global_step == max_steps:
            entry = {
                "pair_step": global_step,
                "losses": result["losses"],
                "region_scores": result["region_scores"],
                "pair_ranking_pass": result["pair_ranking_pass"],
                "lrs": lr_values(optimizer),
                "vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            }
            report["history"].append(entry)
            print(
                f"[task6h.h0] pair step {global_step}/{max_steps} total {entry['losses']['total']:.4f} "
                f"reason {entry['losses']['reasoning_ce']:.4f} heat {entry['losses']['heatmap_sum']:.4f} "
                f"cf {entry['losses']['cf']:.4f} margin {entry['region_scores']['mean_margin']:+.4f} "
                f"lr {entry['lrs'][0]:.3e}",
                flush=True,
            )
        if global_step % int(args.eval_every) == 0 or global_step == max_steps:
            evaluation = evaluate(global_step)
            report["evaluations"].append(evaluation)
            print(
                f"[task6h.h0] step {global_step} query path: pair ranking "
                f"{evaluation['pair']['pair_ranking_pass']}/{evaluation['pair']['pair_count']} strict "
                f"{evaluation['pair']['strict_margin_pass']} inside-own {evaluation['point_inside_own']}/"
                f"{len(evaluation['records'])} paired point "
                f"{evaluation['pair']['paired_point_selection_pass']}/{evaluation['pair']['pair_count']} "
                f"margin {evaluation['pair']['mean_own_minus_cross_margin']:+.4f} dice "
                f"{evaluation['heatmap_dice_diagnostic']:.4f} gate {evaluation['gate']['passed']}",
                flush=True,
            )
            if evaluation["gate"]["passed"] and global_step >= int(args.min_steps_before_stop):
                stop_reason = "gate_passed"
                break

    final = report["evaluations"][-1]
    report["final"] = {
        "pair_step": global_step,
        "pair": final["pair"],
        "point_inside_own": final["point_inside_own"],
        "record_count": len(final["records"]),
        "heatmap_dice_diagnostic": final["heatmap_dice_diagnostic"],
        "gate": final["gate"],
        "records": final["records"],
        "pairs": final["pairs"],
        "stop_reason": stop_reason,
    }
    report["verdict"] = "H0_PASS" if final["gate"]["passed"] else "COUNTERFACTUAL_QUERY_SIGNAL_FAILED"
    report["no_gt_leakage_audit"] = {
        "pair_path_inputs": "image + instruction + constant [BOX] + one shared frozen SAM2 feature",
        "gt_used_for": ["heatmap supervision", "counterfactual region ranking", "evaluation scoring"],
        "gt_enters_inference": False,
    }
    checkpoint = save_stage_checkpoint(
        cfg,
        "H0",
        "h0_final",
        runtime.model,
        global_step,
        metrics={
            "gate_passed": bool(final["gate"]["passed"]),
            "pair_ranking_pass": final["pair"]["pair_ranking_pass"],
            "point_inside_own": final["point_inside_own"],
            "mean_own_minus_cross_margin": final["pair"]["mean_own_minus_cross_margin"],
        },
    )
    report["checkpoint"] = checkpoint
    report["selection"] = {
        "carried_into_h1": False,
        "reason": (
            "H0 is a 10-pair implementation proof; its weights are fitted to those pairs, so H1 "
            "starts from the same clean Task 6G initialization convention."
        ),
    }
    report["seconds"] = round(time.time() - started, 2)
    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    print(
        f"[task6h.h0] {report['verdict']} at pair step {global_step}: ranking "
        f"{final['pair']['pair_ranking_pass']}/{final['pair']['pair_count']}, inside-own "
        f"{final['point_inside_own']}/{len(final['records'])}, paired point "
        f"{final['pair']['paired_point_selection_pass']}/{final['pair']['pair_count']}, margin "
        f"{final['pair']['mean_own_minus_cross_margin']:+.4f}, dice "
        f"{final['heatmap_dice_diagnostic']:.4f}",
        flush=True,
    )
    print(f"[task6h.h0] wrote {output}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if final["gate"]["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
