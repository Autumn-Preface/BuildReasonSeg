"""Task 6H.1 sections 12-13: Stage H0-R -- corrected 10-pair overfit.

Clean Task 6G/6H initialization (never the scale-exploded Task 6H H0 checkpoint), the same 10
canonical pairs, at most 1500 **pair** optimizer steps of the corrected objective:

    L_total = 0.5*(L_reasoning_A+L_reasoning_B) + 1.0*(L_point_A+L_point_B) + 1.0*L_cf

with `L_point` = 65,536-class spatial cross-entropy of the target point-cell and `L_cf` the bounded
own-vs-cross target **probability mass** preference. Evaluated at 250/500/750/1000/1500 pair steps
through the real `argmax` path.

H0-R gate: point-inside-own >= 18/20, paired point-selection >= 9/10, bounded pair ranking >= 9/10,
mean own mass > mean cross mass, and final mean absolute logit < 10.0. Failure after the focused
audit is `BOUNDED_POINT_OBJECTIVE_FAILED`.

Writes `evaluation/task6h1_h0r_overfit.json`.
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
from buildreasonseg_mvp.dense_grounding import feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_objective import (  # noqa: E402
    evaluate_point_pair,
    pair_probability_summary,
    point_record_summary,
)

from task6h1_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    gate_report,
    h0r_pairs,
    load_point_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    pair_dicts_for_eval,
    save_stage_checkpoint,
    write_json,
)

OUT = EVAL / "task6h1_h0r_overfit.json"


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
    cfg, payload = load_point_config()
    grid = int(cfg["dense_grounding"]["grid"])
    h0_cfg = cfg["h0r"]
    max_steps = int(args.max_pair_steps or h0_cfg["max_pair_steps"])
    gate_cfg = h0_cfg["gate"]

    triples = h0r_pairs(canonical_train_pairs(payload), int(args.pairs))
    pair_dicts = {row["a"]: row for row in pair_dicts_for_eval(triples)}
    samples = {pair.a: data_mod.to_sample(ra) for pair, ra, _rb in triples} | {
        pair.b: data_mod.to_sample(rb) for pair, _ra, rb in triples
    }

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg["training"].get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg["training"].get("visual_feature_cache_max_images", 512)),
        )
    first = next(iter(samples.values()))
    probe_image = first.image_rgb()
    probe_features, _cached = runtime.features_for(first, probe_image)
    runtime.install_dense_head(in_channels=int(feature_tensor_for_grid(probe_features, grid).shape[1]))
    trainables = runtime.freeze_for_point_objective()

    print(
        f"[task6h1.h0r] grid {grid}: {len(triples)} pairs / {len(samples)} records, "
        f"budget {max_steps} PAIR steps, eps {cfg['point_objective']['eps']}",
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
            "Task 6H.1 sections 12-13. Stage H0-R: overfit the same 10 canonical pairs as Task 6H H0 "
            "from a clean initialization with the corrected point-aligned bounded objective (spatial "
            "cross-entropy + bounded probability-mass preference). Task 6G BCE+Dice and Task 6H "
            "raw-logit ranking are logged only."
        ),
        "task": "6H.1",
        "stage": "H0-R",
        "grid": grid,
        "trainables": trainables,
        "head": runtime.reports["dense_head"],
        "optimizer_groups": optimizer_coverage(groups),
        "loss": {
            "reasoning_weight": 0.5,
            "classification_weight": 1.0,
            "cf_weight": 1.0,
            "eps": float(cfg["point_objective"]["eps"]),
            "margin": None,
            "legacy_bce_dice_gradient": False,
            "legacy_logit_ranking_gradient": False,
        },
        "budget": {
            "pairs": len(triples),
            "max_pair_steps": max_steps,
            "scheduler_horizon": max_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
            "note": "the scheduler horizon counts PAIR optimizer steps",
        },
        "pair_ids": [f"{pair.a}|{pair.b}" for pair, _a, _b in triples],
        "history": [],
        "evaluations": [],
    }

    def evaluate(step: int) -> dict:
        rows = []
        for pair, sample_a, sample_b, _ba, _bb, _features in prepared:
            rows.append(
                evaluate_point_pair(runtime, pair_dicts[pair.a], sample_a, sample_b, with_hidden=False)
            )
        pair_metrics = pair_probability_summary(rows)
        records = [row["record_a"] for row in rows] + [row["record_b"] for row in rows]
        for row in rows:
            row.pop("record_a", None)
            row.pop("record_b", None)
        record_metrics = point_record_summary(records)
        mean_abs_logit = float(
            sum(record["mean_abs_logit"] for record in records) / max(len(records), 1)
        )
        gate = gate_report(
            {
                "point_inside_own": int(record_metrics["point_inside_count"]),
                "pair": pair_metrics,
                "mean_abs_logit": mean_abs_logit,
                "no_gt_leakage": True,
            },
            gate_cfg,
            "h0r",
        )
        return {
            "pair_step": step,
            "pair": pair_metrics,
            "records": record_metrics,
            "mean_abs_logit": mean_abs_logit,
            "record_rows": records,
            "pair_rows": rows,
            "gate": gate,
        }

    global_step = 0
    stop_reason = "budget_exhausted"
    while global_step < max_steps:
        pair, sample_a, sample_b, batch_a, batch_b, features = prepared[global_step % len(prepared)]
        runtime.set_visual_cache_key(str(pair.image_id))
        result = runtime.bounded_pair_train_step(
            batch_a, batch_b, sample_a.target_mask(), sample_b.target_mask(), features,
            optimizer=optimizer,
        )
        scheduler.step()
        global_step += 1
        if global_step % int(args.log_every) == 0 or global_step == max_steps:
            entry = {
                "pair_step": global_step,
                "losses": result["losses"],
                "region_masses": result["region_masses"],
                "pair_preference_pass": result["pair_preference_pass"],
                "mean_abs_logit": result["mean_abs_logit"],
                "lrs": lr_values(optimizer),
                "vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            }
            report["history"].append(entry)
            print(
                f"[task6h1.h0r] pair step {global_step}/{max_steps} total "
                f"{entry['losses']['total']:.4f} reason {entry['losses']['reasoning_ce']:.4f} "
                f"point {entry['losses']['point_ce_sum']:.4f} cf {entry['losses']['cf']:.4f} "
                f"own {entry['region_masses']['p_aa']:.4f}/{entry['region_masses']['p_bb']:.4f} "
                f"cross {entry['region_masses']['p_ab']:.4f}/{entry['region_masses']['p_ba']:.4f} "
                f"|logit| {entry['mean_abs_logit']:.3f} lr {entry['lrs'][0]:.3e}",
                flush=True,
            )
        if global_step % int(args.eval_every) == 0 or global_step == max_steps:
            evaluation = evaluate(global_step)
            report["evaluations"].append(evaluation)
            print(
                f"[task6h1.h0r] step {global_step} argmax path: inside-own "
                f"{evaluation['records']['point_inside_count']}/{evaluation['records']['count']} "
                f"top1 {evaluation['records']['target_cell_top1_count']} top5 "
                f"{evaluation['records']['target_cell_top5_count']} paired point "
                f"{evaluation['pair']['paired_point_selection_pass']}/{evaluation['pair']['pair_count']} "
                f"ranking {evaluation['pair']['pair_ranking_pass']}/{evaluation['pair']['pair_count']} "
                f"margin {evaluation['pair']['mean_own_minus_cross_probability_mass']:+.4f} "
                f"|logit| {evaluation['mean_abs_logit']:.3f} gate {evaluation['gate']['passed']}",
                flush=True,
            )
            if evaluation["gate"]["passed"] and global_step >= int(args.min_steps_before_stop):
                stop_reason = "gate_passed"
                break

    final = report["evaluations"][-1]
    report["final"] = {**final, "stop_reason": stop_reason}
    report["verdict"] = "H0R_PASS" if final["gate"]["passed"] else "BOUNDED_POINT_OBJECTIVE_FAILED"
    report["no_gt_leakage_audit"] = {
        "pair_path_inputs": "image + instruction + constant [BOX] + one shared frozen SAM2 feature",
        "gt_used_for": ["target point-cell supervision", "bounded region mass", "evaluation scoring"],
        "gt_enters_inference": False,
        "inference_operation": "argmax(heatmap_logits)",
    }
    checkpoint = save_stage_checkpoint(
        cfg,
        "H0R",
        "h0r_final",
        runtime.model,
        global_step,
        metrics={
            "gate_passed": bool(final["gate"]["passed"]),
            "point_inside_own": final["records"]["point_inside_count"],
            "paired_point_selection": final["pair"]["paired_point_selection_pass"],
            "pair_ranking": final["pair"]["pair_ranking_pass"],
            "mean_abs_logit": final["mean_abs_logit"],
        },
    )
    report["checkpoint"] = checkpoint
    report["selection"] = {
        "carried_into_h1r": False,
        "reason": (
            "H0-R is a 10-pair implementation proof; its weights are fitted to those pairs, so H1-R "
            "starts from the same clean Task 6G/6H initialization convention."
        ),
    }
    report["seconds"] = round(time.time() - started, 2)
    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    print(
        f"[task6h1.h0r] {report['verdict']} at pair step {global_step}: inside-own "
        f"{final['records']['point_inside_count']}/{final['records']['count']}, paired point "
        f"{final['pair']['paired_point_selection_pass']}/{final['pair']['pair_count']}, ranking "
        f"{final['pair']['pair_ranking_pass']}/{final['pair']['pair_count']}, own mass "
        f"{final['pair']['mean_own_mass']:.4f} vs cross {final['pair']['mean_cross_mass']:.4f}, "
        f"|logit| {final['mean_abs_logit']:.3f} (Task 6H: 16.32)",
        flush=True,
    )
    print(f"[task6h1.h0r] wrote {output}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if final["gate"]["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
