"""Task 6I section 8: Stage I0 -- 10-pair overfit of the refined path.

Clean initialization (never the failed Task 6H.1 H0-R weights), the same 10 canonical pairs, at
most 1500 **pair** optimizer steps of the frozen Task 6H.1 objective through the refinement block:

    L_total = 0.5*(L_reasoning_A+L_reasoning_B) + 1.0*(L_point_A+L_point_B) + 1.0*L_cf

with `L_point` the 65,536-class spatial cross-entropy of the target point-cell of the **refined**
heatmap and `L_cf` the bounded own-vs-cross probability-mass preference. Evaluated at
250/500/750/1000/1500 pair steps through the real `argmax` path, plus the section 7 attention
diagnostics and same-image q0/q1 separation.

I0 gate: point-inside-own >= 18/20, paired point-selection >= 9/10, bounded pair ranking >= 9/10,
mean normalized point error < 0.08. Failure after the implementation audit is
`VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT`.

Writes `evaluation/task6i_i0_overfit.json`.
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
from buildreasonseg_mvp.query_refine import evaluate_refined_pair  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_objective import pair_probability_summary, point_record_summary  # noqa: E402

from task6i_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    gate_report,
    h0r_pairs,
    load_refinement_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    pair_dicts_for_eval,
    save_stage_checkpoint,
    write_json,
)

OUT = EVAL / "task6i_i0_overfit.json"


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
    cfg, payload = load_refinement_config()
    grid = int(cfg["dense_grounding"]["grid"])
    i0_cfg = cfg["i0"]
    max_steps = int(args.max_pair_steps or i0_cfg["max_pair_steps"])
    gate_cfg = i0_cfg["gate"]

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
    f256 = feature_tensor_for_grid(probe_features, grid)
    runtime.install_dense_head(in_channels=int(f256.shape[1]))  # frozen evidence only
    runtime.install_refinement_block(
        coarse_channels=int(feature_tensor_for_grid(probe_features, 64).shape[1]),
        fine_channels=int(f256.shape[1]),
    )
    trainables = runtime.freeze_for_refinement()

    print(
        f"[task6i.i0] grid {grid}: {len(triples)} pairs / {len(samples)} records, "
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
            "Task 6I section 8. Stage I0: overfit the same 10 canonical pairs as Task 6H H0/H0-R "
            "from a clean initialization, now through the single visual query refinement block "
            "(q0 cross-attends the frozen 64x64 SAM2 embedding once; q1 scores the 256x256 "
            "feature). The scalar objective is the frozen Task 6H.1 point-cell CE + bounded "
            "probability-mass pair loss."
        ),
        "task": "6I",
        "stage": "I0",
        "grid": grid,
        "trainables": trainables,
        "block": runtime.reports["refinement_block"],
        "frozen_task6g_head": runtime.reports["dense_head"],
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
                evaluate_refined_pair(runtime, pair_dicts[pair.a], sample_a, sample_b, with_hidden=False)
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
        attention = {
            "mean_entropy": float(
                sum(record["attn_entropy"] for record in records) / max(len(records), 1)
            ),
            "mean_top10_mass": float(
                sum(record["attn_top10_mass"] for record in records) / max(len(records), 1)
            ),
            "mean_target_mass": float(
                sum(record["attn_target_mass"] for record in records) / max(len(records), 1)
            ),
            "pair_own_mass": float(
                sum(row["attention_own_mass_mean"] for row in rows) / max(len(rows), 1)
            ),
            "pair_cross_mass": float(
                sum(row["attention_cross_mass_mean"] for row in rows) / max(len(rows), 1)
            ),
        }
        separation = {
            "q0_l2": float(sum(row["same_image_q0"]["l2"] for row in rows) / max(len(rows), 1)),
            "q1_l2": float(sum(row["same_image_q1"]["l2"] for row in rows) / max(len(rows), 1)),
            "q0_cosine": float(sum(row["same_image_q0"]["cosine"] for row in rows) / max(len(rows), 1)),
            "q1_cosine": float(sum(row["same_image_q1"]["cosine"] for row in rows) / max(len(rows), 1)),
            "point_distance": pair_metrics.get("mean_same_image_point_distance"),
        }
        gate = gate_report(
            {
                "point_inside_own": int(record_metrics["point_inside_count"]),
                "pair": pair_metrics,
                "mean_normalized_error": record_metrics["mean_normalized_point_error"],
            },
            gate_cfg,
            "i0",
        )
        return {
            "pair_step": step,
            "pair": pair_metrics,
            "records": record_metrics,
            "mean_abs_logit": mean_abs_logit,
            "attention": attention,
            "separation": separation,
            "record_rows": records,
            "pair_rows": rows,
            "gate": gate,
        }

    global_step = 0
    stop_reason = "budget_exhausted"
    while global_step < max_steps:
        pair, sample_a, sample_b, batch_a, batch_b, features = prepared[global_step % len(prepared)]
        runtime.set_visual_cache_key(str(pair.image_id))
        result = runtime.refinement_pair_train_step(
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
                "attention": result["attention"],
                "lrs": lr_values(optimizer),
                "vram_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            }
            report["history"].append(entry)
            print(
                f"[task6i.i0] pair step {global_step}/{max_steps} total "
                f"{entry['losses']['total']:.4f} reason {entry['losses']['reasoning_ce']:.4f} "
                f"point {entry['losses']['point_ce_sum']:.4f} cf {entry['losses']['cf']:.4f} "
                f"own {entry['region_masses']['p_aa']:.4f}/{entry['region_masses']['p_bb']:.4f} "
                f"cross {entry['region_masses']['p_ab']:.4f}/{entry['region_masses']['p_ba']:.4f} "
                f"|logit| {entry['mean_abs_logit']:.3f} attn-tgt "
                f"{entry['attention']['target_mass_a']:.3f}/{entry['attention']['target_mass_b']:.3f} "
                f"lr {entry['lrs'][0]:.3e}",
                flush=True,
            )
        if global_step % int(args.eval_every) == 0 or global_step == max_steps:
            evaluation = evaluate(global_step)
            report["evaluations"].append(evaluation)
            print(
                f"[task6i.i0] step {global_step} argmax path: inside-own "
                f"{evaluation['records']['point_inside_count']}/{evaluation['records']['count']} "
                f"top1 {evaluation['records']['target_cell_top1_count']} top5 "
                f"{evaluation['records']['target_cell_top5_count']} paired point "
                f"{evaluation['pair']['paired_point_selection_pass']}/{evaluation['pair']['pair_count']} "
                f"ranking {evaluation['pair']['pair_ranking_pass']}/{evaluation['pair']['pair_count']} "
                f"norm-err {evaluation['records']['mean_normalized_point_error']:.4f} "
                f"margin {evaluation['pair']['mean_own_minus_cross_probability_mass']:+.4f} "
                f"|logit| {evaluation['mean_abs_logit']:.3f} attn entropy "
                f"{evaluation['attention']['mean_entropy']:.2f} attn own/cross "
                f"{evaluation['attention']['pair_own_mass']:.3f}/{evaluation['attention']['pair_cross_mass']:.3f} "
                f"q0/q1 l2 {evaluation['separation']['q0_l2']:.4f}/{evaluation['separation']['q1_l2']:.4f} "
                f"gate {evaluation['gate']['passed']}",
                flush=True,
            )
            if evaluation["gate"]["passed"] and global_step >= int(args.min_steps_before_stop):
                stop_reason = "gate_passed"
                break

    final = report["evaluations"][-1]
    report["final"] = {**final, "stop_reason": stop_reason}
    report["verdict"] = "I0_PASS" if final["gate"]["passed"] else "VISUAL_QUERY_REFINEMENT_FAILED_AT_OVERFIT"
    report["no_gt_leakage_audit"] = {
        "pair_path_inputs": (
            "image + instruction + constant [BOX] + one shared frozen SAM2 feature set "
            "(64x64 embedding and 256x256 high-res level)"
        ),
        "gt_used_for": ["target point-cell supervision", "bounded region mass", "evaluation scoring"],
        "gt_enters_inference": False,
        "inference_operation": "argmax(heatmap_logits) where heatmap = dot(q1, K256)/sqrt(256)",
    }
    checkpoint = save_stage_checkpoint(
        cfg,
        "I0",
        "i0_final",
        runtime.model,
        global_step,
        metrics={
            "gate_passed": bool(final["gate"]["passed"]),
            "point_inside_own": final["records"]["point_inside_count"],
            "paired_point_selection": final["pair"]["paired_point_selection_pass"],
            "pair_ranking": final["pair"]["pair_ranking_pass"],
            "mean_normalized_error": final["records"]["mean_normalized_point_error"],
            "mean_abs_logit": final["mean_abs_logit"],
        },
    )
    report["checkpoint"] = checkpoint
    report["selection"] = {
        "carried_into_i1": False,
        "reason": (
            "I0 is a 10-pair implementation proof; its weights are fitted to those pairs, so I1 "
            "starts from the same clean initialization convention."
        ),
    }
    report["seconds"] = round(time.time() - started, 2)
    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    print(
        f"[task6i.i0] {report['verdict']} at pair step {global_step}: inside-own "
        f"{final['records']['point_inside_count']}/{final['records']['count']}, paired point "
        f"{final['pair']['paired_point_selection_pass']}/{final['pair']['pair_count']}, ranking "
        f"{final['pair']['pair_ranking_pass']}/{final['pair']['pair_count']}, norm-err "
        f"{final['records']['mean_normalized_point_error']:.4f}, own mass "
        f"{final['pair']['mean_own_mass']:.4f} vs cross {final['pair']['mean_cross_mass']:.4f}, "
        f"|logit| {final['mean_abs_logit']:.3f} (Task 6H: 16.32), attn own/cross "
        f"{final['attention']['pair_own_mass']:.3f}/{final['attention']['pair_cross_mass']:.3f}",
        flush=True,
    )
    print(f"[task6i.i0] wrote {output}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if final["gate"]["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
