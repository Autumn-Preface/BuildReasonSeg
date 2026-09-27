"""Task 6I section 10: Stage I1 -- 240-pair mini-train through the refinement block.

One epoch = 240 canonical pair steps (deterministic order), at most 8 epochs under one corrected
cosine scheduler over the full pair-step budget (no mid-run tuning). Every epoch is validated on the
fixed 20 paired validation images (bounded pair ranking, paired point selection, attention masses)
and on the fixed independent 120 validation records (point-inside rate, target-cell top-1/top-5,
point error, spatial entropy, level and query-family breakdowns).

Selection (section 10, lexicographic): paired point-selection /20, then validation pair ranking /20,
then the 120-record point-inside-target rate. Failure of the gate on the best epoch is
`VISUAL_QUERY_REFINEMENT_NO_GENERALIZATION`.

Writes `evaluation/task6i_i1_training.json`, `evaluation/task6i_spatial_eval.json`,
`evaluation/task6i_representation.json` and `evaluation/task6i_attention_diagnostics.json`.
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
from buildreasonseg_mvp.dense_grounding import feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.query_refine import (  # noqa: E402
    attention_vs_localization,
    evaluate_refined_pair,
    refinement_representation_report,
    refined_point_record,
)
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_objective import (  # noqa: E402
    margin_vs_localization,
    pair_probability_summary,
    point_record_summary,
)

from task6c_train import validation_material  # noqa: E402
from task6i_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    gate_report,
    load_refinement_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    save_stage_checkpoint,
    validation_pair_dicts,
    write_json,
)

OUT = EVAL / "task6i_i1_training.json"
SPATIAL_OUT = EVAL / "task6i_spatial_eval.json"
REPRESENTATION_OUT = EVAL / "task6i_representation.json"
ATTENTION_OUT = EVAL / "task6i_attention_diagnostics.json"


def _selection_key(evaluation: dict) -> tuple:
    """Section 10: paired point-selection, pair ranking, then 120-record inside rate."""

    return (
        int(evaluation["validation_pairs"]["paired_point_selection_pass"]),
        int(evaluation["validation_pairs"]["pair_ranking_pass"]),
        float(evaluation["val"]["point_inside_target_rate"] or 0.0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=60)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    cfg, payload = load_refinement_config()
    grid = int(cfg["dense_grounding"]["grid"])
    eps = float(cfg["point_objective"]["eps"])
    epochs = int(args.epochs or cfg["i1"]["epochs"])
    gate_cfg = cfg["i1"]["gate"]

    triples = canonical_train_pairs(payload)
    pairs_per_epoch = len(triples)
    total_steps = max(1, epochs * pairs_per_epoch)

    val_samples, raw_pairs, _lookup, _audit = validation_material()
    validation_pairs = validation_pair_dicts(raw_pairs)
    validation_samples = {
        pair["a"]: data_mod.to_sample(pair["record_a"]) for pair in validation_pairs
    } | {pair["b"]: data_mod.to_sample(pair["record_b"]) for pair in validation_pairs}

    print(
        f"[task6i.i1] grid {grid}: {pairs_per_epoch} pairs/epoch x {epochs} epochs = "
        f"{total_steps} PAIR steps; val {len(val_samples)} records + {len(validation_pairs)} "
        f"paired images",
        flush=True,
    )

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg["training"].get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg["training"].get("visual_feature_cache_max_images", 512)),
        )
    first_sample = data_mod.to_sample(triples[0][1])
    probe_image = first_sample.image_rgb()
    probe_features, _cached = runtime.features_for(first_sample, probe_image)
    f256 = feature_tensor_for_grid(probe_features, grid)
    runtime.install_dense_head(in_channels=int(f256.shape[1]))  # frozen evidence only
    runtime.install_refinement_block(
        coarse_channels=int(feature_tensor_for_grid(probe_features, 64).shape[1]),
        fine_channels=int(f256.shape[1]),
    )
    trainables = runtime.freeze_for_refinement()
    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime, total_steps=total_steps, warmup_steps=int(cfg["optimizer"]["warmup_steps"])
    )

    report = {
        "_doc": (
            "Task 6I section 10. Stage I1: 240-pair counterfactual mini-train through the visual "
            "query refinement block with the frozen Task 6H.1 point-aligned bounded objective, at "
            "most 8 epochs, pair-step scheduler horizon, per-epoch validation on the fixed 20 "
            "paired validation images and the fixed independent 120 records."
        ),
        "task": "6I",
        "stage": "I1",
        "grid": grid,
        "trainables": trainables,
        "block": runtime.reports["refinement_block"],
        "frozen_task6g_head": runtime.reports["dense_head"],
        "optimizer_groups": optimizer_coverage(groups),
        "loss": {
            "reasoning_weight": 0.5,
            "classification_weight": 1.0,
            "cf_weight": 1.0,
            "eps": eps,
            "margin": None,
        },
        "budget": {
            "epochs_requested": epochs,
            "pairs_per_epoch": pairs_per_epoch,
            "total_pair_steps": total_steps,
            "scheduler_horizon": total_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
            "early_stop": None,
        },
        "validation_material": {
            "val_records": len(val_samples),
            "paired_images": len(validation_pairs),
        },
        "epochs": [],
    }

    def evaluate(tag: str, epoch: int) -> dict:
        pair_rows = []
        for pair in validation_pairs:
            pair_rows.append(
                evaluate_refined_pair(
                    runtime, pair, validation_samples[pair["a"]], validation_samples[pair["b"]],
                    with_hidden=False,
                )
            )
        validation_summary = pair_probability_summary(pair_rows)
        validation_summary["paired_point_selection_pass"] = sum(
            1 for row in pair_rows if row["point_selection_passed"]
        )
        for row in pair_rows:
            row.pop("record_a", None)
            row.pop("record_b", None)
        val_records = [refined_point_record(runtime, sample) for sample in val_samples]
        metrics = point_record_summary(val_records)
        gate = gate_report({"validation_pairs": validation_summary, "val": metrics}, gate_cfg, "i1")
        attention_summary = {
            "mean_pair_attention_own_mass": float(
                sum(row["attention_own_mass_mean"] for row in pair_rows) / max(len(pair_rows), 1)
            ),
            "mean_pair_attention_cross_mass": float(
                sum(row["attention_cross_mass_mean"] for row in pair_rows) / max(len(pair_rows), 1)
            ),
            "mean_attention_entropy": float(
                sum(record["attn_entropy"] for record in val_records) / max(len(val_records), 1)
            ),
            "mean_attention_target_mass": float(
                sum(record["attn_target_mass"] for record in val_records) / max(len(val_records), 1)
            ),
        }
        print(
            f"[{tag}] epoch {epoch} pair probe: ranking "
            f"{validation_summary['pair_ranking_pass']}/{validation_summary['pair_count']} paired "
            f"point {validation_summary['paired_point_selection_pass']}/"
            f"{validation_summary['pair_count']} own {validation_summary['mean_own_mass']:.4f} cross "
            f"{validation_summary['mean_cross_mass']:.4f} margin "
            f"{validation_summary['mean_own_minus_cross_probability_mass']:+.4f} | 120-record inside "
            f"{metrics['point_inside_target_rate']} top1 {metrics['target_cell_top1_count']} "
            f"entropy {metrics['mean_spatial_entropy']:.2f} attn own/cross "
            f"{attention_summary['mean_pair_attention_own_mass']:.3f}/"
            f"{attention_summary['mean_pair_attention_cross_mass']:.3f} gate {gate['passed']}",
            flush=True,
        )
        return {
            "epoch": epoch,
            "validation_pairs": validation_summary,
            "val": metrics,
            "attention": attention_summary,
            "gate": gate,
            "val_records": val_records,
            "pair_rows": pair_rows,
        }

    global_step = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        set_seed(int(cfg["seed"]) + epoch)
        epoch_ranking = 0
        epoch_margins: list[float] = []
        epoch_logits: list[float] = []
        for pair, record_a, record_b in triples:
            sample_a = data_mod.to_sample(record_a)
            sample_b = data_mod.to_sample(record_b)
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
            runtime.set_visual_cache_key(str(pair.image_id))
            result = runtime.refinement_pair_train_step(
                batch_a, batch_b, sample_a.target_mask(), sample_b.target_mask(), features,
                optimizer=optimizer,
            )
            scheduler.step()
            global_step += 1
            epoch_ranking += int(result["pair_preference_pass"])
            epoch_margins.append(result["region_masses"]["mean_margin"])
            epoch_logits.append(result["mean_abs_logit"])
            if global_step % int(args.log_every) == 0 or global_step == total_steps:
                entry = {
                    "epoch": epoch,
                    "global_pair_step": global_step,
                    "losses": result["losses"],
                    "region_masses": result["region_masses"],
                    "pair_preference_pass": result["pair_preference_pass"],
                    "mean_abs_logit": result["mean_abs_logit"],
                    "attention": result["attention"],
                    "lrs": lr_values(optimizer),
                }
                history.append(entry)
                print(
                    f"[task6i.i1] epoch {epoch} pair {global_step}/{total_steps} total "
                    f"{entry['losses']['total']:.4f} point {entry['losses']['point_ce_sum']:.4f} "
                    f"cf {entry['losses']['cf']:.4f} margin "
                    f"{entry['region_masses']['mean_margin']:+.4f} |logit| "
                    f"{entry['mean_abs_logit']:.3f} lr {entry['lrs'][0]:.3e}",
                    flush=True,
                )
            del result, batch_a, batch_b, image, features

        evaluation = evaluate("task6i.i1", epoch)
        evaluation["train_pair_ranking_pass"] = int(epoch_ranking)
        evaluation["train_pair_ranking_rate"] = float(epoch_ranking / max(pairs_per_epoch, 1))
        evaluation["train_mean_probability_margin"] = float(
            sum(epoch_margins) / max(len(epoch_margins), 1)
        )
        evaluation["train_mean_abs_logit"] = float(
            sum(epoch_logits) / max(len(epoch_logits), 1)
        )
        checkpoint = save_stage_checkpoint(
            cfg,
            "I1",
            f"i1_epoch{epoch}",
            runtime.model,
            global_step,
            metrics={
                "epoch": epoch,
                "paired_point_selection_pass": evaluation["validation_pairs"][
                    "paired_point_selection_pass"
                ],
                "pair_ranking_pass": evaluation["validation_pairs"]["pair_ranking_pass"],
                "val_point_inside_rate": evaluation["val"]["point_inside_target_rate"],
                "train_pair_ranking_rate": evaluation["train_pair_ranking_rate"],
                "train_mean_probability_margin": evaluation["train_mean_probability_margin"],
                "train_mean_abs_logit": evaluation["train_mean_abs_logit"],
            },
        )
        evaluation["checkpoint"] = checkpoint
        report["epochs"].append(evaluation)

    report["history"] = history
    best = max(report["epochs"], key=_selection_key)
    report["selection"] = {
        "priority": [
            "paired point-selection /20",
            "validation pair-ranking /20",
            "120-record point-inside-target rate",
        ],
        "selected_epoch": int(best["epoch"]),
        "selected_checkpoint": best["checkpoint"],
        "key": list(_selection_key(best)),
        "per_epoch_keys": [
            {"epoch": int(entry["epoch"]), "key": list(_selection_key(entry))}
            for entry in report["epochs"]
        ],
        "stop_reason": "budget_exhausted",
    }

    # ---- diagnostics on the BEST checkpoint (section 11) ----------------------
    load_report = load_checkpoint(Path(best["checkpoint"]["path"]), runtime.model)
    rep_val_records = [refined_point_record(runtime, sample, with_hidden=True) for sample in val_samples]
    rep_pair_rows = [
        evaluate_refined_pair(
            runtime, pair, validation_samples[pair["a"]], validation_samples[pair["b"]],
            with_hidden=True,
        )
        for pair in validation_pairs
    ]
    rep_paired_records = []
    for row in rep_pair_rows:
        rep_paired_records.extend([row["record_a"], row["record_b"]])
    representation = refinement_representation_report(
        rep_paired_records, rep_val_records, validation_pairs
    )
    representation["bounded_pair"] = {
        "mean_own_mass": pair_probability_summary(rep_pair_rows)["mean_own_mass"],
        "mean_cross_mass": pair_probability_summary(rep_pair_rows)["mean_cross_mass"],
        "mean_own_minus_cross_probability_mass": pair_probability_summary(rep_pair_rows)[
            "mean_own_minus_cross_probability_mass"
        ],
        "pair_ranking_pass": pair_probability_summary(rep_pair_rows)["pair_ranking_pass"],
        "mean_spatial_entropy": pair_probability_summary(rep_pair_rows)["mean_spatial_entropy"],
        "mean_same_image_probability_map_overlap": pair_probability_summary(rep_pair_rows)[
            "mean_same_image_probability_map_overlap"
        ],
        "mean_same_image_point_distance": pair_probability_summary(rep_pair_rows)[
            "mean_same_image_point_distance"
        ],
    }
    representation["margin_vs_localization"] = margin_vs_localization(rep_pair_rows)
    representation["attention_vs_localization"] = attention_vs_localization(rep_pair_rows)
    representation["task6h_reference"] = {
        "mean_abs_logit_after_h0": 16.32,
        "probability_margin_after_h0": 0.1124,
        "point_inside_rate_h0": 0.15,
        "paired_point_selection_h0": 0,
        "task6h1_h0r_reference": {
            "point_inside_own": "7/20",
            "paired_point": "2/10",
            "bounded_ranking": "8/10",
            "own_mass": 0.1835,
            "cross_mass": 0.0456,
        },
        "source": "evaluation/task6h_h0_audit.json + evaluation/task6h1_h0r_overfit.json",
    }

    report["final"] = {
        "epoch": int(best["epoch"]),
        "train_pair_ranking_rate": best["train_pair_ranking_rate"],
        "train_mean_probability_margin": best["train_mean_probability_margin"],
        "train_mean_abs_logit": best["train_mean_abs_logit"],
        "validation_pairs": best["validation_pairs"],
        "val": best["val"],
        "attention": best["attention"],
        "gate": best["gate"],
        "checkpoint": best["checkpoint"],
        "checkpoint_load": load_report,
    }
    report["verdict"] = (
        "I1_GATE_PASS" if best["gate"]["passed"] else "VISUAL_QUERY_REFINEMENT_NO_GENERALIZATION"
    )
    report["seconds"] = round(time.time() - started, 2)

    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    write_json(
        SPATIAL_OUT,
        {
            "_doc": (
                "Task 6I sections 10/11. Point-localization and bounded pair-preference metrics for "
                "the selected I1 model: the point comes from `argmax(refined heatmap logits)` only; "
                "GT is scoring only. Pair metrics use the fixed 20 paired validation images (one "
                "canonical pair each); record metrics use the fixed independent 120-record "
                "validation set."
            ),
            "task": "6I",
            "grid": grid,
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "validation_pairs": best["validation_pairs"],
            "val": best["val"],
            "val_records": best["val_records"],
            "paired_rows": best["pair_rows"],
        },
    )
    write_json(
        REPRESENTATION_OUT,
        {
            "_doc": (
                "Task 6I section 11. q0 vs q1 comparison at the best I1 checkpoint: does one visual "
                "refinement step turn the weak instruction signal in q0 into a target-specific q1?"
            ),
            "task": "6I",
            "grid": grid,
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "representation": representation,
        },
    )
    write_json(
        ATTENTION_OUT,
        {
            "_doc": (
                "Task 6I section 7. Cross-attention diagnostics at the best I1 checkpoint: 1x4096 "
                "attention entropy, top-10 mass and own/cross target mass. GT masks are used for "
                "scoring only and never enter the model."
            ),
            "task": "6I",
            "grid": grid,
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "validation_pairs": best["validation_pairs"],
            "attention": best["attention"],
            "val_attention": {
                "records": [
                    {
                        "sample_id": record["sample_id"],
                        "image_id": record["image_id"],
                        "level": record["level"],
                        "query_family": record["query_family"],
                        "attn_entropy": record["attn_entropy"],
                        "attn_top1_cell": record["attn_top1_cell"],
                        "attn_top1_centre": record["attn_top1_centre"],
                        "attn_top10_mass": record["attn_top10_mass"],
                        "attn_target_mass": record["attn_target_mass"],
                        "point_inside_own": record["point_inside_own"],
                    }
                    for record in best["val_records"]
                ]
            },
            "pair_attention": [
                {
                    "image_id": row["image_id"],
                    "a": row["a"],
                    "b": row["b"],
                    "attention_entropy_a": row["attention_entropy_a"],
                    "attention_entropy_b": row["attention_entropy_b"],
                    "attention_own_mass_a": row["attention_own_mass_a"],
                    "attention_own_mass_b": row["attention_own_mass_b"],
                    "attention_cross_mass_a": row["attention_cross_mass_a"],
                    "attention_cross_mass_b": row["attention_cross_mass_b"],
                    "point_selection_passed": row["point_selection_passed"],
                }
                for row in best["pair_rows"]
            ],
        },
    )
    print(
        f"[task6i.i1] {report['verdict']}: selected epoch {best['epoch']} paired-point "
        f"{best['validation_pairs']['paired_point_selection_pass']}/"
        f"{best['validation_pairs']['pair_count']} ranking "
        f"{best['validation_pairs']['pair_ranking_pass']}/{best['validation_pairs']['pair_count']} "
        f"120-record inside {best['val']['point_inside_target_rate']} margin "
        f"{best['validation_pairs']['mean_own_minus_cross_probability_mass']:+.4f} |logit| "
        f"{best['train_mean_abs_logit']:.3f} attn own/cross "
        f"{best['attention']['mean_pair_attention_own_mass']:.3f}/"
        f"{best['attention']['mean_pair_attention_cross_mass']:.3f}",
        flush=True,
    )
    print(f"[task6i.i1] wrote {output} + spatial/representation/attention artifacts", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if best["gate"]["passed"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
