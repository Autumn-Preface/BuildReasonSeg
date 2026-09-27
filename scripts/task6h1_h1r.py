"""Task 6H.1 sections 15-19: Stage H1-R -- 240-pair mini-train with the corrected objective.

One epoch = 240 canonical pair steps (deterministic order), at most 8 epochs under one corrected
cosine scheduler over the full pair-step budget (no mid-run tuning). Every epoch is validated on the
fixed 20 paired validation images (bounded pair ranking and paired point selection) and on the fixed
independent 120 validation records (point-inside rate, target-cell top-1/top-5, point error, spatial
entropy, level and query-family breakdowns).

Selection (section 17, lexicographic): paired point-selection /20, then validation pair ranking /20,
then the 120-record point-inside-target rate. Failure of the gate on the best epoch is
`BOUNDED_COUNTERFACTUAL_GROUNDING_FAILED`.

Writes `evaluation/task6h1_h1r_training.json`, `evaluation/task6h1_spatial_eval.json` and
`evaluation/task6h1_representation.json`.
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
from buildreasonseg_mvp.dense_eval import representation_report  # noqa: E402
from buildreasonseg_mvp.dense_grounding import feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402
from buildreasonseg_mvp.spatial_objective import (  # noqa: E402
    evaluate_point_pair,
    margin_vs_localization,
    pair_probability_summary,
    point_record,
    point_record_summary,
)

from task6c_train import validation_material  # noqa: E402
from task6h1_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    gate_report,
    load_point_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    save_stage_checkpoint,
    validation_pair_dicts,
    write_json,
)

OUT = EVAL / "task6h1_h1r_training.json"
SPATIAL_OUT = EVAL / "task6h1_spatial_eval.json"
REPRESENTATION_OUT = EVAL / "task6h1_representation.json"


def _selection_key(evaluation: dict) -> tuple:
    """Section 17: paired point-selection, pair ranking, then 120-record inside rate."""

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
    cfg, payload = load_point_config()
    grid = int(cfg["dense_grounding"]["grid"])
    eps = float(cfg["point_objective"]["eps"])
    epochs = int(args.epochs or cfg["h1r"]["epochs"])
    gate_cfg = cfg["h1r"]["gate"]

    triples = canonical_train_pairs(payload)
    pairs_per_epoch = len(triples)
    total_steps = max(1, epochs * pairs_per_epoch)

    val_samples, raw_pairs, _lookup, _audit = validation_material()
    validation_pairs = validation_pair_dicts(raw_pairs)
    validation_samples = {
        pair["a"]: data_mod.to_sample(pair["record_a"]) for pair in validation_pairs
    } | {pair["b"]: data_mod.to_sample(pair["record_b"]) for pair in validation_pairs}

    print(
        f"[task6h1.h1r] grid {grid}: {pairs_per_epoch} pairs/epoch x {epochs} epochs = "
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
    runtime.install_dense_head(in_channels=int(feature_tensor_for_grid(probe_features, grid).shape[1]))
    trainables = runtime.freeze_for_point_objective()
    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime, total_steps=total_steps, warmup_steps=int(cfg["optimizer"]["warmup_steps"])
    )

    report = {
        "_doc": (
            "Task 6H.1 sections 15-19. Stage H1-R: 240-pair counterfactual mini-train with the "
            "corrected point-aligned bounded objective (Task 6G/6H architecture frozen), at most 8 "
            "epochs, pair-step scheduler horizon, per-epoch validation on the fixed 20 paired "
            "validation images and the fixed independent 120 records."
        ),
        "task": "6H.1",
        "stage": "H1-R",
        "grid": grid,
        "trainables": trainables,
        "head": runtime.reports["dense_head"],
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
                evaluate_point_pair(
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
        val_records = [point_record(runtime, sample) for sample in val_samples]
        metrics = point_record_summary(val_records)
        gate = gate_report({"validation_pairs": validation_summary, "val": metrics}, gate_cfg, "h1r")
        print(
            f"[{tag}] epoch {epoch} pair probe: ranking "
            f"{validation_summary['pair_ranking_pass']}/{validation_summary['pair_count']} paired "
            f"point {validation_summary['paired_point_selection_pass']}/"
            f"{validation_summary['pair_count']} own {validation_summary['mean_own_mass']:.4f} cross "
            f"{validation_summary['mean_cross_mass']:.4f} margin "
            f"{validation_summary['mean_own_minus_cross_probability_mass']:+.4f} | 120-record inside "
            f"{metrics['point_inside_target_rate']} top1 {metrics['target_cell_top1_count']} "
            f"entropy {metrics['mean_spatial_entropy']:.2f} gate {gate['passed']}",
            flush=True,
        )
        return {
            "epoch": epoch,
            "validation_pairs": validation_summary,
            "val": metrics,
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
            result = runtime.bounded_pair_train_step(
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
                    "lrs": lr_values(optimizer),
                }
                history.append(entry)
                print(
                    f"[task6h1.h1r] epoch {epoch} pair {global_step}/{total_steps} total "
                    f"{entry['losses']['total']:.4f} point {entry['losses']['point_ce_sum']:.4f} "
                    f"cf {entry['losses']['cf']:.4f} margin "
                    f"{entry['region_masses']['mean_margin']:+.4f} |logit| "
                    f"{entry['mean_abs_logit']:.3f} lr {entry['lrs'][0]:.3e}",
                    flush=True,
                )
            del result, batch_a, batch_b, image, features

        evaluation = evaluate("task6h1.h1r", epoch)
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
            "H1R",
            f"h1r_epoch{epoch}",
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

    # ---- diagnostics on the BEST checkpoint (section 19) ----------------------
    load_report = load_checkpoint(Path(best["checkpoint"]["path"]), runtime.model)
    rep_val_records = [point_record(runtime, sample, with_hidden=True) for sample in val_samples]
    rep_pair_rows = [
        evaluate_point_pair(
            runtime, pair, validation_samples[pair["a"]], validation_samples[pair["b"]],
            with_hidden=True,
        )
        for pair in validation_pairs
    ]
    rep_paired_records = []
    for row in rep_pair_rows:
        rep_paired_records.extend([row["record_a"], row["record_b"]])
    representation = representation_report(rep_paired_records, rep_val_records, validation_pairs)
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
    representation["task6h_reference"] = {
        "mean_abs_logit_after_h0": 16.32,
        "probability_margin_after_h0": 0.1124,
        "point_inside_rate_h0": 0.15,
        "paired_point_selection_h0": 0,
        "source": "evaluation/task6h_h0_audit.json",
    }

    report["final"] = {
        "epoch": int(best["epoch"]),
        "train_pair_ranking_rate": best["train_pair_ranking_rate"],
        "train_mean_probability_margin": best["train_mean_probability_margin"],
        "train_mean_abs_logit": best["train_mean_abs_logit"],
        "validation_pairs": best["validation_pairs"],
        "val": best["val"],
        "gate": best["gate"],
        "checkpoint": best["checkpoint"],
        "checkpoint_load": load_report,
    }
    report["verdict"] = (
        "H1R_GATE_PASS" if best["gate"]["passed"] else "BOUNDED_COUNTERFACTUAL_GROUNDING_FAILED"
    )
    report["seconds"] = round(time.time() - started, 2)

    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    write_json(
        SPATIAL_OUT,
        {
            "_doc": (
                "Task 6H.1 sections 16/19. Point-localization and bounded pair-preference metrics for "
                "the selected H1-R model: the point comes from `argmax(heatmap_logits)` only; GT is "
                "scoring only. Pair metrics use the fixed 20 paired validation images (one canonical "
                "pair each); record metrics use the fixed independent 120-record validation set."
            ),
            "task": "6H.1",
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
                "Task 6H.1 section 19. Query/heatmap diagnostics at the best H1-R checkpoint, "
                "compared with the frozen Task 6H values: did the bounded point-aligned objective "
                "convert same-image instruction differences into correct spatial selection rather "
                "than magnitude-only separation?"
            ),
            "task": "6H.1",
            "grid": grid,
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "representation": representation,
        },
    )
    print(
        f"[task6h1.h1r] {report['verdict']}: selected epoch {best['epoch']} paired-point "
        f"{best['validation_pairs']['paired_point_selection_pass']}/"
        f"{best['validation_pairs']['pair_count']} ranking "
        f"{best['validation_pairs']['pair_ranking_pass']}/{best['validation_pairs']['pair_count']} "
        f"120-record inside {best['val']['point_inside_target_rate']} margin "
        f"{best['validation_pairs']['mean_own_minus_cross_probability_mass']:+.4f} |logit| "
        f"{best['train_mean_abs_logit']:.3f}",
        flush=True,
    )
    print(f"[task6h1.h1r] wrote {output} + spatial/representation artifacts", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if best["gate"]["passed"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
