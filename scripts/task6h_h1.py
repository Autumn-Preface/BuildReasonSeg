"""Task 6H sections 14-17: Stage H1 -- 240-pair counterfactual mini-train.

One epoch = 240 pair optimizer steps in the canonical Task 6C `P` order, at most 8 epochs under one
corrected cosine scheduler over the full pair-step budget (no mid-run recipe change). Every epoch is
validated on the fixed 20 paired validation images (primary target-selection probe: pair ranking and
paired point selection) and on the fixed independent 120 validation records (per-record
localization); no pairs are invented inside the 120-record set.

Model selection (section 15, lexicographic): paired point-selection /20, then validation pair-ranking
/20, then the 120-record point-inside-target rate. Failure of the gate on the best epoch is
`COUNTERFACTUAL_DENSE_GROUNDING_FAILED`.

Writes `evaluation/task6h_h1_training.json`, `evaluation/task6h_spatial_eval.json` and
`evaluation/task6h_representation.json`.
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
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.dense_eval import predict_heatmap, representation_report, spatial_metrics  # noqa: E402
from buildreasonseg_mvp.dense_grounding import feature_tensor_for_grid  # noqa: E402
from buildreasonseg_mvp.pair_eval import (  # noqa: E402
    evaluate_pair,
    gt_point_distance_correlation,
    pair_summary,
)
from buildreasonseg_mvp.runtime import build_runtime, set_seed  # noqa: E402

from task6c_train import validation_material  # noqa: E402
from task6h_common import (  # noqa: E402
    EVAL,
    canonical_train_pairs,
    gate_report,
    load_pair_config,
    lr_values,
    optimizer_and_scheduler,
    optimizer_coverage,
    pair_dicts_for_eval,
    save_stage_checkpoint,
    validation_pair_dicts,
    write_json,
)

OUT = EVAL / "task6h_h1_training.json"
SPATIAL_OUT = EVAL / "task6h_spatial_eval.json"
REPRESENTATION_OUT = EVAL / "task6h_representation.json"


def _selection_key(evaluation: dict) -> tuple:
    """Section 15 lexicographic: paired point-selection, pair ranking, 120-record inside rate."""

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
    cfg, payload = load_pair_config()
    grid = int(cfg["dense_grounding"]["grid"])
    margin = float(cfg["counterfactual"]["margin"])
    epochs = int(args.epochs or cfg["h1"]["epochs"])
    gate_cfg = cfg["h1"]["gate"]

    triples = canonical_train_pairs(payload)
    pairs_per_epoch = len(triples)
    total_steps = max(1, epochs * pairs_per_epoch)

    val_samples, raw_pairs, _lookup, _audit = validation_material()
    validation_pairs = validation_pair_dicts(raw_pairs)
    validation_samples = {
        pair["a"]: data_mod.to_sample(pair["record_a"]) for pair in validation_pairs
    } | {pair["b"]: data_mod.to_sample(pair["record_b"]) for pair in validation_pairs}

    print(
        f"[task6h.h1] grid {grid}: {pairs_per_epoch} pairs/epoch x {epochs} epochs = "
        f"{total_steps} PAIR optimizer steps; val {len(val_samples)} records + "
        f"{len(validation_pairs)} paired images; margin {margin}",
        flush=True,
    )

    set_seed(int(cfg["seed"]))
    runtime = build_runtime(cfg, verbose=True)
    if bool(cfg.get("training", {}).get("visual_feature_cache", False)):
        runtime.install_visual_cache(
            enabled=True,
            max_images=int(cfg.get("training", {}).get("visual_feature_cache_max_images", 512)),
        )
    first_sample = data_mod.to_sample(triples[0][1])
    probe_image = first_sample.image_rgb()
    probe_features, _cached = runtime.features_for(first_sample, probe_image)
    in_channels = int(feature_tensor_for_grid(probe_features, grid).shape[1])
    runtime.install_dense_head(in_channels=in_channels)
    trainables = runtime.freeze_for_counterfactual()
    optimizer, scheduler, groups = optimizer_and_scheduler(
        runtime, total_steps=total_steps, warmup_steps=int(cfg["optimizer"]["warmup_steps"])
    )

    report = {
        "_doc": (
            "Task 6H sections 14-17. Stage H1: 240-pair counterfactual mini-train (Task 6C `P` "
            "subset, canonical order) with the frozen Task 6G dense architecture, at most 8 epochs, "
            "pair-step scheduler horizon, per-epoch validation on the fixed 20 paired validation "
            "images and the fixed independent 120 validation records. No SAM2 training and no "
            "mid-run recipe change."
        ),
        "task": "6H",
        "stage": "H1",
        "grid": grid,
        "pair_manifest_sha256": _manifest_hash(),
        "trainables": trainables,
        "head": runtime.reports["dense_head"],
        "optimizer_groups": optimizer_coverage(groups),
        "loss": {
            "reasoning_weight": 0.5,
            "heatmap_weight": 1.0,
            "cf_weight": 2.0,
            "cf_margin": margin,
        },
        "budget": {
            "epochs_requested": epochs,
            "pairs_per_epoch": pairs_per_epoch,
            "total_pair_steps": total_steps,
            "scheduler_horizon": total_steps,
            "warmup_steps": int(cfg["optimizer"]["warmup_steps"]),
            "early_stop": None,
            "note": "the full budget ran; section 14 specifies no early stopping",
        },
        "validation_material": {
            "val_records": len(val_samples),
            "paired_images": len(validation_pairs),
            "pairs_invented_in_120": 0,
        },
        "epochs": [],
    }

    def evaluate(tag: str, epoch: int) -> dict:
        pair_rows = []
        for pair in validation_pairs:
            pair_rows.append(
                evaluate_pair(
                    runtime, pair, validation_samples[pair["a"]], validation_samples[pair["b"]],
                    with_heatmap=True,
                )
            )
        validation_summary = pair_summary(pair_rows)
        # Strip the in-memory binary heatmaps (kept as the pair scalar `same_image_heatmap_iou`)
        # before serialization, so the per-epoch artifact stays small.
        for row in pair_rows:
            row["record_a"].pop("heatmap_binary", None)
            row["record_b"].pop("heatmap_binary", None)
        val_records = [predict_heatmap(runtime, sample) for sample in val_samples]
        metrics = spatial_metrics(val_records)
        gate = gate_report(
            {"validation_pairs": validation_summary, "val": metrics}, gate_cfg, "h1"
        )
        print(
            f"[{tag}] epoch {epoch} pair probe: ranking "
            f"{validation_summary['pair_ranking_pass']}/{validation_summary['pair_count']} strict "
            f"{validation_summary['strict_margin_pass']} paired point "
            f"{validation_summary['paired_point_selection_pass']}/{validation_summary['pair_count']} "
            f"margin {validation_summary['mean_own_minus_cross_margin']:+.4f} point-dist "
            f"{validation_summary['mean_same_image_point_distance']} heatmapIoU "
            f"{validation_summary['mean_same_image_heatmap_iou']} | 120-record inside "
            f"{metrics['point_inside_target_rate']} dice {metrics['mean_heatmap_dice']} gate "
            f"{gate['passed']}",
            flush=True,
        )
        return {
            "epoch": epoch,
            "validation_pairs": validation_summary,
            "val": metrics,
            "gate": gate,
            "val_records": val_records,
            "pair_rows": [_strip_pair(row) for row in pair_rows],
        }

    global_step = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        set_seed(int(cfg["seed"]) + epoch)
        epoch_margins: list[float] = []
        epoch_ranking = 0
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
            result = runtime.pair_train_step(
                batch_a, batch_b, sample_a.target_mask(), sample_b.target_mask(), features,
                optimizer=optimizer,
            )
            scheduler.step()
            global_step += 1
            epoch_margins.append(result["region_scores"]["mean_margin"])
            epoch_ranking += int(result["pair_ranking_pass"])
            if global_step % int(args.log_every) == 0 or global_step == total_steps:
                entry = {
                    "epoch": epoch,
                    "global_pair_step": global_step,
                    "losses": result["losses"],
                    "region_scores": result["region_scores"],
                    "pair_ranking_pass": result["pair_ranking_pass"],
                    "lrs": lr_values(optimizer),
                }
                history.append(entry)
                print(
                    f"[task6h.h1] epoch {epoch} pair {global_step}/{total_steps} total "
                    f"{entry['losses']['total']:.4f} reason {entry['losses']['reasoning_ce']:.4f} "
                    f"heat {entry['losses']['heatmap_sum']:.4f} cf {entry['losses']['cf']:.4f} "
                    f"margin {entry['region_scores']['mean_margin']:+.4f} lr "
                    f"{entry['lrs'][0]:.3e}",
                    flush=True,
                )
            del result, batch_a, batch_b, image, features

        evaluation = evaluate("task6h.h1", epoch)
        evaluation["train_pair_ranking_pass"] = int(epoch_ranking)
        evaluation["train_pair_ranking_rate"] = float(epoch_ranking / max(pairs_per_epoch, 1))
        evaluation["train_mean_margin"] = float(
            sum(epoch_margins) / max(len(epoch_margins), 1)
        )
        checkpoint = save_stage_checkpoint(
            cfg,
            "H1",
            f"h1_epoch{epoch}",
            runtime.model,
            global_step,
            metrics={
                "epoch": epoch,
                "paired_point_selection_pass": evaluation["validation_pairs"]["paired_point_selection_pass"],
                "pair_ranking_pass": evaluation["validation_pairs"]["pair_ranking_pass"],
                "val_point_inside_rate": evaluation["val"]["point_inside_target_rate"],
                "train_pair_ranking_rate": evaluation["train_pair_ranking_rate"],
                "train_mean_margin": evaluation["train_mean_margin"],
            },
        )
        evaluation["checkpoint"] = checkpoint
        report["epochs"].append(evaluation)

    report["history"] = history
    best = max(report["epochs"], key=_selection_key)
    report["selection"] = {
        "priority": ["paired point-selection /20", "validation pair-ranking /20", "120-record inside rate"],
        "selected_epoch": int(best["epoch"]),
        "selected_checkpoint": best["checkpoint"],
        "key": list(_selection_key(best)),
        "per_epoch_keys": [
            {"epoch": int(entry["epoch"]), "key": list(_selection_key(entry))}
            for entry in report["epochs"]
        ],
        "stop_reason": "budget_exhausted",
    }

    # ---- diagnostics on the BEST checkpoint (section 17) ----------------------
    load_report = load_checkpoint(Path(best["checkpoint"]["path"]), runtime.model)
    rep_val_records = [predict_heatmap(runtime, sample, with_hidden=True) for sample in val_samples]
    rep_pair_rows = [
        evaluate_pair(
            runtime, pair, validation_samples[pair["a"]], validation_samples[pair["b"]],
            with_heatmap=True, with_hidden=True,
        )
        for pair in validation_pairs
    ]
    rep_paired_records = []
    for row in rep_pair_rows:
        rep_paired_records.extend([row["record_a"], row["record_b"]])
    representation = representation_report(rep_paired_records, rep_val_records, validation_pairs)
    representation["pair_distance_correlations"] = gt_point_distance_correlation(rep_pair_rows)
    representation["pair_margin_vs_localization"] = _margin_vs_localization(rep_pair_rows)
    representation["task6f_reference"]["hidden_distance_vs_gt_box_distance_correlation"] = 0.476
    representation["task6g_reference"] = {
        "g0_point_inside_rate": 0.10,
        "g0_heatmap_dice": 0.125,
        "g0_pair_ranking": None,
        "source": "evaluation/task6g_g0_overfit.json",
    }

    report["final"] = {
        "epoch": int(best["epoch"]),
        "train_pair_ranking_rate": best["train_pair_ranking_rate"],
        "train_mean_margin": best["train_mean_margin"],
        "validation_pairs": best["validation_pairs"],
        "val": best["val"],
        "gate": best["gate"],
        "checkpoint": best["checkpoint"],
        "checkpoint_load": load_report,
    }
    report["verdict"] = "H1_GATE_PASS" if best["gate"]["passed"] else "COUNTERFACTUAL_DENSE_GROUNDING_FAILED"
    report["seconds"] = round(time.time() - started, 2)

    output = Path(args.output) if args.output else OUT
    write_json(output, report)
    write_json(
        SPATIAL_OUT,
        {
            "_doc": (
                "Task 6H sections 12/17. Pair-ranking + point-localization metrics for the selected "
                "H1 model: heatmaps and points come from the query-to-map path only; GT is scoring "
                "only. Pair metrics use the fixed 20 paired validation images (one canonical pair "
                "each); record metrics use the fixed independent 120-record validation set."
            ),
            "task": "6H",
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
                "Task 6H section 17. Query/heatmap representation diagnostics at the best H1 "
                "checkpoint, compared with the frozen Task 6F/6G values: did explicit same-image "
                "counterfactual supervision make the query target-specific instead of merely "
                "image-specific?"
            ),
            "task": "6H",
            "grid": grid,
            "selected_epoch": int(best["epoch"]),
            "checkpoint": best["checkpoint"],
            "representation": representation,
        },
    )
    print(
        f"[task6h.h1] {report['verdict']}: selected epoch {best['epoch']} "
        f"paired-point {best['validation_pairs']['paired_point_selection_pass']}/"
        f"{best['validation_pairs']['pair_count']} pair-ranking "
        f"{best['validation_pairs']['pair_ranking_pass']}/{best['validation_pairs']['pair_count']} "
        f"120-record inside {best['val']['point_inside_target_rate']} margin "
        f"{best['validation_pairs']['mean_own_minus_cross_margin']:+.4f}",
        flush=True,
    )
    print(f"[task6h.h1] wrote {output} + spatial/representation artifacts", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if best["gate"]["passed"] else 7


def _strip_pair(row: dict) -> dict:
    row = dict(row)
    row.pop("record_a", None)
    row.pop("record_b", None)
    return row


def _margin_vs_localization(pair_rows: list[dict]) -> dict:
    """Section 17: does a larger own-minus-cross margin mean better localization?"""

    import numpy as np

    margins, localized = [], []
    for row in pair_rows:
        margins.append(float(row["mean_margin"]))
        localized.append(
            1.0
            if (row["record_a"]["point_inside_own"] and row["record_b"]["point_inside_own"])
            else 0.0
        )
    if len(margins) < 2 or len(set(localized)) < 2:
        correlation = None
    else:
        correlation = float(np.corrcoef(margins, localized)[0, 1])
    passed = [row for row in pair_rows if row["point_selection_passed"]]
    return {
        "pairs": len(pair_rows),
        "correlation_margin_vs_paired_localization": correlation,
        "mean_margin_when_both_inside": (
            float(np.mean([row["mean_margin"] for row in passed])) if passed else None
        ),
        "mean_margin_otherwise": (
            float(np.mean([row["mean_margin"] for row in pair_rows if not row["point_selection_passed"]]))
            if len(passed) < len(pair_rows)
            else None
        ),
        "interpretation_note": (
            "a rank correlation between the counterfactual margin and paired localization success "
            "is the concrete test of section 17's question; cosine alone is not used as evidence"
        ),
    }


def _manifest_hash() -> str | None:
    from task6h_common import PAIR_MANIFEST

    if not PAIR_MANIFEST.exists():
        return None
    import json

    return json.loads(PAIR_MANIFEST.read_text(encoding="utf-8")).get("pair_identity_sha256")


if __name__ == "__main__":
    raise SystemExit(main())
