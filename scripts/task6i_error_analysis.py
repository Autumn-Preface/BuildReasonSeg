"""Task 6I: classify refined-path localization errors (no dataset migration).

Reads the selected I1 spatial artifact (or the I0 artifact when the task stopped there) and labels
failures with the pair/localization classes used by Task 6H/6H.1: pair preference correct but point
outside, pair preference wrong, diffuse heatmap, wrong building, tiny target, border truncation,
pseudo-instance ambiguity, complex relation -- plus attention-misses-target as a refinement-specific
signal.

Writes `evaluation/task6i_error_analysis.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.grounding_eval import target_component_stats  # noqa: E402

from task6i_common import EVAL, write_json  # noqa: E402

SPATIAL_JSON = EVAL / "task6i_spatial_eval.json"
I0_JSON = EVAL / "task6i_i0_overfit.json"
OUT = EVAL / "task6i_error_analysis.json"

FAILURE_CLASSES = (
    "pair_preference_correct_point_outside",
    "pair_preference_wrong",
    "attention_misses_target",
    "diffuse_heatmap",
    "wrong_building",
    "tiny_target",
    "border_truncation",
    "pseudo_instance_ambiguity",
    "complex_relation",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="i1 (default) or i0 when the task stopped at I0")
    parser.add_argument("--peakiness-threshold", type=float, default=0.01)
    args = parser.parse_args(argv)

    if args.source == "i0" or not SPATIAL_JSON.exists():
        payload = json.loads(I0_JSON.read_text(encoding="utf-8"))
        stage = "I0"
        pair_rows = payload["final"]["pair_rows"]
        records = payload["final"]["record_rows"]
        selected_epoch = None
        grid = payload.get("grid")
    else:
        payload = json.loads(SPATIAL_JSON.read_text(encoding="utf-8"))
        stage = "I1"
        pair_rows = payload["paired_rows"]
        records = payload["val_records"]
        selected_epoch = payload.get("selected_epoch")
        grid = payload.get("grid")

    train_records = {record["sample_id"]: record for record in data_mod.read_records("train")}
    val_records = {record["sample_id"]: record for record in data_mod.read_records("val")}
    predictions = {row["sample_id"]: row for row in records}

    def stats_for(sample_id: str):
        source = val_records.get(sample_id) or train_records.get(sample_id)
        if source is None:
            return None
        return target_component_stats(data_mod.to_sample(source))

    def prediction_for(sample_id: str, row: dict, side: str) -> dict:
        nested = row.get(f"record_{side}")
        if isinstance(nested, dict):
            return nested
        return predictions.get(sample_id, {})

    analysed_pairs = []
    for row in pair_rows:
        stats_a = stats_for(row["a"])
        stats_b = stats_for(row["b"])
        prediction_a = prediction_for(row["a"], row, "a")
        prediction_b = prediction_for(row["b"], row, "b")
        preference_correct = bool(row["pair_preference_pass"])
        both_inside = bool(row.get("point_selection_passed"))
        entropy = float(row.get("spatial_entropy") or 0.0)
        attention_missed = bool(
            float(row.get("attention_own_mass_a") or 1.0) < float(row.get("attention_cross_mass_a") or 0.0)
            or float(row.get("attention_own_mass_b") or 1.0) < float(row.get("attention_cross_mass_b") or 0.0)
        )
        flags = {
            "pair_preference_correct_point_outside": bool(preference_correct and not both_inside),
            "pair_preference_wrong": bool(not preference_correct),
            "attention_misses_target": bool(attention_missed),
            "diffuse_heatmap": bool(
                not both_inside
                and min(
                    prediction_a.get("max_non_target_probability", 1.0),
                    prediction_b.get("max_non_target_probability", 1.0),
                )
                > args.peakiness_threshold
            ),
            "wrong_building": bool(
                not both_inside
                and min(
                    prediction_a.get("max_non_target_probability", 0.0),
                    prediction_b.get("max_non_target_probability", 0.0),
                )
                <= args.peakiness_threshold
            ),
            "tiny_target": bool(
                (stats_a or {}).get("target_area_fraction", 1.0) < 0.01
                or (stats_b or {}).get("target_area_fraction", 1.0) < 0.01
            ),
            "border_truncation": bool(
                (stats_a or {}).get("touches_border") or (stats_b or {}).get("touches_border")
            ),
            "pseudo_instance_ambiguity": bool(
                (stats_a or {}).get("touching_neighbour_count", 0) > 0
                or (stats_b or {}).get("touching_neighbour_count", 0) > 0
            ),
            "complex_relation": bool(int(row.get("a_level", 1)) >= 2 or int(row.get("b_level", 1)) >= 2),
        }
        analysed_pairs.append(
            {
                "image_id": row["image_id"],
                "a": row["a"],
                "b": row["b"],
                "p_aa": row["p_aa"],
                "p_ab": row["p_ab"],
                "p_bb": row["p_bb"],
                "p_ba": row["p_ba"],
                "mean_margin": row["mean_margin"],
                "cf_loss": row.get("cf_loss"),
                "pair_preference_pass": preference_correct,
                "both_points_inside": both_inside,
                "same_image_point_distance": row.get("same_image_point_distance"),
                "same_image_probability_map_overlap": row.get("same_image_probability_map_overlap"),
                "spatial_entropy": entropy,
                "attention_own_mass_mean": row.get("attention_own_mass_mean"),
                "attention_cross_mass_mean": row.get("attention_cross_mass_mean"),
                "flags": flags,
            }
        )

    analysed_records = []
    for record in records:
        stats = stats_for(record["sample_id"])
        analysed_records.append(
            {
                "sample_id": record["sample_id"],
                "image_id": record["image_id"],
                "level": int(record["level"]),
                "query_family": record["query_family"],
                "point_inside_own": record["point_inside_own"],
                "target_cell_top1": record.get("target_cell_top1"),
                "target_cell_top5": record.get("target_cell_top5"),
                "target_cell_probability": record.get("target_cell_probability"),
                "spatial_entropy": record.get("spatial_entropy"),
                "mean_abs_logit": record.get("mean_abs_logit"),
                "error_512px": record.get("error_512px"),
                "attn_entropy": record.get("attn_entropy"),
                "attn_target_mass": record.get("attn_target_mass"),
                "flags": {
                    "tiny_target": bool((stats or {}).get("target_area_fraction", 1.0) < 0.01),
                    "border_truncation": bool((stats or {}).get("touches_border")),
                    "pseudo_instance_ambiguity": bool(
                        (stats or {}).get("touching_neighbour_count", 0) > 0
                    ),
                    "diffuse_heatmap": bool(
                        not record["point_inside_own"]
                        and float(record.get("max_non_target_probability") or 1.0)
                        > args.peakiness_threshold
                    ),
                    "wrong_building": bool(
                        not record["point_inside_own"]
                        and float(record.get("max_non_target_probability") or 0.0)
                        <= args.peakiness_threshold
                    ),
                    "attention_misses_target": bool(
                        float(record.get("attn_target_mass") or 0.0) < 0.5
                    ),
                    "complex_relation": bool(int(record["level"]) >= 2),
                },
                "component_stats": stats,
            }
        )

    failing_pairs = [row for row in analysed_pairs if not row["both_points_inside"]]
    pair_flag_counts = {
        name: sum(1 for row in failing_pairs if row["flags"][name]) for name in FAILURE_CLASSES
    }
    pair_share = {
        name: (pair_flag_counts[name] / len(failing_pairs) if failing_pairs else None)
        for name in FAILURE_CLASSES
    }
    ranked = sorted(
        ((name, value) for name, value in pair_share.items() if value is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    record_flags = (
        "tiny_target",
        "border_truncation",
        "pseudo_instance_ambiguity",
        "diffuse_heatmap",
        "wrong_building",
        "attention_misses_target",
        "complex_relation",
    )
    failing_records = [row for row in analysed_records if not row["point_inside_own"]]
    record_flag_counts = {
        name: sum(1 for row in failing_records if row["flags"][name]) for name in record_flags
    }
    quality_flags = ("tiny_target", "border_truncation", "pseudo_instance_ambiguity")
    failing_with_quality = [
        row for row in failing_records if any(row["flags"][name] for name in quality_flags)
    ]
    levels = sorted({row["level"] for row in analysed_records})

    report = {
        "_doc": (
            "Task 6I. Refined-path localization error classification. The dataset is not migrated; "
            "the question is whether simple pairs work, whether the attention refinement attends "
            "to the target, and whether data-quality categories dominate the residual."
        ),
        "task": "6I",
        "stage": stage,
        "selected_epoch": selected_epoch,
        "grid": grid,
        "pairs": {
            "count": len(analysed_pairs),
            "preference_pass": sum(1 for row in analysed_pairs if row["pair_preference_pass"]),
            "both_points_inside": sum(1 for row in analysed_pairs if row["both_points_inside"]),
            "attention_misses_target": sum(
                1 for row in analysed_pairs if row["flags"]["attention_misses_target"]
            ),
            "preference_correct_point_outside": sum(
                1 for row in analysed_pairs if row["flags"]["pair_preference_correct_point_outside"]
            ),
            "preference_wrong": sum(1 for row in analysed_pairs if row["flags"]["pair_preference_wrong"]),
            "failure_flag_counts": pair_flag_counts,
            "failure_flag_share": pair_share,
            "dominant_failure": None if not ranked else ranked[0][0],
        },
        "records": {
            "count": len(analysed_records),
            "inside_rate": float(
                sum(1 for row in analysed_records if row["point_inside_own"])
                / max(len(analysed_records), 1)
            ),
            "top1_rate": float(
                sum(1 for row in analysed_records if row.get("target_cell_top1"))
                / max(len(analysed_records), 1)
            ),
            "top5_rate": float(
                sum(1 for row in analysed_records if row.get("target_cell_top5"))
                / max(len(analysed_records), 1)
            ),
            "attention_misses_target_rate": float(
                sum(1 for row in analysed_records if row["flags"]["attention_misses_target"])
                / max(len(analysed_records), 1)
            ),
            "failure_flag_counts": record_flag_counts,
            "failures_with_any_whu_quality_flag": len(failing_with_quality),
            "failures_with_any_whu_quality_flag_share": (
                None if not failing_records else len(failing_with_quality) / len(failing_records)
            ),
            "by_level": {
                str(level): {
                    "count": sum(1 for row in analysed_records if row["level"] == level),
                    "inside_rate": float(
                        sum(1 for row in analysed_records if row["level"] == level and row["point_inside_own"])
                        / max(sum(1 for row in analysed_records if row["level"] == level), 1)
                    ),
                }
                for level in levels
            },
        },
        "whu_data_quality_dominates": bool(
            failing_records and len(failing_with_quality) / len(failing_records) > 0.5
        ),
        "pair_rows": analysed_pairs,
        "record_rows": analysed_records,
    }
    write_json(OUT, report)
    print(
        f"[task6i:errors] stage {stage}: pairs preference "
        f"{report['pairs']['preference_pass']}/{report['pairs']['count']} both-inside "
        f"{report['pairs']['both_points_inside']}/{report['pairs']['count']} dominant "
        f"{report['pairs']['dominant_failure']}; records inside "
        f"{report['records']['inside_rate']:.3f} top1 {report['records']['top1_rate']:.3f} "
        f"attn-miss {report['records']['attention_misses_target_rate']:.3f} "
        f"whu-dominated {report['whu_data_quality_dominates']}",
        flush=True,
    )
    print(f"[task6i:errors] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
