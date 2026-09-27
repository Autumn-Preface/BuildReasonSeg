"""Task 6H section 22: classify pair-ranking and localization errors; do not migrate data.

Reads the selected H1 spatial artifact (or the H0 artifact when the task stopped at H0) and labels
every failure with the section 22 classes: pair-ranking wrong / ranking correct but point outside /
diffuse heatmap / wrong building / tiny target / border truncation / pseudo-instance ambiguity /
complex relation.

Writes `evaluation/task6h_error_analysis.json`.
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

from task6h_common import EVAL, write_json  # noqa: E402

SPATIAL_JSON = EVAL / "task6h_spatial_eval.json"
H0_JSON = EVAL / "task6h_h0_overfit.json"
OUT = EVAL / "task6h_error_analysis.json"

FAILURE_CLASSES = (
    "pair_ranking_correct_point_outside",
    "pair_ranking_wrong",
    "diffuse_heatmap",
    "wrong_building",
    "tiny_target",
    "border_truncation",
    "pseudo_instance_ambiguity",
    "complex_relation",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="h1 (default) or h0 when the task stopped at H0")
    parser.add_argument("--peakiness-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    if args.source == "h0" or not SPATIAL_JSON.exists():
        payload = json.loads(H0_JSON.read_text(encoding="utf-8"))
        stage = "H0"
        pair_rows = payload["final"]["pairs"]
        records = payload["final"]["records"]
        selected_epoch = None
        grid = payload.get("grid")
    else:
        payload = json.loads(SPATIAL_JSON.read_text(encoding="utf-8"))
        stage = "H1"
        pair_rows = payload["paired_rows"]
        records = payload["val_records"]
        selected_epoch = payload.get("selected_epoch")
        grid = payload.get("grid")

    train_records = {record["sample_id"]: record for record in data_mod.read_records("train")}
    val_records = {record["sample_id"]: record for record in data_mod.read_records("val")}
    prediction_records = {row["sample_id"]: row for row in records}

    def stats_for(sample_id: str):
        source = val_records.get(sample_id) or train_records.get(sample_id)
        if source is None:
            return None
        return target_component_stats(data_mod.to_sample(source))

    def prediction_for(sample_id: str, row: dict, side: str) -> dict:
        """Per-sample prediction fields, whether or not the pair row kept the nested records."""

        nested = row.get(f"record_{side}")
        if isinstance(nested, dict):
            return nested
        return prediction_records.get(sample_id, {})

    analysed_pairs = []
    for row in pair_rows:
        stats_a = stats_for(row["a"])
        stats_b = stats_for(row["b"])
        prediction_a = prediction_for(row["a"], row, "a")
        prediction_b = prediction_for(row["b"], row, "b")
        ranking_correct = bool(row["pair_ranking_pass"])
        both_inside = bool(row.get("point_selection_passed"))
        flags = {
            "pair_ranking_correct_point_outside": bool(ranking_correct and not both_inside),
            "pair_ranking_wrong": bool(not ranking_correct),
            "diffuse_heatmap": bool(
                not both_inside
                and min(
                    prediction_a.get("peakiness", 1.0), prediction_b.get("peakiness", 1.0)
                )
                < args.peakiness_threshold
            ),
            "wrong_building": bool(
                not both_inside
                and min(
                    prediction_a.get("peakiness", 0.0), prediction_b.get("peakiness", 0.0)
                )
                >= args.peakiness_threshold
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
                "s_aa": row["s_aa"],
                "s_ab": row["s_ab"],
                "s_bb": row["s_bb"],
                "s_ba": row["s_ba"],
                "margin_a": row["margin_a"],
                "margin_b": row["margin_b"],
                "mean_margin": row["mean_margin"],
                "pair_ranking_pass": ranking_correct,
                "both_points_inside": both_inside,
                "same_image_point_distance": row["same_image_point_distance"],
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
                "error_512px": record["error_512px"],
                "peakiness": record["peakiness"],
                "heatmap_soft_dice": record["heatmap_soft_dice"],
                "flags": {
                    "tiny_target": bool((stats or {}).get("target_area_fraction", 1.0) < 0.01),
                    "border_truncation": bool((stats or {}).get("touches_border")),
                    "pseudo_instance_ambiguity": bool(
                        (stats or {}).get("touching_neighbour_count", 0) > 0
                    ),
                    "diffuse_heatmap": bool(
                        not record["point_inside_own"]
                        and record["peakiness"] < args.peakiness_threshold
                    ),
                    "wrong_building": bool(
                        not record["point_inside_own"]
                        and record["peakiness"] >= args.peakiness_threshold
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

    report = {
        "_doc": (
            "Task 6H section 22. Counterfactual-pair and point-localization error classification. "
            "The dataset is not migrated; the question is whether simple pairs work and whether "
            "data-quality categories dominate the residual."
        ),
        "task": "6H",
        "stage": stage,
        "selected_epoch": selected_epoch,
        "grid": grid,
        "pairs": {
            "count": len(analysed_pairs),
            "ranking_pass": sum(1 for row in analysed_pairs if row["pair_ranking_pass"]),
            "both_points_inside": sum(1 for row in analysed_pairs if row["both_points_inside"]),
            "ranking_correct_point_outside": sum(
                1 for row in analysed_pairs if row["flags"]["pair_ranking_correct_point_outside"]
            ),
            "ranking_wrong": sum(1 for row in analysed_pairs if row["flags"]["pair_ranking_wrong"]),
            "failure_flag_counts": pair_flag_counts,
            "failure_flag_share": pair_share,
            "dominant_failure": None if not ranked else ranked[0][0],
        },
        "records": {
            "count": len(analysed_records),
            "inside_rate": float(
                sum(1 for row in analysed_records if row["point_inside_own"]) / max(len(analysed_records), 1)
            ),
            "failure_flag_counts": record_flag_counts,
            "failures_with_any_whu_quality_flag": len(failing_with_quality),
            "failures_with_any_whu_quality_flag_share": (
                None if not failing_records else len(failing_with_quality) / len(failing_records)
            ),
            "simple_level1_inside_rate": (
                None
                if not [row for row in analysed_records if row["level"] == 1]
                else float(
                    sum(1 for row in analysed_records if row["level"] == 1 and row["point_inside_own"])
                    / len([row for row in analysed_records if row["level"] == 1])
                )
            ),
            "complex_level2_3_inside_rate": (
                None
                if not [row for row in analysed_records if row["level"] >= 2]
                else float(
                    sum(1 for row in analysed_records if row["level"] >= 2 and row["point_inside_own"])
                    / len([row for row in analysed_records if row["level"] >= 2])
                )
            ),
        },
        "whu_data_quality_dominates": bool(
            failing_records and len(failing_with_quality) / len(failing_records) > 0.5
        ),
        "pair_rows": analysed_pairs,
        "record_rows": analysed_records,
    }
    write_json(OUT, report)
    print(
        f"[task6h:errors] stage {stage}: pairs ranking "
        f"{report['pairs']['ranking_pass']}/{report['pairs']['count']} both-inside "
        f"{report['pairs']['both_points_inside']}/{report['pairs']['count']} dominant "
        f"{report['pairs']['dominant_failure']}; records inside "
        f"{report['records']['inside_rate']:.3f} whu-dominated "
        f"{report['whu_data_quality_dominates']}",
        flush=True,
    )
    print(f"[task6h:errors] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
