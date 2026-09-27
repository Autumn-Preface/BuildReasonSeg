"""Task 6G section 20: classify spatial-localization errors; do not change the dataset.

Reads the selected G1 spatial artifact and labels every failure with the WHU target-quality flags
plus the dense-path classes (`heatmap on wrong building`, `diffuse/no-localization heatmap`). If
simple L1 examples work but tiny/ambiguous cases dominate, that is recorded as evidence for a
later dataset task.

Writes `evaluation/task6g_error_analysis.json`.
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

from task6g_common import EVAL, write_json  # noqa: E402

SPATIAL_JSON = EVAL / "task6g_spatial_eval.json"
G0_JSON = EVAL / "task6g_g0_overfit.json"
OUT = EVAL / "task6g_error_analysis.json"

FAILURE_CLASSES = (
    "tiny_target",
    "border_truncation",
    "touching_merged_pseudo_instance",
    "visually_ambiguous_building",
    "relation_complexity",
    "heatmap_on_wrong_building",
    "diffuse_no_localization_heatmap",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside-fail-threshold", type=float, default=0.5)
    parser.add_argument(
        "--source",
        default=None,
        help="G1 spatial artifact by default; --source g0 uses the G0 overfit artifact (task stopped at G0).",
    )
    args = parser.parse_args(argv)

    if args.source == "g0" or not SPATIAL_JSON.exists():
        spatial = json.loads(G0_JSON.read_text(encoding="utf-8"))
        records = spatial["final"]["records"]
        stage = "G0"
        selected_epoch = None
        grid = spatial.get("grid")
    else:
        spatial = json.loads(SPATIAL_JSON.read_text(encoding="utf-8"))
        records = spatial["val_records"]
        stage = "G1"
        selected_epoch = spatial.get("selected_epoch")
        grid = spatial.get("grid")
    val_records = {record["sample_id"]: record for record in data_mod.read_records("val")}
    train_records = {record["sample_id"]: record for record in data_mod.read_records("train")}

    analysed = []
    for record in records:
        source = val_records.get(record["sample_id"]) or train_records.get(record["sample_id"])
        if source is None:
            continue
        sample = data_mod.to_sample(source)
        stats = target_component_stats(sample)
        flags = {
            "tiny_target": bool(stats.get("target_area_fraction", 1.0) < 0.01),
            "border_truncation": bool(stats.get("touches_border")),
            "touching_merged_pseudo_instance": bool(stats.get("touching_neighbour_count", 0) > 0),
            "visually_ambiguous_building": bool(
                stats.get("components_in_tile", 0) <= 2
                or stats.get("target_connected_components", 1) > 1
            ),
            "relation_complexity": bool(int(record["level"]) >= 2 and not record["point_inside_own"]),
            "heatmap_on_wrong_building": bool(
                not record["point_inside_own"] and record["peakiness"] >= 0.5
            ),
            "diffuse_no_localization_heatmap": bool(
                not record["point_inside_own"] and record["peakiness"] < 0.5
            ),
        }
        analysed.append(
            {
                "sample_id": record["sample_id"],
                "image_id": record["image_id"],
                "level": int(record["level"]),
                "query_family": record["query_family"],
                "predicted_point": record["predicted_point"],
                "gt_point": record["gt_point"],
                "point_inside_own": record["point_inside_own"],
                "error_512px": record["error_512px"],
                "peakiness": record["peakiness"],
                "heatmap_soft_dice": record["heatmap_soft_dice"],
                "flags": flags,
                "component_stats": stats,
            }
        )

    failures = [row for row in analysed if not row["point_inside_own"]]
    quality_flags = (
        "tiny_target",
        "border_truncation",
        "touching_merged_pseudo_instance",
        "visually_ambiguous_building",
    )
    failures_with_quality_flag = [row for row in failures if any(row["flags"][name] for name in quality_flags)]
    flag_counts = {
        name: sum(1 for row in failures if row["flags"][name]) for name in FAILURE_CLASSES
    }
    share = {name: (flag_counts[name] / len(failures) if failures else None) for name in FAILURE_CLASSES}
    ranked = sorted(
        ((name, value) for name, value in share.items() if value is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    simple = [row for row in analysed if int(row["level"]) == 1]
    complex_records = [row for row in analysed if int(row["level"]) >= 2]

    report = {
        "_doc": (
            "Task 6G section 20. Dense-path localization error classification. The dataset is not "
            "changed; the question is whether simple L1 examples work and whether the residual "
            "concentrates in WHU data-quality categories. When the task stops at G0 (section 11), "
            "the classification is computed on the G0 evaluation records and says so."
        ),
        "task": "6G",
        "stage": stage,
        "selected_epoch": selected_epoch,
        "grid": grid,
        "count": len(analysed),
        "failures_inside": len(failures),
        "inside_rate": float(
            sum(1 for row in analysed if row["point_inside_own"]) / len(analysed)
        ),
        "flag_counts": flag_counts,
        "flag_share_of_failures": share,
        "dominant_remaining_error": None if not ranked else ranked[0][0],
        "ranked_flags": ranked,
        "failures_with_any_whu_quality_flag": len(failures_with_quality_flag),
        "failures_with_any_whu_quality_flag_share": (
            None if not failures else len(failures_with_quality_flag) / len(failures)
        ),
        "simple_level1_inside_rate": (
            None
            if not simple
            else float(sum(1 for row in simple if row["point_inside_own"]) / len(simple))
        ),
        "complex_level2_3_inside_rate": (
            None
            if not complex_records
            else float(sum(1 for row in complex_records if row["point_inside_own"]) / len(complex_records))
        ),
        "whu_data_quality_dominates": bool(
            failures and len(failures_with_quality_flag) / len(failures) > 0.5
        ),
        "records": analysed,
    }
    write_json(OUT, report)
    print(
        f"[task6g:errors] {len(analysed)} records, {len(failures)} outside: dominant "
        f"{report['dominant_remaining_error']} L1 inside {report['simple_level1_inside_rate']} "
        f"L2/3 inside {report['complex_level2_3_inside_rate']} whu-dominated "
        f"{report['whu_data_quality_dominates']}",
        flush=True,
    )
    print(f"[task6g:errors] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
