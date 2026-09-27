"""Task 6F section 18: classify box failures; do not change the dataset.

Reads the selected F1 geometry artifact and labels every validation failure with the WHU
target-quality flags Task 6D used, plus query-path localization classes. The question is whether
simple cases work and whether the residual concentrates in data-quality categories.

Writes `evaluation/task6f_error_analysis.json`.
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

from task6f_common import EVAL, write_json  # noqa: E402

GEOMETRY_JSON = EVAL / "task6f_geometry_eval.json"
OUT = EVAL / "task6f_error_analysis.json"

FAILURE_CLASSES = (
    "tiny_target",
    "border_truncation",
    "touching_merged_pseudo_instance",
    "visually_ambiguous_building",
    "complex_l3_relation",
    "wrong_target_valid_box",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    geometry = json.loads(GEOMETRY_JSON.read_text(encoding="utf-8"))
    records = geometry["val_records"]
    val_records = {record["sample_id"]: record for record in data_mod.read_records("val")}

    analysed = []
    for record in records:
        source = val_records.get(record["sample_id"])
        if source is None:
            continue
        sample = data_mod.to_sample(source)
        stats = target_component_stats(sample)
        box_iou = record["box_iou"]
        flags = {
            "tiny_target": bool(stats.get("target_area_fraction", 1.0) < 0.01),
            "border_truncation": bool(stats.get("touches_border")),
            "touching_merged_pseudo_instance": bool(stats.get("touching_neighbour_count", 0) > 0),
            "visually_ambiguous_building": bool(
                stats.get("components_in_tile", 0) <= 2
                or stats.get("target_connected_components", 1) > 1
            ),
            "complex_l3_relation": bool(int(record["level"]) == 3 and box_iou < args.iou_threshold),
            "wrong_target_valid_box": bool(
                box_iou < args.iou_threshold and not record["center_inside"]
            ),
        }
        analysed.append(
            {
                "sample_id": record["sample_id"],
                "image_id": record["image_id"],
                "level": int(record["level"]),
                "query_family": record["query_family"],
                "predicted_box": record["predicted_box"],
                "gt_box": record["gt_box"],
                "box_iou": box_iou,
                "center_inside": record["center_inside"],
                "coordinate_mae": record["coordinate_mae"],
                "flags": flags,
                "component_stats": stats,
            }
        )

    failures = [row for row in analysed if row["box_iou"] < args.iou_threshold]
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
    simple_cases = [row for row in analysed if int(row["level"]) == 1]
    complex_cases = [row for row in analysed if int(row["level"]) >= 2]

    report = {
        "_doc": (
            "Task 6F section 18. Query-path box error classification on the fixed 120-record "
            "validation set for the selected F1 model. The dataset is not changed; the question is "
            "whether simple cases work and whether the residual concentrates in WHU data-quality "
            "categories."
        ),
        "task": "6F",
        "selected_epoch": geometry.get("selected_epoch"),
        "count": len(analysed),
        "iou_threshold": float(args.iou_threshold),
        "mean_box_iou": (
            None if not analysed else float(sum(row["box_iou"] for row in analysed) / len(analysed))
        ),
        "failures": len(failures),
        "flag_counts": flag_counts,
        "flag_share_of_failures": share,
        "dominant_remaining_error": None if not ranked else ranked[0][0],
        "ranked_flags": ranked,
        "failures_with_any_whu_quality_flag": len(failures_with_quality_flag),
        "failures_with_any_whu_quality_flag_share": (
            None if not failures else len(failures_with_quality_flag) / len(failures)
        ),
        "simple_level1_mean_box_iou": (
            None if not simple_cases else float(sum(row["box_iou"] for row in simple_cases) / len(simple_cases))
        ),
        "complex_level2_3_mean_box_iou": (
            None if not complex_cases else float(sum(row["box_iou"] for row in complex_cases) / len(complex_cases))
        ),
        "whu_data_quality_dominates": bool(
            failures and len(failures_with_quality_flag) / len(failures) > 0.5
        ),
        "records": analysed,
    }
    write_json(OUT, report)
    print(
        f"[task6f:errors] {len(analysed)} records, {len(failures)} below IoU {args.iou_threshold}: "
        f"dominant {report['dominant_remaining_error']} L1 mean "
        f"{report['simple_level1_mean_box_iou']} L2/3 mean "
        f"{report['complex_level2_3_mean_box_iou']} whu-dominated "
        f"{report['whu_data_quality_dominates']}",
        flush=True,
    )
    print(f"[task6f:errors] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
