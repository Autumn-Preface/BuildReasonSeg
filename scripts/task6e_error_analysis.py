"""Task 6E section 18: classify coordinate-token errors; do not change the dataset.

Reads the selected E1 geometry artifact and labels every validation failure with the WHU
target-quality flags Task 6D already used, plus a quantization-sensitivity ceiling:

* `structural_failure` -- the generated sequence was not a valid spatial target;
* `tiny_target_quantization_sensitivity` -- target area < 1 % and the quantized GT box
  itself already scores below 0.5 IoU against the continuous GT box;
* `border_truncation` -- the target touches the tile border;
* `touching_merged_pseudo_instance` -- the target touches another component;
* `visually_ambiguous_building` -- few components in the tile or a disconnected target;
* `complex_l3_relation` -- a level-3 relation whose box is still wrong;
* `valid_token_sequence_wrong_target` -- a valid sequence that names the wrong building.

Writes `evaluation/task6e_error_analysis.json`.
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
from buildreasonseg_mvp.grounding import target_geometry  # noqa: E402
from buildreasonseg_mvp.grounding_eval import target_component_stats  # noqa: E402
from buildreasonseg_mvp.spatial_tokens import QuantizedBoxCodec, box_iou_values  # noqa: E402

from task6e_common import EVAL, load_spatial_config, write_json  # noqa: E402

GEOMETRY_JSON = EVAL / "task6e_geometry_eval.json"
SEGMENTATION_JSON = EVAL / "task6e_segmentation_eval.json"
OUT = EVAL / "task6e_error_analysis.json"

FAILURE_CLASSES = (
    "structural_failure",
    "tiny_target_quantization_sensitivity",
    "border_truncation",
    "touching_merged_pseudo_instance",
    "visually_ambiguous_building",
    "complex_l3_relation",
    "valid_token_sequence_wrong_target",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    cfg, _payload = load_spatial_config()
    bins = int(cfg["spatial_tokens"]["bins"])
    codec = QuantizedBoxCodec(bins)
    geometry = json.loads(GEOMETRY_JSON.read_text(encoding="utf-8"))
    records = geometry["free_generation_records"]
    segmentation = (
        json.loads(SEGMENTATION_JSON.read_text(encoding="utf-8"))
        if SEGMENTATION_JSON.exists()
        else None
    )
    miou_by_id = {}
    if segmentation is not None:
        miou_by_id = {
            record["sample_id"]: record["strict_miou"]
            for record in segmentation["segmentation"]["records"]
        }

    val_records = {record["sample_id"]: record for record in data_mod.read_records("val")}
    analysed = []
    for record in records:
        source = val_records.get(record["sample_id"])
        if source is None:
            continue
        sample = data_mod.to_sample(source)
        stats = target_component_stats(sample)
        gt_box = [float(value) for value in record["gt_box"]]
        quantized_box = [float(value) for value in codec.decode(codec.encode(gt_box))]
        ceiling = float(box_iou_values(quantized_box, gt_box))
        box_iou = record["box_iou"]
        iou = 0.0 if box_iou is None else float(box_iou)
        flags = {
            "structural_failure": not bool(record["structural_valid"]),
            "tiny_target_quantization_sensitivity": bool(
                stats.get("target_area_fraction", 1.0) < 0.01 and ceiling < args.iou_threshold
            ),
            "border_truncation": bool(stats.get("touches_border")),
            "touching_merged_pseudo_instance": bool(stats.get("touching_neighbour_count", 0) > 0),
            "visually_ambiguous_building": bool(
                stats.get("components_in_tile", 0) <= 2
                or stats.get("target_connected_components", 1) > 1
            ),
            "complex_l3_relation": bool(int(record["level"]) == 3 and iou < args.iou_threshold),
            "valid_token_sequence_wrong_target": bool(
                record["structural_valid"] and iou < args.iou_threshold
            ),
        }
        analysed.append(
            {
                "sample_id": record["sample_id"],
                "image_id": record["image_id"],
                "level": int(record["level"]),
                "query_family": record["query_family"],
                "predicted_codes": record["predicted_codes"],
                "gt_codes": record["gt_codes"],
                "predicted_box": record["predicted_box"],
                "gt_box": gt_box,
                "box_iou": box_iou,
                "quantization_ceiling_box_iou": ceiling,
                "center_inside": record["center_inside"],
                "strict_mask_miou": miou_by_id.get(record["sample_id"]),
                "structural_valid": bool(record["structural_valid"]),
                "failure": record["failure"],
                "box_token_count": int(record["box_token_count"]),
                "loc_token_count": int(record["loc_token_count"]),
                "seg_count": int(record["seg_count"]),
                "generated_token_count": int(record["generated_token_count"]),
                "flags": flags,
                "component_stats": stats,
            }
        )

    failures = [row for row in analysed if row["box_iou"] is None or row["box_iou"] < args.iou_threshold]
    structural_failures = [row for row in analysed if not row["structural_valid"]]
    # WHU target quality can only explain a *localization* error: a sample whose token sequence was
    # valid but whose box named the wrong building. A structural failure (malformed output) cannot be
    # attributed to the target's size, border or ambiguity, and is counted separately.
    localization_failures = [
        row
        for row in failures
        if row["structural_valid"] and (row["box_iou"] or 0.0) < args.iou_threshold
    ]
    quantization_limited = [
        row for row in analysed if row["quantization_ceiling_box_iou"] < args.iou_threshold
    ]
    flag_counts = {
        name: sum(1 for row in failures if row["flags"][name]) for name in FAILURE_CLASSES
    }
    localization_flag_counts = {
        name: sum(1 for row in localization_failures if row["flags"][name])
        for name in FAILURE_CLASSES
    }
    quality_flags = (
        "tiny_target_quantization_sensitivity",
        "border_truncation",
        "touching_merged_pseudo_instance",
        "visually_ambiguous_building",
    )
    localization_with_quality_flag = [
        row for row in localization_failures if any(row["flags"][name] for name in quality_flags)
    ]
    structural_modes = {
        "no_seg_token_emitted": sum(1 for row in structural_failures if row["seg_count"] == 0),
        "no_box_token_emitted": sum(1 for row in structural_failures if row["box_token_count"] == 0),
        "multiple_box_tokens": sum(1 for row in structural_failures if row["box_token_count"] > 1),
        "runaway_location_run": sum(
            1
            for row in structural_failures
            if row["loc_token_count"] > 4 and row["seg_count"] == 0
        ),
        "empty_generation": sum(1 for row in structural_failures if row["generated_token_count"] == 0),
    }
    share = {name: (flag_counts[name] / len(failures) if failures else None) for name in FAILURE_CLASSES}
    ranked = sorted(
        ((name, value) for name, value in share.items() if value is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    localization_share = {
        name: (
            localization_flag_counts[name] / len(localization_failures)
            if localization_failures
            else None
        )
        for name in FAILURE_CLASSES
    }
    localization_ranked = sorted(
        ((name, value) for name, value in localization_share.items() if value is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    report = {
        "_doc": (
            "Task 6E section 18. Coordinate-token error classification on the fixed 120-record "
            "validation set for the selected E1 model. The dataset is not changed; the question is "
            "whether WHU target quality dominates the residual error. Structural failures (malformed "
            "token sequences) are separated from localization failures, because target size, border "
            "or ambiguity cannot explain a sequence that never emitted a valid box."
        ),
        "task": "6E",
        "bins": bins,
        "selected_epoch": geometry.get("selected_epoch"),
        "count": len(analysed),
        "mean_box_iou": (
            None if not analysed else float(sum(row["box_iou"] or 0.0 for row in analysed) / len(analysed))
        ),
        "iou_threshold": float(args.iou_threshold),
        "structural_failures": len(structural_failures),
        "structural_failure_modes": structural_modes,
        "localization_failures": len(localization_failures),
        "failures": len(failures),
        "flag_counts": flag_counts,
        "flag_share_of_failures": share,
        "dominant_remaining_error": None if not ranked else ranked[0][0],
        "ranked_flags": ranked,
        "localization_flag_counts": localization_flag_counts,
        "localization_flag_share": localization_share,
        "dominant_localization_error": None if not localization_ranked else localization_ranked[0][0],
        "localization_failures_with_any_whu_quality_flag": len(localization_with_quality_flag),
        "localization_failures_with_any_whu_quality_flag_share": (
            None
            if not localization_failures
            else len(localization_with_quality_flag) / len(localization_failures)
        ),
        "quantization_ceiling": {
            "samples_whose_quantized_gt_box_is_below_threshold": len(quantization_limited),
            "share": None if not analysed else len(quantization_limited) / len(analysed),
            "mean_ceiling_box_iou": (
                None
                if not analysed
                else float(
                    sum(row["quantization_ceiling_box_iou"] for row in analysed) / len(analysed)
                )
            ),
            "note": (
                "The ceiling is IoU(quantized-and-dequantized GT box, continuous GT box) at the "
                "selected B. A sample whose ceiling is below the threshold cannot be solved by any "
                "token sequence at this bin count."
            ),
        },
        "whu_data_quality_dominates": (
            None
            if not localization_failures
            else len(localization_with_quality_flag) / len(localization_failures) > 0.5
        ),
        "whu_data_quality_note": (
            "computed over localization failures only; with zero localization failures there is no "
            "evidence that target quality (size, border, ambiguity) is the binding constraint here"
        ),
        "records": analysed,
    }
    write_json(OUT, report)
    print(
        f"[task6e:errors] {len(analysed)} records: structural {len(structural_failures)}, "
        f"localization {len(localization_failures)}, quantization-limited "
        f"{len(quantization_limited)}; WHU-quality-dominated "
        f"{report['whu_data_quality_dominates']}",
        flush=True,
    )
    print(f"[task6e:errors] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
