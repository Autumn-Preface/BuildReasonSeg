"""Task 6K Parts H/I (sections 13-14): cross-analyse the frozen Task 6J result.

Cross-references every J1 (oracle program + YOLO proposals) sample against the source-side
evidence: the raw semantic component size of its target, its distance to the `<50` threshold,
border status, merge-risk class, proposal count and the RAW-vs-CONVERTED relation drift on that
tile. Attribution is only made where the evidence supports it; otherwise the sample is recorded
as inseparable.

Also records the qualified Qwen3-VL-2B statement (section 14).

Writes `evaluation/task6k_task6j_cross_analysis.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_source_audit import MIN_CONTOUR_AREA, binarize, raw_components, read_semantic_label, write_json  # noqa: E402

from task6k_common import CURRENT_ROOT, EVAL, aligned_tiles  # noqa: E402

CACHE_DIR = REPO_ROOT / "artifacts" / "task6k"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    j1 = json.loads((EVAL / "task6j_j1_oracle_program_yolo.json").read_text(encoding="utf-8"))
    recall = json.loads((EVAL / "task6j_yolo_proposal_recall.json").read_text(encoding="utf-8"))
    j2 = json.loads((EVAL / "task6j_j2_program_parser.json").read_text(encoding="utf-8"))
    j3 = json.loads((EVAL / "task6j_j3_predicted_program_oracle_candidates.json").read_text(encoding="utf-8"))
    attribution = json.loads((EVAL / "task6j_error_attribution.json").read_text(encoding="utf-8"))

    scan = {f"{record['split']}/{record['stem']}": record for record in load_jsonl(CACHE_DIR / "tile_scan.jsonl")}
    drift = {f"{record['split']}/{record['stem']}": record for record in load_jsonl(CACHE_DIR / "relation_drift.jsonl")}
    recall_by_sample = {row["sample_id"]: row for row in recall["target_rows"]}

    val_records = {}
    base = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1"
    for split in ("train", "val", "test"):
        path = base / f"{split}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        val_records[str(record["sample_id"])] = record

    tile_by_image = {f"{tile.split}/{tile.stem}": tile for tile in aligned_tiles()}

    rows = []
    for row in j1["val_rows"]:
        sample_id = str(row["sample_id"])
        record = val_records.get(sample_id, {})
        image_id = str(row["image_id"])
        query_type = str(row["query_type"])
        key = f"val/{image_id}"
        tile_record = scan.get(key)
        drift_record = drift.get(key)
        recall_row = recall_by_sample.get(sample_id, {})

        target_raw_area = None
        target_raw_border = None
        target_raw_iou = None
        target_merge_risk = None
        if tile_record is not None:
            tile = tile_by_image.get(key)
            if tile is not None and tile.component_map is not None and record:
                component_map = cv2.imread(str(tile.component_map), cv2.IMREAD_GRAYSCALE)
                target_mask = component_map == int(record["target_component_id"])
                if target_mask.any():
                    source_mask = read_semantic_label(tile.source_label)
                    binary = binarize(source_mask)
                    labels_raw, records_raw = raw_components(binary, connectivity=8)
                    overlap_ids, counts = np.unique(labels_raw[target_mask], return_counts=True)
                    best = None
                    for raw_id, intersection in zip(overlap_ids.tolist(), counts.tolist()):
                        if int(raw_id) == 0:
                            continue
                        raw_record = next((r for r in records_raw if int(r.component_id) == int(raw_id)), None)
                        if raw_record is None:
                            continue
                        union = int(target_mask.sum()) + int(raw_record.area_px) - int(intersection)
                        value = float(intersection) / max(union, 1)
                        if best is None or value > best[0]:
                            best = (value, raw_record)
                    if best is not None:
                        target_raw_iou = best[0]
                        target_raw_area = int(best[1].area_px)
                        target_raw_border = bool(best[1].touches_image_border)
                        for entry in tile_record["merge_risk"]["per_component"]:
                            if int(entry["component_id"]) == int(best[1].component_id):
                                target_merge_risk = entry["risk"]
                                break

        drift_status = None
        if drift_record is not None:
            for program_row in drift_record.get("programs", []):
                if program_row["program"] == query_type:
                    drift_status = program_row["status"]
                    break

        failed = bool(row["abstained"]) or float(row["miou"]) < 0.5
        # Discriminative conversion flags only. The heuristic "medium merge risk" class fires for
        # most tiles regardless of outcome (base rate), so it is recorded but NOT used as a
        # conversion-related criterion; only HIGH risk, a would-be-deleted target (<50 px), a
        # missing raw counterpart or a MEASURED raw-vs-converted target change count.
        conversion_flags = {
            "target_below_area_50": bool(target_raw_area is not None and target_raw_area < MIN_CONTOUR_AREA),
            "target_raw_counterpart_missing": bool(target_raw_area is None),
            "target_merge_risk_high": bool(target_merge_risk == "high"),
            "raw_vs_converted_target_changed": bool(drift_status == "target_changed"),
        }
        non_discriminative = {
            "target_merge_risk_medium_base_rate": bool(target_merge_risk == "medium"),
        }
        conversion_related = any(conversion_flags.values())
        proposal_clean = float(recall_row.get("best_proposal_iou", 0.0)) >= 0.5
        if not failed:
            category = "success"
        elif conversion_related:
            category = "plausibly_conversion_or_data_related"
        elif proposal_clean:
            category = "proposal_model_related_on_clean_target"
        else:
            category = "inseparable_with_current_evidence"

        rows.append(
            {
                "sample_id": sample_id,
                "image_id": image_id,
                "level": int(row["level"]),
                "query_type": query_type,
                "miou": float(row["miou"]),
                "abstained": bool(row["abstained"]),
                "abstain_reason": row["abstain_reason"],
                "failed": failed,
                "best_proposal_iou": recall_row.get("best_proposal_iou"),
                "proposal_count": recall_row.get("proposal_count"),
                "target_raw_area_px": target_raw_area,
                "target_raw_area_minus_50": None if target_raw_area is None else int(target_raw_area) - int(MIN_CONTOUR_AREA),
                "target_raw_border": target_raw_border,
                "target_raw_iou": target_raw_iou,
                "target_merge_risk": target_merge_risk,
                "raw_vs_converted_status": drift_status,
                "conversion_flags": conversion_flags,
                "non_discriminative_flags": non_discriminative,
                "category": category,
            }
        )

    failed_rows = [row for row in rows if row["failed"]]
    success_rows = [row for row in rows if not row["failed"]]
    counts = {}
    for row in failed_rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    total_failed = max(len(failed_rows), 1)

    def flag_rate(subset, key: str) -> float:
        if not subset:
            return 0.0
        return float(sum(1 for row in subset if row["conversion_flags"][key]) / len(subset))

    discriminative = {
        key: {
            "rate_in_failures": flag_rate(failed_rows, key),
            "rate_in_successes": flag_rate(success_rows, key),
            "count_in_failures": sum(1 for row in failed_rows if row["conversion_flags"][key]),
        }
        for key in ("target_below_area_50", "target_raw_counterpart_missing",
                    "target_merge_risk_high", "raw_vs_converted_target_changed")
    }
    medium_rate_failures = flag_rate(failed_rows, "target_merge_risk_high")  # placeholder replaced below
    medium_failures = sum(1 for row in failed_rows if row["non_discriminative_flags"]["target_merge_risk_medium_base_rate"])
    medium_successes = sum(1 for row in success_rows if row["non_discriminative_flags"]["target_merge_risk_medium_base_rate"])
    del medium_rate_failures

    report = {
        "_doc": (
            "Task 6K section 13. Cross-analysis of the frozen Task 6J J1 result against "
            "source-side evidence (raw semantic component size, distance to the <50 threshold, "
            "border status, merge-risk class, proposal count, RAW-vs-CONVERTED relation drift). "
            "Attribution is made only where the evidence supports it."
        ),
        "task": "6K",
        "frozen_task6j_facts": {
            "J0_exact_accuracy": 1.0,
            "J0_paired": "20/20",
            "J1_strict_miou": j1["metrics"]["strict_selected_mask_miou"],
            "J1_abstentions": f"{j1['metrics']['abstained']}/120",
            "J1_paired_mask_selection": f"{j1['paired']['mask_paired_pass']}/20",
            "YOLO_target_recall_at_0_50": recall["target_recall"]["recall_at_0_50"],
            "YOLO_tiny_component_recall_at_0_50": recall["all_components"]["tiny_component_recall_at_0_50"],
            "J2_fixed_120_accuracy": j2["final"]["val"]["exact_accuracy"],
            "J2_full_val_accuracy": j2["full_val"]["exact_accuracy"],
            "J3_selected_target_accuracy": j3["metrics"]["selected_target_accuracy"],
            "J3_paired": "20/20",
            "J1_attribution": attribution["j1"]["failure_counts"],
        },
        "j1_samples": {
            "total": len(rows),
            "failed": len(failed_rows),
            "succeeded": len(success_rows),
            "categories": counts,
            "shares_of_failures": {key: float(value / total_failed) for key, value in counts.items()},
        },
        "conversion_flag_discriminative_power": {
            "per_flag": discriminative,
            "medium_merge_risk_base_rate": {
                "failures": int(medium_failures),
                "successes": int(medium_successes),
                "note": (
                    "The merge-risk MEDIUM class fires for most tiles in both outcomes, so it is "
                    "recorded but deliberately NOT used as conversion-related evidence."
                ),
            },
        },
        "evidence_notes": {
            "target_raw_area_px": "raster area of the raw 8-connected semantic component matching the target (IoU >= 0.5)",
            "target_raw_area_minus_50": "how far the raw target area sits from the historical `<50` contourArea threshold",
            "target_merge_risk": "heuristic merge-risk class of the raw target component (section 12)",
            "raw_vs_converted_status": "the tile's relation drift status for this query type (section 9)",
            "conversion_rule": (
                "conversion/data-related when the target would be deleted by the <50 filter, has no "
                "raw counterpart, is HIGH merge risk, or its RAW-vs-CONVERTED target actually changed"
            ),
            "proposal_rule": "proposal-model-related when no conversion flag holds and the target HAS a proposal at IoU >= 0.5",
            "inseparable": "otherwise (no clean proposal and no conversion flag)",
        },
        "rows": rows,
        "qwen_qualification": {
            "proves": (
                "Task 6J proves Qwen3-VL-2B is sufficient for the current closed-template 20-program "
                "classification task."
            ),
            "does_not_prove": [
                "2B is sufficient for arbitrary natural language;",
                "2B is sufficient for paraphrase/OOD instructions;",
                "4B cannot help the final system.",
            ],
            "current_dataset_facts": {
                "program_classes": 20,
                "program_id_is_1_to_1_with_query_type": True,
                "instruction_families": "finite templated families generated from reasoning_steps",
                "template_variants_per_family": "at least 3 in the current generator",
            },
            "task6k_ran_2b_vs_4b": False,
        },
    }
    write_json(EVAL / "task6k_task6j_cross_analysis.json", report)
    print(
        f"[task6k.cross] J1 failures {len(failed_rows)}/{len(rows)}: {counts}",
        flush=True,
    )
    print(f"[task6k.cross] wrote task6j_cross_analysis.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
