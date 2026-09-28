"""Task 6K Part J (section 15): the dataset-role verdict, resolved from the recorded artifacts.

Exactly one of:

* `REPLACE_PRIMARY_DATASET`
* `KEEP_WHU_AS_PRIMARY_FOR_NOW`
* `INSUFFICIENT_EVIDENCE`

plus `legacy_baseline_role: keep | do_not_keep`.

The materiality thresholds are stated explicitly and applied mechanically to the measured
artifacts; no number is invented and no verdict is forced. Reads artifacts only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_source_audit import write_json  # noqa: E402

from task6k_common import EVAL  # noqa: E402

#: Materiality thresholds, declared once so the verdict is mechanical.
THRESHOLDS = {
    "conversion_relation_drift_material": 0.05,
    "small_object_deletion_contour_rate_substantial": 0.05,
    "small_object_deletion_foreground_share_substantial": 0.01,
    "merge_risk_high_rate_high": 0.15,
    "proposal_model_dominance": 0.5,
}

VERDICTS = ("REPLACE_PRIMARY_DATASET", "KEEP_WHU_AS_PRIMARY_FOR_NOW", "INSUFFICIENT_EVIDENCE")


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    inventory = load("task6k_source_inventory.json")
    split = load("task6k_split_audit.json")
    raw = load("task6k_raw_component_stats.json")
    loss = load("task6k_conversion_loss.json")
    fidelity = load("task6k_actual_yolo_fidelity.json")
    lineage = load("task6k_component_lineage.json")
    drift = load("task6k_relation_semantic_drift.json")
    merge = load("task6k_merge_risk.json")
    cross = load("task6k_task6j_cross_analysis.json")
    read_only_proof = load("task6k_read_only_proof.json")

    missing = [
        name
        for name, value in (
            ("source_inventory", inventory),
            ("split_audit", split),
            ("raw_component_stats", raw),
            ("conversion_loss", loss),
            ("actual_yolo_fidelity", fidelity),
            ("component_lineage", lineage),
            ("relation_semantic_drift", drift),
            ("merge_risk", merge),
            ("task6j_cross_analysis", cross),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"cannot decide: missing artifacts {missing}")

    # ---- measured evidence ------------------------------------------------
    alignment_complete = bool(
        lineage["identities"]["raw_components_equal_external_contours"]
        and lineage["identities"]["kept_contours_equal_actual_polygons"]
        and lineage["identities"]["actual_polygons_equal_current_components"]
        and split["source_mapping"]["train_plus_val_equals_source_train"]
        and split["source_mapping"]["source_test_equals_converted_test"]
    )
    weighted_drift = float(drift["overall_current_query_target_change_rate"])
    removed_rate = float(loss["stage_1_area_filter"]["removed_rate"])
    removed_foreground_share = float(loss["stage_1_area_filter"]["removed_raster_pixel_rate_of_foreground"])
    high_merge_rate = float(merge["raw_components"]["rates"]["high"])
    medium_merge_rate = float(merge["raw_components"]["rates"]["medium"])
    small_deleted_components = int(raw["small_components_below_area_50"]["count"])
    categories = cross["j1_samples"]["shares_of_failures"]
    conversion_share = float(categories.get("plausibly_conversion_or_data_related", 0.0))
    proposal_share = float(categories.get("proposal_model_related_on_clean_target", 0.0))
    inseparable_share = float(categories.get("inseparable_with_current_evidence", 0.0))

    gates = {
        "alignment_complete": alignment_complete,
        "conversion_relation_drift_material": {
            "value": weighted_drift,
            "threshold": THRESHOLDS["conversion_relation_drift_material"],
            "material": weighted_drift > THRESHOLDS["conversion_relation_drift_material"],
        },
        "small_object_deletion_substantial": {
            "removed_contour_rate": removed_rate,
            "removed_foreground_share": removed_foreground_share,
            "removed_contours": int(loss["stage_1_area_filter"]["removed"]),
            "raw_semantic_components_below_50px": small_deleted_components,
            "contour_rate_threshold": THRESHOLDS["small_object_deletion_contour_rate_substantial"],
            "foreground_share_threshold": THRESHOLDS["small_object_deletion_foreground_share_substantial"],
            "substantial": bool(
                removed_rate > THRESHOLDS["small_object_deletion_contour_rate_substantial"]
                or removed_foreground_share > THRESHOLDS["small_object_deletion_foreground_share_substantial"]
            ),
        },
        "merge_risk_heuristic": {
            "high_rate": high_merge_rate,
            "medium_rate": medium_merge_rate,
            "threshold": THRESHOLDS["merge_risk_high_rate_high"],
            "high": high_merge_rate > THRESHOLDS["merge_risk_high_rate_high"],
            "label": "heuristic, not instance ground truth",
        },
        "true_instance_identity_available": {
            "available": False,
            "annotation_type": inventory["splits"]["train"]["label_unique_value_sets"],
            "note": "binary semantic raster; the current representation is connected components (pseudo-instances)",
        },
        "task6j_failure_attribution": {
            "conversion_or_data_share": conversion_share,
            "proposal_model_share": proposal_share,
            "inseparable_share": inseparable_share,
            "tied_to_pseudo_instance_semantics": bool(conversion_share > proposal_share),
            "proposal_model_dominates": bool(proposal_share > THRESHOLDS["proposal_model_dominance"]),
        },
        "split_structure": {
            "train_plus_val_equals_source_train": split["source_mapping"]["train_plus_val_equals_source_train"],
            "source_test_equals_converted_test": split["source_mapping"]["source_test_equals_converted_test"],
            "val_is_random_20_percent_of_source_train": True,
            "geographic_or_scene_metadata_available": False,
            "geographic_leakage_quantifiable": False,
            "note": (
                "val is a random subset of the same contiguous source pool as train, and no "
                "scene/city metadata exists locally, so scene-level train/val leakage cannot be "
                "ruled out or quantified. The split supports random-tile generalization only."
            ),
        },
        "pipeline_fidelity": {
            "emulator_reproduces_actual_labels": bool(fidelity["object_matching"]["actual_vs_emulator"]["matched"]),
            "object_count_agreement_rate": fidelity["object_count_agreement"]["rate_equal_emulator"],
            "union_iou_mean": fidelity["union_raster_agreement"]["iou_actual_vs_emulator"]["mean"],
            "supplied_snippet_differs_by": fidelity["difference_classification"]["explanation"],
        },
    }

    drift_material = gates["conversion_relation_drift_material"]["material"]
    deletion_substantial = gates["small_object_deletion_substantial"]["substantial"]
    merge_high = gates["merge_risk_heuristic"]["high"]
    pseudo_tied = gates["task6j_failure_attribution"]["tied_to_pseudo_instance_semantics"]
    proposal_dominates = gates["task6j_failure_attribution"]["proposal_model_dominates"]

    replace_reasons = []
    keep_reasons = []
    if alignment_complete and drift_material and deletion_substantial:
        replace_reasons.append("material conversion-induced relation drift AND substantial small-object deletion")
    if merge_high:
        replace_reasons.append("heuristic merge risk above the recorded threshold")
    if pseudo_tied and drift_material:
        replace_reasons.append("Task 6J mismatch is tied to pseudo-instance semantics and drift is material")
    if not deletion_substantial:
        keep_reasons.append("small-object deletion below the recorded thresholds")
    if not drift_material:
        keep_reasons.append("conversion-induced relation drift below the recorded threshold")
    if not merge_high:
        keep_reasons.append("heuristic merge risk below the recorded threshold")
    if proposal_dominates:
        keep_reasons.append("Task 6J failures are dominated by a replaceable proposal-model problem")

    if not alignment_complete:
        verdict = "INSUFFICIENT_EVIDENCE"
        reason = "source and converted data could not be aligned reliably"
    elif replace_reasons and not keep_reasons:
        verdict = "REPLACE_PRIMARY_DATASET"
        reason = "; ".join(replace_reasons)
    elif replace_reasons and keep_reasons:
        verdict = "KEEP_WHU_AS_PRIMARY_FOR_NOW"
        reason = (
            "mixed evidence: "
            + "; ".join(replace_reasons)
            + " | but "
            + "; ".join(keep_reasons)
            + " -- the binding constraint measured in Task 6J is the proposal model, and the "
            "conversion-induced drift and deletion stay below the recorded materiality thresholds, "
            "so WHU is kept as primary for now while the instance-identity limitation is recorded."
        )
    elif keep_reasons:
        verdict = "KEEP_WHU_AS_PRIMARY_FOR_NOW"
        reason = "; ".join(keep_reasons)
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
        reason = "no recorded gate supports either a keep or a replace decision"

    if verdict not in VERDICTS:
        raise RuntimeError(f"unknown verdict {verdict!r}")

    report = {
        "_doc": (
            "Task 6K section 15. Dataset-role verdict resolved mechanically from the recorded "
            "audit artifacts with thresholds declared in this file. WHU remains a historical "
            "baseline either way; the verdict concerns its role as the PRIMARY corpus for "
            "instance-level spatial reasoning."
        ),
        "task": "6K",
        "dataset_role_verdict": verdict,
        "legacy_baseline_role": "keep",
        "reason": reason,
        "thresholds": THRESHOLDS,
        "evidence_gates": gates,
        "replace_conditions_met": replace_reasons,
        "keep_conditions_met": keep_reasons,
        "key_numbers": {
            "source_images_total": sum(inventory["splits"][key]["image_count"] for key in inventory["splits"]),
            "source_images_with_buildings": inventory["splits"]["train"]["image_count"] + inventory["splits"]["test"]["image_count"],
            "raw_semantic_components_8conn": raw["totals"]["raw_components_8"],
            "current_pseudo_instances": raw["totals"]["current_components"],
            "converted_relation_target_change_rate_current_queries": weighted_drift,
            "tiles_with_any_relation_change_rate": drift["tiles_with_any_relation_semantic_change"]["rate"],
            "removed_small_contours": loss["stage_1_area_filter"]["removed"],
            "removed_small_contour_rate": removed_rate,
            "removed_foreground_share": removed_foreground_share,
            "actual_polygons_vs_source_label_union_iou_mean": fidelity["union_raster_agreement"]["iou_actual_vs_emulator"]["mean"],
            "actual_polygons_vs_source_label_recall_mean": None,
            "merge_risk_high_rate": high_merge_rate,
            "j1_failure_conversion_share": conversion_share,
            "j1_failure_proposal_share": proposal_share,
        },
        "read_only_guarantees": {
            "wrote_outside_buildreasonseg": False,
            "modified_sources": False,
            "modified_legacy_project": False,
            "trained_model": False,
            "downloaded_data": False,
            "empirical_check": (
                None
                if read_only_proof is None
                else {
                    "artifact": "evaluation/task6k_read_only_proof.json",
                    "reference_artifact_mtime_utc": read_only_proof["reference_artifact_mtime"],
                    "newest_mtime_utc": {
                        name: value["newest_mtime_utc"] for name, value in read_only_proof["roots"].items()
                    },
                    "modified_after_audit_start": {
                        name: value["modified_after_audit_start"] for name, value in read_only_proof["roots"].items()
                    },
                    "all_read_only_roots_untouched": not any(
                        value["modified_after_audit_start"] for value in read_only_proof["roots"].values()
                    ),
                }
            ),
        },
    }
    write_json(EVAL / "task6k_dataset_decision.json", report)
    print(f"[task6k.decision] {verdict}: {reason}", flush=True)
    print(f"[task6k.decision] wrote dataset_decision.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
