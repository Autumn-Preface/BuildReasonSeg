"""Task 6K.1 Part G (section 11): re-attribute the frozen Task 6J result against native vector truth.

For every J1 (oracle program + YOLO proposals) sample this asks, using the native vector map:

* is the supervision target a SINGLE native building, or is it part of a pseudo-instance merge of
  several native buildings?
* did the target have a YOLO proposal at IoU >= 0.5 (a clean proposal existed)?
* hence: is the failure attributable to the proposal model on a clean single building, to a merged
  pseudo-instance target, or to something else (inseparable)?

Writes `evaluation/task6k1_task6j_vector_cross_analysis.json`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from task6k1_common import (  # noqa: E402
    CROPPED_ROOT,
    EVAL,
    load_instance_view,
    match_instances,
    write_json,
)

OUT = EVAL / "task6k1_task6j_vector_cross_analysis.json"
CONTAINMENT = 0.80


def main() -> int:
    started = time.time()
    j1 = json.loads((EVAL / "task6j_j1_oracle_program_yolo.json").read_text(encoding="utf-8"))
    recall = json.loads((EVAL / "task6j_yolo_proposal_recall.json").read_text(encoding="utf-8"))
    recall_by_sample = {row["sample_id"]: row for row in recall["target_rows"]}

    records = {}
    base = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1"
    for split in ("train", "val", "test"):
        path = base / f"{split}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[str(record["sample_id"])] = record

    rows = []
    for row in j1["val_rows"]:
        sample_id = str(row["sample_id"])
        record = records.get(sample_id, {})
        image_id = str(row["image_id"])
        view = load_instance_view(image_id)
        target_component = int(record.get("target_component_id", -1))

        target_mask = None
        component_map_path = None
        for split in ("val", "train", "test"):
            candidate = REPO_ROOT / "datasets" / "whu" / "components" / split / f"{image_id}.png"
            if candidate.is_file():
                component_map_path = candidate
                break
        if component_map_path is not None:
            import cv2

            component_map = cv2.imread(str(component_map_path), cv2.IMREAD_GRAYSCALE)
            if component_map is not None:
                target_mask = component_map == target_component

        contained = []
        if view is not None and target_mask is not None and target_mask.any():
            label_map = view["label_map"]
            for local_id in [int(v) for v in np.unique(label_map) if int(v) != 0]:
                mask = label_map == local_id
                area = int(mask.sum())
                if area == 0:
                    continue
                overlap = int(np.logical_and(mask, target_mask).sum())
                if overlap / area >= CONTAINMENT:
                    contained.append(
                        {
                            "global_feature_id": int(view["ids"][local_id - 1]),
                            "area_px": area,
                            "containment": overlap / area,
                        }
                    )

        best_proposal_iou = float(recall_by_sample.get(sample_id, {}).get("best_proposal_iou", 0.0))
        failed = bool(row["abstained"]) or float(row["miou"]) < 0.5
        n_native_in_target = len(contained)
        native_target_areas = [entry["area_px"] for entry in contained]
        # Section 11 classification
        if n_native_in_target >= 2:
            category = "pseudo_label_definition_error"
        elif n_native_in_target == 1 and failed and best_proposal_iou >= 0.5:
            category = "proposal_model_error_against_clean_vector_target"
        elif n_native_in_target == 1 and failed:
            category = "representation_mismatch"
        elif n_native_in_target == 1:
            category = "success_single_vector_target"
        else:
            category = "ambiguous_or_inseparable"

        rows.append(
            {
                "sample_id": sample_id,
                "image_id": image_id,
                "level": int(row["level"]),
                "query_type": str(row["query_type"]),
                "j1_miou": float(row["miou"]),
                "j1_abstained": bool(row["abstained"]),
                "failed": failed,
                "best_proposal_iou": best_proposal_iou,
                "clean_correspondence": bool(n_native_in_target == 1),
                "target_component_id": target_component,
                "native_buildings_inside_target": n_native_in_target,
                "native_buildings": contained,
                "native_target_area_px": native_target_areas[0] if native_target_areas else None,
                "category": category,
            }
        )

    failed_rows = [row for row in rows if row["failed"]]
    success_rows = [row for row in rows if not row["failed"]]
    failure_counts = Counter(row["category"] for row in failed_rows)
    success_counts = Counter(row["category"] for row in success_rows)

    merged_target_rate = failure_counts.get("pseudo_label_definition_error", 0) / max(len(failed_rows), 1)
    proposal_rate = failure_counts.get("proposal_model_error_against_clean_vector_target", 0) / max(
        len(failed_rows), 1
    )
    clean_correspondence_failures = [row for row in failed_rows if row["clean_correspondence"]]
    recall_on_clean_targets = (
        sum(1 for row in clean_correspondence_failures if row["best_proposal_iou"] >= 0.5)
        / max(len(clean_correspondence_failures), 1)
    )

    report = {
        "_doc": (
            "Task 6K.1 section 11. Frozen Task 6J J1 samples re-attributed against the native vector "
            "map: was the supervision target a single native building or part of a pseudo-instance "
            "merge, and did it have a clean YOLO proposal?"
        ),
        "task": "6K.1",
        "frozen_task6j_facts": {
            "J1_strict_miou": j1["metrics"]["strict_selected_mask_miou"],
            "J1_abstentions": j1["metrics"]["abstained"],
            "J1_paired_mask_selection": j1["paired"]["mask_paired_pass"],
            "YOLO_recall_at_0_50": recall["target_recall"]["recall_at_0_50"],
        },
        "j1_samples": {
            "total": len(rows),
            "failed": len(failed_rows),
            "succeeded": len(success_rows),
            "failure_categories": dict(failure_counts),
            "failure_shares": {k: v / max(len(failed_rows), 1) for k, v in failure_counts.items()},
            "success_categories": dict(success_counts),
            "failures_with_clean_vector_correspondence": len(clean_correspondence_failures),
            "yolo_proposal_recall_on_clean_vector_targets": recall_on_clean_targets,
            "share_of_failures_pseudo_label_definition_error": merged_target_rate,
            "share_of_failures_proposal_model_error": proposal_rate,
        },
        "interpretation": {
            "pseudo_label_definition_error": (
                "The supervision target is itself a pseudo-instance covering >= 2 native buildings, "
                "so 'the largest/leftmost building' is ambiguous between them."
            ),
            "proposal_model_error_against_clean_vector_target": (
                "The target is exactly one native building and a YOLO proposal at IoU >= 0.5 existed, "
                "so the failure is in the proposal/executor chain rather than in the instance definition."
            ),
            "representation_mismatch": (
                "The target is one native building but no proposal reached IoU 0.5: proposal or "
                "representation mismatch."
            ),
            "ambiguous_or_inseparable": (
                "No single native building corresponds to the target, so the failure cannot be "
                "attributed with the current evidence."
            ),
        },
        "rows": rows,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6k1.cross] J1 failures {len(failed_rows)}/{len(rows)}: {dict(failure_counts)}",
        flush=True,
    )
    print(f"[task6k1.cross] wrote {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
