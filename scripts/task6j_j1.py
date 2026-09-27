"""Task 6J sections 6-9: Stage J1 -- oracle program + predicted YOLO proposals.

Read-only reuse of the frozen YOLOv8m-seg-WHU baseline: provenance (model hash + environment
versions) is verified from the recorded inference run; proposals (mask / bbox / centroid / area /
confidence) are loaded from the gitignored cache. Two audits run in the mvp environment:

1. proposal recall -- for every GT target component (and, where practical, for every building
   component of the fixed images) the best predicted-proposal IoU, recall @0.25/0.50/0.75,
   missing-target rate, proposal-count distribution and duplicate/border/tiny diagnostics;
2. oracle-program execution -- the canonical program template of each record's query type runs
   over the predicted proposal geometry with NO GT id/mask entering execution; the selected
   proposal mask is scored against the GT target mask (strict mIoU, Dice, paired own-vs-cross).

J1 viability gate: target proposal recall@0.50 >= 0.75, oracle-program selected-mask mIoU >= 0.30,
paired mask selection >= 12/20.

Writes `evaluation/task6j_yolo_proposal_recall.json` and `evaluation/task6j_j1_oracle_program_yolo.json`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "spatial_reasoning"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.structured_grounding import CandidateSet, execute_program_by_id  # noqa: E402
from dataset_access import DEFAULT_DATASET_ROOT  # noqa: E402
from thresholds import load_config as load_relation_config  # noqa: E402

from component_quality import classify_image  # noqa: E402
from task6j_common import (  # noqa: E402
    EVAL,
    fixed_validation_material,
    gate_report,
    geometry_for,
    paired_sample_lists,
    write_json,
)

OUT_RECALL = EVAL / "task6j_yolo_proposal_recall.json"
OUT_J1 = EVAL / "task6j_j1_oracle_program_yolo.json"
CACHE_DIR = REPO_ROOT / "artifacts" / "task6j_yolo_proposals"
MODEL_PATH = (
    REPO_ROOT.parent
    / "WHU_Building_Segment"
    / "runs"
    / "segment"
    / "logs"
    / "whu_building_v1"
    / "weights"
    / "best.pt"
)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def load_proposals(image_id: str):
    data = np.load(CACHE_DIR / f"{image_id}.npz")
    masks = data["masks"]
    boxes = data["boxes"]
    confidences = data["confidences"]
    return [
        {
            "mask": masks[i].astype(bool),
            "bbox_xyxy": [float(v) for v in boxes[i]],
            "confidence": float(confidences[i]),
        }
        for i in range(masks.shape[0])
    ]


def main() -> int:
    started = time.time()
    config = load_relation_config()
    val_samples, pairs = fixed_validation_material()
    a_samples, b_samples = paired_sample_lists(pairs)

    provenance_path = CACHE_DIR / "provenance.json"
    if not provenance_path.is_file():
        raise SystemExit("YOLO proposal cache missing; run task6j_yolo_manifest.py + task6j_yolo_infer.py")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    import hashlib

    actual_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    provenance_check = {
        "model_on_disk_hash": actual_hash,
        "recorded_hash": provenance["provenance"]["model_sha256"],
        "matches": actual_hash == provenance["provenance"]["model_sha256"],
        "env": {
            "python": provenance["provenance"]["python_version"].splitlines()[0],
            "ultralytics": provenance["provenance"]["ultralytics_version"],
            "torch": provenance["provenance"]["torch_version"],
        },
    }

    # ---- 1. proposal recall ------------------------------------------------
    target_rows = []
    image_component_rows = []
    for sample in val_samples + a_samples + b_samples:
        proposals = load_proposals(sample.image_id)
        proposal_masks = [p["mask"] for p in proposals]
        target_mask = sample.target_mask()
        best = max((_iou(m, target_mask) for m in proposal_masks), default=0.0)
        target_rows.append(
            {
                "sample_id": str(sample.sample_id),
                "image_id": str(sample.image_id),
                "level": int(sample.level),
                "query_type": str(sample.query_type),
                "target_component_id": int(sample.target_component_id),
                "best_proposal_iou": best,
                "proposal_count": len(proposals),
            }
        )
        # all-component recall for this image (practical, cached proposals)
        geometry = geometry_for(sample)
        quality = classify_image(geometry, config)
        component_map = geometry.load_map(DEFAULT_DATASET_ROOT)
        for component in geometry.components:
            mask = component_map == component.component_id
            component_best = max((_iou(m, mask) for m in proposal_masks), default=0.0)
            flags = quality.flags(component.component_id)
            image_component_rows.append(
                {
                    "image_id": str(sample.image_id),
                    "component_id": int(component.component_id),
                    "best_proposal_iou": component_best,
                    "is_target": int(component.component_id) == int(sample.target_component_id),
                    "tiny": bool(flags.tiny_component),
                    "border": bool(flags.touches_image_border),
                    "merge_suspected": bool(flags.suspected_large_merge),
                }
            )

    def recall_at(rows, threshold):
        if not rows:
            return None
        return float(sum(1 for row in rows if row["best_proposal_iou"] >= threshold) / len(rows))

    best_ious = [row["best_proposal_iou"] for row in target_rows]
    proposal_counts = [row["proposal_count"] for row in target_rows]
    target_component_rows = [row for row in image_component_rows if row["is_target"]]
    non_target_rows = [row for row in image_component_rows if not row["is_target"]]

    # duplicate/merged diagnostics: within-image proposal pairs with IoU > 0.7
    duplicate_images = 0
    duplicate_pairs = 0
    total_pairs_checked = 0
    for image_id in sorted({row["image_id"] for row in target_rows}):
        proposals = load_proposals(image_id)
        masks = [p["mask"] for p in proposals]
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                total_pairs_checked += 1
                if _iou(masks[i], masks[j]) > 0.7:
                    duplicate_pairs += 1
                    duplicate_images += 1
                    break
            else:
                continue
            break

    recall_report = {
        "_doc": (
            "Task 6J section 8. Proposal recall of the frozen YOLOv8m-seg-WHU baseline over the "
            "fixed 120 val + 40 paired images (131 unique images). GT is evaluation-only: the "
            "proposals come from the model output cache and are never altered with GT."
        ),
        "task": "6J",
        "provenance": provenance["provenance"],
        "provenance_check": provenance_check,
        "target_recall": {
            "samples": len(target_rows),
            "recall_at_0_25": recall_at(target_rows, 0.25),
            "recall_at_0_50": recall_at(target_rows, 0.50),
            "recall_at_0_75": recall_at(target_rows, 0.75),
            "mean_best_iou": _mean(best_ious),
            "median_best_iou": float(np.median(best_ious)) if best_ious else None,
            "missing_target_rate": float(sum(1 for v in best_ious if v < 0.5) / max(len(best_ious), 1)),
            "proposal_count_mean": _mean(proposal_counts),
            "proposal_count_median": float(np.median(proposal_counts)) if proposal_counts else None,
            "proposal_count_min": int(min(proposal_counts)) if proposal_counts else None,
            "proposal_count_max": int(max(proposal_counts)) if proposal_counts else None,
        },
        "all_components": {
            "components": len(image_component_rows),
            "target_recall_at_0_50": recall_at(target_component_rows, 0.50),
            "non_target_recall_at_0_50": recall_at(non_target_rows, 0.50),
            "target_mean_best_iou": _mean([r["best_proposal_iou"] for r in target_component_rows]),
            "tiny_component_recall_at_0_50": recall_at(
                [r for r in image_component_rows if r["tiny"]], 0.50
            ),
            "border_component_recall_at_0_50": recall_at(
                [r for r in image_component_rows if r["border"]], 0.50
            ),
            "merge_suspected_recall_at_0_50": recall_at(
                [r for r in image_component_rows if r["merge_suspected"]], 0.50
            ),
        },
        "duplicates": {
            "proposal_pairs_checked": total_pairs_checked,
            "pairs_with_iou_gt_0_7": duplicate_pairs,
            "images_with_duplicate_pair": duplicate_images,
        },
        "target_rows": target_rows,
    }

    # ---- 2. oracle-program execution over proposals ------------------------
    def execute_sample(sample):
        proposals = load_proposals(sample.image_id)
        image = sample.image_rgb()
        candidates = CandidateSet.from_proposals(image.shape[1], image.shape[0], proposals)
        result = execute_program_by_id(sample.query_type, candidates, config)
        selected_mask = None
        if result.selected_id is not None:
            selected_mask = candidates.by_id()[result.selected_id].mask
        target_mask = sample.target_mask()
        iou = _iou(selected_mask, target_mask) if selected_mask is not None else 0.0
        dice = _dice(selected_mask, target_mask) if selected_mask is not None else 0.0
        return {
            "sample_id": str(sample.sample_id),
            "image_id": str(sample.image_id),
            "level": int(sample.level),
            "query_type": str(sample.query_type),
            "selected_proposal_id": result.selected_id,
            "abstained": bool(result.abstained),
            "abstain_reason": result.reason,
            "miou": iou,
            "dice": dice,
            "proposal_count": len(candidates.candidates),
        }

    val_rows = [execute_sample(sample) for sample in val_samples]
    pair_rows = []
    for sample_a, sample_b, pair in zip(a_samples, b_samples, pairs):
        row_a = execute_sample(sample_a)
        row_b = execute_sample(sample_b)
        mask_a = None
        mask_b = None
        if row_a["selected_proposal_id"] is not None:
            mask_a = load_proposals(sample_a.image_id)[row_a["selected_proposal_id"] - 1]["mask"]
        if row_b["selected_proposal_id"] is not None:
            mask_b = load_proposals(sample_b.image_id)[row_b["selected_proposal_id"] - 1]["mask"]
        target_a = sample_a.target_mask()
        target_b = sample_b.target_mask()
        own_a = _iou(mask_a, target_a) if mask_a is not None else 0.0
        own_b = _iou(mask_b, target_b) if mask_b is not None else 0.0
        cross_a = _iou(mask_a, target_b) if mask_a is not None else 0.0
        cross_b = _iou(mask_b, target_a) if mask_b is not None else 0.0
        own = 0.5 * (own_a + own_b)
        cross = 0.5 * (cross_a + cross_b)
        pair_rows.append(
            {
                "image_id": str(pair["image_id"]),
                "a": str(pair["a"]),
                "b": str(pair["b"]),
                "mask_paired_pass": bool(own > cross),
                "own_iou": float(own),
                "cross_iou": float(cross),
                "own_minus_cross_margin": float(own - cross),
                "row_a": row_a,
                "row_b": row_b,
            }
        )

    paired_pass = sum(1 for row in pair_rows if row["mask_paired_pass"])
    strict_miou = _mean([row["miou"] for row in val_rows]) or 0.0
    mean_dice = _mean([row["dice"] for row in val_rows])
    j1_gate_cfg = {"recall_at_0_50_min": 0.75, "oracle_program_miou_min": 0.30, "paired_mask_min": 12}
    j1_checks = {
        "recall_at_0_50_ge": float(recall_report["target_recall"]["recall_at_0_50"]) >= 0.75,
        "oracle_program_miou_ge": strict_miou >= 0.30,
        "paired_mask_ge": paired_pass >= 12,
        "provenance_matches": provenance_check["matches"],
    }
    j1_gate = gate_report(j1_checks, j1_gate_cfg)

    j1_report = {
        "_doc": (
            "Task 6J section 9. The canonical program template of each record's query type is "
            "executed over the predicted YOLO proposal geometry with NO ground-truth id, mask or "
            "geometry entering execution; the selected proposal mask is then scored against the "
            "GT target mask (strict mIoU, Dice, paired own-vs-cross mask IoU)."
        ),
        "task": "6J",
        "stage": "J1",
        "provenance_check": provenance_check,
        "metrics": {
            "strict_selected_mask_miou": strict_miou,
            "mean_dice": mean_dice,
            "abstained": sum(1 for row in val_rows if row["abstained"]),
            "abstain_reasons": {
                reason: sum(1 for row in val_rows if row["abstain_reason"] == reason)
                for reason in sorted({row["abstain_reason"] for row in val_rows if row["abstain_reason"]})
            },
        },
        "paired": {
            "paired_total": len(pair_rows),
            "mask_paired_pass": paired_pass,
            "mean_own_iou": _mean([row["own_iou"] for row in pair_rows]),
            "mean_cross_iou": _mean([row["cross_iou"] for row in pair_rows]),
            "mean_own_minus_cross_margin": _mean([row["own_minus_cross_margin"] for row in pair_rows]),
            "rows": pair_rows,
        },
        "val_rows": val_rows,
        "gate": j1_gate,
        "viable": bool(j1_gate["passed"]),
        "no_gt_leakage": {
            "execution_inputs": "canonical program template + predicted proposal geometry",
            "gt_used_for": ["post-selection scoring only"],
        },
        "seconds": round(time.time() - started, 2),
    }

    write_json(OUT_RECALL, recall_report)
    write_json(OUT_J1, j1_report)
    print(
        f"[task6j.j1] recall@0.25/0.50/0.75 = "
        f"{recall_report['target_recall']['recall_at_0_25']:.3f}/"
        f"{recall_report['target_recall']['recall_at_0_50']:.3f}/"
        f"{recall_report['target_recall']['recall_at_0_75']:.3f}, mean best IoU "
        f"{recall_report['target_recall']['mean_best_iou']:.3f}, missing "
        f"{recall_report['target_recall']['missing_target_rate']:.3f}, proposals/img "
        f"{recall_report['target_recall']['proposal_count_mean']:.1f}",
        flush=True,
    )
    print(
        f"[task6j.j1] oracle-program+YOLO: mIoU {strict_miou:.4f} Dice {mean_dice:.4f} "
        f"paired {paired_pass}/{len(pair_rows)} own {j1_report['paired']['mean_own_iou']:.3f} "
        f"cross {j1_report['paired']['mean_cross_iou']:.3f} abstained {j1_report['metrics']['abstained']} "
        f"gate {j1_gate['passed']}",
        flush=True,
    )
    print(f"[task6j.j1] wrote {OUT_RECALL.name} + {OUT_J1.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
