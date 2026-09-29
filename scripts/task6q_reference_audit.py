"""Task 6Q Part D (sections 5-8) — reference-level audit of the frozen-proposal resolver.

Verifies the frozen Task 6M.1 YOLO26m-seg checkpoint (SHA256) and the frozen inference configuration
(imgsz 640 / conf 0.10 / max_det 100 / default NMS / no TTA / no tiling / no sweep), then, on the exact
frozen Task 6P **RefValUnique** pack:

* proposal coverage — best IoU among ALL proposals and among the family-ELIGIBLE proposals
  (coverage@0.25/0.50/0.75, overall and by family);
* deterministic selected-reference quality — mask IoU, Dice, normalized centroid error, area ratio,
  abstention (aggregated overall and by family);
* exactly one failure category per record in the fixed priority order.

Each source tile is run through the frozen proposal model **exactly once**; the selected reference
masks are packed into a gitignored resolver cache for the downstream propagation stage so YOLO never
runs twice. Ground truth is used for evaluation only — never for eligibility, ranking or tie-breaks.

Writes:
* `evaluation/task6q_reference_resolver_val.json`
* `evaluation/task6q_reference_failure_attribution.json`
* `evaluation/task6q_proposal_cache_manifest.json` (optional manifest for the resolver cache)
* `artifacts/task6q/resolved_references.npz` (gitignored cache)

    python scripts/task6q_reference_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import canonical_instances, write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import read_pack  # noqa: E402
from buildreasonseg_mvp.task6q_reference_resolver import (  # noqa: E402
    FAMILIES,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    PROPOSAL_CONF,
    PROPOSAL_IMGSZ,
    PROPOSAL_MAX_DET,
    build_proposal,
    config_report,
    pack_masks,
    proposals_from_results,
    select_reference,
)

EVAL = REPO_ROOT / "evaluation"
REF_ROOT = REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
CACHE_ROOT = REPO_ROOT / "artifacts" / "task6q"
OUT_VAL = EVAL / "task6q_reference_resolver_val.json"
OUT_FAILURE = EVAL / "task6q_reference_failure_attribution.json"
OUT_MANIFEST = EVAL / "task6q_proposal_cache_manifest.json"
CACHE_PATH = CACHE_ROOT / "resolved_references.npz"
COVERAGE_THRESHOLDS = (0.25, 0.50, 0.75)
TILE_SIZE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = float(np.logical_and(left, right).sum())
    union = float(np.logical_or(left, right).sum())
    return (intersection + 1e-6) / (union + 1e-6)


def dice(left: np.ndarray, right: np.ndarray) -> float:
    intersection = float(np.logical_and(left, right).sum())
    return (2.0 * intersection + 1e-6) / (float(left.sum()) + float(right.sum()) + 1e-6)


def centroid(mask: np.ndarray) -> tuple[float, float]:
    height, width = mask.shape
    total = float(mask.sum())
    if total <= 0:
        return (0.5, 0.5)
    x = (np.arange(width) + 0.5) / width
    y = (np.arange(height) + 0.5) / height
    cx = float((mask.sum(axis=0) * x).sum() / total)
    cy = float((mask.sum(axis=1) * y).sum() / total)
    return cx, cy


def read_refval() -> list[dict]:
    payload = json.loads((REF_ROOT / "ref_val_unique.json").read_text(encoding="utf-8"))
    return payload["records"]


def gt_reference_mask(record: dict) -> np.ndarray:
    for instance in canonical_instances(record["tile_id"]):
        if instance.source_feature_id == record["reference_source_feature_id"]:
            return np.asarray(instance.mask, dtype=bool)
    raise KeyError(f"{record['tile_id']}: reference not found")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    started = time.time()
    checkpoint = REPO_ROOT / PROPOSAL_CHECKPOINT
    checkpoint_ok = checkpoint.is_file()
    actual_sha = sha256_file(checkpoint) if checkpoint_ok else None
    if not checkpoint_ok or actual_sha != PROPOSAL_CHECKPOINT_SHA256:
        write_json(OUT_VAL, {
            "_doc": "Task 6Q section 4. Frozen proposal checkpoint verification.",
            "task": "6Q", "stage": "B-verification",
            "checkpoint": {"path": str(checkpoint), "present": checkpoint_ok,
                           "expected_sha256": PROPOSAL_CHECKPOINT_SHA256, "actual_sha256": actual_sha,
                           "matches": actual_sha == PROPOSAL_CHECKPOINT_SHA256},
            "verdict": "PROPOSAL_CHECKPOINT_UNAVAILABLE",
        })
        print("[6q.audit] STOP PROPOSAL_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2

    records = read_refval()
    tiles = sorted({record["tile_id"] for record in records})
    print(f"[6q.audit] RefValUnique references {len(records)} over {len(tiles)} tiles", flush=True)

    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    rows = []
    selected_cache: dict[str, np.ndarray] = {}
    cache_meta = {
        "tile_ids": [],
        "family_abstained": [],
        "proposal_counts": [],
    }
    tile_reference_masks: dict[str, np.ndarray] = {}
    for index, tile_id in enumerate(tiles, start=1):
        image_path = next(Path(record["image_path"]) for record in records
                          if record["tile_id"] == tile_id)
        results = model.predict(source=str(image_path), imgsz=PROPOSAL_IMGSZ, conf=PROPOSAL_CONF,
                                max_det=PROPOSAL_MAX_DET, verbose=False, device=args.device,
                                retina_masks=True)[0]
        proposals = proposals_from_results(results)
        cache_meta["tile_ids"].append(tile_id)
        cache_meta["proposal_counts"].append(len(proposals))
        for family in FAMILIES:
            selection = select_reference(proposals, family)
            key = f"{tile_id}|{family}"
            cache_meta["family_abstained"].append(0 if not selection.abstained else 1)
            if selection.mask is not None:
                selected_cache[key] = selection.mask
        for record in [item for item in records if item["tile_id"] == tile_id]:
            gt = gt_reference_mask(record)
            family = record["reference_family"]
            selection = select_reference(proposals, family)
            all_ious = [iou(proposal.mask, gt) for proposal in proposals]
            eligible = [proposal for proposal in proposals
                        if proposal.index in selection.eligible_indices]
            eligible_ious = [iou(proposal.mask, gt) for proposal in eligible]
            best_all = max(all_ious) if all_ious else 0.0
            best_eligible = max(eligible_ious) if eligible_ious else 0.0
            selected_iou = iou(selection.mask, gt) if selection.mask is not None else 0.0
            selected_dice = dice(selection.mask, gt) if selection.mask is not None else 0.0
            if selection.mask is not None:
                pred_cx, pred_cy = centroid(selection.mask)
                gt_cx, gt_cy = centroid(gt)
                centroid_error = float(np.hypot(pred_cx - gt_cx, pred_cy - gt_cy)) / float(np.sqrt(2.0))
                area_ratio = float(selection.mask.sum()) / max(float(gt.sum()), 1.0)
            else:
                centroid_error = None
                area_ratio = None

            if not proposals:
                category = "NO_PROPOSALS"
            elif not eligible:
                category = "NO_ELIGIBLE_PROPOSALS"
            elif best_eligible < 0.50:
                category = "REFERENCE_NOT_COVERED_IOU50"
            elif selected_iou < 0.50:
                category = "EXTREME_SELECTION_WRONG"
            elif centroid_error is not None and centroid_error > 0.05:
                category = "SELECTED_MASK_GEOMETRY_POOR"
            else:
                category = "REFERENCE_OK"

            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "tile_id": tile_id,
                    "reference_family": family,
                    "reference_area_px": int(gt.sum()),
                    "proposals_total": len(proposals),
                    "proposals_eligible": len(eligible),
                    "best_iou_all": best_all,
                    "best_iou_eligible": best_eligible,
                    "selected_iou": selected_iou,
                    "selected_dice": selected_dice,
                    "selected_centroid_error": centroid_error,
                    "selected_area_ratio": area_ratio,
                    "abstained": bool(selection.abstained),
                    "abstention_reason": selection.reason,
                    "failure_category": category,
                }
            )
        if index % 50 == 0:
            print(f"[6q.audit] tiles {index}/{len(tiles)}", flush=True)

    # ---------------- aggregates
    def coverage(rows_subset: list[dict], key: str) -> dict:
        return {
            f"coverage@{threshold:.2f}": (
                float(np.mean([row[key] >= threshold for row in rows_subset])) if rows_subset else None
            )
            for threshold in COVERAGE_THRESHOLDS
        }

    def quality(rows_subset: list[dict]) -> dict:
        if not rows_subset:
            return {"records": 0}
        errors = [row["selected_centroid_error"] for row in rows_subset
                  if row["selected_centroid_error"] is not None]
        ratios = [row["selected_area_ratio"] for row in rows_subset
                  if row["selected_area_ratio"] is not None]
        abstained = sum(1 for row in rows_subset if row["abstained"])
        return {
            "records": len(rows_subset),
            "reference_miou": float(np.mean([row["selected_iou"] for row in rows_subset])),
            "reference_dice": float(np.mean([row["selected_dice"] for row in rows_subset])),
            "precision_at_0_5": float(np.mean([row["selected_iou"] >= 0.5 for row in rows_subset])),
            "median_centroid_error": float(np.median(errors)) if errors else None,
            "p90_centroid_error": float(np.percentile(errors, 90)) if errors else None,
            "median_area_ratio": float(np.median(ratios)) if ratios else None,
            "abstentions": abstained,
            "abstention_rate": abstained / len(rows_subset),
        }

    overall_coverage = {
        "all_proposals": coverage(rows, "best_iou_all"),
        "eligible_proposals": coverage(rows, "best_iou_eligible"),
    }
    per_family_coverage = {
        family: {
            "records": sum(1 for row in rows if row["reference_family"] == family),
            "all_proposals": coverage([row for row in rows if row["reference_family"] == family],
                                      "best_iou_all"),
            "eligible_proposals": coverage([row for row in rows if row["reference_family"] == family],
                                           "best_iou_eligible"),
        }
        for family in FAMILIES
    }
    quality_payload = {
        "overall": quality(rows),
        "per_family": {family: quality([row for row in rows if row["reference_family"] == family])
                       for family in FAMILIES},
    }
    categories = Counter(row["failure_category"] for row in rows)
    categories_by_family = {
        family: dict(Counter(row["failure_category"] for row in rows
                             if row["reference_family"] == family))
        for family in FAMILIES
    }

    coverage_gate = {
        "eligible_coverage_at_0_50_min": 0.70,
        "largest_eligible_coverage_at_0_50_min": 0.75,
        "smallest_eligible_coverage_at_0_50_min": 0.60,
        "measured_overall": overall_coverage["eligible_proposals"]["coverage@0.50"],
        "measured_largest": per_family_coverage["largest"]["eligible_proposals"]["coverage@0.50"],
        "measured_smallest": per_family_coverage["smallest"]["eligible_proposals"]["coverage@0.50"],
    }
    coverage_gate["passed"] = bool(
        coverage_gate["measured_overall"] >= coverage_gate["eligible_coverage_at_0_50_min"]
        and coverage_gate["measured_largest"] >= coverage_gate["largest_eligible_coverage_at_0_50_min"]
        and coverage_gate["measured_smallest"] >= coverage_gate["smallest_eligible_coverage_at_0_50_min"]
    )

    resolver_gate = {
        "miou_min": 0.35, "median_centroid_max": 0.05, "p90_centroid_max": 0.12,
        "abstention_rate_max": 0.10,
        "measured_miou": quality_payload["overall"]["reference_miou"],
        "measured_median_centroid": quality_payload["overall"]["median_centroid_error"],
        "measured_p90_centroid": quality_payload["overall"]["p90_centroid_error"],
        "measured_abstention_rate": quality_payload["overall"]["abstention_rate"],
    }
    resolver_gate["passed"] = bool(
        resolver_gate["measured_miou"] >= resolver_gate["miou_min"]
        and (resolver_gate["measured_median_centroid"] or 1.0) <= resolver_gate["median_centroid_max"]
        and (resolver_gate["measured_p90_centroid"] or 1.0) <= resolver_gate["p90_centroid_max"]
        and resolver_gate["measured_abstention_rate"] <= resolver_gate["abstention_rate_max"]
    )

    payload = {
        "_doc": (
            "Task 6Q Part D. Reference-level audit of the deterministic frozen-proposal resolver on "
            "the exact frozen Task 6P RefValUnique pack: proposal coverage (all vs eligible) and "
            "selected-reference quality, with ground truth used for evaluation only. No training, no "
            "threshold tuning, no test split."
        ),
        "task": "6Q",
        "stage": "D-reference-audit",
        "reference_source": "predicted_proposal_reference",
        "proposal_config": config_report(),
        "checkpoint": {"path": str(checkpoint), "expected_sha256": PROPOSAL_CHECKPOINT_SHA256,
                       "actual_sha256": actual_sha, "matches": True},
        "pack": {"name": "RefValUnique", "count": len(records),
                 "path": str(REF_ROOT / "ref_val_unique.json"),
                 "sha256": sha256_file(REF_ROOT / "ref_val_unique.json")},
        "tiles": len(tiles),
        "coverage": {"overall": overall_coverage, "per_family": per_family_coverage},
        "selected_reference_quality": quality_payload,
        "coverage_gate": coverage_gate,
        "resolver_gate": resolver_gate,
        "failure_categories": dict(categories),
        "failure_categories_by_family": categories_by_family,
        "records": rows,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_VAL, payload)

    failure_payload = {
        "_doc": (
            "Task 6Q section 8. Exactly one failure category per RefValUnique record in the fixed "
            "priority order: NO_PROPOSALS, NO_ELIGIBLE_PROPOSALS, REFERENCE_NOT_COVERED_IOU50, "
            "EXTREME_SELECTION_WRONG, SELECTED_MASK_GEOMETRY_POOR, REFERENCE_OK."
        ),
        "task": "6Q",
        "stage": "D-failure-attribution",
        "pack": {"name": "RefValUnique", "count": len(records)},
        "priority_order": ["NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "REFERENCE_NOT_COVERED_IOU50",
                           "EXTREME_SELECTION_WRONG", "SELECTED_MASK_GEOMETRY_POOR", "REFERENCE_OK"],
        "counts": {category: int(categories.get(category, 0))
                   for category in ("NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS",
                                    "REFERENCE_NOT_COVERED_IOU50", "EXTREME_SELECTION_WRONG",
                                    "SELECTED_MASK_GEOMETRY_POOR", "REFERENCE_OK")},
        "counts_by_family": categories_by_family,
        "records": [{"sample_id": row["sample_id"], "tile_id": row["tile_id"],
                     "reference_family": row["reference_family"],
                     "failure_category": row["failure_category"],
                     "best_iou_eligible": row["best_iou_eligible"],
                     "selected_iou": row["selected_iou"]} for row in rows],
        "test_split_used": False,
    }
    write_json(OUT_FAILURE, failure_payload)

    # ---------------- resolver cache for the downstream stage
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    keys = sorted(selected_cache)
    matrix = (np.stack([selected_cache[key] for key in keys]) if keys
              else np.zeros((0, TILE_SIZE, TILE_SIZE), dtype=bool))
    np.savez_compressed(
        CACHE_PATH,
        keys=np.asarray(keys, dtype=object) if keys else np.asarray([], dtype=object),
        masks=np.packbits(matrix.reshape(len(keys), -1), axis=1) if keys
        else np.zeros((0, TILE_SIZE * TILE_SIZE // 8), dtype=np.uint8),
        checkpoint_sha256=np.asarray(PROPOSAL_CHECKPOINT_SHA256),
        conf=np.asarray(PROPOSAL_CONF),
        imgsz=np.asarray(PROPOSAL_IMGSZ),
        max_det=np.asarray(PROPOSAL_MAX_DET),
    )
    manifest = {
        "_doc": (
            "Task 6Q Part I (optional). Manifest of the gitignored frozen-proposal resolver cache: the "
            "deterministically selected reference mask per (tile, family) plus the per-tile proposal "
            "counts, so the downstream stage never runs the proposal model twice."
        ),
        "task": "6Q",
        "cache_path": str(CACHE_PATH),
        "cache_sha256": sha256_file(CACHE_PATH),
        "cache_entries": len(keys),
        "tiles": len(cache_meta["tile_ids"]),
        "proposal_counts": {
            "min": int(min(cache_meta["proposal_counts"])) if cache_meta["proposal_counts"] else 0,
            "max": int(max(cache_meta["proposal_counts"])) if cache_meta["proposal_counts"] else 0,
            "mean": float(np.mean(cache_meta["proposal_counts"]))
            if cache_meta["proposal_counts"] else 0.0,
            "zero_proposal_tiles": int(sum(1 for count in cache_meta["proposal_counts"] if count == 0)),
        },
        "abstained_family_tiles": int(sum(cache_meta["family_abstained"])),
        "proposal_config": config_report(),
        "checkpoint_sha256": PROPOSAL_CHECKPOINT_SHA256,
        "test_split_used": False,
    }
    write_json(OUT_MANIFEST, manifest)

    print(
        f"[6q.audit] tiles {len(tiles)} | eligible coverage@0.50 overall "
        f"{coverage_gate['measured_overall']:.4f} largest {coverage_gate['measured_largest']:.4f} "
        f"smallest {coverage_gate['measured_smallest']:.4f} -> gate {coverage_gate['passed']}",
        flush=True,
    )
    print(
        f"[6q.audit] selected reference mIoU {resolver_gate['measured_miou']:.6f} | centroid median "
        f"{resolver_gate['measured_median_centroid']:.6f} p90 "
        f"{resolver_gate['measured_p90_centroid']:.6f} | abstention "
        f"{resolver_gate['measured_abstention_rate']:.6f} -> gate {resolver_gate['passed']}",
        flush=True,
    )
    print(f"[6q.audit] failure categories {dict(categories)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
