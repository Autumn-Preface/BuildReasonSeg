"""Shared Task 6U machinery: the four frozen proposal configurations, the proposal cache, canonical GT
reference access and the coverage/metric helpers used by calibration, ranker training, reference
evaluation and the downstream causal comparison.

Frozen rules reused unchanged from Task 6Q: `build_proposal`, `is_eligible` (family eligibility), the
canonical Task 6M mask binarization (`normalize_mask`) and the 512x512 source coordinates.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from buildreasonseg_mvp.task6m_eval import canonical_instances  # noqa: E402
from buildreasonseg_mvp.task6q_reference_resolver import (  # noqa: E402
    PROPOSAL_CHECKPOINT_SHA256,
    build_proposal,
    is_eligible,
    proposals_from_results,
)

EVAL = REPO_ROOT / "evaluation"
TILE = 512
PROPOSAL_ROOT = REPO_ROOT / "artifacts" / "task6u" / "proposals"
PROPOSAL_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m1" / "runs" \
    / "m1_yolo26m_seg_continued" / "weights" / "best.pt"

#: Task 6U section 7 — exactly four frozen proposal inference configurations (no others may be added).
CONFIGS: dict[str, dict] = {
    "U-C0": {"id": "U-C0", "label": "frozen baseline", "imgsz": 640, "conf": 0.10, "max_det": 100,
             "nms": "default", "tta": False, "tiling": False},
    "U-C1": {"id": "U-C1", "label": "lower confidence / larger candidate cap", "imgsz": 640,
             "conf": 0.05, "max_det": 300, "nms": "default", "tta": False, "tiling": False},
    "U-C2": {"id": "U-C2", "label": "higher network input resolution", "imgsz": 1024, "conf": 0.10,
             "max_det": 300, "nms": "default", "tta": False, "tiling": False},
    "U-C3": {"id": "U-C3", "label": "higher resolution + recall-oriented threshold", "imgsz": 1024,
             "conf": 0.05, "max_det": 300, "nms": "default", "tta": False, "tiling": False},
}
CONFIG_ORDER = ("U-C0", "U-C1", "U-C2", "U-C3")
COVERAGE_THRESHOLDS = (0.25, 0.50, 0.75)
FAMILIES = ("largest", "smallest")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- canonical GT references

_INSTANCE_CACHE: dict[str, dict[int, np.ndarray]] = {}


def gt_masks(tile_id: str) -> dict[int, np.ndarray]:
    """source_feature_id -> canonical GT mask for one tile (evaluation only)."""

    if tile_id not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[tile_id] = {
            int(instance.source_feature_id): np.asarray(instance.mask, dtype=bool)
            for instance in canonical_instances(tile_id)
        }
    return _INSTANCE_CACHE[tile_id]


def gt_reference_mask(record: dict) -> np.ndarray:
    masks = gt_masks(str(record["tile_id"]))
    return masks[int(record["reference_source_feature_id"])]


# --------------------------------------------------------------------------- proposal cache


def proposals_path(config_id: str, tile_id: str) -> Path:
    return PROPOSAL_ROOT / config_id / f"{tile_id}.npz"


def save_proposals(config_id: str, tile_id: str, proposals) -> None:
    path = proposals_path(config_id, tile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "packed": np.stack([np.packbits(proposal.mask.reshape(-1)) for proposal in proposals])
        if proposals else np.zeros((0, TILE * TILE // 8), dtype=np.uint8),
        "confidences": np.asarray([proposal.confidence for proposal in proposals], dtype=np.float32),
        "indices": np.asarray([proposal.index for proposal in proposals], dtype=np.int64),
        "count": np.asarray([len(proposals)], dtype=np.int64),
    }
    np.savez_compressed(path, **payload)


def load_proposals(config_id: str, tile_id: str):
    path = proposals_path(config_id, tile_id)
    if not path.is_file():
        return None
    payload = np.load(path, allow_pickle=False)
    count = int(payload["count"][0])
    proposals = []
    for index in range(count):
        mask = np.unpackbits(payload["packed"][index])[: TILE * TILE].reshape(TILE, TILE).astype(bool)
        proposals.append(build_proposal(int(payload["indices"][index]),
                                        float(payload["confidences"][index]), mask))
    return proposals


def run_config_on_tile(model, config: dict, image_path: Path, device: str = "0"):
    """One frozen inference run for one tile under one declared configuration."""

    started = time.perf_counter()
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    results = model.predict(
        source=str(image_path), imgsz=int(config["imgsz"]), conf=float(config["conf"]),
        max_det=int(config["max_det"]), verbose=False, device=device, retina_masks=True,
    )[0]
    seconds = time.perf_counter() - started
    peak = (round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if device != "cpu" else None)
    # masks are restored to exact source 512x512 coordinates by the canonical Task 6M binarization
    return proposals_from_results(results), seconds, peak


def proposals_for_tile(model, config: dict, tile_id: str, image_path: Path, device: str = "0",
                       use_cache: bool = True):
    """Cached proposals for (config, tile). The cache is gitignored under artifacts/task6u/."""

    if use_cache:
        cached = load_proposals(config["id"], tile_id)
        if cached is not None:
            return cached, {"cached": True, "seconds": None, "peak_vram_gb": None}
    proposals, seconds, peak = run_config_on_tile(model, config, image_path, device)
    if use_cache:
        save_proposals(config["id"], tile_id, proposals)
    return proposals, {"cached": False, "seconds": round(seconds, 4), "peak_vram_gb": peak}


# --------------------------------------------------------------------------- metrics


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
    return (float((mask.sum(axis=0) * x).sum() / total),
            float((mask.sum(axis=1) * y).sum() / total))


def centroid_error(predicted: np.ndarray, truth: np.ndarray) -> float:
    px, py = centroid(predicted)
    tx, ty = centroid(truth)
    return float(np.hypot(px - tx, py - ty)) / float(np.sqrt(2.0))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


# --------------------------------------------------------------------------- pack/tile iteration


def group_records_by_tile(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["tile_id"])].append(record)
    return dict(sorted(grouped.items()))


def record_image_path(record: dict) -> Path:
    path = Path(str(record["image_path"]))
    return path if path.is_absolute() else REPO_ROOT / path


def coverage_for_records(records: list[dict], proposals_by_key: dict[tuple, list]) -> dict:
    """All-proposal and eligible coverage at 0.25/0.50/0.75, per family and overall."""

    rows = []
    for record in records:
        family = str(record["reference_family"])
        key = (str(record["tile_id"]), int(record["reference_source_feature_id"]), family)
        proposals = proposals_by_key.get(key, [])
        truth = gt_reference_mask(record)
        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        all_ious = [iou(proposal.mask, truth) for proposal in proposals]
        eligible_ious = [iou(proposal.mask, truth) for proposal in eligible]
        rows.append({
            "tile_id": str(record["tile_id"]), "reference_family": family,
            "proposal_count": len(proposals), "eligible_count": len(eligible),
            "best_all_iou": max(all_ious) if all_ious else 0.0,
            "best_eligible_iou": max(eligible_ious) if eligible_ious else 0.0,
        })

    def summarise(subset: list[dict]) -> dict:
        if not subset:
            return {"records": 0}
        summary = {"records": len(subset)}
        for threshold in COVERAGE_THRESHOLDS:
            tag = f"{threshold:.2f}"
            summary[f"all_coverage@{tag}"] = float(np.mean(
                [row["best_all_iou"] >= threshold for row in subset]))
            summary[f"eligible_coverage@{tag}"] = float(np.mean(
                [row["best_eligible_iou"] >= threshold for row in subset]))
        summary["no_proposals"] = sum(1 for row in subset if row["proposal_count"] == 0)
        summary["no_eligible"] = sum(1 for row in subset if row["eligible_count"] == 0)
        summary["eligible_per_record_mean"] = float(np.mean(
            [row["eligible_count"] for row in subset]))
        return summary

    return {
        "overall": summarise(rows),
        "largest": summarise([row for row in rows if row["reference_family"] == "largest"]),
        "smallest": summarise([row for row in rows if row["reference_family"] == "smallest"]),
        "records": rows,
    }


def selection_priority_key(report: dict) -> tuple:
    """Section 10 priority for one configuration report (higher tuple sorts first).

    1. highest smallest eligible coverage@0.50; 2. then overall; 3. then largest; 4. then lower mean
    eligible proposals per record; 5. then lower imgsz; 6. then higher conf; 7. then lower config id.
    """

    coverage = report["coverage"]
    return (
        coverage["smallest"]["eligible_coverage@0.50"],
        coverage["overall"]["eligible_coverage@0.50"],
        coverage["largest"]["eligible_coverage@0.50"],
        -report["eligible_proposals_per_record"]["mean"],
        -int(report["imgsz"]),
        float(report["conf"]),
        -int(str(report["id"]).split("-C")[1]),
    )


def config_manifest(config: dict, model_sha: str) -> dict:
    return {
        **config,
        "checkpoint": str(PROPOSAL_CHECKPOINT),
        "checkpoint_sha256": model_sha,
        "expected_checkpoint_sha256": PROPOSAL_CHECKPOINT_SHA256,
        "source_image_size": [TILE, TILE],
        "masks_restored_to_source": True,
        "super_resolution": False,
        "proposal_cache": str(PROPOSAL_ROOT / config["id"]),
    }


def write_proposal_cache_manifest(config_ids: tuple[str, ...], entries: dict) -> None:
    path = PROPOSAL_ROOT / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_doc": "Task 6U proposal cache manifest (gitignored artifacts/task6u/proposals/).",
               "task": "6U", "configs": list(config_ids), "tiles": entries}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def attribute_failure(*, proposal_count: int, eligible_count: int, best_eligible_iou: float,
                      selected_iou: float | None, selected_centroid_error: float | None) -> str:
    """Task 6Q / Task 6S reference failure buckets, in the frozen priority order."""

    if proposal_count == 0:
        return "NO_PROPOSALS"
    if eligible_count == 0:
        return "NO_ELIGIBLE_PROPOSALS"
    if best_eligible_iou < 0.50:
        return "REFERENCE_NOT_COVERED_IOU50"
    if selected_iou is None or selected_iou < 0.50:
        return "REFERENCE_SELECTION_WRONG"
    if selected_centroid_error is None or selected_centroid_error > 0.05:
        return "SELECTED_MASK_GEOMETRY_POOR"
    return "REFERENCE_OK"


BUCKET_ORDER = ("NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS", "REFERENCE_NOT_COVERED_IOU50",
                "REFERENCE_SELECTION_WRONG", "SELECTED_MASK_GEOMETRY_POOR", "REFERENCE_OK")


def bucket_counts(rows: list[dict]) -> dict:
    counts = Counter(row["bucket"] for row in rows)
    return {bucket: counts.get(bucket, 0) for bucket in BUCKET_ORDER}
