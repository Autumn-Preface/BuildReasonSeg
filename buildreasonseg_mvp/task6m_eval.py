"""Task 6M shared evaluation helpers (native-vector GT, YOLO label codec, matching metrics).

Ground truth is ALWAYS the canonical native masks from `WHU-EA-NativeVector` v1.0 — never the derived
Ultralytics TXT export. The TXT codec exists only to (a) produce the derived training export and
(b) audit that export's fidelity.

Design rules enforced here:
* `Proposal` objects carry predicted geometry only; GT never repairs or filters a proposal.
* matching is diagnostic (it scores predictions against GT), never generative.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset  # noqa: E402
from buildreasonseg_mvp.whu_native_vector import (  # noqa: E402
    REASONING_VIEW_ROOT,
    instance_rings,
    read_tile_cache,
)

EVAL = REPO_ROOT / "evaluation"
EXPORT_ROOT = REPO_ROOT / "artifacts" / "task6m_yolo_native"
PREDICTION_CACHE = REPO_ROOT / "artifacts" / "task6m_predictions"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task6m"
TILE_SIZE = 512
SPLITS = ("train", "val", "test")
SCENE_SPLIT_OF_RASTER = {"train1": "train", "train2": "val", "test": "test"}


# ------------------------------------------------------------------ canonical GT


@dataclass
class GtInstance:
    """One canonical native instance used as ground truth."""

    tile_instance_id: int
    source_feature_id: int
    mask: np.ndarray
    area_px: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid: tuple[float, float]
    touches_border: bool
    tiny: bool
    n_holes: int
    visible_fraction: float


def canonical_instances(tile_id: str) -> list[GtInstance]:
    """GT instances of one tile, straight from the canonical store (label map + scalars)."""

    cache = read_tile_cache(tile_id)
    if cache is None:
        return []
    label_map = cache["label_map"]
    out: list[GtInstance] = []
    for position, instance_id in enumerate(cache["instance_ids"].tolist()):
        mask = label_map == int(instance_id)
        area = int(mask.sum())
        if area == 0:
            continue
        bbox = [int(v) for v in cache["bboxes"][position]]
        out.append(
            GtInstance(
                tile_instance_id=int(instance_id),
                source_feature_id=int(cache["source_feature_ids"][position]),
                mask=mask,
                area_px=area,
                bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
                centroid=(float(cache["centroids"][position][0]), float(cache["centroids"][position][1])),
                touches_border=bool(cache["touches_border"][position]),
                tiny=bool(cache["tiny"][position]),
                n_holes=int(cache["n_holes"][position]),
                visible_fraction=float(cache["visible_fraction"][position]),
            )
        )
    return out


def tile_ids(split: str) -> list[str]:
    return [str(v) for v in json.loads(
        (REPO_ROOT / "datasets" / "whu_native_vector" / "v1.0" / "splits" / "scene_disjoint_v1.json")
        .read_text(encoding="utf-8")
    )[split]]


def source_image_path(dataset: NativeVectorDataset, source_root: Path, tile_id: str) -> Path:
    view = dataset.load_tile(tile_id)
    assert view is not None, tile_id
    return Path(source_root) / str(view.record["source_image_ref"])


# ------------------------------------------------------------------ YOLO polygon codec


def repair_box_normalized(x0: int, y0: int, x1: int, y1: int, size: int = TILE_SIZE) -> np.ndarray:
    """Smallest box (normalized) whose Ultralytics decode is non-degenerate.

    Ultralytics decodes polygons as ``clip(round(polygon * size), 0, size - 1)``, so a polygon that
    reaches the last row/column loses that edge. For a 1-px sliver on the bottom or right tile edge
    the whole polygon then collapses to fewer than 3 distinct pixels. This grows the box INWARD
    (away from the tile edge) until the decoded polygon has at least 3 distinct pixels.
    """

    for gx in range(0, 3):
        for gy in range(0, 3):
            ax0 = max(0.0, float(x0 - gx))
            ay0 = max(0.0, float(y0 - gy))
            ax1 = min(float(size), float(x1 + gx))
            ay1 = min(float(size), float(y1 + gy))
            if ax1 - ax0 < 1.0:
                ax1 = min(float(size), ax0 + 1.0)
            if ay1 - ay0 < 1.0:
                ay1 = min(float(size), ay0 + 1.0)
            ex1 = ax1 if ax1 < size else size - 0.5
            ey1 = ay1 if ay1 < size else size - 0.5
            box = np.asarray([[ax0, ay0], [ex1, ay0], [ex1, ey1], [ax0, ey1]], dtype=np.float64) / float(size)
            pixels = np.clip(np.round(box * size), 0, size - 1)
            if len(np.unique(pixels, axis=0)) >= 3:
                return box
    # last resort: a 2x2 interior box anchored away from the edges
    bx0 = float(min(max(x0, 0), size - 2))
    by0 = float(min(max(y0, 0), size - 2))
    return np.asarray(
        [[bx0, by0], [bx0 + 1.5, by0], [bx0 + 1.5, by0 + 1.5], [bx0, by0 + 1.5]], dtype=np.float64
    ) / float(size)


def exterior_polygons(tile_id: str, instance_ids: list[int] | None = None) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    """Exterior-only polygons (normalized) for one tile, plus the hole instances affected.

    Ultralytics polygon TXT cannot express holes, so this returns the required exterior form and
    records which instances lost a hole. Multipart instances keep their largest outer ring.

    Sub-pixel instances: a polygon whose rounded pixel form collapses (fewer than 3 distinct pixels)
    would vanish from the label file entirely, which the export gate forbids ("no non-hole instance
    silently missing"). For those instances only, the polygon is replaced by the smallest
    axis-aligned box covering the instance bounding box, and the repair is recorded explicitly.

    Returns `(polygons, holes_dropped, repairs)`.
    """

    cache = read_tile_cache(tile_id)
    if cache is None:
        return [], [], []
    wanted = set(instance_ids) if instance_ids is not None else None
    polygons: list[np.ndarray] = []
    holes_dropped: list[dict] = []
    repairs: list[dict] = []
    for position, instance_id in enumerate(cache["instance_ids"].tolist()):
        if wanted is not None and int(instance_id) not in wanted:
            continue
        rings = instance_rings(cache, int(instance_id))
        outer = [np.asarray(r["points"], dtype=np.float64) for r in rings if r["kind"] == "outer"]
        holes = [r for r in rings if r["kind"] == "hole"]
        if not outer:
            continue
        if len(outer) > 1:
            outer = [max(outer, key=lambda ring: abs(_shoelace(ring)))]
        polygon = outer[0]
        if polygon.shape[0] < 3:
            continue
        normalized = np.clip(polygon / float(TILE_SIZE), 0.0, 1.0)
        pixels = np.clip(np.round(normalized * TILE_SIZE), 0, TILE_SIZE - 1)
        if len(np.unique(pixels, axis=0)) < 3:
            bbox = [int(v) for v in cache["bboxes"][position]]
            box = repair_box_normalized(bbox[0], bbox[1], bbox[2], bbox[3])
            repairs.append(
                {
                    "tile_instance_id": int(instance_id),
                    "area_px": int(cache["clipped_area"][position]),
                    "repaired_bbox_px": [bbox[0], bbox[1], bbox[2], bbox[3]],
                }
            )
            normalized = box
        polygons.append(normalized)
        if holes:
            holes_dropped.append(
                {
                    "tile_instance_id": int(instance_id),
                    "holes": len(holes),
                    "hole_area_px": int(sum(abs(_shoelace(np.asarray(r["points"], dtype=np.float64))) for r in holes)),
                }
            )
    return polygons, holes_dropped, repairs


def _shoelace(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def encode_label_file(polygons: list[np.ndarray], class_id: int = 0) -> str:
    lines = []
    for polygon in polygons:
        if polygon.shape[0] < 3:
            continue
        coords = " ".join(f"{value:.6f}" for value in polygon.reshape(-1))
        lines.append(f"{class_id} {coords}")
    return "\n".join(lines) + ("\n" if lines else "")


def decode_label_file(text: str, size: int = TILE_SIZE) -> tuple[list[np.ndarray], int]:
    """Decode YOLO polygon lines exactly the way Ultralytics does (round + clip to the image)."""

    import cv2

    masks: list[np.ndarray] = []
    malformed = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            malformed += 1
            continue
        try:
            values = np.asarray([float(v) for v in parts[1:]], dtype=np.float64)
        except ValueError:
            malformed += 1
            continue
        polygon = values.reshape(-1, 2)
        if not np.isfinite(polygon).all() or polygon.shape[0] < 3:
            malformed += 1
            continue
        pixels = np.clip(np.round(polygon * size), 0, size - 1).astype(np.int32)
        if len(np.unique(pixels, axis=0)) < 3:
            malformed += 1
            continue
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [pixels], 1)
        masks.append(mask.astype(bool))
    return masks, malformed


# ------------------------------------------------------------------ metrics


def normalize_mask(mask, size: int = TILE_SIZE) -> np.ndarray:
    """Boolean mask at the canonical tile resolution.

    Ultralytics returns masks at the inference resolution; `retina_masks=True` already gives the
    original image size, and this is the belt-and-braces path so a shape mismatch can never
    silently corrupt a comparison.
    """

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[0]
    if array.shape != (size, size):
        import cv2

        array = cv2.resize(array.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return array.astype(bool)


def build_label_map(predicted_masks: list[np.ndarray], size: int = TILE_SIZE) -> np.ndarray:
    """uint16 proposal label map (0 = background, k = proposal k+1)."""

    label_map = np.zeros((size, size), dtype=np.uint16)
    for index, mask in enumerate(predicted_masks, start=1):
        label_map[np.asarray(mask).astype(bool)] = index
    return label_map


def fast_best_iou(gt_mask: np.ndarray, label_map: np.ndarray, proposal_areas: np.ndarray) -> tuple[float, int]:
    """Best IoU between one GT mask and every proposal, in O(pixels) instead of O(pixels x proposals).

    Uses `bincount` over the proposal label map restricted to the GT mask, so a whole tile's
    proposals are compared with one pass over the image.
    """

    gt = np.asarray(gt_mask).astype(bool)
    gt_area = int(gt.sum())
    if gt_area == 0 or label_map.size == 0:
        return 0.0, -1
    counts = np.bincount(label_map[gt], minlength=proposal_areas.size + 1)
    intersections = counts[1:]
    if intersections.size == 0:
        return 0.0, -1
    unions = gt_area + proposal_areas - intersections
    ious = np.where(unions > 0, intersections / np.maximum(unions, 1), 0.0)
    best = int(np.argmax(ious))
    return float(ious[best]), best


def tile_recall(predicted_masks: list[np.ndarray], gt_masks: list[np.ndarray],
                thresholds=(0.25, 0.50, 0.75), size: int = TILE_SIZE) -> dict:
    """Per-tile recall at several IoU thresholds using the fast label-map path."""

    label_map = build_label_map(predicted_masks, size=size)
    areas = np.asarray([int(np.asarray(mask).astype(bool).sum()) for mask in predicted_masks], dtype=np.int64)
    hits = {threshold: 0 for threshold in thresholds}
    best_ious = []
    for gt in gt_masks:
        best, _index = fast_best_iou(gt, label_map, areas)
        best_ious.append(best)
        for threshold in thresholds:
            if best >= threshold:
                hits[threshold] += 1
    return {
        "gt_instances": len(gt_masks),
        "hits": hits,
        "best_ious": best_ious,
        "label_map": label_map,
        "proposal_areas": areas,
    }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denominator)


def match_masks(predicted: list[np.ndarray], ground_truth: list[np.ndarray], threshold: float) -> dict:
    """Greedy IoU matching (diagnostic only)."""

    remaining = list(range(len(ground_truth)))
    pairs = []
    for index, mask in enumerate(predicted):
        best = None
        for candidate in remaining:
            value = iou(mask, ground_truth[candidate])
            if best is None or value > best[0]:
                best = (value, candidate)
        if best is not None and best[0] >= threshold:
            pairs.append({"predicted": index, "gt": best[1], "iou": best[0]})
            remaining.remove(best[1])
    matched_gt = {pair["gt"] for pair in pairs}
    return {
        "pairs": pairs,
        "unmatched_predicted": [i for i in range(len(predicted)) if i not in {p["predicted"] for p in pairs}],
        "unmatched_gt": [i for i in remaining if i not in matched_gt] if False else list(remaining),
    }


def percentile_summary(values) -> dict:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p1": float(np.percentile(array, 1)),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None


def reasoning_view_root(split_view: str = "scene_disjoint_v1") -> Path:
    return REASONING_VIEW_ROOT / split_view
