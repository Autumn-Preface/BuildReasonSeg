"""Task 6K.1 shared helpers: read-only vector access, tile geometry, instance views, metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "datasets" / "transforms"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.shapefile_reader import iter_shp_features  # noqa: E402
from buildreasonseg_mvp.whu_vector_audit import (  # noqa: E402
    CROPPED_ROOT,
    REGION_TO_RASTER,
    SHP_PATH,
    TILE_SIZE,
    WHOLE_AREA_DIR,
    VectorFeature,
    feature_intersects_bbox,
    rasterize_feature,
    read_tiff_metadata,
    tile_grid_for,
    vector_feature_from_shp,
    world_file_for_raster,
)

EVAL = REPO_ROOT / "evaluation"
CACHE_DIR = REPO_ROOT / "artifacts" / "task6k1"
INSTANCE_DIR = CACHE_DIR / "vector_instances"
SAMPLES_DIR = CACHE_DIR / "samples"

#: Task 6K's aligned tiles (current representation) are reused unchanged.
from task6k_common import aligned_tiles  # noqa: E402


def whole_raster_paths() -> dict:
    return {
        "image": {name: WHOLE_AREA_DIR / "image" / f"{name}.tif" for name in REGION_TO_RASTER.values()},
        "label": {name: WHOLE_AREA_DIR / "label" / f"{name}.tif" for name in REGION_TO_RASTER.values()},
        "world_file": {name: WHOLE_AREA_DIR / "label" / f"{name}.tfw" for name in REGION_TO_RASTER.values()},
    }


_metadata_cache: dict[str, dict] = {}
_affine_cache: dict[str, object] = {}
_grid_cache: dict[str, object] = {}


def raster_metadata(name: str) -> dict:
    if name not in _metadata_cache:
        _metadata_cache[name] = read_tiff_metadata(WHOLE_AREA_DIR / "image" / f"{name}.tif")
    return _metadata_cache[name]


def label_metadata(name: str) -> dict:
    key = f"label:{name}"
    if key not in _metadata_cache:
        _metadata_cache[key] = read_tiff_metadata(WHOLE_AREA_DIR / "label" / f"{name}.tif")
    return _metadata_cache[key]


def raster_affine(name: str):
    if name not in _affine_cache:
        _affine_cache[name] = world_file_for_raster(WHOLE_AREA_DIR / "label" / f"{name}.tif")
    return _affine_cache[name]


def raster_grid(name: str):
    if name not in _grid_cache:
        _grid_cache[name] = tile_grid_for(name, raster_metadata(name))
    return _grid_cache[name]


def region_of_stem(stem: str) -> str:
    return stem.split("_", 1)[0] if "_" in stem else "test"


def tile_index_of_stem(stem: str) -> int | None:
    if "_" not in stem:
        try:
            return int(stem)
        except ValueError:
            return None
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return None


@dataclass(frozen=True)
class TileGeometry:
    """Where one cropped tile sits inside its whole-area raster."""

    stem: str
    raster: str
    index: int
    row: int
    col: int
    x0: int
    y0: int
    size: int
    affine: object

    @property
    def map_bbox(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the tile in map units (corner-based)."""

        x_left, y_top = self.affine.pixel_to_map(self.x0, self.y0)
        x_right, y_bottom = self.affine.pixel_to_map(self.x0 + self.size, self.y0 + self.size)
        return (min(x_left, x_right), min(y_top, y_bottom), max(x_left, x_right), max(y_top, y_bottom))

    def as_dict(self) -> dict:
        return {
            "stem": self.stem,
            "raster": self.raster,
            "index": self.index,
            "row": self.row,
            "col": self.col,
            "window": [self.x0, self.y0, self.size, self.size],
            "map_bbox": list(self.map_bbox),
        }


def tile_geometry(stem: str) -> TileGeometry | None:
    region = region_of_stem(stem)
    raster = REGION_TO_RASTER.get(region)
    index = tile_index_of_stem(stem)
    if raster is None or index is None:
        return None
    grid = raster_grid(raster)
    if not grid.contains(index):
        return None
    row, col = grid.index_to_rowcol(index)
    x0, y0, size, _height = grid.window(index)
    return TileGeometry(
        stem=stem, raster=raster, index=index, row=row, col=col,
        x0=x0, y0=y0, size=size, affine=raster_affine(raster),
    )


# ------------------------------------------------------------------ vector corpus


@dataclass
class VectorCorpus:
    features: list[VectorFeature]
    bboxes: np.ndarray

    def candidates(self, bbox: tuple[float, float, float, float]) -> list[int]:
        x0, y0, x1, y1 = bbox
        keep = (
            (self.bboxes[:, 2] >= x0) & (self.bboxes[:, 0] <= x1)
            & (self.bboxes[:, 3] >= y0) & (self.bboxes[:, 1] <= y1)
        )
        return np.nonzero(keep)[0].tolist()

    def __len__(self) -> int:
        return len(self.features)


def load_vector_corpus() -> VectorCorpus:
    features = []
    for index, feature in enumerate(iter_shp_features(SHP_PATH), start=1):
        features.append(vector_feature_from_shp(feature, index))
    bboxes = np.asarray([f.bbox_map for f in features], dtype=np.float64)
    return VectorCorpus(features=features, bboxes=bboxes)


# ------------------------------------------------------------------ clipping / instances


@dataclass
class TileInstances:
    """The vector instances clipped to one tile window."""

    stem: str
    geometry: TileGeometry
    ids: list[int]                 # global stable feature ids
    masks: list[np.ndarray]        # 512x512 boolean masks
    areas_px: list[int]
    bboxes_px: list[tuple[int, int, int, int]]
    centroids_px: list[tuple[float, float]]
    touches_border: list[bool]
    multipart: list[bool]
    clipped: list[bool]            # feature extends beyond the tile window

    def __len__(self) -> int:
        return len(self.ids)

    def label_map(self) -> np.ndarray:
        out = np.zeros((self.geometry.size, self.geometry.size), dtype=np.int32)
        for local_id, mask in enumerate(self.masks, start=1):
            out[mask] = local_id
        return out

    def as_candidate_set(self, source: str = "vector"):
        from buildreasonseg_mvp.structured_grounding import Candidate, CandidateSet

        height = width = self.geometry.size
        candidates = []
        for local_id, (global_id, mask, area, bbox, centroid, border) in enumerate(
            zip(self.ids, self.masks, self.areas_px, self.bboxes_px, self.centroids_px, self.touches_border),
            start=1,
        ):
            candidates.append(
                Candidate(
                    candidate_id=local_id,
                    mask=mask,
                    bbox_xyxy_px=tuple(int(v) for v in bbox),
                    centroid_px=(float(centroid[0]), float(centroid[1])),
                    area_px=int(area),
                    touches_image_border=bool(border),
                    confidence=None,
                    source=source,
                )
            )
        label_map = self.label_map().astype(np.int32)
        return CandidateSet(width=width, height=height, candidates=candidates, label_map=label_map, source=source)


def clip_vector_to_tile(corpus: VectorCorpus, geometry: TileGeometry) -> TileInstances:
    """Rasterize every intersecting feature into the tile window (holes subtracted)."""

    bbox = geometry.map_bbox
    ids, masks, areas, boxes, centroids, borders, multiparts, clipped = [], [], [], [], [], [], [], []
    for index in corpus.candidates(bbox):
        feature = corpus.features[index]
        mask = rasterize_feature(
            feature, geometry.affine, (geometry.size, geometry.size),
            origin_col=geometry.x0, origin_row=geometry.y0,
        )
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        touches = bool(x0 == 0 or y0 == 0 or x1 == geometry.size - 1 or y1 == geometry.size - 1)
        inside = (
            feature.bbox_map[0] >= bbox[0] and feature.bbox_map[1] >= bbox[1]
            and feature.bbox_map[2] <= bbox[2] and feature.bbox_map[3] <= bbox[3]
        )
        ids.append(feature.feature_id)
        masks.append(mask)
        areas.append(int(mask.sum()))
        boxes.append((x0, y0, x1 + 1, y1 + 1))
        centroids.append((float(xs.mean()), float(ys.mean())))
        borders.append(touches)
        multiparts.append(feature.n_parts > 1)
        clipped.append(not inside)
    return TileInstances(
        stem=geometry.stem, geometry=geometry, ids=ids, masks=masks, areas_px=areas,
        bboxes_px=boxes, centroids_px=centroids, touches_border=borders,
        multipart=multiparts, clipped=clipped,
    )


def save_tile_instances(instances: TileInstances) -> Path:
    """Persist the canonical instance view for one tile (gitignored cache)."""

    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    path = INSTANCE_DIR / f"{instances.stem}.npz"
    np.savez_compressed(
        path,
        ids=np.asarray(instances.ids, dtype=np.int64),
        label_map=instances.label_map().astype(np.int32),
        areas_px=np.asarray(instances.areas_px, dtype=np.int64),
        bboxes_px=np.asarray(instances.bboxes_px, dtype=np.int64).reshape(-1, 4),
        centroids_px=np.asarray(instances.centroids_px, dtype=np.float64).reshape(-1, 2),
        touches_border=np.asarray(instances.touches_border, dtype=bool),
        multipart=np.asarray(instances.multipart, dtype=bool),
        clipped=np.asarray(instances.clipped, dtype=bool),
        map_bbox=np.asarray(instances.geometry.map_bbox, dtype=np.float64),
        window=np.asarray([instances.geometry.x0, instances.geometry.y0,
                           instances.geometry.size, instances.geometry.size], dtype=np.int64),
    )
    return path


# ------------------------------------------------------------------ metrics


def iou_masks(left, right, *, both_empty_value: float = 1.0) -> float:
    """IoU with an explicit convention for the both-empty case (a match, not a 0)."""

    a = np.asarray(left).astype(bool)
    b = np.asarray(right).astype(bool)
    a_empty = not a.any()
    b_empty = not b.any()
    if a_empty and b_empty:
        return float(both_empty_value)
    if a_empty or b_empty:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float(both_empty_value)


def dice_masks(left, right) -> float:
    a = np.asarray(left).astype(bool)
    b = np.asarray(right).astype(bool)
    denominator = int(a.sum()) + int(b.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denominator)


def precision_recall(prediction, target) -> tuple[float, float]:
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return float(precision), float(recall)


def match_instances(left_masks, right_masks, threshold: float) -> dict:
    """Greedy best-match between two instance lists at an IoU threshold."""

    remaining = list(range(len(right_masks)))
    pairs = []
    unmatched_left = []
    for index, mask in enumerate(left_masks):
        best = None
        for candidate in remaining:
            value = iou_masks(mask, right_masks[candidate])
            if best is None or value > best[0]:
                best = (value, candidate)
        if best is not None and best[0] >= threshold:
            pairs.append({"left": index, "right": best[1], "iou": best[0]})
            remaining.remove(best[1])
        else:
            unmatched_left.append(index)
    return {"pairs": pairs, "unmatched_left": unmatched_left, "unmatched_right": remaining}


def percentile_summary(values) -> dict:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p5": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def masks_from_label_map(label_map: np.ndarray) -> list[np.ndarray]:
    """Boolean mask per non-zero instance id in a label map (order = ascending id)."""

    label_map = np.asarray(label_map)
    return [label_map == value for value in np.unique(label_map) if int(value) != 0]


def load_instance_view(stem: str) -> dict | None:
    """Load one canonical vector instance view from the gitignored cache."""

    path = INSTANCE_DIR / f"{stem}.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return {
            "ids": data["ids"].astype(np.int64),
            "label_map": data["label_map"].astype(np.int32),
            "areas_px": data["areas_px"].astype(np.int64),
            "bboxes_px": data["bboxes_px"].astype(np.int64),
            "centroids_px": data["centroids_px"].astype(np.float64),
            "touches_border": data["touches_border"].astype(bool),
            "multipart": data["multipart"].astype(bool),
            "clipped": data["clipped"].astype(bool),
        }


def candidate_set_from_instance_view(view: dict, source: str = "vector"):
    """Build a Task 6J `CandidateSet` from a stored vector instance view."""

    from buildreasonseg_mvp.structured_grounding import Candidate, CandidateSet

    label_map = np.asarray(view["label_map"], dtype=np.int32)
    height, width = label_map.shape
    masks = {int(value): label_map == value for value in np.unique(label_map) if int(value) != 0}
    candidates = []
    for local_id, global_id in enumerate(view["ids"].tolist(), start=1):
        mask = masks.get(local_id)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        candidates.append(
            Candidate(
                candidate_id=int(local_id),
                mask=mask,
                bbox_xyxy_px=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                centroid_px=(float(xs.mean()), float(ys.mean())),
                area_px=int(mask.sum()),
                touches_image_border=bool(view["touches_border"][local_id - 1]),
                confidence=None,
                source=source,
            )
        )
    return CandidateSet(width=width, height=height, candidates=candidates, label_map=label_map, source=source)


def match_components_by_overlap(labels_left: np.ndarray, labels_right: np.ndarray,
                                areas_right: dict[int, int], threshold: float = 0.5) -> dict:
    """Mask-overlap matching left -> right (IoU >= threshold), computed per left component."""

    matches = {}
    for left_id in [int(v) for v in np.unique(labels_left) if int(v) != 0]:
        left_mask = labels_left == left_id
        left_area = int(left_mask.sum())
        if left_area == 0:
            continue
        overlap_ids, counts = np.unique(labels_right[left_mask], return_counts=True)
        best = None
        for right_id, intersection in zip(overlap_ids.tolist(), counts.tolist()):
            if int(right_id) == 0:
                continue
            union = left_area + int(areas_right.get(int(right_id), 0)) - int(intersection)
            value = float(intersection) / max(union, 1)
            if best is None or value > best["iou"]:
                best = {
                    "right_id": int(right_id),
                    "iou": value,
                    "area_left": left_area,
                    "area_right": int(areas_right.get(int(right_id), 0)),
                    "containment_of_left": float(intersection) / max(left_area, 1),
                }
        if best is not None and best["iou"] >= float(threshold):
            matches[left_id] = best
    return matches


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
