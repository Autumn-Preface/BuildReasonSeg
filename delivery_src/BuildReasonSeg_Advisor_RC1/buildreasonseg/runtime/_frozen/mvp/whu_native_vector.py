"""Task 6L: canonical WHU East-Asia native-vector instance dataset (`WHU-EA-NativeVector` v1.0).

This module builds and reads the canonical dataset that replaces semantic connected components as
the PRIMARY instance truth (Task 6K.1 verdict `MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES`).

Design rules
------------
1. **All 17,388 cropped tiles** are indexed, including `_no` tiles and empty tiles.
2. Native `EA.shp` geometry is preserved: no `<50` deletion, no connected-component merging, no
   `RETR_EXTERNAL` hole loss, no `approxPolyDP` simplification.
3. Identity is `(EA.shp SHA256, source_feature_id)`; a tile-clipped instance additionally carries
   `tile_id` and a deterministic `tile_instance_id`. DBF fields are never identity (Task 6K.1 showed
   they are degenerate).
4. Committed metadata never hard-codes an absolute source path: the external WHU root is supplied at
   runtime (`--source-root`) and only *logical* paths relative to it are stored.
5. Large regenerable per-tile geometry caches live under the gitignored `artifacts/whu_native_vector/`.

Everything is read-only with respect to the source archive, the historical YOLO dataset and the
legacy project.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .shapefile_reader import iter_shp_features
from .whu_vector_audit import (
    CROPPED_ROOT,
    DBF_PATH,
    PRJ_PATH,
    REGION_TO_RASTER,
    SHP_PATH,
    SHX_PATH,
    WHOLE_AREA_DIR,
    VectorFeature,
    WorldFile,
    rasterize_feature,
    read_tiff_metadata,
    tile_grid_for,
    vector_feature_from_shp,
    world_file_for_raster,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "whu_native_vector" / "v1.0"
CACHE_ROOT = REPO_ROOT / "artifacts" / "whu_native_vector"
REASONING_VIEW_ROOT = CACHE_ROOT / "reasoning_view"

DATASET_NAME = "WHU-EA-NativeVector"
DATASET_VERSION = "v1.0"
SCHEMA_VERSION = "1.0"

#: Historical cropped-archive folders and their whole-raster provenance.
LEGACY_CATEGORIES = ("train", "train_no", "test", "test_no")
CATEGORY_RASTER = {
    "train": ("train1", "train2"),
    "train_no": ("train1", "train2"),
    "test": ("test",),
    "test_no": ("test",),
}
EXPECTED_CATEGORY_COUNTS = {"train": 3135, "train_no": 10527, "test": 903, "test_no": 2823}
EXPECTED_TOTAL_TILES = 17388

TILE_SIZE = 512
MAX_TILE_INSTANCES = 255  # the frozen reasoning stack reads label maps as uint8

#: Scene-disjoint split view: whole-raster provenance.
SCENE_DISJOINT_VIEW = "scene_disjoint_v1"
LEGACY_COMPAT_VIEW = "legacy_compat_v1"
SCENE_DISJOINT_SPLITS = {"train1": "train", "train2": "val", "test": "test"}

#: Repository-relative paths used inside generated metadata (never absolute).
CACHE_REL = "artifacts/whu_native_vector"


# ------------------------------------------------------------------ provenance


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def shapefile_provenance() -> dict:
    """Identity of the exact archive this dataset indexes."""

    return {
        "shp_path_logical": "2. The shape file of the whole images/EA.shp",
        "shp_sha256": sha256_file(SHP_PATH),
        "shp_bytes": SHP_PATH.stat().st_size,
        "shx_sha256": sha256_file(SHX_PATH),
        "dbf_sha256": sha256_file(DBF_PATH),
        "prj_sha256": sha256_file(PRJ_PATH),
        "uid_definition": "stable instance UID = (EA.shp SHA256, source_feature_id)",
        "identity_note": (
            "source_feature_id is the 1-based .shp record order within this exact archive; the DBF "
            "attribute table is degenerate (Task 6K.1) and is never used for identity"
        ),
    }


def source_root_default() -> Path:
    """The external WHU archive root (read-only). Runtime-overridable via --source-root."""

    return CROPPED_ROOT.parent


# ------------------------------------------------------------------ tiles


@dataclass(frozen=True)
class TileRecord:
    """One canonical tile record (metadata only; geometry lives in the cache)."""

    tile_id: str
    grid_id: str
    category: str
    raster: str
    row: int
    col: int
    source_image_rel: str
    source_label_rel: str
    width: int
    height: int
    split: str

    def as_dict(self, instance_stats: dict | None = None, historical_split: str | None = None) -> dict:
        record = {
            "tile_id": self.tile_id,
            "grid_id": self.grid_id,
            "legacy_category": self.category,
            "source_raster": self.raster,
            "grid_row": self.row,
            "grid_column": self.col,
            "source_image_ref": self.source_image_rel,
            "source_label_ref": self.source_label_rel,
            "width": self.width,
            "height": self.height,
            "scene_disjoint_split": self.split,
            "legacy_compat_split": historical_split,
            "prov_version": DATASET_VERSION,
        }
        if instance_stats is not None:
            record.update(instance_stats)
        return record


def iter_source_tiles(source_root: Path) -> list[TileRecord]:
    """Enumerate every cropped tile in the archive deterministically."""

    cropped = Path(source_root) / "1. The cropped image data and raster labels"
    records: list[TileRecord] = []
    for category in LEGACY_CATEGORIES:
        image_dir = cropped / category / "image"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"missing cropped image directory: {image_dir}")
        for path in sorted(image_dir.glob("*.tif"), key=lambda p: _stem_sort_key(p.stem)):
            stem = path.stem
            region = stem.split("_", 1)[0] if "_" in stem else "test"
            raster = REGION_TO_RASTER.get(region)
            if raster is None:
                raise ValueError(f"{stem}: cannot map to a whole raster")
            index = int(stem.split("_", 1)[1]) if "_" in stem else int(stem)
            grid = tile_grid_for(raster)
            if not grid.contains(index):
                raise ValueError(f"{stem}: index {index} outside the {raster} grid")
            row, col = grid.index_to_rowcol(index)
            records.append(
                TileRecord(
                    tile_id=stem,
                    grid_id=f"{raster}:{row}:{col}",
                    category=category,
                    raster=raster,
                    row=row,
                    col=col,
                    source_image_rel=f"1. The cropped image data and raster labels/{category}/image/{stem}.tif",
                    source_label_rel=f"1. The cropped image data and raster labels/{category}/label/{stem}.tif",
                    width=TILE_SIZE,
                    height=TILE_SIZE,
                    split=SCENE_DISJOINT_SPLITS[raster],
                )
            )
    return records


def _stem_sort_key(stem: str):
    region = stem.split("_", 1)[0] if "_" in stem else "test"
    index = int(stem.split("_", 1)[1]) if "_" in stem else int(stem)
    return (region, index)


def tile_geometry_for(record: TileRecord):
    """Where a tile sits inside its whole raster (window + affine)."""

    grid = tile_grid_for(record.raster)
    x0, y0, size, _height = grid.window(grid.rowcol_to_index(record.row, record.col))
    return x0, y0, size, raster_affine(record.raster)


_raster_affine_cache: dict[str, WorldFile] = {}


def raster_affine(raster: str) -> WorldFile:
    if raster not in _raster_affine_cache:
        _raster_affine_cache[raster] = world_file_for_raster(WHOLE_AREA_DIR / "label" / f"{raster}.tif")
    return _raster_affine_cache[raster]


def tile_map_bbox(record: TileRecord) -> tuple[float, float, float, float]:
    x0, y0, size, affine = tile_geometry_for(record)
    x_left, y_top = affine.pixel_to_map(x0, y0)
    x_right, y_bottom = affine.pixel_to_map(x0 + size, y0 + size)
    return (min(x_left, x_right), min(y_top, y_bottom), max(x_left, x_right), max(y_top, y_bottom))


# ------------------------------------------------------------------ corpus


@dataclass
class VectorCorpus:
    """Every native feature plus a bbox index for window queries."""

    features: list[VectorFeature]
    bboxes: np.ndarray

    def __len__(self) -> int:
        return len(self.features)

    def candidates(self, bbox: tuple[float, float, float, float]) -> list[int]:
        x0, y0, x1, y1 = bbox
        keep = (
            (self.bboxes[:, 2] >= x0) & (self.bboxes[:, 0] <= x1)
            & (self.bboxes[:, 3] >= y0) & (self.bboxes[:, 1] <= y1)
        )
        return np.nonzero(keep)[0].tolist()


def load_vector_corpus() -> VectorCorpus:
    features: list[VectorFeature] = []
    for index, feature in enumerate(iter_shp_features(SHP_PATH), start=1):
        features.append(vector_feature_from_shp(feature, index))
    bboxes = np.asarray([f.bbox_map for f in features], dtype=np.float64)
    return VectorCorpus(features=features, bboxes=bboxes)


# ------------------------------------------------------------------ clipping


def clip_ring_to_rect(ring: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a ring against an axis-aligned rectangle.

    The ring may be open or closed; the result is a closed vertex list (first == last) or an empty
    array when the ring lies outside the window. Used so tile-clipped polygons are exact rather than
    re-derived from a raster.
    """

    points = np.asarray(ring, dtype=np.float64)
    if points.shape[0] > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if points.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)

    def clip_edge(polygon: np.ndarray, inside, intersect) -> np.ndarray:
        if polygon.shape[0] == 0:
            return polygon
        out: list[np.ndarray] = []
        count = polygon.shape[0]
        for i in range(count):
            current = polygon[i]
            previous = polygon[(i - 1) % count]
            current_in = inside(current)
            previous_in = inside(previous)
            if current_in:
                if not previous_in:
                    out.append(intersect(previous, current))
                out.append(current)
            elif previous_in:
                out.append(intersect(previous, current))
        return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 2), dtype=np.float64)

    def vertical(value):
        return (
            (lambda p: p[0] >= value),
            (lambda a, b: np.array([value, a[1] + (b[1] - a[1]) * (value - a[0]) / (b[0] - a[0])])),
        )

    def horizontal(value):
        return (
            (lambda p: p[1] >= value),
            (lambda a, b: np.array([a[0] + (b[0] - a[0]) * (value - a[1]) / (b[1] - a[1]), value])),
        )

    polygon = points
    for check, cut in (
        (lambda p: p[0] >= x0, lambda a, b: np.array([x0, a[1] + (b[1] - a[1]) * (x0 - a[0]) / (b[0] - a[0])])),
        (lambda p: p[0] <= x1, lambda a, b: np.array([x1, a[1] + (b[1] - a[1]) * (x1 - a[0]) / (b[0] - a[0])])),
        (lambda p: p[1] >= y0, lambda a, b: np.array([a[0] + (b[0] - a[0]) * (y0 - a[1]) / (b[1] - a[1]), y0])),
        (lambda p: p[1] <= y1, lambda a, b: np.array([a[0] + (b[0] - a[0]) * (y1 - a[1]) / (b[1] - a[1]), y1])),
    ):
        polygon = clip_edge(polygon, check, cut)
        if polygon.shape[0] == 0:
            return polygon
    if polygon.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack([polygon, polygon[:1]])


@dataclass
class NativeInstance:
    """One tile-clipped native feature (geometry preserved, nothing simplified away)."""

    source_feature_id: int
    tile_instance_id: int
    tile_id: str
    rings: list[dict]                 # [{"kind": "outer"|"hole", "points": [[x, y], ...]}]
    bbox_xyxy_px: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    clipped_area_px: int
    full_area_px: int
    touched_tile_border: bool
    visible_fraction: float
    multipart: bool
    n_rings: int
    n_holes: int
    n_outer_rings_full: int
    n_hole_rings_full: int
    tiny_area: bool
    clipped_by_tile: bool
    area_map_units: float

    def as_dict(self, with_geometry: bool = False) -> dict:
        record = {
            "tile_id": self.tile_id,
            "tile_instance_id": self.tile_instance_id,
            "source_feature_id": self.source_feature_id,
            "bbox_xyxy_px": list(self.bbox_xyxy_px),
            "centroid_px": [self.centroid_px[0], self.centroid_px[1]],
            "clipped_area_px": self.clipped_area_px,
            "full_area_px": self.full_area_px,
            "visible_fraction": self.visible_fraction,
            "touches_tile_border": self.touched_tile_border,
            "clipped_by_tile": self.clipped_by_tile,
            "multipart": self.multipart,
            "n_rings": self.n_rings,
            "n_holes": self.n_holes,
            "n_outer_rings_full": self.n_outer_rings_full,
            "n_hole_rings_full": self.n_hole_rings_full,
            "tiny_area": self.tiny_area,
            "area_map_units": self.area_map_units,
        }
        if with_geometry:
            record["rings"] = self.rings
        return record


def _polygon_area_points(points: np.ndarray) -> float:
    if points.shape[0] < 4:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


class FullAreaCache:
    """Full (unclipped) rasterized area per source feature, in tile pixels.

    Rasterising each feature once into its own window keeps the geometry identical to the clipped
    comparison; the value is in *pixels*, never converted to ground m^2 (the CRS is World Mercator).
    """

    def __init__(self, corpus: VectorCorpus) -> None:
        self.corpus = corpus
        self._cache: dict[int, int] = {}

    def area_px(self, feature: VectorFeature, affine: WorldFile) -> int:
        if feature.feature_id in self._cache:
            return self._cache[feature.feature_id]
        x0 = math.floor(min(affine.map_to_pixel(feature.bbox_map[0], feature.bbox_map[1])[0],
                            affine.map_to_pixel(feature.bbox_map[2], feature.bbox_map[3])[0])) - 2
        y0 = math.floor(min(affine.map_to_pixel(feature.bbox_map[0], feature.bbox_map[1])[1],
                            affine.map_to_pixel(feature.bbox_map[2], feature.bbox_map[3])[1])) - 2
        x1 = math.ceil(max(affine.map_to_pixel(feature.bbox_map[0], feature.bbox_map[1])[0],
                           affine.map_to_pixel(feature.bbox_map[2], feature.bbox_map[3])[0])) + 2
        y1 = math.ceil(max(affine.map_to_pixel(feature.bbox_map[0], feature.bbox_map[1])[1],
                           affine.map_to_pixel(feature.bbox_map[2], feature.bbox_map[3])[1])) + 2
        width = max(int(x1 - x0), 1)
        height = max(int(y1 - y0), 1)
        mask = rasterize_feature(feature, affine, (height, width), origin_col=x0, origin_row=y0)
        area = int(mask.sum())
        self._cache[feature.feature_id] = area
        return area


def clip_native_instances(
    corpus: VectorCorpus,
    record: TileRecord,
    full_area_cache: FullAreaCache,
    tiny_area_px: int = 50,
) -> list[NativeInstance]:
    """Clip every intersecting native feature into the tile window.

    Returns instances ordered by `source_feature_id`; `tile_instance_id` is 1-based in that order.
    A feature whose clipped rasterisation is empty is skipped (it contributes no pixels to the tile),
    and a feature that produces a RING but no pixel is still reported when the polygon has area.
    """

    x0, y0, size, affine = tile_geometry_for(record)
    bbox = tile_map_bbox(record)
    instances: list[NativeInstance] = []
    next_id = 0
    for index in corpus.candidates(bbox):
        feature = corpus.features[index]
        pixel_rings: list[dict] = []
        for ring_index, kind in (
            *((i, "outer") for i in feature.outer_rings),
            *((i, "hole") for i in feature.hole_rings),
        ):
            pixel = np.empty((feature.ring(ring_index).shape[0], 2), dtype=np.float64)
            for position, (x, y) in enumerate(feature.ring(ring_index)):
                col, row = affine.map_to_pixel(float(x), float(y))
                pixel[position, 0] = col - x0
                pixel[position, 1] = row - y0
            clipped = clip_ring_to_rect(pixel, 0.0, 0.0, float(size), float(size))
            if clipped.shape[0] >= 3:
                pixel_rings.append({"kind": kind, "points": np.round(clipped, 3).tolist()})
        if not pixel_rings:
            continue
        mask = _rasterize_rings(pixel_rings, size)
        area = int(mask.sum())
        if area == 0:
            continue
        ys, xs = np.nonzero(mask)
        bx0, bx1 = int(xs.min()), int(xs.max())
        by0, by1 = int(ys.min()), int(ys.max())
        touched = bool(bx0 == 0 or by0 == 0 or bx1 == size - 1 or by1 == size - 1)
        full_area = full_area_cache.area_px(feature, affine)
        visible = float(min(1.0, area / full_area)) if full_area else 1.0
        next_id += 1
        instances.append(
            NativeInstance(
                source_feature_id=int(feature.feature_id),
                tile_instance_id=next_id,
                tile_id=record.tile_id,
                rings=pixel_rings,
                bbox_xyxy_px=(bx0, by0, bx1 + 1, by1 + 1),
                centroid_px=(float(xs.mean()), float(ys.mean())),
                clipped_area_px=area,
                full_area_px=int(full_area),
                touched_tile_border=touched,
                visible_fraction=visible,
                multipart=bool(len([r for r in pixel_rings if r["kind"] == "outer"]) > 1),
                n_rings=len(pixel_rings),
                n_holes=len([r for r in pixel_rings if r["kind"] == "hole"]),
                n_outer_rings_full=len(feature.outer_rings),
                n_hole_rings_full=len(feature.hole_rings),
                tiny_area=bool(area < tiny_area_px),
                clipped_by_tile=bool(visible < 0.999),
                area_map_units=float(feature.area_map_units),
            )
        )
    return instances


def _rasterize_rings(rings: list[dict], size: int) -> np.ndarray:
    import cv2

    mask = np.zeros((size, size), dtype=np.uint8)
    outer = [np.asarray(r["points"], dtype=np.float64) for r in rings if r["kind"] == "outer"]
    holes = [np.asarray(r["points"], dtype=np.float64) for r in rings if r["kind"] == "hole"]
    if outer:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in outer], 1)
    if holes:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in holes], 0)
    return mask.astype(bool)


def label_map_from_instances(instances: list[NativeInstance], size: int = TILE_SIZE) -> np.ndarray:
    """uint8 instance-index map (0 = background, k = tile_instance_id)."""

    if len(instances) > MAX_TILE_INSTANCES:
        raise ValueError(
            f"{len(instances)} instances exceed the {MAX_TILE_INSTANCES}-id uint8 label-map limit"
        )
    out = np.zeros((size, size), dtype=np.uint8)
    for instance in instances:
        out[_rasterize_rings(instance.rings, size)] = instance.tile_instance_id
    return out


# ------------------------------------------------------------------ cache I/O


def cache_dir(kind: str) -> Path:
    return CACHE_ROOT / kind


def write_tile_cache(record: TileRecord, instances: list[NativeInstance], label_map: np.ndarray) -> Path:
    """Persist one tile's canonical geometry (gitignored, regenerable)."""

    directory = cache_dir("instances")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.tile_id.replace('/', '_')}.npz"
    flat_points: list[list[float]] = []
    ring_offsets: list[int] = []
    ring_lengths: list[int] = []
    ring_kinds: list[int] = []
    instance_ids: list[int] = []
    for instance in instances:
        for ring in instance.rings:
            ring_offsets.append(len(flat_points))
            ring_lengths.append(len(ring["points"]))
            ring_kinds.append(1 if ring["kind"] == "outer" else 0)
            flat_points.extend(ring["points"])
            instance_ids.append(instance.tile_instance_id)
    np.savez_compressed(
        path,
        label_map=label_map,
        points=np.asarray(flat_points, dtype=np.float32).reshape(-1, 2),
        ring_offsets=np.asarray(ring_offsets, dtype=np.int32),
        ring_lengths=np.asarray(ring_lengths, dtype=np.int32),
        ring_kinds=np.asarray(ring_kinds, dtype=np.int8),
        ring_instance_ids=np.asarray(instance_ids, dtype=np.int32),
        instance_ids=np.asarray([i.tile_instance_id for i in instances], dtype=np.int32),
        source_feature_ids=np.asarray([i.source_feature_id for i in instances], dtype=np.int64),
        bboxes=np.asarray([i.bbox_xyxy_px for i in instances], dtype=np.int32).reshape(-1, 4),
        centroids=np.asarray([i.centroid_px for i in instances], dtype=np.float64).reshape(-1, 2),
        clipped_area=np.asarray([i.clipped_area_px for i in instances], dtype=np.int64),
        full_area=np.asarray([i.full_area_px for i in instances], dtype=np.int64),
        visible_fraction=np.asarray([i.visible_fraction for i in instances], dtype=np.float64),
        touches_border=np.asarray([i.touched_tile_border for i in instances], dtype=bool),
        tiny=np.asarray([i.tiny_area for i in instances], dtype=bool),
        multipart=np.asarray([i.multipart for i in instances], dtype=bool),
        n_holes=np.asarray([i.n_holes for i in instances], dtype=np.int32),
    )
    return path


def read_tile_cache(tile_id: str) -> dict | None:
    path = cache_dir("instances") / f"{tile_id.replace('/', '_')}.npz"
    if not path.is_file():
        return None
    with np.load(path) as data:
        return {
            "label_map": data["label_map"],
            "points": data["points"],
            "ring_offsets": data["ring_offsets"],
            "ring_lengths": data["ring_lengths"],
            "ring_kinds": data["ring_kinds"],
            "ring_instance_ids": data["ring_instance_ids"],
            "instance_ids": data["instance_ids"],
            "source_feature_ids": data["source_feature_ids"],
            "bboxes": data["bboxes"],
            "centroids": data["centroids"],
            "clipped_area": data["clipped_area"],
            "full_area": data["full_area"],
            "visible_fraction": data["visible_fraction"],
            "touches_border": data["touches_border"],
            "tiny": data["tiny"],
            "multipart": data["multipart"],
            "n_holes": data["n_holes"],
        }


def instance_rings(cache: dict, tile_instance_id: int) -> list[dict]:
    """Rings of one cached instance, in tile pixel coordinates."""

    rings: list[dict] = []
    for index in range(len(cache["ring_offsets"])):
        if int(cache["ring_instance_ids"][index]) != int(tile_instance_id):
            continue
        start = int(cache["ring_offsets"][index])
        length = int(cache["ring_lengths"][index])
        rings.append(
            {
                "kind": "outer" if int(cache["ring_kinds"][index]) == 1 else "hole",
                "points": cache["points"][start: start + length].tolist(),
            }
        )
    return rings


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
