"""Task 6K.1: read-only WHU native-vector audit library.

Complements ``buildreasonseg.runtime._frozen.mvp/shapefile_reader.py`` with:

* a minimal standard-library TIFF IFD reader (metadata only — never a full decode);
* world-file (``.tfw``) georeferencing and map<->pixel transforms;
* the 512-tile grid of each whole-area raster and the tile-index -> window mapping;
* per-feature ring classification, bbox indexing and window rasterization;
* windowed comparison utilities used to VALIDATE the mapping against the cropped tiles.

Nothing here writes to the source archive; the only pixel decoding happens through small PIL
crops of the whole-area rasters (an existing read-only tool), never a whole-image load.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .shapefile_reader import ShpFeature, classify_rings, ring_signed_area

# ------------------------------------------------------------------ paths (read-only)

ORIGINAL_ROOT = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")
SHP_DIR = ORIGINAL_ROOT / "2. The shape file of the whole images"
WHOLE_AREA_DIR = ORIGINAL_ROOT / "3. The whole area image dataset"
CROPPED_ROOT = ORIGINAL_ROOT / "1. The cropped image data and raster labels"

SHP_PATH = SHP_DIR / "EA.shp"
SHX_PATH = SHP_DIR / "EA.shx"
DBF_PATH = SHP_DIR / "EA.dbf"
PRJ_PATH = SHP_DIR / "EA.prj"

#: Whole-area raster name -> the source-split prefix used in the cropped tile names.
WHOLE_RASTERS = ("train1", "train2", "test")
#: Cropped tile prefix -> whole-area raster name.
REGION_TO_RASTER = {"1": "train1", "2": "train2", "test": "test"}

TILE_SIZE = 512

# ------------------------------------------------------------------ TIFF metadata (stdlib)


TIFF_TAGS = {
    256: "ImageWidth", 257: "ImageLength", 258: "BitsPerSample", 259: "Compression",
    262: "PhotometricInterpretation", 273: "StripOffsets", 277: "SamplesPerPixel",
    278: "RowsPerStrip", 279: "StripByteCounts", 284: "PlanarConfiguration",
    317: "Predictor", 322: "TileWidth", 323: "TileLength", 324: "TileOffsets",
    325: "TileByteCounts", 339: "SampleFormat", 254: "NewSubfileType", 274: "Orientation",
}

COMPRESSION_NAMES = {
    1: "none", 2: "CCITT_RLE", 3: "CCITT_G3", 4: "CCITT_G4", 5: "LZW", 6: "JPEG_old",
    7: "JPEG", 8: "Deflate", 32773: "PackBits", 32946: "Deflate_old",
}

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


def read_tiff_metadata(path: Path) -> dict:
    """Minimal read-only TIFF/BigTIFF IFD0 reader: dimensions, compression, tiling, layout."""

    with Path(path).open("rb") as handle:
        header = handle.read(16)
        if len(header) < 8:
            raise ValueError(f"{path}: truncated TIFF header")
        if header[:2] == b"II":
            endian = "<"
        elif header[:2] == b"MM":
            endian = ">"
        else:
            raise ValueError(f"{path}: not a TIFF (bad byte order)")
        magic = struct.unpack(endian + "H", header[2:4])[0]
        bigtiff = magic == 43
        if magic not in (42, 43):
            raise ValueError(f"{path}: bad TIFF magic {magic}")
        if bigtiff:
            offset_size = struct.unpack(endian + "H", header[4:6])[0]
            if offset_size != 8:
                raise ValueError(f"{path}: unsupported BigTIFF offset size {offset_size}")
            ifd_offset = struct.unpack(endian + "Q", header[8:16])[0]
        else:
            ifd_offset = struct.unpack(endian + "I", header[4:8])[0]
        handle.seek(ifd_offset)
        if bigtiff:
            count = struct.unpack(endian + "Q", handle.read(8))[0]
            entry_size = 20
            entries = handle.read(count * entry_size)
        else:
            count = struct.unpack(endian + "H", handle.read(2))[0]
            entry_size = 12
            entries = handle.read(count * entry_size)

    tags: dict[int, object] = {}
    for index in range(count):
        entry = entries[index * entry_size: index * entry_size + entry_size]
        if bigtiff:
            tag, type_code, n_values = struct.unpack(endian + "HHQ", entry[0:12])
            value_field = entry[12:20]
        else:
            tag, type_code, n_values = struct.unpack(endian + "HHI", entry[0:8])
            value_field = entry[8:12]
        unit = _TYPE_SIZES.get(type_code, 0)
        size = unit * n_values
        if size <= len(value_field):
            data = value_field[:size]
        else:
            offset = struct.unpack(endian + ("Q" if bigtiff else "I"), value_field)[0]
            with Path(path).open("rb") as handle:
                handle.seek(offset)
                data = handle.read(int(size))
        values = _decode_tiff_values(endian, type_code, int(n_values), data)
        tags[tag] = values if isinstance(values, list) and len(values) > 1 else (
            values[0] if isinstance(values, list) and values else values
        )

    width = int(tags.get(256))
    height = int(tags.get(257))
    compression = int(tags.get(259, 1))
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "endianness": "little" if endian == "<" else "big",
        "bigtiff": bool(bigtiff),
        "width": width,
        "height": height,
        "pixels": width * height,
        "bits_per_sample": tags.get(258),
        "samples_per_pixel": int(tags.get(277, 1)),
        "sample_format": tags.get(339),
        "compression": compression,
        "compression_name": COMPRESSION_NAMES.get(compression, f"unknown({compression})"),
        "photometric": int(tags.get(262, 1)),
        "planar_config": int(tags.get(284, 1)),
        "predictor": int(tags.get(317, 1)),
        "tiled": 322 in tags and 323 in tags,
        "tile_width": int(tags.get(322, 0)) or None,
        "tile_height": int(tags.get(323, 0)) or None,
        "tile_count": len(tags.get(324, [])) if isinstance(tags.get(324), list) else (1 if 324 in tags else None),
        "rows_per_strip": int(tags.get(278, 0)) or None,
    }


def _decode_tiff_values(endian: str, type_code: int, n_values: int, data: bytes):
    formats = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i",
               10: "ii", 11: "f", 12: "d"}
    fmt = formats.get(type_code)
    if fmt is None:
        return None
    unit = struct.calcsize(endian + fmt)
    values = []
    for index in range(n_values):
        chunk = data[index * unit: (index + 1) * unit]
        if len(chunk) < unit:
            break
        value = struct.unpack(endian + fmt, chunk)
        values.append(value[0] if len(value) == 1 else value)
    if type_code == 2:
        return b"".join(v for v in values if isinstance(v, bytes)).decode("ascii", errors="replace")
    return values


# ------------------------------------------------------------------ world files / affine


@dataclass(frozen=True)
class WorldFile:
    """ESRI world file: x' = a*col + b*row + c ; y' = d*col + e*row + f (corner-based)."""

    a: float
    d: float
    b: float
    e: float
    c: float
    f: float
    path: Path

    @property
    def pixel_size_x(self) -> float:
        return self.a

    @property
    def pixel_size_y(self) -> float:
        return -self.e

    @property
    def rotated(self) -> bool:
        return abs(self.b) > 1e-12 or abs(self.d) > 1e-12

    def pixel_to_map(self, col: float, row: float) -> tuple[float, float]:
        return (
            self.c + self.a * col + self.b * row,
            self.f + self.d * col + self.e * row,
        )

    def map_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Corner-based inverse (exact for the unrotated case)."""

        if self.rotated:
            det = self.a * self.e - self.b * self.d
            col = (self.e * (x - self.c) - self.b * (y - self.f)) / det
            row = (-self.d * (x - self.c) + self.a * (y - self.f)) / det
            return col, row
        col = (x - self.c) / self.a
        row = (y - self.f) / self.e
        return col, row

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "a": self.a, "d": self.d, "b": self.b, "e": self.e, "c": self.c, "f": self.f,
            "pixel_size_x": self.pixel_size_x,
            "pixel_size_y": self.pixel_size_y,
            "rotated": self.rotated,
            "upper_left_map": [self.c, self.f],
        }


def read_world_file(path: Path) -> WorldFile:
    values = [float(token) for token in Path(path).read_text(encoding="ascii").split()]
    if len(values) != 6:
        raise ValueError(f"{path}: expected 6 world-file values, got {len(values)}")
    a, d, b, e, c, f = values
    return WorldFile(a=a, d=d, b=b, e=e, c=c, f=f, path=Path(path))


def world_file_for_raster(raster: Path) -> WorldFile | None:
    """The label-side ``.tfw`` that georeferences a whole-area raster."""

    stem = Path(raster).stem
    candidate = WHOLE_AREA_DIR / "label" / f"{stem}.tfw"
    if candidate.is_file():
        return read_world_file(candidate)
    return None


# ------------------------------------------------------------------ tile grid


@dataclass(frozen=True)
class TileGrid:
    """The 512-tile grid of one whole-area raster (only full tiles are cropped)."""

    raster: str
    width: int
    height: int
    tile: int = TILE_SIZE

    @property
    def columns(self) -> int:
        return self.width // self.tile

    @property
    def rows(self) -> int:
        return self.height // self.tile

    @property
    def capacity(self) -> int:
        return self.columns * self.rows

    def index_to_rowcol(self, index: int) -> tuple[int, int]:
        row, col = divmod(int(index), self.columns)
        return row, col

    def rowcol_to_index(self, row: int, col: int) -> int:
        return int(row) * self.columns + int(col)

    def window(self, index: int) -> tuple[int, int, int, int]:
        row, col = self.index_to_rowcol(index)
        x0 = col * self.tile
        y0 = row * self.tile
        return x0, y0, self.tile, self.tile

    def contains(self, index: int) -> bool:
        return 0 <= int(index) < self.capacity

    def as_dict(self) -> dict:
        return {
            "raster": self.raster,
            "width": self.width,
            "height": self.height,
            "tile": self.tile,
            "columns": self.columns,
            "rows": self.rows,
            "capacity": self.capacity,
            "right_margin_px": self.width - self.columns * self.tile,
            "bottom_margin_px": self.height - self.rows * self.tile,
        }


def tile_grid_for(raster: str, metadata: dict | None = None) -> TileGrid:
    if metadata is None:
        metadata = read_tiff_metadata(WHOLE_AREA_DIR / "image" / f"{raster}.tif")
    return TileGrid(raster=raster, width=int(metadata["width"]), height=int(metadata["height"]))


# ------------------------------------------------------------------ vector features


@dataclass
class VectorFeature:
    """One native vector record with its ring classification and derived geometry."""

    feature_id: int              # 1-based record index (stable: the .shp record order)
    record_number: int
    bbox_map: tuple[float, float, float, float]
    n_parts: int
    n_points: int
    outer_rings: list[int]
    hole_rings: list[int]
    degenerate_rings: list[int]
    area_map_units: float
    perimeter_map_units: float
    parts: list[tuple[int, int]]
    points: np.ndarray = field(repr=False)

    def ring(self, index: int) -> np.ndarray:
        start, end = self.parts[index]
        return self.points[start:end]

    def as_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "record_number": self.record_number,
            "bbox_map": list(self.bbox_map),
            "n_parts": self.n_parts,
            "n_points": self.n_points,
            "n_outer_rings": len(self.outer_rings),
            "n_hole_rings": len(self.hole_rings),
            "n_degenerate_rings": len(self.degenerate_rings),
            "area_map_units": self.area_map_units,
            "perimeter_map_units": self.perimeter_map_units,
        }


def vector_feature_from_shp(feature: ShpFeature, feature_id: int) -> VectorFeature:
    classification = classify_rings(feature)
    outer_area = sum(abs(ring_signed_area(feature.ring(i))) for i in classification["outer_rings"])
    hole_area = sum(abs(ring_signed_area(feature.ring(i))) for i in classification["hole_rings"])
    perimeter = 0.0
    for index in range(feature.n_parts):
        ring = feature.ring(index)
        if ring.shape[0] < 2:
            continue
        delta = np.diff(np.vstack([ring, ring[:1]]), axis=0)
        perimeter += float(np.sqrt((delta ** 2).sum(axis=1)).sum())
    return VectorFeature(
        feature_id=int(feature_id),
        record_number=int(feature.record_number),
        bbox_map=tuple(float(v) for v in (feature.bbox or (0.0, 0.0, 0.0, 0.0))),
        n_parts=int(feature.n_parts),
        n_points=int(feature.n_points),
        outer_rings=list(classification["outer_rings"]),
        hole_rings=list(classification["hole_rings"]),
        degenerate_rings=list(classification["degenerate_rings"]),
        area_map_units=float(outer_area - hole_area),
        perimeter_map_units=float(perimeter),
        parts=list(feature.parts),
        points=np.asarray(feature.points, dtype=np.float64),
    )


def feature_intersects_bbox(feature: VectorFeature, bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    fx0, fy0, fx1, fy1 = feature.bbox_map
    return not (fx1 < x0 or fx0 > x1 or fy1 < y0 or fy0 > y1)


def count_features_per_component(component_masks, feature_masks, overlap_share: float = 0.5) -> list[int]:
    """Primary touching-building merge metric.

    For every reference mask (a semantic connected component) count how many native feature masks
    have at least ``overlap_share`` of *their own* pixels inside it. A component with >= 2 counts
    contains that many distinct manually delineated buildings. Pure function so it can be unit
    tested on synthetic masks.
    """

    counts: list[int] = []
    for component_mask in component_masks:
        if not np.asarray(component_mask).any():
            counts.append(0)
            continue
        count = 0
        for feature_mask in feature_masks:
            feature_mask = np.asarray(feature_mask).astype(bool)
            area = int(feature_mask.sum())
            if area == 0:
                continue
            overlap = int(np.logical_and(feature_mask, np.asarray(component_mask).astype(bool)).sum())
            if overlap / area >= overlap_share:
                count += 1
        counts.append(count)
    return counts


def count_targets_per_feature(feature_masks, target_masks, share: float = 0.25) -> list[int]:
    """Split metric: for every feature mask count the reference masks holding >= ``share`` of it."""

    counts: list[int] = []
    for feature_mask in feature_masks:
        feature_mask = np.asarray(feature_mask).astype(bool)
        area = int(feature_mask.sum())
        if area == 0:
            counts.append(0)
            continue
        count = 0
        for target_mask in target_masks:
            overlap = int(np.logical_and(feature_mask, np.asarray(target_mask).astype(bool)).sum())
            if overlap / area >= share:
                count += 1
        counts.append(count)
    return counts


def rasterize_feature(feature: VectorFeature, affine: WorldFile, shape: tuple[int, int],
                      origin_col: float = 0.0, origin_row: float = 0.0) -> np.ndarray:
    """Rasterize one feature into a window whose pixel (0, 0) is (origin_col, origin_row)."""

    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    outer_polygons = []
    for ring_index in feature.outer_rings:
        ring = feature.ring(ring_index)
        if ring.shape[0] < 3:
            continue
        pixels = np.empty_like(ring)
        for i, (x, y) in enumerate(ring):
            col, row = affine.map_to_pixel(float(x), float(y))
            pixels[i, 0] = col - origin_col
            pixels[i, 1] = row - origin_row
        polygon = np.round(pixels).astype(np.int32)
        if polygon.shape[0] >= 3:
            outer_polygons.append(polygon)
    if not outer_polygons:
        return mask.astype(bool)
    cv2.fillPoly(mask, outer_polygons, 1)
    holes = []
    for ring_index in feature.hole_rings:
        ring = feature.ring(ring_index)
        if ring.shape[0] < 3:
            continue
        pixels = np.empty_like(ring)
        for i, (x, y) in enumerate(ring):
            col, row = affine.map_to_pixel(float(x), float(y))
            pixels[i, 0] = col - origin_col
            pixels[i, 1] = row - origin_row
        polygon = np.round(pixels).astype(np.int32)
        if polygon.shape[0] >= 3:
            holes.append(polygon)
    if holes:
        cv2.fillPoly(mask, holes, 0)
    return mask.astype(bool)
