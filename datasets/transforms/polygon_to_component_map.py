"""Polygon -> building component representation.

Converts legacy WHU YOLO polygon labels into a stable, deterministic
component-index representation:

    YOLO polygon labels
        -> deterministic rasterization
        -> component-index map (one indexed PNG per source image)
        -> geometry metadata (JSONL, one line per source image)

Terminology
-----------
A polygon in the legacy labels is a *building connected component*, not a
verified physical building instance. The source raster was a binary semantic
mask, and the legacy pipeline applied connected components + RETR_EXTERNAL, so
mutually touching buildings were already merged before this code ever runs.

Accordingly this module uses ``component_id`` / ``component_map`` /
``building_component``. The word "instance" is deliberately avoided; where a
legacy term must be referenced it is written ``pseudo_instance``.

Geometry source of truth
------------------------
All geometry (area, centroid, bbox, width, height) is derived from the
**rasterized component map**, never from polygon vertex means, bbox centres,
the original raster labels, or any model prediction. This guarantees that mask
supervision and geometry supervision share exactly one representation.

Provenance note
---------------
Polygon vertices are retained verbatim in the metadata as provenance. They are
NOT the geometry source of truth.

Known, deliberate bias
----------------------
The legacy polygons came from ``RETR_EXTERNAL`` contours, so interior holes
(light wells, courtyards) were already lost before rasterization. A rasterized
component is therefore a *filled exterior polygon*. It is not a pixel-exact
recovery of the original building footprint, and this representation must not
be described as equal to the original raster ground truth.

No dependency on anything outside this repository except reading the legacy
label/image files themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

REPRESENTATION_VERSION = "v1.0"
REPRESENTATION_NAME = "building_connected_component"
BACKGROUND_ID = 0
MAX_UINT8_ID = 255
ENTITY_SEMANTICS = "connected_component"

#: Source-image file extensions to probe, in priority order.
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp",
)


class RasterizationError(RuntimeError):
    """Raised when rasterization cannot proceed safely."""


class ComponentIdOverflowError(RasterizationError):
    """Raised when the component count exceeds what the chosen dtype can hold."""


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PolygonRecord:
    """A single polygon line from a legacy YOLO label file."""

    source_polygon_index: int
    """Zero-based index of this polygon's line within the label file."""

    class_id: int
    """YOLO class id. Always 0 (`building`) in the current dataset."""

    vertices_normalized: np.ndarray
    """Raw normalized vertices, shape (N, 2), float64, values nominally [0, 1].

    Retained as provenance only. Not the geometry source of truth.
    """

    @property
    def n_vertices(self) -> int:
        return int(self.vertices_normalized.shape[0])


@dataclass
class ComponentGeometry:
    """Geometry derived from a rasterized component's pixel set."""

    component_id: int
    source_polygon_index: int
    area_px: int
    area_ratio: float
    centroid_px: tuple[float, float]
    centroid_normalized: tuple[float, float]
    bbox_xyxy_px: tuple[int, int, int, int]
    bbox_xyxy_normalized: tuple[float, float, float, float]
    width_px: int
    height_px: int
    touches_image_border: bool
    continuous_polygon_area_px: float


@dataclass
class RasterizationDiagnostics:
    """Per-image rasterization diagnostics. Never silently swallowed."""

    n_polygons_read: int = 0
    n_components_written: int = 0
    empty_mask_polygons: list[int] = field(default_factory=list)
    """Source polygon indices whose rasterization produced zero pixels."""

    conflict_pairs: list[tuple[int, int, int]] = field(default_factory=list)
    """(source_polygon_index_a, source_polygon_index_b, overlap_px) records."""

    out_of_bound_vertices: int = 0
    """Vertex coordinates that fell outside [0, width-1] x [0, height-1].

    Vertices are never clipped: rasterization clamps to the image viewport
    itself, so clamping here would silently move geometry. These vertices are
    counted and reported instead.
    """

    malformed_lines: list[int] = field(default_factory=list)
    """1-based line numbers that could not be parsed as a polygon."""

    non_building_class_lines: list[int] = field(default_factory=list)
    """1-based line numbers whose class id was not 0."""

    def is_valid(self) -> bool:
        return not (
            self.conflict_pairs
            or self.empty_mask_polygons
            or self.malformed_lines
            or self.non_building_class_lines
        )


@dataclass
class ComponentMapResult:
    """Full result of converting one label file."""

    image_id: str
    split: str
    width: int
    height: int
    component_map: np.ndarray
    components: list[ComponentGeometry]
    diagnostics: RasterizationDiagnostics
    polygons: list[PolygonRecord]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_yolo_polygon_label(
    label_path: Path,
) -> tuple[list[PolygonRecord], list[int], list[int]]:
    """Parse a YOLO segmentation polygon label file.

    Every non-empty line must become exactly one :class:`PolygonRecord`.
    No filtering is applied: the legacy labels were already filtered
    historically (``contourArea < 50`` was dropped upstream), and re-filtering
    here would alter the legacy geometry ground truth.

    Returns
    -------
    (polygons, malformed_line_numbers, non_building_class_line_numbers)
        Line numbers are 1-based, referring to the original file.
    """

    text = label_path.read_text(encoding="utf-8")
    polygons: list[PolygonRecord] = []
    malformed: list[int] = []
    wrong_class: list[int] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        tokens = raw_line.split()
        if not tokens:
            continue  # blank lines carry no polygon; not an error

        if len(tokens) < 7:
            # 1 class id + at least 3 xy pairs = 7 tokens
            malformed.append(line_no)
            continue

        try:
            class_id = int(float(tokens[0]))
            coords = [float(tok) for tok in tokens[1:]]
        except ValueError:
            malformed.append(line_no)
            continue

        if len(coords) % 2 != 0:
            malformed.append(line_no)
            continue

        vertices = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
        if vertices.shape[0] < 3:
            malformed.append(line_no)
            continue

        if class_id != 0:
            wrong_class.append(line_no)

        polygons.append(
            PolygonRecord(
                source_polygon_index=len(polygons),
                class_id=class_id,
                vertices_normalized=vertices,
            )
        )

    return polygons, malformed, wrong_class


# --------------------------------------------------------------------------
# Coordinate conversion
# --------------------------------------------------------------------------


def normalized_to_pixel(
    vertices_normalized: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Convert normalized vertices to float pixel coordinates.

    Uses ``x_px = x_norm * width`` (no ``-1``), matching the legacy conversion
    direction where the denominator was the image width/height. This keeps this
    representation consistent with how the polygons were originally produced.
    """

    scale = np.array([float(width), float(height)], dtype=np.float64)
    return vertices_normalized * scale


def compute_continuous_polygon_area(vertices_px: np.ndarray) -> float:
    """Shoelace area of a float pixel-coordinate polygon.

    Diagnostic only -- used to characterise rasterization discretization error.
    Never used as the geometry ground truth.
    """

    x = vertices_px[:, 0]
    y = vertices_px[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def to_raster_vertices(
    vertices_px: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    """Round float pixel vertices to integer pixel indices.

    Vertices are intentionally **not** clipped. Rasterization itself skips
    pixels outside the image, so clamping is unnecessary and would silently
    move geometry. Keeping the true coordinates means out-of-range vertices are
    faithfully reported rather than hidden.

    Returns
    -------
    (vertices_int, n_out_of_bound)
        ``n_out_of_bound`` counts vertex coordinates outside the inclusive
        valid range ``[0, width-1] x [0, height-1]``.
    """

    lower = np.array([0.0, 0.0], dtype=np.float64)
    upper = np.array([float(width - 1), float(height - 1)], dtype=np.float64)

    out_of_bound = int(np.count_nonzero((vertices_px < lower) | (vertices_px > upper)))

    # np.rint is banker's rounding and fully deterministic.
    return np.rint(vertices_px).astype(np.int64), out_of_bound


# --------------------------------------------------------------------------
# Deterministic scanline polygon rasterization
# --------------------------------------------------------------------------


def rasterize_polygon(
    vertices_int: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Rasterize a single polygon into a boolean mask of shape (height, width).

    Backend
    -------
    Uses ``cv2.fillPoly`` as the rasterization backend of this canonical
    representation.

    Note on provenance: the legacy pipeline produced the polygon labels with
    ``cv2.findContours`` (RETR_EXTERNAL) plus ``cv2.approxPolyDP``. It never
    called ``fillPoly``. This project goes the opposite direction -- polygon ->
    filled region -- so ``fillPoly`` is simply the rasterizer of the current
    representation, not the routine that created the labels.

    Consequently, agreement with ``cv2.fillPoly`` means the polygon -> mask
    direction is self-consistent and deterministic. It does **not** mean the
    component map equals the original raster ground truth: interior holes were
    already lost at the ``RETR_EXTERNAL`` step and the boundary was already
    simplified by ``approxPolyDP``.

    This was a deliberate decision. A hand-written scanline rasterizer was
    implemented and validated first: it matched ``cv2.fillPoly`` exactly on all
    axis-aligned shapes (rectangle, thin sliver, full-image extent, L-shape,
    square) but over-filled non-convex polygons by up to 46% because a
    min/max span fill implicitly takes the convex hull. Reproducing OpenCV's
    exact boundary rule for non-convex input would mean reimplementing its
    span-fill conventions with no benefit to this project.

    Determinism
    -----------
    ``cv2.fillPoly`` on integer vertices is a pure integer scanline fill with
    no threading, sampled state, or randomness, so repeated calls on identical
    input produce byte-identical output. ``tests/test_component_conversion.py``
    asserts this rather than trusting it.

    This function does NOT read the legacy raster labels; it only rasterizes
    the polygon vertices that are already present in the workspace. No external
    ground-truth data is accessed.
    """

    mask = np.zeros((height, width), dtype=np.uint8)

    if vertices_int.shape[0] < 3:
        return mask.astype(bool)

    import cv2

    contour = vertices_int.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [contour], 1)

    return mask.astype(bool)


# --------------------------------------------------------------------------
# Geometry from rasterized pixel set
# --------------------------------------------------------------------------


def geometry_from_mask(
    mask: np.ndarray,
    component_id: int,
    source_polygon_index: int,
    width: int,
    height: int,
    continuous_polygon_area_px: float,
) -> ComponentGeometry | None:
    """Derive geometry from the rasterized pixel set of one component.

    Returns ``None`` if the mask is empty (caller records that as a defect).
    """

    ys, xs = np.nonzero(mask)
    area_px = int(xs.size)
    if area_px == 0:
        return None

    total = float(area_px)
    cx = float(xs.sum()) / total
    cy = float(ys.sum()) / total

    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1

    touches = bool(x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1)

    return ComponentGeometry(
        component_id=component_id,
        source_polygon_index=source_polygon_index,
        area_px=area_px,
        area_ratio=area_px / float(width * height),
        centroid_px=(cx, cy),
        centroid_normalized=(cx / width, cy / height),
        bbox_xyxy_px=(x0, y0, x1, y1),
        bbox_xyxy_normalized=(x0 / width, y0 / height, x1 / width, y1 / height),
        width_px=w,
        height_px=h,
        touches_image_border=touches,
        continuous_polygon_area_px=continuous_polygon_area_px,
    )


# --------------------------------------------------------------------------
# Image size discovery
# --------------------------------------------------------------------------


def find_source_image(images_dir: Path, image_id: str) -> Path | None:
    """Locate the source image for ``image_id`` by probing known extensions."""

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.is_file():
            return candidate
        upper = images_dir / f"{image_id}{ext.upper()}"
        if upper.is_file():
            return upper
    return None


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read (width, height) from an image without decoding pixel data.

    Uses Pillow lazily so that the true image size is always authoritative.
    ``512 x 512`` is never assumed anywhere in this module.
    """

    from PIL import Image

    with Image.open(image_path) as im:
        width, height = im.size
    return int(width), int(height)


# --------------------------------------------------------------------------
# Full conversion for one image
# --------------------------------------------------------------------------


def build_component_map(
    label_path: Path,
    image_size: tuple[int, int],
    image_id: str,
    split: str,
) -> ComponentMapResult:
    """Convert one polygon label file into a component-index map plus geometry.

    Component ids are assigned as ``source_polygon_index + 1`` and are stable:
    a polygon that rasterizes to zero pixels consumes its id rather than having
    ids silently renumbered, so that id gaps always correspond to a recorded
    defect.
    """

    width, height = image_size
    if width <= 0 or height <= 0:
        raise RasterizationError(f"invalid image size {image_size} for {image_id}")
    if width > 65535 or height > 65535:
        raise RasterizationError(
            f"image size {image_size} exceeds supported rasterization range"
        )

    polygons, malformed, wrong_class = parse_yolo_polygon_label(label_path)

    n_components = len(polygons)
    if n_components > MAX_UINT8_ID:
        raise ComponentIdOverflowError(
            f"{image_id}: {n_components} components exceeds uint8 capacity "
            f"({MAX_UINT8_ID}). Upgrade the component map dtype to uint16 "
            f"before proceeding -- refusing to overflow silently."
        )

    component_map = np.zeros((height, width), dtype=np.uint8)
    components: list[ComponentGeometry] = []
    diag = RasterizationDiagnostics(
        n_polygons_read=n_components,
        malformed_lines=malformed,
        non_building_class_lines=wrong_class,
    )

    for record in polygons:
        component_id = record.source_polygon_index + 1

        vertices_px = normalized_to_pixel(
            record.vertices_normalized, width, height
        )
        continuous_area = compute_continuous_polygon_area(vertices_px)

        vertices_int, n_oob = to_raster_vertices(
            vertices_px, width, height
        )
        diag.out_of_bound_vertices += n_oob

        mask = rasterize_polygon(vertices_int, width, height)

        if not mask.any():
            diag.empty_mask_polygons.append(record.source_polygon_index)
            # Geometry still recorded so the defect is inspectable.
            components.append(
                ComponentGeometry(
                    component_id=component_id,
                    source_polygon_index=record.source_polygon_index,
                    area_px=0,
                    area_ratio=0.0,
                    centroid_px=(float("nan"), float("nan")),
                    centroid_normalized=(float("nan"), float("nan")),
                    bbox_xyxy_px=(0, 0, 0, 0),
                    bbox_xyxy_normalized=(0.0, 0.0, 0.0, 0.0),
                    width_px=0,
                    height_px=0,
                    touches_image_border=False,
                    continuous_polygon_area_px=continuous_area,
                )
            )
            continue

        # --- conflict detection: never overwrite an existing component id ---
        occupied = component_map != BACKGROUND_ID
        overlap = np.logical_and(mask, occupied)
        if overlap.any():
            overlap_px = int(np.count_nonzero(overlap))
            existing_ids = np.unique(component_map[overlap])
            for existing_id in existing_ids:
                diag.conflict_pairs.append(
                    (record.source_polygon_index, int(existing_id) - 1, overlap_px)
                )
            # Deliberately do NOT paint over the existing component.
            # The conflict is recorded and the image is reported as invalid.
            geom = geometry_from_mask(
                mask, component_id, record.source_polygon_index,
                width, height, continuous_area,
            )
            if geom is not None:
                components.append(geom)
            continue

        component_map[mask] = np.uint8(component_id)

        geom = geometry_from_mask(
            mask, component_id, record.source_polygon_index,
            width, height, continuous_area,
        )
        if geom is None:  # defensive; mask.any() was already true
            diag.empty_mask_polygons.append(record.source_polygon_index)
            continue
        components.append(geom)

    diag.n_components_written = len(components)

    return ComponentMapResult(
        image_id=image_id,
        split=split,
        width=width,
        height=height,
        component_map=component_map,
        components=components,
        diagnostics=diag,
        polygons=polygons,
    )


# --------------------------------------------------------------------------
# Metadata serialisation
# --------------------------------------------------------------------------


def component_to_metadata(
    geom: ComponentGeometry,
    record: PolygonRecord,
) -> dict:
    """Build the metadata dict for a single component."""

    def _clean(value: float) -> float | None:
        """JSON has no NaN; represent undefined numeric geometry as null."""
        if value != value:  # NaN
            return None
        return value

    return {
        "component_id": geom.component_id,
        "source_polygon_index": geom.source_polygon_index,
        "entity_semantics": ENTITY_SEMANTICS,
        "source_polygon_class_id": record.class_id,
        "polygon_normalized": [
            [float(x), float(y)] for x, y in record.vertices_normalized
        ],
        "geometry_source": "rasterized_component_map",
        "area_px": geom.area_px,
        "area_ratio": geom.area_ratio,
        "centroid_px": [_clean(geom.centroid_px[0]), _clean(geom.centroid_px[1])],
        "centroid_normalized": [
            _clean(geom.centroid_normalized[0]),
            _clean(geom.centroid_normalized[1]),
        ],
        "bbox_xyxy_px": list(geom.bbox_xyxy_px),
        "bbox_xyxy_normalized": [float(v) for v in geom.bbox_xyxy_normalized],
        "width_px": geom.width_px,
        "height_px": geom.height_px,
        "touches_image_border": geom.touches_image_border,
        "continuous_polygon_area_px": geom.continuous_polygon_area_px,
        "rasterization_area_error_ratio": (
            _clean(
                abs(geom.area_px - geom.continuous_polygon_area_px)
                / geom.continuous_polygon_area_px
            )
            if geom.continuous_polygon_area_px > 0
            else None
        ),
    }


def result_to_metadata(
    result: ComponentMapResult,
    image_path_rel: str,
    label_path_rel: str,
    component_map_rel: str,
) -> dict:
    """Build the full per-image metadata dict (one JSONL line)."""

    by_index = {p.source_polygon_index: p for p in result.polygons}

    return {
        "image_id": result.image_id,
        "split": result.split,
        "image_path": image_path_rel,
        "label_path": label_path_rel,
        "component_map": component_map_rel,
        "width": result.width,
        "height": result.height,
        "n_components": len(result.components),
        "components": [
            component_to_metadata(geom, by_index[geom.source_polygon_index])
            for geom in result.components
        ],
        "rasterization": {
            "is_valid": result.diagnostics.is_valid(),
            "empty_mask_polygons": result.diagnostics.empty_mask_polygons,
            "conflict_pairs": [
                {
                    "source_polygon_index": a,
                    "conflicts_with_source_polygon_index": b,
                    "overlap_px": n,
                }
                for a, b, n in result.diagnostics.conflict_pairs
            ],
            "malformed_lines": result.diagnostics.malformed_lines,
            "non_building_class_lines": result.diagnostics.non_building_class_lines,
            "out_of_bound_vertices": result.diagnostics.out_of_bound_vertices,
        },
        "representation_version": REPRESENTATION_VERSION,
    }


def write_indexed_png(component_map: np.ndarray, out_path: Path) -> None:
    """Write a component-index map as an indexed PNG.

    Deterministic encoding parameters are set explicitly so that re-running the
    conversion produces byte-identical files.
    """

    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(component_map, mode="L")
    image.save(
        out_path,
        format="PNG",
        optimize=False,
        compress_level=6,
    )


def read_indexed_png(path: Path) -> np.ndarray:
    """Read a component-index PNG back as a uint8 array."""

    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


# --------------------------------------------------------------------------
# Split iteration
# --------------------------------------------------------------------------


def iter_split(
    dataset_root: Path, split: str
) -> Iterator[tuple[str, Path, Path]]:
    """Yield ``(image_id, image_path, label_path)`` for a split.

    Only image ids that have BOTH an image and a label are yielded; ids missing
    either side are skipped here and must be reported by the caller.
    """

    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    if not labels_dir.is_dir():
        return

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_id = label_path.stem
        image_path = find_source_image(images_dir, image_id)
        if image_path is None:
            continue
        yield image_id, image_path, label_path


def unique_component_ids(component_map: np.ndarray) -> np.ndarray:
    """Sorted unique non-background ids present in a component map."""

    ids = np.unique(component_map)
    return ids[ids != BACKGROUND_ID]


def check_id_contiguity(component_map: np.ndarray, n_components: int) -> list[int]:
    """Return ids that are missing from ``{1..n_components}``."""

    present = set(int(v) for v in unique_component_ids(component_map))
    expected = set(range(1, n_components + 1))
    return sorted(expected - present)
