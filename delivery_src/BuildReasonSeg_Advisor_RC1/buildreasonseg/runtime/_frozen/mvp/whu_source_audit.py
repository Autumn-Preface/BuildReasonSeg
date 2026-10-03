"""Task 6K: read-only audit library for the WHU source -> pseudo-instance chain.

Three representations are compared, all read-only:

    (1) ORIGINAL   WHU Satellite Dataset II (East Asia) raster semantic labels
    (2) CONVERTED  the historical semantic -> YOLO polygon conversion (mask_to_yolo.py)
    (3) CURRENT    BuildReasonSeg's building-connected-component representation
                   (datasets/whu/components + metadata), derived from (2) by
                   datasets/transforms/polygon_to_component_map.py

Nothing here writes to the source data, the converted dataset or the legacy project. The
historical conversion semantics are reproduced IN MEMORY exactly as recorded in
``mask_to_yolo.py``: threshold 127, ``RETR_EXTERNAL``, ``CHAIN_APPROX_SIMPLE``,
``cv2.contourArea(contour) < 50`` removal, ``epsilon = 0.001 * arcLength`` simplification,
plus the two details the supplied snippet omits: the ``len(approx) < 3`` skip and the
[0, 1] coordinate clipping.

Terminology rule: connected components of a binary semantic mask are NOT verified physical
building instances. They are called "semantic components" throughout.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRANSFORM_DIR = REPO_ROOT / "datasets" / "transforms"
if str(TRANSFORM_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORM_DIR))

# ------------------------------------------------------------------ read-only source paths

ORIGINAL_ROOT = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")
CROPPED_ROOT = ORIGINAL_ROOT / "1. The cropped image data and raster labels"
SHAPEFILE_ROOT = ORIGINAL_ROOT / "2. The shape file of the whole images"
WHOLE_AREA_ROOT = ORIGINAL_ROOT / "3. The whole area image dataset"
HISTORICAL_CONVERTER = ORIGINAL_ROOT / "mask_to_yolo.py"

#: Candidate locations of the historical converted dataset (the path recorded inside
#: ``mask_to_yolo.py`` no longer exists; the surviving copy lives in the legacy project).
CONVERTED_CANDIDATES = (
    Path(r"C:\D\resources\WHU_YOLO_dataset"),
    REPO_ROOT.parent / "WHU_Building_Segment" / "dataset" / "WHU_YOLO_dataset",
    Path(r"C:\D\resources\project\WHU_Building_Segment\dataset\WHU_YOLO_dataset"),
)

LEGACY_PROJECTS = (
    REPO_ROOT.parent / "WHU_Building_Segment",
    Path(r"C:\D\resources\project\WHU_Building_Segment"),
)

CURRENT_ROOT = REPO_ROOT / "datasets" / "whu"

#: Historical conversion constants (exact).
THRESHOLD = 127
MIN_CONTOUR_AREA = 50.0
EPSILON_RATIO = 0.001
MIN_APPROX_VERTICES = 3

#: Original split directory names inside the cropped subtree.
SOURCE_SPLITS = ("train", "train_no", "test", "test_no")

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# ------------------------------------------------------------------ paths / provenance


def converted_root() -> Path | None:
    for candidate in CONVERTED_CANDIDATES:
        if (candidate / "images").is_dir():
            return candidate
    return None


def path_provenance() -> dict:
    """Record every resolved path and its existence, without inventing any."""

    return {
        "original_root": {
            "path": str(ORIGINAL_ROOT),
            "exists": ORIGINAL_ROOT.is_dir(),
            "cropped_subtree": {
                "path": str(CROPPED_ROOT),
                "exists": CROPPED_ROOT.is_dir(),
            },
            "shapefile_subtree": {
                "path": str(SHAPEFILE_ROOT),
                "exists": SHAPEFILE_ROOT.is_dir(),
            },
            "whole_area_subtree": {
                "path": str(WHOLE_AREA_ROOT),
                "exists": WHOLE_AREA_ROOT.is_dir(),
            },
            "historical_converter_script": {
                "path": str(HISTORICAL_CONVERTER),
                "exists": HISTORICAL_CONVERTER.is_file(),
            },
        },
        "converted_candidates": [
            {"path": str(candidate), "exists": (candidate / "images").is_dir()}
            for candidate in CONVERTED_CANDIDATES
        ],
        "converted_resolved": str(converted_root()) if converted_root() else None,
        "legacy_projects": [
            {"path": str(project), "exists": project.is_dir()} for project in LEGACY_PROJECTS
        ],
        "current_representation": {"path": str(CURRENT_ROOT), "exists": CURRENT_ROOT.is_dir()},
    }


# ------------------------------------------------------------------ mask I/O and thresholding


def read_semantic_label(path: Path) -> np.ndarray:
    """Read a raster label with the historical I/O order: OpenCV first, Pillow fallback."""

    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        from PIL import Image

        mask = np.array(Image.open(path).convert("L"))
    return mask


def read_semantic_label_pillow(path: Path) -> np.ndarray:
    """Pillow-only read, used to prove the I/O fallback does not change binary semantics."""

    from PIL import Image

    return np.array(Image.open(path).convert("L"))


def binarize(mask: np.ndarray, threshold: int = THRESHOLD) -> np.ndarray:
    """Historical binarization: ``cv2.threshold(mask, 127, 255, THRESH_BINARY)`` semantics."""

    import cv2

    _, binary = cv2.threshold(np.asarray(mask), int(threshold), 255, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def binary_foreground(binary: np.ndarray) -> np.ndarray:
    return np.asarray(binary) > 0


# ------------------------------------------------------------------ raw semantic components


@dataclass
class ComponentRecord:
    """One semantic connected component (NOT a verified physical building)."""

    component_id: int
    area_px: int
    bbox_xyxy_px: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    touches_image_border: bool
    width_px: int
    height_px: int


def raw_components(binary: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, list[ComponentRecord]]:
    """8-connected (primary) or 4-connected (sensitivity) components of a binary mask.

    Component ids start at 1; 0 is background, matching the current representation's
    convention so the two candidate sets are directly comparable.
    """

    import cv2

    n_labels, labels = cv2.connectedComponents(
        (np.asarray(binary) > 0).astype(np.uint8), connectivity=int(connectivity), ltype=cv2.CV_32S
    )
    height, width = labels.shape
    records: list[ComponentRecord] = []
    for component_id in range(1, int(n_labels)):
        mask = labels == component_id
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        touches = bool(x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1)
        records.append(
            ComponentRecord(
                component_id=component_id,
                area_px=int(mask.sum()),
                bbox_xyxy_px=(x0, y0, x1 + 1, y1 + 1),
                centroid_px=(float(xs.mean()), float(ys.mean())),
                touches_image_border=touches,
                width_px=x1 - x0 + 1,
                height_px=y1 - y0 + 1,
            )
        )
    return labels, records


# ------------------------------------------------------------------ historical contour pipeline


@dataclass
class ContourAudit:
    """Per-image result of reproducing the historical contour pipeline in memory."""

    n_contours_before: int = 0
    n_contours_kept: int = 0
    n_contours_removed: int = 0
    removed_contour_areas: list[float] = field(default_factory=list)          # cv2.contourArea
    removed_raster_pixels: list[int] = field(default_factory=list)            # rasterized area
    kept_contour_areas: list[float] = field(default_factory=list)
    kept_raster_pixels: list[int] = field(default_factory=list)
    approx_vertex_counts: list[int] = field(default_factory=list)
    dropped_below_three_vertices: int = 0


def historical_contours(binary: np.ndarray):
    """Exactly the historical call: ``cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)``."""

    import cv2

    contours, _hierarchy = cv2.findContours(
        np.asarray(binary).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def rasterize_contour(contour, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize one contour to a boolean mask (same fill routine as the current transform)."""

    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool)


def simplify_contour(contour):
    """``approxPolyDP(contour, 0.001 * arcLength(contour, True), True)``."""

    import cv2

    epsilon = EPSILON_RATIO * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True), float(epsilon)


def audit_contours(binary: np.ndarray) -> ContourAudit:
    """Reproduce the historical pipeline in memory and record every loss component."""

    import cv2

    shape = np.asarray(binary).shape
    audit = ContourAudit()
    contours = historical_contours(binary)
    audit.n_contours_before = len(contours)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        raster = int(rasterize_contour(contour, shape).sum())
        if area < MIN_CONTOUR_AREA:
            audit.n_contours_removed += 1
            audit.removed_contour_areas.append(area)
            audit.removed_raster_pixels.append(raster)
            continue
        approx, _epsilon = simplify_contour(contour)
        if len(approx) < MIN_APPROX_VERTICES:
            audit.dropped_below_three_vertices += 1
            audit.n_contours_removed += 1
            audit.removed_contour_areas.append(area)
            audit.removed_raster_pixels.append(raster)
            continue
        audit.n_contours_kept += 1
        audit.kept_contour_areas.append(area)
        audit.kept_raster_pixels.append(raster)
        audit.approx_vertex_counts.append(int(len(approx)))
    return audit


def external_contour_target(binary: np.ndarray) -> np.ndarray:
    """Rasterized union of ALL external contours (no area filter) — the RETR_EXTERNAL target."""

    import cv2

    shape = np.asarray(binary).shape
    mask = np.zeros(shape, dtype=np.uint8)
    contours = historical_contours(binary)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool)


def post_filter_target(binary: np.ndarray, min_area: float = MIN_CONTOUR_AREA) -> np.ndarray:
    """Rasterized union of external contours that survive the ``contourArea < 50`` filter."""

    import cv2

    shape = np.asarray(binary).shape
    mask = np.zeros(shape, dtype=np.uint8)
    kept = [c for c in historical_contours(binary) if float(cv2.contourArea(c)) >= float(min_area)]
    if kept:
        cv2.drawContours(mask, kept, -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool)


def simplified_polygon_target(binary: np.ndarray, min_area: float = MIN_CONTOUR_AREA) -> np.ndarray:
    """Rasterized union of the SIMPLIFIED polygons kept by the historical pipeline."""

    import cv2

    shape = np.asarray(binary).shape
    mask = np.zeros(shape, dtype=np.uint8)
    kept = []
    for contour in historical_contours(binary):
        if float(cv2.contourArea(contour)) < float(min_area):
            continue
        approx, _epsilon = simplify_contour(contour)
        if len(approx) < MIN_APPROX_VERTICES:
            continue
        kept.append(approx)
    if kept:
        cv2.drawContours(mask, kept, -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool)


def hole_analysis(binary: np.ndarray) -> dict:
    """RETR_EXTERNAL topology loss: interior holes of the semantic mask."""

    import cv2

    foreground = np.asarray(binary) > 0
    filled = external_contour_target(binary)
    holes = filled & ~foreground
    n_hole_regions, _labels = cv2.connectedComponents(holes.astype(np.uint8), connectivity=8)
    raw_labels, raw_records = raw_components(binary, connectivity=8)
    components_with_holes = 0
    for record in raw_records:
        component = raw_labels == record.component_id
        x0, y0, x1, y1 = record.bbox_xyxy_px
        if (filled[y0:y1, x0:x1] & ~component[y0:y1, x0:x1]).any():
            components_with_holes += 1
    return {
        "hole_pixels": int(holes.sum()),
        "hole_region_count": int(max(0, n_hole_regions - 1)),
        "components_with_holes": int(components_with_holes),
        "reconstruction_difference_pixels": int((filled & ~foreground).sum()),
        "filled_pixels": int(filled.sum()),
        "foreground_pixels": int(foreground.sum()),
    }


# ------------------------------------------------------------------ YOLO polygon labels


def parse_yolo_polygons(label_path: Path):
    """Parse one YOLO polygon label with the project's canonical parser (read-only)."""

    import polygon_to_component_map as P

    polygons, malformed, wrong_class = P.parse_yolo_polygon_label(Path(label_path))
    return polygons, {
        "malformed_lines": [int(value) for value in malformed],
        "wrong_class_lines": [int(value) for value in wrong_class],
    }


def rasterize_polygons(polygons, width: int, height: int) -> np.ndarray:
    """Rasterize parsed polygons with the same backend as the current representation."""

    import polygon_to_component_map as P

    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    for record in polygons:
        vertices_px = P.normalized_to_pixel(record.vertices_normalized, int(width), int(height))
        vertices_int, _n_oob = P.to_raster_vertices(vertices_px, int(width), int(height))
        sub = P.rasterize_polygon(vertices_int, int(width), int(height))
        mask |= sub.astype(np.uint8)
    return mask.astype(bool)


def polygon_vertex_clip_stats(polygons) -> dict:
    """Recoverable clipping evidence: normalized vertices outside [0, 1] before clipping."""

    out_of_range = 0
    total = 0
    for record in polygons:
        vertices = np.asarray(record.vertices_normalized, dtype=np.float64)
        total += int(vertices.size)
        out_of_range += int(((vertices < 0.0) | (vertices > 1.0)).sum())
    return {"vertices": total, "out_of_range_vertices": out_of_range}


# ------------------------------------------------------------------ masks / metrics


def iou(a, b) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def dice(a, b) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    denom = int(a.sum()) + int(b.sum())
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def precision_recall(prediction, target) -> tuple[float, float]:
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return float(precision), float(recall)


def boundary_displacement(left, right) -> dict:
    """Mean symmetric boundary distance (pixels) via distance transforms; cheap and exact."""

    from scipy import ndimage

    import cv2

    a = np.asarray(left).astype(bool)
    b = np.asarray(right).astype(bool)
    if not a.any() or not b.any():
        return {"mean_boundary_distance_px": None, "p95_boundary_distance_px": None}
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    boundary_a = a & ~cv2.erode(a.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary_b = b & ~cv2.erode(b.astype(np.uint8), kernel, iterations=1).astype(bool)
    distance_to_b = ndimage.distance_transform_edt(~boundary_b)
    distance_to_a = ndimage.distance_transform_edt(~boundary_a)
    forward = distance_to_b[boundary_a] if boundary_a.any() else np.array([0.0])
    backward = distance_to_a[boundary_b] if boundary_b.any() else np.array([0.0])
    both = np.concatenate([forward, backward])
    return {
        "mean_boundary_distance_px": float(both.mean()),
        "p95_boundary_distance_px": float(np.percentile(both, 95)),
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
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def count_distribution(values, boundaries) -> dict:
    counts: dict[str, int] = {}
    for index, low in enumerate(boundaries):
        high = boundaries[index + 1] if index + 1 < len(boundaries) else None
        label = f"{low}" if high is None else f"{low}-{high - 1}" if high - low > 1 else f"{low}"
        counts[label] = 0
    for value in values:
        for index, low in enumerate(boundaries):
            high = boundaries[index + 1] if index + 1 < len(boundaries) else None
            label = f"{low}" if high is None else f"{low}-{high - 1}" if high - low > 1 else f"{low}"
            if value >= low and (high is None or value < high):
                counts[label] += 1
                break
    return counts


def write_json(path: Path, payload: dict) -> None:
    """Deterministic JSON output (sorted keys, stable formatting)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    path.write_text(text + "\n", encoding="utf-8")
