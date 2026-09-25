"""Geometry primitives for building components.

Everything here operates on the **canonical component representation** produced
by ``datasets/transforms/polygon_to_component_map.py``:

    component-index map (uint8, 0 = background, k = component k)
    + metadata JSONL providing width / height / area / centroid / bbox

Design rules
------------
1. Geometry comes from the rasterized component pixel set, never from polygon
   vertex means or from the original raster labels.
2. Two distinct distances are provided and they are **not** interchangeable:

   ``centroid_distance``
       Euclidean distance between component centroids. Cheap, smooth, and easy
       to turn into a differentiable surrogate later.

   ``boundary_distance``
       Minimum distance between the two components' boundaries. This is what
       ``nearest`` / ``near`` / ``adjacent`` actually mean geometrically, so
       relation ground truth uses **boundary_distance**, not the centroid
       shortcut.

3. Nothing here mutates the dataset. Component maps are read, never written.

Terminology: a component is a building connected component (see
``docs/data_representation.md``), not a verified physical building instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

BACKGROUND_ID = 0

#: Image border sides, used for tile-relative direction analysis.
BORDER_SIDES = ("left", "right", "top", "bottom")


# --------------------------------------------------------------------------
# Component record
# --------------------------------------------------------------------------


@dataclass
class Component:
    """One building component with its geometry resolved for one image.

    ``centroid_px`` and ``bbox_xyxy_px`` are taken from the metadata JSONL,
    which itself derived them from the rasterized mask. They are not recomputed
    here, so that a single geometry source of truth is preserved.
    """

    component_id: int
    source_polygon_index: int
    area_px: int
    area_ratio: float
    centroid_px: tuple[float, float]
    bbox_xyxy_px: tuple[int, int, int, int]
    width_px: int
    height_px: int
    touches_image_border: bool
    continuous_polygon_area_px: float = 0.0

    @property
    def centroid_x(self) -> float:
        return self.centroid_px[0]

    @property
    def centroid_y(self) -> float:
        return self.centroid_px[1]

    @property
    def aspect_ratio(self) -> float:
        """width / height. ``inf`` for a zero-height component (defensive)."""
        if self.height_px <= 0:
            return float("inf")
        return self.width_px / self.height_px

    @property
    def equivalent_scale_px(self) -> float:
        """sqrt(area), a rotation-invariant size measure."""
        return float(np.sqrt(max(self.area_px, 0)))

    @property
    def fill_ratio(self) -> float:
        """area / bbox area (a compactness proxy)."""
        bbox_area = self.width_px * self.height_px
        if bbox_area <= 0:
            return 0.0
        return self.area_px / float(bbox_area)

    def bbox_extent_ratio(self, image_area: int) -> float:
        """Bounding-box area as a fraction of image area.

        A high value means the component spans a large part of the tile, which
        is one signal for a suspected merged blob.
        """
        if image_area <= 0:
            return 0.0
        return (self.width_px * self.height_px) / float(image_area)


@dataclass
class ImageGeometry:
    """All components of one image plus image-level context."""

    image_id: str
    split: str
    width: int
    height: int
    component_map_path: str
    components: list[Component]
    _map_cache: np.ndarray | None = field(default=None, repr=False)

    @property
    def image_area(self) -> int:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def center(self) -> tuple[float, float]:
        return ((self.width - 1) / 2.0, (self.height - 1) / 2.0)

    @property
    def n_components(self) -> int:
        return len(self.components)

    def by_id(self) -> dict[int, Component]:
        return {c.component_id: c for c in self.components}

    def get(self, component_id: int) -> Component:
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise KeyError(f"{self.image_id}: unknown component_id {component_id}")

    def load_map(self, root: Path) -> np.ndarray:
        """Load (and cache) this image's component-index map."""
        if self._map_cache is None:
            from dataset_access import read_component_map

            self._map_cache = read_component_map(root, self.component_map_path)
        return self._map_cache

    def mask_of(self, component_id: int, root: Path) -> np.ndarray:
        """Boolean mask of one component."""
        return self.load_map(root) == component_id


# --------------------------------------------------------------------------
# Metadata loading
# --------------------------------------------------------------------------


def component_from_metadata(entry: dict) -> Component:
    """Build a :class:`Component` from one metadata JSON entry."""

    bbox = entry["bbox_xyxy_px"]
    centroid = entry["centroid_px"]
    return Component(
        component_id=int(entry["component_id"]),
        source_polygon_index=int(entry["source_polygon_index"]),
        area_px=int(entry["area_px"]),
        area_ratio=float(entry["area_ratio"]),
        centroid_px=(float(centroid[0]), float(centroid[1])),
        bbox_xyxy_px=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
        width_px=int(entry["width_px"]),
        height_px=int(entry["height_px"]),
        touches_image_border=bool(entry["touches_image_border"]),
        continuous_polygon_area_px=float(entry.get("continuous_polygon_area_px") or 0.0),
    )


def image_geometry_from_record(record: dict) -> ImageGeometry:
    """Build an :class:`ImageGeometry` from one metadata JSONL record."""

    return ImageGeometry(
        image_id=str(record["image_id"]),
        split=str(record["split"]),
        width=int(record["width"]),
        height=int(record["height"]),
        component_map_path=str(record["component_map"]),
        components=[component_from_metadata(e) for e in record["components"]],
    )


def iter_metadata(root: Path, split: str) -> Iterator[dict]:
    """Yield metadata records for one split."""

    path = root / "metadata" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_image_geometry(root: Path, split: str) -> Iterator[ImageGeometry]:
    """Yield :class:`ImageGeometry` for every image in a split."""

    for record in iter_metadata(root, split):
        yield image_geometry_from_record(record)


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------


def centroid_distance(a: Component, b: Component) -> float:
    """Euclidean distance between two component centroids, in pixels."""

    return float(
        np.hypot(a.centroid_px[0] - b.centroid_px[0], a.centroid_px[1] - b.centroid_px[1])
    )


def subject_to_object_delta(
    subject: Component, object_: Component
) -> tuple[float, float]:
    """Centroid offset from ``subject`` to ``object``, in pixels: ``object - subject``.

    Returns ``(dx_subject_to_object, dy_subject_to_object)`` where::

        dx_subject_to_object = object.centroid_x - subject.centroid_x
        dy_subject_to_object = object.centroid_y - subject.centroid_y

    Sign meaning (image coordinates: x grows right, **y grows downward**):

    * ``dx > 0``  -> the object is to the RIGHT of the subject
    * ``dx < 0``  -> the object is to the LEFT of the subject
    * ``dy > 0``  -> the object is BELOW the subject
    * ``dy < 0``  -> the object is ABOVE the subject

    This function replaces the earlier ambiguous ``centroid_delta(a, b)``. The
    name now states the argument order explicitly so the sign cannot be
    misread. Relation evaluation uses it as
    ``subject_to_object_delta(subject, object)``.
    """

    return (
        float(object_.centroid_px[0] - subject.centroid_px[0]),
        float(object_.centroid_px[1] - subject.centroid_px[1]),
    )


def subject_minus_object_delta(
    subject: Component, object_: Component
) -> tuple[float, float]:
    """Centroid offset ``subject - object``, in pixels.

    Provided only for symmetry with :func:`subject_to_object_delta`; the
    relation engine does not use it. Sign meaning is the mirror image::

        dx > 0 -> the subject is to the RIGHT of the object
        dx < 0 -> the subject is to the LEFT of the object
        dy > 0 -> the subject is BELOW the object
        dy < 0 -> the subject is ABOVE the object
    """

    return (
        float(subject.centroid_px[0] - object_.centroid_px[0]),
        float(subject.centroid_px[1] - object_.centroid_px[1]),
    )


def bbox_gap(a: Component, b: Component) -> float:
    """Gap between two axis-aligned bounding boxes; 0 if they intersect.

    This is a *lower* bound on the point-to-point distance between the boxes,
    and therefore also on the components' region distance, but it is an
    **upper bound on the empty-pixel gap** returned by :func:`boundary_distance`
    (which subtracts 1 and is smaller than the corner-to-corner distance).

    Practical use: ``bbox_gap`` is cheap (no image access) and is used to order
    candidate pairs. It must NOT be used as an admissible lower bound for
    pruning ``nearest``; see :func:`nearest_prune_lower_bound`.
    """

    ax0, ay0, ax1, ay1 = a.bbox_xyxy_px
    bx0, by0, bx1, by1 = b.bbox_xyxy_px
    dx = max(0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0, max(ay0, by0) - min(ay1, by1))
    return float(np.hypot(dx, dy))


def nearest_prune_upper_bound(a: Component, b: Component) -> float:
    """Upper bound on ``boundary_distance(a, b)``, usable to prune safely.

    ``gap(A, B) <= d(A, B) <= bbox_corner_distance(A, B)``, because for any two
    points ``p`` in box A and ``q`` in box B the triangle inequality through the
    nearest box corners gives ``||p - q|| <= ||boxA_corner - boxB_corner||``.

    Therefore, once an exact gap ``g`` has been found, every candidate whose
    ``nearest_prune_upper_bound`` is ``<= g`` cannot beat it and may be skipped.
    """

    return bbox_gap(a, b)


def component_mask(component_map: np.ndarray, component_id: int) -> np.ndarray:
    """Boolean mask for one component id."""
    return component_map == component_id


def component_boundary(component_map: np.ndarray, component_id: int) -> np.ndarray:
    """Boundary pixels of one component, as a boolean mask.

    A pixel is on the boundary when it belongs to the component and at least
    one of its 4-neighbours is background or outside the image. 8-connectivity
    is deliberately not used: with 8-connectivity a pixel that only touches the
    background diagonally would still count as boundary, which inflates the
    boundary and shifts ``boundary_distance`` by up to ``sqrt(2)``.

    Pixels at the image edge count as boundary, because that is where the
    component is clipped. Note that this is a property of the *clipped* shape;
    a component cut by the tile edge really does have a boundary there.
    """

    import cv2

    mask = component_mask(component_map, component_id).astype(np.uint8)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask.astype(bool) & (~eroded.astype(bool))


def boundary_distance(
    component_map: np.ndarray,
    id_a: int,
    id_b: int,
) -> float:
    """Minimum distance between two components, in pixels.

    This is the geometric meaning of ``nearest`` / ``near`` / ``adjacent`` and
    is the distance used for relation ground truth. Centroid distance is
    deliberately not used as a substitute: it misranks elongated and L-shaped
    buildings, which this dataset is full of.

    Convention
    ----------
    ``boundary_distance`` is the **gap** between the two components, in pixels,
    defined as the number of empty pixels separating them::

        gap(A, B) = 0                    if the components touch or overlap
        gap(A, B) = d(A, B) - 1          otherwise

    where ``d(A, B)`` is the distance between the two pixel sets, each pixel
    represented by its centre::

        d(A, B) = min_{p in A, q in B} || centre(p) - centre(q) ||

    ``d`` is computed exactly (not sampled) by evaluating, at every pixel of A,
    the Euclidean distance transform of B's boundary and taking the minimum.

    Consequences, all of which are the desired semantics:

    * two components sharing an edge give exactly ``0.0``;
    * two components separated by one empty pixel give exactly ``1.0``;
    * two components separated by N empty pixels give exactly ``N.0`` in an
      axis-aligned arrangement;
    * the value is invariant to the components' sizes, unlike centroid distance.

    Accuracy note: for a diagonal arrangement the value is
    ``sqrt(2) - 1 ~= 0.414`` rather than the geometric gap of ``1.0``, i.e. the
    measure can under-report a diagonal gap by up to about half a pixel. This
    does not affect relation semantics: ``nearest`` depends on the *ordering* of
    distances, and the identical convention applies to every pair.

    Returns ``inf`` when either component is empty.
    """

    from scipy import ndimage

    if not component_mask(component_map, id_a).any():
        return float("inf")
    if not component_mask(component_map, id_b).any():
        return float("inf")

    # Exact: distance to B's boundary, sampled at A's boundary.
    boundary_a = component_boundary(component_map, id_a)
    boundary_b = component_boundary(component_map, id_b)
    if not boundary_a.any() or not boundary_b.any():
        return float("inf")

    distance_to_b = ndimage.distance_transform_edt(~boundary_b)
    region_distance = float(distance_to_b[boundary_a].min())
    if region_distance <= 0.0:
        return 0.0
    return max(0.0, region_distance - 1.0)


def component_box_distance(
    component_map: np.ndarray,
    id_a: int,
    id_b: int,
    pad: int = 4,
) -> float:
    """Gap between two components, computed inside a local window.

    Equivalent to :func:`boundary_distance` (including the ``- 1`` gap
    convention) but restricts the distance transform to the union of the two
    components' bounding boxes plus ``pad``. Using ``pad >= 2`` guarantees the
    unrestricted minimum is still found, because that minimum is attained
    between boundary pixels of A and B, both of which lie inside this window.

    Returns ``0.0`` for touching components and ``inf`` if either is empty.
    """

    from scipy import ndimage

    mask_a = component_mask(component_map, id_a)
    mask_b = component_mask(component_map, id_b)
    if not mask_a.any() or not mask_b.any():
        return float("inf")

    rows = np.nonzero(mask_a.any(axis=1) | mask_b.any(axis=1))[0]
    cols = np.nonzero(mask_a.any(axis=0) | mask_b.any(axis=0))[0]
    y0 = max(0, int(rows.min()) - pad)
    y1 = min(component_map.shape[0], int(rows.max()) + pad + 1)
    x0 = max(0, int(cols.min()) - pad)
    x1 = min(component_map.shape[1], int(cols.max()) + pad + 1)

    window_a = mask_a[y0:y1, x0:x1]
    window_b = mask_b[y0:y1, x0:x1]

    # Distance transform of the complement of B, sampled at A's pixels.
    distance_to_b = ndimage.distance_transform_edt(~window_b)
    samples = distance_to_b[window_a]
    if samples.size == 0:
        return float("inf")
    region_distance = float(samples.min())
    if region_distance <= 0.0:
        return 0.0
    return max(0.0, region_distance - 1.0)


def boundary_distance_between(
    component_map: np.ndarray,
    a: Component,
    b: Component,
) -> float:
    """Convenience wrapper around :func:`boundary_distance`."""

    return boundary_distance(component_map, a.component_id, b.component_id)


def distance_to_image_border(
    component_map: np.ndarray, component_id: int
) -> float:
    """Number of empty pixels between a component and the image edge.

    Defined directly from the component's extent::

        distance = min(x_min, y_min, (width - 1) - x_max, (height - 1) - y_max)

    This is the count of empty pixel columns/rows separating the component from
    the tile edge. ``0.0`` means the component reaches the border, i.e. it is
    clipped by the tile.

    Note this deliberately uses the component's full pixel set, not
    :func:`component_boundary`: a 4-neighbour erosion leaves pixels on the image
    edge classified as interior, so the boundary mask alone cannot express
    "clipped by the tile".
    """

    mask = component_mask(component_map, component_id)
    if not mask.any():
        return float("inf")

    height, width = component_map.shape
    ys, xs = np.nonzero(mask)
    return float(
        min(
            int(xs.min()),
            int(ys.min()),
            (width - 1) - int(xs.max()),
            (height - 1) - int(ys.max()),
        )
    )


# --------------------------------------------------------------------------
# Extreme (tile-relative rank) helpers
# --------------------------------------------------------------------------


def rank_by(
    components: Sequence[Component],
    key,
    reverse: bool = False,
) -> list[Component]:
    """Sort components by a key, deterministically.

    Ties are broken by ``component_id`` so the ordering never depends on input
    order or an unstable sort.
    """

    return sorted(components, key=lambda c: (key(c), c.component_id), reverse=reverse)


def extreme_pair(
    components: Sequence[Component],
    key,
    reverse: bool = False,
) -> tuple[Component, Component, float] | None:
    """Return ``(best, second, margin)`` for an extreme relation.

    ``margin`` is ``key(best) - key(second)`` in the ordering direction, so it
    is non-negative. Returns ``None`` when fewer than two components are given.
    """

    ordered = rank_by(components, key, reverse=reverse)
    if len(ordered) < 2:
        return None
    best, second = ordered[0], ordered[1]
    margin = abs(key(best) - key(second))
    return best, second, float(margin)
