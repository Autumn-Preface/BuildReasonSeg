"""Tests for the geometry primitives.

Covers centroid, bbox, boundary distance, centroid distance and normalization
on a synthetic component map with known ground truth.

Run with pytest, or directly::

    python tests/test_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))

import geometry as G  # noqa: E402


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def make_map(width: int = 64, height: int = 64, boxes=None) -> np.ndarray:
    """Build a component map from ``{component_id: (x0, y0, x1, y1)}`` boxes."""

    component_map = np.zeros((height, width), dtype=np.uint8)
    for component_id, (x0, y0, x1, y1) in (boxes or {}).items():
        component_map[y0:y1, x0:x1] = component_id
    return component_map


def component_from_map(component_map: np.ndarray, component_id: int) -> G.Component:
    mask = component_map == component_id
    ys, xs = np.nonzero(mask)
    height, width = component_map.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return G.Component(
        component_id=component_id,
        source_polygon_index=component_id - 1,
        area_px=int(mask.sum()),
        area_ratio=float(mask.sum()) / (width * height),
        centroid_px=(float(xs.mean()), float(ys.mean())),
        bbox_xyxy_px=(x0, y0, x1, y1),
        width_px=x1 - x0 + 1,
        height_px=y1 - y0 + 1,
        touches_image_border=bool(x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1),
    )


BOXES = {
    1: (2, 2, 12, 12),     # 10x10 at top-left
    2: (40, 4, 60, 24),    # 20x20 top-right
    3: (2, 40, 10, 60),    # 8x20 bottom-left
}


def _fixture():
    component_map = make_map(boxes=BOXES)
    components = [component_from_map(component_map, cid) for cid in sorted(BOXES)]
    return component_map, components


# --------------------------------------------------------------------------
# Geometry: centroid and bbox
# --------------------------------------------------------------------------


def test_centroid_matches_pixel_mean():
    component_map, components = _fixture()
    for component in components:
        mask = component_map == component.component_id
        ys, xs = np.nonzero(mask)
        assert component.centroid_x == float(xs.mean())
        assert component.centroid_y == float(ys.mean())
        # Centroid must lie inside the component's own bbox.
        x0, y0, x1, y1 = component.bbox_xyxy_px
        assert x0 <= component.centroid_x <= x1
        assert y0 <= component.centroid_y <= y1
    print("  [geometry] centroid == pixel mean OK")


def test_bbox_and_size():
    component_map, components = _fixture()
    for component in components:
        expected = BOXES[component.component_id]
        x0, y0, x1, y1 = component.bbox_xyxy_px
        assert (x0, y0) == (expected[0], expected[1])
        assert (x1, y1) == (expected[2] - 1, expected[3] - 1)
        assert component.width_px == expected[2] - expected[0]
        assert component.height_px == expected[3] - expected[1]
        assert component.area_px == component.width_px * component.height_px
    print("  [geometry] bbox / width / height / area OK")


def test_aspect_ratio_and_scale():
    component_map, components = _fixture()
    by_id = {c.component_id: c for c in components}
    assert abs(by_id[1].aspect_ratio - 1.0) < 1e-9            # 10x10
    assert abs(by_id[2].aspect_ratio - 1.0) < 1e-9            # 20x20
    assert abs(by_id[3].aspect_ratio - (8 / 20)) < 1e-9       # 8x20
    assert abs(by_id[2].equivalent_scale_px - 20.0) < 1e-9    # sqrt(400)
    assert abs(by_id[1].fill_ratio - 1.0) < 1e-9              # rectangle fills bbox
    print("  [geometry] aspect ratio / equivalent scale / fill ratio OK")


# --------------------------------------------------------------------------
# Geometry: distances
# --------------------------------------------------------------------------


def test_centroid_distance_known_value():
    component_map, components = _fixture()
    by_id = {c.component_id: c for c in components}
    a, b = by_id[1], by_id[2]
    expected = float(np.hypot(a.centroid_x - b.centroid_x, a.centroid_y - b.centroid_y))
    assert abs(G.centroid_distance(a, b) - expected) < 1e-9
    # Symmetric.
    assert abs(G.centroid_distance(a, b) - G.centroid_distance(b, a)) < 1e-9
    print("  [geometry] centroid distance OK")


def test_offset_sign_convention():
    """Pin the sign convention of both offset helpers.

    ``subject_to_object_delta`` is ``object - subject`` and is the direction the
    relation engine reasons about; ``subject_minus_object_delta`` is its mirror.
    """

    component_map, components = _fixture()
    by_id = {c.component_id: c for c in components}
    a, b = by_id[1], by_id[2]   # 1 is left of and slightly above 2

    dx, dy = G.subject_to_object_delta(a, b)
    assert dx > 0, "object is to the RIGHT of subject -> dx > 0"
    assert dy > 0, "object is BELOW subject -> dy > 0 (image y grows downward)"

    sdx, sdy = G.subject_minus_object_delta(a, b)
    assert sdx < 0, "subject is to the LEFT of object -> dx < 0"
    assert sdy < 0, "subject is ABOVE object -> dy < 0"

    # The two helpers must be exact mirrors.
    assert abs(dx + sdx) < 1e-9
    assert abs(dy + sdy) < 1e-9
    # And anti-symmetric under argument swap.
    r_dx, r_dy = G.subject_to_object_delta(b, a)
    assert abs(dx + r_dx) < 1e-9 and abs(dy + r_dy) < 1e-9
    print("  [geometry] offset sign convention OK")


def test_boundary_distance_axis_aligned_exact():
    """Two axis-aligned rectangles: gap is the number of empty columns."""

    component_map = make_map(boxes={1: (2, 2, 12, 12), 2: (20, 2, 30, 12)})
    a = component_from_map(component_map, 1)
    b = component_from_map(component_map, 2)
    # A covers x[2..11], B covers x[20..29]; gap = 20 - 11 - 1 = 8 empty columns.
    expected = float(20 - 11 - 1)
    distance = G.boundary_distance_between(component_map, a, b)
    assert abs(distance - expected) < 1e-9, f"expected {expected}, got {distance}"
    # Local-window variant must agree exactly.
    assert abs(G.component_box_distance(component_map, a.component_id, b.component_id) - distance) < 1e-9
    # Symmetric.
    assert abs(G.boundary_distance_between(component_map, b, a) - distance) < 1e-9
    print(f"  [geometry] boundary gap exact on axis-aligned case OK ({distance})")


def test_component_distance_counts_gap_pixels():
    """Adjacent / one-apart / three-apart must be distinguishable and ordered."""

    boxes = {1: (2, 2, 10, 10), 2: (10, 2, 18, 10)}   # touching
    component_map = make_map(boxes=boxes)
    assert G.component_box_distance(component_map, 1, 2) == 0.0

    boxes = {1: (2, 2, 10, 10), 2: (11, 2, 19, 10)}   # 1 empty column
    component_map = make_map(boxes=boxes)
    d1 = G.component_box_distance(component_map, 1, 2)

    boxes = {1: (2, 2, 10, 10), 2: (13, 2, 21, 10)}   # 3 empty columns
    component_map = make_map(boxes=boxes)
    d3 = G.component_box_distance(component_map, 1, 2)

    assert d1 == 1.0, d1
    assert d3 == 3.0, d3
    assert d1 < d3
    print(f"  [geometry] gap pixels counted correctly OK (1 empty -> {d1}, 3 empty -> {d3})")


def test_boundary_distance_diagonal_exact():
    """Diagonal offset: gap is the pixel-centre distance minus one."""

    component_map = make_map(boxes={1: (0, 0, 10, 10), 2: (11, 11, 21, 21)})
    a = component_from_map(component_map, 1)
    b = component_from_map(component_map, 2)
    # A covers x[0..9] y[0..9]; B covers x[11..20] y[11..20].
    # Nearest pixel centres are (9,9) and (11,11) -> hypot(2,2) = 2.8284.
    # The documented convention subtracts 1 pixel of extent per pixel => 1.8284.
    expected = float(np.hypot(2, 2)) - 1.0
    distance = G.boundary_distance_between(component_map, a, b)
    assert abs(distance - expected) < 1e-6, f"expected {expected}, got {distance}"
    assert abs(G.component_box_distance(component_map, 1, 2) - distance) < 1e-6
    # The documented under-report for a diagonal arrangement: the true
    # continuous gap is 1.0, and the measure gives 1.8284.
    assert 1.0 <= distance < 2.0
    print(f"  [geometry] boundary gap on diagonal case OK ({distance:.4f}, true gap 1.0)")

def test_boundary_distance_zero_when_touching():
    """Two components sharing an edge must give exactly 0.0."""

    component_map = make_map(boxes={1: (2, 2, 12, 12), 2: (12, 2, 22, 12)})
    a = component_from_map(component_map, 1)
    b = component_from_map(component_map, 2)
    assert G.boundary_distance_between(component_map, a, b) == 0.0
    assert G.component_box_distance(component_map, 1, 2) == 0.0
    print("  [geometry] touching components give distance 0 OK")


def test_boundary_distance_differs_from_centroid_distance():
    """The two distances must not be conflated.

    An elongated component can be close at its boundary yet far at its centroid,
    which is exactly why relations use boundary distance.
    """

    # A is a long thin bar; B sits right beside its middle.
    component_map = make_map(width=120, height=60, boxes={1: (2, 28, 100, 32), 2: (60, 40, 70, 50)})
    a = component_from_map(component_map, 1)
    b = component_from_map(component_map, 2)
    boundary = G.boundary_distance_between(component_map, a, b)
    centroid = G.centroid_distance(a, b)
    assert boundary < centroid, "boundary distance must be below centroid distance here"
    # A's bottom edge is y=31, B's top edge is y=40 -> 8 empty rows.
    assert abs(boundary - 8.0) < 1e-9, boundary
    assert centroid > 10.0
    print(f"  [geometry] boundary ({boundary}) != centroid ({centroid:.2f}) OK")


def test_component_box_distance_matches_full_map():
    """The local-window optimisation must equal the unrestricted computation."""

    component_map, components = _fixture()
    for i in range(len(components)):
        for j in range(len(components)):
            if i == j:
                continue
            full = G.boundary_distance_between(component_map, components[i], components[j])
            local = G.component_box_distance(
                component_map, components[i].component_id, components[j].component_id
            )
            assert abs(full - local) < 1e-9, (
                f"{components[i].component_id}-{components[j].component_id}: full={full} local={local}"
            )
    print("  [geometry] local-window distance == full-map distance OK")


def test_bbox_gap_upper_bounds_the_gap():
    """``bbox_gap`` is an UPPER bound on the empty-pixel gap.

    This is the direction that matters for the ``nearest`` pruning rule: a pair
    whose bbox gap already exceeds the best exact gap cannot win. The reverse
    is NOT true, which is why bbox gap is never used as a lower bound.
    """

    component_map, components = _fixture()
    for i in range(len(components)):
        for j in range(len(components)):
            if i == j:
                continue
            upper = G.nearest_prune_upper_bound(components[i], components[j])
            gap = G.boundary_distance_between(component_map, components[i], components[j])
            assert gap <= upper + 1e-9, (
                f"gap {gap} must not exceed the pruning upper bound {upper}"
            )
    print("  [geometry] bbox gap is an upper bound on the gap OK")


def test_distance_to_image_border():
    """Border distance: 0 when clipped by the tile, exact gap otherwise."""

    component_map = make_map(boxes={1: (5, 5, 15, 15), 4: (0, 0, 10, 10)})
    inside = component_from_map(component_map, 1)
    corner = component_from_map(component_map, 4)
    # Component 1's left face is at x=5, so 5 empty columns separate it.
    assert G.distance_to_image_border(component_map, inside.component_id) == 5.0
    assert G.distance_to_image_border(component_map, corner.component_id) == 0.0
    # A component one pixel in from the edge.
    component_map2 = make_map(boxes={9: (1, 10, 8, 20)})
    assert G.distance_to_image_border(component_map2, 9) == 1.0
    print("  [geometry] distance to image border OK")


# --------------------------------------------------------------------------
# Geometry: normalization
# --------------------------------------------------------------------------


def test_normalization_consistency():
    """Normalized coordinates must be recoverable and within [0, 1]."""

    component_map, components = _fixture()
    width = height = 64
    for component in components:
        # Metadata construction normalizes by width/height.
        assert 0.0 <= component.centroid_x / width <= 1.0
        assert 0.0 <= component.centroid_y / height <= 1.0
        for value in component.bbox_xyxy_px:
            assert 0 <= value <= 63
    print("  [geometry] normalization bounds OK")


def test_bbox_extent_ratio_and_area_ratio():
    component_map, components = _fixture()
    image_area = 64 * 64
    by_id = {c.component_id: c for c in components}
    b = by_id[2]
    assert abs(b.bbox_extent_ratio(image_area) - (20 * 20) / image_area) < 1e-12
    assert abs(b.area_ratio - (20 * 20) / image_area) < 1e-12
    print("  [geometry] bbox extent ratio / area ratio OK")


# --------------------------------------------------------------------------
# Geometry: extreme helpers
# --------------------------------------------------------------------------


def test_rank_by_is_deterministic_on_ties():
    """Ties must break by component_id, not by input order."""

    # Two components with the SAME area but different ids and positions, so
    # both actually exist in the map (overlapping boxes would erase one).
    component_map = make_map(boxes={7: (2, 2, 12, 12), 3: (30, 30, 40, 40)})
    components = [component_from_map(component_map, cid) for cid in (7, 3)]
    assert components[0].area_px == components[1].area_px
    ordered = G.rank_by(components, lambda c: c.area_px)
    assert [c.component_id for c in ordered] == [3, 7]
    # Reversed input order must give the same result.
    ordered2 = G.rank_by(list(reversed(components)), lambda c: c.area_px)
    assert [c.component_id for c in ordered2] == [3, 7]
    print("  [geometry] deterministic tie-break OK")


def test_extreme_pair_margin():
    component_map, components = _fixture()
    result = G.extreme_pair(components, lambda c: c.centroid_x)
    assert result is not None
    best, second, margin = result
    assert best.centroid_x <= second.centroid_x
    assert margin >= 0
    assert abs(margin - abs(best.centroid_x - second.centroid_x)) < 1e-9
    assert G.extreme_pair(components[:1], lambda c: c.centroid_x) is None
    print("  [geometry] extreme_pair margin OK")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("centroid", test_centroid_matches_pixel_mean),
        ("bbox/size", test_bbox_and_size),
        ("aspect/scale", test_aspect_ratio_and_scale),
        ("centroid_distance", test_centroid_distance_known_value),
        ("offset_sign", test_offset_sign_convention),
        ("boundary_axis", test_boundary_distance_axis_aligned_exact),
        ("gap_pixels", test_component_distance_counts_gap_pixels),
        ("boundary_diagonal", test_boundary_distance_diagonal_exact),
        ("boundary_touch", test_boundary_distance_zero_when_touching),
        ("boundary_vs_centroid", test_boundary_distance_differs_from_centroid_distance),
        ("box_distance_equiv", test_component_box_distance_matches_full_map),
        ("bbox_gap_upper_bound", test_bbox_gap_upper_bounds_the_gap),
        ("border_distance", test_distance_to_image_border),
        ("normalization", test_normalization_consistency),
        ("ratios", test_bbox_extent_ratio_and_area_ratio),
        ("rank_ties", test_rank_by_is_deterministic_on_ties),
        ("extreme_pair", test_extreme_pair_margin),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} geometry checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
