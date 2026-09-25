"""Tests for the spatial relation engine.

Builds synthetic component maps with known geometry and asserts the documented
behaviour of every v1 relation, plus symmetry, ambiguity handling, border
semantics and the merge heuristic.

Important fixture constraint
----------------------------
``suspected_large_merge`` fires when a component's bounding box covers more than
20% of the tile. Test components that are meant to be *normal* must therefore
stay well under that, or they will be silently excluded from size and nearest
relations. Canvases here are 300x300; a 120x120 component is 16% (fine) while a
250x250 component is 69% (triggers the heuristic on purpose).

Run with pytest, or directly::

    python tests/test_relations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))

import geometry as G  # noqa: E402
import relations as R  # noqa: E402
import thresholds as T  # noqa: E402
from component_quality import classify_image  # noqa: E402

CONFIG = T.load_config()

CANVAS = 300
PLAIN_COMPONENT = 120 * 120          # 16.0% of a 300x300 tile -> not a merge
MERGE_COMPONENT_MIN_EXTENT = 0.20    # config threshold, asserted below


def make_image(boxes: dict[int, tuple[int, int, int, int]], width: int = CANVAS, height: int = CANVAS):
    """Build ``(component_map, ImageGeometry)`` from id -> (x0, y0, x1, y1)."""

    component_map = np.zeros((height, width), dtype=np.uint8)
    for component_id, (x0, y0, x1, y1) in boxes.items():
        component_map[y0:y1, x0:x1] = component_id

    components = []
    for component_id in sorted(boxes):
        mask = component_map == component_id
        assert mask.any(), f"component {component_id} is empty (boxes overlap?)"
        ys, xs = np.nonzero(mask)
        cx0, cy0, cx1, cy1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        components.append(
            G.Component(
                component_id=component_id,
                source_polygon_index=component_id - 1,
                area_px=int(mask.sum()),
                area_ratio=float(mask.sum()) / (width * height),
                centroid_px=(float(xs.mean()), float(ys.mean())),
                bbox_xyxy_px=(cx0, cy0, cx1, cy1),
                width_px=cx1 - cx0 + 1,
                height_px=cy1 - cy0 + 1,
                touches_image_border=bool(
                    cx0 == 0 or cy0 == 0 or cx1 == width - 1 or cy1 == height - 1
                ),
            )
        )

    image = G.ImageGeometry(
        image_id="synthetic",
        split="train",
        width=width,
        height=height,
        component_map_path="unused",
        components=components,
    )
    return component_map, image


def _quality(image):
    return classify_image(image, CONFIG)


# --------------------------------------------------------------------------
# FROZEN predicate semantics (absolute coordinates, natural language)
#
# relation(subject, object) always means "the SUBJECT satisfies the relation
# with respect to the OBJECT":
#
#     left_of(A, B)  ==  "A is left of B"   ==  cx_A < cx_B
#     right_of(A, B) ==  "A is right of B"  ==  cx_A > cx_B
#     above(A, B)    ==  "A is above B"     ==  cy_A < cy_B
#     below(A, B)    ==  "A is below B"     ==  cy_A > cy_B
# --------------------------------------------------------------------------


def _pair_image(box_a, box_b, width=600, height=600):
    """Two component boxes in a large canvas, returned as (map, image)."""

    component_map, image = make_image({1: box_a, 2: box_b}, width=width, height=height)
    return component_map, image


def test_predicate_semantics_left_right_absolute_coordinates():
    """A centroid = (100, 200), B centroid = (300, 200).

    A is visually LEFT of B. The predicate must say so when called as
    left_of(A, B), and must refuse left_of(B, A).
    """

    # A centred at (100, 200); B centred at (300, 200).
    # Boxes are chosen so the pixel-index mean is exactly the stated centroid:
    # a box (x0, .., x1) inclusive covers indices x0..x1-1, mean = (x0+x1-1)/2.
    component_map, image = _pair_image((85, 185, 116, 216), (285, 185, 316, 216))
    quality = _quality(image)
    A, B = image.get(1), image.get(2)

    assert abs(A.centroid_x - 100.0) < 1e-6, A.centroid_x
    assert abs(A.centroid_y - 200.0) < 1e-6, A.centroid_y
    assert abs(B.centroid_x - 300.0) < 1e-6, B.centroid_x
    assert abs(B.centroid_y - 200.0) < 1e-6, B.centroid_y

    # A is visually left of B.
    assert R.evaluate_direction("left_of", image, A, B, CONFIG, quality).valid, \
        "left_of(A, B) must be valid: A is left of B"
    # A is not right of B.
    assert not R.evaluate_direction("right_of", image, A, B, CONFIG, quality).valid, \
        "right_of(A, B) must be invalid: A is not right of B"

    # B is visually right of A.
    assert R.evaluate_direction("right_of", image, B, A, CONFIG, quality).valid, \
        "right_of(B, A) must be valid: B is right of A"
    # B is not left of A.
    assert not R.evaluate_direction("left_of", image, B, A, CONFIG, quality).valid, \
        "left_of(B, A) must be invalid: B is not left of A"
    print("  [semantics] left_of / right_of absolute coordinates OK")


def test_predicate_semantics_above_below_absolute_coordinates():
    """A centroid = (200, 100), B centroid = (200, 300).

    A is visually ABOVE B (image y grows downward, so smaller y is higher).
    """

    component_map, image = _pair_image((185, 85, 216, 116), (185, 285, 216, 316))
    quality = _quality(image)
    A, B = image.get(1), image.get(2)

    assert abs(A.centroid_x - 200.0) < 1e-6
    assert abs(A.centroid_y - 100.0) < 1e-6
    assert abs(B.centroid_x - 200.0) < 1e-6
    assert abs(B.centroid_y - 300.0) < 1e-6

    # A is visually above B.
    assert R.evaluate_direction("above", image, A, B, CONFIG, quality).valid, \
        "above(A, B) must be valid: A is above B"
    assert not R.evaluate_direction("below", image, A, B, CONFIG, quality).valid, \
        "below(A, B) must be invalid: A is not below B"

    # B is visually below A.
    assert R.evaluate_direction("below", image, B, A, CONFIG, quality).valid, \
        "below(B, A) must be valid: B is below A"
    assert not R.evaluate_direction("above", image, B, A, CONFIG, quality).valid, \
        "above(B, A) must be invalid: B is not above A"
    print("  [semantics] above / below absolute coordinates OK")


def test_predicate_evidence_is_self_describing():
    """Evidence must let a reviewer read the inequality directly."""

    component_map, image = _pair_image((85, 185, 115, 215), (285, 185, 315, 215))
    quality = _quality(image)
    A, B = image.get(1), image.get(2)
    result = R.evaluate_direction("left_of", image, A, B, CONFIG, quality)
    assert result.valid, result.reason

    evidence = result.evidence
    # Absolute centroids present.
    assert evidence["subject_centroid_x"] < evidence["object_centroid_x"], (
        "left_of must have subject_centroid_x < object_centroid_x"
    )
    # Subject-relative offset present and negative on the dominant axis.
    assert evidence["subject_minus_object_dx"] < 0
    assert evidence["relationship"] == "subject is left of object"
    print("  [semantics] evidence is self-describing OK")


def test_round_trip_semantic_mapping():
    """Relation -> natural language -> geometry must agree for all four.

    This is the minimal semantic mapping test. The future instruction generator
    must use the same template so wording and logic cannot drift apart.
    """

    cases = [
        # (relation, subject box, object box, description, geometry predicate)
        ("left_of", (85, 185, 115, 215), (285, 185, 315, 215),
         "A is left of B", lambda s, o: s.centroid_x < o.centroid_x),
        ("right_of", (285, 185, 315, 215), (85, 185, 115, 215),
         "A is right of B", lambda s, o: s.centroid_x > o.centroid_x),
        ("above", (185, 85, 215, 115), (185, 285, 215, 315),
         "A is above B", lambda s, o: s.centroid_y < o.centroid_y),
        ("below", (185, 285, 215, 315), (185, 85, 215, 115),
         "A is below B", lambda s, o: s.centroid_y > o.centroid_y),
    ]
    for relation, box_a, box_b, expected_text, geometry_predicate in cases:
        component_map, image = _pair_image(box_a, box_b)
        quality = _quality(image)
        A, B = image.get(1), image.get(2)

        # 1. the relation is valid
        result = R.evaluate_direction(relation, image, A, B, CONFIG, quality)
        assert result.valid, f"{relation}: {result.reason}"
        # 2. it renders as the natural sentence
        text = R.describe_relation(relation, "A", "B")
        assert text == expected_text, f"{text!r} != {expected_text!r}"
        # 3. the geometry predicate that the sentence denotes is true
        assert geometry_predicate(A, B), (relation, A.centroid_px, B.centroid_px)
        # 4. the reverse reading must be false
        assert not geometry_predicate(B, A), f"{relation} must not hold in reverse"
    print("  [semantics] round-trip relation -> text -> geometry OK")


def test_left_of_and_right_of():
    """Frozen convention: ``left_of(A, B)`` means "A is left of B"."""

    component_map, image = make_image({1: (10, 120, 60, 180), 2: (200, 120, 260, 180)})
    quality = _quality(image)
    a, b = image.get(1), image.get(2)   # a is LEFT of b

    # a is visually left of b -> left_of(a, b) holds.
    left = R.evaluate_direction("left_of", image, a, b, CONFIG, quality)
    assert left.valid, left.reason
    assert left.subject == 1 and left.object == 2
    assert left.margin > 0
    # b is not left of a.
    assert not R.evaluate_direction("left_of", image, b, a, CONFIG, quality).valid

    # b is visually right of a -> right_of(b, a) holds.
    right = R.evaluate_direction("right_of", image, b, a, CONFIG, quality)
    assert right.valid, right.reason
    assert right.subject == 2 and right.object == 1
    print("  [relations] left_of / right_of OK")


def test_above_and_below():
    component_map, image = make_image({1: (120, 10, 180, 60), 2: (120, 200, 180, 260)})
    quality = _quality(image)
    a, b = image.get(1), image.get(2)   # a is ABOVE b

    # a is visually above b -> above(a, b) holds.
    above = R.evaluate_direction("above", image, a, b, CONFIG, quality)
    assert above.valid, above.reason
    assert not R.evaluate_direction("above", image, b, a, CONFIG, quality).valid

    # b is visually below a -> below(b, a) holds.
    below = R.evaluate_direction("below", image, b, a, CONFIG, quality)
    assert below.valid, below.reason
    print("  [relations] above / below OK")


def test_wrong_direction_rejected():
    component_map, image = make_image({1: (10, 120, 60, 180), 2: (200, 120, 260, 180)})
    quality = _quality(image)
    a, b = image.get(1), image.get(2)   # a is LEFT of b
    # a is left of b, so claiming a is right of b must fail with wrong_direction.
    result = R.evaluate_direction("right_of", image, a, b, CONFIG, quality)
    assert not result.valid
    assert result.reason == R.REASON_WRONG_DIRECTION, result.reason
    print("  [relations] wrong direction rejected OK")


def test_axis_not_dominant_rejected():
    """Nearly diagonal: dx and dy similar -> no clean left/right claim."""

    component_map, image = make_image({1: (10, 10, 60, 60), 2: (90, 80, 140, 130)})
    quality = _quality(image)
    a, b = image.get(1), image.get(2)
    # a is up-left of b, but only slightly -> neither left_of nor above holds.
    result = R.evaluate_direction("left_of", image, a, b, CONFIG, quality)
    assert not result.valid
    assert result.reason == R.REASON_AXIS_NOT_DOMINANT, result.reason
    vertical = R.evaluate_direction("above", image, a, b, CONFIG, quality)
    assert not vertical.valid
    print("  [relations] axis dominance rejection OK")


def test_margin_too_small_rejected():
    """A separation below the normalized margin must be rejected.

    Wide canvas so the components can be small in area while their centroids
    are still only ~13 px apart, i.e. 13/400 = 0.0325 < tau (0.04).
    """

    component_map, image = make_image(
        {1: (198, 190, 210, 202), 2: (211, 190, 223, 202)},
        width=400,
        height=400,
    )
    quality = _quality(image)
    a, b = image.get(1), image.get(2)
    dx = abs(a.centroid_x - b.centroid_x)
    assert dx / image.width < CONFIG.direction.tau, (dx, image.width)
    # a is visually left of b, but the margin is too small to accept.
    result = R.evaluate_direction("left_of", image, a, b, CONFIG, quality)
    assert not result.valid
    assert result.reason == R.REASON_MARGIN_TOO_SMALL, result.reason
    print(f"  [relations] minimum-margin rejection OK (dx_norm {dx / image.width:.4f})")


def test_axis_ratio_threshold_boundary():
    """A pair just inside the dominance rule passes."""

    component_map, image = make_image({1: (10, 10, 60, 60), 2: (110, 70, 160, 120)})
    quality = _quality(image)
    a, b = image.get(1), image.get(2)
    sdx, sdy = G.subject_minus_object_delta(a, b)
    ratio = abs(sdx) / abs(sdy)
    result = R.evaluate_direction("left_of", image, a, b, CONFIG, quality)
    assert result.valid, f"ratio {ratio:.3f} should pass alpha={CONFIG.direction.alpha}: {result.reason}"
    print(f"  [relations] dominance boundary pass OK (ratio {ratio:.3f})")


def test_direction_symmetry():
    """left_of(A,B) valid <=> right_of(B,A) valid; same for above/below."""

    boxes = {1: (10, 10, 60, 60), 2: (200, 20, 260, 70), 3: (15, 200, 65, 260)}
    component_map, image = make_image(boxes)
    quality = _quality(image)

    cases = [
        ("left_of", "right_of", 1, 2),
        ("right_of", "left_of", 2, 1),
        ("above", "below", 1, 3),
        ("below", "above", 3, 1),
    ]
    for relation, opposite, subject_id, object_id in cases:
        forward = R.evaluate_direction(relation, image, image.get(subject_id), image.get(object_id), CONFIG, quality)
        backward = R.evaluate_direction(opposite, image, image.get(object_id), image.get(subject_id), CONFIG, quality)
        assert forward.valid == backward.valid, (
            f"{relation}({subject_id},{object_id}) valid={forward.valid} but "
            f"{opposite}({object_id},{subject_id}) valid={backward.valid}"
        )
        if forward.valid:
            same = R.evaluate_direction(relation, image, image.get(object_id), image.get(subject_id), CONFIG, quality)
            assert not same.valid, f"{relation} held in both directions"
    print("  [relations] direction symmetry OK")


# --------------------------------------------------------------------------
# Extreme relations
# --------------------------------------------------------------------------


def test_extreme_relations():
    """Only 3 eligible components so each extreme has a clear winner."""

    boxes = {
        1: (5, 120, 55, 170),     # leftmost
        2: (200, 120, 260, 180),  # rightmost
        3: (120, 5, 180, 55),     # topmost
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)

    leftmost = R.evaluate_extreme("leftmost", image, CONFIG, quality)
    rightmost = R.evaluate_extreme("rightmost", image, CONFIG, quality)
    topmost = R.evaluate_extreme("topmost", image, CONFIG, quality)
    bottommost = R.evaluate_extreme("bottommost", image, CONFIG, quality)

    assert leftmost.valid and leftmost.subject == 1, (leftmost.valid, leftmost.subject)
    assert rightmost.valid and rightmost.subject == 2, (rightmost.valid, rightmost.subject)
    assert topmost.valid and topmost.subject == 3, (topmost.valid, topmost.subject)
    assert bottommost.valid and bottommost.subject == 2, (bottommost.valid, bottommost.subject)
    for result in (leftmost, rightmost, topmost, bottommost):
        assert result.margin >= CONFIG.extreme_margin_px
    print("  [relations] extreme relations OK")


def test_extreme_ambiguous_when_margin_small():
    """Two components with almost the same x must be ambiguous, not picked."""

    # centroid x differ by 1 px, far below the 4 px margin.
    component_map, image = make_image({1: (40, 10, 60, 40), 2: (41, 200, 61, 230)})
    quality = _quality(image)
    result = R.evaluate_extreme("leftmost", image, CONFIG, quality)
    assert not result.valid
    assert result.ambiguous, "expected ambiguous, not a hard rejection"
    assert result.reason == R.STATUS_AMBIGUOUS
    print("  [relations] extreme ambiguity detected OK")


# --------------------------------------------------------------------------
# Size relations
# --------------------------------------------------------------------------


def test_largest_and_smallest():
    """Normal geometry only: nothing exceeds the merge extent threshold."""

    boxes = {
        1: (10, 10, 130, 130),     # 120x120 = 14400 -> 16% extent, largest
        2: (150, 150, 200, 200),   # 50x50 = 2500
        3: (210, 210, 240, 240),   # 30x30 = 900 -> smallest
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)
    for component in image.components:
        flags = quality.flags(component.component_id)
        assert not flags.suspected_large_merge, component.component_id
        assert not flags.touches_image_border, component.component_id

    largest = R.evaluate_size_rank("largest", image, CONFIG, quality)
    smallest = R.evaluate_size_rank("smallest", image, CONFIG, quality)
    assert largest.valid and largest.subject == 1, largest.reason
    assert smallest.valid and smallest.subject == 3, smallest.reason
    print("  [relations] largest / smallest OK")


def test_size_rank_ambiguous_on_near_tie():
    boxes = {1: (10, 10, 60, 70), 2: (150, 150, 200, 210)}  # equal 50x60 areas
    component_map, image = make_image(boxes)
    quality = _quality(image)
    result = R.evaluate_size_rank("largest", image, CONFIG, quality)
    assert not result.valid
    assert result.ambiguous, "a near tie must be ambiguous"
    print("  [relations] size-rank ambiguity detected OK")


# --------------------------------------------------------------------------
# nearest
# --------------------------------------------------------------------------


def test_nearest_uses_boundary_distance():
    """The nearest target must be decided by boundary distance.

    Component 2 is close at the boundary but far at the centroid (it is tall);
    component 3 is farther at the boundary but closer at the centroid.
    """

    boxes = {
        1: (20, 130, 60, 170),    # anchor, 40x40
        2: (65, 20, 85, 280),     # tall bar: boundary gap 5, centroid 35 px away
        3: (100, 130, 140, 170),  # boundary gap 40, centroid 75 px away
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)

    result = R.evaluate_nearest(image, 1, CONFIG, quality, component_map)
    assert result.valid, result.reason
    assert result.object == 2, f"expected 2 (nearest by boundary), got {result.object}"
    assert result.evidence["distance_metric"] == "boundary_distance"

    anchor, tall, near = image.get(1), image.get(2), image.get(3)
    assert G.boundary_distance_between(component_map, anchor, tall) < G.centroid_distance(anchor, tall)
    assert G.boundary_distance_between(component_map, anchor, tall) < G.boundary_distance_between(component_map, anchor, near)
    print("  [relations] nearest uses boundary distance OK")


def test_nearest_ambiguous_on_near_tie():
    boxes = {
        1: (120, 120, 150, 150),  # anchor
        2: (154, 120, 184, 150),  # gap 3
        3: (120, 154, 150, 184),  # gap 3 as well
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)
    result = R.evaluate_nearest(image, 1, CONFIG, quality, component_map)
    assert not result.valid
    assert result.ambiguous, "near-equal distances must be ambiguous"
    print("  [relations] nearest ambiguity detected OK")


def test_nearest_respects_eligibility():
    """A component rejected by the nearest rule must not be chosen as target."""

    # Component 2 covers 27.6% of the tile -> suspected merge -> ineligible.
    # Canvas is 400x400 and component 3 sits clear of both component 2 and the
    # tile border (a border-touching target would itself be ineligible).
    # Layout verified: gap(1,3) = 20 px (nearest), gap(1,2) = 40 px.
    boxes = {
        1: (10, 10, 50, 50),
        2: (70, 90, 300, 300),     # triggers suspected_large_merge
        3: (70, 10, 110, 50),      # valid fallback target, not on the border
    }
    component_map, image = make_image(boxes, width=400, height=400)
    quality = _quality(image)
    assert quality.flags(2).suspected_large_merge
    assert not quality.flags(2).geometry_valid
    assert not quality.flags(3).touches_image_border
    assert not quality.flags(3).suspected_large_merge
    assert not quality.flags(3).tiny_component

    result = R.evaluate_nearest(image, 1, CONFIG, quality, component_map)
    assert result.valid, result.reason
    assert result.object == 3, (
        f"suspected-merge component 2 must be ineligible, got {result.object}"
    )
    # Only one admissible candidate remains, so the answer is unique and the
    # margin is not applicable. This must NOT be reported as ambiguous.
    assert not result.ambiguous
    assert result.evidence["n_scored"] == 1
    assert result.evidence["margin_px"] is None
    print("  [relations] nearest eligibility OK")


def test_nearest_single_candidate_is_valid_not_ambiguous():
    """One admissible candidate is a unique answer, so it is valid.

    Ambiguity in ``nearest`` means the distance margin is too small, not that
    the candidate set is small.
    """

    boxes = {
        1: (10, 10, 50, 50),          # anchor
        2: (70, 90, 300, 300),        # suspected merge -> ineligible
    }
    component_map, image = make_image(boxes, width=400, height=400)
    quality = _quality(image)
    eligible = quality.eligible_ids("nearest", CONFIG)
    assert eligible == [1], eligible
    result = R.evaluate_nearest(image, 1, CONFIG, quality, component_map)
    assert not result.valid
    assert result.reason == R.REASON_TOO_FEW_CANDIDATES, result.reason
    print("  [relations] nearest with no candidate rejected cleanly OK")


# --------------------------------------------------------------------------
# Per-relation eligibility (why there is no single global flag)
# --------------------------------------------------------------------------


def test_border_component_allowed_in_tile_relative_leftmost():
    """A border-clipped component may still be the leftmost of its tile."""

    boxes = {
        1: (0, 100, 40, 140),      # clipped by the left edge, clearly leftmost
        2: (200, 100, 260, 160),
        3: (100, 20, 150, 70),
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)
    assert quality.flags(1).touches_image_border

    leftmost = R.evaluate_extreme("leftmost", image, CONFIG, quality)
    assert leftmost.valid, leftmost.reason
    assert leftmost.subject == 1, "a border component must be allowed to be leftmost"

    # 1 is to the left of 2, so left_of(1, 2) must hold.
    relation = R.evaluate_direction("left_of", image, image.get(1), image.get(2), CONFIG, quality)
    assert relation.valid, relation.reason
    print("  [relations] border component allowed in tile-relative relations OK")


def test_border_component_excluded_from_largest():
    """A border-clipped component must NOT be eligible for largest."""

    boxes = {
        1: (0, 0, 130, 130),       # clipped by two edges, largest area (17% extent)
        2: (150, 150, 270, 270),   # 120x120
        3: (280, 280, 295, 295),
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)
    assert quality.flags(1).touches_image_border
    assert quality.flags(1).area_px > quality.flags(2).area_px

    largest = R.evaluate_size_rank("largest", image, CONFIG, quality)
    assert largest.valid, largest.reason
    assert largest.subject != 1, "border component must be excluded from largest"
    assert largest.subject == 2

    leftmost = R.evaluate_extreme("leftmost", image, CONFIG, quality)
    assert leftmost.valid and leftmost.subject == 1
    print("  [relations] border component excluded from largest OK")


def test_merge_heuristic_only_affects_specified_relations():
    """A suspected-merge flag must change size/nearest but not direction."""

    boxes = {
        1: (0, 0, 250, 250),       # suspected merge, 69% extent
        2: (255, 100, 290, 140),
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)
    assert quality.flags(1).suspected_large_merge

    assert 1 in quality.eligible_ids("left_of", CONFIG)
    assert 1 in quality.eligible_ids("leftmost", CONFIG)
    assert 1 not in quality.eligible_ids("largest", CONFIG)
    assert 1 not in quality.eligible_ids("smallest", CONFIG)
    assert 1 not in quality.eligible_ids("nearest", CONFIG)

    result = R.evaluate_size_rank("largest", image, CONFIG, quality)
    assert not result.valid
    assert result.reason == R.REASON_TOO_FEW_CANDIDATES
    print("  [relations] merge heuristic scope OK")


def test_tiny_component_allowed_as_largest_but_not_smallest():
    """The asymmetric eligibility rule for size relations."""

    boxes = {
        1: (5, 5, 8, 8),           # 3x3 = 9 px, below the tiny threshold
        2: (60, 60, 180, 180),     # 120x120, largest, 16% extent
        3: (200, 200, 260, 260),   # 60x60, smallest valid
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)
    assert quality.flags(1).tiny_component

    assert 1 in quality.eligible_ids("largest", CONFIG), "tiny must be allowed for largest"
    assert 1 not in quality.eligible_ids("smallest", CONFIG), "tiny must be excluded for smallest"

    largest = R.evaluate_size_rank("largest", image, CONFIG, quality)
    assert largest.valid and largest.subject == 2, largest.reason
    smallest = R.evaluate_size_rank("smallest", image, CONFIG, quality)
    assert smallest.valid and smallest.subject == 3, smallest.reason
    print("  [relations] tiny eligibility asymmetry OK")


def test_smallest_ratio_uses_larger_over_smaller():
    """Regression: ``smallest`` must be able to be valid.

    The separation ratio is always larger/smaller, so the same threshold means
    the same thing for ``largest`` and ``smallest``. Comparing rank1/rank2
    directly would give a ratio below 1 for ``smallest`` and it could never fire.
    """

    # Areas 14400 / 3600 / 361; component 3 must not touch the tile border,
    # because `smallest` excludes border-truncated components.
    boxes = {1: (60, 60, 180, 180), 2: (200, 200, 260, 260), 3: (280, 280, 299, 299)}
    component_map, image = make_image(boxes)
    quality = _quality(image)
    assert not quality.flags(3).touches_image_border
    smallest = R.evaluate_size_rank("smallest", image, CONFIG, quality)
    assert smallest.valid, f"smallest must be valid here, got {smallest.reason}"
    assert smallest.subject == 3, smallest.subject
    assert smallest.evidence["ratio"] >= CONFIG.size_rank.ratio_margin
    assert smallest.evidence["ratio_definition"] == "larger_area / smaller_area"
    assert smallest.evidence["ratio"] > 1.0, "ratio must exceed 1 for a valid rank gap"

    largest = R.evaluate_size_rank("largest", image, CONFIG, quality)
    assert largest.valid and largest.subject == 1
    assert largest.evidence["ratio_definition"] == "larger_area / smaller_area"
    print(f"  [relations] smallest ratio definition OK (ratio {smallest.evidence['ratio']:.3f})")


def test_no_such_component():
    component_map, image = make_image({1: (10, 10, 60, 60), 2: (200, 200, 260, 260)})
    quality = _quality(image)
    result = R.evaluate_nearest(image, 999, CONFIG, quality, component_map)
    assert not result.valid
    assert result.reason == R.REASON_NO_SUCH_COMPONENT
    print("  [relations] unknown component handled OK")


def test_single_component_image_is_rejected_cleanly():
    component_map, image = make_image({1: (10, 10, 60, 60)})
    quality = _quality(image)
    extreme = R.evaluate_extreme("leftmost", image, CONFIG, quality)
    assert not extreme.valid
    assert extreme.reason == R.REASON_TOO_FEW_CANDIDATES
    size = R.evaluate_size_rank("largest", image, CONFIG, quality)
    assert not size.valid
    assert size.reason == R.REASON_TOO_FEW_CANDIDATES
    print("  [relations] single-component image handled OK")


# --------------------------------------------------------------------------
# nearest border policy (FROZEN)
# --------------------------------------------------------------------------


def test_nearest_rejects_border_anchor():
    """A border-truncated component must NOT be a `nearest` anchor.

    `nearest` uses boundary_distance; a clipped component has an incomplete
    boundary, so its distances are unreliable.
    """

    boxes = {
        1: (0, 200, 40, 240),      # touches the left edge -> border
        2: (100, 200, 160, 260),   # interior
        3: (250, 200, 300, 250),   # interior
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)
    assert quality.flags(1).touches_image_border

    # The border component is still eligible for tile-relative relations.
    assert 1 in quality.eligible_ids("leftmost", CONFIG)
    assert 1 in quality.eligible_ids("left_of", CONFIG)

    result = R.evaluate_nearest(image, 1, CONFIG, quality, component_map)
    assert not result.valid, "border anchor must be rejected for nearest"
    assert result.reason == R.REASON_ANCHOR_NOT_ELIGIBLE, result.reason
    assert result.evidence.get("anchor_rejection") == "component_touches_image_border"
    print("  [relations] nearest rejects border anchor OK")


def test_nearest_rejects_border_target():
    """A border-truncated component must NOT be selected as a `nearest` target.

    Layout is chosen so the border component really IS the geometrically nearer
    one (gap 10 vs 100), so the exclusion is what produces the answer.
    """

    boxes = {
        1: (40, 200, 100, 260),    # interior anchor, near the left edge
        2: (0, 200, 30, 260),      # touches the left edge, and is NEARER (gap 10)
        3: (200, 200, 260, 260),   # interior, farther (gap 100)
    }
    component_map, image = make_image(boxes, width=400, height=400)
    quality = _quality(image)
    assert quality.flags(1).touches_image_border is False
    assert quality.flags(2).touches_image_border is True
    assert quality.flags(3).touches_image_border is False
    assert 2 not in quality.eligible_ids("nearest", CONFIG)

    # Sanity: the border component really is the closer one.
    assert G.component_box_distance(component_map, 1, 2) < G.component_box_distance(component_map, 1, 3)

    result = R.evaluate_nearest(image, 1, CONFIG, quality, component_map)
    assert result.valid, result.reason
    assert result.object == 3, (
        f"border target 2 must be excluded even though it is closer, got {result.object}"
    )
    print("  [relations] nearest rejects border target OK")


def test_nearest_border_policy_does_not_affect_extremes():
    """The tightened nearest policy must not change tile-relative extremes."""

    boxes = {
        1: (0, 100, 40, 140),      # border, and leftmost
        2: (200, 100, 260, 160),
        3: (100, 200, 150, 250),
    }
    component_map, image = make_image(boxes)
    quality = _quality(image)
    leftmost = R.evaluate_extreme("leftmost", image, CONFIG, quality)
    assert leftmost.valid and leftmost.subject == 1, (
        "border components must remain eligible for leftmost"
    )
    print("  [relations] nearest border policy leaves extremes unchanged OK")


# --------------------------------------------------------------------------
# Threshold preset monotonicity
# --------------------------------------------------------------------------


def test_preset_monotonicity_on_all_pairs():
    """Valid(strict) subset of Valid(medium) subset of Valid(loose).

    This is the defining semantic of a strict -> medium -> loose preset family.
    Tested on a spread of synthetic pairs covering clean, near-diagonal and
    small-margin geometry.
    """

    presets = CONFIG.direction.candidates
    for name, params in presets.items():
        assert "alpha" in params and "tau" in params, name

    # Monotonicity must hold in the config itself: alpha decreases and tau
    # decreases from strict to loose.
    assert presets["strict"]["alpha"] >= presets["medium"]["alpha"] >= presets["loose"]["alpha"]
    assert presets["strict"]["tau"] >= presets["medium"]["tau"] >= presets["loose"]["tau"]

    pairs = [
        ("clean_horizontal", (10, 120, 60, 180), (250, 120, 300, 180)),
        ("clean_vertical", (120, 10, 180, 60), (120, 250, 180, 300)),
        ("mild_diagonal", (10, 10, 60, 60), (150, 100, 200, 150)),
        ("strong_diagonal", (10, 10, 60, 60), (90, 80, 140, 130)),
        ("tiny_margin", (198, 190, 210, 202), (211, 190, 223, 202)),
        ("medium_margin", (100, 100, 130, 130), (145, 100, 175, 130)),
    ]

    checked = 0
    for tag, box_a, box_b in pairs:
        width = 400 if tag == "tiny_margin" else 400
        component_map, image = make_image({1: box_a, 2: box_b}, width=width, height=width)
        base_quality = _quality(image)
        a, b = image.get(1), image.get(2)

        verdicts = {}
        for name in ("strict", "medium", "loose"):
            cfg = T.with_direction_candidate(CONFIG, name)
            # a is left of b in every fixture above.
            verdicts[name] = R.evaluate_direction("left_of", image, a, b, cfg, base_quality).valid

        assert not (verdicts["strict"] and not verdicts["medium"]), (
            f"{tag}: strict valid but medium invalid {verdicts}"
        )
        assert not (verdicts["medium"] and not verdicts["loose"]), (
            f"{tag}: medium valid but loose invalid {verdicts}"
        )
        checked += 1
    print(f"  [relations] preset monotonicity strict<=medium<=loose OK ({checked} pairs)")


def test_preset_monotonicity_over_real_dataset_sample():
    """The same subset relation must hold on real image pairs.

    Skipped cleanly when the Task 2 component representation is not present.
    """

    dataset_root = _REPO_ROOT / "datasets" / "whu"
    if not (dataset_root / "metadata" / "train.jsonl").is_file():
        print("  [relations] real-dataset monotonicity SKIPPED (no component dataset)")
        return

    import dataset_access

    presets = {
        name: T.with_direction_candidate(CONFIG, name)
        for name in ("strict", "medium", "loose")
    }

    checked = 0
    scanned = 0
    for record in G.iter_metadata(dataset_root, "train"):
        image = G.image_geometry_from_record(record)
        if len(image.components) < 2:
            continue
        # No component map needed: direction relations use centroids only.
        quality = classify_image(image, CONFIG)
        components = image.components
        for i in range(len(components)):
            for j in range(len(components)):
                if i == j:
                    continue
                subject, object_ = components[i], components[j]
                verdicts = {
                    name: R.evaluate_direction(
                        "left_of", image, subject, object_, cfg, quality
                    ).valid
                    for name, cfg in presets.items()
                }
                assert not (verdicts["strict"] and not verdicts["medium"]), (
                    f"{image.image_id} {subject.component_id}->{object_.component_id}: {verdicts}"
                )
                assert not (verdicts["medium"] and not verdicts["loose"]), (
                    f"{image.image_id} {subject.component_id}->{object_.component_id}: {verdicts}"
                )
                checked += 1
        scanned += 1
        if checked >= 1000:
            break
    print(f"  [relations] preset monotonicity on real pairs OK ({checked} pairs, {scanned} images)")


# --------------------------------------------------------------------------
# Config integrity and documented scope
# --------------------------------------------------------------------------


def test_fixture_geometry_matches_stated_extents():
    """Guard the fixture premise: plain components must not trip the merge rule."""

    component_map, image = make_image({1: (10, 10, 130, 130)})
    quality = _quality(image)
    extent = quality.flags(1).bbox_extent_ratio
    assert abs(extent - PLAIN_COMPONENT / (CANVAS * CANVAS)) < 1e-12
    assert extent < CONFIG.quality.merge_bbox_extent_ratio
    assert CONFIG.quality.merge_bbox_extent_ratio == MERGE_COMPONENT_MIN_EXTENT
    print(f"  [relations] fixture extents consistent with config OK ({extent:.3f})")


def test_config_rejects_centroid_distance_for_nearest():
    """The centroid shortcut must be explicitly rejected by config validation."""

    import copy

    import thresholds

    raw = thresholds._load_yaml(thresholds.DEFAULT_CONFIG_PATH)
    broken = copy.deepcopy(raw)
    broken["nearest"]["distance_metric"] = "centroid_distance"
    bad_path = _REPO_ROOT / ".pytest_tmp_bad_config.yaml"
    try:
        import yaml

        bad_path.write_text(yaml.safe_dump(broken), encoding="utf-8")
        try:
            thresholds.load_config(bad_path)
        except thresholds.ConfigError:
            print("  [relations] config rejects centroid distance for nearest OK")
            return
        raise AssertionError("config accepted centroid_distance for nearest")
    finally:
        bad_path.unlink(missing_ok=True)


def test_core_relation_set_is_frozen():
    excluded = {"overlap", "contain", "inside", "adjacent_to", "near", "far"}
    assert excluded.isdisjoint(T.CORE_RELATIONS), (
        "relations that are structurally empty or uncalibrated must stay out of v1"
    )
    assert len(T.CORE_RELATIONS) == 11
    print("  [relations] frozen core relation set OK")


def test_scope_is_tile_relative():
    assert CONFIG.scope_kind == "tile_relative"
    print("  [relations] scope declared as tile-relative OK")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("semantics_left_right", test_predicate_semantics_left_right_absolute_coordinates),
        ("semantics_above_below", test_predicate_semantics_above_below_absolute_coordinates),
        ("semantics_evidence", test_predicate_evidence_is_self_describing),
        ("semantics_round_trip", test_round_trip_semantic_mapping),
        ("left/right", test_left_of_and_right_of),
        ("above/below", test_above_and_below),
        ("wrong_direction", test_wrong_direction_rejected),
        ("axis_dominance", test_axis_not_dominant_rejected),
        ("small_margin", test_margin_too_small_rejected),
        ("dominance_boundary", test_axis_ratio_threshold_boundary),
        ("symmetry", test_direction_symmetry),
        ("extreme", test_extreme_relations),
        ("extreme_ambiguous", test_extreme_ambiguous_when_margin_small),
        ("largest/smallest", test_largest_and_smallest),
        ("size_ambiguous", test_size_rank_ambiguous_on_near_tie),
        ("nearest_boundary", test_nearest_uses_boundary_distance),
        ("nearest_ambiguous", test_nearest_ambiguous_on_near_tie),
        ("nearest_eligibility", test_nearest_respects_eligibility),
        ("nearest_no_candidate", test_nearest_single_candidate_is_valid_not_ambiguous),
        ("nearest_border_anchor", test_nearest_rejects_border_anchor),
        ("nearest_border_target", test_nearest_rejects_border_target),
        ("nearest_border_extremes", test_nearest_border_policy_does_not_affect_extremes),
        ("preset_monotonic", test_preset_monotonicity_on_all_pairs),
        ("preset_monotonic_real", test_preset_monotonicity_over_real_dataset_sample),
        ("border_leftmost", test_border_component_allowed_in_tile_relative_leftmost),
        ("border_largest", test_border_component_excluded_from_largest),
        ("merge_scope", test_merge_heuristic_only_affects_specified_relations),
        ("tiny_asymmetry", test_tiny_component_allowed_as_largest_but_not_smallest),
        ("smallest_ratio", test_smallest_ratio_uses_larger_over_smaller),
        ("no_such_component", test_no_such_component),
        ("single_component", test_single_component_image_is_rejected_cleanly),
        ("fixture_extents", test_fixture_geometry_matches_stated_extents),
        ("config_nearest_metric", test_config_rejects_centroid_distance_for_nearest),
        ("config_frozen_set", test_core_relation_set_is_frozen),
        ("scope", test_scope_is_tile_relative),
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
    print(f"{len(tests) - failures}/{len(tests)} relation checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
