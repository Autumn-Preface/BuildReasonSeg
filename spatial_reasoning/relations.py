"""Spatial relation engine.

Implements the frozen v1 core relation set over the canonical building
component representation::

    left_of   right_of   above   below
    leftmost  rightmost  topmost  bottommost
    nearest
    largest   smallest

Deliberately NOT implemented in v1, with reasons:

``overlap`` / ``contain`` / ``inside``
    Structurally empty. The component maps come from disjoint external contours,
    so no two components share a pixel. Measured across the full dataset:
    0 polygon conflicts. These relations could never fire.

``adjacent_to``
    No dependable ground truth. Mutually touching buildings were already merged
    upstream into a single component, so strict-touch adjacency cannot be
    distinguished from merge.

``near`` / ``far``
    Require a calibrated scale threshold that has not been validated yet.

Every relation returns a structured :class:`RelationResult` carrying a status,
a score, a margin, and the numeric evidence that produced the decision, so a
downstream validator can audit why a candidate relation was accepted,
discarded, or marked ambiguous.

Scope
-----
This is **tile-relative** spatial reasoning. ``leftmost`` means leftmost within
one image tile, not leftmost in the world. A border-truncated component may
legitimately be the leftmost component of its tile. Magnitude relations
(``largest`` / ``smallest``) are not tile-relative and therefore exclude
truncated and suspected-merged components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from component_quality import ImageQuality, classify_image
from geometry import (
    Component,
    ImageGeometry,
    boundary_distance_between,
    bbox_gap,
    component_box_distance,
    rank_by,
    subject_minus_object_delta,
)
from thresholds import (
    DIRECTION_RELATIONS,
    EXTREME_RELATIONS,
    SIZE_RELATIONS,
    RelationConfig,
)

# --------------------------------------------------------------------------
# Status vocabulary (auditable, stable strings)
# --------------------------------------------------------------------------

STATUS_VALID = "valid"
STATUS_AMBIGUOUS = "ambiguous"

# Rejection reasons
REASON_NO_SUCH_COMPONENT = "no_such_component"
REASON_INVALID_COMPONENT = "invalid_component"
REASON_WRONG_DIRECTION = "wrong_direction"
REASON_AXIS_NOT_DOMINANT = "axis_not_dominant"
REASON_MARGIN_TOO_SMALL = "margin_too_small"
REASON_TOO_FEW_CANDIDATES = "too_few_eligible_candidates"
REASON_ANCHOR_NOT_ELIGIBLE = "anchor_not_eligible"
REASON_ANCHOR_EQUALS_TARGET = "anchor_equals_target"

#: Directional relation -> (axis, required sign of the subject-relative offset).
#:
#: FROZEN PREDICATE CONVENTION
#: --------------------------
#: ``relation(subject, object)`` always means "the SUBJECT satisfies the
#: relation with respect to the OBJECT". Natural-language reading:
#:
#:     left_of(A, B)   ==  "A is left of B"   ==  cx_A < cx_B
#:     right_of(A, B)  ==  "A is right of B"  ==  cx_A > cx_B
#:     above(A, B)     ==  "A is above B"     ==  cy_A < cy_B
#:     below(A, B)     ==  "A is below B"     ==  cy_A > cy_B
#:
#: The offsets below are expressed as ``subject - object``
#: (:func:`geometry.subject_minus_object_delta`), so:
#:
#:     left_of  requires subject_minus_object_dx < 0
#:     right_of requires subject_minus_object_dx > 0
#:     above    requires subject_minus_object_dy < 0   (image y grows downward)
#:     below    requires subject_minus_object_dy > 0
#:
#: This was previously INVERTED (it required the object to be on the left for
#: ``left_of``), which contradicted standard predicate semantics.
#: ``tests/test_relations.py`` now pins the absolute-coordinate behaviour so it
#: cannot regress.
DIRECTION_SIGN: dict[str, tuple[str, int]] = {
    "left_of": ("dx", -1),
    "right_of": ("dx", +1),
    "above": ("dy", -1),
    "below": ("dy", +1),
}

#: Human-readable predicate templates, used by the round-trip semantic test and
#: available to the future instruction generator so wording and logic cannot
#: drift apart.
PREDICATE_TEMPLATE: dict[str, str] = {
    "left_of": "{subject} is left of {object}",
    "right_of": "{subject} is right of {object}",
    "above": "{subject} is above {object}",
    "below": "{subject} is below {object}",
}


def describe_relation(relation: str, subject, object_) -> str:
    """Render a relation as natural language in the frozen convention.

    ``describe_relation("left_of", "A", "B")`` -> ``"A is left of B"``.

    This is the minimal semantic mapping used by the round-trip test: the
    sentence must always be interpretable as ``subject <relation> object``.
    """

    if relation not in PREDICATE_TEMPLATE:
        raise ValueError(f"no predicate template for relation: {relation}")
    return PREDICATE_TEMPLATE[relation].format(subject=subject, object=object_)

#: Opposite relation, used to assert symmetry.
OPPOSITE_RELATION: dict[str, str] = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
}


@dataclass
class RelationResult:
    """Structured outcome of evaluating one relation.

    ``valid`` is True only when the relation holds unambiguously.
    ``ambiguous`` marks a case that is geometrically real but too close to a
    decision boundary to be used as supervision. ``reason`` explains any
    non-valid outcome so a validator can group and audit them.
    """

    valid: bool
    relation: str
    subject: int | None = None
    object: int | None = None
    ambiguous: bool = False
    reason: str | None = None
    score: float | None = None
    margin: float | None = None
    evidence: dict = field(default_factory=dict)

    def __bool__(self) -> bool:  # convenience: `if result:`
        return self.valid

    def contradicts(self, other: "RelationResult") -> bool:
        """True when two results both claim to be valid but disagree.

        Used to check that a relation and its converse are never both valid.
        """

        if not (self.valid and other.valid):
            return False
        return (
            self.subject == other.subject
            and self.object == other.object
            and self.relation != other.relation
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _reject(relation: str, reason: str, subject=None, object_=None, evidence=None) -> RelationResult:
    return RelationResult(
        valid=False,
        relation=relation,
        subject=subject,
        object=object_,
        ambiguous=reason == STATUS_AMBIGUOUS,
        reason=reason,
        evidence=evidence or {},
    )


def _normalize(value: float, scale: float) -> float:
    return float(value / scale) if scale > 0 else 0.0


# --------------------------------------------------------------------------
# Directional relations
# --------------------------------------------------------------------------


def evaluate_direction(
    relation: str,
    image: ImageGeometry,
    subject: Component,
    object_: Component,
    config: RelationConfig,
    quality: ImageQuality | None = None,
) -> RelationResult:
    """Evaluate ``left_of`` / ``right_of`` / ``above`` / ``below``.

    A directional relation holds only when all three conditions hold:

    1. **Direction** -- the dominant-axis delta has the required sign.
    2. **Axis dominance** -- ``dominant >= alpha * other``, so near-diagonal
       pairs are not claimed as a clean left/right or above/below relation.
    3. **Minimum normalized margin** -- ``dominant_normalized >= tau``, so the
       separation is not a rounding artifact.

    Margin is measured on the dominant axis and normalized by the image extent
    on that axis, which keeps it comparable across images.
    """

    if relation not in DIRECTION_RELATIONS:
        raise ValueError(f"not a directional relation: {relation}")

    quality = quality or classify_image(image, config)
    rule = config.eligibility_for(relation)

    for side, component in (("subject", subject), ("object", object_)):
        accepted, reason = rule.accepts(quality.flags(component.component_id).as_dict())
        if not accepted:
            return _reject(
                relation,
                REASON_INVALID_COMPONENT,
                subject.component_id,
                object_.component_id,
                {"rejected_side": side, "rejection": reason},
            )

    # Offsets are (subject - object) so the sign reads directly as
    # "where is the subject relative to the object".
    subject_minus_object_dx, subject_minus_object_dy = subject_minus_object_delta(
        subject, object_
    )
    axis, sign = DIRECTION_SIGN[relation]

    dominant_raw = (
        subject_minus_object_dx if axis == "dx" else subject_minus_object_dy
    )
    other_raw = (
        subject_minus_object_dy if axis == "dx" else subject_minus_object_dx
    )
    dominant_norm = _normalize(dominant_raw, image.width if axis == "dx" else image.height)
    other_norm = _normalize(other_raw, image.height if axis == "dx" else image.width)

    abs_dom = abs(dominant_norm)
    abs_other = abs(other_norm)
    axis_ratio = float(abs_dom / abs_other) if abs_other > 0 else float("inf")

    evidence = {
        # Explicitly named so the sign convention is readable from evidence
        # alone, with no implicit agreement required.
        "subject_centroid_x": float(subject.centroid_x),
        "subject_centroid_y": float(subject.centroid_y),
        "object_centroid_x": float(object_.centroid_x),
        "object_centroid_y": float(object_.centroid_y),
        "subject_minus_object_dx": float(subject_minus_object_dx),
        "subject_minus_object_dy": float(subject_minus_object_dy),
        "subject_minus_object_dx_norm": _normalize(subject_minus_object_dx, image.width),
        "subject_minus_object_dy_norm": _normalize(subject_minus_object_dy, image.height),
        "relationship": describe_relation(relation, "subject", "object"),
        "dominant_axis": "x" if axis == "dx" else "y",
        "dominant_norm": dominant_norm,
        "other_norm": other_norm,
        "axis_ratio": axis_ratio,
        "alpha": config.direction.alpha,
        "tau": config.direction.tau,
        "direction_candidate": config.direction.active_candidate,
    }

    # 1. direction
    if np.sign(dominant_raw) != sign:
        result = _reject(relation, REASON_WRONG_DIRECTION, subject.component_id, object_.component_id, evidence)
        result.score = float(dominant_norm)
        result.margin = float(abs_dom)
        return result

    # 2. axis dominance
    if abs_dom < config.direction.alpha * abs_other:
        result = _reject(relation, REASON_AXIS_NOT_DOMINANT, subject.component_id, object_.component_id, evidence)
        result.score = float(dominant_norm)
        result.margin = float(abs_dom)
        return result

    # 3. minimum normalized margin
    if abs_dom < config.direction.tau:
        result = _reject(relation, REASON_MARGIN_TOO_SMALL, subject.component_id, object_.component_id, evidence)
        result.score = float(dominant_norm)
        result.margin = float(abs_dom)
        return result

    return RelationResult(
        valid=True,
        relation=relation,
        subject=subject.component_id,
        object=object_.component_id,
        score=float(dominant_norm),
        margin=float(abs_dom),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# Extreme relations
# --------------------------------------------------------------------------


def evaluate_extreme(
    relation: str,
    image: ImageGeometry,
    config: RelationConfig,
    quality: ImageQuality | None = None,
) -> RelationResult:
    """Evaluate ``leftmost`` / ``rightmost`` / ``topmost`` / ``bottommost``.

    The best and second-best components must be separated by at least
    ``extreme.margin_px``; otherwise the result is ambiguous.

    Border-truncated components ARE eligible here, because being leftmost is a
    tile-relative property: a component clipped by the left tile edge is still
    the leftmost component of this tile.
    """

    if relation not in EXTREME_RELATIONS:
        raise ValueError(f"not an extreme relation: {relation}")

    quality = quality or classify_image(image, config)
    eligible_ids = set(quality.eligible_ids(relation, config))
    candidates = [c for c in image.components if c.component_id in eligible_ids]

    if len(candidates) < 2:
        return _reject(relation, REASON_TOO_FEW_CANDIDATES, evidence={"n_eligible": len(candidates)})

    reverse = relation in ("rightmost", "bottommost")
    axis = "x" if relation in ("leftmost", "rightmost") else "y"
    key: Callable[[Component], float] = (
        (lambda c: c.centroid_x) if axis == "x" else (lambda c: c.centroid_y)
    )

    ordered = rank_by(candidates, key, reverse=reverse)
    best, second = ordered[0], ordered[1]
    margin_px = abs(key(best) - key(second))
    span = image.width if axis == "x" else image.height

    evidence = {
        "axis": axis,
        "value_px": float(key(best)),
        "second_value_px": float(key(second)),
        "margin_px": float(margin_px),
        "margin_norm": _normalize(margin_px, span),
        "margin_threshold_px": config.extreme_margin_px,
        "n_eligible": len(candidates),
    }

    if margin_px < config.extreme_margin_px:
        result = _reject(relation, STATUS_AMBIGUOUS, best.component_id, second.component_id, evidence)
        result.ambiguous = True
        result.score = float(key(best))
        result.margin = float(margin_px)
        return result

    return RelationResult(
        valid=True,
        relation=relation,
        subject=best.component_id,
        score=float(key(best)),
        margin=float(margin_px),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# Size relations
# --------------------------------------------------------------------------


def evaluate_size_rank(
    relation: str,
    image: ImageGeometry,
    config: RelationConfig,
    quality: ImageQuality | None = None,
) -> RelationResult:
    """Evaluate ``largest`` / ``smallest``.

    Eligibility is asymmetric by design:

    * both exclude border-truncated and suspected-merged components, because
      their area is not a reliable magnitude;
    * ``smallest`` additionally excludes tiny components, since otherwise it
      would select rasterization noise rather than a small building;
    * ``largest`` does not exclude tiny components (a tiny component can hardly
      be the largest, and excluding it would only shrink the candidate pool).
    """

    if relation not in SIZE_RELATIONS:
        raise ValueError(f"not a size relation: {relation}")

    quality = quality or classify_image(image, config)
    eligible_ids = set(quality.eligible_ids(relation, config))
    candidates = [c for c in image.components if c.component_id in eligible_ids]

    if len(candidates) < 2:
        return _reject(relation, REASON_TOO_FEW_CANDIDATES, evidence={"n_eligible": len(candidates)})

    reverse = relation == "largest"
    ordered = rank_by(candidates, lambda c: c.area_px, reverse=reverse)
    best, second = ordered[0], ordered[1]

    # Separation is ALWAYS expressed as larger/smaller >= ratio_margin, so the
    # same threshold means the same thing for both relations. (Comparing
    # rank1/rank2 directly would make the ratio < 1 for `smallest` and the
    # relation could never be valid.)
    larger, smaller = (best, second) if best.area_px >= second.area_px else (second, best)
    ratio = larger.area_px / max(smaller.area_px, 1)
    evidence = {
        "area_px": int(best.area_px),
        "second_area_px": int(second.area_px),
        "ratio": float(ratio),
        "ratio_definition": "larger_area / smaller_area",
        "ratio_threshold": config.size_rank.ratio_margin,
        "n_eligible": len(candidates),
        "excluded_tiny": config.eligibility_for(relation).reject_tiny_component,
    }

    if ratio < config.size_rank.ratio_margin:
        result = _reject(relation, STATUS_AMBIGUOUS, best.component_id, second.component_id, evidence)
        result.ambiguous = True
        result.score = float(best.area_px)
        result.margin = float(ratio)
        return result

    return RelationResult(
        valid=True,
        relation=relation,
        subject=best.component_id,
        score=float(best.area_px),
        margin=float(ratio),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# nearest
# --------------------------------------------------------------------------


def evaluate_nearest(
    image: ImageGeometry,
    anchor_id: int,
    config: RelationConfig,
    quality: ImageQuality | None = None,
    component_map: np.ndarray | None = None,
) -> RelationResult:
    """Evaluate ``nearest(anchor)`` -> unique target component.

    Distance is **boundary_distance**, i.e. the minimum distance between the two
    components' boundaries. The centroid-distance shortcut is deliberately NOT
    used: it misranks elongated and L-shaped buildings, which is exactly the
    geometry this dataset is full of.

    The target must win by a margin, evaluated both absolutely (``margin_px_floor``)
    and normalized to the image diagonal (``margin_diag_fraction``). Both are
    reported in the evidence; the normalized term is authoritative with the
    absolute floor as a backstop.

    Border policy (FROZEN): both the anchor and every candidate must satisfy the
    relation's eligibility rule, which for `nearest` excludes border-truncated
    components. Because the metric is a boundary distance, a component clipped by
    the tile edge has an incomplete boundary, making both its distances and the
    ordering of distances around it unreliable. See docs/relation_definitions.md.
    """

    quality = quality or classify_image(image, config)
    rule = config.eligibility_for("nearest")

    anchor = next((c for c in image.components if c.component_id == anchor_id), None)
    if anchor is None:
        return _reject("nearest", REASON_NO_SUCH_COMPONENT, anchor_id)

    if anchor.area_px <= 0:
        return _reject("nearest", REASON_ANCHOR_NOT_ELIGIBLE, anchor_id)

    # Border policy for `nearest` (FROZEN): the anchor must ALSO satisfy the
    # relation's eligibility rule, not just the candidates.
    #
    # Reason: `nearest` is measured with boundary_distance. A component clipped
    # by the tile edge has an incomplete boundary, so both the distance from it
    # and the ordering of distances around it are unreliable. Accepting a border
    # anchor would produce ground truth that looks fine but is not.
    #
    # This does NOT change the tile-relative policy for leftmost / rightmost /
    # topmost / bottommost, which still accept border components.
    anchor_accepted, anchor_rejection = rule.accepts_as_anchor(
        quality.flags(anchor.component_id).as_dict()
    )
    if not anchor_accepted:
        return _reject(
            "nearest",
            REASON_ANCHOR_NOT_ELIGIBLE,
            anchor_id,
            evidence={"anchor_rejection": anchor_rejection, "rejected_side": "anchor"},
        )

    eligible_ids = set(quality.eligible_ids("nearest", config))
    candidates = [c for c in image.components if c.component_id in eligible_ids and c.component_id != anchor_id]
    if len(candidates) < rule.min_valid_components - 1:
        return _reject(
            "nearest",
            REASON_TOO_FEW_CANDIDATES,
            anchor_id,
            evidence={"n_eligible_candidates": len(candidates)},
        )

    if component_map is None:
        from dataset_access import DEFAULT_DATASET_ROOT

        component_map = image.load_map(DEFAULT_DATASET_ROOT)

    # Compute the EXACT gap for every eligible candidate.
    #
    # An earlier version pruned candidates using bbox_gap as if it were a lower
    # bound. It is not: bbox_gap upper-bounds the empty-pixel gap this engine
    # defines (which subtracts one pixel and shrinks further for diagonal
    # arrangements), so pruning with it could discard the true nearest
    # neighbour. Correctness matters more than a micro-optimisation here, and
    # the windowed distance transform in `component_box_distance` already keeps
    # each pair cheap.
    scored: list[tuple[float, int]] = []
    for candidate in candidates:
        distance = component_box_distance(
            component_map, anchor.component_id, candidate.component_id
        )
        scored.append((distance, candidate.component_id))

    if not scored:
        return _reject("nearest", REASON_TOO_FEW_CANDIDATES, anchor_id)

    if len(scored) < 1:
        return _reject("nearest", REASON_TOO_FEW_CANDIDATES, anchor_id)

    scored.sort(key=lambda t: (t[0], t[1]))
    d1, id1 = scored[0]

    if len(scored) < 2:
        # Exactly one admissible candidate: the target is UNIQUE, so the answer
        # is not ambiguous -- there is simply no runner-up to compare against.
        # Ambiguity in this relation means the distance MARGIN is too small, so
        # a lone candidate is accepted. Measured on the dataset: only about 6.8%
        # of anchors have exactly one candidate.
        evidence = {
            "distance_metric": "boundary_distance",
            "d1_px": float(d1),
            "nearest_id": int(id1),
            "margin_px": None,
            "margin_diag_fraction": None,
            "margin_px_floor": config.nearest.margin_px_floor,
            "margin_diag_threshold": config.nearest.margin_diag_fraction,
            "n_candidates": len(candidates),
            "n_scored": len(scored),
            "note": "single admissible candidate: unique answer, margin not applicable",
        }
        return RelationResult(
            valid=True,
            relation="nearest",
            subject=anchor_id,
            object=int(id1),
            score=float(d1),
            margin=None,
            evidence=evidence,
        )

    d2, id2 = scored[1]
    margin_abs = d2 - d1
    margin_norm = _normalize(margin_abs, image.diagonal)

    evidence = {
        "distance_metric": "boundary_distance",
        "d1_px": float(d1),
        "d2_px": float(d2),
        "nearest_id": int(id1),
        "second_id": int(id2),
        "margin_px": float(margin_abs),
        "margin_diag_fraction": float(margin_norm),
        "margin_px_floor": config.nearest.margin_px_floor,
        "margin_diag_threshold": config.nearest.margin_diag_fraction,
        "n_candidates": len(candidates),
        "n_scored": len(scored),
    }

    if margin_norm < config.nearest.margin_diag_fraction or margin_abs < config.nearest.margin_px_floor:
        result = _reject("nearest", STATUS_AMBIGUOUS, anchor_id, int(id1), evidence)
        result.ambiguous = True
        result.score = float(d1)
        result.margin = float(margin_abs)
        return result

    return RelationResult(
        valid=True,
        relation="nearest",
        subject=anchor_id,
        object=int(id1),
        score=float(d1),
        margin=float(margin_abs),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# Conditional nearest: nearest WITHIN a restricted candidate set
# --------------------------------------------------------------------------


@dataclass
class NearestWithinResult:
    """Outcome of :func:`nearest_within`.

    ``trivial_selection`` is True when the filtered candidate set contains
    exactly one component. In that case the answer is forced by the filter
    alone and the model does not actually have to compare distances, so such
    samples must be tracked separately from genuine nearest comparisons.
    """

    valid: bool
    relation: str = "nearest_within"
    anchor: int | None = None
    target: int | None = None
    ambiguous: bool = False
    reason: str | None = None
    trivial_selection: bool = False
    candidate_ids: list[int] = field(default_factory=list)
    score: float | None = None
    margin: float | None = None
    evidence: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid


def nearest_within(
    image: ImageGeometry,
    anchor_id: int,
    candidate_ids: Sequence[int],
    config: RelationConfig,
    quality: ImageQuality | None = None,
    component_map: np.ndarray | None = None,
) -> NearestWithinResult:
    """Nearest component to ``anchor`` **restricted to ``candidate_ids``**.

    This is deliberately NOT ``nearest(anchor)`` followed by a filter. The two
    differ: a global nearest neighbour may lie outside the filtered set, so
    filtering its result can lose the true answer, and it answers a different
    question. Here the comparison happens *inside* the restricted set.

    It otherwise follows the frozen ``nearest`` semantics exactly:

    * distance is ``boundary_distance`` (never the centroid shortcut);
    * the anchor must be non-border (anchor eligibility rule);
    * every candidate must also satisfy the ``nearest`` eligibility rule;
    * the same margin policy applies when two or more candidates remain.

    Outcomes:

    * ``candidate_count == 0`` -> invalid, ``too_few_eligible_candidates``;
    * ``candidate_count == 1`` -> valid with ``trivial_selection = True``;
    * ``candidate_count >= 2`` passing the margin -> valid, nontrivial;
    * ``candidate_count >= 2`` failing the margin -> ambiguous, discarded by the
      caller (no secondary tie-break is ever applied).
    """

    quality = quality or classify_image(image, config)
    rule = config.eligibility_for("nearest")

    anchor = next((c for c in image.components if c.component_id == anchor_id), None)
    if anchor is None:
        return NearestWithinResult(valid=False, reason=REASON_NO_SUCH_COMPONENT, anchor=anchor_id)
    if anchor.area_px <= 0:
        return NearestWithinResult(valid=False, reason=REASON_ANCHOR_NOT_ELIGIBLE, anchor=anchor_id)

    anchor_accepted, anchor_rejection = rule.accepts_as_anchor(
        quality.flags(anchor.component_id).as_dict()
    )
    if not anchor_accepted:
        return NearestWithinResult(
            valid=False,
            reason=REASON_ANCHOR_NOT_ELIGIBLE,
            anchor=anchor_id,
            evidence={"anchor_rejection": anchor_rejection, "rejected_side": "anchor"},
        )

    # Restrict to the supplied set AND the relation's eligibility, then drop the
    # anchor itself. Order does not matter; the set is sorted for determinism.
    eligible = set(quality.eligible_ids("nearest", config))
    candidates: list[Component] = []
    rejected: dict[int, str] = {}
    for component_id in sorted(set(int(i) for i in candidate_ids)):
        if component_id == anchor_id:
            continue
        component = next((c for c in image.components if c.component_id == component_id), None)
        if component is None:
            rejected[component_id] = "no_such_component"
            continue
        if component_id not in eligible:
            rejected[component_id] = "component_ineligible_for_nearest"
            continue
        candidates.append(component)

    if not candidates:
        return NearestWithinResult(
            valid=False,
            reason=REASON_TOO_FEW_CANDIDATES,
            anchor=anchor_id,
            candidate_ids=[],
            evidence={"n_supplied": len(set(candidate_ids)), "rejected": rejected},
        )

    if component_map is None:
        from dataset_access import DEFAULT_DATASET_ROOT

        component_map = image.load_map(DEFAULT_DATASET_ROOT)

    scored: list[tuple[float, int]] = []
    for component in candidates:
        distance = component_box_distance(
            component_map, anchor.component_id, component.component_id
        )
        scored.append((distance, component.component_id))
    scored.sort(key=lambda t: (t[0], t[1]))

    d1, id1 = scored[0]
    candidate_id_list = sorted(c.component_id for c in candidates)

    base_evidence = {
        "distance_metric": "boundary_distance",
        "anchor_id": int(anchor_id),
        "candidate_ids": candidate_id_list,
        "n_candidates": len(candidates),
        "rejected_candidates": rejected,
        "d1_px": float(d1),
        "nearest_id": int(id1),
        "margin_px_floor": config.nearest.margin_px_floor,
        "margin_diag_threshold": config.nearest.margin_diag_fraction,
    }

    if len(scored) == 1:
        # The filter alone determines the answer.
        base_evidence["margin_px"] = None
        base_evidence["note"] = "single candidate after filtering: trivial_selection"
        return NearestWithinResult(
            valid=True,
            anchor=anchor_id,
            target=int(id1),
            trivial_selection=True,
            candidate_ids=candidate_id_list,
            score=float(d1),
            margin=None,
            evidence=base_evidence,
        )

    d2, id2 = scored[1]
    margin_abs = d2 - d1
    margin_norm = _normalize(margin_abs, image.diagonal)
    base_evidence.update(
        {
            "d2_px": float(d2),
            "second_id": int(id2),
            "margin_px": float(margin_abs),
            "margin_diag_fraction": float(margin_norm),
        }
    )

    if margin_norm < config.nearest.margin_diag_fraction or margin_abs < config.nearest.margin_px_floor:
        return NearestWithinResult(
            valid=False,
            anchor=anchor_id,
            target=int(id1),
            ambiguous=True,
            reason=STATUS_AMBIGUOUS,
            trivial_selection=False,
            candidate_ids=candidate_id_list,
            score=float(d1),
            margin=float(margin_abs),
            evidence=base_evidence,
        )

    return NearestWithinResult(
        valid=True,
        anchor=anchor_id,
        target=int(id1),
        trivial_selection=False,
        candidate_ids=candidate_id_list,
        score=float(d1),
        margin=float(margin_abs),
        evidence=base_evidence,
    )


# --------------------------------------------------------------------------
# Image-level relation set
# --------------------------------------------------------------------------


@dataclass
class ImageRelations:
    """All evaluated relations for one image."""

    image_id: str
    split: str
    quality: ImageQuality
    directional: list[RelationResult] = field(default_factory=list)
    extremes: dict[str, RelationResult] = field(default_factory=dict)
    size_rank: dict[str, RelationResult] = field(default_factory=dict)
    nearest: dict[int, RelationResult] = field(default_factory=dict)

    def all_results(self) -> list[RelationResult]:
        out = list(self.directional)
        out.extend(self.extremes.values())
        out.extend(self.size_rank.values())
        out.extend(self.nearest.values())
        return out

    def valid_results(self) -> list[RelationResult]:
        return [r for r in self.all_results() if r.valid]

    def ambiguous_results(self) -> list[RelationResult]:
        return [r for r in self.all_results() if r.ambiguous]

    def discarded_results(self) -> list[RelationResult]:
        return [r for r in self.all_results() if not r.valid and not r.ambiguous]

    def count_by_relation(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for result in self.all_results():
            bucket = counts.setdefault(result.relation, {"valid": 0, "ambiguous": 0, "discarded": 0})
            if result.valid:
                bucket["valid"] += 1
            elif result.ambiguous:
                bucket["ambiguous"] += 1
            else:
                bucket["discarded"] += 1
        return counts


def evaluate_image(
    image: ImageGeometry,
    config: RelationConfig,
    *,
    compute_directional: bool = True,
    compute_nearest: bool = True,
    component_map: np.ndarray | None = None,
) -> ImageRelations:
    """Evaluate every relation for one image."""

    quality = classify_image(image, config)
    result = ImageRelations(image_id=image.image_id, split=image.split, quality=quality)

    if compute_directional:
        for relation in DIRECTION_RELATIONS:
            for subject in image.components:
                for object_ in image.components:
                    if subject.component_id == object_.component_id:
                        continue
                    result.directional.append(
                        evaluate_direction(relation, image, subject, object_, config, quality)
                    )

    for relation in EXTREME_RELATIONS:
        result.extremes[relation] = evaluate_extreme(relation, image, config, quality)

    for relation in SIZE_RELATIONS:
        result.size_rank[relation] = evaluate_size_rank(relation, image, config, quality)

    if compute_nearest:
        if component_map is None:
            from dataset_access import DEFAULT_DATASET_ROOT

            component_map = image.load_map(DEFAULT_DATASET_ROOT)
        for component in image.components:
            result.nearest[component.component_id] = evaluate_nearest(
                image, component.component_id, config, quality, component_map
            )

    return result
