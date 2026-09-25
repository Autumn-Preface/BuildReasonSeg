"""Independent audit-only semantic oracle for BuildSpatialReason.

Why this file exists
--------------------
The v0.1.1 generator and the v0.1.1 acceptance validator both call
``spatial_reasoning/semantic_policy.py``. That is good for *consistency* — the
two can no longer disagree about what a question means — but it is not
sufficient *evidence*: a shared implementation that is wrong is wrong twice, and
the acceptance audit would still say PASS.

This module is therefore a **second, deliberately independent implementation**
of the same semantics, used only for auditing and testing. It answers:

1. ``largest``                            -- global area maximum
2. ``smallest``                           -- global area minimum
3. ``reference -> nearest``               -- nearest over all other visible
4. ``reference -> direction``             -- frozen direction predicate
5. ``reference -> direction -> nearest``  -- nearest within the direction set

Independence contract
---------------------
This module MUST NOT import, call, or otherwise delegate to:

* ``semantic_policy`` (any name),
* ``SP.resolve_size_extreme`` / ``SP.resolve_nearest`` /
  ``SP.direction_candidates_over_visible`` / ``SP.admissible_nearest_ids``,
* any generator helper that internally calls them.

``tests/test_semantic_oracle.py`` enforces this twice: a static source scan and a
runtime ``sys.meta_path`` poison that makes importing ``semantic_policy`` raise.

What it *may* reuse (explicitly allowed by the task specification)
-----------------------------------------------------------------
* component metadata structures and the loader (``geometry``);
* the low-level geometric primitive ``geometry.component_box_distance``;
* frozen threshold/config loading (``thresholds``).

Everything that constitutes the *acceptance policy* — which universe a question
ranges over, what "largest" means, when a question is ambiguous, when a component
is eligible, when a chain is trivial — is re-derived here from the frozen
configuration and raw component fields. Component quality flags are recomputed
here from ``area_px`` / bounding box / border flag rather than read from
``component_quality``, so a defect in that module cannot hide behind this oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import geometry as G
import thresholds as T

#: Version of the oracle implementation itself (not of the policy it audits).
ORACLE_VERSION = "1.0"

#: Operation name per direct grounding query type (own copy, not imported).
LEVEL1_OPERATIONS: dict[str, str] = {
    "leftmost": "argmin_centroid_x",
    "rightmost": "argmax_centroid_x",
    "topmost": "argmin_centroid_y",
    "bottommost": "argmax_centroid_y",
    "largest": "argmax_area",
    "smallest": "argmin_area",
}

#: Extreme query type -> (axis attribute, reverse) using the FROZEN reading:
#: leftmost/rightmost on x, topmost/bottommost on y (image y grows downward).
EXTREME_AXES: dict[str, tuple[str, bool]] = {
    "leftmost": ("x", False),
    "rightmost": ("x", True),
    "topmost": ("y", False),
    "bottommost": ("y", True),
}

#: Frozen predicate convention (ADR-006): relation(subject, object) means the
#: SUBJECT satisfies the relation with respect to the OBJECT.
#: (axis, required sign of subject_centroid - object_centroid)
DIRECTION_SIGN: dict[str, tuple[str, int]] = {
    "left_of": ("dx", -1),
    "right_of": ("dx", +1),
    "above": ("dy", -1),
    "below": ("dy", +1),
}

DIRECTION_RELATIONS: tuple[str, ...] = ("left_of", "right_of", "above", "below")

#: Reason codes. Deliberately separate constants from semantic_policy's so a
#: shared typo cannot make both sides agree.
ORACLE_OK = "ok"
ORACLE_NO_COMPONENTS = "oracle_no_visible_components"
ORACLE_AMBIGUOUS = "oracle_ambiguous"
ORACLE_TARGET_INELIGIBLE = "oracle_target_ineligible"
ORACLE_MARGIN_FAIL = "oracle_margin_fail"
ORACLE_ANCHOR_INELIGIBLE = "oracle_anchor_ineligible"
ORACLE_NO_DIRECTION_CANDIDATE = "oracle_no_direction_candidate"
ORACLE_MULTIPLE_DIRECTION_CANDIDATES = "oracle_multiple_direction_candidates"
ORACLE_UNKNOWN_QUERY = "oracle_unknown_query_type"


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleResolution:
    """What the oracle says one query means, and whether it is usable."""

    query_type: str
    level: int
    admissible: bool
    target_id: int | None
    reference_id: int | None = None
    candidate_ids: tuple[int, ...] = ()
    #: ids admissible as the nearest answer, used for the Level-3 trivial flag.
    admissible_nearest_ids: tuple[int, ...] = ()
    reason: str = ORACLE_OK
    ranked: tuple[tuple[float, int], ...] = ()
    detail: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Independent quality / eligibility recomputation
# --------------------------------------------------------------------------


def component_flags(component, image: G.ImageGeometry, config: T.RelationConfig) -> dict[str, bool]:
    """Recompute the three quality flags directly from frozen thresholds.

    Independent of ``component_quality.classify_component``: the merge flag is
    re-derived from the bounding box, and the tiny flag from ``area_px``.
    """

    image_area = image.width * image.height
    extent = component.bbox_extent_ratio(image_area)
    return {
        "touches_image_border": bool(component.touches_image_border),
        "suspected_large_merge": bool(extent > config.quality.merge_bbox_extent_ratio),
        "tiny_component": bool(component.area_px < config.quality.tiny_area_px),
    }


def rule_accepts(rule: T.EligibilityRule, flags: dict[str, bool], as_anchor: bool = False) -> bool:
    """Apply a frozen eligibility rule without touching ``component_quality``."""

    if as_anchor and rule.reject_touches_image_border_anchor and flags["touches_image_border"]:
        return False
    if rule.reject_touches_image_border and flags["touches_image_border"]:
        return False
    if rule.reject_suspected_large_merge and flags["suspected_large_merge"]:
        return False
    if rule.reject_tiny_component and flags["tiny_component"]:
        return False
    return True


def _sign(value: float) -> int:
    """Integer sign, matching ``numpy.sign`` for finite floats."""

    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


# --------------------------------------------------------------------------
# Per-image oracle context (caching only; holds no policy)
# --------------------------------------------------------------------------


class OracleImage:
    """Caches the expensive geometric primitive for one image.

    The cache is pure memoisation of ``geometry.component_box_distance`` — a
    symmetric geometric primitive, not a policy decision.
    """

    def __init__(
        self,
        image: G.ImageGeometry,
        config: T.RelationConfig,
        component_map,
        flags_by_component: dict[int, dict[str, bool]] | None = None,
    ) -> None:
        self.image = image
        self.config = config
        self.component_map = component_map
        self._by_id = {c.component_id: c for c in image.components}
        self._gaps: dict[tuple[int, int], float] = {}
        self._flags = flags_by_component if flags_by_component is not None else {
            c.component_id: component_flags(c, image, config) for c in image.components
        }

    # -- basics ---------------------------------------------------------

    @property
    def components(self) -> list:
        return self.image.components

    def component(self, component_id: int):
        return self._by_id.get(int(component_id))

    def flags(self, component_id: int) -> dict[str, bool]:
        return self._flags[int(component_id)]

    def gap(self, a_id: int, b_id: int) -> float:
        key = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
        cached = self._gaps.get(key)
        if cached is None:
            cached = float(G.component_box_distance(self.component_map, key[0], key[1]))
            self._gaps[key] = cached
        return cached

    # -- primitive rankings ---------------------------------------------

    def rank_by_area(self, reverse: bool) -> list:
        """Area ranking; ties broken by component_id (identical to frozen order)."""

        return sorted(self.components, key=lambda c: (c.area_px, c.component_id), reverse=reverse)

    def rank_by_axis(self, axis: str, reverse: bool) -> list:
        attr = "centroid_x" if axis == "x" else "centroid_y"
        return sorted(
            self.components, key=lambda c: (getattr(c, attr), c.component_id), reverse=reverse
        )

    def rank_by_gap(self, anchor_id: int, candidates: Iterable[int]) -> list[tuple[float, int]]:
        """(gap, id) ascending over the given candidates, anchor excluded.

        No eligibility filtering whatsoever: this is the raw perception ranking.
        """

        anchor = self.component(anchor_id)
        if anchor is None:
            return []
        scored: list[tuple[float, int]] = []
        for candidate_id in sorted({int(c) for c in candidates}):
            if candidate_id == anchor_id:
                continue
            if self.component(candidate_id) is None:
                continue
            scored.append((self.gap(anchor_id, candidate_id), candidate_id))
        scored.sort(key=lambda pair: (pair[0], pair[1]))
        return scored


# --------------------------------------------------------------------------
# 1 & 2.  largest / smallest
# --------------------------------------------------------------------------


def resolve_size_extreme(oracle: OracleImage, which: str) -> OracleResolution:
    """Global area extreme over ALL VISIBLE components, then admissibility.

    The universe is every component of the tile. Eligibility is applied only
    afterwards, to decide whether the *global answer* may be used as supervision.
    A different, eligible component is never substituted for it.
    """

    if which not in ("largest", "smallest"):
        raise ValueError(f"which must be 'largest' or 'smallest', got {which!r}")

    image = oracle.image
    config = oracle.config
    components = oracle.components
    if not components:
        return OracleResolution(which, 1, False, None, reason=ORACLE_NO_COMPONENTS)

    ordered = oracle.rank_by_area(reverse=(which == "largest"))
    best = ordered[0]

    detail = {
        "which": which,
        "n_visible": len(components),
        "semantic_area_px": int(best.area_px),
    }

    if len(ordered) >= 2:
        second = ordered[1]
        larger, smaller = (best, second) if best.area_px >= second.area_px else (second, best)
        ratio = larger.area_px / max(smaller.area_px, 1)
        detail["runner_up_area_px"] = int(second.area_px)
        detail["ratio"] = float(ratio)
        detail["ratio_margin"] = config.size_rank.ratio_margin
        if ratio < config.size_rank.ratio_margin:
            return OracleResolution(
                which, 1, False, best.component_id, reason=ORACLE_AMBIGUOUS, detail=detail
            )

    flags = oracle.flags(best.component_id)
    detail["semantic_target_flags"] = sorted(k for k, v in flags.items() if v)
    if not rule_accepts(config.eligibility_for(which), flags):
        return OracleResolution(
            which, 1, False, best.component_id, reason=ORACLE_TARGET_INELIGIBLE, detail=detail
        )

    return OracleResolution(which, 1, True, best.component_id, reason=ORACLE_OK, detail=detail)


# --------------------------------------------------------------------------
# 4.  direction predicate
# --------------------------------------------------------------------------


def direction_holds(oracle: OracleImage, relation: str, subject, reference) -> bool:
    """Re-implementation of the frozen direction predicate.

    ``relation(subject, reference)``: the subject must satisfy the relation with
    respect to the reference. Three frozen conditions (see
    ``configs/spatial_relations_v1.yaml``):

    1. the dominant-axis delta has the required sign;
    2. axis dominance: ``|dominant| >= alpha * |other|``;
    3. minimum normalised margin: ``|dominant| >= tau``.

    The predicate is evaluated directly from centroids and the frozen config; the
    relation engine is not consulted.
    """

    if relation not in DIRECTION_SIGN:
        raise ValueError(f"not a directional relation: {relation}")

    image = oracle.image
    config = oracle.config
    axis, required_sign = DIRECTION_SIGN[relation]

    delta_x = float(subject.centroid_x) - float(reference.centroid_x)
    delta_y = float(subject.centroid_y) - float(reference.centroid_y)
    dominant_raw = delta_x if axis == "dx" else delta_y
    other_raw = delta_y if axis == "dx" else delta_x

    dominant_span = image.width if axis == "dx" else image.height
    other_span = image.height if axis == "dx" else image.width
    dominant_norm = dominant_raw / dominant_span if dominant_span > 0 else 0.0
    other_norm = other_raw / other_span if other_span > 0 else 0.0

    if _sign(dominant_raw) != required_sign:
        return False
    abs_dominant = abs(dominant_norm)
    if abs_dominant < config.direction.alpha * abs(other_norm):
        return False
    if abs_dominant < config.direction.tau:
        return False
    return True


def direction_candidates(oracle: OracleImage, reference_id: int, relation: str) -> tuple[int, ...]:
    """Every visible component satisfying ``relation(candidate, reference)``.

    This is the full semantic set. Eligibility for direction relations rejects
    nothing in the frozen config, and that fact is asserted here rather than
    assumed: if a future config started rejecting components for a direction
    relation, the oracle would still return the complete set and the difference
    would surface as a candidate-set mismatch.
    """

    reference = oracle.component(reference_id)
    if reference is None:
        return ()
    kept = [
        component.component_id
        for component in oracle.components
        if component.component_id != reference_id
        and direction_holds(oracle, relation, component, reference)
    ]
    return tuple(sorted(kept))


# --------------------------------------------------------------------------
# 3 & 5.  nearest, with and without a direction filter
# --------------------------------------------------------------------------


def nearest_margin_fails(
    oracle: OracleImage, ranked: Sequence[tuple[float, int]]
) -> tuple[bool, dict]:
    """Frozen nearest ambiguity rule, applied at rank 1 vs rank 2."""

    if len(ranked) < 2:
        return False, {"comparison": "single_candidate"}
    first, second = ranked[0][0], ranked[1][0]
    margin_abs = float(second - first)
    diagonal = float(oracle.image.diagonal)
    margin_norm = margin_abs / diagonal if diagonal > 0 else 0.0
    fails = (
        margin_norm < oracle.config.nearest.margin_diag_fraction
        or margin_abs < oracle.config.nearest.margin_px_floor
    )
    return fails, {
        "d1_px": float(first),
        "d2_px": float(second),
        "margin_px": margin_abs,
        "margin_diag_fraction": margin_norm,
        "margin_px_floor": oracle.config.nearest.margin_px_floor,
        "margin_diag_threshold": oracle.config.nearest.margin_diag_fraction,
    }


def resolve_nearest(
    oracle: OracleImage,
    anchor_id: int,
    candidates: Sequence[int],
    level: int = 3,
    query_type: str = "nearest",
) -> OracleResolution:
    """Nearest over the supplied candidate set, BEFORE any eligibility filter.

    The semantic answer is the globally closest candidate. It is admissible only
    when (a) the anchor is itself eligible as an anchor, (b) that exact component
    is eligible as a nearest target, and (c) the frozen margin rule passes. A
    farther eligible component is never substituted.
    """

    config = oracle.config
    rule = config.eligibility_for("nearest")
    ranked = tuple(oracle.rank_by_gap(anchor_id, candidates))

    if not ranked:
        return OracleResolution(
            query_type, level, False, None, reference_id=anchor_id, reason=ORACLE_NO_COMPONENTS
        )

    semantic_target = ranked[0][1]
    detail: dict = {
        "anchor_id": int(anchor_id),
        "n_candidates": len(ranked),
        "ranked": [{"id": cid, "gap": float(gap)} for gap, cid in ranked[:6]],
    }

    anchor = oracle.component(anchor_id)
    anchor_ok = anchor is not None and rule_accepts(
        rule, oracle.flags(anchor_id), as_anchor=True
    )
    detail["anchor_eligible"] = bool(anchor_ok)
    if not anchor_ok:
        return OracleResolution(
            query_type,
            level,
            False,
            semantic_target,
            reference_id=anchor_id,
            candidate_ids=tuple(int(c) for c in candidates),
            reason=ORACLE_ANCHOR_INELIGIBLE,
            ranked=ranked,
            detail=detail,
        )

    target_flags = oracle.flags(semantic_target)
    detail["semantic_target_flags"] = sorted(k for k, v in target_flags.items() if v)
    if not rule_accepts(rule, target_flags):
        detail["closer_than_or_equal_ineligible"] = [
            {"id": cid, "gap": float(gap)}
            for gap, cid in ranked
            if gap <= ranked[0][0] + 1e-9 and not rule_accepts(rule, oracle.flags(cid))
        ]
        return OracleResolution(
            query_type,
            level,
            False,
            semantic_target,
            reference_id=anchor_id,
            candidate_ids=tuple(int(c) for c in candidates),
            reason=ORACLE_TARGET_INELIGIBLE,
            ranked=ranked,
            detail=detail,
        )

    fails, margin_detail = nearest_margin_fails(oracle, ranked)
    detail.update(margin_detail)
    if fails:
        return OracleResolution(
            query_type,
            level,
            False,
            semantic_target,
            reference_id=anchor_id,
            candidate_ids=tuple(int(c) for c in candidates),
            reason=ORACLE_MARGIN_FAIL,
            ranked=ranked,
            detail=detail,
        )

    return OracleResolution(
        query_type,
        level,
        True,
        semantic_target,
        reference_id=anchor_id,
        candidate_ids=tuple(int(c) for c in candidates),
        reason=ORACLE_OK,
        ranked=ranked,
        detail=detail,
    )


def admissible_nearest_ids(
    oracle: OracleImage, anchor_id: int, candidates: Sequence[int]
) -> tuple[int, ...]:
    """Re-derivation of the frozen "admissible nearest answer" set.

    Used only for the Level-3 ``trivial_selection`` flag: a chain is trivial when
    the direction filter plus the frozen margin rule leave exactly one admissible
    candidate, so no distance comparison is needed.

    Definition (frozen for v0.1.1, re-derived here from geometry): a candidate is
    admissible when it is nearest-eligible; additionally, the top-ranked
    candidate is dropped when the margin rule fails, because in that case the
    ranking itself is not reliable enough to be an answer.
    """

    rule = oracle.config.eligibility_for("nearest")
    ranked = oracle.rank_by_gap(anchor_id, candidates)
    if not ranked:
        return ()

    out: list[int] = []
    for index, (gap, candidate_id) in enumerate(ranked):
        if not rule_accepts(rule, oracle.flags(candidate_id)):
            continue
        if index == 0 and len(ranked) >= 2:
            second = ranked[1][0]
            margin_abs = float(second - gap)
            diagonal = float(oracle.image.diagonal)
            margin_norm = margin_abs / diagonal if diagonal > 0 else 0.0
            if (
                margin_norm < oracle.config.nearest.margin_diag_fraction
                or margin_abs < oracle.config.nearest.margin_px_floor
            ):
                continue
        out.append(candidate_id)
    return tuple(out)


# --------------------------------------------------------------------------
# 1-5.  whole-query resolution
# --------------------------------------------------------------------------


def resolve_extreme(oracle: OracleImage, query_type: str) -> OracleResolution:
    """Tile-relative extreme: leftmost / rightmost / topmost / bottommost."""

    axis, reverse = EXTREME_AXES[query_type]
    ordered = oracle.rank_by_axis(axis, reverse=reverse)
    if len(ordered) < 2:
        return OracleResolution(
            query_type, 1, False, None, reason=ORACLE_NO_COMPONENTS,
            detail={"n_visible": len(ordered)},
        )

    best, second = ordered[0], ordered[1]
    attr = "centroid_x" if axis == "x" else "centroid_y"
    margin_px = abs(float(getattr(best, attr)) - float(getattr(second, attr)))
    detail = {
        "axis": axis,
        "value_px": float(getattr(best, attr)),
        "second_value_px": float(getattr(second, attr)),
        "margin_px": margin_px,
        "margin_threshold_px": oracle.config.extreme_margin_px,
    }
    if margin_px < oracle.config.extreme_margin_px:
        return OracleResolution(
            query_type, 1, False, best.component_id, reason=ORACLE_AMBIGUOUS, detail=detail
        )

    rule = oracle.config.eligibility_for(query_type)
    if not rule_accepts(rule, oracle.flags(best.component_id)):
        return OracleResolution(
            query_type, 1, False, best.component_id, reason=ORACLE_TARGET_INELIGIBLE, detail=detail
        )
    return OracleResolution(query_type, 1, True, best.component_id, reason=ORACLE_OK, detail=detail)


def resolve_query(
    oracle: OracleImage,
    level: int,
    query_type: str,
) -> OracleResolution:
    """Resolve one query type end to end, from level + query_type alone.

    Deliberately does **not** read the stored ``target_component_id``,
    ``reference_component_ids`` or ``candidate_component_ids``: the answer is
    derived from geometry so a stored value can never verify itself.
    """

    if level == 1:
        if query_type in ("largest", "smallest"):
            return resolve_size_extreme(oracle, query_type)
        if query_type in EXTREME_AXES:
            return resolve_extreme(oracle, query_type)
        return OracleResolution(query_type, level, False, None, reason=ORACLE_UNKNOWN_QUERY)

    if level == 2:
        reference_type, separator, tail = query_type.partition("_to_")
        if not separator or reference_type not in ("largest", "smallest"):
            return OracleResolution(query_type, level, False, None, reason=ORACLE_UNKNOWN_QUERY)

        reference = resolve_size_extreme(oracle, reference_type)
        if not reference.admissible:
            return OracleResolution(
                query_type, level, False, None, reason=reference.reason, detail=reference.detail
            )
        reference_id = int(reference.target_id)

        if tail == "nearest":
            others = [c.component_id for c in oracle.components if c.component_id != reference_id]
            # resolve_nearest already records the anchor as reference_id.
            return resolve_nearest(oracle, reference_id, others, level, query_type)

        if tail in DIRECTION_RELATIONS:
            candidates = direction_candidates(oracle, reference_id, tail)
            if not candidates:
                return OracleResolution(
                    query_type, level, False, None, reference_id=reference_id,
                    reason=ORACLE_NO_DIRECTION_CANDIDATE,
                    detail={"direction_candidates": []},
                )
            if len(candidates) > 1:
                return OracleResolution(
                    query_type, level, False, candidates[0], reference_id=reference_id,
                    candidate_ids=candidates, reason=ORACLE_MULTIPLE_DIRECTION_CANDIDATES,
                    detail={"direction_candidates": list(candidates)},
                )
            return OracleResolution(
                query_type, level, True, candidates[0], reference_id=reference_id,
                candidate_ids=candidates, reason=ORACLE_OK,
                detail={"direction_candidates": list(candidates)},
            )
        return OracleResolution(query_type, level, False, None, reason=ORACLE_UNKNOWN_QUERY)

    if level == 3:
        parts = query_type.split("_to_")
        if len(parts) != 3 or parts[0] not in ("largest", "smallest"):
            return OracleResolution(query_type, level, False, None, reason=ORACLE_UNKNOWN_QUERY)
        reference_type, relation, tail = parts
        if tail != "nearest" or relation not in DIRECTION_RELATIONS:
            return OracleResolution(query_type, level, False, None, reason=ORACLE_UNKNOWN_QUERY)

        reference = resolve_size_extreme(oracle, reference_type)
        if not reference.admissible:
            return OracleResolution(
                query_type, level, False, None, reason=reference.reason, detail=reference.detail
            )
        reference_id = int(reference.target_id)

        filtered = direction_candidates(oracle, reference_id, relation)
        if not filtered:
            return OracleResolution(
                query_type, level, False, None, reference_id=reference_id,
                reason=ORACLE_NO_DIRECTION_CANDIDATE,
                detail={"direction_candidates": []},
            )

        resolution = resolve_nearest(oracle, reference_id, filtered, level, query_type)
        admissible_set = admissible_nearest_ids(oracle, reference_id, filtered)
        return OracleResolution(
            **{
                **_as_kwargs(resolution),
                "reference_id": reference_id,
                "admissible_nearest_ids": admissible_set,
                "detail": {
                    **resolution.detail,
                    "direction_candidates": list(filtered),
                    "admissible_nearest_ids": list(admissible_set),
                },
            }
        )

    return OracleResolution(query_type, level, False, None, reason=ORACLE_UNKNOWN_QUERY)


def _as_kwargs(resolution: OracleResolution) -> dict:
    return {
        "query_type": resolution.query_type,
        "level": resolution.level,
        "admissible": resolution.admissible,
        "target_id": resolution.target_id,
        "reference_id": resolution.reference_id,
        "candidate_ids": resolution.candidate_ids,
        "admissible_nearest_ids": resolution.admissible_nearest_ids,
        "reason": resolution.reason,
        "ranked": resolution.ranked,
        "detail": dict(resolution.detail),
    }


# --------------------------------------------------------------------------
# Record-level comparison
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleComparison:
    """Oracle vs one stored record."""

    sample_id: str
    level: int
    query_type: str
    checked: bool
    target_match: bool
    reference_match: bool
    candidate_set_match: bool
    admissible: bool
    trivial_match: bool | None
    oracle_target: int | None
    oracle_reference: int | None
    oracle_candidates: tuple[int, ...]
    oracle_reason: str
    oracle_admissible_nearest_ids: tuple[int, ...]
    stored_target: int | None
    stored_references: tuple[int, ...]
    stored_candidates: tuple[int, ...]
    stored_trivial: bool | None
    mismatch_kinds: tuple[str, ...]


def stored_candidate_set(level: int, steps: Sequence[dict]) -> tuple[int, ...]:
    """The stored semantic candidate set of a record.

    Level 1 has no candidate set. Levels 2 and 3 record it in step 2, which is
    the step that carries the semantic filter (``all others`` for
    reference->nearest, the direction-valid set otherwise).
    """

    if level == 1:
        return ()
    if len(steps) < 2:
        return ()
    return tuple(int(c) for c in steps[1].get("candidate_component_ids", []))


def compare_record(oracle: OracleImage, record: dict) -> OracleComparison:
    """Independently recompute one record and compare every semantic field."""

    level = int(record["level"])
    query_type = record["query_type"]
    resolution = resolve_query(oracle, level, query_type)

    stored_target = record.get("target_component_id")
    stored_references = tuple(int(r) for r in record.get("reference_component_ids", []))
    stored_candidates = stored_candidate_set(level, record.get("reasoning_steps", []))
    stored_trivial = record.get("trivial_selection") if level == 3 else None

    target_match = resolution.admissible and resolution.target_id == stored_target
    reference_match = (
        resolution.reference_id is None and not stored_references
    ) or (
        resolution.reference_id is not None and tuple(stored_references) == (resolution.reference_id,)
    )
    candidate_set_match = level == 1 or resolution.candidate_ids == stored_candidates

    trivial_match: bool | None = None
    if level == 3:
        trivial_match = bool(stored_trivial) == (len(resolution.admissible_nearest_ids) == 1)

    kinds: list[str] = []
    if not resolution.admissible:
        kinds.append("ambiguity_policy_mismatch")
    if not target_match:
        kinds.append("target_mismatch")
    if not reference_match:
        kinds.append("reference_mismatch")
    if not candidate_set_match:
        kinds.append("candidate_set_mismatch")
    if trivial_match is False:
        kinds.append("trivial_flag_mismatch")

    return OracleComparison(
        sample_id=record.get("sample_id", ""),
        level=level,
        query_type=query_type,
        checked=True,
        target_match=target_match,
        reference_match=reference_match,
        candidate_set_match=candidate_set_match,
        admissible=resolution.admissible,
        trivial_match=trivial_match,
        oracle_target=resolution.target_id,
        oracle_reference=resolution.reference_id,
        oracle_candidates=resolution.candidate_ids,
        oracle_reason=resolution.reason,
        oracle_admissible_nearest_ids=resolution.admissible_nearest_ids,
        stored_target=stored_target,
        stored_references=stored_references,
        stored_candidates=stored_candidates,
        stored_trivial=stored_trivial,
        mismatch_kinds=tuple(kinds),
    )


def oracle_hidden_violations(oracle: OracleImage, record: dict) -> tuple[str, ...]:
    """Independent hidden-eligibility audit for one record.

    Returns the violation codes (empty when clean). These are derived purely from
    the oracle's own geometry reading, never from ``semantic_policy``.

    A record violates visible-component semantics when the stored answer differs
    from what a reader looking at the whole tile would select — i.e. when the
    generator answered over a hidden, eligibility-filtered universe.
    """

    level = int(record["level"])
    query_type = record["query_type"]
    stored_references = tuple(int(r) for r in record.get("reference_component_ids", []))
    stored_target = record.get("target_component_id")
    codes: list[str] = []

    reference_type = query_type.split("_to_", 1)[0] if level >= 2 else query_type
    if reference_type in ("largest", "smallest"):
        extreme = resolve_size_extreme(oracle, reference_type)
        stored_answer = stored_target if level == 1 else (
            stored_references[0] if stored_references else None
        )
        if not extreme.admissible or extreme.target_id != stored_answer:
            codes.append(f"hidden_eligibility_{reference_type}")

    if level == 2 and query_type.endswith("_to_nearest") and stored_references:
        anchor_id = stored_references[0]
        others = [c.component_id for c in oracle.components if c.component_id != anchor_id]
        nearest = resolve_nearest(oracle, anchor_id, others, level, query_type)
        if nearest.target_id != stored_target:
            codes.append("hidden_eligibility_nearest")

    if level == 3 and stored_references:
        anchor_id = stored_references[0]
        relation = query_type.split("_to_")[1]
        filtered = direction_candidates(oracle, anchor_id, relation)
        nearest = resolve_nearest(oracle, anchor_id, filtered, level, query_type)
        if nearest.target_id != stored_target:
            codes.append("hidden_eligibility_level3_nearest")

    return tuple(codes)
