"""Semantic visibility policy for BuildSpatialReason (v0.1.1).

This module is the **single shared implementation** of the corrected semantic
policy, used by:

* the dataset generator       (``spatial_reasoning/annotator.py``)
* the acceptance validator    (``spatial_reasoning/dataset_validator.py``)
* target recomputation

Sharing one implementation is deliberate. In v0.1 the generator and the relation
engine disagreed about what "the largest building" means, which is exactly how
7,086 semantically false records were produced. Here there is one definition, so
they cannot drift apart again.

The policy
----------
**Natural language is defined over ALL VISIBLE COMPONENTS. Eligibility may
reject a sample; it may never silently change the answer.**

v0.1's relation engine answered "the largest *eligible* building", then the
generator presented that as "the largest building". Whenever the true global
extreme was excluded by border/tiny/merge quality logic, the emitted instruction
was false. The correction is to invert the order:

1. compute the **semantic** answer over every visible component;
2. check whether that exact component is admissible (eligibility + frozen margin);
3. if admissible -> keep the query;
4. otherwise -> **discard the query**. Never substitute the runner-up.

This module therefore exposes two things per question:

* ``semantic_target`` — what a reader looking at the image would pick;
* ``admissible``      — whether that answer is usable as supervision.

A sample is only emittable when ``semantic_target == eligible answer`` and the
frozen ambiguity rule passes.

Frozen rules are untouched: ``ratio_margin = 1.10``, direction preset ``medium``,
``nearest`` anchor and target both non-border, ambiguity -> discard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import geometry as G
import thresholds as T
from component_quality import ImageQuality

#: Version of the semantic visibility policy. Recorded in every sample and in
#: the dataset manifest so a future change is auditable.
SEMANTIC_VISIBILITY_POLICY_VERSION = "1.0"

#: Reason codes returned when a semantic answer cannot be used.
REASON_SEMANTIC_TARGET_INELIGIBLE = "semantic_target_ineligible"
REASON_SEMANTIC_AMBIGUOUS = "semantic_ambiguous"
REASON_SEMANTIC_MARGIN_FAIL = "semantic_margin_fail"
REASON_NO_COMPONENTS = "no_visible_components"


@dataclass
class SemanticOutcome:
    """Result of resolving one semantic question over visible components."""

    admissible: bool
    semantic_target: int | None = None
    reason: str | None = None
    evidence: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.admissible


# --------------------------------------------------------------------------
# Size extremes: largest / smallest
# --------------------------------------------------------------------------


def resolve_size_extreme(
    image: G.ImageGeometry,
    which: str,
    config: T.RelationConfig,
    quality: ImageQuality | None = None,
) -> SemanticOutcome:
    """Resolve "the largest / smallest building region" over ALL visible components.

    ``which`` is ``"largest"`` or ``"smallest"``.

    The global extreme is taken over every visible component regardless of
    quality flags. It is then checked against the frozen eligibility rule for
    that relation and the frozen size ambiguity margin. If either check fails the
    outcome is not admissible and the caller must discard the query.

    The runner-up eligible component is **never** returned as a substitute.
    """

    if which not in ("largest", "smallest"):
        raise ValueError(f"which must be 'largest' or 'smallest', got {which!r}")

    quality = quality or _quality_of(image, config)
    components = image.components
    if not components:
        return SemanticOutcome(False, reason=REASON_NO_COMPONENTS)

    reverse = which == "largest"
    ordered = G.rank_by(components, lambda c: c.area_px, reverse=reverse)
    semantic_target = ordered[0]

    evidence = {
        "which": which,
        "semantic_target_id": semantic_target.component_id,
        "semantic_target_area_px": int(semantic_target.area_px),
        "n_visible": len(components),
    }
    if len(ordered) > 1:
        evidence["runner_up_id"] = ordered[1].component_id
        evidence["runner_up_area_px"] = int(ordered[1].area_px)

    # Frozen ambiguity margin. Expressed as larger/smaller so the same threshold
    # means the same thing for both relations.
    if len(ordered) >= 2:
        larger, smaller = ordered[0], ordered[1]
        if larger.area_px < smaller.area_px:
            larger, smaller = smaller, larger
        ratio = larger.area_px / max(smaller.area_px, 1)
        evidence["ratio"] = float(ratio)
        evidence["ratio_threshold"] = config.size_rank.ratio_margin
        if ratio < config.size_rank.ratio_margin:
            return SemanticOutcome(
                False, semantic_target.component_id, REASON_SEMANTIC_AMBIGUOUS, evidence
            )

    # Eligibility of the SEMANTIC target itself.
    flags = quality.flags(semantic_target.component_id).as_dict()
    accepted, rejection = config.eligibility_for(which).accepts(flags)
    evidence["semantic_target_flags"] = [k for k, v in flags.items() if v]
    if not accepted:
        evidence["rejection"] = rejection
        return SemanticOutcome(
            False, semantic_target.component_id, REASON_SEMANTIC_TARGET_INELIGIBLE, evidence
        )

    return SemanticOutcome(True, semantic_target.component_id, None, evidence)


# --------------------------------------------------------------------------
# Nearest over all visible components
# --------------------------------------------------------------------------


def nearest_over_visible(
    image: G.ImageGeometry,
    anchor_id: int,
    candidates: Sequence[int],
    component_map,
    config: T.RelationConfig,
) -> list[tuple[float, int]]:
    """(gap, id) ascending for ``candidates``, using frozen boundary distance.

    No eligibility filtering is applied: this is the raw semantic ranking a
    reader would perceive.
    """

    scored: list[tuple[float, int]] = []
    for candidate_id in sorted(set(int(c) for c in candidates)):
        if candidate_id == anchor_id:
            continue
        scored.append(
            (G.component_box_distance(component_map, anchor_id, candidate_id), candidate_id)
        )
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored


def resolve_nearest(
    image: G.ImageGeometry,
    anchor_id: int,
    candidates: Sequence[int],
    component_map,
    config: T.RelationConfig,
    quality: ImageQuality | None = None,
) -> SemanticOutcome:
    """Resolve "the nearest one" over the supplied visible candidate set.

    The semantic nearest is the closest candidate **before** eligibility
    filtering. It is kept only if that exact component is nearest-eligible and
    the frozen nearest margin rule passes. Otherwise the query is discarded —
    a farther eligible component is never substituted.
    """

    quality = quality or _quality_of(image, config)
    scored = nearest_over_visible(image, anchor_id, candidates, component_map, config)
    if not scored:
        return SemanticOutcome(False, reason=REASON_NO_COMPONENTS)

    d1, semantic_target = scored[0]
    evidence = {
        "anchor_id": int(anchor_id),
        "semantic_target_id": int(semantic_target),
        "semantic_target_gap_px": float(d1),
        "n_candidates": len(scored),
        "ranked": [{"id": cid, "gap": float(gap)} for gap, cid in scored[:6]],
    }

    eligible = set(quality.eligible_ids("nearest", config))
    evidence["semantic_target_nearest_eligible"] = semantic_target in eligible

    if semantic_target not in eligible:
        flags = quality.flags(semantic_target).as_dict()
        evidence["semantic_target_flags"] = [k for k, v in flags.items() if v]
        # Report which nearer-but-ineligible candidates exist, for the audit.
        ineligible_closer = [
            {"id": cid, "gap": float(gap)}
            for gap, cid in scored
            if cid not in eligible and gap < d1 + 1e-9
        ]
        evidence["ineligible_candidates_at_or_before_target"] = ineligible_closer
        return SemanticOutcome(
            False, int(semantic_target), REASON_SEMANTIC_TARGET_INELIGIBLE, evidence
        )

    # Frozen nearest margin rule, applied at the semantic answer.
    if len(scored) >= 2:
        d2 = scored[1][0]
        margin_abs = d2 - d1
        margin_norm = margin_abs / image.diagonal if image.diagonal > 0 else 0.0
        evidence["d2_px"] = float(d2)
        evidence["margin_px"] = float(margin_abs)
        evidence["margin_diag_fraction"] = float(margin_norm)
        evidence["margin_px_floor"] = config.nearest.margin_px_floor
        evidence["margin_diag_threshold"] = config.nearest.margin_diag_fraction
        if (
            margin_norm < config.nearest.margin_diag_fraction
            or margin_abs < config.nearest.margin_px_floor
        ):
            return SemanticOutcome(
                False, int(semantic_target), REASON_SEMANTIC_MARGIN_FAIL, evidence
            )

    return SemanticOutcome(True, int(semantic_target), None, evidence)


# --------------------------------------------------------------------------
# Direction over visible components
# --------------------------------------------------------------------------


def direction_candidates_over_visible(
    image: G.ImageGeometry,
    reference_id: int,
    relation: str,
    config: T.RelationConfig,
    quality: ImageQuality | None = None,
) -> list[int]:
    """Components satisfying ``relation(candidate, reference)`` over all visible ones.

    Uses the frozen predicate convention (ADR-006): the candidate is the SUBJECT
    and the reference is the OBJECT. This applies the frozen direction predicate
    (sign + axis dominance + margin); it does **not** apply nearest eligibility.
    """

    import relations as R

    quality = quality or _quality_of(image, config)
    reference = next((c for c in image.components if c.component_id == reference_id), None)
    if reference is None:
        return []

    kept: list[int] = []
    for subject in image.components:
        if subject.component_id == reference_id:
            continue
        result = R.evaluate_direction(relation, image, subject, reference, config, quality)
        if result.valid:
            kept.append(subject.component_id)
    return sorted(kept)


def nearest_eligible_ids(
    image: G.ImageGeometry,
    anchor_id: int,
    candidates: Sequence[int],
    config: T.RelationConfig,
    quality: ImageQuality | None = None,
) -> list[int]:
    """Subset of ``candidates`` admissible as a nearest target (frozen rule)."""

    quality = quality or _quality_of(image, config)
    eligible = set(quality.eligible_ids("nearest", config))
    return sorted(
        int(c) for c in set(int(x) for x in candidates)
        if int(c) in eligible and int(c) != anchor_id
    )


def admissible_nearest_ids(
    image: G.ImageGeometry,
    anchor_id: int,
    candidates: Sequence[int],
    component_map,
    config: T.RelationConfig,
    quality: ImageQuality | None = None,
) -> list[int]:
    """Candidates that could serve as an admissible nearest answer.

    A Level-3 chain is classified TRIVIAL when the direction filter leaves
    exactly one admissible candidate, because then no distance comparison is
    required. This helper recomputes that set independently of the generator's
    ``trivial_selection`` flag so the validator can cross-check it.

    Definition used here, matching the generator: a candidate is admissible when
    it is nearest-eligible **and** the frozen margin rule holds for it. Because
    the margin rule compares rank 1 against rank 2 over the full ranked list, the
    top-ranked eligible candidate is admissible when it clears the margin, and
    every other eligible candidate is also counted (it could win if all closer
    candidates were ineligible — the generator never does that substitution, but
    counting it keeps this helper a faithful mirror of the filter size).
    """

    quality = quality or _quality_of(image, config)
    eligible_set = set(quality.eligible_ids("nearest", config))
    ranked = nearest_over_visible(image, anchor_id, candidates, component_map, config)
    if not ranked:
        return []

    admissible: list[int] = []
    for index, (gap, candidate_id) in enumerate(ranked):
        if candidate_id not in eligible_set:
            continue
        if index == 0 and len(ranked) >= 2:
            margin = ranked[1][0] - gap
            norm = margin / image.diagonal if image.diagonal > 0 else 0.0
            if (
                norm < config.nearest.margin_diag_fraction
                or margin < config.nearest.margin_px_floor
            ):
                # Top candidate fails the margin: ambiguous, contributes nothing.
                continue
        admissible.append(candidate_id)
    return admissible


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _quality_of(image: G.ImageGeometry, config: T.RelationConfig) -> ImageQuality:
    from component_quality import classify_image

    return classify_image(image, config)
