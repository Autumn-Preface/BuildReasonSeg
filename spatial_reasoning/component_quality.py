"""Component quality classification.

Computes per-component quality flags and per-relation eligibility.

Why per-relation eligibility instead of one global flag
------------------------------------------------------
A single ``geometry_valid`` flag applied to every relation would be wrong. The
project's v1 scope is **tile-relative** spatial reasoning:

* ``leftmost`` asks "which component is furthest left **in this tile**". A
  component clipped by the tile edge is still legitimately the leftmost one, so
  border truncation must NOT disqualify it.
* ``largest`` asks about physical size. A border-truncated component's area is a
  lower bound on its true area, and a suspected merged blob is not one building,
  so those ARE disqualified.

Therefore each relation carries its own eligibility rule (see
``configs/spatial_relations_v1.yaml`` -> ``eligibility``), and
``geometry_valid`` is provided only as a general-purpose flag and is **not**
applied uniformly.

The merge flag is an explicitly labelled heuristic, not merge ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thresholds import RelationConfig

# --------------------------------------------------------------------------
# Flag names
# --------------------------------------------------------------------------

FLAG_TOUCHES_IMAGE_BORDER = "touches_image_border"
FLAG_SUSPECTED_LARGE_MERGE = "suspected_large_merge"
FLAG_TINY_COMPONENT = "tiny_component"
FLAG_GEOMETRY_VALID = "geometry_valid"

ALL_FLAGS = (
    FLAG_TOUCHES_IMAGE_BORDER,
    FLAG_SUSPECTED_LARGE_MERGE,
    FLAG_TINY_COMPONENT,
    FLAG_GEOMETRY_VALID,
)


@dataclass(frozen=True)
class QualityFlags:
    """Quality flags for one component, plus the evidence behind each flag."""

    component_id: int
    touches_image_border: bool
    suspected_large_merge: bool
    tiny_component: bool
    geometry_valid: bool
    bbox_extent_ratio: float
    area_ratio: float
    area_px: int

    def as_dict(self) -> dict[str, bool]:
        return {
            FLAG_TOUCHES_IMAGE_BORDER: self.touches_image_border,
            FLAG_SUSPECTED_LARGE_MERGE: self.suspected_large_merge,
            FLAG_TINY_COMPONENT: self.tiny_component,
            FLAG_GEOMETRY_VALID: self.geometry_valid,
        }

    def evidence(self) -> dict[str, float | int]:
        """Numeric basis for the flags, so a reviewer can audit the decision."""

        return {
            "bbox_extent_ratio": self.bbox_extent_ratio,
            "area_ratio": self.area_ratio,
            "area_px": self.area_px,
        }


@dataclass
class ImageQuality:
    """Quality flags for every component of one image."""

    image_id: str
    split: str
    flags_by_component: dict[int, QualityFlags]
    image_area: int
    diagonal: float
    counts: dict[str, int] = field(default_factory=dict)

    def flags(self, component_id: int) -> QualityFlags:
        return self.flags_by_component[component_id]

    def eligible_ids(self, relation: str, config: RelationConfig) -> list[int]:
        """Component ids eligible as anchor/target for a relation.

        Returns ids in ascending order for determinism.
        """

        rule = config.eligibility_for(relation)
        out: list[int] = []
        for component_id in sorted(self.flags_by_component):
            accepted, _ = rule.accepts(self.flags_by_component[component_id].as_dict())
            if accepted:
                out.append(component_id)
        return out

    def rejection_reasons(self, relation: str, config: RelationConfig) -> dict[int, str]:
        """Map component id -> rejection reason for a relation (for auditing)."""

        rule = config.eligibility_for(relation)
        out: dict[int, str] = {}
        for component_id in sorted(self.flags_by_component):
            accepted, reason = rule.accepts(self.flags_by_component[component_id].as_dict())
            if not accepted and reason:
                out[component_id] = reason
        return out


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def classify_component(
    component,
    image_area: int,
    config: RelationConfig,
) -> QualityFlags:
    """Compute quality flags for one component.

    ``geometry_valid`` is the conjunction of "not border truncated", "not a
    suspected merge" and "not tiny". It is a convenience summary and is NOT
    applied uniformly to all relations -- use :meth:`ImageQuality.eligible_ids`
    for per-relation eligibility.
    """

    bbox_extent_ratio = component.bbox_extent_ratio(image_area)

    touches = bool(component.touches_image_border)

    # Heuristic, not ground truth: a component spanning more than the configured
    # fraction of the tile is implausible as a single building.
    suspected_merge = bbox_extent_ratio > config.quality.merge_bbox_extent_ratio

    tiny = component.area_px < config.quality.tiny_area_px

    geometry_valid = not (touches or suspected_merge or tiny)

    return QualityFlags(
        component_id=component.component_id,
        touches_image_border=touches,
        suspected_large_merge=bool(suspected_merge),
        tiny_component=bool(tiny),
        geometry_valid=bool(geometry_valid),
        bbox_extent_ratio=float(bbox_extent_ratio),
        area_ratio=float(component.area_ratio),
        area_px=int(component.area_px),
    )


def classify_image(image_geometry, config: RelationConfig) -> ImageQuality:
    """Compute quality flags for every component in one image."""

    image_area = image_geometry.width * image_geometry.height
    flags_by_component = {
        component.component_id: classify_component(component, image_area, config)
        for component in image_geometry.components
    }

    counts = {flag: 0 for flag in ALL_FLAGS}
    for quality in flags_by_component.values():
        for flag, value in quality.as_dict().items():
            if value:
                counts[flag] += 1
    counts["total"] = len(flags_by_component)

    return ImageQuality(
        image_id=image_geometry.image_id,
        split=image_geometry.split,
        flags_by_component=flags_by_component,
        image_area=image_area,
        diagonal=float(image_geometry.diagonal),
        counts=counts,
    )
