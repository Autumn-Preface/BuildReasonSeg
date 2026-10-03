"""Deterministic 512×512 reasoning context (Task 8B section 16).

One and only one context is created per sample from the reference building's global geometry, using the frozen
directional anchor positions. Integer positions use round-half-up on the exact float anchor (`int(x + 0.5)`),
documented here and used consistently. There is no adaptive retry ladder: a reference that cannot fit triggers
`E404 reference_too_large_for_rc1_context`, and directional candidates that all fall outside the single context
trigger `E404 directional_candidates_outside_rc1_context`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from buildreasonseg.errors import BuildReasonSegError
from buildreasonseg.runtime.detector import TILE_SIZE, GlobalProposal

CONTEXT_SIZE = 512
ANCHOR_FRACTIONS = {"right_of": (0.30, 0.50), "left_of": (0.70, 0.50),
                    "above": (0.50, 0.70), "below": (0.50, 0.30)}
MAX_REFERENCE_EXTENT = 0.80


def round_half_up(value: float) -> int:
    """Deterministic rounding for anchor pixel positions: floor(value + 0.5)."""

    return int(np.floor(value + 0.5))


def anchor_pixel(direction: str, *, size: int = CONTEXT_SIZE) -> tuple[int, int]:
    if direction not in ANCHOR_FRACTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    fraction_x, fraction_y = ANCHOR_FRACTIONS[direction]
    return round_half_up(fraction_x * size), round_half_up(fraction_y * size)


@dataclass
class ReasoningContext:
    top: int
    left: int
    size: int
    direction: str
    anchor_in_context: tuple[int, int]
    reference_centroid_global: tuple[float, float]
    padding: dict
    reference_too_large: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"origin": [int(self.left), int(self.top)], "size": int(self.size),
                "direction": self.direction,
                "anchor_in_context": [int(self.anchor_in_context[0]), int(self.anchor_in_context[1])],
                "reference_centroid_global": [float(self.reference_centroid_global[0]),
                                              float(self.reference_centroid_global[1])],
                "padding": dict(self.padding), "reference_too_large": self.reference_too_large,
                "notes": list(self.notes)}


def plan_context(reference: GlobalProposal, direction: str, *,
                 size: int = CONTEXT_SIZE) -> ReasoningContext:
    """Anchor the reference centroid at the frozen position and derive the crop origin."""

    top_box, left_box, bottom_box, right_box = reference.global_bbox
    box_height = bottom_box - top_box + 1
    box_width = right_box - left_box + 1
    if box_width > MAX_REFERENCE_EXTENT * size or box_height > MAX_REFERENCE_EXTENT * size:
        context = ReasoningContext(top=0, left=0, size=size, direction=direction,
                                   anchor_in_context=anchor_pixel(direction, size=size),
                                   reference_centroid_global=reference.centroid,
                                   padding={"applied": False}, reference_too_large=True,
                                   notes=["reference bbox exceeds 0.80 * 512"])
        raise BuildReasonSegError(
            "E404",
            detail="reference_too_large_for_rc1_context：参考建筑 bbox 超过 0.80×512，RC1 不实现自适应 "
                   "resize / context ladder。",
            context={"reason": "reference_too_large_for_rc1_context",
                     "context": context.to_dict()})
    anchor_x, anchor_y = anchor_pixel(direction, size=size)
    centroid_y, centroid_x = reference.centroid
    left = round_half_up(centroid_x - anchor_x)
    top = round_half_up(centroid_y - anchor_y)
    return ReasoningContext(top=int(top), left=int(left), size=size, direction=direction,
                           anchor_in_context=(anchor_x, anchor_y),
                           reference_centroid_global=reference.centroid,
                           padding={"applied": None, "mode": "reflect"})


def directional_candidates(reference: GlobalProposal, proposals: list[GlobalProposal],
                           direction: str) -> list[GlobalProposal]:
    """Proposals whose centroid lies in the requested half-plane relative to the reference centroid."""

    ref_y, ref_x = reference.centroid
    candidates = []
    for proposal in proposals:
        if proposal.proposal_id == reference.proposal_id:
            continue
        row, column = proposal.centroid
        if direction == "right_of" and column > ref_x:
            candidates.append(proposal)
        elif direction == "left_of" and column < ref_x:
            candidates.append(proposal)
        elif direction == "above" and row < ref_y:
            candidates.append(proposal)
        elif direction == "below" and row > ref_y:
            candidates.append(proposal)
    return candidates


def candidate_inside_context(proposal: GlobalProposal, context: ReasoningContext) -> bool:
    row, column = proposal.centroid
    return (context.top <= row < context.top + context.size
            and context.left <= column < context.left + context.size)


def guard_directional_candidates(reference: GlobalProposal, proposals: list[GlobalProposal],
                                 direction: str, context: ReasoningContext) -> dict:
    """Diagnostic guard: report (and fail) when every directional candidate is outside the context."""

    candidates = directional_candidates(reference, proposals, direction)
    inside = [proposal for proposal in candidates if candidate_inside_context(proposal, context)]
    report = {"direction": direction, "candidate_count": len(candidates),
              "inside_context": len(inside),
              "candidate_ids": [proposal.proposal_id for proposal in candidates]}
    if candidates and not inside:
        raise BuildReasonSegError(
            "E404",
            detail="directional_candidates_outside_rc1_context：该方向上存在候选建筑，但全部位于本次 "
                   "512 reasoning context 之外；RC1 不扩大 context。",
            context={"reason": "directional_candidates_outside_rc1_context", "guard": report})
    return report


def context_to_global(mask_context: np.ndarray, context: ReasoningContext,
                      image_size: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Map a 512 context mask back to original image pixels, cropping away the padded region."""

    height, width = image_size
    full = np.zeros((height, width), dtype=bool)
    valid_top = max(0, context.top)
    valid_left = max(0, context.left)
    valid_bottom = min(height, context.top + context.size)
    valid_right = min(width, context.left + context.size)
    source_top = valid_top - context.top
    source_left = valid_left - context.left
    crop = mask_context[source_top:source_top + (valid_bottom - valid_top),
                        source_left:source_left + (valid_right - valid_left)]
    full[valid_top:valid_bottom, valid_left:valid_right] = crop
    padding = {"top": max(0, -context.top), "left": max(0, -context.left),
               "bottom": max(0, (context.top + context.size) - height),
               "right": max(0, (context.left + context.size) - width)}
    padding["applied"] = bool(padding["top"] or padding["left"] or padding["bottom"]
                              or padding["right"])
    return full, padding


def direction_satisfied(target_centroid: tuple[float, float], reference_centroid: tuple[float, float],
                        direction: str) -> bool:
    """Hard post-inference check only (section 22); no 'pick the best' logic."""

    target_row, target_column = target_centroid
    reference_row, reference_column = reference_centroid
    if direction == "right_of":
        return target_column > reference_column
    if direction == "left_of":
        return target_column < reference_column
    if direction == "above":
        return target_row < reference_row
    if direction == "below":
        return target_row > reference_row
    raise ValueError(f"unknown direction {direction!r}")


__all__ = ["ANCHOR_FRACTIONS", "CONTEXT_SIZE", "MAX_REFERENCE_EXTENT", "ReasoningContext",
           "anchor_pixel", "candidate_inside_context", "context_to_global", "direction_satisfied",
           "directional_candidates", "guard_directional_candidates", "plan_context", "round_half_up"]
