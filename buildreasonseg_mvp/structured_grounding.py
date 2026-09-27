"""Task 6J: Structured Proposal Grounding — reusable core API.

Task 6J pivots from direct pixel grounding to the project's original structured route:

    instruction -> canonical relation program -> building candidates
    -> explicit geometry execution (frozen Task 3B relation engine) -> selected building mask

This module owns the pieces that are independent of any model:

* the canonical program vocabulary, derived 1:1 from the ACTUAL frozen BuildSpatialReason
  v0.1.1 ``query_type`` values (20 programs; never invented semantics);
* a candidate representation that can be filled from the frozen component maps (oracle,
  diagnostic-only) or from predicted instance proposals (YOLO, inference-valid);
* a compact, deterministic program executor that consumes ONLY a program and candidate
  geometry and returns one candidate id. It reuses the frozen relation engine
  (``spatial_reasoning/relations.py`` + ``thresholds.py`` + ``component_quality.py``) and
  mirrors ``spatial_reasoning/annotator.recompute_target_from_steps`` exactly — the
  equivalence is asserted on real data in the J0 stage, not assumed.

The executor never reads the target component id, GT reasoning text, or the target mask:
GT is used exclusively for post-selection scoring by the callers.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "spatial_reasoning") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "spatial_reasoning"))

import geometry as G  # noqa: E402
import relations as R  # noqa: E402
from component_quality import classify_image  # noqa: E402
from thresholds import load_config as load_relation_config  # noqa: E402

#: The canonical program ids are exactly the frozen dataset query types (1:1; asserted in
#: ``build_program_spec`` against the real records, so no semantics are invented here).
EXPECTED_QUERY_TYPES = (
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "largest",
    "smallest",
    "largest_to_nearest",
    "smallest_to_nearest",
    "largest_to_above",
    "largest_to_below",
    "largest_to_left_of",
    "largest_to_right_of",
    "smallest_to_above",
    "smallest_to_below",
    "smallest_to_left_of",
    "smallest_to_right_of",
    "largest_to_above_to_nearest",
    "largest_to_below_to_nearest",
    "largest_to_left_of_to_nearest",
    "largest_to_right_of_to_nearest",
)

#: Directional relations used inside ``filter_relation`` steps, keyed by the query-type suffix.
DIRECTION_BY_SUFFIX = {
    "above": "above",
    "below": "below",
    "left_of": "left_of",
    "right_of": "right_of",
}

#: Extreme operations for the six level-1 programs.
EXTREME_OPERATION = {
    "leftmost": "argmin_centroid_x",
    "rightmost": "argmax_centroid_x",
    "topmost": "argmin_centroid_y",
    "bottommost": "argmax_centroid_y",
    "largest": "argmax_area",
    "smallest": "argmin_area",
}

#: size rank relation used for the size-extreme steps (frozen eligibility names).
SIZE_RELATION = {"largest": "largest", "smallest": "smallest"}


# ------------------------------------------------------------------ program templates


def canonical_program_template(query_type: str) -> list[dict]:
    """The deterministic program template of one frozen query type.

    References use symbolic roles: ``"@1"`` = the single output of step 1 (the anchor
    established by the first argument step). Candidate sources are ``"all"`` (every visible
    component) or ``"previous"`` (the previous filter step's kept set). ``build_program_spec``
    verifies these templates against the actual stored ``reasoning_steps`` of every record.
    """

    if query_type in EXTREME_OPERATION:
        return [{"operation": EXTREME_OPERATION[query_type]}]

    # Level 3: X_to_{direction}_to_nearest (checked before the shorter suffixes).
    for suffix in DIRECTION_BY_SUFFIX:
        if query_type.endswith(f"_to_{suffix}_to_nearest"):
            base = query_type[: -len(f"_to_{suffix}_to_nearest")]
            return [
                {"operation": EXTREME_OPERATION[base]},
                {
                    "operation": "filter_relation",
                    "relation": DIRECTION_BY_SUFFIX[suffix],
                    "reference_component_id": "@1",
                    "input_component_ids": "all",
                },
                {
                    "operation": "argmin_boundary_distance",
                    "reference_component_id": "@1",
                    "candidate_component_ids": "previous",
                },
            ]

    # Level 2 type-B: X_to_{direction}.
    for suffix in DIRECTION_BY_SUFFIX:
        if query_type.endswith(f"_to_{suffix}"):
            base = query_type[: -len(f"_to_{suffix}")]
            return [
                {"operation": EXTREME_OPERATION[base]},
                {
                    "operation": "filter_relation",
                    "relation": DIRECTION_BY_SUFFIX[suffix],
                    "reference_component_id": "@1",
                    "input_component_ids": "all",
                },
            ]

    # Level 2 type-A: X_to_nearest.
    if query_type.endswith("_to_nearest"):
        base = query_type[: -len("_to_nearest")]
        return [
            {"operation": EXTREME_OPERATION[base]},
            {
                "operation": "argmin_boundary_distance",
                "reference_component_id": "@1",
                "candidate_component_ids": "all",
            },
        ]

    raise ValueError(f"unknown query type: {query_type}")


def build_program_spec(records: Sequence[dict]) -> dict:
    """Section 2: canonical program vocabulary from the ACTUAL frozen query types.

    Asserts (a) every record's query type is in the expected set, (b) every query type maps
    to exactly one operation pattern across the supplied records, and (c) the template's
    operations match that pattern. Returns the spec payload for
    ``evaluation/task6j_program_spec.json`` (counts are filled in by the caller).
    """

    from collections import Counter, defaultdict

    patterns: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for record in records:
        query_type = str(record["query_type"])
        if query_type not in EXPECTED_QUERY_TYPES:
            raise ValueError(f"query type outside the frozen vocabulary: {query_type}")
        patterns[query_type].add(
            tuple(step["operation"] for step in record["reasoning_steps"])
        )

    programs = {}
    for query_type in EXPECTED_QUERY_TYPES:
        observed = patterns.get(query_type)
        if not observed:
            continue  # query type absent from this record set (possible for small subsets)
        if len(observed) != 1:
            raise ValueError(
                f"{query_type}: expected one operation pattern, found {sorted(observed)}"
            )
        operations = list(observed.pop())
        template = canonical_program_template(query_type)
        template_ops = [step["operation"] for step in template]
        if operations != template_ops:
            raise ValueError(
                f"{query_type}: stored operations {operations} != template {template_ops}"
            )
        programs[query_type] = {
            "program_id": query_type,
            "query_type": query_type,
            "operations": [
                {
                    key: value
                    for key, value in step.items()
                    if key
                    in (
                        "operation",
                        "relation",
                        "reference_component_id",
                        "input_component_ids",
                        "candidate_component_ids",
                    )
                }
                for step in template
            ],
            "reference_roles": ["@1"] if len(template) > 1 else [],
            "output_role": "single_candidate"
            if template[-1]["operation"] != "filter_relation"
            else "unique_filtered_candidate",
            "level": _level_for(query_type),
        }
    return programs


def _level_for(query_type: str) -> int:
    if query_type in EXTREME_OPERATION:
        return 1
    if query_type.endswith("_to_nearest"):
        return 3 if query_type.count("_to_") == 2 else 2
    return 2


# ------------------------------------------------------------------ candidates


@dataclass
class Candidate:
    """One building candidate with the geometry the relation engine consumes."""

    candidate_id: int
    mask: np.ndarray
    bbox_xyxy_px: tuple[int, int, int, int]
    centroid_px: tuple[float, float]
    area_px: int
    touches_image_border: bool
    confidence: float | None = None
    source: str = "oracle"

    def as_component(self) -> G.Component:
        x0, y0, x1, y1 = self.bbox_xyxy_px
        return G.Component(
            component_id=int(self.candidate_id),
            source_polygon_index=int(self.candidate_id),
            area_px=int(self.area_px),
            area_ratio=float(self.area_px) / max(1, int(self.mask.size)),
            centroid_px=(float(self.centroid_px[0]), float(self.centroid_px[1])),
            bbox_xyxy_px=(int(x0), int(y0), int(x1), int(y1)),
            width_px=max(1, int(x1) - int(x0)),
            height_px=max(1, int(y1) - int(y0)),
            touches_image_border=bool(self.touches_image_border),
            continuous_polygon_area_px=float(self.area_px),
        )


@dataclass
class CandidateSet:
    """All building candidates of one image plus the id-per-pixel label map."""

    width: int
    height: int
    candidates: list[Candidate]
    label_map: np.ndarray = field(repr=False)
    source: str = "oracle"

    @classmethod
    def from_component_map(cls, component_map: np.ndarray, geometry: G.ImageGeometry) -> "CandidateSet":
        """Oracle candidates (diagnostic-only): every visible building component."""

        height, width = component_map.shape
        candidates = []
        for component in geometry.components:
            mask = component_map == component.component_id
            if not mask.any():
                continue
            candidates.append(
                Candidate(
                    candidate_id=component.component_id,
                    mask=mask,
                    bbox_xyxy_px=component.bbox_xyxy_px,
                    centroid_px=component.centroid_px,
                    area_px=component.area_px,
                    touches_image_border=component.touches_image_border,
                    confidence=None,
                    source="oracle",
                )
            )
        return cls(width=width, height=height, candidates=candidates, label_map=component_map, source="oracle")

    @classmethod
    def from_proposals(cls, width: int, height: int, proposals: Sequence[dict]) -> "CandidateSet":
        """Predicted candidates: proposal masks only; no GT touches this path."""

        label_map = np.zeros((height, width), dtype=np.uint8)
        candidates = []
        # Assign in ASCENDING confidence order so higher-confidence proposals overwrite lower
        # ones on overlapping pixels in the label map (documented duplicate-proposal policy).
        ordered = sorted(
            proposals, key=lambda p: float(p.get("confidence", 0.0) or 0.0)
        )
        for index, proposal in enumerate(ordered, start=1):
            mask = np.asarray(proposal["mask"], dtype=bool)
            if mask.shape != (height, width):
                raise ValueError(f"proposal mask {mask.shape} != image {(height, width)}")
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                continue
            bbox = (
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            )
            touches = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == width - 1 or ys.max() == height - 1)
            candidates.append(
                Candidate(
                    candidate_id=index,
                    mask=mask,
                    bbox_xyxy_px=bbox,
                    centroid_px=(float(xs.mean()), float(ys.mean())),
                    area_px=int(mask.sum()),
                    touches_image_border=touches,
                    confidence=float(proposal.get("confidence") or 0.0),
                    source="predicted",
                )
            )
            label_map[mask] = index
        return cls(width=width, height=height, candidates=candidates, label_map=label_map, source="predicted")

    # -- frozen geometry bridge -----------------------------------------

    def as_geometry(self) -> G.ImageGeometry:
        return G.ImageGeometry(
            image_id="candidates",
            split="val",
            width=self.width,
            height=self.height,
            component_map_path="",
            components=[candidate.as_component() for candidate in self.candidates],
        )

    def by_id(self) -> dict[int, Candidate]:
        return {candidate.candidate_id: candidate for candidate in self.candidates}


# ------------------------------------------------------------------ executor


#: Stable abstention reasons for the executor (reported under "abstentions/ambiguities").
ABSTAIN_NO_ELIGIBLE = "no_eligible_candidates"
ABSTAIN_EXTREME_INVALID = "extreme_relation_invalid"
ABSTAIN_REFERENCE_UNRESOLVED = "reference_unresolved"
ABSTAIN_FILTER_MULTI = "filter_multi_candidate"
ABSTAIN_NEAREST_INVALID = "nearest_relation_invalid"
ABSTAIN_FINAL_EMPTY = "final_step_has_no_output"


@dataclass
class ExecutionResult:
    """The executor's answer: one candidate id, or a documented abstention."""

    selected_id: int | None
    abstained: bool
    reason: str | None = None
    trace: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "selected_id": self.selected_id,
            "abstained": self.abstained,
            "reason": self.reason,
            "trace": self.trace,
        }


def execute_program(program: Sequence[dict], candidates: CandidateSet,
                    config=None) -> ExecutionResult:
    """Execute one canonical program over candidate geometry; return one candidate id.

    Section 4 contract: the executor consumes ONLY the program and the candidate geometry.
    It never reads the target component id, GT reasoning text, or the target mask. Steps may
    carry explicit instance ids (the J0 oracle-program form) or symbolic role references
    (``"@k"``, the predicted-program form); the operation semantics are the frozen relation
    engine's, mirroring ``spatial_reasoning/annotator.recompute_target_from_steps`` exactly.
    """

    config = config or load_relation_config()
    geometry = candidates.as_geometry()
    quality = classify_image(geometry, config)
    component_map = candidates.label_map
    by_id = geometry.by_id()

    final_step_is_filter = bool(program) and program[-1]["operation"] == "filter_relation"

    previous_outputs: list[int] = []
    single_outputs: dict[int, int] = {}
    trace: list[dict] = []

    def _resolve_reference(step: dict, step_index: int) -> int | None:
        reference = step.get("reference_component_id")
        if isinstance(reference, str):
            if not reference.startswith("@"):
                return None
            anchor_step = int(reference[1:]) - 1
            if anchor_step not in single_outputs:
                return None
            return single_outputs[anchor_step]
        if reference is None:
            return None
        return int(reference)

    def _candidate_ids(step: dict) -> list[int]:
        stored = step.get("candidate_component_ids")
        if stored == "all":
            return [component.component_id for component in geometry.components]
        if stored == "previous":
            return list(previous_outputs)
        if stored:
            return [int(value) for value in stored]
        return list(previous_outputs)

    for step_index, step in enumerate(program):
        operation = step["operation"]
        entry: dict = {"step": step_index + 1, "operation": operation}

        if operation == "argmax_area" or operation == "argmin_area":
            relation = SIZE_RELATION["largest" if operation == "argmax_area" else "smallest"]
            eligible = set(quality.eligible_ids(relation, config))
            pool = [c for c in geometry.components if c.component_id in eligible]
            entry["n_eligible"] = len(pool)
            if not pool:
                return ExecutionResult(None, True, ABSTAIN_NO_ELIGIBLE, [*trace, entry])
            reverse = operation == "argmax_area"
            best = G.rank_by(pool, lambda c: c.area_px, reverse=reverse)[0]
            outputs = [best.component_id]
            entry["output"] = int(best.component_id)
            entry["area_px"] = int(best.area_px)

        elif operation in ("argmin_centroid_x", "argmax_centroid_x", "argmin_centroid_y", "argmax_centroid_y"):
            relation = {
                "argmin_centroid_x": "leftmost",
                "argmax_centroid_x": "rightmost",
                "argmin_centroid_y": "topmost",
                "argmax_centroid_y": "bottommost",
            }[operation]
            result = R.evaluate_extreme(relation, geometry, config, quality)
            entry["valid"] = result.valid
            entry["reason"] = result.reason
            if not result.valid:
                return ExecutionResult(None, True, ABSTAIN_EXTREME_INVALID, [*trace, entry])
            outputs = [int(result.subject)]
            entry["output"] = int(result.subject)

        elif operation == "filter_relation":
            reference = _resolve_reference(step, step_index)
            if reference is None:
                return ExecutionResult(None, True, ABSTAIN_REFERENCE_UNRESOLVED, [*trace, entry])
            relation = str(step["relation"])
            subjects_spec = step.get("input_component_ids")
            if subjects_spec == "all" or not subjects_spec:
                subjects = [component.component_id for component in geometry.components]
            else:
                subjects = [int(value) for value in subjects_spec]
            kept: list[int] = []
            for subject_id in sorted(set(subjects)):
                if subject_id == reference:
                    continue
                subject = by_id.get(subject_id)
                reference_component = by_id.get(reference)
                if subject is None or reference_component is None:
                    continue
                result = R.evaluate_direction(
                    relation, geometry, subject, reference_component, config, quality
                )
                if result.valid:
                    kept.append(int(subject_id))
            previous_outputs = sorted(kept)
            outputs = sorted(kept)
            entry["kept"] = sorted(kept)
            entry["n_kept"] = len(kept)

        elif operation == "argmin_boundary_distance":
            reference = _resolve_reference(step, step_index)
            if reference is None:
                return ExecutionResult(None, True, ABSTAIN_REFERENCE_UNRESOLVED, [*trace, entry])
            within = R.nearest_within(
                geometry, reference, _candidate_ids(step), config, quality, component_map
            )
            entry["valid"] = within.valid
            entry["reason"] = within.reason
            entry["trivial_selection"] = within.trivial_selection
            if not within.valid:
                return ExecutionResult(None, True, ABSTAIN_NEAREST_INVALID, [*trace, entry])
            outputs = [int(within.target)]
            entry["output"] = int(within.target)

        else:
            raise ValueError(f"unknown operation: {operation}")

        if len(outputs) == 1:
            single_outputs[step_index] = int(outputs[0])
        trace.append(entry)

    if final_step_is_filter:
        if len(previous_outputs) == 1:
            return ExecutionResult(int(previous_outputs[0]), False, None, trace)
        reason = ABSTAIN_FILTER_MULTI if previous_outputs else ABSTAIN_NO_ELIGIBLE
        return ExecutionResult(None, True, reason, trace)

    if not outputs:
        return ExecutionResult(None, True, ABSTAIN_FINAL_EMPTY, trace)
    return ExecutionResult(int(outputs[0]), False, None, trace)


# ------------------------------------------------------------------ reusable API (Part K)


def program_id_for_query_type(query_type: str) -> str:
    """The canonical program id is the frozen query type (1:1 vocabulary)."""

    if query_type not in EXPECTED_QUERY_TYPES:
        raise ValueError(f"unknown query type: {query_type}")
    return query_type


def parse_program(instruction: str, parser, template_map) -> dict:
    """Predict the canonical program id for one instruction (ProgramHead branch, J2+).

    ``parser`` is the trained Qwen ProgramHead runtime; ``template_map`` maps program ids to
    templates. This wrapper is inference-only: query_type never enters as input, only as the
    classifier's output space.
    """

    program_id = parser.predict(instruction)
    return {"program_id": program_id, "program": template_map[program_id]}


def execute_program_by_id(program_id: str, candidates: CandidateSet, config=None) -> ExecutionResult:
    """Execute a canonical program identified by its frozen program id."""

    return execute_program(canonical_program_template(program_id), candidates, config=config)


def predict_structured_mask(image: np.ndarray, instruction: str, *, parser,
                            proposal_loader, template_map, config=None) -> dict:
    """Part K: predicted program + predicted candidates -> selected proposal mask.

    ``proposal_loader(image) -> CandidateSet`` is the (frozen YOLO) candidate extractor.
    No GT geometry, mask or id enters this function.
    """

    parsed = parse_program(instruction, parser, template_map)
    candidates = proposal_loader(image)
    result = execute_program_by_id(parsed["program_id"], candidates, config=config)
    mask = None
    if result.selected_id is not None:
        candidate = candidates.by_id()[result.selected_id]
        mask = np.asarray(candidate.mask, dtype=bool)
    return {
        "program_id": parsed["program_id"],
        "execution": result.as_dict(),
        "mask": mask,
        "candidate_count": len(candidates.candidates),
    }
