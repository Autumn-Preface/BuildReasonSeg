"""BuildSpatialReason dataset generator (v0.1).

Core data chain::

    source image
      -> building component representation   (datasets/whu, Task 2)
      -> frozen Spatial Relation Engine      (spatial_reasoning, Task 3A/3B)
      -> structured reasoning program        (reasoning_steps -- SOURCE OF TRUTH)
      -> deterministic target
      -> bilingual natural-language instruction

Design rules enforced here
--------------------------
1. **The structured reasoning program is the source of truth.** Natural-language
   reasoning is rendered *from* ``reasoning_steps``; steps are never inferred
   from text. ``recompute_target_from_steps`` re-executes the stored steps and
   must reproduce ``target_component_id``.
2. **No LLM-guessed targets.** Every target is produced by the relation engine
   plus argmax/argmin over the stored geometry.
3. **Frozen thresholds.** All thresholds come from
   ``configs/spatial_relations_v1.yaml``. This module never overrides one.
4. **Ambiguity is discarded, never tie-broken.** Any ambiguous or invalid
   relation removes the candidate query.
5. **No hidden disambiguation.** A Level-2 direction query is emitted only when
   exactly one candidate satisfies the stated relation. If more than one does,
   the query is discarded rather than reduced by an unstated criterion (ADR-004).
6. **Model-agnostic.** No ``[SEG]`` / ``[REF]`` or other architecture-specific
   token appears in any instruction or annotation field.
7. **Determinism.** Template choice, sample ids, and any downsampling use
   SHA256, never Python's salted ``hash()``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import geometry as G
import relations as R
import templates as TP
import thresholds as T
from component_quality import ImageQuality, classify_image

GENERATOR_VERSION = "v0.1"

#: Version string written into every sample for the relation layer.
RELATION_CONFIG_VERSION = "spatial_relations_v1"

#: Level-1 query type -> (operation, natural-language extreme name)
LEVEL1_OPERATIONS: dict[str, str] = {
    "leftmost": "argmin_centroid_x",
    "rightmost": "argmax_centroid_x",
    "topmost": "argmin_centroid_y",
    "bottommost": "argmax_centroid_y",
    "largest": "argmax_area",
    "smallest": "argmin_area",
}

#: Operation -> the size/extreme relation whose frozen validity gates it.
LEVEL1_GATING_RELATION: dict[str, str] = {
    "leftmost": "leftmost",
    "rightmost": "rightmost",
    "topmost": "topmost",
    "bottommost": "bottommost",
    "largest": "largest",
    "smallest": "smallest",
}

#: Module-relative path used for the source-of-truth image/geometry metadata.
IMAGE_METADATA_TEMPLATE = "datasets/whu/metadata/{split}.jsonl"

#: Noun used when describing a component in generated language. It maps to a
#: building CONNECTED COMPONENT, not a verified physical building.
COMPONENT_NOUN_ZH = "建筑区域"
COMPONENT_NOUN_EN = "building region"


# --------------------------------------------------------------------------
# Discard accounting
# --------------------------------------------------------------------------

DISCARD_REASONS = (
    "ambiguous",
    "no_reference",
    "no_direction_candidate",
    "multiple_direction_candidates",
    "nearest_margin_fail",
    "border_ineligible",
    "duplicate",
    "quota_downsample",
    "single_component_image",
    "level_disabled",
)


@dataclass
class DiscardCounter:
    """Counts why candidate queries were dropped. Every drop is attributable."""

    counts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in DISCARD_REASONS})

    def add(self, reason: str, n: int = 1) -> None:
        if reason not in self.counts:
            self.counts[reason] = 0
        self.counts[reason] += n

    def as_dict(self) -> dict[str, int]:
        return {k: v for k, v in sorted(self.counts.items()) if v}


# --------------------------------------------------------------------------
# Structured reasoning program
# --------------------------------------------------------------------------


def semantic_key(image_id: str, steps: Sequence[dict], target_id: int) -> str:
    """Canonical identity of a query: image + program + target.

    Two queries with the same semantic key are the SAME question even if a
    different template would word it differently, so only one canonical record
    may exist.
    """

    payload = json.dumps(
        {
            "image_id": image_id,
            "steps": _canonical_steps(steps),
            "target": int(target_id),
            "generator_version": GENERATOR_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_steps(steps: Sequence[dict]) -> list[dict]:
    """Reduce steps to the fields that define the program's meaning."""

    out = []
    for step in steps:
        entry = {"operation": step["operation"]}
        for key in (
            "relation",
            "reference_component_id",
            "candidate_component_ids",
            "output_component_id",
            "unique_candidate",
        ):
            if key in step and step[key] is not None:
                value = step[key]
                if isinstance(value, list):
                    value = sorted(int(v) for v in value)
                entry[key] = value
        out.append(entry)
    return out


def make_sample_id(split: str, image_id: str, level: int, query_type: str, steps: Sequence[dict], target_id: int) -> str:
    """Deterministic sample id.

    Hash inputs: image id, structured reasoning program, target component id,
    generator version. Re-running the generator reproduces the same ids.
    """

    digest = semantic_key(image_id, steps, target_id)
    safe_type = query_type.replace("/", "-")
    return f"buildsr_{split}_{image_id}_{level}_{safe_type}_{digest[:12]}"


def recompute_target_from_steps(
    image: G.ImageGeometry,
    steps: Sequence[dict],
    config: T.RelationConfig,
    quality: ImageQuality | None = None,
    component_map: np.ndarray | None = None,
) -> int | None:
    """Re-execute a stored reasoning program and return the final component id.

    This is the independent check that ``reasoning_steps`` really determine the
    target. It is also what makes the program the source of truth: the generator
    stores steps, and the target is whatever this function computes.
    """

    quality = quality or classify_image(image, config)
    outputs: list[int] = []
    # A `filter_relation` step is a *set* operation: it may leave exactly one
    # component, in which case that component IS the answer (this is the Level-2
    # Type-B case). Only a genuinely multi-valued final set has no single target.
    final_step_is_filter = bool(steps) and steps[-1]["operation"] == "filter_relation"

    for step in steps:
        operation = step["operation"]

        if operation == "argmax_area":
            eligible = set(quality.eligible_ids("largest", config))
            pool = [c for c in image.components if c.component_id in eligible]
            if not pool:
                return None
            outputs = [G.rank_by(pool, lambda c: c.area_px, reverse=True)[0].component_id]

        elif operation == "argmin_area":
            eligible = set(quality.eligible_ids("smallest", config))
            pool = [c for c in image.components if c.component_id in eligible]
            if not pool:
                return None
            outputs = [G.rank_by(pool, lambda c: c.area_px)[0].component_id]

        elif operation in ("argmin_centroid_x", "argmax_centroid_x", "argmin_centroid_y", "argmax_centroid_y"):
            relation = {
                "argmin_centroid_x": "leftmost",
                "argmax_centroid_x": "rightmost",
                "argmin_centroid_y": "topmost",
                "argmax_centroid_y": "bottommost",
            }[operation]
            result = R.evaluate_extreme(relation, image, config, quality)
            if not result.valid:
                return None
            outputs = [result.subject]

        elif operation == "filter_relation":
            # Filters the CURRENT output set by the stated relation against the
            # stated reference. Uses the frozen engine, so predicate semantics
            # are exactly the frozen ones: relation(subject, reference).
            reference = step["reference_component_id"]
            relation = step["relation"]
            subjects = step.get("input_component_ids")
            if subjects is None:
                subjects = [c.component_id for c in image.components]
            kept = []
            for subject_id in sorted(set(int(s) for s in subjects)):
                if subject_id == reference:
                    continue
                subject = next((c for c in image.components if c.component_id == subject_id), None)
                if subject is None:
                    continue
                reference_component = next((c for c in image.components if c.component_id == reference), None)
                if reference_component is None:
                    continue
                # subject is the subject; reference is the object.
                result = R.evaluate_direction(
                    relation, image, subject, reference_component, config, quality
                )
                if result.valid:
                    kept.append(subject_id)
            outputs = sorted(kept)

        elif operation == "argmin_boundary_distance":
            anchor = step["reference_component_id"]
            candidates = step.get("candidate_component_ids") or outputs
            within = R.nearest_within(image, anchor, candidates, config, quality, component_map)
            if not within.valid:
                return None
            outputs = [within.target]

        else:
            raise ValueError(f"unknown operation: {operation}")

    if final_step_is_filter:
        # A `filter_relation` is a SET operation. It yields a target only when it
        # leaves exactly one component, and that uniqueness is the whole reason
        # the query is answerable: there is no hidden tie-break. Returning None
        # for a multi-valued set is deliberate -- never pick one arbitrarily.
        if len(outputs) == 1:
            return outputs[0]
        return None

    return outputs[0] if len(outputs) == 1 else None


def filter_relation_candidates(
    image: G.ImageGeometry,
    step: dict,
    config: T.RelationConfig,
    quality: ImageQuality | None = None,
) -> list[int]:
    """Independently recompute the candidate set of a ``filter_relation`` step.

    Used to audit a stored program against the relation engine *without*
    trusting the stored ``candidate_component_ids``: the stored list is only
    ever compared against this recomputation.
    """

    quality = quality or classify_image(image, config)
    reference = step["reference_component_id"]
    relation = step["relation"]
    subjects = step.get("input_component_ids") or [c.component_id for c in image.components]

    reference_component = next(
        (c for c in image.components if c.component_id == reference), None
    )
    if reference_component is None:
        return []

    kept: list[int] = []
    for subject_id in sorted(set(int(s) for s in subjects)):
        if subject_id == reference:
            continue
        subject = next((c for c in image.components if c.component_id == subject_id), None)
        if subject is None:
            continue
        # subject is the subject, reference is the object: the frozen convention.
        result = R.evaluate_direction(relation, image, subject, reference_component, config, quality)
        if result.valid:
            kept.append(subject_id)
    return kept


# --------------------------------------------------------------------------
# Natural-language reasoning rendered FROM the steps
# --------------------------------------------------------------------------


def render_reasoning(steps: Sequence[dict]) -> tuple[str, str]:
    """Render short, verifiable reasoning text from the structured steps.

    Every statement is directly checkable against ``reasoning_steps``. This is
    deliberately not a long chain of thought.
    """

    zh: list[str] = []
    en: list[str] = []

    for step_index, step in enumerate(steps):
        operation = step["operation"]

        if operation in TP.OPERATION_PHRASE:
            phrase = TP.OPERATION_PHRASE[operation]
            out = step.get("output_component_id")
            zh.append(f"{phrase['zh']}是 component {out}。")
            en.append(f"The {phrase['en']} is component {out}.")

        elif operation == "filter_relation":
            relation = step["relation"]
            reference = step["reference_component_id"]
            candidates = step.get("candidate_component_ids", [])
            if len(candidates) == 1:
                zh.append(
                    f"在 component {reference} {TP.DIRECTION_PHRASE[relation]['zh']}"
                    f"的建筑区域中仅有 component {candidates[0]}，因此它是目标。"
                )
                en.append(
                    f"Component {candidates[0]} is the only building region to the "
                    f"{TP.DIRECTION_PHRASE[relation]['en']} of component {reference}, "
                    f"so it is the target."
                )
            else:
                zh.append(
                    f"位于 component {reference} {TP.DIRECTION_PHRASE[relation]['zh']}的候选为 "
                    f"{TP.join_components_zh(list(candidates))}。"
                )
                en.append(
                    f"The candidates {TP.DIRECTION_RELATIVE_OF[relation]['en']} "
                    f"component {reference} are {TP.join_components_en(list(candidates))}."
                )

            # When a stricter nearest-eligibility subset exists, state it too, so
            # the text accounts for every component the computation actually used.
            nearest_eligible = step.get("nearest_eligible_component_ids")
            if nearest_eligible is not None and sorted(nearest_eligible) != sorted(candidates):
                zh.append(
                    f"其中可用于最近距离比较的为 {TP.join_components_zh(list(nearest_eligible))}。"
                )
                en.append(
                    f"Of those, the ones admissible for the nearest-distance "
                    f"comparison are {TP.join_components_en(list(nearest_eligible))}."
                )

        elif operation == "argmin_boundary_distance":
            anchor = step["reference_component_id"]
            candidates = step.get("candidate_component_ids", [])
            out = step.get("output_component_id")
            previous_was_filter = (
                step_index > 0 and steps[step_index - 1]["operation"] == "filter_relation"
            )
            if len(candidates) == 1:
                if not previous_was_filter:
                    # The filter step already stated the uniqueness; avoid repeating it.
                    zh.append(f"筛选后仅剩 component {out}，因此它是目标。")
                    en.append(f"Only component {out} remains after filtering, so it is the target.")
                # else: the filter step already concluded, nothing to add.
            else:
                zh.append(
                    f"其中 component {out} 与 component {anchor} 的边界距离最近，因此它是目标。"
                )
                en.append(
                    f"Component {out} has the smallest boundary distance to component "
                    f"{anchor}, so it is the target."
                )
        else:
            raise ValueError(f"cannot render reasoning for operation: {operation}")

    return "".join(zh), " ".join(en)


# --------------------------------------------------------------------------
# Sample construction
# --------------------------------------------------------------------------


def build_sample(
    image: G.ImageGeometry,
    image_record: dict,
    level: int,
    query_type: str,
    steps: list[dict],
    target_id: int,
    instruction_zh: str,
    instruction_en: str,
    reference_ids: list[int],
    candidate_ids: list[int],
    *,
    dataset_meta: dict,
    trivial_selection: bool = False,
) -> dict:
    """Assemble one canonical sample record.

    Geometry (polygon, bbox, centroid, area) is deliberately NOT duplicated:
    ``image_metadata_ref`` points at the Task 2 metadata, which remains the
    single geometry source.
    """

    reasoning_zh, reasoning_en = render_reasoning(steps)
    distractor_ids = sorted(
        c.component_id
        for c in image.components
        if c.component_id != target_id and c.component_id not in set(candidate_ids)
    )

    return {
        "sample_id": make_sample_id(image.split, image.image_id, level, query_type, steps, target_id),
        "project": dataset_meta["project"],
        "dataset_name": dataset_meta["name"],
        "dataset_version": dataset_meta["version"],
        "source_dataset": dataset_meta["source_dataset"],
        "source_subset": dataset_meta["source_subset"],
        "image_id": image.image_id,
        "split": image.split,
        "image_path": image_record["image_path"],
        "component_map_path": image_record["component_map"],
        "image_metadata_ref": IMAGE_METADATA_TEMPLATE.format(split=image.split),
        "level": level,
        "query_type": query_type,
        "instruction_zh": instruction_zh,
        "instruction_en": instruction_en,
        "reference_component_ids": sorted(reference_ids),
        "target_component_id": int(target_id),
        "candidate_component_ids": sorted(candidate_ids),
        "distractor_component_ids": distractor_ids,
        "reasoning_steps": steps,
        "reasoning_zh": reasoning_zh,
        "reasoning_en": reasoning_en,
        "target_mask": {
            "representation": "component_map_selector",
            "component_id": int(target_id),
        },
        "trivial_selection": bool(trivial_selection),
        "relation_config_version": RELATION_CONFIG_VERSION,
        "generator_version": GENERATOR_VERSION,
    }


# --------------------------------------------------------------------------
# Query family generators
# --------------------------------------------------------------------------


@dataclass
class CandidateQuery:
    """A generated query before quota selection."""

    level: int
    query_type: str
    steps: list[dict]
    target_id: int
    instruction_zh: str
    instruction_en: str
    reference_ids: list[int]
    candidate_ids: list[int]
    trivial_selection: bool = False

    @property
    def semantic_key(self) -> str:
        return semantic_key("", self.steps, self.target_id)


def generate_level1(
    image: G.ImageGeometry,
    image_relations: R.ImageRelations,
    config: T.RelationConfig,
    enabled_types: Sequence[str],
    counter: DiscardCounter,
) -> list[CandidateQuery]:
    """Direct spatial grounding queries."""

    queries: list[CandidateQuery] = []
    quality = image_relations.quality

    for query_type in enabled_types:
        relation = LEVEL1_GATING_RELATION[query_type]
        result = image_relations.extremes.get(relation) or image_relations.size_rank.get(relation)
        if result is None:
            counter.add("no_reference")
            continue
        if result.ambiguous:
            counter.add("ambiguous")
            continue
        if not result.valid:
            counter.add("border_ineligible" if result.reason == R.REASON_TOO_FEW_CANDIDATES else "no_reference")
            continue

        target_id = result.subject
        operation = LEVEL1_OPERATIONS[query_type]
        steps = [{"step": 1, "operation": operation, "output_component_id": int(target_id)}]

        family = TP.LEVEL1_TEMPLATES[query_type]
        template = TP.select_template(family, semantic_key(image.image_id, steps, target_id))

        queries.append(
            CandidateQuery(
                level=1,
                query_type=query_type,
                steps=steps,
                target_id=target_id,
                instruction_zh=template.zh,
                instruction_en=template.en,
                reference_ids=[],
                candidate_ids=[],
            )
        )
    return queries


def generate_level2(
    image: G.ImageGeometry,
    image_relations: R.ImageRelations,
    config: T.RelationConfig,
    nearest_refs: Sequence[str],
    direction_refs: Sequence[str],
    direction_relations: Sequence[str],
    component_map: np.ndarray,
    counter: DiscardCounter,
) -> list[CandidateQuery]:
    """Reference-based reasoning: Type A (reference -> nearest) and Type B
    (reference -> direction with a unique candidate)."""

    queries: list[CandidateQuery] = []

    # ---- Type A: reference -> nearest -------------------------------------
    for reference_type in nearest_refs:
        ref_result = image_relations.size_rank.get(reference_type)
        if ref_result is None or not ref_result.valid:
            counter.add("ambiguous" if (ref_result is not None and ref_result.ambiguous) else "no_reference")
            continue
        reference_id = ref_result.subject

        # The reference must itself be eligible as a nearest anchor.
        rule = config.eligibility_for("nearest")
        accepted, _ = rule.accepts_as_anchor(image_relations.quality.flags(reference_id).as_dict())
        if not accepted:
            counter.add("border_ineligible")
            continue

        nearest_result = image_relations.nearest.get(reference_id)
        if nearest_result is None or not nearest_result.valid:
            if nearest_result is not None and nearest_result.ambiguous:
                counter.add("nearest_margin_fail")
            else:
                counter.add("no_reference")
            continue
        target_id = nearest_result.object

        ref_operation = LEVEL1_OPERATIONS[reference_type]
        steps = [
            {"step": 1, "operation": ref_operation, "output_component_id": int(reference_id)},
            {
                "step": 2,
                "operation": "argmin_boundary_distance",
                "reference_component_id": int(reference_id),
                "candidate_component_ids": [
                    c.component_id for c in image.components if c.component_id != reference_id
                ],
                "output_component_id": int(target_id),
            },
        ]
        family = TP.LEVEL2_NEAREST_TEMPLATES[reference_type]
        template = TP.select_template(family, semantic_key(image.image_id, steps, target_id))

        queries.append(
            CandidateQuery(
                level=2,
                query_type=f"{reference_type}_to_nearest",
                steps=steps,
                target_id=target_id,
                instruction_zh=template.zh,
                instruction_en=template.en,
                reference_ids=[reference_id],
                candidate_ids=[
                    c.component_id for c in image.components if c.component_id != reference_id
                ],
            )
        )

    # ---- Type B: reference -> direction, ONLY when unique ------------------
    for reference_type in direction_refs:
        ref_result = image_relations.size_rank.get(reference_type)
        if ref_result is None or not ref_result.valid:
            counter.add("ambiguous" if (ref_result is not None and ref_result.ambiguous) else "no_reference")
            continue
        reference_id = ref_result.subject

        for relation in direction_relations:
            # PREDICATE CONVENTION (ADR-006): relation(subject, object) means the
            # SUBJECT satisfies the relation against the OBJECT.
            #
            # The instruction says "the building region to the right of the
            # reference", i.e. the CANDIDATE is the subject:
            #
            #     right_of(candidate, reference)   == candidate is right of reference
            #
            # So the candidate is the subject and the reference is the object.
            # Querying with subject == reference would ask "is the reference to
            # the right of the candidate", which is a different question.
            candidates = sorted(
                r.subject
                for r in image_relations.directional
                if r.valid and r.relation == relation and r.object == reference_id
            )
            if not candidates:
                counter.add("no_direction_candidate")
                continue
            # ADR-004: more than one candidate means the sentence as stated does
            # not determine the answer. Discard; never apply a hidden tie-break.
            if len(candidates) > 1:
                counter.add("multiple_direction_candidates")
                continue

            target_id = candidates[0]
            ref_operation = LEVEL1_OPERATIONS[reference_type]
            # NOTE: the filter step deliberately carries NO output_component_id.
            # Storing it would let `recompute_target_from_steps` short-circuit and
            # "verify" the target by reading it back instead of deriving it. The
            # target is derived from the filter set, whose uniqueness is recorded
            # in candidate_component_ids and re-checked independently.
            steps = [
                {"step": 1, "operation": ref_operation, "output_component_id": int(reference_id)},
                {
                    "step": 2,
                    "operation": "filter_relation",
                    "relation": relation,
                    "reference_component_id": int(reference_id),
                    "input_component_ids": [
                        c.component_id for c in image.components if c.component_id != reference_id
                    ],
                    "candidate_component_ids": [int(target_id)],
                    "unique_candidate": True,
                },
            ]
            family = TP.LEVEL2_DIRECTION_TEMPLATES[relation]
            template = TP.select_template(family, semantic_key(image.image_id, steps, target_id))
            reference_phrase = TP.REFERENCE_PHRASE[reference_type]

            queries.append(
                CandidateQuery(
                    level=2,
                    query_type=f"{reference_type}_to_{relation}",
                    steps=steps,
                    target_id=target_id,
                    instruction_zh=template.zh.format(ref=reference_phrase["zh"]),
                    instruction_en=template.en.format(ref=reference_phrase["en"]),
                    reference_ids=[reference_id],
                    candidate_ids=[int(target_id)],
                )
            )

    return queries


def generate_level3(
    image: G.ImageGeometry,
    image_relations: R.ImageRelations,
    config: T.RelationConfig,
    references: Sequence[str],
    relations_wanted: Sequence[str],
    component_map: np.ndarray,
    counter: DiscardCounter,
    emit_trivial: bool,
) -> list[CandidateQuery]:
    """Multi-hop: reference -> direction filter -> nearest within filtered set."""

    queries: list[CandidateQuery] = []

    for reference_type in references:
        ref_result = image_relations.size_rank.get(reference_type)
        if ref_result is None or not ref_result.valid:
            counter.add("ambiguous" if (ref_result is not None and ref_result.ambiguous) else "no_reference")
            continue
        reference_id = ref_result.subject

        rule = config.eligibility_for("nearest")
        accepted, _ = rule.accepts_as_anchor(image_relations.quality.flags(reference_id).as_dict())
        if not accepted:
            counter.add("border_ineligible")
            continue

        for relation in relations_wanted:
            # The candidate is the SUBJECT (see the convention note in
            # generate_level2): "the regions to the right of the reference".
            filtered = sorted(
                r.subject
                for r in image_relations.directional
                if r.valid and r.relation == relation and r.object == reference_id
            )
            if not filtered:
                counter.add("no_direction_candidate")
                continue

            within = R.nearest_within(
                image, reference_id, filtered, config, image_relations.quality, component_map
            )
            if not within.valid:
                if within.ambiguous:
                    counter.add("nearest_margin_fail")
                else:
                    counter.add("border_ineligible")
                continue
            if within.trivial_selection and not emit_trivial:
                counter.add("level_disabled")
                continue

            target_id = within.target
            ref_operation = LEVEL1_OPERATIONS[reference_type]
            steps = [
                {"step": 1, "operation": ref_operation, "output_component_id": int(reference_id)},
                {
                    "step": 2,
                    "operation": "filter_relation",
                    "relation": relation,
                    "reference_component_id": int(reference_id),
                    "input_component_ids": [
                        c.component_id for c in image.components if c.component_id != reference_id
                    ],
                    # Everything that satisfies the stated direction relation.
                    "candidate_component_ids": list(filtered),
                    # The subset that is also admissible for nearest. These two
                    # sets differ because `nearest` is stricter (a border
                    # component can satisfy `right_of` yet be ineligible as a
                    # nearest candidate). Both are stored so the reasoning text
                    # and the nearest computation stay auditable.
                    "nearest_eligible_component_ids": list(within.candidate_ids),
                },
                {
                    "step": 3,
                    "operation": "argmin_boundary_distance",
                    "reference_component_id": int(reference_id),
                    "candidate_component_ids": list(within.candidate_ids),
                    "output_component_id": int(target_id),
                },
            ]

            template = TP.select_template(
                TP.LEVEL3_TEMPLATES, semantic_key(image.image_id, steps, target_id)
            )
            reference_phrase = TP.REFERENCE_PHRASE[reference_type]
            direction_phrase = TP.DIRECTION_PREPOSITION[relation]

            queries.append(
                CandidateQuery(
                    level=3,
                    query_type=f"{reference_type}_to_{relation}_to_nearest",
                    steps=steps,
                    target_id=target_id,
                    instruction_zh=template.zh.format(
                        ref=reference_phrase["zh"], dir=direction_phrase["zh"]
                    ),
                    instruction_en=template.en.format(
                        ref=reference_phrase["en"], dir=direction_phrase["en"]
                    ),
                    reference_ids=[reference_id],
                    candidate_ids=list(within.candidate_ids),
                    trivial_selection=bool(within.trivial_selection),
                )
            )

    return queries


# --------------------------------------------------------------------------
# Deterministic quota selection
# --------------------------------------------------------------------------


def selection_order(seed: int, image_id: str, level: int, query_type: str, semantic_key_value: str) -> str:
    """Stable ordering key for quota selection."""

    return hashlib.sha256(
        f"{seed}|{image_id}|{level}|{query_type}|{semantic_key_value}".encode("utf-8")
    ).hexdigest()


def apply_quotas(
    image_id: str,
    queries: list[CandidateQuery],
    level_limits: dict[int, int],
    per_type_limit: int,
    seed: int,
    counter: DiscardCounter,
) -> list[CandidateQuery]:
    """Deterministically enforce per-image, per-level and per-type quotas.

    Selection is by ascending SHA256 ordering key, so it is reproducible and
    independent of iteration order. Nothing is dropped for a threshold reason.
    """

    kept: list[CandidateQuery] = []
    for level in (1, 2, 3):
        level_queries = [q for q in queries if q.level == level]
        if not level_queries:
            continue

        # Group by query type, ordered deterministically within each group.
        by_type: dict[str, list[CandidateQuery]] = {}
        for query in level_queries:
            by_type.setdefault(query.query_type, []).append(query)

        selected: list[CandidateQuery] = []
        for query_type, group in by_type.items():
            ordered = sorted(
                group,
                key=lambda q: selection_order(seed, image_id, q.level, q.query_type, q.semantic_key),
            )
            if per_type_limit is not None and len(ordered) > per_type_limit:
                counter.add("quota_downsample", len(ordered) - per_type_limit)
                ordered = ordered[:per_type_limit]
            selected.extend(ordered)

        limit = level_limits.get(level)
        if limit is not None and len(selected) > limit:
            selected = sorted(
                selected,
                key=lambda q: selection_order(seed, image_id, q.level, q.query_type, q.semantic_key),
            )
            counter.add("quota_downsample", len(selected) - limit)
            selected = selected[:limit]

        kept.extend(selected)

    return kept


# --------------------------------------------------------------------------
# Per-image generation
# --------------------------------------------------------------------------


def generate_for_image(
    image: G.ImageGeometry,
    image_record: dict,
    config: T.RelationConfig,
    gen_config: dict,
    dataset_meta: dict,
    counter: DiscardCounter,
    seen_keys: set[str],
) -> list[dict]:
    """Generate every canonical sample for one image."""

    if len(image.components) < 2:
        counter.add("single_component_image")
        return []

    quality = classify_image(image, config)
    component_map = image.load_map(_dataset_root())
    image_relations = R.evaluate_image(image, config, component_map=component_map)

    candidates: list[CandidateQuery] = []

    l1 = gen_config.get("level1", {})
    if l1.get("enabled"):
        candidates.extend(
            generate_level1(image, image_relations, config, l1["query_types"], counter)
        )
    else:
        counter.add("level_disabled")

    l2 = gen_config.get("level2", {})
    if l2.get("enabled"):
        candidates.extend(
            generate_level2(
                image,
                image_relations,
                config,
                l2.get("nearest", {}).get("references", []),
                l2.get("direction", {}).get("references", []),
                l2.get("direction", {}).get("relations", []),
                component_map,
                counter,
            )
        )
    else:
        counter.add("level_disabled")

    l3 = gen_config.get("level3", {})
    if l3.get("enabled"):
        candidates.extend(
            generate_level3(
                image,
                image_relations,
                config,
                l3.get("references", []),
                l3.get("relations", []),
                component_map,
                counter,
                l3.get("emit_trivial", True),
            )
        )
    else:
        counter.add("level_disabled")

    quotas = gen_config["quotas"]
    level_limits = {
        1: quotas["max_per_level_per_image"]["level1"],
        2: quotas["max_per_level_per_image"]["level2"],
        3: quotas["max_per_level_per_image"]["level3"],
    }
    selected = apply_quotas(
        image.image_id,
        candidates,
        level_limits,
        quotas.get("max_per_query_type_per_image"),
        quotas.get("downsample_seed", 0),
        counter,
    )

    samples: list[dict] = []
    for query in selected:
        key = semantic_key(image.image_id, query.steps, query.target_id)
        if key in seen_keys:
            # Same image + same program + same target is one canonical question.
            counter.add("duplicate")
            continue
        seen_keys.add(key)

        samples.append(
            build_sample(
                image=image,
                image_record=image_record,
                level=query.level,
                query_type=query.query_type,
                steps=query.steps,
                target_id=query.target_id,
                instruction_zh=query.instruction_zh,
                instruction_en=query.instruction_en,
                reference_ids=query.reference_ids,
                candidate_ids=query.candidate_ids,
                dataset_meta=dataset_meta,
                trivial_selection=query.trivial_selection,
            )
        )
    return samples


# --------------------------------------------------------------------------
# Dataset root resolution
# --------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset_root() -> Path:
    return _REPO_ROOT / "datasets" / "whu"


def dataset_meta_from_config(gen_config: dict) -> dict:
    return dict(gen_config["dataset"])
