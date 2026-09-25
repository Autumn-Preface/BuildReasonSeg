"""Independent validator for the BuildSpatialReason dataset.

This module audits a generated dataset **from primary sources**, not from the
generator's own bookkeeping:

* source component metadata (Task 2) for geometry;
* the frozen relation engine and threshold config for relation semantics;
* the raw JSONL records (no reliance on ``statistics.json`` / ``manifest.json``).

Design principle: the validator must be able to disagree with the generator.
It never feeds a stored ``target_component_id`` into a recomputation, and it
never trusts a stored ``candidate_component_ids`` list without recomputing it.

Key audits
----------
1. Independent counts from JSONL.
2. Full target recomputation from the intended structured program.
3. **Hidden-eligibility semantic audit.** Natural language such as "the largest
   building" is normally read over *visible* components, but the engine answers
   over an *eligibility-filtered* subset. Where those disagree, the instruction
   can be false for a reader who cannot see the hidden filter. Those cases are
   flagged; they are the primary semantic risk of this dataset.
4. Component-ID leakage into natural language.
5. Structured-program, candidate/distractor, mask-selector and reference integrity.
6. Template reconstruction and language parity.
7. Exact cross-split image duplicate detection.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

import geometry as G
import relations as R
import templates as TP
import thresholds as T
from component_quality import classify_image

# --------------------------------------------------------------------------
# Issue model
# --------------------------------------------------------------------------

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

#: Codes that make the natural-language instruction false or misleading.
SEMANTIC_CODES = (
    "hidden_eligibility_largest",
    "hidden_eligibility_smallest",
    "hidden_eligibility_nearest",
    "hidden_eligibility_level3_nearest",
)

#: Codes that make the dataset unusable as supervision.
BLOCKING_CODES = (
    "target_recompute_mismatch",
    "target_mask_mismatch",
    "duplicate_sample_id",
    "duplicate_semantic_key",
    *SEMANTIC_CODES,
)


@dataclass
class Issue:
    code: str
    severity: str
    sample_id: str | None = None
    image_id: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "sample_id": self.sample_id,
            "image_id": self.image_id,
            "detail": self.detail,
        }


class IssueCollector:
    """Collects issues with per-code caps so a systemic bug stays reportable."""

    def __init__(self, per_code_limit: int = 40) -> None:
        self.counts: Counter = Counter()
        self.severity: dict[str, str] = {}
        self.examples: dict[str, list[dict]] = defaultdict(list)
        self.per_code_limit = per_code_limit

    def add(self, code: str, severity: str, sample_id=None, image_id=None, **detail) -> None:
        self.counts[code] += 1
        self.severity[code] = severity
        if len(self.examples[code]) < self.per_code_limit:
            self.examples[code].append(
                {"sample_id": sample_id, "image_id": image_id, "detail": detail}
            )

    def has(self, code: str) -> bool:
        return self.counts.get(code, 0) > 0

    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict:
        return {
            "counts": {k: v for k, v in sorted(self.counts.items())},
            "severity": {k: self.severity[k] for k in sorted(self.severity)},
            "examples": {k: v for k, v in sorted(self.examples.items())},
        }


# --------------------------------------------------------------------------
# Per-image cached context
# --------------------------------------------------------------------------


class ImageContext:
    """Lazily-computed, cached relation structures for one image.

    Relation evaluation is per-image, so recomputing the same structures once per
    sample would be quadratically wasteful. This makes the full-dataset
    recomputation tractable.
    """

    def __init__(self, image: G.ImageGeometry, config: T.RelationConfig,
                 component_map: np.ndarray | None = None) -> None:
        self.image = image
        self.config = config
        self.quality = classify_image(image, config)
        self._component_map = component_map
        self._relations: R.ImageRelations | None = None
        self._nearest_all: dict[int, R.RelationResult] | None = None
        self._boundary_cache: dict[tuple[int, int], float] = {}

    @property
    def component_map(self) -> np.ndarray:
        if self._component_map is None:
            self._component_map = self.image.load_map(_dataset_root())
        return self._component_map

    @property
    def relations(self) -> R.ImageRelations:
        if self._relations is None:
            self._relations = R.evaluate_image(
                self.image, self.config, component_map=self.component_map
            )
        return self._relations

    def component(self, component_id: int):
        return next((c for c in self.image.components if c.component_id == component_id), None)

    def gap(self, a_id: int, b_id: int) -> float:
        key = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
        if key not in self._boundary_cache:
            self._boundary_cache[key] = G.component_box_distance(self.component_map, key[0], key[1])
        return self._boundary_cache[key]

    # ---- operation-level semantics, mirroring the frozen engine ----

    def eligible_ids(self, relation: str) -> list[int]:
        return self.quality.eligible_ids(relation, self.config)

    def extreme_target(self, operation: str) -> int | None:
        """Target of a global extreme operation, or None when not unique."""

        relation = {
            "argmin_centroid_x": "leftmost",
            "argmax_centroid_x": "rightmost",
            "argmin_centroid_y": "topmost",
            "argmax_centroid_y": "bottommost",
        }.get(operation)
        if relation is None:
            return None
        result = self.relations.extremes.get(relation)
        if result is None or not result.valid:
            return None
        return result.subject

    def size_target(self, operation: str) -> int | None:
        relation = "largest" if operation == "argmax_area" else "smallest"
        result = self.relations.size_rank.get(relation)
        if result is None or not result.valid:
            return None
        return result.subject

    def direction_candidates(self, reference_id: int, relation: str) -> list[int]:
        """Candidates satisfying ``relation(candidate, reference)``.

        Uses the frozen predicate convention: the candidate is the SUBJECT and
        the reference is the OBJECT.
        """

        return sorted(
            r.subject
            for r in self.relations.directional
            if r.valid and r.relation == relation and r.object == reference_id
        )

    def nearest_eligible(self, anchor_id: int, candidates: Sequence[int]) -> list[int]:
        """Subset of ``candidates`` admissible as a nearest target."""

        eligible = set(self.eligible_ids("nearest"))
        return sorted(int(c) for c in set(candidates) if int(c) in eligible and int(c) != anchor_id)

    def nearest_gap(self, anchor_id: int, candidates: Sequence[int]) -> list[tuple[float, int]]:
        """(gap, id) for every candidate, ascending. No eligibility filtering."""

        anchor = self.component(anchor_id)
        if anchor is None:
            return []
        scored = []
        for candidate_id in sorted(set(int(c) for c in candidates)):
            if candidate_id == anchor_id:
                continue
            if self.component(candidate_id) is None:
                continue
            scored.append((self.gap(anchor_id, candidate_id), candidate_id))
        scored.sort(key=lambda t: (t[0], t[1]))
        return scored


# --------------------------------------------------------------------------
# Reusable helpers shared with the generator's semantics
# --------------------------------------------------------------------------

LEVEL1_OPERATIONS = {
    "leftmost": "argmin_centroid_x",
    "rightmost": "argmax_centroid_x",
    "topmost": "argmin_centroid_y",
    "bottommost": "argmax_centroid_y",
    "largest": "argmax_area",
    "smallest": "argmin_area",
}

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _dataset_root() -> Path:
    return _REPO_ROOT / "datasets" / "whu"


def iteration_metadata(dataset_root: Path, split: str) -> Iterator[dict]:
    path = dataset_root / "metadata" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_dataset_records(dataset_root: Path, split: str) -> list[dict]:
    path = dataset_root / "build_spatial_reason" / "v0.1" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------
# Independent target recomputation
# --------------------------------------------------------------------------


def recompute_target_independent(
    ctx: ImageContext, level: int, query_type: str, steps: Sequence[dict]
) -> int | None:
    """Recompute the target from the intended program, using the query_type too.

    Deliberately independent of the stored ``target_component_id``. For the
    Level-2 Type-B case the answer is the unique direction candidate, derived
    from the frozen direction predicate rather than read from the record.
    """

    if level == 1:
        if len(steps) != 1:
            return None
        return ctx.extreme_target(steps[0]["operation"]) if "area" not in steps[0]["operation"] \
            else ctx.size_target(steps[0]["operation"])

    if level == 2:
        reference_type, _, tail = query_type.partition("_to_")
        reference_operation = LEVEL1_OPERATIONS.get(reference_type)
        if reference_operation is None:
            return None
        reference_id = ctx.extreme_target(reference_operation) if "area" not in reference_operation \
            else ctx.size_target(reference_operation)
        if reference_id is None:
            return None

        if tail == "nearest":
            # Nearest over the nearest-eligible set, using boundary distance.
            all_others = [c.component_id for c in ctx.image.components if c.component_id != reference_id]
            eligible = ctx.nearest_eligible(reference_id, all_others)
            scored = ctx.nearest_gap(reference_id, eligible)
            if not scored:
                return None
            if len(scored) >= 2:
                d1, id1 = scored[0]
                d2, _ = scored[1]
                margin = d2 - d1
                if (
                    margin / ctx.image.diagonal < ctx.config.nearest.margin_diag_fraction
                    or margin < ctx.config.nearest.margin_px_floor
                ):
                    return None
            return scored[0][1]

        # Type B: unique direction candidate.
        candidates = ctx.direction_candidates(reference_id, tail)
        if len(candidates) != 1:
            return None
        return candidates[0]

    if level == 3:
        reference_type = query_type.split("_to_", 1)[0]
        reference_operation = LEVEL1_OPERATIONS.get(reference_type)
        if reference_operation is None:
            return None
        reference_id = ctx.extreme_target(reference_operation) if "area" not in reference_operation \
            else ctx.size_target(reference_operation)
        if reference_id is None:
            return None

        relation = None
        for candidate_relation in ("left_of", "right_of", "above", "below"):
            if f"_to_{candidate_relation}_to_nearest" == query_type[len(reference_type):]:
                relation = candidate_relation
                break
        if relation is None:
            return None

        filtered = ctx.direction_candidates(reference_id, relation)
        eligible = ctx.nearest_eligible(reference_id, filtered)
        scored = ctx.nearest_gap(reference_id, eligible)
        if not scored:
            return None
        if len(scored) >= 2:
            d1, _ = scored[0]
            d2, _ = scored[1]
            margin = d2 - d1
            if (
                margin / ctx.image.diagonal < ctx.config.nearest.margin_diag_fraction
                or margin < ctx.config.nearest.margin_px_floor
            ):
                return None
        return scored[0][1]

    return None


# --------------------------------------------------------------------------
# Hidden-eligibility semantic audit
# --------------------------------------------------------------------------


def audit_hidden_largest(ctx: ImageContext, reference_id: int | None) -> dict | None:
    """True global argmax over ALL visible components vs the eligible argmax."""

    if reference_id is None:
        return None
    components = ctx.image.components
    if not components:
        return None
    global_largest = G.rank_by(components, lambda c: c.area_px, reverse=True)[0].component_id
    if global_largest == reference_id:
        return None
    flags = ctx.quality.flags(global_largest).as_dict()
    reasons = [k for k, v in flags.items() if v]
    return {
        "global_largest_id": global_largest,
        "engine_reference_id": reference_id,
        "global_largest_flags": reasons,
    }


def audit_hidden_smallest(ctx: ImageContext, reference_id: int | None) -> dict | None:
    if reference_id is None:
        return None
    components = ctx.image.components
    if not components:
        return None
    global_smallest = G.rank_by(components, lambda c: c.area_px)[0].component_id
    if global_smallest == reference_id:
        return None
    flags = ctx.quality.flags(global_smallest).as_dict()
    reasons = [k for k, v in flags.items() if v]
    return {
        "global_smallest_id": global_smallest,
        "engine_reference_id": reference_id,
        "global_smallest_flags": reasons,
    }


def audit_hidden_nearest(ctx: ImageContext, anchor_id: int, target_id: int) -> dict | None:
    """Semantic nearest over ALL visible components vs the eligible target."""

    all_others = [c.component_id for c in ctx.image.components if c.component_id != anchor_id]
    semantic = ctx.nearest_gap(anchor_id, all_others)
    if not semantic:
        return None
    semantic_target = semantic[0][1]
    if semantic_target == target_id:
        return None
    flags = ctx.quality.flags(semantic_target).as_dict()
    reasons = [k for k, v in flags.items() if v]
    return {
        "semantic_nearest_id": semantic_target,
        "semantic_nearest_gap": semantic[0][0],
        "engine_target_id": target_id,
        "semantic_nearest_flags": reasons,
        "all_semantic_gaps": [
            {"id": cid, "gap": gap, "eligible": cid in set(ctx.eligible_ids("nearest"))}
            for gap, cid in semantic[:6]
        ],
    }


def audit_hidden_level3_nearest(
    ctx: ImageContext, reference_id: int, relation: str, target_id: int
) -> dict | None:
    """Semantic nearest within the direction set, BEFORE nearest eligibility."""

    filtered = ctx.direction_candidates(reference_id, relation)
    if not filtered:
        return None
    semantic = ctx.nearest_gap(reference_id, filtered)
    if not semantic:
        return None
    semantic_target = semantic[0][1]
    if semantic_target == target_id:
        return None
    flags = ctx.quality.flags(semantic_target).as_dict()
    return {
        "relation": relation,
        "direction_candidates": filtered,
        "semantic_nearest_id": semantic_target,
        "semantic_nearest_gap": semantic[0][0],
        "engine_target_id": target_id,
        "semantic_nearest_flags": [k for k, v in flags.items() if v],
        "all_semantic_gaps": [
            {"id": cid, "gap": gap, "eligible": cid in set(ctx.eligible_ids("nearest"))}
            for gap, cid in semantic[:6]
        ],
    }


# --------------------------------------------------------------------------
# Component-ID leakage
# --------------------------------------------------------------------------

import re

#: Matches "component 12" style internal annotation ids.
COMPONENT_ID_PATTERN_EN = re.compile(r"\bcomponents?\s+\d+", re.IGNORECASE)
#: Matches Chinese equivalents such as 组件3 / 构件3.
COMPONENT_ID_PATTERN_ZH = re.compile(r"(组件|构件)\s*\d+")


def find_component_id_leaks(text: str) -> list[str]:
    """Return internal component-id mentions in a natural-language string."""

    if not text:
        return []
    return COMPONENT_ID_PATTERN_EN.findall(text) + COMPONENT_ID_PATTERN_ZH.findall(text)


# --------------------------------------------------------------------------
# Template reconstruction
# --------------------------------------------------------------------------


def reconstruct_template_id(query_type: str, instruction_zh: str, instruction_en: str) -> str | None:
    """Identify which template pair produced this instruction.

    Both languages must resolve to the SAME template id, which is what
    ``language_parity`` means operationally: a zh/en pair that matches templates
    with different ids is a parity failure.

    Query type shapes:
      Level 1  `leftmost`
      Level 2A `largest_to_nearest`
      Level 2B `largest_to_right_of`
      Level 3  `largest_to_right_of_to_nearest`
    """

    to_count = query_type.count("_to_")

    if to_count == 2:
        # Level 3: the renderer uses the shared LEVEL3_TEMPLATES family.
        families = [TP.LEVEL3_TEMPLATES]
    elif to_count == 1:
        if query_type.endswith("_to_nearest"):
            reference = query_type[: -len("_to_nearest")]
            families = [TP.LEVEL2_NEAREST_TEMPLATES.get(reference, ())]
        else:
            relation = query_type.rsplit("_to_", 1)[1]
            families = [TP.LEVEL2_DIRECTION_TEMPLATES.get(relation, ())]
    else:
        families = [TP.LEVEL1_TEMPLATES.get(query_type, ())]

    for family in families:
        for pair in family:
            if _render_matches(pair.zh, instruction_zh) and _render_matches(pair.en, instruction_en):
                return pair.template_id
    return None


def _render_matches(template: str, text: str) -> bool:
    """True when ``text`` could be this template with some placeholder fill."""

    if "{" not in template:
        return template.strip() == text.strip()
    pattern = re.escape(template)
    # Replace escaped placeholders with a permissive pattern.
    pattern = re.sub(r"\\\{[a-z_]+\\\}", ".*", pattern)
    return re.fullmatch(pattern, text.strip()) is not None


# --------------------------------------------------------------------------
# Image hashing for cross-split duplicate detection
# --------------------------------------------------------------------------


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_cross_split_image_duplicates(
    dataset_root: Path, splits: Sequence[str], record_by_split: dict[str, list[dict]]
) -> dict:
    """Hash the underlying source images and look for cross-split duplicates.

    Only images actually referenced by the dataset are hashed (a subset is
    enough to detect leakage and keeps I/O bounded).
    """

    hashes: dict[str, dict[str, set[str]]] = {}
    for split in splits:
        seen_paths: dict[str, str] = {}
        for record in record_by_split[split]:
            path = record["image_path"]
            if path in seen_paths:
                continue
            resolved = (_REPO_ROOT / path).resolve()
            if not resolved.is_file():
                continue
            seen_paths[path] = hash_file(resolved)
        by_hash: dict[str, set[str]] = defaultdict(set)
        for path, digest in seen_paths.items():
            by_hash[digest].add(path)
        hashes[split] = by_hash

    duplicates: list[dict] = []
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = set(hashes[a]) & set(hashes[b])
            for digest in shared:
                duplicates.append(
                    {
                        "split_a": a,
                        "split_b": b,
                        "sha256": digest,
                        "paths_a": sorted(hashes[a][digest]),
                        "paths_b": sorted(hashes[b][digest]),
                    }
                )

    return {
        "images_hashed_by_split": {split: len(hashes[split]) for split in splits},
        "exact_image_duplicate_cross_split": len(duplicates),
        "duplicates": duplicates[:20],
    }


# --------------------------------------------------------------------------
# Distribution helpers
# --------------------------------------------------------------------------


def quantiles(values: Sequence[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def proportions(counter: Counter) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: round(v / total, 6) for k, v in sorted(counter.items())}


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def decide_verdict(issues: "IssueCollector") -> str:
    """Exactly one of PASS / PASS_WITH_WARNINGS / FAIL_REQUIRES_REVISION.

    ``FAIL_REQUIRES_REVISION`` is returned when any blocking code fired:
    target recomputation failures, instruction semantic mismatches (hidden
    eligibility), or key-integrity failures such as duplicate sample ids.
    """

    for code in BLOCKING_CODES:
        if issues.has(code):
            return "FAIL_REQUIRES_REVISION"
    if issues.total() > 0:
        return "PASS_WITH_WARNINGS"
    return "PASS"
