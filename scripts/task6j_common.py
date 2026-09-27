"""Task 6J shared script helpers: fixed subsets, geometry, gates, JSON output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "spatial_reasoning") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "spatial_reasoning"))

import geometry as G  # noqa: E402
from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from dataset_access import DEFAULT_DATASET_ROOT, read_component_map  # noqa: E402
from thresholds import load_config as load_relation_config  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
SPEC_OUT = EVAL / "task6j_program_spec.json"

#: Fixed validation material, identical to Tasks 6B-6I (no new subsets, no test split).
def fixed_validation_material() -> tuple[list, list[dict]]:
    from task6c_train import validation_material
    from task6h1_common import validation_pair_dicts

    val_samples, raw_pairs, _lookup, _audit = validation_material()
    pairs = validation_pair_dicts(raw_pairs)
    return val_samples, pairs


_geometry_cache: dict[str, dict[str, G.ImageGeometry]] = {}


def geometry_by_image(split: str) -> dict[str, G.ImageGeometry]:
    """The frozen metadata geometry for every image of one split (cached per process)."""

    if split not in _geometry_cache:
        _geometry_cache[split] = {
            geometry.image_id: geometry for geometry in G.iter_image_geometry(DEFAULT_DATASET_ROOT, split)
        }
    return _geometry_cache[split]


def geometry_for(sample) -> G.ImageGeometry:
    return geometry_by_image(sample.split)[str(sample.image_id)]


def component_map_for(sample) -> np.ndarray:
    geometry = geometry_for(sample)
    return geometry.load_map(DEFAULT_DATASET_ROOT)


def oracle_candidate_set(sample):
    """J0 oracle candidates: every visible component of the frozen component map."""

    from buildreasonseg_mvp.structured_grounding import CandidateSet

    geometry = geometry_for(sample)
    component_map = geometry.load_map(DEFAULT_DATASET_ROOT)
    return CandidateSet.from_component_map(component_map, geometry)


def paired_sample_lists(pairs: list[dict]) -> tuple[list, list]:
    """The A-side and B-side samples of the fixed paired validation images."""

    a_samples = [data_mod.to_sample(pair["record_a"]) for pair in pairs]
    b_samples = [data_mod.to_sample(pair["record_b"]) for pair in pairs]
    return a_samples, b_samples


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def breakdown(rows: list[dict], key: str) -> dict:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row[key]), []).append(row)
    return {
        name: {"count": len(items), "accuracy": _mean([1.0 if item["correct"] else 0.0 for item in items])}
        for name, items in sorted(buckets.items())
    }


def gate_report(checks: dict[str, bool], gate: dict) -> dict:
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "gate": {key: value for key, value in gate.items()},
    }


def record_accuracy_summary(rows: list[dict]) -> dict:
    correct = sum(1 for row in rows if row["correct"])
    return {
        "count": len(rows),
        "exact_correct": correct,
        "exact_accuracy": float(correct / max(len(rows), 1)),
        "abstained": sum(1 for row in rows if row.get("abstained")),
        "by_level": breakdown(rows, "level"),
        "by_query_type": breakdown(rows, "query_type"),
    }


def write_json_artifact(path: Path, payload: dict) -> None:
    write_json(path, payload)
