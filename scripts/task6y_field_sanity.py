"""Task 6Y Part D — parameter-free nearest-field semantic sanity on the frozen Y-MiniVal240.

For every record (section 7):

1. compute `P_near_512` from the oracle reference mask;
2. consider every canonical native building other than the reference that satisfies the frozen nearest
   target eligibility (`nearest` eligibility rule from `configs/spatial_relations_v1.yaml`);
3. candidate score `field_score(C) = max(P_near_512[p] for p in C)`.

Reported: target top-1 / top-3 rate, mean target score, mean best non-target distractor score, mean
target-minus-distractor score, per reference family, mean eligible candidate count, and the per-record
Spearman correlation between `field_score(C)` and `-canonical boundary_distance(reference, C)` (then mean).

Sanity gate (section 7): top-1 >= 0.85, top-3 >= 0.97, mean Spearman >= 0.90. If any fails the task stops
with `NEAREST_BOUNDARY_FIELD_SEMANTIC_MISMATCH`; sigma is never tuned.

Writes `evaluation/task6y_field_sanity.json`.

    python scripts/task6y_field_sanity.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.nearest_boundary_field import (  # noqa: E402
    SIGMA_DIAG,
    candidate_field_score,
    nearest_boundary_field_512,
    scipy_available,
    spearman_correlation,
)

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6y_field_sanity.json"
PACK = REPO_ROOT / "artifacts" / "task6y" / "packs" / "y_mini_val_240.json"
CONFIG = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"
GATE = {"top1_min": 0.85, "top3_min": 0.97, "spearman_min": 0.90}


def load_engine():
    """The frozen canonical nearest engine (relations/geometry/quality) and its config loader."""

    import geometry as G
    import relations as R
    from component_quality import classify_image
    from thresholds import load_config

    return G, R, classify_image, load_config


def component_map_for(record: dict) -> np.ndarray | None:
    path = record.get("metadata", {}).get("component_map_path")
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / path
    if not resolved.is_file():
        return None
    from PIL import Image

    return np.asarray(Image.open(resolved))


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    if not scipy_available():
        write_json(OUT, {"_doc": "Task 6Y section 5.", "task": "6Y",
                         "verdict": "NEAREST_FIELD_DEPENDENCY_UNAVAILABLE",
                         "reason": "scipy.ndimage.distance_transform_edt is unavailable"})
        print("[6y.sanity] STOP NEAREST_FIELD_DEPENDENCY_UNAVAILABLE", flush=True)
        return 2
    if not PACK.is_file():
        write_json(OUT, {"_doc": "Task 6Y section 7.", "task": "6Y",
                         "verdict": "INVALID_EXPERIMENT", "reason": "Y-MiniVal240 pack missing"})
        return 2

    payload = json.loads(PACK.read_text(encoding="utf-8"))
    records = payload["records"]
    G, R, classify_image, load_config = load_engine()
    config = load_config(str(CONFIG))

    from task6n_train import canonical_instances

    rows = []
    for record in records:
        tile_id = record["tile_id"]
        instances = canonical_instances(tile_id)
        by_feature = {instance.source_feature_id: instance for instance in instances}
        reference = by_feature.get(int(record["reference_source_feature_id"]))
        if reference is None:
            continue
        image = _image_geometry(G, tile_id, instances)
        if image is None:
            continue
        quality = classify_image(image, config)
        eligible_ids = set(quality.eligible_ids("nearest", config))
        candidates = [instance for instance in instances
                      if instance.tile_instance_id in eligible_ids
                      and instance.tile_instance_id != reference.tile_instance_id]
        if not candidates:
            continue
        field = nearest_boundary_field_512(reference.mask)
        component_map = component_map_for(record)
        scores, distances, labels = [], [], []
        for candidate in candidates:
            scores.append(candidate_field_score(field, candidate.mask))
            if component_map is None:
                distances.append(np.nan)
            else:
                try:
                    distances.append(float(G.component_box_distance(
                        component_map, reference.tile_instance_id, candidate.tile_instance_id)))
                except Exception:
                    distances.append(np.nan)
            labels.append(int(candidate.source_feature_id) == int(record["target_source_feature_id"]))
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
        target_index = int(np.argmax(labels)) if any(labels) else None
        target_score = float(scores[target_index]) if target_index is not None else None
        distractor_scores = [score for score, is_target in zip(scores, labels) if not is_target]
        best_distractor = max(distractor_scores) if distractor_scores else None
        finite = [index for index, value in enumerate(distances) if np.isfinite(value)]
        correlation = float("nan")
        if len(finite) >= 2 and any(labels[index] for index in finite):
            correlation = spearman_correlation(
                np.asarray([scores[index] for index in finite]),
                -np.asarray([distances[index] for index in finite]))
        rows.append({
            "sample_id": record["sample_id"], "tile_id": tile_id,
            "reference_family": ("largest" if record["program_id"] == "largest_to_nearest"
                                 else "smallest"),
            "eligible_candidates": len(candidates),
            "target_rank": (int(np.where(order == target_index)[0][0]) + 1)
            if target_index is not None else None,
            "is_top1": bool(target_index is not None and int(order[0]) == target_index),
            "is_top3": bool(target_index is not None
                            and target_index in set(int(value) for value in order[:3])),
            "target_score": target_score,
            "best_distractor_score": best_distractor,
            "target_minus_distractor": (None if target_score is None or best_distractor is None
                                        else target_score - best_distractor),
            "spearman": correlation,
        })

    def summarise(subset: list[dict]) -> dict:
        if not subset:
            return {"records": 0}
        spearmans = [row["spearman"] for row in subset if np.isfinite(row["spearman"])]
        deltas = [row["target_minus_distractor"] for row in subset
                  if row["target_minus_distractor"] is not None]
        return {
            "records": len(subset),
            "top1_rate": float(np.mean([row["is_top1"] for row in subset])),
            "top3_rate": float(np.mean([row["is_top3"] for row in subset])),
            "mean_target_score": float(np.mean([row["target_score"] for row in subset
                                                if row["target_score"] is not None])),
            "mean_best_distractor_score": float(np.mean([row["best_distractor_score"] for row in subset
                                                         if row["best_distractor_score"] is not None])),
            "mean_target_minus_distractor": float(np.mean(deltas)) if deltas else None,
            "mean_eligible_candidates": float(np.mean([row["eligible_candidates"]
                                                       for row in subset])),
            "mean_spearman": float(np.mean(spearmans)) if spearmans else None,
            "records_with_spearman": len(spearmans),
        }

    overall = summarise(rows)
    by_family = {
        family: summarise([row for row in rows if row["reference_family"] == family])
        for family in ("largest", "smallest")
    }
    gate = {
        "top1": {"required": GATE["top1_min"], "measured": overall["top1_rate"],
                 "passed": overall["top1_rate"] >= GATE["top1_min"]},
        "top3": {"required": GATE["top3_min"], "measured": overall["top3_rate"],
                 "passed": overall["top3_rate"] >= GATE["top3_min"]},
        "spearman": {"required": GATE["spearman_min"], "measured": overall["mean_spearman"],
                     "passed": (overall["mean_spearman"] or 0.0) >= GATE["spearman_min"]},
    }
    passed = all(entry["passed"] for entry in gate.values())

    result = {
        "_doc": (
            "Task 6Y section 7. Parameter-free semantic sanity of the oracle-reference "
            "NearestBoundaryField v0.1 on the frozen Y-MiniVal240: candidate score is the maximum field "
            "value inside the candidate mask, compared against the canonical frozen boundary_distance "
            "ordering. sigma_diag is fixed at 0.05 and was not tuned."
        ),
        "task": "6Y", "stage": "D-field-sanity",
        "pack": {"path": str(PACK), "records": len(records)},
        "field": {"sigma_diag": SIGMA_DIAG, "reference": "oracle_native_gt",
                  "candidate_score": "max(P_near_512 inside candidate)",
                  "distance_metric": "boundary_distance",
                  "eligibility": "frozen nearest rule (configs/spatial_relations_v1.yaml)",
                  "component_map_used_for_distance_only": True},
        "overall": overall, "by_family": by_family,
        "gate_constants": GATE, "gate": gate,
        "sanity_passed": passed,
        "verdict": ("NEAREST_BOUNDARY_FIELD_SANITY_PASS" if passed
                    else "NEAREST_BOUNDARY_FIELD_SEMANTIC_MISMATCH"),
        "sigma_tuned": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, result)
    print(f"[6y.sanity] records {overall['records']} top1 {overall['top1_rate']:.4f} "
          f"top3 {overall['top3_rate']:.4f} mean Spearman {overall['mean_spearman']:.4f} "
          f"mean target {overall['mean_target_score']:.4f} best distractor "
          f"{overall['mean_best_distractor_score']:.4f} candidates "
          f"{overall['mean_eligible_candidates']:.2f} -> {result['verdict']}", flush=True)
    return 0 if passed else 3


def _image_geometry(G, tile_id: str, instances: list):
    """Build the frozen `ImageGeometry` for one tile from the canonical native instances."""

    components = []
    for instance in instances:
        x0, y0, x1, y1 = (int(value) for value in instance.bbox_xyxy)
        components.append(G.Component(
            component_id=int(instance.tile_instance_id),
            source_polygon_index=int(instance.tile_instance_id),
            area_px=int(instance.area_px),
            area_ratio=float(instance.area_px) / float(512 * 512),
            centroid_px=(float(instance.centroid[0]), float(instance.centroid[1])),
            bbox_xyxy_px=(x0, y0, x1, y1),
            width_px=max(1, x1 - x0),
            height_px=max(1, y1 - y0),
            touches_image_border=bool(instance.touches_border),
            continuous_polygon_area_px=float(instance.area_px),
        ))
    return G.ImageGeometry(
        image_id=tile_id,
        split="val",
        width=512,
        height=512,
        component_map_path="",
        components=components,
    )


if __name__ == "__main__":
    raise SystemExit(main())
