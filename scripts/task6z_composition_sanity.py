"""Task 6Z Part D — parameter-free L3 composition sanity on the frozen Z-MiniVal240.

For every record (section 7):

* **7.1 all eligible non-reference candidates** — canonical native instances, not the reference, satisfying
  the frozen nearest target eligibility; score `score_all(C) = max(P_prod_512[p] for p in C)`; report target
  top-1, target top-3, mean target score, mean best distractor score and target-minus-distractor
  (diagnostic only, never stops the task).
* **7.2 exact direction-valid subset** — keep only candidates that are direction-valid under the frozen
  Task 3B predicate (`relations.evaluate_direction` with the record's direction and the oracle reference);
  score with the same `P_prod_512`; report target top-1, target top-3 and the mean Spearman correlation
  between the product-field score and `-canonical boundary_distance` inside that subset.

Sanity gate (section 8, direction-valid subset only): top-1 >= 0.95, top-3 >= 0.99, mean Spearman >= 0.90.
Any failure stops the task with `L3_COMPOSITION_FIELD_SEMANTIC_MISMATCH`; field constants are never tuned.

Writes `evaluation/task6z_composition_sanity.json`.

    python scripts/task6z_composition_sanity.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
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
    spearman_correlation,
)
from buildreasonseg_mvp.task6z_field_composition import (  # noqa: E402
    L3_PROGRAMS,
    PROGRAM_TO_RELATION,
    field_report,
)
from scripts.task6y_field_sanity import _image_geometry, component_map_for, load_engine  # noqa: E402
from scripts.task6z_fields import compute_product_field  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6z_composition_sanity.json"
PACK = REPO_ROOT / "artifacts" / "task6z" / "packs" / "z_mini_val_240.json"
CONFIG = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"
GATE = {"top1_min": 0.95, "top3_min": 0.99, "spearman_min": 0.90}


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    spearmans = [row["spearman"] for row in rows if np.isfinite(row.get("spearman", np.nan))]
    deltas = [row["target_minus_distractor"] for row in rows
              if row.get("target_minus_distractor") is not None]
    return {
        "records": len(rows),
        "top1_rate": float(np.mean([row["is_top1"] for row in rows])),
        "top3_rate": float(np.mean([row["is_top3"] for row in rows])),
        "mean_target_score": float(np.mean([row["target_score"] for row in rows
                                            if row["target_score"] is not None]))
        if any(row["target_score"] is not None for row in rows) else None,
        "mean_best_distractor_score": float(np.mean([row["best_distractor_score"] for row in rows
                                                     if row["best_distractor_score"] is not None]))
        if any(row["best_distractor_score"] is not None for row in rows) else None,
        "mean_target_minus_distractor": float(np.mean(deltas)) if deltas else None,
        "mean_eligible_candidates": float(np.mean([row["candidates"] for row in rows])),
        "mean_spearman": float(np.mean(spearmans)) if spearmans else None,
        "records_with_spearman": len(spearmans),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    if not PACK.is_file():
        write_json(OUT, {"_doc": "Task 6Z section 7.", "task": "6Z",
                         "verdict": "INVALID_EXPERIMENT", "reason": "Z-MiniVal240 pack missing"})
        return 2
    records = json.loads(PACK.read_text(encoding="utf-8"))["records"]
    G, R, classify_image, load_config = load_engine()
    config = load_config(str(CONFIG))
    from task6n_train import canonical_instances

    all_rows, direction_rows, canonical_rows = [], [], []
    agreement = Counter()
    by_direction: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        tile_id = record["tile_id"]
        instances = canonical_instances(tile_id)
        by_feature = {instance.source_feature_id: instance for instance in instances}
        reference = by_feature.get(int(record["reference_source_feature_id"]))
        if reference is None:
            continue
        image = _image_geometry(G, tile_id, instances)
        quality = classify_image(image, config)
        nearest_eligible = set(quality.eligible_ids("nearest", config))
        candidates = [instance for instance in instances
                      if instance.tile_instance_id in nearest_eligible
                      and instance.tile_instance_id != reference.tile_instance_id]
        if not candidates:
            continue
        relation = PROGRAM_TO_RELATION[record["program_id"]]
        reference_component = image.get(reference.tile_instance_id)
        direction_valid = []
        for candidate in candidates:
            subject = image.get(candidate.tile_instance_id)
            result = R.evaluate_direction(relation, image, subject, reference_component, config,
                                          quality)
            if result.valid:
                direction_valid.append(candidate)
        # cross-check against the frozen canonical step-2 candidate list
        canonical_ids = set(record["metadata"].get("candidate_component_ids", []))
        engine_ids = {instance.tile_instance_id for instance in direction_valid}
        agreement[bool(engine_ids == canonical_ids)] += 1
        if engine_ids > canonical_ids:
            agreement["engine_superset"] += 1
        elif engine_ids < canonical_ids:
            agreement["engine_subset"] += 1

        field = compute_product_field(reference.mask, record["program_id"])
        component_map = component_map_for(record)

        def score_set(subset: list, with_distance: bool) -> dict:
            scores, distances, labels = [], [], []
            for candidate in subset:
                scores.append(candidate_field_score(field, candidate.mask))
                if with_distance and component_map is not None:
                    try:
                        distances.append(float(G.component_box_distance(
                            component_map, reference.tile_instance_id, candidate.tile_instance_id)))
                    except Exception:
                        distances.append(np.nan)
                else:
                    distances.append(np.nan)
                labels.append(int(candidate.source_feature_id)
                              == int(record["target_source_feature_id"]))
            order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
            target_index = int(np.argmax(labels)) if any(labels) else None
            target_score = float(scores[target_index]) if target_index is not None else None
            distractors = [score for score, is_target in zip(scores, labels) if not is_target]
            finite = [index for index, value in enumerate(distances) if np.isfinite(value)]
            correlation = float("nan")
            if len(finite) >= 2 and any(labels[index] for index in finite):
                correlation = spearman_correlation(
                    np.asarray([scores[index] for index in finite]),
                    -np.asarray([distances[index] for index in finite]))
            return {
                "candidates": len(subset),
                "is_top1": bool(target_index is not None and int(order[0]) == target_index),
                "is_top3": bool(target_index is not None
                                and target_index in set(int(value) for value in order[:3])),
                "target_score": target_score,
                "best_distractor_score": max(distractors) if distractors else None,
                "target_minus_distractor": (None if target_score is None or not distractors
                                            else target_score - max(distractors)),
                "spearman": correlation,
            }

        all_row = score_set(candidates, with_distance=False)
        all_row.update({"sample_id": record["sample_id"], "direction": record["direction"]})
        all_rows.append(all_row)
        if direction_valid:
            direction_row = score_set(direction_valid, with_distance=True)
            direction_row.update({"sample_id": record["sample_id"], "direction": record["direction"]})
            direction_rows.append(direction_row)
            by_direction[record["direction"]].append(direction_row)
        # diagnostic: the record's own frozen step-2 candidate list (generator-consistent)
        canonical_instances_subset = [instance for instance in instances
                                      if instance.tile_instance_id in canonical_ids
                                      and instance.tile_instance_id != reference.tile_instance_id]
        if canonical_instances_subset:
            canonical_row = score_set(canonical_instances_subset, with_distance=True)
            canonical_row.update({"sample_id": record["sample_id"],
                                  "direction": record["direction"]})
            canonical_rows.append(canonical_row)

    overall_all = summarise(all_rows)
    overall_direction = summarise(direction_rows)
    gate = {
        "top1": {"required": GATE["top1_min"], "measured": overall_direction["top1_rate"],
                 "passed": overall_direction["top1_rate"] >= GATE["top1_min"]},
        "top3": {"required": GATE["top3_min"], "measured": overall_direction["top3_rate"],
                 "passed": overall_direction["top3_rate"] >= GATE["top3_min"]},
        "spearman": {"required": GATE["spearman_min"], "measured": overall_direction["mean_spearman"],
                     "passed": (overall_direction["mean_spearman"] or 0.0) >= GATE["spearman_min"]},
    }
    passed = all(entry["passed"] for entry in gate.values())

    payload = {
        "_doc": (
            "Task 6Z sections 7-8. Parameter-free semantic sanity of the deterministic L3 product field "
            "P_prod = clamp(P_dir * P_near) on the frozen Z-MiniVal240: candidate score is the maximum "
            "product-field value inside the candidate mask. The gate applies to the exact direction-valid "
            "subset produced by the frozen Task 3B predicate; the all-candidate metrics are diagnostic."
        ),
        "task": "6Z", "stage": "D-composition-sanity",
        "pack": {"path": str(PACK), "records": len(records),
                 "by_direction": dict(Counter(record["direction"] for record in records))},
        "fields": field_report(),
        "sigma_diag": SIGMA_DIAG,
        "all_candidates": overall_all,
        "direction_valid_subset": overall_direction,
        "canonical_direction_valid_subset": summarise(canonical_rows),
        "direction_valid_by_direction": {direction: summarise(rows)
                                         for direction, rows in sorted(by_direction.items())},
        "dynamic_relation_engine": {
            "module": "spatial_reasoning/relations.py::evaluate_direction",
            "frozen": True,
            "canonical_step2_agreement_records": agreement.get(True, 0),
            "canonical_step2_disagreement_records": agreement.get(False, 0),
            "engine_superset_records": agreement.get("engine_superset", 0),
            "engine_subset_records": agreement.get("engine_subset", 0),
            "note": ("the gate uses the frozen engine recomputation; the canonical step-2 list of the "
                     "frozen record is reported alongside as a generator-consistent diagnostic"),
        },
        "gate_constants": GATE, "gate": gate, "sanity_passed": passed,
        "verdict": ("L3_COMPOSITION_SANITY_PASS" if passed
                    else "L3_COMPOSITION_FIELD_SEMANTIC_MISMATCH"),
        "constants_tuned": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[6z.sanity] all-candidates top1 {overall_all['top1_rate']:.4f} top3 "
          f"{overall_all['top3_rate']:.4f} | direction-valid top1 "
          f"{overall_direction['top1_rate']:.4f} top3 {overall_direction['top3_rate']:.4f} Spearman "
          f"{overall_direction['mean_spearman']:.4f} (n={overall_direction['records']}) | canonical-list "
          f"top1 {summarise(canonical_rows)['top1_rate']:.4f} Spearman "
          f"{summarise(canonical_rows)['mean_spearman']:.4f} | engine agreement "
          f"{agreement.get(True, 0)}/{agreement.get(True, 0) + agreement.get(False, 0)} -> "
          f"{payload['verdict']}", flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
