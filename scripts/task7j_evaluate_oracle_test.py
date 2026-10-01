"""Task 7J Part 7 — oracle-reference final test evaluation of the six frozen checkpoints.

Every frozen final L3 test record is evaluated with the canonical oracle GT largest reference for each of the
six Task 7I `best.pt` checkpoints: mIoU/Dice/Pr@0.5, per relation, target-area and boundary-distance quartiles,
the complete `I-FinalTestPairsAll` counterfactual set (pass count/rate, own IoU, cross IoU, own-cross margin —
each pair shares one tile/reference), and efficiency (trainable parameters, total inference wall time,
ms/record, peak inference VRAM).

Per model the three seeds are aggregated with mean and sample std (`ddof=1`) plus matched-seed D-B1 − Z-B3
deltas. **No seed is selected** and no performance gate exists.

Writes `evaluation/task7j_oracle_test_results.json`.

    python scripts/task7j_evaluate_oracle_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
)
from scripts.task6n_train import FEATURE_ROOT, MaskStore, canonical_instances  # noqa: E402
from scripts.task7d_evaluate import component_map_for  # noqa: E402
from scripts.task7i_evaluate_oracle_val import (  # noqa: E402
    load_model,
    metric_row,
    predict,
    quartiles,
    summarise,
)
from scripts.task7i_train import SEEDS, sha256_file  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task7i"
TEST_ROWS = REPO_ROOT / "artifacts" / "task7j" / "packs" / "final_l3_test.jsonl"
PAIRS = EVAL / "task7j_test_pair_manifest.json"
POPULATION = EVAL / "task7j_test_population_manifest.json"
CHECKPOINTS = EVAL / "task7j_checkpoint_manifest.json"
OUT = EVAL / "task7j_oracle_test_results.json"
TOLERANCE = 1.0e-6
DIRECTIONS = ("above", "below", "left", "right")
MODELS = ("Z-B3", "D-B1")


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"values": [float(value) for value in values], "mean": float(array.mean()),
            "std_ddof1": float(array.std(ddof=1)) if array.size > 1 else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    population = json.loads(POPULATION.read_text(encoding="utf-8"))
    checkpoints = json.loads(CHECKPOINTS.read_text(encoding="utf-8"))
    if population["verdict"] != "FINAL_TEST_POPULATION_FROZEN":
        write_json(OUT, {"_doc": "Task 7J section 7.", "task": "7J",
                         "verdict": "FINAL_TEST_POPULATION_INVALID"})
        print("[7j.oracle] STOP FINAL_TEST_POPULATION_INVALID", flush=True)
        return 4
    if not checkpoints["all_verified"]:
        write_json(OUT, {"_doc": "Task 7J section 3.", "task": "7J",
                         "verdict": "FORMAL_CHECKPOINT_MISMATCH"})
        print("[7j.oracle] STOP FORMAL_CHECKPOINT_MISMATCH", flush=True)
        return 3

    records = read_rows(TEST_ROWS)
    pair_payload = json.loads(PAIRS.read_text(encoding="utf-8"))
    pairs = pair_payload["pairs_list"]
    by_id = {record["sample_id"]: record for record in records}
    encoder, _report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()

    results = {}
    for model_name in MODELS:
        for seed in SEEDS:
            key = f"{model_name}/{seed}"
            precomputed: dict[str, dict] = {}
            model, payload = load_model(model_name, seed, args.device)
            if args.device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            inference_started = time.time()
            predictions = predict(model, records, store, masks, args.device, precomputed, model_name)
            inference_seconds = time.time() - inference_started
            peak_vram = (round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
                         if args.device != "cpu" else None)
            rows = []
            for record, prediction in zip(records, predictions):
                truth = np.asarray(masks.mask(record["tile_id"],
                                              record["target_source_feature_id"]), dtype=bool)
                instance = next((item for item in canonical_instances(record["tile_id"])
                                 if item.source_feature_id == record["target_source_feature_id"]), None)
                row = metric_row(prediction, truth)
                row.update({"sample_id": record["sample_id"], "program_id": record["program_id"],
                            "direction": record["direction"], "tile_id": record["tile_id"],
                            "target_area_px": float(truth.sum()),
                            "target_touches_border": bool(instance.touches_border) if instance else None,
                            "target_tiny": bool(instance.tiny) if instance else None})
                component_map = component_map_for(record)
                distance = None
                if component_map is not None:
                    try:
                        import geometry as G

                        distance = float(G.component_box_distance(component_map,
                                                                  record["reference_instance_id"],
                                                                  record["target_instance_id"]))
                    except Exception:
                        distance = None
                row["boundary_distance_px"] = distance
                rows.append(row)

            pair_results = {"passed": 0, "pairs": len(pairs)}
            own_values, cross_values = [], []
            for pair in pairs:
                members = [by_id[member["sample_id"]] for member in pair["members"]]
                pair_predictions = predict(model, members, store, masks, args.device, precomputed,
                                           model_name, batch_size=2)
                own, cross = [], []
                for position, member in enumerate(members):
                    own_mask = np.asarray(masks.mask(member["tile_id"],
                                                     member["target_source_feature_id"]), dtype=bool)
                    other = members[1 - position]
                    other_mask = np.asarray(masks.mask(other["tile_id"],
                                                       other["target_source_feature_id"]), dtype=bool)
                    own.append(float((pair_predictions[position] & own_mask).sum())
                               / (float((pair_predictions[position] | own_mask).sum()) + TOLERANCE))
                    cross.append(float((pair_predictions[position] & other_mask).sum())
                                 / (float((pair_predictions[position] | other_mask).sum()) + TOLERANCE))
                if own[0] > cross[0] and own[1] > cross[1]:
                    pair_results["passed"] += 1
                own_values.extend(own)
                cross_values.extend(cross)
            pair_results.update({
                "pass_rate": pair_results["passed"] / max(1, len(pairs)),
                "mean_own_iou": float(np.mean(own_values)),
                "mean_cross_iou": float(np.mean(cross_values)),
                "own_cross_margin": float(np.mean(own_values) - np.mean(cross_values))})

            results[key] = {
                "model": model_name, "seed": seed, "reference_source": "oracle_native_gt",
                "selected_epoch": int(payload["epoch"]),
                "checkpoint_sha256": sha256_file(CHECKPOINT_ROOT / model_name / str(seed) / "best.pt"),
                "overall": summarise(rows),
                "per_direction": {direction: summarise([row for row in rows
                                                        if row["direction"] == direction])
                                  for direction in DIRECTIONS},
                "target_area_quartiles": quartiles(rows, "target_area_px"),
                "boundary_distance_quartiles": quartiles(rows, "boundary_distance_px"),
                "pairs": pair_results,
                "efficiency": {"trainable_parameters": checkpoints["checkpoints"][key]
                               ["trainable_parameters"],
                               "inference_seconds": round(inference_seconds, 2),
                               "ms_per_record": round(1000.0 * inference_seconds / max(1, len(records)), 3),
                               "peak_inference_vram_gb": peak_vram},
            }
            print(f"[7j.oracle] {key}: mIoU {results[key]['overall']['miou']:.4f} Dice "
                  f"{results[key]['overall']['dice']:.4f} pairs {pair_results['passed']}/"
                  f"{pair_results['pairs']} margin {pair_results['own_cross_margin']:+.4f} "
                  f"({results[key]['efficiency']['ms_per_record']:.2f} ms/record)", flush=True)
            del model
            if args.device != "cpu":
                torch.cuda.empty_cache()

    aggregates = {}
    for model_name in MODELS:
        entries = [results[f"{model_name}/{seed}"] for seed in SEEDS]
        aggregates[model_name] = {
            "miou": aggregate([entry["overall"]["miou"] for entry in entries]),
            "dice": aggregate([entry["overall"]["dice"] for entry in entries]),
            "precision_at_0_5": aggregate([entry["overall"]["precision_at_0_5"] for entry in entries]),
            "pair_pass_rate": aggregate([entry["pairs"]["pass_rate"] for entry in entries]),
            "pair_margin": aggregate([entry["pairs"]["own_cross_margin"] for entry in entries]),
            "parameters": entries[0]["efficiency"]["trainable_parameters"],
        }
    matched = [{"seed": seed,
                "z_b3_miou": results[f"Z-B3/{seed}"]["overall"]["miou"],
                "d_b1_miou": results[f"D-B1/{seed}"]["overall"]["miou"],
                "delta": (results[f"D-B1/{seed}"]["overall"]["miou"]
                          - results[f"Z-B3/{seed}"]["overall"]["miou"])} for seed in SEEDS]
    write_json(OUT, {
        "_doc": ("Task 7J section 7. Oracle-reference final test evaluation of the six frozen Task 7I "
                 "checkpoints on the frozen final L3 test population. No seed selection, no performance gate; "
                 "all three seeds are reported for both models."),
        "task": "7J", "stage": "7-oracle-final-test",
        "population": {"records": population["records"], "per_program": population["per_program"],
                       "sample_id_sha256": population["sample_id_sha256"],
                       "pairs": len(pairs), "pair_id_sha256": pair_payload["pair_id_sha256"]},
        "results": results, "aggregates": aggregates, "matched_seed": matched,
        "matched_seed_wins": sum(1 for entry in matched if entry["delta"] > 0),
        "seeds": list(SEEDS), "models": list(MODELS),
        "seed_selected": False, "training_performed": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7j.oracle] Z-B3 {aggregates['Z-B3']['miou']['mean']:.4f} ± "
          f"{aggregates['Z-B3']['miou']['std_ddof1']:.4f} | D-B1 "
          f"{aggregates['D-B1']['miou']['mean']:.4f} ± {aggregates['D-B1']['miou']['std_ddof1']:.4f} | "
          f"wins {sum(1 for entry in matched if entry['delta'] > 0)}/3", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
