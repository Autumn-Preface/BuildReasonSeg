"""Task 7I Part I — final oracle-reference full-val evaluation of the six frozen checkpoints.

After all six best checkpoints exist, each is evaluated once on the full 936-record formal val population with
the canonical oracle GT largest reference: mIoU, Dice, Pr@0.5, per direction, target-area quartiles,
boundary-distance quartiles, and the complete `I-FormalValPairsAll` counterfactual set (pass count/rate, own
IoU, cross IoU, own-cross margin; each pair shares one tile/reference, so the oracle reference is used for both
members).

Writes `evaluation/task7i_oracle_val_results.json`.

    python scripts/task7i_evaluate_oracle_val.py
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
from buildreasonseg_mvp.task6z_l3_decoder import L3TargetDecoder  # noqa: E402
from buildreasonseg_mvp.task7d_global_competition_decoder import GlobalCompetitionDecoder  # noqa: E402
from scripts.task6n_train import FEATURE_ROOT, MaskStore  # noqa: E402
from scripts.task7d_evaluate import component_map_for  # noqa: E402
from scripts.task7i_train import (  # noqa: E402
    PACK_ROOT,
    SEEDS,
    make_batch,
    forward_batch,
    read_rows,
)

EVAL = REPO_ROOT / "evaluation"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task7i"
VAL_ROWS = PACK_ROOT / "formal_val_936.jsonl"
PAIRS = EVAL / "task7i_formal_val_pair_manifest.json"
OUT = EVAL / "task7i_oracle_val_results.json"
TOLERANCE = 1.0e-6
DIRECTIONS = ("above", "below", "left", "right")
MODELS = ("Z-B3", "D-B1")


def load_model(variant: str, seed: int, device: str):
    path = CHECKPOINT_ROOT / variant / str(seed) / "best.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = (L3TargetDecoder("Z-B3") if variant == "Z-B3"
             else GlobalCompetitionDecoder("D-B1")).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def predict(model, records: list[dict], store, masks, device, precomputed, variant: str,
            batch_size: int = 8) -> list[np.ndarray]:
    predictions = []
    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        batch = make_batch(chunk, store, masks, device, precomputed, variant)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = forward_batch(model, batch, variant)
        upsampled = torch.nn.functional.interpolate(logits.float(), size=(512, 512), mode="bilinear",
                                                   align_corners=False)
        predictions.extend((upsampled > 0.0).cpu().numpy()[:, 0])
    return predictions


def metric_row(prediction: np.ndarray, truth: np.ndarray) -> dict:
    intersection = float((prediction & truth).sum())
    return {"miou": (intersection + TOLERANCE) / (float((prediction | truth).sum()) + TOLERANCE),
            "dice": (2.0 * intersection + TOLERANCE)
            / (float(prediction.sum()) + float(truth.sum()) + TOLERANCE),
            "precision_at_0_5": (intersection + TOLERANCE)
            / (float(prediction.sum()) + TOLERANCE)}


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    return {"records": len(rows),
            "miou": float(np.mean([row["miou"] for row in rows])),
            "dice": float(np.mean([row["dice"] for row in rows])),
            "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows]))}


def quartiles(rows: list[dict], key: str) -> dict:
    values = [(row[key], row["miou"]) for row in rows if row.get(key) is not None]
    if not values:
        return {}
    array = np.asarray([value for value, _ in values], dtype=np.float64)
    edges = np.percentile(array, [25, 50, 75])
    buckets = {}
    for index, label in enumerate(("q1", "q2", "q3", "q4")):
        lower = -np.inf if index == 0 else edges[index - 1]
        upper = np.inf if index == 3 else edges[index]
        subset = [miou for value, miou in values
                  if ((lower <= value <= upper) if index == 3 else (lower <= value < upper))]
        if subset:
            buckets[label] = {"records": len(subset), "miou": float(np.mean(subset))}
    return {"edges": [float(value) for value in edges], "buckets": buckets}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    records = read_rows(VAL_ROWS)
    pair_payload = json.loads(PAIRS.read_text(encoding="utf-8"))
    pairs = pair_payload["pairs_list"]
    by_id = {record["sample_id"]: record for record in records}
    encoder, _report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()
    precomputed: dict[str, dict] = {}

    missing = [f"{model}/{seed}" for model in MODELS for seed in SEEDS
               if not (CHECKPOINT_ROOT / model / str(seed) / "best.pt").is_file()]
    if missing:
        write_json(OUT, {"_doc": "Task 7I section 19.", "task": "7I",
                         "verdict": "FORMAL_TRAINING_INCOMPLETE", "missing_checkpoints": missing})
        print(f"[7i.oracle] STOP FORMAL_TRAINING_INCOMPLETE ({missing})", flush=True)
        return 4

    results = {}
    for model_name in MODELS:
        for seed in SEEDS:
            model, payload = load_model(model_name, seed, args.device)
            instance_started = time.time()
            predictions = predict(model, records, store, masks, args.device, precomputed,
                                  model_name)
            inference_seconds = time.time() - instance_started
            rows = []
            for record, prediction in zip(records, predictions):
                truth = np.asarray(masks.mask(record["tile_id"],
                                              record["target_source_feature_id"]), dtype=bool)
                instance = next((item for item in
                                 __import__("scripts.task6n_train", fromlist=["canonical_instances"])
                                 .canonical_instances(record["tile_id"])
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

            # counterfactual pairs (oracle reference is shared within a pair)
            pair_results = {"passed": 0, "pairs": len(pairs)}
            own_values, cross_values = [], []
            for pair in pairs:
                members = [by_id[member["sample_id"]] for member in pair["members"]]
                predictions_pair = predict(model, members, store, masks, args.device, precomputed,
                                           model_name, batch_size=2)
                own, cross = [], []
                for position, member in enumerate(members):
                    own_mask = np.asarray(masks.mask(member["tile_id"],
                                                     member["target_source_feature_id"]), dtype=bool)
                    other = members[1 - position]
                    other_mask = np.asarray(masks.mask(other["tile_id"],
                                                       other["target_source_feature_id"]), dtype=bool)
                    own.append(float((predictions_pair[position] & own_mask).sum())
                               / (float((predictions_pair[position] | own_mask).sum()) + TOLERANCE))
                    cross.append(float((predictions_pair[position] & other_mask).sum())
                                 / (float((predictions_pair[position] | other_mask).sum()) + TOLERANCE))
                if own[0] > cross[0] and own[1] > cross[1]:
                    pair_results["passed"] += 1
                own_values.extend(own)
                cross_values.extend(cross)
            pair_results.update({
                "pass_rate": pair_results["passed"] / max(1, len(pairs)),
                "mean_own_iou": float(np.mean(own_values)),
                "mean_cross_iou": float(np.mean(cross_values)),
                "own_cross_margin": float(np.mean(own_values) - np.mean(cross_values))})

            results[f"{model_name}/{seed}"] = {
                "model": model_name, "seed": seed,
                "selected_epoch": int(payload["epoch"]),
                "checkpoint_sha256": __import__("scripts.task7i_train", fromlist=["sha256_file"])
                .sha256_file(CHECKPOINT_ROOT / model_name / str(seed) / "best.pt"),
                "overall": summarise(rows),
                "per_direction": {direction: summarise([row for row in rows
                                                        if row["direction"] == direction])
                                  for direction in DIRECTIONS},
                "target_area_quartiles": quartiles(rows, "target_area_px"),
                "boundary_distance_quartiles": quartiles(rows, "boundary_distance_px"),
                "pairs": pair_results,
                "inference_seconds": round(inference_seconds, 2),
            }
            print(f"[7i.oracle] {model_name}/{seed}: mIoU {results[f'{model_name}/{seed}']['overall']['miou']:.4f} "
                  f"Dice {results[f'{model_name}/{seed}']['overall']['dice']:.4f} pairs "
                  f"{pair_results['passed']}/{pair_results['pairs']} margin "
                  f"{pair_results['own_cross_margin']:+.4f}", flush=True)
            del model
            if args.device != "cpu":
                torch.cuda.empty_cache()

    write_json(OUT, {
        "_doc": ("Task 7I sections 19-20. Final oracle-reference full-val evaluation of the six frozen formal "
                 "checkpoints on all 936 formal val records, including the complete I-FormalValPairsAll "
                 "counterfactual set (326 pairs). Oracle GT largest reference only."),
        "task": "7I", "stage": "I-oracle-val-evaluation",
        "population": {"val_records": len(records), "pairs": len(pairs),
                       "pair_key_hash": pair_payload["pair_key_hash"]},
        "results": results, "models": list(MODELS), "seeds": list(SEEDS),
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7i.oracle] wrote {OUT.name} for {len(results)} checkpoints", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
