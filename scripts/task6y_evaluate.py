"""Task 6Y Parts J — MiniVal240 and PairedVal evaluation for the four nearest variants.

MiniVal240 (section 23): mIoU, Dice, Pr@0.5, largest/smallest-reference mIoU, border-target mIoU,
tiny-target mIoU when present, target-area quartiles, canonical target boundary-distance quartiles,
parameters, best epoch, wall time and peak VRAM.

PairedVal (section 24): for each pair the largest-reference and smallest-reference queries are run
independently; the pair passes iff both predictions have a higher IoU with their own pair target than with
the other pair target. Reported: pass / N_pair, pass rate, mean own IoU, mean cross IoU, own-cross margin.

Writes `evaluation/task6y_mini_val.json` and `evaluation/task6y_paired_val.json`.

    python scripts/task6y_evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
)
from buildreasonseg_mvp.task6y_nearest_decoder import (  # noqa: E402
    ALL_VARIANTS,
    NearestTargetDecoder,
    variant_report,
)
from scripts.task6n_train import FEATURE_ROOT, MaskStore  # noqa: E402
from scripts.task6y_train import build_batch, forward_variant  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6y" / "packs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task6y"
TRAINING = EVAL / "task6y_training.json"
OUT_MINI = EVAL / "task6y_mini_val.json"
OUT_PAIRED = EVAL / "task6y_paired_val.json"
PROGRAM_TO_FAMILY = {"largest_to_nearest": "largest", "smallest_to_nearest": "smallest"}


def load_pack(name: str) -> list[dict]:
    return json.loads((PACK_ROOT / f"{name}.json").read_text(encoding="utf-8"))["records"]


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


@torch.no_grad()
def predict(model: NearestTargetDecoder, records: list[dict], store, masks, device, precomputed,
            batch_size: int = 8) -> list[np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        batch = build_batch(chunk, store, masks, device, precomputed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = forward_variant(model, batch)
        upsampled = torch.nn.functional.interpolate(
            logits.float(), size=batch.target_size, mode="bilinear", align_corners=False)
        predictions.extend((upsampled > 0.0).cpu().numpy())
    return predictions


def load_variant(variant: str, device: str) -> NearestTargetDecoder:
    checkpoint = CHECKPOINT_ROOT / f"{variant.lower()}_minitrain1000.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = NearestTargetDecoder(variant).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    if not TRAINING.is_file():
        write_json(OUT_MINI, {"_doc": "Task 6Y section 23.", "task": "6Y",
                              "verdict": "INVALID_EXPERIMENT", "reason": "training stage missing"})
        return 2
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    records = load_pack("y_mini_val_240")
    pairs = json.loads((PACK_ROOT / "y_paired_val.json").read_text(encoding="utf-8"))["records"]
    pair_entries = []
    for index in range(0, len(pairs), 2):
        pair_entries.append({"tile_id": pairs[index]["tile_id"], "largest": pairs[index],
                             "smallest": pairs[index + 1]})

    encoder, _report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()
    from task6n_train import canonical_instances

    # canonical boundary distances for the quartile breakdown (evaluation metadata only)
    distances = {}
    for record in records:
        key = record["sample_id"]
        component_map = component_map_for(record)
        if component_map is None:
            distances[key] = None
            continue
        try:
            import geometry as G

            distances[key] = float(G.component_box_distance(
                component_map, record["reference_instance_id"], record["target_instance_id"]))
        except Exception:
            distances[key] = None

    precomputed: dict[str, dict] = {}
    results = {}
    predictions_by_variant = {}
    for variant in ALL_VARIANTS:
        if args.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        model = load_variant(variant, args.device)
        variant_started = time.time()
        predictions = predict(model, records, store, masks, args.device, precomputed)
        predictions_by_variant[variant] = predictions
        evaluation_seconds = time.time() - variant_started
        rows = []
        for record, prediction in zip(records, predictions):
            truth = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                               dtype=bool)
            instances = {instance.source_feature_id: instance
                         for instance in canonical_instances(record["tile_id"])}
            truth_instance = instances.get(record["target_source_feature_id"])
            intersection = float((prediction & truth).sum())
            union = float((prediction | truth).sum())
            rows.append({
                "sample_id": record["sample_id"],
                "reference_family": PROGRAM_TO_FAMILY[record["program_id"]],
                "miou": (intersection + 1e-6) / (union + 1e-6),
                "dice": (2.0 * intersection + 1e-6)
                / (float(prediction.sum()) + float(truth.sum()) + 1e-6),
                "precision_at_0_5": (intersection + 1e-6) / (float(prediction.sum()) + 1e-6),
                "target_area_px": float(truth.sum()),
                "boundary_distance_px": distances.get(record["sample_id"]),
                "target_touches_border": bool(truth_instance.touches_border)
                if truth_instance is not None else None,
                "target_tiny": bool(truth_instance.tiny) if truth_instance is not None else None,
            })

        def summarise(subset: list[dict]) -> dict:
            if not subset:
                return {"records": 0}
            return {
                "records": len(subset),
                "miou": float(np.mean([row["miou"] for row in subset])),
                "dice": float(np.mean([row["dice"] for row in subset])),
                "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in subset])),
            }

        def quartiles(key: str) -> dict:
            values = [row[key] for row in rows if row[key] is not None]
            if not values:
                return {}
            array = np.asarray(values, dtype=np.float64)
            edges = np.percentile(array, [25, 50, 75])
            out = {}
            for index, label in enumerate(("q1", "q2", "q3", "q4")):
                lower = -np.inf if index == 0 else edges[index - 1]
                upper = np.inf if index == 3 else edges[index]
                subset = [row for row in rows if row[key] is not None
                          and lower <= row[key] <= upper] if index == 3 else \
                         [row for row in rows if row[key] is not None
                          and lower <= row[key] < upper]
                if subset:
                    out[label] = {"records": len(subset), "miou": float(np.mean([row["miou"]
                                                                                for row in subset]))}
            return {"edges": [float(value) for value in edges], "buckets": out}

        best = training["results"][variant]["best"]
        results[variant] = {
            "overall": summarise(rows),
            "largest_reference": summarise([row for row in rows
                                            if row["reference_family"] == "largest"]),
            "smallest_reference": summarise([row for row in rows
                                             if row["reference_family"] == "smallest"]),
            "border_target": summarise([row for row in rows if row["target_touches_border"]]),
            "tiny_target": summarise([row for row in rows if row["target_tiny"]]),
            "target_area_quartiles": quartiles("target_area_px"),
            "boundary_distance_quartiles": quartiles("boundary_distance_px"),
            "params": training["results"][variant]["params"],
            "best_epoch": best["epoch"], "best_step": best["step"],
            "best_selection_miou": best["miou"],
            "variant_wall_seconds": training["results"][variant]["wall_seconds"],
            "evaluation_seconds": round(evaluation_seconds, 1),
            "training_peak_vram_gb": training["results"][variant]["peak_vram_gb"],
            "checkpoint_sha256": training["results"][variant]["checkpoint"]["sha256"],
        }
        print(f"[6y.eval] {variant}: mIoU {results[variant]['overall']['miou']:.4f} Dice "
              f"{results[variant]['overall']['dice']:.4f} Pr@0.5 "
              f"{results[variant]['overall']['precision_at_0_5']:.4f} best epoch {best['epoch']}",
              flush=True)

    mini_payload = {
        "_doc": (
            "Task 6Y section 23. MiniVal240 evaluation of the four predeclared nearest variants trained on "
            "Y-MiniTrain1000 with the oracle reference and the frozen NearestBoundaryField v0.1. GT masks "
            "are evaluation labels only. The test split is never read."
        ),
        "task": "6Y", "stage": "J-mini-val-240",
        "pack": {"name": "y_mini_val_240", "records": len(records),
                 "by_program": dict(Counter(record["program_id"] for record in records))},
        "variants": variant_report(),
        "selection_metric": training["selection_metric"],
        "results": results,
        "best_variant": max(ALL_VARIANTS, key=lambda variant: results[variant]["overall"]["miou"]),
        "field": {"sigma_diag": SIGMA_DIAG, "reference": "oracle_native_gt"},
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_MINI, mini_payload)

    pair_rows = []
    for entry in pair_entries:
        members = [entry["largest"], entry["smallest"]]
        row = {"tile_id": entry["tile_id"], "systems": {}}
        for variant in ALL_VARIANTS:
            model = load_variant(variant, args.device)
            chunk_predictions = predict(model, members, store, masks, args.device, precomputed,
                                        batch_size=2)
            own, cross = [], []
            for index, member in enumerate(members):
                own_mask = np.asarray(masks.mask(member["tile_id"],
                                                 member["target_source_feature_id"]), dtype=bool)
                other = members[1 - index]
                other_mask = np.asarray(masks.mask(other["tile_id"],
                                                   other["target_source_feature_id"]), dtype=bool)
                prediction = chunk_predictions[index]
                own.append(float((prediction & own_mask).sum())
                           / float((prediction | own_mask).sum() + 1e-6))
                cross.append(float((prediction & other_mask).sum())
                             / float((prediction | other_mask).sum() + 1e-6))
            row["systems"][variant] = {"own": own, "cross": cross,
                                       "passes": bool(own[0] > cross[0] and own[1] > cross[1])}
        pair_rows.append(row)

    paired_payload = {
        "_doc": (
            "Task 6Y section 24. PairedVal nearest counterfactual: for each pair the largest-reference and "
            "smallest-reference queries are executed independently and the pair passes iff both "
            "predictions match their own target better than the other pair target."
        ),
        "task": "6Y", "stage": "J-paired-val",
        "pack": {"name": "y_paired_val", "pairs": len(pair_entries)},
        "systems": {}, "test_split_used": False,
    }
    for variant in ALL_VARIANTS:
        rows = [row["systems"][variant] for row in pair_rows]
        own = [value for row in rows for value in row["own"]]
        cross = [value for row in rows for value in row["cross"]]
        passed = sum(1 for row in rows if row["passes"])
        paired_payload["systems"][variant] = {
            "passed": passed, "pairs": len(rows),
            "pass_rate": passed / max(1, len(rows)),
            "mean_own_iou": float(np.mean(own)), "mean_cross_iou": float(np.mean(cross)),
            "own_cross_margin": float(np.mean(own) - np.mean(cross)),
        }
        print(f"[6y.eval] paired {variant}: {passed}/{len(rows)} margin "
              f"{paired_payload['systems'][variant]['own_cross_margin']:+.4f}", flush=True)
    write_json(OUT_PAIRED, paired_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
