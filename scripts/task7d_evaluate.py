"""Task 7D Parts L — MiniVal240 and PairedVal20 evaluation for D-B0 ... D-B4.

MiniVal240 (section 26): mIoU, Dice, Pr@0.5, per-direction mIoU, target-area quartiles, canonical target
boundary-distance quartiles, parameter count, best epoch, wall time and peak VRAM for every variant, plus the
competition diagnostics for the four trainable variants (written separately by
`scripts/task7d_competition_diagnostics.py`).

PairedVal20 (section 27): the exact frozen Z-PairedVal20 pairs, reporting pass/20, mean own IoU, mean cross
IoU and the own-cross margin.

Writes `evaluation/task7d_mini_val.json` and `evaluation/task7d_paired_val.json`.

    python scripts/task7d_evaluate.py
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
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
)
from buildreasonseg_mvp.task7d_data import (  # noqa: E402
    CHECKPOINT_ROOT,
    TRAINABLE_VARIANTS,
    build_batch,
    forward_variant,
    load_pack,
)
from buildreasonseg_mvp.task7d_global_competition_decoder import (  # noqa: E402
    ALL_VARIANTS,
    GlobalCompetitionDecoder,
    variant_report,
)
from scripts.task6n_train import FEATURE_ROOT, MaskStore  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
OUT_MINI = EVAL / "task7d_mini_val.json"
OUT_PAIRED = EVAL / "task7d_paired_val.json"
TASK6Z_VAL = EVAL / "task6z_mini_val.json"
TASK6Z_PAIRED = EVAL / "task6z_paired_val.json"
TOLERANCE = 1.0e-6
DIRECTIONS = ("above", "below", "left", "right")


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


def load_variant_model(variant: str, device: str) -> GlobalCompetitionDecoder:
    checkpoint = CHECKPOINT_ROOT / f"{variant.lower().replace('-', '')}_minitrain1200.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = GlobalCompetitionDecoder(variant).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, records: list[dict], store, masks, device, precomputed: dict,
            batch_size: int = 8) -> list[np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        batch = build_batch(chunk, store, masks, device, precomputed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = forward_variant(model, batch)
        upsampled = model.upsampled(logits, 512)
        predictions.extend((upsampled > 0.0).cpu().numpy()[:, 0])
    return predictions


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    return {"records": len(rows),
            "miou": float(np.mean([row["miou"] for row in rows])),
            "dice": float(np.mean([row["dice"] for row in rows])),
            "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows]))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    training = json.loads((EVAL / "task7d_training.json").read_text(encoding="utf-8"))
    encoder, _report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()
    from task6n_train import canonical_instances

    import geometry as G

    records = load_pack("z_mini_val_240")
    distances = {}
    for record in records:
        component_map = component_map_for(record)
        if component_map is None:
            distances[record["sample_id"]] = None
            continue
        try:
            distances[record["sample_id"]] = float(G.component_box_distance(
                component_map, record["reference_instance_id"], record["target_instance_id"]))
        except Exception:
            distances[record["sample_id"]] = None

    precomputed: dict[str, dict] = {}
    results = {variant: {} for variant in ALL_VARIANTS}
    task6z_val = json.loads(TASK6Z_VAL.read_text(encoding="utf-8"))["results"]["Z-B3"]
    results["D-B0"] = {
        "overall": {"records": task6z_val["overall"]["records"],
                    "miou": task6z_val["overall"]["miou"], "dice": task6z_val["overall"]["dice"],
                    "precision_at_0_5": task6z_val["overall"]["precision_at_0_5"]},
        "source": "frozen Task 6Z Z-B3 evaluation artifact",
        "params": json.loads((EVAL / "task6z_training.json").read_text(encoding="utf-8"))
        ["results"]["Z-B3"]["params"],
        "reproduction": json.loads((EVAL / "task7d_baseline_reproduction.json")
                                   .read_text(encoding="utf-8"))["reproduction_passed"],
    }
    for variant in TRAINABLE_VARIANTS:
        if args.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        model = load_variant_model(variant, args.device)
        variant_started = time.time()
        predictions = predict(model, records, store, masks, args.device, precomputed)
        evaluation_seconds = time.time() - variant_started
        rows = []
        for record, prediction in zip(records, predictions):
            truth = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                               dtype=bool)
            instance = next((item for item in canonical_instances(record["tile_id"])
                             if item.source_feature_id == record["target_source_feature_id"]), None)
            intersection = float((prediction & truth).sum())
            rows.append({
                "sample_id": record["sample_id"], "direction": record["direction"],
                "miou": (intersection + TOLERANCE) / (float((prediction | truth).sum()) + TOLERANCE),
                "dice": (2.0 * intersection + TOLERANCE)
                / (float(prediction.sum()) + float(truth.sum()) + TOLERANCE),
                "precision_at_0_5": (intersection + TOLERANCE)
                / (float(prediction.sum()) + TOLERANCE),
                "target_area_px": float(truth.sum()),
                "boundary_distance_px": distances.get(record["sample_id"]),
                "target_touches_border": bool(instance.touches_border) if instance else None,
                "target_tiny": bool(instance.tiny) if instance else None})

        def quartiles(key: str) -> dict:
            values = [row[key] for row in rows if row[key] is not None]
            if not values:
                return {}
            array = np.asarray(values, dtype=np.float64)
            edges = np.percentile(array, [25, 50, 75])
            buckets = {}
            for index, label in enumerate(("q1", "q2", "q3", "q4")):
                lower = -np.inf if index == 0 else edges[index - 1]
                upper = np.inf if index == 3 else edges[index]
                subset = [row for row in rows if row[key] is not None
                          and ((lower <= row[key] <= upper) if index == 3
                               else (lower <= row[key] < upper))]
                if subset:
                    buckets[label] = {"records": len(subset),
                                      "miou": float(np.mean([row["miou"] for row in subset]))}
            return {"edges": [float(value) for value in edges], "buckets": buckets}

        best = training["results"][variant]["best"]
        results[variant] = {
            "overall": summarise(rows),
            "per_direction": {direction: summarise([row for row in rows
                                                    if row["direction"] == direction])
                              for direction in DIRECTIONS},
            "target_area_quartiles": quartiles("target_area_px"),
            "boundary_distance_quartiles": quartiles("boundary_distance_px"),
            "border_target": summarise([row for row in rows if row["target_touches_border"]]),
            "tiny_target": summarise([row for row in rows if row["target_tiny"]]),
            "params": training["results"][variant]["params"],
            "best_epoch": best["epoch"], "best_step": best["step"],
            "variant_wall_seconds": training["results"][variant]["wall_seconds"],
            "evaluation_seconds": round(evaluation_seconds, 1),
            "peak_vram_gb": training["results"][variant]["peak_vram_gb"],
            "checkpoint_sha256": training["results"][variant]["checkpoint"]["sha256"],
        }
        print(f"[7d.eval] {variant}: mIoU {results[variant]['overall']['miou']:.4f} Dice "
              f"{results[variant]['overall']['dice']:.4f} Pr@0.5 "
              f"{results[variant]['overall']['precision_at_0_5']:.4f} params "
              f"{results[variant]['params']} epoch {best['epoch']}", flush=True)

    write_json(OUT_MINI, {
        "_doc": ("Task 7D section 26. MiniVal240 evaluation of D-B0 (frozen Task 6Z Z-B3) and the four "
                 "trainable global-competition variants. The oracle largest reference, the frozen fields and "
                 "the frozen SAM2 features are the only inputs; GT is an evaluation label."),
        "task": "7D", "stage": "L-mini-val-240",
        "pack": {"name": "z_mini_val_240", "records": len(records),
                 "by_direction": dict(Counter(record["direction"] for record in records))},
        "variants": variant_report(), "results": results,
        "best_trainable_variant": max(TRAINABLE_VARIANTS,
                                      key=lambda variant: results[variant]["overall"]["miou"]),
        "training_performed": True, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })

    paired_records = load_pack("z_paired_val20")
    pairs = [{"tile_id": paired_records[index]["tile_id"], "a": paired_records[index],
              "b": paired_records[index + 1]} for index in range(0, len(paired_records), 2)]
    paired_results = {}
    task6z_paired = json.loads(TASK6Z_PAIRED.read_text(encoding="utf-8"))["systems"]["Z-B3"]
    paired_results["D-B0"] = {"passed": task6z_paired["passed"], "pairs": task6z_paired["pairs"],
                              "pass_rate": task6z_paired["pass_rate"],
                              "mean_own_iou": task6z_paired["mean_own_iou"],
                              "mean_cross_iou": task6z_paired["mean_cross_iou"],
                              "own_cross_margin": task6z_paired["own_cross_margin"],
                              "source": "frozen Task 6Z artifact"}
    for variant in TRAINABLE_VARIANTS:
        model = load_variant_model(variant, args.device)
        own_values, cross_values = [], []
        passed = 0
        for entry in pairs:
            members = [entry["a"], entry["b"]]
            pair_predictions = predict(model, members, store, masks, args.device, precomputed,
                                       batch_size=2)
            own, cross = [], []
            for index, member in enumerate(members):
                own_mask = np.asarray(masks.mask(member["tile_id"],
                                                 member["target_source_feature_id"]), dtype=bool)
                other = members[1 - index]
                other_mask = np.asarray(masks.mask(other["tile_id"],
                                                   other["target_source_feature_id"]), dtype=bool)
                own.append(float((pair_predictions[index] & own_mask).sum())
                           / (float((pair_predictions[index] | own_mask).sum()) + TOLERANCE))
                cross.append(float((pair_predictions[index] & other_mask).sum())
                             / (float((pair_predictions[index] | other_mask).sum()) + TOLERANCE))
            if own[0] > cross[0] and own[1] > cross[1]:
                passed += 1
            own_values.extend(own)
            cross_values.extend(cross)
        paired_results[variant] = {
            "passed": passed, "pairs": len(pairs), "pass_rate": passed / max(1, len(pairs)),
            "mean_own_iou": float(np.mean(own_values)), "mean_cross_iou": float(np.mean(cross_values)),
            "own_cross_margin": float(np.mean(own_values) - np.mean(cross_values))}
        print(f"[7d.eval] paired {variant}: {passed}/{len(pairs)} margin "
              f"{paired_results[variant]['own_cross_margin']:+.4f}", flush=True)
    write_json(OUT_PAIRED, {
        "_doc": ("Task 7D section 27. PairedVal20 evaluation on the exact frozen Z-PairedVal20 pack for "
                 "D-B0 (frozen task 6Z Z-B3) and the four trainable variants."),
        "task": "7D", "stage": "L-paired-val20",
        "pack": {"name": "z_paired_val20", "pairs": len(pairs)}, "results": paired_results,
        "training_performed": True, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
