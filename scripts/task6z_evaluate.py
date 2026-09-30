"""Task 6Z Part J — MiniVal240 and PairedVal evaluation for the six L3 variants.

MiniVal240 (section 26): mIoU, Dice, Pr@0.5, per-direction mIoU, target-area quartiles, canonical target
boundary-distance quartiles, parameter count, best epoch, wall time and peak VRAM.

PairedVal (section 27): the two direction programs of each pair are run independently; the pair passes iff
both predictions prefer their own GT target over the other member's target. Reported per variant: pass /
N_pair, pass rate, mean own IoU, mean cross IoU, own-cross margin.

Writes `evaluation/task6z_mini_val.json` and `evaluation/task6z_paired_val.json`.

    python scripts/task6z_evaluate.py
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
from buildreasonseg_mvp.task6z_l3_decoder import ALL_VARIANTS, L3TargetDecoder, variant_report  # noqa: E402
from scripts.task6n_train import FEATURE_ROOT, MaskStore  # noqa: E402
from scripts.task6z_train import build_batch, forward_variant, image_path_of, load_pack  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
CHECKPOINT_ROOT = REPO_ROOT / "artifacts" / "checkpoints" / "task6z"
TRAINING = EVAL / "task6z_training.json"
OUT_MINI = EVAL / "task6z_mini_val.json"
OUT_PAIRED = EVAL / "task6z_paired_val.json"


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


def load_variant(variant: str, device: str) -> L3TargetDecoder:
    checkpoint = CHECKPOINT_ROOT / f"{variant.lower().replace('-', '')}_minitrain1200.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = L3TargetDecoder(variant).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: L3TargetDecoder, records: list[dict], store, masks, device, precomputed,
            batch_size: int = 8) -> list[np.ndarray]:
    model.eval()
    predictions = []
    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        batch = build_batch(chunk, store, masks, device, precomputed)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = forward_variant(model, batch)
        upsampled = torch.nn.functional.interpolate(logits.float(), size=batch.target_size,
                                                    mode="bilinear", align_corners=False)
        predictions.extend((upsampled > 0.0).cpu().numpy())
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

    if not TRAINING.is_file():
        write_json(OUT_MINI, {"_doc": "Task 6Z section 26.", "task": "6Z",
                              "verdict": "INVALID_EXPERIMENT", "reason": "training stage missing"})
        return 2
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    records = load_pack("z_mini_val_240")
    paired = load_pack("z_paired_val20")
    pairs = [{"tile_id": paired[index]["tile_id"], "a": paired[index], "b": paired[index + 1]}
             for index in range(0, len(paired), 2)]

    encoder, _report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()
    from task6n_train import canonical_instances

    import geometry as G

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
    results = {}
    for variant in ALL_VARIANTS:
        if args.device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        model = load_variant(variant, args.device)
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
            union = float((prediction | truth).sum())
            rows.append({
                "sample_id": record["sample_id"], "direction": record["direction"],
                "miou": (intersection + 1e-6) / (union + 1e-6),
                "dice": (2.0 * intersection + 1e-6)
                / (float(prediction.sum()) + float(truth.sum()) + 1e-6),
                "precision_at_0_5": (intersection + 1e-6) / (float(prediction.sum()) + 1e-6),
                "target_area_px": float(truth.sum()),
                "boundary_distance_px": distances.get(record["sample_id"]),
                "target_touches_border": bool(instance.touches_border) if instance else None,
                "target_tiny": bool(instance.tiny) if instance else None,
            })

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
                              for direction in ("above", "below", "left", "right")},
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
        print(f"[6z.eval] {variant}: mIoU {results[variant]['overall']['miou']:.4f} Dice "
              f"{results[variant]['overall']['dice']:.4f} Pr@0.5 "
              f"{results[variant]['overall']['precision_at_0_5']:.4f} best epoch {best['epoch']}",
              flush=True)

    mini_payload = {
        "_doc": (
            "Task 6Z section 26. MiniVal240 evaluation of the six predeclared L3 variants trained on "
            "Z-MiniTrain1200 with the oracle largest reference, the frozen directional field v0.2, the "
            "frozen nearest field v0.1 and the exact deterministic product field. GT masks are evaluation "
            "labels only; the test split is never read."
        ),
        "task": "6Z", "stage": "J-mini-val-240",
        "pack": {"name": "z_mini_val_240", "records": len(records),
                 "by_direction": dict(Counter(record["direction"] for record in records))},
        "variants": variant_report(), "selection_metric": training["selection_metric"],
        "results": results,
        "best_variant": max(ALL_VARIANTS, key=lambda variant: results[variant]["overall"]["miou"]),
        "test_split_used": False, "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_MINI, mini_payload)

    pair_rows = []
    for entry in pairs:
        members = [entry["a"], entry["b"]]
        row = {"tile_id": entry["tile_id"], "systems": {}}
        for variant in ALL_VARIANTS:
            model = load_variant(variant, args.device)
            predictions = predict(model, members, store, masks, args.device, precomputed, batch_size=2)
            own, cross = [], []
            for index, member in enumerate(members):
                own_mask = np.asarray(masks.mask(member["tile_id"],
                                                 member["target_source_feature_id"]), dtype=bool)
                other = members[1 - index]
                other_mask = np.asarray(masks.mask(other["tile_id"],
                                                   other["target_source_feature_id"]), dtype=bool)
                prediction = predictions[index]
                own.append(float((prediction & own_mask).sum())
                           / float((prediction | own_mask).sum() + 1e-6))
                cross.append(float((prediction & other_mask).sum())
                             / float((prediction | other_mask).sum() + 1e-6))
            row["systems"][variant] = {"own": own, "cross": cross,
                                       "passes": bool(own[0] > cross[0] and own[1] > cross[1])}
        pair_rows.append(row)

    paired_payload = {
        "_doc": (
            "Task 6Z section 27. Same-reference directional PairedVal: for each pair both direction "
            "programs are executed independently and the pair passes iff both predictions prefer their own "
            "GT target over the other member's target."
        ),
        "task": "6Z", "stage": "J-paired-val",
        "pack": {"name": "z_paired_val20", "pairs": len(pair_rows)},
        "systems": {}, "test_split_used": False,
    }
    for variant in ALL_VARIANTS:
        rows = [row["systems"][variant] for row in pair_rows]
        own = [value for row in rows for value in row["own"]]
        cross = [value for row in rows for value in row["cross"]]
        passed = sum(1 for row in rows if row["passes"])
        paired_payload["systems"][variant] = {
            "passed": passed, "pairs": len(rows), "pass_rate": passed / max(1, len(rows)),
            "mean_own_iou": float(np.mean(own)), "mean_cross_iou": float(np.mean(cross)),
            "own_cross_margin": float(np.mean(own) - np.mean(cross)),
        }
        print(f"[6z.eval] paired {variant}: {passed}/{len(rows)} margin "
              f"{paired_payload['systems'][variant]['own_cross_margin']:+.4f}", flush=True)
    write_json(OUT_PAIRED, paired_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
