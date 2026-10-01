"""Task 7E Parts F-H — E0 reproduction, E1 oracle-reference holdout and the oracle generalization gate.

E0 (section 9): on the exact Z-MiniVal240 pack, Z-B3 mIoU must equal `0.3242128982543474` and D-B1
`0.3978996298363562` with |delta| <= 1e-6, and the frozen paired results must be Z-B3 15/20 and D-B1 19/20;
otherwise STOP `TASK7D_REPRODUCTION_FAIL`.

E1 (sections 10-12): on every `E-HoldoutL3` record, compare the frozen Z-B3 (E-O0) with the frozen D-B1
(E-O1) using the canonical program id and the canonical **oracle** largest reference mask only — no parser, no
predicted reference, no retraining. Report record count, mIoU, Dice, Pr@0.5, per program/direction mIoU,
target-area and boundary-distance quartiles, the overall and per-direction delta, and a deterministic paired
bootstrap (seed 20261001, 2000 resamples by record id) with the 95 % percentile CI of the mean IoU delta. The
new `E-PairedHoldout20` is scored for both decoders.

Section 13 gate `DB1_HOLDOUT_GENERALIZES` requires D-B1 holdout mIoU >= 0.36, delta >= +0.05, improvement or
tie in >= 3/4 directions, bootstrap CI lower bound > 0.00, paired >= 16/20 and margin >= 0.30.

    python scripts/task7e_evaluate_oracle.py
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
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import FrozenFeatureStore, load_frozen_sam2_encoder  # noqa: E402
from buildreasonseg_mvp.task7d_data import load_pack  # noqa: E402
from buildreasonseg_mvp.task7e_l3_decoder_adapter import FrozenL3Decoder, frozen_metadata  # noqa: E402
from scripts.task6n_train import FEATURE_ROOT, MaskStore, canonical_instances  # noqa: E402
from scripts.task7e_build_holdout import HOLDOUT_ROOT  # noqa: E402
from scripts.task7d_evaluate import component_map_for  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT_MINIVAL = EVAL / "task7e_minival_reproduction.json"
OUT_ORACLE = EVAL / "task7e_oracle_holdout.json"
OUT_PAIRED = EVAL / "task7e_oracle_holdout_paired.json"
HOLDOUT_MANIFEST = EVAL / "task7e_holdout_manifest.json"
TOLERANCE = 1.0e-6
EXPECTED = {"Z-B3": 0.3242128982543474, "D-B1": 0.3978996298363562}
EXPECTED_PAIRED = {"Z-B3": 15, "D-B1": 19}
PAIRED_COUNT = 20
BOOTSTRAP_SEED = 20261001
BOOTSTRAP_RESAMPLES = 2000
CI = (2.5, 97.5)
GATE = {"d_b1_miou": 0.36, "delta": 0.05, "directions_improved": 3, "ci_lower": 0.0,
        "paired": 16, "margin": 0.30}
DIRECTIONS = ("above", "below", "left", "right")


def read_holdout_rows() -> list[dict]:
    rows = []
    with (HOLDOUT_ROOT / "holdout_l3_rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_pairs() -> list[dict]:
    return json.loads((HOLDOUT_ROOT / "holdout_pairs.json").read_text(encoding="utf-8"))["pairs"]


def record_rows(predictions: list[np.ndarray], records: list[dict], masks) -> list[dict]:
    rows = []
    for record, prediction in zip(records, predictions):
        truth = np.asarray(masks.mask(record["tile_id"], record["target_source_feature_id"]),
                           dtype=bool)
        instance = next((item for item in canonical_instances(record["tile_id"])
                         if item.source_feature_id == record["target_source_feature_id"]), None)
        intersection = float((prediction & truth).sum())
        rows.append({
            "sample_id": record["sample_id"], "program_id": record["program_id"],
            "direction": record["direction"],
            "miou": (intersection + TOLERANCE) / (float((prediction | truth).sum()) + TOLERANCE),
            "dice": (2.0 * intersection + TOLERANCE)
            / (float(prediction.sum()) + float(truth.sum()) + TOLERANCE),
            "precision_at_0_5": (intersection + TOLERANCE)
            / (float(prediction.sum()) + TOLERANCE),
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
        rows[-1]["boundary_distance_px"] = distance
    return rows


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"records": 0}
    return {"records": len(rows),
            "miou": float(np.mean([row["miou"] for row in rows])),
            "dice": float(np.mean([row["dice"] for row in rows])),
            "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows]))}


def quartiles(rows: list[dict], key: str) -> dict:
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
                  and ((lower <= row[key] <= upper) if index == 3 else (lower <= row[key] < upper))]
        if subset:
            buckets[label] = {"records": len(subset),
                              "miou": float(np.mean([row["miou"] for row in subset]))}
    return {"edges": [float(value) for value in edges], "buckets": buckets}


def evaluate_pairs(decoder: FrozenL3Decoder, pairs: list[dict], records_by_id: dict, store, masks,
                   precomputed: dict) -> dict:
    passed = 0
    own_values, cross_values = [], []
    per_pair = []
    for pair in pairs:
        members = [records_by_id[member["sample_id"]] for member in pair["members"]]
        predictions = decoder.predict(members, store, masks, precomputed, batch_size=2)
        own, cross = [], []
        for index, member in enumerate(members):
            own_mask = np.asarray(masks.mask(member["tile_id"],
                                             member["target_source_feature_id"]), dtype=bool)
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other["tile_id"],
                                               other["target_source_feature_id"]), dtype=bool)
            own.append(float((predictions[index] & own_mask).sum())
                       / (float((predictions[index] | own_mask).sum()) + TOLERANCE))
            cross.append(float((predictions[index] & other_mask).sum())
                         / (float((predictions[index] | other_mask).sum()) + TOLERANCE))
        pair_passed = bool(own[0] > cross[0] and own[1] > cross[1])
        passed += int(pair_passed)
        own_values.extend(own)
        cross_values.extend(cross)
        per_pair.append({"pair_id": pair["pair_id"], "passed": pair_passed,
                         "own_iou": own, "cross_iou": cross,
                         "margin": float(np.mean(own) - np.mean(cross))})
    return {"passed": passed, "pairs": len(pairs), "pass_rate": passed / max(1, len(pairs)),
            "mean_own_iou": float(np.mean(own_values)), "mean_cross_iou": float(np.mean(cross_values)),
            "own_cross_margin": float(np.mean(own_values) - np.mean(cross_values)),
            "per_pair": per_pair}


def bootstrap_ci(deltas: np.ndarray) -> dict:
    """Deterministic paired bootstrap over record ids (seed and count are predeclared)."""

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    count = deltas.shape[0]
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = generator.integers(0, count, size=count)
        means[index] = deltas[sample].mean()
    lower, upper = np.percentile(means, CI)
    return {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
            "unit": "record_id", "paired": True, "ci_percent": list(CI),
            "mean_delta": float(deltas.mean()), "ci_lower": float(lower), "ci_upper": float(upper),
            "bootstrap_mean": float(means.mean())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    metadata = frozen_metadata()
    if not metadata["d_b1"]["matches"]:
        write_json(OUT_MINIVAL, {"_doc": "Task 7E section 2.", "task": "7E",
                                 "verdict": "DB1_CHECKPOINT_UNAVAILABLE", "checkpoint": metadata})
        print("[7e.oracle] STOP DB1_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2
    manifest = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    if manifest["verdict"] != "L3_HOLDOUT_READY":
        write_json(OUT_MINIVAL, {"_doc": "Task 7E section 7.", "task": "7E",
                                 "verdict": manifest["verdict"]})
        print(f"[7e.oracle] STOP {manifest['verdict']}", flush=True)
        return 3

    encoder, _report = load_frozen_sam2_encoder(device=args.device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.device)
    masks = MaskStore()
    decoders = {name: FrozenL3Decoder(name, args.device) for name in ("Z-B3", "D-B1")}

    # ---------------- E0: MiniVal240 reproduction
    mini_records = load_pack("z_mini_val_240")
    precomputed: dict[str, dict] = {}
    mini_metrics, mini_pairs, mini_predicted = {}, {}, {}
    for name, decoder in decoders.items():
        predictions = decoder.predict(mini_records, store, masks, precomputed, batch_size=8)
        mini_predicted[name] = predictions
        rows = record_rows(predictions, mini_records, masks)
        mini_metrics[name] = summarise(rows)
    paired_records = load_pack("z_paired_val20")
    mini_pairs_source = [{"pair_id": f"task6z-{index // 2}", "members": [
        {"sample_id": paired_records[index]["sample_id"],
         "target_source_feature_id": paired_records[index]["target_source_feature_id"]},
        {"sample_id": paired_records[index + 1]["sample_id"],
         "target_source_feature_id": paired_records[index + 1]["target_source_feature_id"]}]}
        for index in range(0, len(paired_records), 2)]
    mini_by_id = {record["sample_id"]: record for record in paired_records}
    for name, decoder in decoders.items():
        mini_pairs[name] = evaluate_pairs(decoder, mini_pairs_source, mini_by_id, store, masks,
                                          precomputed)
    reproduction = {}
    for name in ("Z-B3", "D-B1"):
        delta = abs(mini_metrics[name]["miou"] - EXPECTED[name])
        reproduction[name] = {"miou": mini_metrics[name]["miou"], "expected_miou": EXPECTED[name],
                              "delta": delta, "miou_ok": delta <= TOLERANCE,
                              "paired_passed": mini_pairs[name]["passed"],
                              "expected_paired": EXPECTED_PAIRED[name],
                              "paired_ok": mini_pairs[name]["passed"] == EXPECTED_PAIRED[name],
                              "dice": mini_metrics[name]["dice"]}
    reproduction_passed = all(entry["miou_ok"] and entry["paired_ok"]
                              for entry in reproduction.values())
    write_json(OUT_MINIVAL, {
        "_doc": ("Task 7E section 9. E0: exact Z-MiniVal240 / Z-PairedVal20 reproduction of the frozen Task "
                 "7D measurements before any holdout evaluation. Both decoders are loaded read-only."),
        "task": "7E", "stage": "F-minival-reproduction",
        "checkpoints": metadata, "pack": {"name": "z_mini_val_240", "records": len(mini_records),
                                          "paired": len(mini_pairs_source)},
        "results": {"metrics": mini_metrics,
                    "paired": {name: {key: value for key, value in mini_pairs[name].items()
                                      if key != "per_pair"} for name in mini_pairs}},
        "reproduction": reproduction, "tolerance": TOLERANCE,
        "reproduction_passed": reproduction_passed,
        "verdict": ("TASK7D_REPRODUCTION_PASS" if reproduction_passed
                    else "TASK7D_REPRODUCTION_FAIL"),
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7e.E0] Z-B3 {mini_metrics['Z-B3']['miou']:.10f} (Δ"
          f"{reproduction['Z-B3']['delta']:.2e}) paired {mini_pairs['Z-B3']['passed']}/20 | D-B1 "
          f"{mini_metrics['D-B1']['miou']:.10f} (Δ{reproduction['D-B1']['delta']:.2e}) paired "
          f"{mini_pairs['D-B1']['passed']}/20 -> "
          f"{'PASS' if reproduction_passed else 'FAIL'}", flush=True)
    if not reproduction_passed:
        return 4

    # ---------------- E1: oracle-reference holdout
    holdout_records = read_holdout_rows()
    oracle_predictions, oracle_rows = {}, {}
    for name, decoder in decoders.items():
        predictions = decoder.predict(holdout_records, store, masks, precomputed, batch_size=8)
        oracle_predictions[name] = predictions
        oracle_rows[name] = record_rows(predictions, holdout_records, masks)
    by_id = {row["sample_id"]: row for row in oracle_rows["Z-B3"]}
    deltas = np.asarray([oracle_rows["D-B1"][index]["miou"] - oracle_rows["Z-B3"][index]["miou"]
                         for index in range(len(holdout_records))], dtype=np.float64)
    bootstrap = bootstrap_ci(deltas)
    per_direction = {}
    for direction in DIRECTIONS:
        z_rows = [row for row in oracle_rows["Z-B3"] if row["direction"] == direction]
        d_rows = [row for row in oracle_rows["D-B1"] if row["direction"] == direction]
        per_direction[direction] = {
            "records": len(z_rows), "z_b3_miou": summarise(z_rows)["miou"],
            "d_b1_miou": summarise(d_rows)["miou"],
            "delta": summarise(d_rows)["miou"] - summarise(z_rows)["miou"]}
    directions_improved = sum(1 for entry in per_direction.values() if entry["delta"] >= 0.0)
    oracle_holdout = {
        "_doc": ("Task 7E sections 10-11. E1 oracle-reference L3 holdout: frozen Z-B3 (E-O0) vs frozen D-B1 "
                 "(E-O1) on the untouched E-HoldoutL3 remainder, canonical program id + canonical oracle "
                 "largest reference mask only, no parser, no predicted reference, no retraining."),
        "task": "7E", "stage": "G-oracle-holdout",
        "checkpoints": metadata, "holdout": manifest["holdout"],
        "source": "evaluation/task7e_holdout_manifest.json",
        "records": len(holdout_records),
        "results": {name: {"overall": summarise(oracle_rows[name]),
                           "per_program": {program: summarise([row for row in oracle_rows[name]
                                                               if row["program_id"] == program])
                                           for program in sorted(manifest["holdout"]["per_program"])},
                           "per_direction": {direction: summarise([row for row in oracle_rows[name]
                                                                   if row["direction"] == direction])
                                             for direction in DIRECTIONS},
                           "target_area_quartiles": quartiles(oracle_rows[name], "target_area_px"),
                           "boundary_distance_quartiles": quartiles(oracle_rows[name],
                                                                    "boundary_distance_px"),
                           "border_target": summarise([row for row in oracle_rows[name]
                                                       if row["target_touches_border"]]),
                           "tiny_target": summarise([row for row in oracle_rows[name]
                                                     if row["target_tiny"]]),
                           "params": decoders[name].params,
                           "checkpoint_sha256": decoders[name].sha256}
                   for name in ("Z-B3", "D-B1")},
        "delta": {"overall": (summarise(oracle_rows["D-B1"])["miou"]
                              - summarise(oracle_rows["Z-B3"])["miou"]),
                  "per_direction": {direction: entry["delta"]
                                    for direction, entry in per_direction.items()},
                  "dice": (summarise(oracle_rows["D-B1"])["dice"]
                           - summarise(oracle_rows["Z-B3"])["dice"]),
                  "precision_at_0_5": (summarise(oracle_rows["D-B1"])["precision_at_0_5"]
                                       - summarise(oracle_rows["Z-B3"])["precision_at_0_5"])},
        "per_direction": per_direction,
        "bootstrap": bootstrap, "bootstrap_sample_ids_hash": manifest["holdout"]["record_id_hash"],
        "gate_constants": GATE, "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_ORACLE, oracle_holdout)
    z_overall = oracle_holdout["results"]["Z-B3"]["overall"]["miou"]
    d_overall = oracle_holdout["results"]["D-B1"]["overall"]["miou"]
    print(f"[7e.E1] holdout {len(holdout_records)} | Z-B3 {z_overall:.4f} D-B1 {d_overall:.4f} delta "
          f"{d_overall - z_overall:+.4f} | CI [{bootstrap['ci_lower']:+.4f}, "
          f"{bootstrap['ci_upper']:+.4f}] | improved directions {directions_improved}/4", flush=True)

    # ---------------- E1 paired
    pairs = read_pairs()
    holdout_by_id = {record["sample_id"]: record for record in holdout_records}
    paired_results = {name: evaluate_pairs(decoders[name], pairs, holdout_by_id, store, masks,
                                           precomputed) for name in ("Z-B3", "D-B1")}
    write_json(OUT_PAIRED, {
        "_doc": ("Task 7E section 12. E-PairedHoldout20 oracle audit: newly frozen held-out pairs from "
                 "E-HoldoutL3 (same tile, same oracle largest reference, different direction programs and "
                 "different targets), scored for frozen Z-B3 and frozen D-B1."),
        "task": "7E", "stage": "G-oracle-holdout-paired",
        "pairs": {"count": len(pairs), "pair_ids": manifest["paired"]["pair_ids"],
                  "pair_id_hash": manifest["paired"]["pair_id_hash"],
                  "overlap_with_task6z_paired_members":
                      manifest["paired"]["overlap_with_task6z_paired_members"]},
        "results": paired_results, "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    d_b1_paired = paired_results["D-B1"]
    print(f"[7e.E1-paired] Z-B3 {paired_results['Z-B3']['passed']}/20 (margin "
          f"{paired_results['Z-B3']['own_cross_margin']:+.4f}) | D-B1 {d_b1_paired['passed']}/20 (margin "
          f"{d_b1_paired['own_cross_margin']:+.4f})", flush=True)

    # ---------------- section 13 gate
    gate = {
        "1_d_b1_miou": {"required": GATE["d_b1_miou"], "measured": d_overall,
                        "passed": d_overall >= GATE["d_b1_miou"]},
        "2_delta_overall": {"required": GATE["delta"], "measured": d_overall - z_overall,
                            "passed": (d_overall - z_overall) >= GATE["delta"]},
        "3_directions": {"required": f">= {GATE['directions_improved']}/4",
                         "measured": f"{directions_improved}/4",
                         "passed": directions_improved >= GATE["directions_improved"]},
        "4_bootstrap_ci_lower": {"required": f"> {GATE['ci_lower']}",
                                 "measured": bootstrap["ci_lower"],
                                 "passed": bootstrap["ci_lower"] > GATE["ci_lower"]},
        "5_paired": {"required": f">= {GATE['paired']}/20", "measured": d_b1_paired["passed"],
                     "passed": d_b1_paired["passed"] >= GATE["paired"]},
        "6_margin": {"required": GATE["margin"], "measured": d_b1_paired["own_cross_margin"],
                     "passed": d_b1_paired["own_cross_margin"] >= GATE["margin"]},
    }
    passed = all(entry["passed"] for entry in gate.values())
    oracle_holdout["gate"] = gate
    oracle_holdout["DB1_HOLDOUT_GENERALIZES"] = passed
    write_json(OUT_ORACLE, oracle_holdout)
    print(f"[7e.gate] DB1_HOLDOUT_GENERALIZES {passed} "
          f"({sum(1 for entry in gate.values() if entry['passed'])}/6 conditions)", flush=True)
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
