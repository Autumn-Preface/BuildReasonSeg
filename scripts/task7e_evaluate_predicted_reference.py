"""Task 7E Part I — E2 predicted-reference holdout audit (runs only after the section-13 gate passes).

Uses exactly the frozen U-C1 practical resolver from Task 6Q/7A: YOLO26m-seg Task 6M.1 checkpoint, imgsz 640,
conf 0.05, max_det 300, default NMS, no TTA, no tiling; largest eligibility = not border-touching and bbox
extent ratio <= 0.20; selection = maximum predicted mask area, ties by higher confidence then lower original
index. No ranker, no quality filter, no SAM2 refinement.

The **same** resolved reference mask feeds both pipelines per record (and is resolved once and reused for both
members of a pair):

* E-P0: canonical program -> predicted U-C1 largest reference -> frozen Z-B3
* E-P1: canonical program -> predicted U-C1 largest reference -> frozen D-B1

No GT reference or GT target enters inference; GT is used only for offline reference diagnostics. Reference
abstention counts the strict target IoU as 0.

Writes `evaluation/task7e_predicted_reference_quality.json`,
`evaluation/task7e_predicted_reference_holdout.json`, `evaluation/task7e_predicted_reference_paired.json`.

    python scripts/task7e_evaluate_predicted_reference.py
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
from buildreasonseg_mvp.task6q_reference_resolver import config_report  # noqa: E402
from buildreasonseg_mvp.task7e_l3_decoder_adapter import sha256_file  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import L3Pipeline, default_parser_checkpoint  # noqa: E402
from buildreasonseg_mvp.task7e_l3_decoder_adapter import FrozenL3Decoder, frozen_metadata  # noqa: E402
from scripts.task6n_train import FEATURE_ROOT, MaskStore  # noqa: E402
from scripts.task6u_common import CONFIGS, PROPOSAL_CHECKPOINT, iou  # noqa: E402
from scripts.task7a_evaluate_reference import aggregate, reference_bucket, target_row  # noqa: E402
from scripts.task7e_build_holdout import HOLDOUT_ROOT  # noqa: E402
from scripts.task7e_evaluate_oracle import (  # noqa: E402
    DIRECTIONS,
    OUT_ORACLE,
    TOLERANCE,
    read_holdout_rows,
    read_pairs,
    record_rows,
)

EVAL = REPO_ROOT / "evaluation"
OUT_QUALITY = EVAL / "task7e_predicted_reference_quality.json"
OUT_HOLDOUT = EVAL / "task7e_predicted_reference_holdout.json"
OUT_PAIRED = EVAL / "task7e_predicted_reference_paired.json"
GATE = {"strict_miou": 0.24, "answered_miou": 0.25, "delta_vs_p0": 0.03, "retention": 0.62,
        "paired": 11, "margin": 0.20, "abstention_rate": 0.10}
BUCKET_MAP = {"REFERENCE_NOT_COVERED_IOU50": "NOT_COVERED",
              "REFERENCE_SELECTION_WRONG": "SELECTION_WRONG",
              "REFERENCE_GEOMETRY_POOR": "GEOMETRY_POOR",
              "REFERENCE_OK": "REFERENCE_OK",
              "NO_PROPOSALS": "ABSTENTION", "NO_ELIGIBLE_PROPOSALS": "ABSTENTION"}


def effective_resolver_report() -> dict:
    """The frozen U-C1 configuration actually used (section 14): conf 0.05, max_det 300, imgsz 640."""

    config = dict(CONFIGS["U-C1"])
    return {
        "config": "U-C1", **config,
        "frozen_config_source": "scripts/task6u_common.CONFIGS['U-C1']",
        "checkpoint": str(PROPOSAL_CHECKPOINT),
        "checkpoint_sha256": sha256_file(PROPOSAL_CHECKPOINT),
        "family": "YOLO26m-seg",
        "mask_threshold": "canonical Task 6M (Ultralytics retina mask > 0 via normalize_mask)",
        "eligibility": {"not_border_touching": True, "bbox_extent_ratio_max": 0.20},
        "selection": ["max predicted mask area", "tie higher confidence", "then lower original index"],
        "ranker": False, "quality_filter": False, "sam2_refinement": False,
        "task6q_resolver_report": config_report(),
        "note": ("Task 6Q's read-only `config_report()` prints the resolver module defaults (conf 0.10 / "
                 "max_det 100); the Task 6U frozen `CONFIGS['U-C1']` entry used by the pipeline is conf "
                 "0.05 / max_det 300, which is what Task 7E section 14 mandates and what this run used."),
    }


def aggregate_with_answered(rows: list[dict]) -> dict:
    return aggregate(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    oracle = json.loads(OUT_ORACLE.read_text(encoding="utf-8"))
    if not oracle.get("DB1_HOLDOUT_GENERALIZES"):
        write_json(OUT_HOLDOUT, {"_doc": "Task 7E section 13.", "task": "7E",
                                 "verdict": "DB1_HOLDOUT_GENERALIZATION_FAIL",
                                 "reason": "section 13 gate did not pass; E2 must not run"})
        print("[7e.pred] STOP DB1_HOLDOUT_GENERALIZATION_FAIL", flush=True)
        return 3
    metadata = frozen_metadata()
    records = read_holdout_rows()
    pairs = read_pairs()

    # frozen U-C1 resolver + frozen SAM2 features; the parser is NOT used for inference
    pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device)
    store = pipeline.store
    masks = MaskStore()
    decoders = {name: FrozenL3Decoder(name, args.device) for name in ("Z-B3", "D-B1")}

    reference_cache: dict[str, dict] = {}

    def resolve(tile_id: str, image_path: Path) -> dict:
        """Resolve the U-C1 largest reference once per tile (shared by P0/P1 and by both pair members)."""

        if tile_id not in reference_cache:
            proposals = pipeline.proposals(tile_id, Path(image_path))
            bundle = pipeline.resolve_reference(proposals)
            reference_cache[tile_id] = {"proposals": len(proposals),
                                        "eligible": len(bundle.eligible),
                                        "selection": (np.asarray(bundle.selection["mask"], dtype=bool)
                                                      if bundle.selection is not None else None),
                                        "abstention_reason": bundle.abstention_reason,
                                        "eligible_masks": [np.asarray(proposal.mask, dtype=bool)
                                                           for proposal in bundle.eligible]}
        return reference_cache[tile_id]

    # ---------------- reference diagnostics (GT offline only) and target metrics
    reference_rows = []
    target_rows = {"Z-B3": [], "D-B1": []}
    for record in records:
        resolved = resolve(record["tile_id"], record["image_path"])
        truth_reference = np.asarray(masks.mask(record["tile_id"],
                                                record["reference_source_feature_id"]), dtype=bool)
        best_eligible = max((iou(mask, truth_reference) for mask in resolved["eligible_masks"]),
                            default=0.0)
        row = {"sample_id": record["sample_id"], "tile_id": record["tile_id"],
               "direction": record["direction"], "program_id": record["program_id"],
               "proposal_count": resolved["proposals"], "eligible_count": resolved["eligible"],
               "abstained": resolved["selection"] is None,
               "abstention_reason": resolved["abstention_reason"],
               "best_eligible_coverage_at_0_5": float(best_eligible >= 0.50)}
        if resolved["selection"] is None:
            row.update({"reference_iou": None, "reference_dice": None,
                        "bucket": "NO_PROPOSALS" if not resolved["proposals"]
                        else "NO_ELIGIBLE_PROPOSALS"})
        else:
            selected = resolved["selection"]
            intersection = float((selected & truth_reference).sum())
            row.update({"reference_iou": (intersection + TOLERANCE)
                        / (float((selected | truth_reference).sum()) + TOLERANCE),
                        "reference_dice": (2.0 * intersection + TOLERANCE)
                        / (float(selected.sum()) + float(truth_reference.sum()) + TOLERANCE),
                        "reference_precision_at_0_5": (intersection + TOLERANCE)
                        / (float(selected.sum()) + TOLERANCE)})
            row["bucket"] = reference_bucket(proposals=resolved["proposals"],
                                             eligible=resolved["eligible"],
                                             selected_iou=row["reference_iou"], centroid=None,
                                             best_eligible_iou=best_eligible)
        row["bucket_semantic"] = BUCKET_MAP[row["bucket"]]
        reference_rows.append(row)

        truth_target = np.asarray(masks.mask(record["tile_id"],
                                             record["target_source_feature_id"]), dtype=bool)
        if resolved["selection"] is None:
            for name in ("Z-B3", "D-B1"):
                target_rows[name].append({"sample_id": record["sample_id"],
                                          "direction": record["direction"],
                                          "miou": 0.0, "dice": 0.0, "precision_at_0_5": 0.0,
                                          "abstained": True, "bucket": row["bucket"],
                                          "bucket_semantic": row["bucket_semantic"],
                                          "reference_iou": None})
            continue
        predicted = _predict_with_reference(decoders, record, resolved["selection"], store,
                                            masks, args.device)
        for name in ("Z-B3", "D-B1"):
            target = target_row(predicted[name], truth_target, record)
            target.update({"abstained": False, "bucket": row["bucket"],
                           "bucket_semantic": row["bucket_semantic"],
                           "reference_iou": row["reference_iou"]})
            target_rows[name].append(target)

    answered = [row for row in reference_rows if row["reference_iou"] is not None]
    reference_metrics = {
        "records": len(reference_rows),
        "proposal_count_mean": float(np.mean([row["proposal_count"] for row in reference_rows])),
        "eligible_count_mean": float(np.mean([row["eligible_count"] for row in reference_rows])),
        "reference_miou": float(np.mean([row["reference_iou"] for row in answered]))
        if answered else None,
        "reference_dice": float(np.mean([row["reference_dice"] for row in answered]))
        if answered else None,
        "reference_precision_at_0_5": float(np.mean([row["reference_precision_at_0_5"]
                                                     for row in answered])) if answered else None,
        "abstentions": sum(1 for row in reference_rows if row["abstained"]),
        "abstention_rate": sum(1 for row in reference_rows if row["abstained"])
        / max(1, len(reference_rows)),
        "best_eligible_coverage_at_0_5": float(np.mean([row["best_eligible_coverage_at_0_5"]
                                                        for row in reference_rows])),
        "buckets": dict(Counter(row["bucket_semantic"] for row in reference_rows)),
        "raw_buckets": dict(Counter(row["bucket"] for row in reference_rows)),
        "frozen_bucket_definitions": ["NO_PROPOSALS", "NO_ELIGIBLE_PROPOSALS",
                                      "REFERENCE_NOT_COVERED_IOU50", "REFERENCE_SELECTION_WRONG",
                                      "REFERENCE_GEOMETRY_POOR", "REFERENCE_OK"],
    }
    write_json(OUT_QUALITY, {
        "_doc": ("Task 7E section 16. Predicted-reference quality on E-HoldoutL3 with the frozen U-C1 "
                 "resolver. GT is used offline only; abstention means no eligible largest reference, which "
                 "counts as strict target IoU 0."),
        "task": "7E", "stage": "I-reference-quality",
        "resolver": effective_resolver_report(),
        "checkpoints": metadata, "records": len(reference_rows),
        "metrics": reference_metrics, "rows": reference_rows,
        "training_performed": False, "test_split_used": False,
    })
    print(f"[7e.E2-ref] ref mIoU {reference_metrics['reference_miou']:.4f} abstentions "
          f"{reference_metrics['abstentions']} ({reference_metrics['abstention_rate']:.3%}) buckets "
          f"{reference_metrics['buckets']}", flush=True)

    # ---------------- target metrics
    oracle_db1 = oracle["results"]["D-B1"]["overall"]["miou"]
    oracle_zb3 = oracle["results"]["Z-B3"]["overall"]["miou"]
    results = {}
    for name in ("Z-B3", "D-B1"):
        rows = target_rows[name]
        ok_rows = [row for row in rows if row["bucket_semantic"] == "REFERENCE_OK"]
        results[name] = {
            "strict": aggregate_with_answered(rows),
            "reference_ok_subset": aggregate_with_answered(ok_rows),
            "per_direction": {direction: aggregate_with_answered(
                [row for row in rows if row["direction"] == direction]) for direction in DIRECTIONS},
            "abstentions": sum(1 for row in rows if row["abstained"]),
            "params": decoders[name].params, "checkpoint_sha256": decoders[name].sha256,
        }
    pred_db1_strict = results["D-B1"]["strict"]["miou"]
    retention = pred_db1_strict / oracle_db1 if oracle_db1 else None
    write_json(OUT_HOLDOUT, {
        "_doc": ("Task 7E section 17. E2 predicted-reference holdout: E-P0 (canonical program -> predicted "
                 "U-C1 largest reference -> frozen Z-B3) vs E-P1 (... -> frozen D-B1) on every E-HoldoutL3 "
                 "record. Both pipelines share the same resolved reference mask. No GT reference or target "
                 "enters inference."),
        "task": "7E", "stage": "I-predicted-reference-holdout",
        "checkpoints": metadata, "resolver": effective_resolver_report(),
        "records": len(records), "results": results,
        "retention": {"oracle_d_b1_miou": oracle_db1, "oracle_z_b3_miou": oracle_zb3,
                      "pred_d_b1_strict_miou": pred_db1_strict, "retention": retention,
                      "definition": "pred_db1_strict / oracle_db1"},
        "delta": {"d_b1_minus_z_b3_strict": (pred_db1_strict
                                             - results["Z-B3"]["strict"]["miou"]),
                  "answered_only": (results["D-B1"]["strict"]["answered_only_miou"]
                                    - results["Z-B3"]["strict"]["answered_only_miou"])},
        "gate_constants": GATE, "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7e.E2] E-P0 strict {results['Z-B3']['strict']['miou']:.4f} | E-P1 strict "
          f"{pred_db1_strict:.4f} answered {results['D-B1']['strict']['answered_only_miou']:.4f} | "
          f"delta {pred_db1_strict - results['Z-B3']['strict']['miou']:+.4f} | retention "
          f"{retention:.4f}", flush=True)

    # ---------------- paired (resolve reference once per pair)
    paired_results = {}
    per_pair = []
    reference_abstention_pairs = 0
    for name in ("Z-B3", "D-B1"):
        decoders[name].model.eval()
    for pair in pairs:
        members = [next(record for record in records if record["sample_id"] == member["sample_id"])
                   for member in pair["members"]]
        resolved = resolve(pair["tile_id"], members[0]["image_path"])
        if resolved["selection"] is None:
            reference_abstention_pairs += 1
        own = {name: [] for name in ("Z-B3", "D-B1")}
        cross = {name: [] for name in ("Z-B3", "D-B1")}
        predictions = {name: [] for name in ("Z-B3", "D-B1")}
        for record in members:
            if resolved["selection"] is None:
                for name in ("Z-B3", "D-B1"):
                    predictions[name].append(np.zeros((512, 512), dtype=bool))
                continue
            predicted = _predict_with_reference(decoders, record, resolved["selection"], store,
                                                masks, args.device)
            for name in ("Z-B3", "D-B1"):
                predictions[name].append(predicted[name])
        for index, record in enumerate(members):
            own_mask = np.asarray(masks.mask(record["tile_id"],
                                             record["target_source_feature_id"]), dtype=bool)
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other["tile_id"],
                                               other["target_source_feature_id"]), dtype=bool)
            for name in ("Z-B3", "D-B1"):
                prediction = predictions[name][index]
                own[name].append(float((prediction & own_mask).sum())
                                 / (float((prediction | own_mask).sum()) + TOLERANCE))
                cross[name].append(float((prediction & other_mask).sum())
                                   / (float((prediction | other_mask).sum()) + TOLERANCE))
        entry = {"pair_id": pair["pair_id"], "reference_abstained": resolved["selection"] is None}
        for name in ("Z-B3", "D-B1"):
            passed = bool(own[name][0] > cross[name][0] and own[name][1] > cross[name][1])
            entry[name] = {"passed": passed,
                           "own_iou": own[name], "cross_iou": cross[name],
                           "margin": float(np.mean(own[name]) - np.mean(cross[name]))}
        per_pair.append(entry)
    for name in ("Z-B3", "D-B1"):
        own_values = [value for entry in per_pair for value in entry[name]["own_iou"]]
        cross_values = [value for entry in per_pair for value in entry[name]["cross_iou"]]
        paired_results[name] = {
            "passed": sum(1 for entry in per_pair if entry[name]["passed"]),
            "pairs": len(per_pair),
            "pass_rate": sum(1 for entry in per_pair if entry[name]["passed"]) / max(1, len(per_pair)),
            "mean_own_iou": float(np.mean(own_values)), "mean_cross_iou": float(np.mean(cross_values)),
            "own_cross_margin": float(np.mean(own_values) - np.mean(cross_values))}
    write_json(OUT_PAIRED, {
        "_doc": ("Task 7E section 18. Predicted-reference E-PairedHoldout20: because pair members share the "
                 "same tile and oracle reference, the U-C1 reference is resolved once per pair and reused "
                 "for both members and both decoders."),
        "task": "7E", "stage": "I-predicted-reference-paired",
        "pairs": {"count": len(pairs), "reference_abstention_pairs": reference_abstention_pairs,
                  "pair_ids": [pair["pair_id"] for pair in pairs]},
        "results": paired_results, "per_pair": per_pair,
        "training_performed": False, "test_split_used": False,
    })
    print(f"[7e.E2-paired] Z-B3 {paired_results['Z-B3']['passed']}/20 | D-B1 "
          f"{paired_results['D-B1']['passed']}/20 margin {paired_results['D-B1']['own_cross_margin']:+.4f}"
          f" | reference-abstention pairs {reference_abstention_pairs}", flush=True)

    # ---------------- section 19 gate
    gate = {
        "1_strict_miou": {"required": GATE["strict_miou"], "measured": pred_db1_strict,
                          "passed": pred_db1_strict >= GATE["strict_miou"]},
        "2_answered_only_miou": {"required": GATE["answered_miou"],
                                 "measured": results["D-B1"]["strict"]["answered_only_miou"],
                                 "passed": (results["D-B1"]["strict"]["answered_only_miou"] or 0.0)
                                 >= GATE["answered_miou"]},
        "3_delta_vs_p0": {"required": GATE["delta_vs_p0"],
                          "measured": pred_db1_strict - results["Z-B3"]["strict"]["miou"],
                          "passed": (pred_db1_strict - results["Z-B3"]["strict"]["miou"])
                          >= GATE["delta_vs_p0"]},
        "4_retention": {"required": GATE["retention"], "measured": retention,
                        "passed": (retention or 0.0) >= GATE["retention"]},
        "5_paired": {"required": f">= {GATE['paired']}/20",
                     "measured": paired_results["D-B1"]["passed"],
                     "passed": paired_results["D-B1"]["passed"] >= GATE["paired"]},
        "6_margin": {"required": GATE["margin"],
                     "measured": paired_results["D-B1"]["own_cross_margin"],
                     "passed": paired_results["D-B1"]["own_cross_margin"] >= GATE["margin"]},
        "7_abstention_rate": {"required": f"<= {GATE['abstention_rate']:.0%}",
                              "measured": reference_metrics["abstention_rate"],
                              "passed": reference_metrics["abstention_rate"]
                              <= GATE["abstention_rate"]},
    }
    passed = all(entry["passed"] for entry in gate.values())
    payload = json.loads(OUT_HOLDOUT.read_text(encoding="utf-8"))
    payload["gate"] = gate
    payload["DB1_PREDICTED_REFERENCE_USABLE"] = passed
    write_json(OUT_HOLDOUT, payload)
    print(f"[7e.gate2] DB1_PREDICTED_REFERENCE_USABLE {passed} "
          f"({sum(1 for entry in gate.values() if entry['passed'])}/7 conditions)", flush=True)
    return 0 if passed else 5


def _predict_with_reference(decoders: dict, record: dict, reference_mask: np.ndarray,
                            store, masks, device: str):
    """Run both frozen decoders on one record with the shared predicted reference (no code change).

    The predicted mask is injected through a read-only `MaskStore` wrapper under a synthetic reference key, so
    each decoder keeps running its own frozen batch builder, field computation and prediction path; the only
    difference from the oracle run is which mask the store returns.
    """

    synthetic_id = -1
    overridden = dict(record)
    overridden["reference_source_feature_id"] = synthetic_id
    overridden["sample_id"] = f"{record['sample_id']}:pred"
    wrapper = _ReferenceOverride(masks, {(record["tile_id"], synthetic_id): reference_mask})
    predictions = {}
    for name in ("Z-B3", "D-B1"):
        predicted = decoders[name].predict([overridden], store, wrapper, {}, batch_size=1)
        predictions[name] = predicted[0]
    return predictions


class _ReferenceOverride:
    """Read-only `MaskStore` facade returning the predicted reference for one synthetic key."""

    def __init__(self, base, overrides: dict) -> None:
        self._base = base
        self._overrides = overrides

    def mask(self, tile_id: str, source_feature_id: int) -> np.ndarray:
        key = (tile_id, source_feature_id)
        if key in self._overrides:
            return self._overrides[key]
        return self._base.mask(tile_id, source_feature_id)

    def instances(self, tile_id: str):
        return self._base.instances(tile_id)


if __name__ == "__main__":
    raise SystemExit(main())
