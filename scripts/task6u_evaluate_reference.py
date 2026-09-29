"""Task 6U Parts F and H — reference candidate audit and the three-selector comparison on RefValUnique.

Part F (section 11): compare U-C0 with the frozen selected configuration on the untouched Task 6P
RefValUnique (219 references), including the **oracle-selection ceiling** (best eligible proposal chosen
with GT — diagnostic only, never in inference).

Part H (sections 17-18): exactly three systems

* **U-S0** — U-C0 proposals + frozen Task 6Q deterministic area selector;
* **U-S1** — selected U-C* proposals + the same deterministic selector;
* **U-S2** — selected U-C* proposals + ProposalSetRanker v0.1.

with selected-reference mIoU/Dice/Pr@0.5, centroid error mean/median/p90, area-ratio median, abstention
rate, largest/smallest breakdowns and the six Task 6Q failure buckets.

Writes `evaluation/task6u_refval_candidate_audit.json` and
`evaluation/task6u_refval_selector_comparison.json`.

    python scripts/task6u_evaluate_reference.py
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
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible, select_reference  # noqa: E402
from buildreasonseg_mvp.task6u_reference_ranker import (  # noqa: E402
    ProposalSetRanker,
    proposal_features,
    select_with_ranker,
)
from scripts.task6u_common import (  # noqa: E402
    BUCKET_ORDER,
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    attribute_failure,
    bucket_counts,
    centroid_error,
    coverage_for_records,
    dice,
    gt_reference_mask,
    group_records_by_tile,
    iou,
    percentile,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)

OUT_AUDIT = EVAL / "task6u_refval_candidate_audit.json"
OUT_COMPARISON = EVAL / "task6u_refval_selector_comparison.json"
PACK = REPO_ROOT / "artifacts" / "task6p" / "reference_packs" / "ref_val_unique.json"
SELECTED = EVAL / "task6u_selected_proposal_config.json"
RANKER_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6u" / "reference_ranker_v01.pt"
COVERED_IOU = 0.50
CENTROID_POOR = 0.05


def load_records() -> list[dict]:
    return json.loads(PACK.read_text(encoding="utf-8"))["records"]


def run_pack(model, config: dict, records: list[dict], device: str) -> dict:
    """Proposal sets per record for one configuration (cached), keyed by unique reference key."""

    grouped = group_records_by_tile(records)
    proposals_by_key: dict[tuple, list] = {}
    tile_runs = {}
    for tile_id, tile_records in grouped.items():
        image_path = record_image_path(tile_records[0])
        proposals, run = proposals_for_tile(model, config, tile_id, image_path, device,
                                           use_cache=True)
        tile_runs[tile_id] = run
        for record in tile_records:
            key = (tile_id, int(record["reference_source_feature_id"]),
                   str(record["reference_family"]))
            proposals_by_key[key] = proposals
    return {"proposals_by_key": proposals_by_key, "tile_runs": tile_runs}


def evaluate_selector(records: list[dict], proposals_by_key: dict, selector: str, ranker=None,
                      device: str = "cpu") -> dict:
    rows = []
    for record in records:
        family = str(record["reference_family"])
        key = (str(record["tile_id"]), int(record["reference_source_feature_id"]), family)
        proposals = proposals_by_key.get(key, [])
        truth = gt_reference_mask(record)
        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        eligible_ious = [iou(proposal.mask, truth) for proposal in eligible]
        best_eligible = max(eligible_ious) if eligible_ious else 0.0

        selected_mask = None
        selection_detail = {}
        if selector == "deterministic":
            selection = select_reference(proposals, family)
            if not selection.abstained:
                selected_mask = selection.mask
                selection_detail = {
                    "selected_area_px": int(selection.proposal.area_px),
                    "selected_confidence": float(selection.proposal.confidence),
                    "selected_index": int(selection.proposal.index),
                }
        else:
            if eligible:
                outcome = select_with_ranker(ranker, eligible, family, device=device)
                selected_mask = eligible[outcome["selected_index"]].mask
                selection_detail = {
                    "selected_area_px": int(eligible[outcome["selected_index"]].area_px),
                    "selected_confidence": float(eligible[outcome["selected_index"]].confidence),
                    "selected_index": int(eligible[outcome["selected_index"]].index),
                    "ranker_score": outcome["scores"][outcome["selected_index"]],
                }

        row = {
            "reference_key": list(key), "reference_family": family,
            "proposal_count": len(proposals), "eligible_count": len(eligible),
            "best_eligible_iou": best_eligible,
            "abstained": selected_mask is None,
            **selection_detail,
        }
        if selected_mask is None:
            row.update({"selected_iou": None, "selected_dice": None, "centroid_error": None,
                        "area_ratio": None,
                        "bucket": attribute_failure(proposal_count=len(proposals),
                                                    eligible_count=len(eligible),
                                                    best_eligible_iou=best_eligible,
                                                    selected_iou=None,
                                                    selected_centroid_error=None)})
        else:
            selected_iou = iou(selected_mask, truth)
            selected_dice = dice(selected_mask, truth)
            error = centroid_error(selected_mask, truth)
            area_pred = float(selected_mask.sum())
            area_truth = float(truth.sum())
            row.update({
                "selected_iou": selected_iou, "selected_dice": selected_dice,
                "centroid_error": error,
                "area_ratio": (area_pred / area_truth) if area_truth > 0 else None,
                "bucket": attribute_failure(proposal_count=len(proposals),
                                            eligible_count=len(eligible),
                                            best_eligible_iou=best_eligible,
                                            selected_iou=selected_iou,
                                            selected_centroid_error=error),
            })
        rows.append(row)

    answered = [row for row in rows if not row["abstained"]]

    def summarise(subset: list[dict], label: str) -> dict:
        if not subset:
            return {"records": 0}
        ious = [row["selected_iou"] for row in subset if row["selected_iou"] is not None]
        dices = [row["selected_dice"] for row in subset if row["selected_dice"] is not None]
        errors = [row["centroid_error"] for row in subset if row["centroid_error"] is not None]
        ratios = [row["area_ratio"] for row in subset if row["area_ratio"] is not None]
        return {
            "records": len(subset),
            "answered": len(ious),
            "selected_reference_miou": float(np.mean(ious)) if ious else None,
            "selected_reference_dice": float(np.mean(dices)) if dices else None,
            "precision_at_0_5": float(np.mean([value >= 0.50 for value in ious])) if ious else None,
            "centroid_error_mean": float(np.mean(errors)) if errors else None,
            "centroid_error_median": float(np.median(errors)) if errors else None,
            "centroid_error_p90": percentile(errors, 0.90),
            "area_ratio_median": float(np.median(ratios)) if ratios else None,
            "abstention_rate": sum(1 for row in subset if row["abstained"]) / len(subset),
            "buckets": bucket_counts(subset),
        }

    overall = summarise(rows, "overall")
    largest = summarise([row for row in rows if row["reference_family"] == "largest"], "largest")
    smallest = summarise([row for row in rows if row["reference_family"] == "smallest"], "smallest")
    return {
        "selector": selector,
        "overall": overall,
        "largest": largest,
        "smallest": smallest,
        "largest_miou": largest.get("selected_reference_miou"),
        "largest_precision_at_0_5": largest.get("precision_at_0_5"),
        "smallest_miou": smallest.get("selected_reference_miou"),
        "smallest_precision_at_0_5": smallest.get("precision_at_0_5"),
        "rows": rows,
    }


def oracle_ceiling(records: list[dict], proposals_by_key: dict) -> dict:
    """Diagnostic ceiling: GT chooses the best eligible proposal (never part of inference)."""

    rows = []
    for record in records:
        family = str(record["reference_family"])
        key = (str(record["tile_id"]), int(record["reference_source_feature_id"]), family)
        proposals = proposals_by_key.get(key, [])
        truth = gt_reference_mask(record)
        eligible = [proposal for proposal in proposals if is_eligible(proposal, family)]
        if not eligible:
            rows.append({"reference_key": list(key), "family": family, "available": False})
            continue
        ious = [iou(proposal.mask, truth) for proposal in eligible]
        best = int(np.argmax(ious))
        mask = eligible[best].mask
        rows.append({
            "reference_key": list(key), "family": family, "available": True,
            "miou": float(ious[best]), "dice": dice(mask, truth),
            "centroid_error": centroid_error(mask, truth),
            "area_ratio": float(mask.sum()) / max(1.0, float(truth.sum())),
        })
    available = [row for row in rows if row["available"]]

    def summarise(subset: list[dict]) -> dict:
        if not subset:
            return {"records": 0}
        return {
            "records": len(subset),
            "selected_reference_miou": float(np.mean([row["miou"] for row in subset])),
            "dice": float(np.mean([row["dice"] for row in subset])),
            "precision_at_0_5": float(np.mean([row["miou"] >= 0.50 for row in subset])),
            "centroid_error_median": float(np.median([row["centroid_error"] for row in subset])),
            "centroid_error_p90": percentile([row["centroid_error"] for row in subset], 0.90),
        }

    return {
        "available_records": len(available),
        "overall": summarise(available),
        "largest": summarise([row for row in available if row["family"] == "largest"]),
        "smallest": summarise([row for row in available if row["family"] == "smallest"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256:
        write_json(OUT_AUDIT, {"_doc": "Task 6U section 11.", "task": "6U",
                               "verdict": "PROPOSAL_CHECKPOINT_UNAVAILABLE"})
        return 2
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    selected_config = CONFIGS[selected["selected_config"]]
    baseline_config = CONFIGS["U-C0"]
    records = load_records()
    print(f"[6u.ref] RefValUnique {len(records)} references; baseline {baseline_config['id']}, "
          f"selected {selected_config['id']}", flush=True)

    from ultralytics import YOLO

    model = YOLO(str(PROPOSAL_CHECKPOINT))
    baseline_run = run_pack(model, baseline_config, records, args.device)
    selected_run = run_pack(model, selected_config, records, args.device)

    baseline_coverage = coverage_for_records(records, baseline_run["proposals_by_key"])
    selected_coverage = coverage_for_records(records, selected_run["proposals_by_key"])

    def coverage_summary(coverage: dict) -> dict:
        return {key: value for key, value in coverage.items() if key != "records"}

    def tile_stats(run: dict) -> dict:
        times = [value["seconds"] for value in run["tile_runs"].values()
                 if value.get("seconds") is not None]
        peaks = [value["peak_vram_gb"] for value in run["tile_runs"].values()
                 if value.get("peak_vram_gb") is not None]
        return {"tiles": len(run["tile_runs"]),
                "mean_seconds_per_tile": float(np.mean(times)) if times else None,
                "peak_vram_gb": max(peaks) if peaks else None}

    audit = {
        "_doc": (
            "Task 6U section 11. Reference candidate audit on the untouched Task 6P RefValUnique: the "
            "frozen U-C0 baseline configuration versus the frozen selected configuration, with the "
            "oracle-selection ceiling (diagnostic only; GT never enters inference)."
        ),
        "task": "6U", "stage": "F-refval-candidate-audit",
        "pack": {"path": str(PACK), "records": len(records),
                 "sha256": sha256_file(PACK),
                 "by_family": dict(Counter(record["reference_family"] for record in records))},
        "checkpoint": {"path": str(PROPOSAL_CHECKPOINT), "sha256": checkpoint_sha},
        "configs": {
            "baseline": {**baseline_config, "coverage": coverage_summary(baseline_coverage),
                         "tiles": tile_stats(baseline_run)},
            "selected": {**selected_config, "coverage": coverage_summary(selected_coverage),
                         "tiles": tile_stats(selected_run)},
        },
        "oracle_selection_ceiling": {
            "baseline": oracle_ceiling(records, baseline_run["proposals_by_key"]),
            "selected": oracle_ceiling(records, selected_run["proposals_by_key"]),
        },
        "candidate_coverage_improved": bool(
            selected_coverage["overall"]["eligible_coverage@0.50"]
            >= baseline_coverage["overall"]["eligible_coverage@0.50"] + 0.04
            and selected_coverage["smallest"]["eligible_coverage@0.50"]
            >= baseline_coverage["smallest"]["eligible_coverage@0.50"] + 0.06),
        "delta": {
            "overall_eligible_coverage@0.50":
                selected_coverage["overall"]["eligible_coverage@0.50"]
                - baseline_coverage["overall"]["eligible_coverage@0.50"],
            "smallest_eligible_coverage@0.50":
                selected_coverage["smallest"]["eligible_coverage@0.50"]
                - baseline_coverage["smallest"]["eligible_coverage@0.50"],
            "largest_eligible_coverage@0.50":
                selected_coverage["largest"]["eligible_coverage@0.50"]
                - baseline_coverage["largest"]["eligible_coverage@0.50"],
        },
        "task6q_reference_values": {"overall": 0.6895, "largest": 0.8364, "smallest": 0.5413},
        "test_split_used": False,
    }
    write_json(OUT_AUDIT, audit)

    ranker = None
    ranker_sha = None
    if RANKER_CHECKPOINT.is_file():
        ranker = ProposalSetRanker().to("cpu")
        payload = torch.load(RANKER_CHECKPOINT, map_location="cpu", weights_only=False)
        ranker.load_state_dict(payload["state_dict"])
        ranker.eval()
        ranker_sha = sha256_file(RANKER_CHECKPOINT)

    systems = {
        "U-S0": evaluate_selector(records, baseline_run["proposals_by_key"], "deterministic"),
        "U-S1": evaluate_selector(records, selected_run["proposals_by_key"], "deterministic"),
        "U-S2": evaluate_selector(records, selected_run["proposals_by_key"], "ranker", ranker, "cpu"),
    }
    comparison = {
        "_doc": (
            "Task 6U sections 17-18. Exactly three reference systems on the untouched RefValUnique: "
            "U-S0 (U-C0 proposals + frozen Task 6Q deterministic area selector), U-S1 (selected U-C* "
            "proposals + the same deterministic selector), U-S2 (selected U-C* proposals + "
            "ProposalSetRanker v0.1). Failure buckets use the Task 6Q definitions."
        ),
        "task": "6U", "stage": "H-selector-comparison",
        "pack": {"path": str(PACK), "records": len(records), "sha256": sha256_file(PACK)},
        "systems": {
            name: {key: value for key, value in system.items() if key != "rows"}
            for name, system in systems.items()
        },
        "buckets": {name: systems[name]["overall"]["buckets"] for name in systems},
        "bucket_order": list(BUCKET_ORDER),
        "ranker_checkpoint": {"path": str(RANKER_CHECKPOINT), "sha256": ranker_sha,
                              "exists": RANKER_CHECKPOINT.is_file()},
        "ranker_improved_selection": bool(
            systems["U-S2"]["overall"]["selected_reference_miou"] is not None
            and systems["U-S1"]["overall"]["selected_reference_miou"] is not None
            and systems["U-S2"]["overall"]["selected_reference_miou"]
            >= systems["U-S1"]["overall"]["selected_reference_miou"] + 0.05
            and systems["U-S2"]["overall"]["buckets"]["REFERENCE_SELECTION_WRONG"]
            <= 0.70 * systems["U-S1"]["overall"]["buckets"]["REFERENCE_SELECTION_WRONG"]
            and systems["U-S2"]["overall"]["abstention_rate"] <= 0.10),
        "deltas": {
            "U-S1_minus_U-S0_miou":
                (systems["U-S1"]["overall"]["selected_reference_miou"] or 0.0)
                - (systems["U-S0"]["overall"]["selected_reference_miou"] or 0.0),
            "U-S2_minus_U-S1_miou":
                (systems["U-S2"]["overall"]["selected_reference_miou"] or 0.0)
                - (systems["U-S1"]["overall"]["selected_reference_miou"] or 0.0),
            "U-S1_minus_U-S0_selection_wrong":
                systems["U-S0"]["overall"]["buckets"]["REFERENCE_SELECTION_WRONG"]
                - systems["U-S1"]["overall"]["buckets"]["REFERENCE_SELECTION_WRONG"],
            "U-S2_minus_U-S1_selection_wrong":
                systems["U-S1"]["overall"]["buckets"]["REFERENCE_SELECTION_WRONG"]
                - systems["U-S2"]["overall"]["buckets"]["REFERENCE_SELECTION_WRONG"],
        },
        "selectors_compared": 3,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_COMPARISON, comparison)

    for name in ("U-S0", "U-S1", "U-S2"):
        overall = systems[name]["overall"]
        print(f"[6u.ref] {name}: mIoU {overall['selected_reference_miou']:.4f} "
              f"Pr@0.5 {overall['precision_at_0_5']:.4f} centroid med "
              f"{overall['centroid_error_median']:.4f} abstain {overall['abstention_rate']:.4f} "
              f"selection_wrong {overall['buckets']['REFERENCE_SELECTION_WRONG']} "
              f"not_covered {overall['buckets']['REFERENCE_NOT_COVERED_IOU50']}", flush=True)
    print(f"[6u.ref] coverage improved {audit['candidate_coverage_improved']} | ranker improved "
          f"{comparison['ranker_improved_selection']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
