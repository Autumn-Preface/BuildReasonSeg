"""Task 6U Parts D-E — U0 candidate-coverage selection on the train-only U-Calib200 split.

Runs the exact four declared proposal inference configurations (U-C0..U-C3) of the frozen Task 6M.1
YOLO26m-seg checkpoint over the 200 U-Calib200 references and reports all-proposal and eligible coverage
at 0.25/0.50/0.75 (overall / largest / smallest), proposals and eligible proposals per tile, no-proposal
and no-eligible counts, inference wall time per tile and peak VRAM.

Then freezes exactly one configuration with the section 10 priority order:

1. highest smallest eligible coverage@0.50; 2. then overall; 3. then largest; 4. then fewer mean eligible
proposals per tile; 5. then lower imgsz; 6. then higher conf; 7. then lower config id.

RefValUnique and MiniVal240 never participate in the choice. The selected config may not change once
`evaluation/task6u_selected_proposal_config.json` exists.

    python scripts/task6u_candidate_coverage.py
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
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible  # noqa: E402
from scripts.task6u_common import (  # noqa: E402
    CONFIG_ORDER,
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    config_manifest,
    coverage_for_records,
    group_records_by_tile,
    percentile,
    proposals_for_tile,
    record_image_path,
    selection_priority_key,
    sha256_file,
    write_proposal_cache_manifest,
)

OUT_COVERAGE = EVAL / "task6u_calibration_candidate_coverage.json"
OUT_SELECTED = EVAL / "task6u_selected_proposal_config.json"
SPLIT = EVAL / "task6u_reference_train_split.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()

    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256:
        write_json(OUT_COVERAGE, {"_doc": "Task 6U sections 7-10.", "task": "6U",
                                  "verdict": "PROPOSAL_CHECKPOINT_UNAVAILABLE",
                                  "checkpoint": str(PROPOSAL_CHECKPOINT),
                                  "sha256": checkpoint_sha})
        print("[6u.calib] STOP PROPOSAL_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    calib_records = split["u_calib200"]["records"]
    grouped = group_records_by_tile(calib_records)
    print(f"[6u.calib] U-Calib200 {len(calib_records)} references over {len(grouped)} tiles; "
          f"configs {list(CONFIG_ORDER)}", flush=True)

    from ultralytics import YOLO

    model = YOLO(str(PROPOSAL_CHECKPOINT))
    proposal_cache: dict[tuple, list] = {}
    cache_entries: dict[str, dict] = {}
    config_reports: dict[str, dict] = {}

    for config_id in CONFIG_ORDER:
        config = CONFIGS[config_id]
        tile_seconds, tile_peaks = [], []
        proposals_per_tile, eligible_per_tile = [], []
        per_tile_times = {}
        for tile_id, tile_records in grouped.items():
            image_path = record_image_path(tile_records[0])
            proposals, run = proposals_for_tile(model, config, tile_id, image_path, args.device,
                                                use_cache=not args.no_cache)
            if not run["cached"]:
                tile_seconds.append(run["seconds"])
                if run["peak_vram_gb"] is not None:
                    tile_peaks.append(run["peak_vram_gb"])
            per_tile_times[tile_id] = run["seconds"]
            for record in tile_records:
                family = str(record["reference_family"])
                proposal_cache[(tile_id, int(record["reference_source_feature_id"]), family)] = \
                    proposals
            proposals_per_tile.append(len(proposals))
        # eligible counts are per reference record (family dependent)
        for record in calib_records:
            family = str(record["reference_family"])
            proposals = proposal_cache[(str(record["tile_id"]),
                                        int(record["reference_source_feature_id"]), family)]
            eligible_per_tile.append(sum(1 for proposal in proposals
                                         if is_eligible(proposal, family)))
        coverage = coverage_for_records(calib_records, proposal_cache)
        config_reports[config_id] = {
            **config_manifest(config, checkpoint_sha),
            "coverage": {key: value for key, value in coverage.items() if key != "records"},
            "proposals_per_tile": {
                "records": len(proposals_per_tile),
                "mean": float(np.mean(proposals_per_tile)) if proposals_per_tile else None,
                "median": float(np.median(proposals_per_tile)) if proposals_per_tile else None,
                "p90": percentile([float(value) for value in proposals_per_tile], 0.90),
            },
            "eligible_proposals_per_record": {
                "records": len(eligible_per_tile),
                "mean": float(np.mean(eligible_per_tile)) if eligible_per_tile else None,
                "median": float(np.median(eligible_per_tile)) if eligible_per_tile else None,
                "p90": percentile([float(value) for value in eligible_per_tile], 0.90),
            },
            "no_proposal_tiles": sum(1 for value in proposals_per_tile if value == 0),
            "no_eligible_records": sum(1 for value in eligible_per_tile if value == 0),
            "inference": {
                "tiles": len(grouped),
                "mean_seconds_per_tile": float(np.mean(tile_seconds)) if tile_seconds else None,
                "median_seconds_per_tile": float(np.median(tile_seconds)) if tile_seconds else None,
                "total_seconds": float(np.sum(tile_seconds)) if tile_seconds else None,
                "peak_vram_gb": max(tile_peaks) if tile_peaks else None,
                "cached_tiles": sum(1 for value in per_tile_times.values() if value is None),
            },
        }
        cache_entries[config_id] = per_tile_times
        smallest = config_reports[config_id]["coverage"]["smallest"]
        overall = config_reports[config_id]["coverage"]["overall"]
        print(f"[6u.calib] {config_id} imgsz {config['imgsz']} conf {config['conf']}: smallest "
              f"eligible@0.50 {smallest['eligible_coverage@0.50']:.4f} | overall "
              f"{overall['eligible_coverage@0.50']:.4f} | largest "
              f"{config_reports[config_id]['coverage']['largest']['eligible_coverage@0.50']:.4f} | "
              f"eligible/record "
              f"{config_reports[config_id]['eligible_proposals_per_record']['mean']:.2f}",
              flush=True)

    ranked = sorted(CONFIG_ORDER,
                    key=lambda config_id: selection_priority_key(config_reports[config_id]),
                    reverse=True)
    selected_id = ranked[0]
    selected = config_reports[selected_id]
    payload = {
        "_doc": (
            "Task 6U sections 7-10. U0 candidate-coverage selection: the exact four declared proposal "
            "inference configurations of the frozen Task 6M.1 YOLO26m-seg checkpoint evaluated on the "
            "train-only U-Calib200 split (100 largest + 100 smallest references), with the unchanged "
            "Task 6Q family eligibility. GT reference masks are used for scoring only."
        ),
        "task": "6U", "stage": "E-candidate-coverage",
        "checkpoint": {"path": str(PROPOSAL_CHECKPOINT), "sha256": checkpoint_sha,
                       "expected_sha256": PROPOSAL_CHECKPOINT_SHA256, "retrained": False},
        "calibration_split": {"path": str(SPLIT), "records": len(calib_records),
                              "tiles": len(grouped),
                              "by_family": dict(Counter(record["reference_family"]
                                                        for record in calib_records))},
        "eligibility": "Task 6Q family rules unchanged (largest: no border + extent<=0.20; smallest: "
                       "same + area>=150)",
        "configurations": config_reports,
        "ranking": ranked,
        "runtime_seconds": round(time.time() - started, 1),
        "test_split_used": False,
        "refval_used_for_selection": False,
        "minival_used_for_selection": False,
    }
    write_json(OUT_COVERAGE, payload)
    write_proposal_cache_manifest(CONFIG_ORDER, cache_entries)

    if OUT_SELECTED.is_file():
        existing = json.loads(OUT_SELECTED.read_text(encoding="utf-8"))
        if existing.get("selected_config") != selected_id:
            print(f"[6u.calib] the frozen selected config is {existing.get('selected_config')}; "
                  f"keeping it (it may not change)", flush=True)
            selected_id = existing["selected_config"]
            selected = config_reports[selected_id]
        else:
            print(f"[6u.calib] selected config already frozen: {selected_id}", flush=True)

    selection_payload = {
        "_doc": (
            "Task 6U section 10. The single frozen proposal inference configuration, selected on the "
            "train-only U-Calib200 split by the exact section 10 priority order and immutable "
            "afterwards."
        ),
        "task": "6U", "stage": "E-selected-config",
        "selected_config": selected_id,
        "selected": {key: selected[key] for key in (
            "id", "label", "imgsz", "conf", "max_det", "nms", "tta", "tiling",
            "checkpoint", "checkpoint_sha256", "source_image_size", "masks_restored_to_source")},
        "priority_rule": [
            "highest smallest eligible coverage@0.50",
            "then highest overall eligible coverage@0.50",
            "then highest largest eligible coverage@0.50",
            "then lower mean eligible proposals per record",
            "then lower imgsz",
            "then higher conf",
            "then lower config id",
        ],
        "priority_key_of_selected": list(selection_priority_key(selected)),
        "ranking": ranked,
        "candidates_considered": list(CONFIG_ORDER),
        "frozen": True,
        "immutable_after_creation": True,
        "selection_inputs": {"u_calib200": True, "refval_unique": False, "minival240": False,
                             "test_split": False},
        "selected_metrics": {
            "smallest_eligible_coverage@0.50":
                selected["coverage"]["smallest"]["eligible_coverage@0.50"],
            "overall_eligible_coverage@0.50":
                selected["coverage"]["overall"]["eligible_coverage@0.50"],
            "largest_eligible_coverage@0.50":
                selected["coverage"]["largest"]["eligible_coverage@0.50"],
            "mean_eligible_proposals_per_record":
                selected["eligible_proposals_per_record"]["mean"],
        },
        "evidence": {"calibration": str(OUT_COVERAGE)},
        "test_split_used": False,
    }
    write_json(OUT_SELECTED, selection_payload)
    print(f"[6u.calib] ranking {ranked} -> selected {selected_id} "
          f"(smallest eligible@0.50 "
          f"{selected['coverage']['smallest']['eligible_coverage@0.50']:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
