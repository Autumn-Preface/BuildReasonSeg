"""Task 7G Parts C-D — build the train-only largest-reference selection dataset.

Source (section 6): **BuildSpatialReason v0.2 train split only**, every record whose canonical program contains
a `largest` reference **and** whose metadata provides the canonical `reference_source_feature_id`. Records
without an explicit reference id (the L1 `largest` program, whose target *is* the reference) and every
`smallest`-reference record are excluded, as are val/test records.

Deduplication (section 7) by the exact key `(split, tile_id, reference_source_feature_id)` gives
`G-RefTrainUniqueLargest`: one proposal-set example per unique canonical largest reference.

Per unique reference (section 8): frozen U-C1 proposals → exact `largest` eligibility → GT IoU of every eligible
proposal → the label is the maximum-GT-IoU candidate (tie higher YOLO confidence, then lower original index).
No eligible proposal is counted as `no_eligible`, best eligible IoU < 0.50 as `untrainable_not_covered`, and both
are excluded from the selector loss. GT is label construction only and is never an inference input.

Section 9 gate: selector-trainable unique references >= 300 and >= 50 unique train tiles available for the
internal holdout, otherwise STOP `SELECTOR_TRAIN_DATA_INSUFFICIENT`.

Writes `evaluation/task7g_training_dataset_manifest.json`; the large per-reference rows (18-D features + labels)
are cached under the gitignored `artifacts/task7g/selector_dataset/`.

    python scripts/task7g_build_selector_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task6q_reference_resolver import is_eligible  # noqa: E402
from buildreasonseg_mvp.task7g_largest_reference_selector import (  # noqa: E402
    FEATURE_DIM,
    SelectorDependencyUnavailable,
    oracle_best,
    proposal_feature_matrix,
    selector_report,
)
from scripts.task6n_train import MaskStore  # noqa: E402
from scripts.task6u_common import CONFIGS, PROPOSAL_CHECKPOINT, proposals_for_tile  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
CACHE_ROOT = REPO_ROOT / "artifacts" / "task7g" / "selector_dataset"
ROWS_PATH = CACHE_ROOT / "selector_rows.jsonl"
OUT = EVAL / "task7g_training_dataset_manifest.json"
YOLO_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
MIN_TRAINABLE = 300
MIN_HOLDOUT_TILES = 50
COVERAGE_MIN = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_path_of(record: dict) -> Path:
    from buildreasonseg_mvp.task6n_relation_decoder import WHU_SOURCE_ROOT

    relative = str(record["image_path"]).replace("/", "\\")
    return Path(WHU_SOURCE_ROOT) / relative


def tile_of(record: dict) -> str:
    native = record.get("native_vector") or {}
    return str(native.get("tile_id") or record.get("image_id"))


def reference_id_of(record: dict) -> int | None:
    native = record.get("native_vector") or {}
    references = native.get("references") or []
    if not references or "source_feature_id" not in references[0]:
        return None
    return int(references[0]["source_feature_id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    try:
        import scipy  # noqa: F401
    except ImportError:
        write_json(OUT, {"_doc": "Task 7G section 12.", "task": "7G",
                         "verdict": "SELECTOR_DEPENDENCY_UNAVAILABLE"})
        print("[7g.data] STOP SELECTOR_DEPENDENCY_UNAVAILABLE", flush=True)
        return 2

    train_rows = []
    with (DATA / "train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                train_rows.append(json.loads(line))
    candidates = [record for record in train_rows if "largest" in str(record["query_type"])]
    with_reference = [record for record in candidates if reference_id_of(record) is not None]
    without_reference = len(candidates) - len(with_reference)
    excluded_smallest = sum(1 for record in train_rows if "smallest" in str(record["query_type"]))

    unique: dict[tuple[str, str, int], dict] = {}
    duplicates = 0
    for record in with_reference:
        key = (str(record.get("split", "train")), tile_of(record), reference_id_of(record))
        if key in unique:
            duplicates += 1
            continue
        unique[key] = record
    per_program = Counter(str(record["query_type"]) for record in unique.values())

    from ultralytics import YOLO

    from buildreasonseg_mvp.task6n_relation_decoder import load_frozen_sam2_encoder  # noqa: F401

    model = YOLO(str(PROPOSAL_CHECKPOINT))
    masks = MaskStore()
    config = CONFIGS["U-C1"]
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"no_eligible": 0, "untrainable_not_covered": 0, "trainable": 0}
    by_tile: dict[str, list[dict]] = defaultdict(list)
    for key, record in sorted(unique.items()):
        split, tile_id, reference_id = key
        proposals, _run = proposals_for_tile(model, config, tile_id,
                                            image_path_of(record), args.yolo_device,
                                            use_cache=True)
        eligible = [proposal for proposal in proposals
                    if np.asarray(proposal.mask, dtype=bool).any()
                    and is_eligible(proposal, "largest")]
        gt_reference = np.asarray(masks.mask(tile_id, reference_id), dtype=bool)
        if not eligible:
            counts["no_eligible"] += 1
            by_tile[tile_id].append({"key": list(key), "state": "no_eligible",
                                     "programs": [str(record["query_type"])]})
            continue
        ious = []
        for proposal in eligible:
            mask = np.asarray(proposal.mask, dtype=bool)
            intersection = float((mask & gt_reference).sum())
            union = float((mask | gt_reference).sum())
            ious.append(intersection / max(union, 1e-6))
        best_iou = max(ious)
        if best_iou < COVERAGE_MIN:
            counts["untrainable_not_covered"] += 1
            by_tile[tile_id].append({"key": list(key), "state": "untrainable_not_covered",
                                     "programs": [str(record["query_type"])],
                                     "best_eligible_iou": best_iou})
            continue
        features = proposal_feature_matrix(eligible)
        target = oracle_best(eligible, gt_reference)
        deterministic = max(range(len(eligible)),
                            key=lambda position: (int(np.asarray(eligible[position].mask).sum()),
                                                  float(eligible[position].confidence),
                                                  -int(getattr(eligible[position], "index", position))))
        counts["trainable"] += 1
        row = {
            "split": split, "tile_id": tile_id, "reference_source_feature_id": reference_id,
            "program_ids": [str(record["query_type"])], "state": "trainable",
            "eligible_count": len(eligible), "best_eligible_iou": float(best_iou),
            "target_index": int(target),
            # GT-derived bookkeeping for labels/offline metrics only - never a selector feature
            "candidate_ious": [float(value) for value in ious],
            "candidate_statistics": [{"index": int(getattr(proposal, "index", position)),
                                      "area": int(np.asarray(proposal.mask).sum()),
                                      "confidence": float(proposal.confidence)}
                                     for position, proposal in enumerate(eligible)],
            "selected_iou_deterministic": float(ious[deterministic]),
            "features": features.tolist(),
        }
        rows.append(row)
        by_tile[tile_id].append({"key": list(key), "state": "trainable"})

    trainable_tiles = sorted({row["tile_id"] for row in rows})
    manifest = {
        "_doc": ("Task 7G sections 6-9. G-RefTrainUniqueLargest: the train-only deduplicated largest-reference "
                 "selection dataset. GT is used for label construction only (best eligible IoU >= 0.50) and is "
                 "never an inference input; no eligible or uncovered reference is excluded from the loss."),
        "task": "7G", "stage": "C-selector-dataset",
        "source": {"split": "train", "dataset": "BuildSpatialReason v0.2",
                   "train_rows": len(train_rows), "largest_reference_rows": len(candidates),
                   "rows_with_reference_id": len(with_reference),
                   "rows_without_reference_id": without_reference,
                   "excluded_smallest_reference_rows": excluded_smallest,
                   "unique_references": len(unique), "unique_tiles": len({key[1] for key in unique}),
                   "duplicate_rows_removed": duplicates,
                   "per_program": dict(per_program)},
        "proposal_generation": {"config": "U-C1", **config,
                                "checkpoint": str(PROPOSAL_CHECKPOINT),
                                "checkpoint_sha256": sha256_file(PROPOSAL_CHECKPOINT),
                                "eligibility": {"non_empty": True, "not_border_touching": True,
                                                "bbox_extent_ratio_max": 0.20,
                                                "area_min_rule": None}},
        "labels": {"coverage_min_iou": COVERAGE_MIN, "tie_break": ["higher GT IoU",
                                                                   "higher YOLO confidence",
                                                                   "lower original proposal index"],
                   "counts": counts, "trainable_tiles": len(trainable_tiles)},
        "selector": selector_report(),
        "rows_path": str(ROWS_PATH), "rows_gitignored": True,
        "gate": {"min_trainable": MIN_TRAINABLE, "measured_trainable": counts["trainable"],
                 "min_holdout_tiles": MIN_HOLDOUT_TILES,
                 "unique_train_tiles": len(trainable_tiles),
                 "passed": bool(counts["trainable"] >= MIN_TRAINABLE
                                and len(trainable_tiles) >= MIN_HOLDOUT_TILES)},
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    manifest["verdict"] = ("SELECTOR_TRAIN_DATA_READY" if manifest["gate"]["passed"]
                           else "SELECTOR_TRAIN_DATA_INSUFFICIENT")
    write_json(OUT, manifest)
    with ROWS_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[7g.data] train rows {len(train_rows)} | largest-reference rows {len(candidates)} | with "
          f"reference id {len(with_reference)} | unique references {len(unique)} on "
          f"{manifest['source']['unique_tiles']} tiles | no_eligible {counts['no_eligible']} uncovered "
          f"{counts['untrainable_not_covered']} trainable {counts['trainable']} | "
          f"{manifest['verdict']}", flush=True)
    return 0 if manifest["gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
