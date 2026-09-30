"""Task 6W Part D — build the proposal-quality training dataset (train-only, deduplicated by tile).

Tiles come **only** from the frozen Task 6U U-RankerTrain split (no U-Calib200 tile, no RefVal tile, no
MiniVal/PairedVal evaluation record, no test). U-C1 proposals are generated/read **once per unique tile**
so a proposal is never duplicated because several reference records share the tile.

Candidate population uses the common structural eligibility only (section 9): the proposal does not touch
the source-image border and `bbox_extent_ratio <= 0.20`. The smallest-family `area >= 150` rule is **not**
applied, because proposal quality is family-independent and family eligibility stays a later resolver rule.

Binary label (section 10): `q_gt = max IoU(proposal_mask, any native GT building instance on the tile)`,
`y_quality = 1 if q_gt >= 0.50 else 0`; `q_gt` is stored for analysis. GT is never a model input.

Deterministic tile split (section 14): seed 20260930, 80 % train / 20 % internal holdout by tile-id hash,
no tile in both. Writes `artifacts/task6w/quality_dataset/` (gitignored) and the tracked manifest
`evaluation/task6w_quality_dataset_manifest.json`.

    python scripts/task6w_build_quality_dataset.py --device 0
"""

from __future__ import annotations

import argparse
import hashlib
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
from buildreasonseg_mvp.task6n_relation_decoder import (  # noqa: E402
    FrozenFeatureStore,
    load_frozen_sam2_encoder,
)
from buildreasonseg_mvp.task6w_proposal_quality import (  # noqa: E402
    GEOMETRY_FEATURE_NAMES,
    QUALITY_THRESHOLD,
    VISUAL_DIM,
    proposal_quality_features,
)
from scripts.task6u_common import (  # noqa: E402
    CONFIGS,
    EVAL,
    PROPOSAL_CHECKPOINT,
    PROPOSAL_CHECKPOINT_SHA256,
    gt_masks,
    iou,
    proposals_for_tile,
    record_image_path,
    sha256_file,
)
from task6n_train import FEATURE_ROOT  # noqa: E402

OUT_MANIFEST = EVAL / "task6w_quality_dataset_manifest.json"
SPLIT = EVAL / "task6u_reference_train_split.json"
DATASET_ROOT = REPO_ROOT / "artifacts" / "task6w" / "quality_dataset"
SEED = 20260930
HOLDOUT_FRACTION = 0.20
EXTENT_MAX = 0.20


def tile_hash(tile_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{tile_id}".encode("utf-8")).hexdigest()


def common_eligible(proposal) -> bool:
    """Section 9 structural eligibility: no border touch and bbox extent <= 0.20 (no area floor)."""

    return (not proposal.touches_border) and proposal.bbox_extent_ratio <= EXTENT_MAX


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--torch-device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    checkpoint_sha = sha256_file(PROPOSAL_CHECKPOINT) if PROPOSAL_CHECKPOINT.is_file() else None
    if checkpoint_sha != PROPOSAL_CHECKPOINT_SHA256:
        write_json(OUT_MANIFEST, {"_doc": "Task 6W section 8.", "task": "6W",
                                  "verdict": "FROZEN_ASSET_UNAVAILABLE"})
        return 2

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ranker_records = split["u_rankertrain"].get("records")
    if ranker_records is None:
        from scripts.task6u_freeze_reference_split import unique_key

        packs = json.loads((REPO_ROOT / "artifacts" / "task6p" / "reference_packs"
                            / "ref_train_unique.json").read_text(encoding="utf-8"))["records"]
        calib_keys = {unique_key(record) for record in split["u_calib200"]["records"]}
        ranker_records = [record for record in packs if unique_key(record) not in calib_keys]

    def image_of(record) -> Path:
        path = Path(str(record.get("image_path")))
        return path if path.is_absolute() else REPO_ROOT / path

    tiles: dict[str, dict] = {}
    for record in ranker_records:
        tile_id = str(record["tile_id"])
        tiles.setdefault(tile_id, record)
    calib_tiles = {str(record["tile_id"]) for record in split["u_calib200"]["records"]}
    val_tiles = {str(record["tile_id"]) for record in json.loads(
        (REPO_ROOT / "artifacts" / "task6p" / "reference_packs" / "ref_val_unique.json")
        .read_text(encoding="utf-8"))["records"]}
    # section 8: no U-Calib200 tile and no RefVal tile may enter quality training. The Task 6U split is
    # reference-key based, so a tile can legitimately hold both a U-Calib200 and a U-RankerTrain
    # reference; such tiles are excluded here on purpose.
    excluded_calib = sorted(set(tiles) & calib_tiles)
    excluded_refval = sorted(set(tiles) & val_tiles)
    for tile_id in excluded_calib + excluded_refval:
        tiles.pop(tile_id, None)
    print(f"[6w.data] U-RankerTrain {len(ranker_records)} references; tiles after exclusion "
          f"{len(tiles)} (excluded {len(excluded_calib)} calib-overlapping, "
          f"{len(excluded_refval)} refval-overlapping)", flush=True)

    from ultralytics import YOLO

    yolo = YOLO(str(PROPOSAL_CHECKPOINT))
    encoder, _report = load_frozen_sam2_encoder(device=args.torch_device)
    store = FrozenFeatureStore(FEATURE_ROOT, encoder=encoder, device=args.torch_device)

    geometry_rows, visual_rows, label_rows, quality_rows, tile_rows = [], [], [], [], []
    invalid_small = 0
    ring_empty = 0
    per_tile_counts = {}
    for index, (tile_id, record) in enumerate(sorted(tiles.items()), start=1):
        image_path = image_of(record)
        proposals, _run = proposals_for_tile(yolo, CONFIGS["U-C1"], tile_id, image_path, args.device,
                                             use_cache=True)
        instances = gt_masks(tile_id)
        feature = store.get(tile_id, image_path).float().cpu().numpy()
        kept = 0
        for proposal in proposals:
            if not common_eligible(proposal):
                continue
            best = max((iou(proposal.mask, truth) for truth in instances.values()), default=0.0)
            bundle = proposal_quality_features(proposal, feature)
            if not bundle.valid:
                invalid_small += 1
                continue
            if bundle.ring_empty:
                ring_empty += 1
            geometry_rows.append(bundle.geometry)
            visual_rows.append(bundle.visual)
            quality_rows.append(float(best))
            label_rows.append(1.0 if best >= QUALITY_THRESHOLD else 0.0)
            tile_rows.append(tile_id)
            kept += 1
        per_tile_counts[tile_id] = kept
        if index % 100 == 0:
            print(f"[6w.data] {index}/{len(tiles)} tiles, {len(label_rows)} proposals", flush=True)

    geometry = np.asarray(geometry_rows, dtype=np.float32)
    visual = np.asarray(visual_rows, dtype=np.float32)
    labels = np.asarray(label_rows, dtype=np.float32)
    qualities = np.asarray(quality_rows, dtype=np.float32)

    unique_tiles = sorted(tiles)
    holdout_tiles = {tile_id for tile_id in unique_tiles
                     if int(tile_hash(tile_id)[:8], 16) / 0xFFFFFFFF < HOLDOUT_FRACTION}
    train_tiles = [tile_id for tile_id in unique_tiles if tile_id not in holdout_tiles]
    train_mask = np.asarray([tile_id not in holdout_tiles for tile_id in tile_rows])
    holdout_mask = ~train_mask

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_path = DATASET_ROOT / "quality_rows.npz"
    np.savez_compressed(dataset_path, geometry=geometry, visual=visual, labels=labels,
                        quality=qualities, tiles=np.asarray(tile_rows))
    histogram = Counter(f"{min(int(value * 10) / 10, 0.9):.1f}" for value in qualities)

    payload = {
        "_doc": (
            "Task 6W sections 8-10 and 14. Proposal-quality training dataset built from U-C1 proposals on "
            "the train-only U-RankerTrain tiles (deduplicated by tile id), using the common structural "
            "eligibility (no border touch, bbox extent <= 0.20; no smallest area floor). Labels are "
            "y_quality = 1 iff q_gt = max IoU with any native GT building instance on the tile >= 0.50; "
            "GT is training metadata only and never a model input. The 80/20 train/internal-holdout split "
            "is by tile-id hash with seed 20260930."
        ),
        "task": "6W", "stage": "D-quality-dataset", "seed": SEED,
        "checkpoint": {"path": str(PROPOSAL_CHECKPOINT), "sha256": checkpoint_sha,
                       "expected": PROPOSAL_CHECKPOINT_SHA256, "retrained": False},
        "config": {key: CONFIGS["U-C1"][key]
                   for key in ("id", "imgsz", "conf", "max_det", "nms", "tta", "tiling")},
        "universe": {
            "source_split": "U-RankerTrain (Task 6U)",
            "references": len(ranker_records),
            "unique_tiles": len(tiles),
            "calib_tile_overlap": len(set(tiles) & calib_tiles),
            "refval_tile_overlap": len(set(tiles) & val_tiles),
            "excluded_calib_overlapping_tiles": len(excluded_calib),
            "excluded_refval_overlapping_tiles": len(excluded_refval),
            "excluded_calib_tile_ids_sample": excluded_calib[:5],
            "no_calib_tile": len(set(tiles) & calib_tiles) == 0,
            "no_refval_tile": len(set(tiles) & val_tiles) == 0,
            "test_split_used": False,
        },
        "eligibility": {"no_border_touch": True, "bbox_extent_ratio_max": EXTENT_MAX,
                        "smallest_area_floor_applied": False,
                        "rationale": "proposal quality is family-independent; family eligibility remains "
                                     "a later resolver rule"},
        "labels": {
            "quality_threshold": QUALITY_THRESHOLD,
            "definition": "q_gt = max IoU(proposal_mask, any native GT building instance on the tile)",
            "binary": "y_quality = 1 if q_gt >= 0.50 else 0",
            "proposals": int(len(labels)),
            "positives": int((labels >= 0.5).sum()),
            "negatives": int((labels < 0.5).sum()),
            "positive_rate": float((labels >= 0.5).mean()) if len(labels) else None,
            "q_gt_histogram": {key: histogram[key] for key in sorted(histogram)},
            "q_gt_mean": float(qualities.mean()) if len(qualities) else None,
        },
        "features": {
            "geometry_dim": int(geometry.shape[1]) if len(geometry) else 0,
            "geometry_names": list(GEOMETRY_FEATURE_NAMES),
            "visual_dim": int(visual.shape[1]) if len(visual) else 0,
            "visual_source": "frozen SAM2 256x64x64 inside-mean + one-cell-ring-mean",
            "invalid_small_excluded": invalid_small,
            "ring_empty_using_zeros": ring_empty,
        },
        "split": {
            "method": "tile-id hash, seed 20260930, 80/20",
            "train_tiles": len(train_tiles), "holdout_tiles": len(holdout_tiles),
            "train_proposals": int(train_mask.sum()), "holdout_proposals": int(holdout_mask.sum()),
            "train_positives": int((labels[train_mask] >= 0.5).sum()),
            "train_negatives": int((labels[train_mask] < 0.5).sum()),
            "holdout_positives": int((labels[holdout_mask] >= 0.5).sum()),
            "holdout_negatives": int((labels[holdout_mask] < 0.5).sum()),
            "tile_overlap": len(set(train_tiles) & holdout_tiles),
            "no_overlap": len(set(train_tiles) & holdout_tiles) == 0,
            "train_tile_hash": hashlib.sha256("".join(sorted(train_tiles)).encode()).hexdigest(),
            "holdout_tile_hash": hashlib.sha256("".join(sorted(holdout_tiles)).encode()).hexdigest(),
        },
        "dataset": {"path": str(dataset_path), "bytes": dataset_path.stat().st_size,
                    "gitignored": True,
                    "arrays": {"geometry": list(geometry.shape), "visual": list(visual.shape),
                               "labels": list(labels.shape)}},
        "per_tile_proposals_mean": float(np.mean(list(per_tile_counts.values())))
        if per_tile_counts else None,
        "verdict": "QUALITY_DATASET_READY" if len(labels) > 0 else "QUALITY_DATASET_EMPTY",
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_MANIFEST, payload)
    print(f"[6w.data] proposals {len(labels)} ({int((labels >= 0.5).sum())} positive / "
          f"{int((labels < 0.5).sum())} negative); train {int(train_mask.sum())} / holdout "
          f"{int(holdout_mask.sum())}; invalid_small {invalid_small}; ring_empty {ring_empty}",
          flush=True)
    return 0 if len(labels) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
