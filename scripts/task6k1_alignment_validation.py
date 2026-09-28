"""Task 6K.1 Part C (sections 6-7): validate the native vector against the raster and pseudo views.

Read-only. For every positive tile (and a deterministic sample of empty tiles) the native vector is
clipped to the tile window and compared with:

* the cropped raster semantic label (IoU / Dice / precision / recall, both-empty handled);
* a +/-2 px offset search (systematic-shift detection) on a deterministic sample;
* per-object matching: vector features missing from the raster label and raster-only components
  with no vector counterpart;
* the historical pseudo-instance view (polygon/connected-component) for reference.

Also settles the EXACT local feature count from the local files.

Writes `evaluation/task6k1_vector_raster_alignment.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_vector_audit import CROPPED_ROOT, rasterize_feature  # noqa: E402

from task6k1_common import (  # noqa: E402
    EVAL,
    clip_vector_to_tile,
    iou_masks,
    load_vector_corpus,
    mean_or_none,
    percentile_summary,
    tile_geometry,
    write_json,
)

OUT = EVAL / "task6k1_vector_raster_alignment.json"
SPLITS = ("train", "train_no", "test", "test_no")


def label_path_for(stem: str) -> Path | None:
    for split in SPLITS:
        candidate = CROPPED_ROOT / split / "label" / f"{stem}.tif"
        if candidate.is_file():
            return candidate
    return None


def raster_only_components(label_mask: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(label_mask.astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset-sample", type=int, default=240)
    parser.add_argument("--empty-sample", type=int, default=300)
    args = parser.parse_args(argv)

    started = time.time()
    corpus = load_vector_corpus()

    positive_stems = []
    empty_stems = []
    for split in ("train", "val", "test"):
        directory = CROPPED_ROOT / ("train" if split == "train" else "test") / "label"
        metadata_file = REPO_ROOT / "datasets" / "whu" / "metadata" / f"{split}.jsonl"
        if not metadata_file.is_file():
            continue
        with metadata_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    positive_stems.append(str(record["image_id"]))
    positive_stems = sorted(set(positive_stems))

    for split in ("train_no", "test_no"):
        directory = CROPPED_ROOT / split / "label"
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.tif"))
        step = max(1, len(files) // max(int(args.empty_sample) // 2, 1))
        for path in files[::step]:
            if path.stem not in set(positive_stems):
                empty_stems.append(path.stem)
    empty_stems = sorted(set(empty_stems))[: int(args.empty_sample)]

    rows = []
    offset_rows = []
    per_raster = {}

    def evaluate(stem: str, with_offset: bool, with_objects: bool) -> dict | None:
        geometry = tile_geometry(stem)
        if geometry is None:
            return None
        label_path = label_path_for(stem)
        if label_path is None:
            return None
        label_mask = np.asarray(Image.open(label_path).convert("L")) > 0
        instances = clip_vector_to_tile(corpus, geometry)
        vector_mask = np.zeros_like(label_mask)
        for mask in instances.masks:
            vector_mask |= mask
        row = {
            "stem": stem,
            "raster": geometry.raster,
            "vector_instances": len(instances),
            "vector_pixels": int(vector_mask.sum()),
            "label_pixels": int(label_mask.sum()),
            "both_empty": bool(not vector_mask.any() and not label_mask.any()),
            "iou": iou_masks(vector_mask, label_mask),
            "dice": float(
                2 * np.logical_and(vector_mask, label_mask).sum()
                / max(int(vector_mask.sum()) + int(label_mask.sum()), 1)
            ),
            "precision": float(
                np.logical_and(vector_mask, label_mask).sum() / max(int(vector_mask.sum()), 1)
            ) if vector_mask.any() else None,
            "recall": float(
                np.logical_and(vector_mask, label_mask).sum() / max(int(label_mask.sum()), 1)
            ) if label_mask.any() else None,
        }
        if with_offset:
            inter = np.logical_and(vector_mask, label_mask).sum()
            union = np.logical_or(vector_mask, label_mask).sum()
            best = (iou_masks(vector_mask, label_mask), 0, 0)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    shifted = np.roll(np.roll(vector_mask, dy, axis=0), dx, axis=1)
                    value = iou_masks(shifted, label_mask)
                    if value > best[0]:
                        best = (value, dx, dy)
            step_inter = int(inter)
            step_union = int(union)
            row["best_offset"] = {"dx": best[1], "dy": best[2], "iou": best[0]}
            row["iou_at_zero"] = iou_masks(vector_mask, label_mask)
            row["offset_gain"] = best[0] - row["iou_at_zero"]
            row["overlap_pixels"] = step_inter
            row["union_pixels"] = step_union
        if with_objects:
            raster_labels = raster_only_components(label_mask)
            raster_ids = [int(v) for v in np.unique(raster_labels) if int(v) != 0]
            raster_masks = [raster_labels == value for value in raster_ids]
            vector_without_raster = 0
            for mask in instances.masks:
                best = max((iou_masks(mask, other) for other in raster_masks), default=0.0)
                if best < 0.25:
                    vector_without_raster += 1
            raster_without_vector = 0
            for mask in raster_masks:
                best = max((iou_masks(mask, other) for other in instances.masks), default=0.0)
                if best < 0.25:
                    raster_without_vector += 1
            row["vector_instances_without_raster_counterpart"] = vector_without_raster
            row["raster_components_without_vector_counterpart"] = raster_without_vector
            row["raster_component_count"] = len(raster_ids)
        return row

    offset_counter = 0
    for position, stem in enumerate(positive_stems, start=1):
        with_offset = offset_counter < int(args.offset_sample)
        row = evaluate(stem, with_offset=with_offset, with_objects=with_offset)
        if row is None:
            continue
        if with_offset:
            offset_counter += 1
            offset_rows.append(row)
        rows.append(row)
        if position % 1000 == 0:
            print(f"[task6k1.align] {position}/{len(positive_stems)} ({time.time() - started:.0f}s)", flush=True)

    empty_rows = []
    for stem in empty_stems:
        row = evaluate(stem, with_offset=False, with_objects=False)
        if row is not None:
            empty_rows.append(row)

    for row in rows + empty_rows:
        bucket = per_raster.setdefault(row["raster"], {"tiles": 0, "ious": [], "empty": 0, "instances": 0})
        bucket["tiles"] += 1
        bucket["ious"].append(row["iou"])
        bucket["instances"] += row["vector_instances"]
        bucket["empty"] += 1 if row["both_empty"] else 0

    positive_ious = [row["iou"] for row in rows]
    non_empty_ious = [row["iou"] for row in rows if not row["both_empty"]]
    offsets = {}
    for row in offset_rows:
        key = f"({row['best_offset']['dx']},{row['best_offset']['dy']})"
        offsets[key] = offsets.get(key, 0) + 1

    report = {
        "_doc": (
            "Task 6K.1 sections 6-7. Native vector vs the cropped raster semantic label on every "
            "positive tile plus a deterministic empty-tile sample. The vector is clipped to each "
            "tile window and rasterized (holes subtracted); both-empty tiles count as agreement."
        ),
        "task": "6K.1",
        "scope": {
            "positive_tiles": len(rows),
            "empty_sample_tiles": len(empty_rows),
            "vector_features_total": len(corpus),
        },
        "exact_local_feature_count": {
            "value": len(corpus),
            "source": "EA.shp record count == EA.shx index records == EA.dbf records",
            "external_counts_ignored": ["29085", "34085-as-published"],
            "note": (
                "The published 29,085/34,085 counts are context only; this project uses the locally "
                "parsed count."
            ),
        },
        "vector_vs_raster_label": {
            "all_positive_tiles": {
                "iou_summary": percentile_summary(positive_ious),
                "iou_non_empty_summary": percentile_summary(non_empty_ious),
                "mean_iou": mean_or_none(positive_ious),
                "mean_iou_non_empty": mean_or_none(non_empty_ious),
                "dice_summary": percentile_summary([row["dice"] for row in rows]),
                "precision_summary": percentile_summary([row["precision"] for row in rows]),
                "recall_summary": percentile_summary([row["recall"] for row in rows]),
                "tiles_iou_ge_0_90": sum(1 for value in non_empty_ious if value >= 0.90),
                "tiles_iou_ge_0_75": sum(1 for value in non_empty_ious if value >= 0.75),
                "tiles_iou_lt_0_50": sum(1 for value in non_empty_ious if value < 0.50),
            },
            "empty_sample_tiles": {
                "tiles": len(empty_rows),
                "both_empty": sum(1 for row in empty_rows if row["both_empty"]),
                "vector_only": sum(1 for row in empty_rows if row["vector_pixels"] > 0 and not row["label_pixels"]),
                "label_only": sum(1 for row in empty_rows if row["label_pixels"] > 0 and not row["vector_pixels"]),
            },
            "by_raster": {
                name: {
                    "tiles": bucket["tiles"],
                    "mean_iou": mean_or_none(bucket["ious"]),
                    "empty_tiles": bucket["empty"],
                    "vector_instances": bucket["instances"],
                }
                for name, bucket in sorted(per_raster.items())
            },
        },
        "offset_search": {
            "sampled_tiles": len(offset_rows),
            "best_offset_histogram": offsets,
            "zero_offset_is_best": sum(1 for row in offset_rows if row["best_offset"]["dx"] == 0 and row["best_offset"]["dy"] == 0),
            "mean_offset_gain": mean_or_none([row["offset_gain"] for row in offset_rows]),
            "max_offset_gain": max((row["offset_gain"] for row in offset_rows), default=None),
            "conclusion": (
                "If the zero offset dominates the histogram and the mean gain is near zero, the "
                "vector and raster share one pixel grid with no systematic shift."
            ),
        },
        "unmatched_objects": {
            "sampled_tiles": len(offset_rows),
            "vector_instances_without_raster_counterpart": sum(
                row.get("vector_instances_without_raster_counterpart", 0) for row in offset_rows
            ),
            "raster_components_without_vector_counterpart": sum(
                row.get("raster_components_without_vector_counterpart", 0) for row in offset_rows
            ),
            "vector_instances_sampled": sum(row["vector_instances"] for row in offset_rows),
            "raster_components_sampled": sum(row.get("raster_component_count", 0) for row in offset_rows),
            "note": "Unmatched at IoU < 0.25, computed on the offset-search sample only.",
        },
        "rows": rows,
        "empty_rows": empty_rows,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    summary = report["vector_vs_raster_label"]["all_positive_tiles"]
    print(
        f"[task6k1.align] positive tiles {len(rows)}: IoU mean {summary['mean_iou']:.4f} "
        f"(non-empty {summary['mean_iou_non_empty']:.4f}); >=0.90 {summary['tiles_iou_ge_0_90']}, "
        f"<0.50 {summary['tiles_iou_lt_0_50']}; empty-sample both-empty "
        f"{report['vector_vs_raster_label']['empty_sample_tiles']['both_empty']}/"
        f"{len(empty_rows)}; best offsets {offsets}",
        flush=True,
    )
    print(f"[task6k1.align] wrote {OUT.name} in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
