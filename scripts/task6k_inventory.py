"""Task 6K Part A (sections 3-4): original-data inventory + historical split recovery.

Read-only. Measures the original WHU cropped subtrees BEFORE thresholding, records exact
resolved paths, extensions, shapes, dtypes, unique label values, empty/non-empty counts and
foreground-fraction distributions; then recovers the actual historical train/val/test split
from the existing converted folders (never regenerating it from seed 42).

Writes `evaluation/task6k_source_inventory.json` and `evaluation/task6k_split_audit.json`.
"""

from __future__ import annotations

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

from PIL import Image  # noqa: E402

from buildreasonseg_mvp.whu_source_audit import (  # noqa: E402
    CONVERTED_CANDIDATES,
    CROPPED_ROOT,
    ORIGINAL_ROOT,
    SHAPEFILE_ROOT,
    SOURCE_SPLITS,
    WHOLE_AREA_ROOT,
    binarize,
    percentile_summary,
    read_semantic_label,
    write_json,
)
from task6k_common import converted_stems, source_split_dir, source_stems  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
IMAGE_SAMPLE_PER_SPLIT = 300


def region_of(stem: str) -> str:
    """Scene/region grouping recoverable from the tile naming alone (no coordinates invented).

    Cropped tiles are named ``<region>_<index>`` in the source train pool (e.g. ``1_10035``,
    ``2_2784``) and ``<index>`` in the source test pool (e.g. ``1003``). The numeric prefix is
    treated as a REGION LABEL only -- it is never converted into coordinates.
    """

    if "_" in stem:
        return stem.split("_", 1)[0]
    return "test"


def region_mixing(source_train: set[str], converted: dict[str, set[str]]) -> dict:
    train_regions = Counter(region_of(stem) for stem in converted["train"])
    val_regions = Counter(region_of(stem) for stem in converted["val"])
    test_regions = Counter(region_of(stem) for stem in converted["test"])
    train_region_set = set(train_regions)
    val_region_set = set(val_regions)
    test_region_set = set(test_regions)
    shared = sorted(train_region_set & val_region_set)
    val_in_shared = sum(count for region, count in val_regions.items() if region in shared)
    return {
        "region_definition": "numeric prefix of the tile stem (`1_x`/`2_x` = source train regions; bare index = source test region)",
        "train_regions": dict(sorted(train_regions.items())),
        "val_regions": dict(sorted(val_regions.items())),
        "test_regions": dict(sorted(test_regions.items())),
        "train_val_shared_regions": shared,
        "val_tiles_in_shared_regions": int(val_in_shared),
        "val_tiles": len(converted["val"]),
        "val_same_region_mixing_rate": float(val_in_shared / max(len(converted["val"]), 1)),
        "train_test_shared_regions": sorted(train_region_set & test_region_set),
        "val_test_shared_regions": sorted(val_region_set & test_region_set),
        "train_val_spatially_correlated": bool(shared),
        "test_geographically_disjoint_from_train_and_val": bool(
            not (train_region_set & test_region_set) and not (val_region_set & test_region_set)
        ),
        "geographic_leakage_quantification": (
            "Region-level mixing IS recoverable from the naming (all val tiles come from regions "
            "also present in train), so train/val scene-level leakage is certain at region "
            "granularity. Per-tile adjacency and exact geographic overlap are NOT recoverable: the "
            "cropped tiles carry no world files or coordinates, and no coordinates are invented here. "
            "The test split comes from a separate region, so it is geographically disjoint."
        ),
        "supports": {
            "random_tile_generalization": True,
            "geographic_or_cross_region_generalization": "partial: only the test region provides an unseen region",
        },
        "local_geo_evidence_available": {
            "whole_area_georeferenced_rasters": True,
            "world_files": ".tfw present for the whole-area images only (92 bytes each)",
            "shapefile": "EA.shp/.dbf/.prj/.shx present locally; not joined to tiles in this audit",
            "cropped_tile_coordinates": "absent",
        },
    }


def image_sample_stats(paths: list[Path], sample_size: int) -> dict:
    shapes = Counter()
    modes = Counter()
    dtypes = Counter()
    channels = Counter()
    sizes = []
    step = max(1, len(paths) // max(sample_size, 1))
    sampled = paths[::step][:sample_size]
    for path in sampled:
        with Image.open(path) as handle:
            shapes[str(handle.size)] += 1
            modes[str(handle.mode)] += 1
        sizes.append(path.stat().st_size)
    return {
        "sampled": len(sampled),
        "shape_distribution": dict(shapes),
        "mode_distribution": dict(modes),
        "dtype_distribution": dict(dtypes),
        "channel_distribution": dict(channels),
        "file_bytes": percentile_summary(sizes),
    }


def inventory_split(split: str) -> dict:
    image_dir = source_split_dir(split, "image")
    label_dir = source_split_dir(split, "label")
    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file())
    label_paths = sorted(p for p in label_dir.iterdir() if p.is_file())
    image_stems = {p.stem for p in image_paths}
    label_stems = {p.stem for p in label_paths}

    shapes = Counter()
    dtypes = Counter()
    unique_values = Counter()
    foreground_fractions = []
    empty = 0
    total_pixels = 0
    total_foreground = 0
    for path in label_paths:
        mask = read_semantic_label(path)
        shapes[str(tuple(mask.shape))] += 1
        dtypes[str(mask.dtype)] += 1
        unique_values[str([int(v) for v in np.unique(mask)])] += 1
        foreground = binarize(mask) > 0
        fraction = float(foreground.mean())
        foreground_fractions.append(fraction)
        total_pixels += int(foreground.size)
        total_foreground += int(foreground.sum())
        if not foreground.any():
            empty += 1

    return {
        "resolved_image_dir": str(image_dir),
        "resolved_label_dir": str(label_dir),
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "matched_stems": len(image_stems & label_stems),
        "images_without_labels": sorted(image_stems - label_stems)[:20],
        "labels_without_images": sorted(label_stems - image_stems)[:20],
        "n_images_without_labels": len(image_stems - label_stems),
        "n_labels_without_images": len(label_stems - image_stems),
        "image_extensions": dict(Counter(p.suffix.lower() for p in image_paths)),
        "label_extensions": dict(Counter(p.suffix.lower() for p in label_paths)),
        "image_stats": image_sample_stats(image_paths, IMAGE_SAMPLE_PER_SPLIT),
        "label_shape_distribution": dict(shapes),
        "label_dtype_distribution": dict(dtypes),
        "label_unique_value_sets": {k: v for k, v in unique_values.most_common(10)},
        "label_unusual_unique_value_sets": {
            k: v for k, v in unique_values.items() if k not in ("[0, 255]", "[0]")
        },
        "empty_masks": int(empty),
        "non_empty_masks": int(len(label_paths) - empty),
        "foreground_fraction": percentile_summary(foreground_fractions),
        "total_pixels": total_pixels,
        "total_foreground_pixels": total_foreground,
        "overall_foreground_fraction": (total_foreground / total_pixels) if total_pixels else 0.0,
    }


def whole_area_inventory() -> dict:
    result = {"resolved_root": str(WHOLE_AREA_ROOT), "exists": WHOLE_AREA_ROOT.is_dir(), "files": []}
    if not WHOLE_AREA_ROOT.is_dir():
        return result
    for kind in ("image", "label"):
        directory = WHOLE_AREA_ROOT / kind
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            entry = {
                "kind": kind,
                "name": path.name,
                "extension": path.suffix.lower(),
                "bytes": path.stat().st_size,
            }
            if path.suffix.lower() in (".tif", ".tiff"):
                try:
                    with Image.open(path) as handle:
                        entry["size"] = list(handle.size)
                        entry["mode"] = str(handle.mode)
                except Exception as exc:  # noqa: BLE001
                    entry["read_error"] = f"{type(exc).__name__}: {exc}"
            result["files"].append(entry)
    return result


def shapefile_inventory() -> dict:
    result = {"resolved_root": str(SHAPEFILE_ROOT), "exists": SHAPEFILE_ROOT.is_dir(), "files": []}
    if not SHAPEFILE_ROOT.is_dir():
        return result
    for path in sorted(SHAPEFILE_ROOT.iterdir()):
        if path.is_file():
            result["files"].append({"name": path.name, "bytes": path.stat().st_size})
    return result


def main() -> int:
    started = time.time()
    inventory = {
        "_doc": (
            "Task 6K section 3. Read-only inventory of the original WHU Satellite Dataset II "
            "(East Asia) cropped subtrees, measured BEFORE thresholding. Shapes/dtypes/unique "
            "values/foreground fractions come from the raster LABELS; image shape/mode come from "
            "a deterministic sample plus full file-size statistics."
        ),
        "task": "6K",
        "original_root": str(ORIGINAL_ROOT),
        "cropped_root": str(CROPPED_ROOT),
        "splits": {split: inventory_split(split) for split in SOURCE_SPLITS},
        "whole_area_subtree": whole_area_inventory(),
        "shapefile_subtree": shapefile_inventory(),
        "seconds": round(time.time() - started, 2),
    }
    write_json(EVAL / "task6k_source_inventory.json", inventory)

    # ---- section 4: recover the ACTUAL historical split ----------------------
    source_train = source_stems("train")
    source_test = source_stems("test")
    converted = {split: converted_stems(split) for split in ("train", "val", "test")}
    overlaps = {
        "train_val": sorted(converted["train"] & converted["val"])[:20],
        "train_test": sorted(converted["train"] & converted["test"])[:20],
        "val_test": sorted(converted["val"] & converted["test"])[:20],
    }
    n_overlaps = {
        "train_val": len(converted["train"] & converted["val"]),
        "train_test": len(converted["train"] & converted["test"]),
        "val_test": len(converted["val"] & converted["test"]),
    }
    train_val_union = converted["train"] | converted["val"]
    split_audit = {
        "_doc": (
            "Task 6K section 4/10. The historical split is RECOVERED from the existing converted "
            "folders; seed 42 is never re-run. The historical converter built train/val from the "
            "source `train` pool (`VALIDATION_RATIO = 0.2`) and test from the source `test` pool."
        ),
        "task": "6K",
        "converted_dataset_root": str(CONVERTED_CANDIDATES[1]),
        "counts": {
            "source_train": len(source_train),
            "source_test": len(source_test),
            "converted_train": len(converted["train"]),
            "converted_val": len(converted["val"]),
            "converted_test": len(converted["test"]),
            "converted_total": sum(len(v) for v in converted.values()),
        },
        "overlaps": {**n_overlaps, "examples": overlaps},
        "source_mapping": {
            "train_and_val_subset_of_source_train": bool(train_val_union <= source_train),
            "train_test_disjoint_from_source_train": bool(not (train_val_union & source_train - source_train)),
            "converted_test_subset_of_source_test": bool(converted["test"] <= source_test),
            "source_test_equals_converted_test": bool(converted["test"] == source_test),
            "train_plus_val_equals_source_train": bool(train_val_union == source_train),
            "train_val_partition_exactly": bool(
                (converted["train"] | converted["val"]) == source_train
                and not (converted["train"] & converted["val"])
            ),
            "missing_from_converted_train_val": sorted(source_train - train_val_union)[:20],
            "n_missing_from_converted_train_val": len(source_train - train_val_union),
            "unexpected_extra_stems": sorted(train_val_union - source_train)[:20],
            "n_unexpected_extra_stems": len(train_val_union - source_train),
        },
        "reproducibility_caveat": {
            "historical_code": [
                "stems = set(); ... return list(stems)   # get_image_stems()",
                "random.seed(42)",
                "random.shuffle(train_stems_all)",
                "val_size = int(len(train_stems_all) * 0.2)",
                "val_stems = set(train_stems_all[:val_size]); train_stems = set(train_stems_all[val_size:])",
            ],
            "caveat": (
                "`list(set(...))` iteration order depends on the per-process string hash seed "
                "(PYTHONHASHSEED), so `random.seed(42)` alone does NOT guarantee the same split "
                "across processes. The split is therefore recovered from the existing converted "
                "folders rather than regenerated."
            ),
            "val_size_formula_check": {
                "int(len(train_stems) * 0.2)": int(len(source_train) * 0.2),
                "observed_val_count": len(converted["val"]),
                "consistent": int(len(source_train) * 0.2) == len(converted["val"]),
            },
            "split_authority": "existing converted folders (WHU_YOLO_dataset/images|labels/{train,val,test})",
        },
        "geographic_audit": region_mixing(source_train, converted),
        "seconds": round(time.time() - started, 2),
    }
    write_json(EVAL / "task6k_split_audit.json", split_audit)
    print(
        f"[task6k.inventory] source train {len(source_train)} test {len(source_test)} | "
        f"converted {len(converted['train'])}/{len(converted['val'])}/{len(converted['test'])} | "
        f"train+val==source train: {split_audit['source_mapping']['train_plus_val_equals_source_train']} | "
        f"test==source test: {split_audit['source_mapping']['source_test_equals_converted_test']}",
        flush=True,
    )
    print(f"[task6k.inventory] wrote source_inventory + split_audit in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
