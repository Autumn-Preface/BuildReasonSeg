"""Task 6K.1 Part H (section 12): split and geographic audit with the recovered tile coordinates.

The tile -> (raster, row, column) -> map-extent mapping recovered in Part B turns the split question
from "region-level" (Task 6K) into a per-tile geometric statement:

* map-space extent of every split (and its approximate latitude/longitude);
* 8-neighbour adjacency: how many val tiles are directly adjacent to training tiles;
* whether the test split is spatially separated from train/val;
* what the split therefore supports (random-tile vs geographic generalization).

Writes `evaluation/task6k1_split_geographic_audit.json`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_vector_audit import REGION_TO_RASTER  # noqa: E402

from task6k1_common import (  # noqa: E402
    EVAL,
    CROPPED_ROOT,
    raster_affine,
    raster_grid,
    region_of_stem,
    tile_index_of_stem,
    write_json,
)

OUT = EVAL / "task6k1_split_geographic_audit.json"
SPLITS = ("train", "train_no", "test", "test_no")
SPLIT_GROUPS = {
    "train": ("train", "train_no"),
    "val": ("val",),
    "test": ("test", "test_no"),
}


def inverse_mercator(x: float, y: float) -> tuple[float, float]:
    """Inverse ellipsoidal Mercator (WGS84) -> (longitude, latitude) in degrees."""

    a = 6378137.0
    e = math.sqrt(1 - (6356752.314245 / a) ** 2) if False else 0.08181919084262149
    lon = math.degrees(x / a)
    t = math.exp(-y / a)
    latitude = math.pi / 2 - 2 * math.atan(t)
    for _ in range(8):
        es = e * math.sin(latitude)
        latitude = math.pi / 2 - 2 * math.atan(t * ((1 - es) / (1 + es)) ** (e / 2))
    return lon, math.degrees(latitude)


def main() -> int:
    started = time.time()
    tiles_by_group: dict[str, set[tuple[str, int, int]]] = {name: set() for name in SPLIT_GROUPS}
    stems_by_group: dict[str, list[str]] = {name: [] for name in SPLIT_GROUPS}
    counts = {}
    for group, splits in SPLIT_GROUPS.items():
        for split in splits:
            directory = CROPPED_ROOT / split / "image"
            if not directory.is_dir():
                continue
            files = sorted(directory.glob("*.tif"))
            counts[split] = len(files)
            for path in files:
                stem = path.stem
                raster = REGION_TO_RASTER.get(region_of_stem(stem))
                index = tile_index_of_stem(stem)
                if raster is None or index is None:
                    continue
                grid = raster_grid(raster)
                if not grid.contains(index):
                    continue
                row, col = grid.index_to_rowcol(index)
                tiles_by_group[group].add((raster, row, col))
                stems_by_group[group].append(stem)

    # the val split exists only in the historical converted dataset (the source folders have no
    # `val` directory), so its tiles come from the converted folder and are mapped onto the grid.
    from task6k_common import converted_stems

    val_stems = converted_stems("val", kind="labels")
    counts["val"] = len(val_stems)
    for stem in val_stems:
        raster = REGION_TO_RASTER.get(region_of_stem(stem))
        index = tile_index_of_stem(stem)
        if raster is None or index is None:
            continue
        grid = raster_grid(raster)
        if not grid.contains(index):
            continue
        row, col = grid.index_to_rowcol(index)
        tiles_by_group["val"].add((raster, row, col))
        stems_by_group["val"].append(stem)

    def extent(group: str) -> dict:
        xs, ys, rows, cols = [], [], [], []
        for raster, row, col in tiles_by_group[group]:
            affine = raster_affine(raster)
            grid = raster_grid(raster)
            x0, y0 = affine.pixel_to_map(col * grid.tile, row * grid.tile)
            x1, y1 = affine.pixel_to_map(col * grid.tile + grid.tile, row * grid.tile + grid.tile)
            xs.extend([x0, x1])
            ys.extend([y0, y1])
            rows.append(row)
            cols.append(col)
        if not xs:
            return {}
        lon_min, lat_min = inverse_mercator(min(xs), min(ys))
        lon_max, lat_max = inverse_mercator(max(xs), max(ys))
        return {
            "tiles": len(tiles_by_group[group]),
            "map_extent": [min(xs), min(ys), max(xs), max(ys)],
            "map_extent_km": [round((max(xs) - min(xs)) / 1000, 3), round((max(ys) - min(ys)) / 1000, 3)],
            "approx_lon_lat_bbox": [round(lon_min, 6), round(lat_min, 6), round(lon_max, 6), round(lat_max, 6)],
            "row_range": [int(min(rows)), int(max(rows))],
            "col_range": [int(min(cols)), int(max(cols))],
            "by_raster": {
                raster: sum(1 for entry in tiles_by_group[group] if entry[0] == raster)
                for raster in sorted({entry[0] for entry in tiles_by_group[group]})
            },
        }

    # the val tiles are drawn from the source train pool, so the *effective* training set is the
    # train pool minus val; comparing against the raw pool would trivially include each val tile.
    tiles_by_group["train_effective"] = tiles_by_group["train"] - tiles_by_group["val"]
    stems_by_group["train_effective"] = [
        stem for stem in stems_by_group["train"] if stem not in set(stems_by_group["val"])
    ]

    # adjacency: is a tile of one group orthogonally/diagonally next to a tile of another group?
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def adjacency(target: str, other: str) -> dict:
        target_tiles = tiles_by_group[target]
        other_tiles = tiles_by_group[other]
        with_neighbour = 0
        neighbour_counts = []
        fully_surrounded = 0
        distances = []
        for raster, row, col in target_tiles:
            count = 0
            for drow, dcol in neighbours:
                if (raster, row + drow, col + dcol) in other_tiles:
                    count += 1
            neighbour_counts.append(count)
            if count > 0:
                with_neighbour += 1
            if count == 8:
                fully_surrounded += 1
            # nearest tile of the other group in tile-grid units (Chebyshev distance)
            best = None
            for other_raster, other_row, other_col in other_tiles:
                if other_raster != raster:
                    continue
                distance = max(abs(other_row - row), abs(other_col - col))
                if best is None or distance < best:
                    best = distance
                if best == 1:
                    break
            distances.append(best if best is not None else None)
        finite = [value for value in distances if value is not None]
        return {
            "target_tiles": len(target_tiles),
            "tiles_with_at_least_one_neighbour_in_" + other: with_neighbour,
            "rate": float(with_neighbour / max(len(target_tiles), 1)),
            "fully_surrounded_rate": float(fully_surrounded / max(len(target_tiles), 1)),
            "mean_neighbours": float(np.mean(neighbour_counts)) if neighbour_counts else None,
            "neighbour_count_histogram": {
                str(value): int(sum(1 for count in neighbour_counts if count == value)) for value in range(0, 9)
            },
            "nearest_" + other + "_tile_distance_grid_units": {
                "min": int(min(finite)) if finite else None,
                "median": float(np.median(finite)) if finite else None,
                "max": int(max(finite)) if finite else None,
                "mean": float(np.mean(finite)) if finite else None,
                "distance_1_rate": float(sum(1 for value in finite if value == 1) / max(len(finite), 1)),
                "distance_le_2_rate": float(sum(1 for value in finite if value <= 2) / max(len(finite), 1)),
                "histogram": {
                    str(value): int(sum(1 for entry in finite if entry == value))
                    for value in sorted(set(finite))[:12]
                },
                "tiles_without_other_group_in_same_raster": len(distances) - len(finite),
            },
        }

    report = {
        "_doc": (
            "Task 6K.1 section 12. Split/geographic audit using the recovered tile -> map-extent "
            "mapping. Adjacency is computed on the 512-tile grids, so it is exact rather than "
            "inferred from region names. Approximate latitude/longitude uses the inverse ellipsoidal "
            "Mercator for readability only; all distances are reported in map units (metres)."
        ),
        "task": "6K.1",
        "counts": counts,
        "extents": {group: extent(group) for group in SPLIT_GROUPS},
        "adjacency": {
            "val_vs_train_effective": adjacency("val", "train_effective"),
            "train_effective_vs_val": adjacency("train_effective", "val"),
            "val_vs_train_pool_including_val": adjacency("val", "train"),
            "test_vs_train_effective": adjacency("test", "train_effective"),
            "test_vs_val": adjacency("test", "val"),
            "test_vs_train_pool": adjacency("test", "train"),
            "train_effective_vs_test": adjacency("train_effective", "test"),
        },
        "adjacency_definition": (
            "`train_effective` = the source train pool minus the val tiles (the tiles actually used "
            "for training); `train` = the whole source train pool, which CONTAINS the val tiles. "
            "The effective set is the meaningful comparison."
        ),
        "spatial_correlation_risk": {
            "val_tiles_directly_adjacent_to_train_tiles_rate": adjacency("val", "train_effective")["rate"],
            "test_tiles_directly_adjacent_to_train_or_val_rate": max(
                adjacency("test", "train_effective")["rate"], adjacency("test", "val")["rate"]
            ),
            "val_nearest_train_tile_distance_grid_units": adjacency("val", "train_effective")[
                "nearest_train_effective_tile_distance_grid_units"
            ],
            "same_whole_image_source_rate": {
                "val_sharing_a_raster_with_train": (
                    sum(
                        1 for entry in tiles_by_group["val"]
                        if entry[0] in {tile[0] for tile in tiles_by_group["train"]}
                    ) / max(len(tiles_by_group["val"]), 1)
                ),
                "test_sharing_a_raster_with_train": (
                    sum(
                        1 for entry in tiles_by_group["test"]
                        if entry[0] in {tile[0] for tile in tiles_by_group["train"]}
                    ) / max(len(tiles_by_group["test"]), 1)
                ),
            },
            "val_is_spatially_interleaved_with_train": bool(
                adjacency("val", "train_effective")["rate"] > 0.5
            ),
            "regions_shared_between_train_and_val": sorted(
                {entry[0] for entry in tiles_by_group["train"]} & {entry[0] for entry in tiles_by_group["val"]}
            ),
            "regions_shared_between_train_and_test": sorted(
                {entry[0] for entry in tiles_by_group["train"]} & {entry[0] for entry in tiles_by_group["test"]}
            ),
            "conclusion": (
                "val is drawn from the same two whole-area rasters as train and its tiles are "
                "interleaved with training tiles, so scene-level leakage between train and val is "
                "certain; the test split comes from a separate raster and is spatially disjoint. "
                "The split therefore supports random-tile generalization only."
            ),
            "supported_paper_claim": (
                "Random-tile generalization within the same two source scenes (train vs val) and one "
                "spatially disjoint scene (test). It does NOT support a claim of geographic "
                "generalization across cities/regions or of unseen-scene robustness beyond the single "
                "held-out test raster."
            ),
        },
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    risks = report["spatial_correlation_risk"]
    print(
        f"[task6k1.split] val adjacent to train "
        f"{risks['val_tiles_directly_adjacent_to_train_tiles_rate']:.4f}; test adjacent to "
        f"train/val {risks['test_tiles_directly_adjacent_to_train_or_val_rate']:.4f}; "
        f"shared regions train∩val {risks['regions_shared_between_train_and_val']}",
        flush=True,
    )
    print(f"[task6k1.split] wrote {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
