"""Task 6K.1 Part D (section 8): build the canonical native-vector instance view per tile.

Read-only w.r.t. the archive; writes only into the gitignored BuildReasonSeg cache. For every tile
of the audit corpus the intersecting native features are clipped to the 512x512 window, rasterized
(holes subtracted) and stored with stable ids (= the .shp record order, since the DBF attributes are
degenerate), geometry, border-truncation and multipart flags.

Writes `evaluation/task6k1_vector_instance_stats.json` plus the gitignored per-tile cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from task6k1_common import (  # noqa: E402
    CACHE_DIR,
    EVAL,
    INSTANCE_DIR,
    clip_vector_to_tile,
    load_vector_corpus,
    mean_or_none,
    percentile_summary,
    raster_affine,
    raster_grid,
    save_tile_instances,
    tile_geometry,
    write_json,
)

OUT = EVAL / "task6k1_vector_instance_stats.json"


def count_distribution(values, boundaries) -> dict:
    counts: dict[str, int] = {}
    for index, low in enumerate(boundaries):
        high = boundaries[index + 1] if index + 1 < len(boundaries) else None
        label = f"{low}" if high is None else f"{low}-{high - 1}" if high - low > 1 else f"{low}"
        counts[label] = 0
    for value in values:
        for index, low in enumerate(boundaries):
            high = boundaries[index + 1] if index + 1 < len(boundaries) else None
            label = f"{low}" if high is None else f"{low}-{high - 1}" if high - low > 1 else f"{low}"
            if value >= low and (high is None or value < high):
                counts[label] += 1
                break
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    corpus = load_vector_corpus()
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

    # the audit corpus: the same 4,038 positive tiles used by every earlier task
    from task6k_common import aligned_tiles

    tiles = aligned_tiles()
    if args.limit:
        tiles = tiles[: int(args.limit)]
    print(f"[task6k1.instances] {len(tiles)} corpus tiles", flush=True)

    per_tile = []
    instance_counts = []
    border_instances = 0
    multipart_instances = 0
    clipped_features = 0
    total_instances = 0
    empty_tiles = 0
    all_ids = set()

    for position, tile in enumerate(tiles, start=1):
        geometry = tile_geometry(tile.stem)
        if geometry is None:
            per_tile.append({"stem": tile.stem, "split": tile.split, "skipped": "no_geometry"})
            continue
        instances = clip_vector_to_tile(corpus, geometry)
        save_tile_instances(instances)
        instance_counts.append(len(instances))
        total_instances += len(instances)
        border_instances += sum(1 for flag in instances.touches_border if flag)
        multipart_instances += sum(1 for flag in instances.multipart if flag)
        clipped_features += sum(1 for flag in instances.clipped if flag)
        empty_tiles += 1 if len(instances) == 0 else 0
        all_ids.update(instances.ids)
        per_tile.append(
            {
                "stem": tile.stem,
                "split": tile.split,
                "raster": geometry.raster,
                "row": geometry.row,
                "col": geometry.col,
                "vector_instances": len(instances),
                "pixels": int(sum(instances.areas_px)),
                "border_instances": int(sum(1 for flag in instances.touches_border if flag)),
                "multipart_instances": int(sum(1 for flag in instances.multipart if flag)),
                "clipped_features": int(sum(1 for flag in instances.clipped if flag)),
                "feature_ids": instances.ids,
            }
        )
        if position % 500 == 0:
            print(f"[task6k1.instances] {position}/{len(tiles)} ({time.time() - started:.0f}s)", flush=True)

    # global coverage: assign every feature centroid to its tile across the three full grids
    coverage = {}
    for region, raster in (("1", "train1"), ("2", "train2"), ("test", "test")):
        grid = raster_grid(raster)
        affine = raster_affine(raster)
        inside = 0
        for feature in corpus.features:
            cx = 0.5 * (feature.bbox_map[0] + feature.bbox_map[2])
            cy = 0.5 * (feature.bbox_map[1] + feature.bbox_map[3])
            col, row = affine.map_to_pixel(cx, cy)
            if 0 <= col < grid.columns * grid.tile and 0 <= row < grid.rows * grid.tile:
                inside += 1
        coverage[raster] = {
            "tile_grid_capacity": grid.capacity,
            "features_with_centroid_inside_grid": inside,
        }

    split_counts = {}
    for entry in per_tile:
        split = entry.get("split", "unknown")
        bucket = split_counts.setdefault(split, {"tiles": 0, "instances": 0, "empty_tiles": 0})
        bucket["tiles"] += 1
        if "vector_instances" in entry:
            bucket["instances"] += entry["vector_instances"]
            bucket["empty_tiles"] += 1 if entry["vector_instances"] == 0 else 0

    report = {
        "_doc": (
            "Task 6K.1 section 8. Canonical native-vector instance view for the audit corpus: each "
            "intersecting feature is clipped to the tile window and rasterized with its holes "
            "subtracted. Identity is the .shp record order (1-based) because the DBF attributes are "
            "degenerate. Masks live in the gitignored per-tile cache."
        ),
        "task": "6K.1",
        "corpus": {
            "tiles": len(per_tile),
            "tiles_with_geometry": sum(1 for entry in per_tile if "vector_instances" in entry),
            "tiles_with_zero_vector_instances": empty_tiles,
            "total_instances": total_instances,
            "distinct_features_touched": len(all_ids),
            "vector_features_total": len(corpus),
            "instance_count_distribution": count_distribution(instance_counts, [0, 1, 2, 3, 5, 10, 20]),
            "instances_per_tile": percentile_summary(instance_counts),
            "border_truncated_instances": border_instances,
            "border_truncation_rate": (border_instances / total_instances) if total_instances else None,
            "multipart_instances": multipart_instances,
            "features_clipped_by_tile_boundary": clipped_features,
            "clipped_feature_rate": (clipped_features / total_instances) if total_instances else None,
            "cache_dir": str(INSTANCE_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "by_split": split_counts,
        "global_coverage": coverage,
        "per_tile": per_tile,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6k1.instances] {total_instances} instances over {len(per_tile)} tiles "
        f"({empty_tiles} tiles with none); border {border_instances}; multipart {multipart_instances}; "
        f"distinct features touched {len(all_ids)}; wrote {OUT.name} in {time.time() - started:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
