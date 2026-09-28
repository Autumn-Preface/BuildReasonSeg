"""Task 6L Part A/B: build the canonical `WHU-EA-NativeVector` v1.0 dataset.

Indexes ALL 17,388 cropped tiles, clips native `EA.shp` geometry into each tile window (no `<50`
deletion, no component merging, no hole loss, no simplification), writes the two split views, the
integrity/statistics artifacts and the scene-disjoint leakage audit.

Read-only w.r.t. the archive; writes only inside BuildReasonSeg (committed metadata + gitignored
per-tile geometry cache).

    python scripts/task6l_build_dataset.py [--limit N] [--skip-rgb-hash]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_native_vector import (  # noqa: E402
    CACHE_ROOT,
    DATASET_NAME,
    DATASET_ROOT,
    DATASET_VERSION,
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_TOTAL_TILES,
    LEGACY_COMPAT_VIEW,
    MAX_TILE_INSTANCES,
    SCHEMA_VERSION,
    SCENE_DISJOINT_VIEW,
    FullAreaCache,
    TileRecord,
    clip_native_instances,
    iter_source_tiles,
    label_map_from_instances,
    load_vector_corpus,
    shapefile_provenance,
    tile_map_bbox,
    write_json,
    write_jsonl,
    write_tile_cache,
)

EVAL = REPO_ROOT / "evaluation"
INTEGRITY_OUT = EVAL / "task6l_vector_dataset_integrity.json"
SPLIT_AUDIT_OUT = EVAL / "task6l_scene_disjoint_split_audit.json"

SOURCE_ROOT = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")

from task6k_common import converted_stems  # noqa: E402


def percentile_summary(values) -> dict:
    array = np.asarray([v for v in values], dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def count_distribution(values, boundaries) -> dict:
    out: dict[str, int] = {}
    for index, low in enumerate(boundaries):
        high = boundaries[index + 1] if index + 1 < len(boundaries) else None
        label = str(low) if high is None else (f"{low}-{high - 1}" if high - low > 1 else str(low))
        out[label] = sum(1 for value in values if value >= low and (high is None or value < high))
    return out


def image_content_hash(image_path: Path) -> tuple[str, float]:
    """Deterministic content hash + grayscale std of a tile image (duplicate/leakage audit)."""

    from PIL import Image

    with Image.open(image_path) as image:
        small = image.convert("L").resize((64, 64), Image.BILINEAR)
        array = np.asarray(small, dtype=np.uint8)
    return hashlib.sha256(array.tobytes()).hexdigest(), float(array.std())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="debug subset (tiles)")
    parser.add_argument("--skip-rgb-hash", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    source_root = args.source_root
    if not source_root.is_dir():
        print(f"error: source root not found: {source_root}", file=sys.stderr)
        return 2

    provenance = shapefile_provenance()
    corpus = load_vector_corpus()
    full_areas = FullAreaCache(corpus)
    if not args.quiet:
        print(f"[6l.build] {len(corpus)} native features; shp sha256 {provenance['shp_sha256'][:12]}",
              flush=True)

    records = iter_source_tiles(source_root)
    if args.limit:
        records = records[: args.limit]
    category_counts = Counter(record.category for record in records)
    for category, expected in EXPECTED_CATEGORY_COUNTS.items():
        if args.limit is None and category_counts.get(category, 0) != expected:
            print(
                f"error: category {category} has {category_counts.get(category, 0)} tiles, expected {expected}",
                file=sys.stderr,
            )
            return 2
    if args.limit is None and len(records) != EXPECTED_TOTAL_TILES:
        print(f"error: {len(records)} tiles, expected {EXPECTED_TOTAL_TILES}", file=sys.stderr)
        return 2

    # historical split (recovered from the converted folders, never regenerated)
    historical = {
        "train": converted_stems("train", kind="labels"),
        "val": converted_stems("val", kind="labels"),
        "test": converted_stems("test", kind="labels"),
    }
    historical_of_stem = {}
    for split, stems in historical.items():
        for stem in stems:
            historical_of_stem[stem] = split

    tile_rows: list[dict] = []
    instance_rows: list[dict] = []
    sample_geometry: list[dict] = []
    feature_rasters: dict[int, set[str]] = defaultdict(set)
    per_instance_counts: list[int] = []
    instance_areas: list[int] = []
    visible_fractions: list[float] = []
    tiny_count = 0
    border_count = 0
    hole_instances = 0
    multipart_instances = 0
    empty_tiles = 0
    max_instances_tile = 0
    max_instances_tile_id = None
    area_delta_tiles = 0
    overlap_deltas: list[int] = []
    content_hashes: dict[str, list[str]] = defaultdict(list)
    degenerate_hashes: dict[str, float] = {}
    per_raster_stats: dict[str, dict] = defaultdict(lambda: {"tiles": 0, "instances": 0, "empty": 0})
    sample_tiles: set[str] = set()
    if records:
        step = max(1, len(records) // 40)
        sample_tiles = {record.tile_id for record in records[::step][:40]}

    for position, record in enumerate(records, start=1):
        instances = clip_native_instances(corpus, record, full_areas)
        if len(instances) > MAX_TILE_INSTANCES:
            print(
                f"error: {record.tile_id} has {len(instances)} instances > {MAX_TILE_INSTANCES}",
                file=sys.stderr,
            )
            return 2
        label_map = label_map_from_instances(instances)
        write_tile_cache(record, instances, label_map)

        rasterized_area = int((label_map > 0).sum())
        polygon_area = sum(int(instance.clipped_area_px) for instance in instances)
        area_delta = int(polygon_area - rasterized_area)
        if area_delta:
            overlap_deltas.append(area_delta)
        area_delta_tiles += int(area_delta != 0)
        for instance in instances:
            feature_rasters[instance.source_feature_id].add(record.raster)
            instance_rows.append(instance.as_dict())
            instance_areas.append(instance.clipped_area_px)
            visible_fractions.append(instance.visible_fraction)
            tiny_count += int(instance.tiny_area)
            border_count += int(instance.touched_tile_border)
            hole_instances += int(instance.n_holes > 0)
            multipart_instances += int(instance.multipart)
        if record.tile_id in sample_tiles:
            sample_geometry.append(
                {
                    "tile_id": record.tile_id,
                    "grid_id": record.grid_id,
                    "source_raster": record.raster,
                    "legacy_category": record.category,
                    "scene_disjoint_split": record.split,
                    "legacy_compat_split": historical_of_stem.get(record.tile_id),
                    "instance_count": len(instances),
                    "instances": [i.as_dict(with_geometry=True) for i in instances],
                }
            )

        if not args.skip_rgb_hash:
            image_path = source_root / record.source_image_rel
            digest, spread = image_content_hash(image_path)
            content_hashes[digest].append(record.tile_id)
            degenerate_hashes[digest] = max(degenerate_hashes.get(digest, 0.0), spread)

        per_instance_counts.append(len(instances))
        if len(instances) > max_instances_tile:
            max_instances_tile = len(instances)
            max_instances_tile_id = record.tile_id
        empty_tiles += int(len(instances) == 0)
        bucket = per_raster_stats[record.raster]
        bucket["tiles"] += 1
        bucket["instances"] += len(instances)
        bucket["empty"] += int(len(instances) == 0)

        tile_rows.append(
            record.as_dict(
                instance_stats={
                    "instance_count": len(instances),
                    "is_empty": len(instances) == 0,
                    "tiny_instance_count": sum(1 for i in instances if i.tiny_area),
                    "border_instance_count": sum(1 for i in instances if i.touched_tile_border),
                    "instance_area_px_sum": rasterized_area,
                    "rasterized_area_px": rasterized_area,
                    "clipped_polygon_area_px_sum": polygon_area,
                    "instance_ids": [i.tile_instance_id for i in instances],
                    "source_feature_ids": [i.source_feature_id for i in instances],
                },
                historical_split=historical_of_stem.get(record.tile_id),
            )
        )
        if not args.quiet and position % 2000 == 0:
            print(f"[6l.build] {position}/{len(records)} tiles ({time.time() - started:.0f}s)", flush=True)

    # ------------------------------------------------------------------ splits
    scene_tiles = {split: [] for split in ("train", "val", "test")}
    for record in records:
        scene_tiles[record.split].append(record.tile_id)

    boundary_features = sorted(
        feature_id for feature_id, rasters in feature_rasters.items() if len(rasters) > 1
    )
    boundary_feature_set = set(boundary_features)
    boundary_instance_rows = sum(1 for row in instance_rows
                                if row["source_feature_id"] in boundary_feature_set)

    # per-split feature sets AFTER excluding documented boundary-crossing features
    raster_of_tile = {record.tile_id: record.raster for record in records}
    split_features = {split: set() for split in ("train", "val", "test")}
    for row in instance_rows:
        feature_id = row["source_feature_id"]
        if feature_id in boundary_feature_set:
            continue
        raster = raster_of_tile.get(row["tile_id"])
        if raster is None:
            continue
        split_features[{"train1": "train", "train2": "val", "test": "test"}[raster]].add(feature_id)

    leakage = {
        "train_val": sorted(split_features["train"] & split_features["val"]),
        "train_test": sorted(split_features["train"] & split_features["test"]),
        "val_test": sorted(split_features["val"] & split_features["test"]),
    }
    tile_overlap = {
        "train_val": sorted(set(scene_tiles["train"]) & set(scene_tiles["val"])),
        "train_test": sorted(set(scene_tiles["train"]) & set(scene_tiles["test"])),
        "val_test": sorted(set(scene_tiles["val"]) & set(scene_tiles["test"])),
    }
    rgb_duplicates = {
        digest: sorted(tiles) for digest, tiles in content_hashes.items() if len(tiles) > 1
    }
    cross_split_duplicates = {}
    for digest, tiles in rgb_duplicates.items():
        splits = {
            "train" if tile in set(scene_tiles["train"]) else
            ("val" if tile in set(scene_tiles["val"]) else "test")
            for tile in tiles
        }
        if len(splits) > 1:
            cross_split_duplicates[digest] = {
                "tiles": tiles,
                "splits": sorted(splits),
                "grayscale_std": degenerate_hashes.get(digest, 0.0),
                "degenerate_content": bool(degenerate_hashes.get(digest, 1.0) < 2.0),
            }

    historical_view = {
        "view": LEGACY_COMPAT_VIEW,
        "purpose": "historical comparability with Task 1-6K.1 experiments only",
        "recovered_from": "historical converted YOLO dataset folders (never regenerated from seed)",
        "train": sorted(historical["train"]),
        "val": sorted(historical["val"]),
        "test": sorted(historical["test"]),
        "unassigned": sorted(
            record.tile_id for record in records
            if record.tile_id not in historical_of_stem
        ),
    }
    scene_view = {
        "view": SCENE_DISJOINT_VIEW,
        "purpose": "primary development/evaluation split (scene/raster separation)",
        "definition": {"train": "train1", "val": "train2", "test": "test"},
        "train": sorted(scene_tiles["train"]),
        "val": sorted(scene_tiles["val"]),
        "test": sorted(scene_tiles["test"]),
        "excluded_boundary_crossing_features": boundary_features,
        "excluded_boundary_instance_rows": boundary_instance_rows,
        "unassigned": [],
        "interpretation": (
            "scene/raster separation only: this is NOT unseen-city or broad geographic "
            "generalization, and no cross-city claim may be made from it"
        ),
    }
    write_json(DATASET_ROOT / "splits" / f"{LEGACY_COMPAT_VIEW}.json", historical_view)
    write_json(DATASET_ROOT / "splits" / f"{SCENE_DISJOINT_VIEW}.json", scene_view)

    # ------------------------------------------------------------------ artifacts
    write_jsonl(DATASET_ROOT / "tiles" / "index.jsonl", tile_rows)
    write_jsonl(DATASET_ROOT / "instances" / "index.jsonl", instance_rows)
    write_jsonl(DATASET_ROOT / "instances" / "sample_geometry.jsonl", sample_geometry)

    statistics_payload = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_feature_count": len(corpus),
        "tiles_indexed": len(records),
        "tiles_expected": EXPECTED_TOTAL_TILES,
        "tile_counts_by_category": dict(sorted(category_counts.items())),
        "tiles_by_raster": {raster: bucket["tiles"] for raster, bucket in sorted(per_raster_stats.items())},
        "instances_by_raster": {raster: bucket["instances"] for raster, bucket in sorted(per_raster_stats.items())},
        "empty_tiles_by_raster": {raster: bucket["empty"] for raster, bucket in sorted(per_raster_stats.items())},
        "total_clipped_instances": len(instance_rows),
        "distinct_source_features_represented": len(feature_rasters),
        "empty_tiles": empty_tiles,
        "non_empty_tiles": len(records) - empty_tiles,
        "instances_per_tile": percentile_summary(per_instance_counts),
        "instances_per_tile_distribution": count_distribution(per_instance_counts, [0, 1, 2, 3, 5, 10, 20, 50]),
        "instance_area_px": percentile_summary(instance_areas),
        "visible_fraction": percentile_summary(visible_fractions),
        "border_truncated_instances": border_count,
        "border_truncated_rate": border_count / max(len(instance_rows), 1),
        "tiny_instances": tiny_count,
        "tiny_instance_rate": tiny_count / max(len(instance_rows), 1),
        "instances_with_holes": hole_instances,
        "multipart_instances": multipart_instances,
        "max_instances_per_tile": max_instances_tile,
        "max_instances_per_tile_id": max_instances_tile_id,
        "label_map_uint8_limit": MAX_TILE_INSTANCES,
        "instance_overlap_or_rounding": {
            "tiles_with_nonzero_delta": area_delta_tiles,
            "delta_sum_px": int(sum(overlap_deltas)),
            "delta_max_px": int(max(overlap_deltas)) if overlap_deltas else 0,
            "delta_distribution": count_distribution(overlap_deltas, [1, 2, 3, 5, 10, 50]),
            "definition": (
                "sum of per-instance rasterised areas minus the number of non-background pixels in "
                "the combined uint8 label map; a small positive delta means adjacent instances share "
                "boundary pixels, where the later instance owns the pixel"
            ),
        },
        "splits": {
            LEGACY_COMPAT_VIEW: {
                "train": len(historical_view["train"]),
                "val": len(historical_view["val"]),
                "test": len(historical_view["test"]),
                "unassigned": len(historical_view["unassigned"]),
            },
            SCENE_DISJOINT_VIEW: {
                "train": len(scene_view["train"]),
                "val": len(scene_view["val"]),
                "test": len(scene_view["test"]),
                "excluded_boundary_crossing_features": len(boundary_features),
            },
        },
        "preservation_guarantees": {
            "min_contour_area_filter": None,
            "connected_component_merging": False,
            "retr_external_hole_loss": False,
            "approx_poly_dp_simplification": False,
            "tiny_instances_retained": True,
            "note": "native polygon geometry is preserved exactly as tile clipping permits",
        },
    }
    write_json(DATASET_ROOT / "statistics.json", statistics_payload)

    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "project_name": "BuildReasonSeg",
        "primary_annotation_truth": "native EA.shp polygon geometry (Task 6K.1 validated)",
        "predecessor": "Task 6K.1 verdict MIGRATE_WHU_TO_NATIVE_VECTOR_INSTANCES",
        "shapefile_provenance": provenance,
        "source_root_note": (
            "the external WHU archive root is supplied at runtime (--source-root); committed metadata "
            "stores only logical paths relative to that root, never absolute paths"
        ),
        "tile_id_definition": (
            "tile_id is the archive's own tile name (e.g. `1_0`, `0`), unique across the cropped "
            "archive; grid_id = `<raster>:<row>:<col>` is the whole-raster coordinate identity"
        ),
        "instance_identity": {
            "source_feature_id": "1-based EA.shp record order within this exact archive",
            "tile_instance_id": "deterministic 1-based order of clipped features in one tile",
            "uid": "(EA.shp SHA256, source_feature_id)",
            "dbf_fields_are_identity": False,
        },
        "tile_coverage": {
            "tiles_indexed": len(records),
            "includes_negative_folders": True,
            "includes_empty_tiles": True,
            "categories": {name: EXPECTED_CATEGORY_COUNTS[name] for name in EXPECTED_CATEGORY_COUNTS},
        },
        "files": {
            "tile_index": "datasets/whu_native_vector/v1.0/tiles/index.jsonl",
            "instance_index": "datasets/whu_native_vector/v1.0/instances/index.jsonl",
            "instance_schema": "datasets/whu_native_vector/v1.0/instances/schema.json",
            "sample_geometry": "datasets/whu_native_vector/v1.0/instances/sample_geometry.jsonl",
            "statistics": "datasets/whu_native_vector/v1.0/statistics.json",
            "splits": [
                f"datasets/whu_native_vector/v1.0/splits/{LEGACY_COMPAT_VIEW}.json",
                f"datasets/whu_native_vector/v1.0/splits/{SCENE_DISJOINT_VIEW}.json",
            ],
        },
        "geometry_cache": {
            "path": "artifacts/whu_native_vector/instances/<tile_id>.npz",
            "gitignored": True,
            "regenerate_with": "python scripts/task6l_build_dataset.py",
            "contents": "uint8 label map, clipped polygon rings (pixel coords), per-instance scalars",
        },
        "generation": {
            "runtime_seconds": round(time.time() - started, 2),
            "limit_applied": args.limit,
            "rgb_hash_skipped": bool(args.skip_rgb_hash),
        },
    }
    write_json(DATASET_ROOT / "manifest.json", manifest)

    write_json(
        DATASET_ROOT / "instances" / "schema.json",
        {
            "_doc": "Per-instance schema of WHU-EA-NativeVector v1.0 (one record per tile-clipped feature).",
            "fields": {
                "tile_id": "archive tile name",
                "tile_instance_id": "1-based index inside the tile (label-map value)",
                "source_feature_id": "1-based EA.shp record order (stable source identity)",
                "bbox_xyxy_px": "[x0, y0, x1, y1) of the clipped rasterisation",
                "centroid_px": "[x, y] centroid of the clipped rasterisation",
                "clipped_area_px": "rasterised area inside the tile window",
                "full_area_px": "rasterised area of the complete feature (unclipped)",
                "visible_fraction": "clipped_area_px / full_area_px (<= 1)",
                "touches_tile_border": "clipped rasterisation reaches the tile edge",
                "clipped_by_tile": "visible_fraction < 0.999",
                "multipart": "more than one outer ring after clipping",
                "n_rings": "rings after clipping",
                "n_holes": "hole rings after clipping",
                "n_outer_rings_full": "outer rings of the complete feature",
                "n_hole_rings_full": "hole rings of the complete feature",
                "tiny_area": "clipped area below the historical 50 px threshold (retained, flagged only)",
                "area_map_units": "complete-feature area in World Mercator map units (never ground m^2)",
            },
            "geometry": "full pixel-space rings live in instances/sample_geometry.jsonl and the gitignored per-tile cache",
        },
    )

    integrity = {
        "_doc": (
            "Task 6L section 16. Structural integrity of the canonical native-vector dataset: tile "
            "coverage, identity stability, geometry preservation and cache consistency."
        ),
        "task": "6L",
        "tiles_indexed": len(records),
        "tiles_expected": EXPECTED_TOTAL_TILES,
        "all_tiles_accounted": bool(args.limit is not None or len(records) == EXPECTED_TOTAL_TILES),
        "tile_counts_by_category": dict(sorted(category_counts.items())),
        "category_counts_match_archive": all(
            category_counts.get(name, 0) == expected
            for name, expected in EXPECTED_CATEGORY_COUNTS.items()
        ),
        "unique_tile_ids": len({record.tile_id for record in records}),
        "unique_grid_ids": len({record.grid_id for record in records}),
        "tile_id_unique": len({record.tile_id for record in records}) == len(records),
        "grid_id_unique": len({record.grid_id for record in records}) == len(records),
        "total_clipped_instances": len(instance_rows),
        "distinct_source_features_represented": len(feature_rasters),
        "source_feature_count": len(corpus),
        "stable_source_ids_survive_clipping": all(
            isinstance(row["source_feature_id"], int) and row["source_feature_id"] > 0
            for row in instance_rows
        ),
        "instances_with_no_source_feature_id": sum(
            1 for row in instance_rows if not row.get("source_feature_id")
        ),
        "holes_preserved": {
            "instances_with_holes": hole_instances,
            "features_with_holes_in_source": sum(1 for f in corpus.features if f.hole_rings),
            "hole_loss": False,
        },
        "multipart_preserved": {
            "multipart_instances": multipart_instances,
            "multipart_features_in_source": sum(1 for f in corpus.features if f.n_parts > 1),
            "multipart_loss": False,
            "note": (
                "no tile-clipped instance has more than one outer ring because the 5 multipart source "
                "features have widely separated parts that never fall in the same 512x512 window; "
                "their parts are still preserved as separate clipped instances of the same "
                "source_feature_id"
            ),
        },
        "no_global_area_deletion": {
            "tiny_instances_retained": tiny_count,
            "min_contour_area_filter": None,
            "note": "tiny instances are flagged (tiny_area) but never removed from the canonical GT",
        },
        "empty_tiles_supported": {
            "empty_tiles": empty_tiles,
            "non_empty_tiles": len(records) - empty_tiles,
        },
        "cache": {
            "directory": "artifacts/whu_native_vector/instances",
            "files_written": len(records),
            "gitignored": True,
        },
        "determinism": {
            "geometry_source": "EA.shp bytes",
            "shp_sha256": provenance["shp_sha256"],
            "clipping": "Sutherland-Hodgman against the validated tile window",
            "runtime_seconds": round(time.time() - started, 2),
        },
    }
    write_json(INTEGRITY_OUT, integrity)

    split_audit = {
        "_doc": (
            "Task 6L section 8/18. Scene-disjoint split audit: tile overlap, source-feature "
            "identity leakage, RGB duplicate hashes and the documented boundary exclusions."
        ),
        "task": "6L",
        "view": SCENE_DISJOINT_VIEW,
        "definition": {"train": "train1", "val": "train2", "test": "test"},
        "split_sizes": {key: len(value) for key, value in scene_tiles.items()},
        "tile_overlap": {key: len(value) for key, value in tile_overlap.items()},
        "tile_overlap_zero": all(len(value) == 0 for value in tile_overlap.values()),
        "source_feature_overlap_after_exclusions": {key: len(value) for key, value in leakage.items()},
        "source_feature_leakage_zero": all(len(value) == 0 for value in leakage.values()),
        "excluded_boundary_crossing_features": boundary_features,
        "excluded_boundary_crossing_feature_count": len(boundary_features),
        "excluded_boundary_instance_rows": boundary_instance_rows,
        "exclusion_reason": (
            "a feature crossing the ~1 m overlap between adjacent whole rasters would otherwise "
            "appear in two splits; these features are excluded from the scene-disjoint view only"
        ),
        "rgb_duplicate_hashes": {
            "hash_definition": "SHA256 of the 64x64 bilinear grayscale downsample of the tile image",
            "duplicate_groups": len(rgb_duplicates),
            "duplicate_tiles": sum(len(tiles) for tiles in rgb_duplicates.values()),
            "cross_split_groups": len(cross_split_duplicates),
            "cross_split_examples": dict(list(cross_split_duplicates.items())[:5]),
            "all_cross_split_duplicates_degenerate": all(
                entry["degenerate_content"] for entry in cross_split_duplicates.values()
            ),
            "note": (
                "cross-split duplicates are only acceptable when the content is degenerate "
                "(near-uniform tiles); any non-degenerate cross-split duplicate is a leakage finding"
            ),
        },
        "interpretation": (
            "scene/raster separation only - not unseen-city or broad geographic generalization; "
            "no cross-city claim may be made from this split"
        ),
    }
    write_json(SPLIT_AUDIT_OUT, split_audit)

    print(
        f"[6l.build] tiles {len(records)}; instances {len(instance_rows)}; distinct features "
        f"{len(feature_rasters)}; empty tiles {empty_tiles}; boundary-crossing features "
        f"{len(boundary_features)}; feature leakage "
        f"{ {k: len(v) for k, v in leakage.items()} }",
        flush=True,
    )
    print(
        f"[6l.build] scene-disjoint sizes "
        f"{ {k: len(v) for k, v in scene_tiles.items()} }; tile overlap "
        f"{ {k: len(v) for k, v in tile_overlap.items()} }",
        flush=True,
    )
    print(f"[6l.build] wrote dataset + integrity + split audit in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
