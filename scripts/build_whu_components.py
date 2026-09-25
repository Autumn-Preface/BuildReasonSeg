#!/usr/bin/env python
"""CLI: convert legacy WHU polygon labels into building component representations.

    python scripts/build_whu_components.py [--split train|val|test|all]
                                           [--dataset-root PATH] [--out-dir PATH]
                                           [--limit N] [--quiet]

Produces, under the output directory:

    components/<split>/<image_id>.png     indexed component map (uint8, 0 = background)
    metadata/<split>.jsonl                one JSON line per source image
    component_manifest.json               dataset-level summary

Every path written is relative to this repository. The legacy dataset is only
ever read.

Terminology: a polygon is a *building connected component*, not a verified
physical building instance. See docs/data_representation.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running as a plain script without requiring an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "datasets" / "transforms"))

import numpy as np  # noqa: E402

from polygon_to_component_map import (  # noqa: E402
    BACKGROUND_ID,
    REPRESENTATION_NAME,
    REPRESENTATION_VERSION,
    build_component_map,
    check_id_contiguity,
    iter_split,
    read_image_size,
    result_to_metadata,
    unique_component_ids,
    write_indexed_png,
)

DEFAULT_DATASET_ROOT = _REPO_ROOT.parent / "WHU_Building_Segment" / "dataset" / "WHU_YOLO_dataset"
DEFAULT_OUT_DIR = _REPO_ROOT / "datasets" / "whu"
SPLITS = ("train", "val", "test")


WORKSPACE_ROOT = _REPO_ROOT.parent


def _relative(path: Path) -> str:
    """Return a POSIX path relative to the repository root.

    Paths therefore read naturally from inside this repository, e.g.::

        datasets/whu/components/train/1_0.png
        ../WHU_Building_Segment/dataset/WHU_YOLO_dataset/images/train/1_0.tif

    The legacy WHU dataset deliberately lives outside this repository, so
    dataset paths legitimately start with ``../``. Paths are never absolute.

    Absolute paths outside the shared workspace are refused, so this stays safe
    if the project is moved or a wrong dataset path is passed in.
    """

    resolved = path.resolve()
    workspace = WORKSPACE_ROOT.resolve()
    if not resolved.is_relative_to(workspace):
        raise SystemExit(
            f"refusing to record a path outside the workspace: {resolved}"
        )
    return os.path.relpath(resolved, _REPO_ROOT.resolve()).replace(os.sep, "/")


def collect_split_stats(records: list[dict]) -> dict:
    """Aggregate dataset-level statistics for one split."""

    n_components = int(sum(r["n_components"] for r in records))
    per_image = [r["n_components"] for r in records]
    areas: list[int] = []
    for r in records:
        for c in r["components"]:
            areas.append(int(c["area_px"]))

    stats = {
        "images": len(records),
        "components": n_components,
        "components_per_image_min": int(min(per_image)) if per_image else 0,
        "components_per_image_median": float(np.median(per_image)) if per_image else 0.0,
        "components_per_image_mean": float(np.mean(per_image)) if per_image else 0.0,
        "components_per_image_max": int(max(per_image)) if per_image else 0,
        "component_area_px_min": int(min(areas)) if areas else 0,
        "component_area_px_median": float(np.median(areas)) if areas else 0.0,
        "component_area_px_p90": float(np.percentile(areas, 90)) if areas else 0.0,
        "component_area_px_max": int(max(areas)) if areas else 0,
        "empty_mask_polygons": int(sum(len(r["rasterization"]["empty_mask_polygons"]) for r in records)),
        "polygon_conflicts": int(sum(len(r["rasterization"]["conflict_pairs"]) for r in records)),
        "malformed_lines": int(sum(len(r["rasterization"]["malformed_lines"]) for r in records)),
        "non_building_class_lines": int(sum(len(r["rasterization"]["non_building_class_lines"]) for r in records)),
        "out_of_bound_vertices": int(sum(r["rasterization"]["out_of_bound_vertices"] for r in records)),
        "images_with_defects": int(sum(1 for r in records if not r["rasterization"]["is_valid"])),
        "touching_image_border_components": int(
            sum(1 for r in records for c in r["components"] if c["touches_image_border"])
        ),
    }
    return stats


def write_polygon_archive(
    records: list[dict],
    polygons_by_image: dict[str, list[np.ndarray]],
    out_path: Path,
) -> int:
    """Write full-precision polygon provenance to a compressed archive.

    Polygon vertices are recorded exactly as they appear in the legacy labels
    (float64, unrounded). They are kept as provenance only -- the geometry
    ground truth is the rasterized component map.

    They live in a separate compressed ``.npz`` rather than inline in the JSONL
    because embedding 36,926 high-vertex-count polygons as decimal text inflates
    the metadata to tens of megabytes for no benefit. The JSONL keeps a
    per-component ``polygon_vertices`` count and a pointer to this archive, so
    the provenance is still complete and reachable.

    Layout: one flat float64 array of all vertices per split, plus an offsets
    and lengths array giving each component's slice, indexed in the same order
    as the JSONL records/components.
    """

    offsets: list[int] = []
    lengths: list[int] = []
    chunks: list[np.ndarray] = []

    cursor = 0
    for record in records:
        image_id = record["image_id"]
        image_polygons = polygons_by_image.get(image_id, [])
        for index in range(len(record["components"])):
            vertices = (
                image_polygons[index]
                if index < len(image_polygons)
                else np.zeros((0, 2), dtype=np.float64)
            )
            flat = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
            offsets.append(cursor)
            lengths.append(int(flat.shape[0]))
            chunks.append(flat)
            cursor += int(flat.shape[0])

    vertices_all = (
        np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), dtype=np.float64)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        vertices=vertices_all,
        offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
    )
    return int(vertices_all.shape[0])


def convert_split(
    dataset_root: Path,
    out_dir: Path,
    split: str,
    limit: int | None,
    quiet: bool,
) -> tuple[list[dict], dict, dict[str, list[np.ndarray]]]:
    """Convert one split. Returns (metadata records, split statistics, polygons)."""

    components_dir = out_dir / "components" / split
    metadata_dir = out_dir / "metadata"
    polygon_dir = out_dir / "polygons"
    components_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    polygon_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    polygons_by_image: dict[str, list[np.ndarray]] = {}
    started = time.time()

    for i, (image_id, image_path, label_path) in enumerate(iter_split(dataset_root, split), start=1):
        if limit is not None and i > limit:
            break

        width, height = read_image_size(image_path)
        result = build_component_map(
            label_path=label_path,
            image_size=(width, height),
            image_id=image_id,
            split=split,
        )

        # Invariant: component ids must be exactly {1..n_components}, with no
        # gaps. A gap means a polygon produced no pixels, which is a reported
        # defect rather than something to silently renumber.
        gaps = check_id_contiguity(result.component_map, len(result.components))
        if gaps:
            result.diagnostics.empty_mask_polygons.extend(
                [g - 1 for g in gaps if (g - 1) not in result.diagnostics.empty_mask_polygons]
            )

        map_path = components_dir / f"{image_id}.png"
        write_indexed_png(result.component_map, map_path)

        record = result_to_metadata(
            result=result,
            image_path_rel=_relative(image_path),
            label_path_rel=_relative(label_path),
            component_map_rel=_relative(map_path),
        )

        # Full-precision polygon provenance moves to the compressed archive, so
        # the JSONL stays small. The per-component vertex count stays inline so
        # the archive entry can be validated without loading the archive.
        polygons_by_image[image_id] = [
            np.asarray(p.vertices_normalized, dtype=np.float64) for p in result.polygons
        ]
        for component in record["components"]:
            component["polygon_vertices"] = int(
                polygons_by_image[image_id][component["source_polygon_index"]].shape[0]
            )
            component["polygon_vertices_ref"] = f"datasets/whu/polygons/{split}.npz"
            del component["polygon_normalized"]

        records.append(record)

        if not quiet and i % 250 == 0:
            print(f"  [{split}] {i} images...", flush=True)

    stats = collect_split_stats(records)
    stats["elapsed_seconds"] = round(time.time() - started, 3)

    jsonl_path = metadata_dir / f"{split}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    archive_path = polygon_dir / f"{split}.npz"
    n_vertices = write_polygon_archive(records, polygons_by_image, archive_path)
    stats["polygon_archive"] = _relative(archive_path)
    stats["polygon_archive_bytes"] = archive_path.stat().st_size
    stats["polygon_vertices_total"] = n_vertices

    if not quiet:
        print(
            f"  [{split}] done: {stats['images']} images, "
            f"{stats['components']} components in {stats['elapsed_seconds']}s"
        )

    return records, stats, polygons_by_image


def build_manifest(all_stats: dict[str, dict], n_images: int, n_components: int, max_comp: int) -> dict:
    """Build the dataset-level manifest."""

    return {
        "representation": REPRESENTATION_NAME,
        "version": REPRESENTATION_VERSION,
        "source": "legacy_yolo_polygon",
        "source_dataset_root": "../../../WHU_Building_Segment/dataset/WHU_YOLO_dataset",
        "terminology_note": (
            "A polygon is a building connected component, not a verified physical "
            "building instance. The source raster was a binary semantic mask, so "
            "mutually touching buildings were already merged upstream."
        ),
        "background_id": BACKGROUND_ID,
        "dtype": "uint8",
        "dtype_justification": (
            "Max components per image is well below 255, so uint8 suffices. "
            "The converter raises ComponentIdOverflowError rather than "
            "overflowing silently if this is ever exceeded."
        ),
        "geometry_source": "rasterized_component_map",
        "rasterization_backend": "cv2.fillPoly",
        "rasterization_backend_note": (
            "cv2.fillPoly rasterizes the ALREADY EXISTING polygons into component "
            "maps. It is NOT the routine that created the polygon labels: those came "
            "from cv2.findContours (RETR_EXTERNAL) + cv2.approxPolyDP in the legacy "
            "conversion script. Polygon->mask agreement therefore means this "
            "representation is self-consistent and deterministic; it does NOT mean the "
            "component map equals the original raster ground truth."
        ),
        "splits": all_stats,
        "total_images": n_images,
        "total_components": n_components,
        "max_components_per_image": max_comp,
        "polygon_conflicts": int(sum(s["polygon_conflicts"] for s in all_stats.values())),
        "known_biases": [
            {
                "id": "KB-01",
                "bias": "touching buildings may already be merged",
                "detail": (
                    "The source raster was a binary semantic mask and the legacy "
                    "conversion used connected components, so mutually touching "
                    "buildings became a single component. Measured upstream: 2.601% "
                    "of sampled component pairs have a contour distance <= 1.5 px."
                ),
                "impact": "Component counts underestimate physical building counts in dense areas.",
            },
            {
                "id": "KB-02",
                "bias": "internal holes were lost by RETR_EXTERNAL",
                "detail": (
                    "Legacy polygons came from external contours, so interior "
                    "courtyards and light wells were filled. A rasterized component "
                    "is a filled exterior polygon, not a pixel-exact footprint."
                ),
                "impact": "area_px slightly overestimates the true footprint; area-based relations inherit this.",
            },
            {
                "id": "KB-03",
                "bias": "not authoritative physical-building instance ground truth",
                "detail": (
                    "This representation must not be described as, or interpreted as, "
                    "a physical building instance segmentation ground truth."
                ),
                "impact": "Any claim of 'building instance' accuracy must state this caveat.",
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="all", choices=(*SPLITS, "all"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="max images per split (debug)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    dataset_root: Path = args.dataset_root
    out_dir: Path = args.out_dir

    if not dataset_root.is_dir():
        print(f"error: dataset root not found: {dataset_root}", file=sys.stderr)
        return 2

    splits = SPLITS if args.split == "all" else (args.split,)
    if not args.quiet:
        print(f"dataset root : {dataset_root}")
        print(f"output dir   : {out_dir}")

    all_stats: dict[str, dict] = {}
    n_images = 0
    n_components = 0
    max_comp = 0

    for split in splits:
        records, stats, _ = convert_split(dataset_root, out_dir, split, args.limit, args.quiet)
        all_stats[split] = stats
        n_images += stats["images"]
        n_components += stats["components"]
        max_comp = max(max_comp, stats["components_per_image_max"])

    manifest = build_manifest(all_stats, n_images, n_components, max_comp)
    manifest_path = out_dir / "component_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not args.quiet:
        print()
        print(f"total images     : {n_images}")
        print(f"total components : {n_components}")
        print(f"max components   : {max_comp}")
        print(f"conflicts        : {manifest['polygon_conflicts']}")
        print(f"manifest         : {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
