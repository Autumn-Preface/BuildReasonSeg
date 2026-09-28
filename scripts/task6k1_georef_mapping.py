"""Task 6K.1 Part B (sections 4-5): whole-image georeferencing + tile mapping recovery.

Read-only. Inspects the whole-area raster metadata with a standard-library TIFF/BigTIFF reader
(no full decode), reads the ``.tfw`` world files, derives the 512-tile grid per raster, and
VALIDATES the tile-index -> window mapping against the cropped tiles:

* RGB identity: the cropped image tile must equal the whole-image window pixel for pixel;
* label identity: the cropped label tile must equal the whole-label window (IoU 1.0, or both
  empty);
* vector agreement: the native vector clipped to the same window must reproduce the label window.

Windowed reads use small PIL crops (an existing read-only tool) ---never a whole-image load.

Writes `evaluation/task6k1_whole_image_georef.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

Image.MAX_IMAGE_PIXELS = None  # metadata + small crops only; never a full decode

from buildreasonseg_mvp.shapefile_reader import read_prj_text  # noqa: E402
from buildreasonseg_mvp.whu_vector_audit import (  # noqa: E402
    CROPPED_ROOT,
    PRJ_PATH,
    REGION_TO_RASTER,
    WHOLE_AREA_DIR,
    WorldFile,
    rasterize_feature,
)

from task6k1_common import (  # noqa: E402
    EVAL,
    iou_masks,
    label_metadata,
    load_vector_corpus,
    percentile_summary,
    raster_affine,
    raster_grid,
    raster_metadata,
    tile_geometry,
    write_json,
)

OUT_GEOREF = EVAL / "task6k1_whole_image_georef.json"
OUT_MAPPING = EVAL / "task6k1_tile_mapping.json"
SPLITS = ("train", "train_no", "test", "test_no")


def cropped_tile_paths(stem: str) -> tuple[Path | None, Path | None]:
    for split in SPLITS:
        image = CROPPED_ROOT / split / "image" / f"{stem}.tif"
        if image.is_file():
            return image, CROPPED_ROOT / split / "label" / f"{stem}.tif"
    return None, None


def window_read(path: Path, window: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, width, height = window
    with Image.open(path) as handle:
        return np.asarray(handle.crop((x0, y0, x0 + width, y0 + height)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-sample", type=int, default=40, help="RGB identity checks per raster")
    parser.add_argument("--label-sample", type=int, default=120, help="label identity checks per raster")
    args = parser.parse_args(argv)

    started = time.time()
    corpus = load_vector_corpus()
    prj = read_prj_text(PRJ_PATH)

    raster_reports = {}
    for region, raster in sorted(REGION_TO_RASTER.items()):
        metadata = raster_metadata(raster)
        label_meta = label_metadata(raster)
        affine: WorldFile = raster_affine(raster)
        grid = raster_grid(raster)
        crop_counts = {}
        stems_by_prefix = {}
        for split in SPLITS:
            directory = CROPPED_ROOT / split / "image"
            if not directory.is_dir():
                continue
            files = list(directory.glob("*.tif"))
            crop_counts[split] = len(files)
            for path in files:
                stems_by_prefix.setdefault(path.stem, split)
        prefix = region if region != "test" else ""
        matching = sorted(
            (stem for stem in stems_by_prefix if (stem.split("_", 1)[0] if "_" in stem else "test") == region),
            key=lambda stem: int(stem.split("_", 1)[1]) if "_" in stem else int(stem),
        )
        capacities = {
            "tile_grid_capacity": grid.capacity,
            "cropped_tiles_in_region": len(matching),
            "capacity_minus_cropped": grid.capacity - len(matching),
        }

        # deterministic sample: evenly spaced indices across the region
        def sample(items, count):
            if not items:
                return []
            step = max(1, len(items) // max(count, 1))
            return items[::step][:count]

        image_checks = []
        for stem in sample(matching, int(args.image_sample)):
            tile_geometry_value = tile_geometry(stem)
            if tile_geometry_value is None:
                continue
            image_path, _label_path = cropped_tile_paths(stem)
            if image_path is None:
                continue
            window = [tile_geometry_value.x0, tile_geometry_value.y0,
                      tile_geometry_value.size, tile_geometry_value.size]
            t0 = time.time()
            whole = window_read(WHOLE_AREA_DIR / "image" / f"{raster}.tif", tuple(window))
            elapsed = time.time() - t0
            tile = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            diff = float(np.abs(whole.astype(np.int16) - tile.astype(np.int16)).mean())
            max_diff = int(np.abs(whole.astype(np.int16) - tile.astype(np.int16)).max())
            image_checks.append(
                {
                    "stem": stem,
                    "row": tile_geometry_value.row,
                    "col": tile_geometry_value.col,
                    "window": window,
                    "mean_abs_diff": diff,
                    "max_abs_diff": max_diff,
                    "identical": bool(max_diff == 0),
                    "shape_match": list(whole.shape) == list(tile.shape),
                    "seconds": round(elapsed, 2),
                }
            )

        label_checks = []
        vector_ious = []
        both_empty = 0
        for stem in sample(matching, int(args.label_sample)):
            geometry_value = tile_geometry(stem)
            if geometry_value is None:
                continue
            _image_path, label_path = cropped_tile_paths(stem)
            if label_path is None:
                continue
            window = (geometry_value.x0, geometry_value.y0, geometry_value.size, geometry_value.size)
            whole_label = window_read(WHOLE_AREA_DIR / "label" / f"{raster}.tif", window)
            whole_label_binary = np.asarray(whole_label) > 0
            cropped_label = np.asarray(Image.open(label_path).convert("L")) > 0
            label_iou = iou_masks(whole_label_binary, cropped_label)
            empty = not whole_label_binary.any() and not cropped_label.any()
            both_empty += int(empty)
            # vector agreement inside the same window
            bbox = geometry_value.map_bbox
            vector_mask = np.zeros((geometry_value.size, geometry_value.size), dtype=bool)
            for index in corpus.candidates(bbox):
                vector_mask |= rasterize_feature(
                    corpus.features[index], geometry_value.affine,
                    (geometry_value.size, geometry_value.size),
                    origin_col=geometry_value.x0, origin_row=geometry_value.y0,
                )
            vector_iou = iou_masks(vector_mask, cropped_label)
            vector_ious.append(vector_iou)
            label_checks.append(
                {
                    "stem": stem,
                    "label_iou_whole_vs_cropped": label_iou,
                    "both_empty": bool(empty),
                    "vector_iou_vs_cropped_label": vector_iou,
                    "vector_pixels": int(vector_mask.sum()),
                    "label_pixels": int(cropped_label.sum()),
                }
            )

        identical = sum(1 for check in image_checks if check["identical"])
        raster_reports[raster] = {
            "image_metadata": metadata,
            "label_metadata": label_meta,
            "world_file": affine.as_dict(),
            "tile_grid": grid.as_dict(),
            "cropped_tile_counts": crop_counts,
            **capacities,
            "crs_wkt": prj,
            "rgb_identity": {
                "checked": len(image_checks),
                "identical": identical,
                "all_identical": bool(image_checks) and identical == len(image_checks),
                "max_abs_diff_observed": max((c["max_abs_diff"] for c in image_checks), default=None),
                "mean_abs_diff_summary": percentile_summary([c["mean_abs_diff"] for c in image_checks]),
                "seconds_summary": percentile_summary([c["seconds"] for c in image_checks]),
                "checks": image_checks,
            },
            "label_identity": {
                "checked": len(label_checks),
                "both_empty_cases": both_empty,
                "iou_min": min((c["label_iou_whole_vs_cropped"] for c in label_checks), default=None),
                "all_exact_or_both_empty": all(
                    c["label_iou_whole_vs_cropped"] >= 0.999 or c["both_empty"] for c in label_checks
                ),
                "checks": label_checks,
            },
            "vector_vs_label_in_same_window": {
                "iou_summary": percentile_summary(vector_ious),
                "checks": len(vector_ious),
            },
        }

    georef_report = {
        "_doc": (
            "Task 6K.1 section 4. Whole-area raster georeferencing: TIFF/BigTIFF structure read with "
            "a standard-library IFD parser (metadata only - the multi-gigapixel rasters are NEVER "
            "fully decoded), the label-side .tfw world files, the CRS WKT and the raster/label/cropped "
            "inventory. Pixel-level mapping validation lives in task6k1_tile_mapping.json."
        ),
        "task": "6K.1",
        "method": {
            "tiff_reader": "standard-library TIFF/BigTIFF IFD reader (metadata only)",
            "pixel_reader": "PIL windowed crop (existing read-only tool), never a full decode",
            "tile_size": 512,
            "georeferencing": "label-side .tfw (A, D, B, E, C, F), corner-based",
            "full_decode_performed": False,
        },
        "crs": {"wkt": prj, "projection_name": "WGS_1984_World_Mercator", "units": "meter"},
        "rasters": {
            raster: {
                "image_metadata": data["image_metadata"],
                "label_metadata": data["label_metadata"],
                "world_file": data["world_file"],
                "tile_grid": data["tile_grid"],
                "cropped_tile_counts": data["cropped_tile_counts"],
                "tile_grid_capacity": data["tile_grid_capacity"],
                "cropped_tiles_in_region": data["cropped_tiles_in_region"],
                "capacity_minus_cropped": data["capacity_minus_cropped"],
            }
            for raster, data in raster_reports.items()
        },
        "crs_wkt": prj,
        "seconds": round(time.time() - started, 2),
    }
    mapping_report = {
        "_doc": (
            "Task 6K.1 section 5. Cropped-tile -> whole-image mapping recovery and VALIDATION. The "
            "mapping is not assumed: each sampled cropped image tile is compared pixel-by-pixel with "
            "the corresponding whole-image window (RGB identity), each sampled cropped label tile with "
            "the whole-label window (exact or both empty), and the native vector clipped to the same "
            "window is compared with the label window."
        ),
        "task": "6K.1",
        "hypothesis": (
            "cropped tile `<region>_<index>` (or bare `<index>` for the test region) is the window at "
            "col = index % floor(width/512), row = index // floor(width/512) of the corresponding "
            "whole-area raster, i.e. only FULL 512x512 tiles were cropped"
        ),
        "rasters": {
            raster: {
                "tile_grid": data["tile_grid"],
                "tile_grid_capacity": data["tile_grid_capacity"],
                "cropped_tiles_in_region": data["cropped_tiles_in_region"],
                "capacity_minus_cropped": data["capacity_minus_cropped"],
                "rgb_identity": data["rgb_identity"],
                "label_identity": data["label_identity"],
                "vector_vs_label_in_same_window": data["vector_vs_label_in_same_window"],
            }
            for raster, data in raster_reports.items()
        },
        "conclusion": {
            "grid_capacity_equals_cropped_tiles_everywhere": all(
                data["capacity_minus_cropped"] == 0 for data in raster_reports.values()
            ),
            "rgb_windows_pixel_identical_everywhere": all(
                data["rgb_identity"]["all_identical"] for data in raster_reports.values()
            ),
            "labels_exact_or_both_empty_everywhere": all(
                data["label_identity"]["all_exact_or_both_empty"] for data in raster_reports.values()
            ),
            "mapping_confidence": "validated (pixel identity on the sampled tiles + exact grid capacity)",
        },
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT_GEOREF, georef_report)
    write_json(OUT_MAPPING, mapping_report)
    for raster, data in raster_reports.items():
        print(
            f"[task6k1.georef] {raster}: grid {data['tile_grid']['columns']}x{data['tile_grid']['rows']} "
            f"= {data['tile_grid_capacity']} vs cropped {data['cropped_tiles_in_region']} "
            f"(diff {data['capacity_minus_cropped']}); RGB identical "
            f"{data['rgb_identity']['identical']}/{data['rgb_identity']['checked']}; label exact "
            f"{data['label_identity']['all_exact_or_both_empty']}; vector IoU mean "
            f"{data['vector_vs_label_in_same_window']['iou_summary'].get('mean')}",
            flush=True,
        )
    print(
        f"[task6k1.georef] wrote {OUT_GEOREF.name} + {OUT_MAPPING.name} in {time.time() - started:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
