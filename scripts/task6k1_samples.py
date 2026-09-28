"""Task 6K.1: small representative overlay panels (tracked, deliberately tiny).

Each panel shows, for one deterministic tile: the RGB crop, the raster semantic label, the native
vector instances (instance-coloured), the current pseudo-instance view, and the disagreement map
(native-only / pseudo-only / shared). Panels are written to `evaluation/task6k1_samples/`.

Writes nothing else; large caches stay in the gitignored tree.
"""

from __future__ import annotations

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

from task6k1_common import (  # noqa: E402
    CROPPED_ROOT,
    EVAL,
    load_instance_view,
    masks_from_label_map,
    write_json,
)

SAMPLES = EVAL / "task6k1_samples"
SCALE = 1  # 512 px panels, kept small by PNG compression


def raster_label_path(stem: str, split: str) -> Path | None:
    for candidate_split in (split, "train", "test", "train_no", "test_no"):
        candidate = CROPPED_ROOT / candidate_split / "label" / f"{stem}.tif"
        if candidate.is_file():
            return candidate
    return None


def crop_image_path(stem: str, split: str) -> Path | None:
    for candidate_split in (split, "train", "test", "train_no", "test_no"):
        candidate = CROPPED_ROOT / candidate_split / "image" / f"{stem}.tif"
        if candidate.is_file():
            return candidate
    return None


def colorize(label_map: np.ndarray) -> np.ndarray:
    """Deterministic instance-coloured rendering of a label map."""

    ids = [int(v) for v in np.unique(label_map) if int(v) != 0]
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for index, value in enumerate(ids):
        hue = int((index * 47) % 180)
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        out[label_map == value] = bgr
    return out


def main() -> int:
    started = time.time()
    SAMPLES.mkdir(parents=True, exist_ok=True)
    from task6k_common import aligned_tiles, converted_candidate_set

    tiles = aligned_tiles()
    # deterministic spread across the corpus (train/val/test, sparse and dense tiles)
    picks = [tiles[index] for index in (0, 300, 900, 1500, 2100, 2700, 3300, 3900, 4030)]
    written = []
    for tile in picks:
        view = load_instance_view(tile.stem)
        if view is None:
            continue
        image_path = crop_image_path(tile.stem, tile.source_split)
        label_path = raster_label_path(tile.stem, tile.split)
        if image_path is None or label_path is None:
            continue
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        label = np.asarray(Image.open(label_path).convert("L")) > 0
        vector_map = np.asarray(view["label_map"], dtype=np.int32)
        vector_mask = vector_map > 0
        pseudo_set = converted_candidate_set(tile)
        pseudo_map = pseudo_set.label_map if pseudo_set is not None else np.zeros_like(vector_map)
        pseudo_mask = pseudo_map > 0

        shared = np.logical_and(vector_mask, pseudo_mask)
        vector_only = np.logical_and(vector_mask, ~pseudo_mask)
        pseudo_only = np.logical_and(~vector_mask, pseudo_mask)
        disagreement = np.zeros((*label.shape, 3), dtype=np.uint8)
        disagreement[shared] = (80, 80, 80)
        disagreement[pseudo_only] = (0, 0, 255)   # pseudo-only (conversion artefacts, merges)
        disagreement[vector_only] = (0, 255, 0)   # vector-only (buildings missing from pseudo)

        label_vis = np.zeros((*label.shape, 3), dtype=np.uint8)
        label_vis[label] = (255, 255, 255)

        panel = np.concatenate(
            [rgb, label_vis, colorize(vector_map), colorize(pseudo_map), disagreement], axis=1
        )
        # halve the panel so the tracked overlays stay small (the full-resolution data is reproducible)
        panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2), interpolation=cv2.INTER_AREA)
        path = SAMPLES / f"{tile.stem}_panel.png"
        Image.fromarray(panel).save(path, optimize=True)
        written.append(
            {
                "stem": tile.stem,
                "split": tile.split,
                "file": path.name,
                "vector_instances": len(masks_from_label_map(vector_map)),
                "pseudo_instances": int(len(pseudo_set.candidates)) if pseudo_set else 0,
                "disagreement_pixels": int(disagreement.any(axis=2).sum()),
                "pseudo_only_pixels": int(pseudo_only.sum()),
                "vector_only_pixels": int(vector_only.sum()),
            }
        )
    write_json(
        SAMPLES / "index.json",
        {
            "_doc": (
                "Task 6K.1 sample overlays. Columns: RGB crop | raster semantic label | native vector "
                "instances | current pseudo-instances | disagreement (grey shared, red pseudo-only, "
                "green vector-only). Small representative panels only; large caches are gitignored."
            ),
            "task": "6K.1",
            "columns": ["rgb", "raster_label", "vector_instances", "pseudo_instances", "disagreement"],
            "panels": written,
            "seconds": round(time.time() - started, 2),
        },
    )
    total_bytes = sum(path.stat().st_size for path in SAMPLES.glob("*.png"))
    print(f"[task6k1.samples] {len(written)} panels, {total_bytes / 1024:.0f} KiB total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
