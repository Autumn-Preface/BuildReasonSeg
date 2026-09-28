"""Task 6K shared script helpers: read-only paths, split recovery, alignment, JSON output."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "datasets" / "transforms"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_source_audit import (  # noqa: E402
    CONVERTED_CANDIDATES,
    CROPPED_ROOT,
    CURRENT_ROOT,
    ORIGINAL_ROOT,
    SOURCE_SPLITS,
    converted_root,
    path_provenance,
    write_json,
)

EVAL = REPO_ROOT / "evaluation"

#: The frozen BuildSpatialReason relation configuration (Task 3B, v1).
RELATION_CONFIG = REPO_ROOT / "configs" / "spatial_relations_v1.yaml"


def source_split_dir(split: str, kind: str) -> Path:
    """`kind` is 'image' or 'label' inside the cropped source subtree."""

    return CROPPED_ROOT / split / kind


def source_stems(split: str, kind: str = "image") -> set[str]:
    directory = source_split_dir(split, kind)
    if not directory.is_dir():
        return set()
    stems = set()
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            stems.add(path.stem)
    return stems


def source_path(split: str, kind: str, stem: str) -> Path | None:
    directory = source_split_dir(split, kind)
    for extension in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".TIF", ".TIFF", ".PNG"):
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class AlignedTile:
    """One tile aligned across the three representations."""

    split: str          # current BuildReasonSeg split: train | val | test
    source_split: str   # recovered original source split: train | test
    stem: str
    source_label: Path
    source_image: Path | None
    yolo_label: Path
    yolo_image: Path
    component_map: Path | None
    metadata: dict | None

    @property
    def width(self) -> int:
        if self.metadata:
            return int(self.metadata["width"])
        return 512

    @property
    def height(self) -> int:
        if self.metadata:
            return int(self.metadata["height"])
        return 512


def converted_stems(split: str, kind: str = "labels") -> set[str]:
    root = converted_root()
    if root is None:
        return set()
    directory = root / kind / split
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.iterdir() if path.is_file()}


def current_metadata(split: str) -> dict[str, dict]:
    path = CURRENT_ROOT / "metadata" / f"{split}.jsonl"
    records = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records[str(record["image_id"])] = record
    return records


def aligned_tiles() -> list[AlignedTile]:
    """Every tile present in the historical converted dataset, aligned to source + current.

    The historical split is RECOVERED from the existing converted folders; it is never
    regenerated from seed 42 (section 4/10).
    """

    root = converted_root()
    if root is None:
        raise RuntimeError("historical converted dataset not found")
    tiles: list[AlignedTile] = []
    # The converter drew train/val from the source `train` pool and test from source `test`.
    source_split_for = {"train": "train", "val": "train", "test": "test"}
    for split in ("train", "val", "test"):
        metadata = current_metadata(split)
        for stem in sorted(converted_stems(split)):
            source_split = source_split_for[split]
            source_label = source_path(source_split, "label", stem)
            source_image = source_path(source_split, "image", stem)
            yolo_label = root / "labels" / split / f"{stem}.txt"
            yolo_image = root / "images" / split / f"{stem}.tif"
            component_map = CURRENT_ROOT / "components" / split / f"{stem}.png"
            if not component_map.is_file():
                component_map = None
            tiles.append(
                AlignedTile(
                    split=split,
                    source_split=source_split,
                    stem=stem,
                    source_label=source_label,
                    source_image=source_image,
                    yolo_label=yolo_label,
                    yolo_image=yolo_image,
                    component_map=component_map,
                    metadata=metadata.get(stem),
                )
            )
    return tiles


def dataset_query_type_counts() -> dict[str, int]:
    """Current query-type record counts over the frozen BuildSpatialReason v0.1.1 splits."""

    counts: dict[str, int] = {}
    base = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1.1"
    for split in ("train", "val", "test"):
        path = base / f"{split}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                query_type = str(record["query_type"])
                counts[query_type] = counts.get(query_type, 0) + 1
    return counts


def frozen_relation_config():
    sys.path.insert(0, str(REPO_ROOT / "spatial_reasoning"))
    from thresholds import load_config

    return load_config(RELATION_CONFIG)


def candidate_set_from_labels(labels: np.ndarray, records, source: str):
    """Build a Task 6J `CandidateSet` from a labeled array plus component records."""

    from buildreasonseg_mvp.structured_grounding import Candidate, CandidateSet

    height, width = labels.shape
    candidates = []
    for record in records:
        mask = labels == record.component_id
        if not mask.any():
            continue
        candidates.append(
            Candidate(
                candidate_id=int(record.component_id),
                mask=mask,
                bbox_xyxy_px=tuple(int(v) for v in record.bbox_xyxy_px),
                centroid_px=(float(record.centroid_px[0]), float(record.centroid_px[1])),
                area_px=int(record.area_px),
                touches_image_border=bool(record.touches_image_border),
                confidence=None,
                source=source,
            )
        )
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    label_map = np.where(np.isin(labels, list(candidate_ids)), labels, 0).astype(np.uint8)
    return CandidateSet(width=width, height=height, candidates=candidates, label_map=label_map, source=source)


def converted_candidate_set(tile: AlignedTile):
    """The candidate set implied by the CURRENT component representation (from the polygons)."""

    import cv2

    from buildreasonseg_mvp.whu_source_audit import ComponentRecord

    if tile.component_map is None or tile.metadata is None:
        return None
    labels = cv2.imread(str(tile.component_map), cv2.IMREAD_GRAYSCALE)
    if labels is None:
        return None
    records = []
    for entry in tile.metadata["components"]:
        bbox = entry["bbox_xyxy_px"]
        records.append(
            ComponentRecord(
                component_id=int(entry["component_id"]),
                area_px=int(entry["area_px"]),
                bbox_xyxy_px=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                centroid_px=(float(entry["centroid_px"][0]), float(entry["centroid_px"][1])),
                touches_image_border=bool(entry["touches_image_border"]),
                width_px=int(entry["width_px"]),
                height_px=int(entry["height_px"]),
            )
        )
    return candidate_set_from_labels(labels.astype(np.int32), records, source="converted")


def raw_candidate_set(labels: np.ndarray, records):
    return candidate_set_from_labels(labels, records, source="raw")


def deterministic_json(path: Path, payload: dict) -> None:
    write_json(path, payload)
