"""Task 6L Part C: reusable adapter for the canonical native-vector dataset.

Independent of YOLO and of the old pseudo-component format. Provides the interface required by the
task file and a bridge that exposes native instances through the geometry fields the frozen Task 3B
relation engine already consumes (`ImageGeometry` + `Component` + a uint8 label map), so the relation
engine itself does not change.

    load_tile(tile_id, split_view=...)
    list_instances(tile_id)
    get_instance_geometry(tile_id, tile_instance_id)
    get_source_feature_id(tile_id, tile_instance_id)
    iter_tiles(split, split_view=...)

The per-tile geometry cache is regenerated deterministically from the read-only archive on demand.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.whu_native_vector import (  # noqa: E402
    DATASET_ROOT,
    DATASET_VERSION,
    LEGACY_COMPAT_VIEW,
    SCENE_DISJOINT_VIEW,
    instance_rings,
    read_jsonl,
    read_tile_cache,
)

SPLIT_VIEWS = (SCENE_DISJOINT_VIEW, LEGACY_COMPAT_VIEW)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TileView:
    """Resolved tile metadata (from the committed tile index)."""

    tile_id: str
    grid_id: str
    source_raster: str
    legacy_category: str
    grid_row: int
    grid_column: int
    width: int
    height: int
    instance_count: int
    is_empty: bool
    split: str
    legacy_split: str | None
    record: dict


class NativeVectorDataset:
    """Read-only access to `WHU-EA-NativeVector` v1.0."""

    def __init__(self, root: Path = DATASET_ROOT) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.statistics = json.loads((self.root / "statistics.json").read_text(encoding="utf-8"))
        self._tiles: dict[str, dict] = {}
        for row in read_jsonl(self.root / "tiles" / "index.jsonl"):
            self._tiles[str(row["tile_id"])] = row
        self._splits: dict[str, dict[str, list[str]]] = {}
        for view in SPLIT_VIEWS:
            payload = json.loads((self.root / "splits" / f"{view}.json").read_text(encoding="utf-8"))
            self._splits[view] = {
                split: [str(tile) for tile in payload.get(split, [])] for split in SPLITS
            }
        self._instance_cache: dict[str, list[dict]] = {}

    # ---------------------------------------------------------------- tiles

    @property
    def dataset_version(self) -> str:
        return str(self.manifest["dataset_version"])

    @property
    def dataset_name(self) -> str:
        return str(self.manifest["dataset_name"])

    def iter_tiles(self, split: str, split_view: str = SCENE_DISJOINT_VIEW):
        """Yield `TileView` for every tile of one split under one split view."""

        if split_view not in self._splits:
            raise KeyError(f"unknown split view {split_view!r}")
        if split not in SPLITS:
            raise KeyError(f"unknown split {split!r}")
        for tile_id in self._splits[split_view][split]:
            view = self.load_tile(tile_id, split_view=split_view)
            if view is not None:
                yield view

    def load_tile(self, tile_id: str, split_view: str = SCENE_DISJOINT_VIEW) -> TileView | None:
        record = self._tiles.get(str(tile_id))
        if record is None:
            return None
        return TileView(
            tile_id=str(record["tile_id"]),
            grid_id=str(record["grid_id"]),
            source_raster=str(record["source_raster"]),
            legacy_category=str(record["legacy_category"]),
            grid_row=int(record["grid_row"]),
            grid_column=int(record["grid_column"]),
            width=int(record["width"]),
            height=int(record["height"]),
            instance_count=int(record.get("instance_count", 0)),
            is_empty=bool(record.get("is_empty", True)),
            split=str(record["scene_disjoint_split"]),
            legacy_split=record.get("legacy_compat_split"),
            record=record,
        )

    # ---------------------------------------------------------------- instances

    def list_instances(self, tile_id: str) -> list[dict]:
        """Per-instance scalar records for one tile (scalars only, no ring geometry)."""

        if str(tile_id) in self._instance_cache:
            return self._instance_cache[str(tile_id)]
        cache = read_tile_cache(str(tile_id))
        if cache is None:
            self._instance_cache[str(tile_id)] = []
            return []
        rows = []
        for position, instance_id in enumerate(cache["instance_ids"].tolist()):
            rows.append(
                {
                    "tile_id": str(tile_id),
                    "tile_instance_id": int(instance_id),
                    "source_feature_id": int(cache["source_feature_ids"][position]),
                    "bbox_xyxy_px": [int(v) for v in cache["bboxes"][position]],
                    "centroid_px": [float(v) for v in cache["centroids"][position]],
                    "clipped_area_px": int(cache["clipped_area"][position]),
                    "full_area_px": int(cache["full_area"][position]),
                    "visible_fraction": float(cache["visible_fraction"][position]),
                    "touches_tile_border": bool(cache["touches_border"][position]),
                    "tiny_area": bool(cache["tiny"][position]),
                    "multipart": bool(cache["multipart"][position]),
                    "n_holes": int(cache["n_holes"][position]),
                }
            )
        self._instance_cache[str(tile_id)] = rows
        return rows

    def get_instance_geometry(self, tile_id: str, tile_instance_id: int) -> list[dict] | None:
        """Pixel-space rings of one instance: `[{"kind": "outer"|"hole", "points": [[x, y], ...]}]`."""

        cache = read_tile_cache(str(tile_id))
        if cache is None:
            return None
        if int(tile_instance_id) not in {int(v) for v in cache["instance_ids"].tolist()}:
            return None
        return instance_rings(cache, int(tile_instance_id))

    def get_source_feature_id(self, tile_id: str, tile_instance_id: int) -> int | None:
        for row in self.list_instances(tile_id):
            if row["tile_instance_id"] == int(tile_instance_id):
                return int(row["source_feature_id"])
        return None

    def label_map(self, tile_id: str) -> np.ndarray | None:
        cache = read_tile_cache(str(tile_id))
        return None if cache is None else cache["label_map"]

    def instance_count(self, split: str, split_view: str = SCENE_DISJOINT_VIEW) -> int:
        return sum(view.instance_count for view in self.iter_tiles(split, split_view=split_view))

    def split_sizes(self, split_view: str = SCENE_DISJOINT_VIEW) -> dict:
        return {split: len(self._splits[split_view][split]) for split in SPLITS}


# ------------------------------------------------------------------ relation-engine bridge


def image_record_for_reasoning(
    dataset: NativeVectorDataset,
    tile_id: str,
    split: str,
    *,
    split_view: str = SCENE_DISJOINT_VIEW,
    label_map_path: Path | None = None,
) -> dict:
    """One metadata record in the exact schema `geometry.image_geometry_from_record` consumes.

    `component_id` is the tile-instance id; `component_map` is the workspace-relative path of the
    uint8 label map. Nothing else in the frozen reasoning stack has to change.
    """

    instances = dataset.list_instances(tile_id)
    view = dataset.load_tile(tile_id, split_view=split_view)
    if view is None:
        raise KeyError(f"unknown tile {tile_id}")
    image_area = float(view.width * view.height)
    components = []
    for row in instances:
        bbox = row["bbox_xyxy_px"]
        components.append(
            {
                "component_id": int(row["tile_instance_id"]),
                "source_polygon_index": int(row["source_feature_id"]),
                "area_px": int(row["clipped_area_px"]),
                "area_ratio": float(row["clipped_area_px"]) / image_area,
                "centroid_px": [float(row["centroid_px"][0]), float(row["centroid_px"][1])],
                "bbox_xyxy_px": [int(v) for v in bbox],
                "width_px": int(bbox[2] - bbox[0]),
                "height_px": int(bbox[3] - bbox[1]),
                "touches_image_border": bool(row["touches_tile_border"]),
                "continuous_polygon_area_px": float(row["clipped_area_px"]),
                "source_feature_id": int(row["source_feature_id"]),
                "visible_fraction": float(row["visible_fraction"]),
                "tiny_area": bool(row["tiny_area"]),
            }
        )
    record = {
        "image_id": str(tile_id),
        "split": str(split),
        "width": int(view.width),
        "height": int(view.height),
        "component_map": str(label_map_path).replace("\\", "/") if label_map_path else "",
        # The cropped imagery lives in the external read-only archive, which is outside the
        # repository AND outside the workspace, so an absolute path is forbidden by the path policy.
        # `image_path` is therefore the logical path relative to the runtime `--source-root`, and
        # `image_path_root` says so explicitly.
        "image_path": str(view.record.get("source_image_ref", "")),
        "image_path_root": "whu_source_root",
        "source_image_ref": str(view.record.get("source_image_ref", "")),
        "source_label_ref": str(view.record.get("source_label_ref", "")),
        "components": components,
        "source_raster": view.source_raster,
        "scene_disjoint_split": view.split,
        "legacy_compat_split": view.legacy_split,
        "native_vector_dataset_version": DATASET_VERSION,
        "split_view": split_view,
    }
    return record


def resolve_source_image(record: dict, source_root: Path) -> Path:
    """Absolute path of a record's source image, given the runtime source root."""

    return Path(source_root) / str(record["image_path"])


def validate_reasoning_record(record: dict, dataset: NativeVectorDataset) -> list[str]:
    """Validate one v0.2 reasoning record against the canonical dataset.

    Returns a list of problems (empty when the record is fully consistent). Used by the test suite to
    prove that the validator actually catches corrupted target provenance.
    """

    problems: list[str] = []
    tile_id = str(record.get("image_id", ""))
    if not tile_id:
        return ["missing image_id"]
    instances = {row["tile_instance_id"]: row for row in dataset.list_instances(tile_id)}
    if not instances and dataset.load_tile(tile_id) is None:
        return [f"unknown tile {tile_id}"]

    provenance = record.get("native_vector")
    if not isinstance(provenance, dict):
        return [f"{tile_id}: missing native_vector provenance"]

    def check(entry, role: str) -> None:
        if entry is None:
            return
        instance_id = entry.get("tile_instance_id")
        feature_id = entry.get("source_feature_id")
        row = instances.get(int(instance_id)) if instance_id is not None else None
        if row is None:
            problems.append(f"{tile_id}: {role} tile_instance_id {instance_id} does not exist")
            return
        if int(row["source_feature_id"]) != int(feature_id):
            problems.append(
                f"{tile_id}: {role} source_feature_id {feature_id} != recorded {row['source_feature_id']}"
            )

    check(provenance.get("target"), "target")
    for entry in provenance.get("references") or []:
        check(entry, "reference")
    geometry_ref = record.get("target_geometry_ref") or {}
    if geometry_ref:
        if str(geometry_ref.get("tile_id")) != tile_id:
            problems.append(f"{tile_id}: target_geometry_ref tile_id mismatch")
        check(
            {
                "tile_instance_id": geometry_ref.get("tile_instance_id"),
                "source_feature_id": geometry_ref.get("source_feature_id"),
            },
            "target_geometry_ref",
        )
    return problems


# ------------------------------------------------------------------ Task 6J bridge


def proposal_evaluation_interface(dataset: NativeVectorDataset, tile_id: str, proposals):
    """Task 6L section 20 bridge: score proposals against native-vector instance masks.

    `proposals` is any iterable of objects exposing `.mask` (bool array), `.bbox_xyxy_px` and
    `.area_px`. Returns per-proposal best IoU against the native instances plus the recall of native
    instances covered at 0.5. No model is trained or selected here.
    """

    native = []
    label_map = dataset.label_map(tile_id)
    if label_map is None:
        return None
    for row in dataset.list_instances(tile_id):
        native.append(label_map == int(row["tile_instance_id"]))

    def iou(a, b) -> float:
        a = np.asarray(a).astype(bool)
        b = np.asarray(b).astype(bool)
        union = int(np.logical_or(a, b).sum())
        return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)

    per_proposal = []
    for index, proposal in enumerate(proposals):
        mask = np.asarray(proposal.mask).astype(bool)
        best = 0.0
        best_id = None
        for row, native_mask in zip(dataset.list_instances(tile_id), native):
            value = iou(mask, native_mask)
            if value > best:
                best = value
                best_id = int(row["tile_instance_id"])
        per_proposal.append(
            {
                "proposal_index": index,
                "best_native_instance_id": best_id,
                "best_iou": best,
                "matched": bool(best >= 0.5),
                "area_px": int(getattr(proposal, "area_px", mask.sum())),
                "touches_image_border": bool(getattr(proposal, "touches_image_border", False)),
            }
        )
    covered = {entry["best_native_instance_id"] for entry in per_proposal if entry["matched"]}
    return {
        "tile_id": tile_id,
        "native_instances": len(native),
        "proposals": len(per_proposal),
        "native_recall_at_0_5": len(covered) / len(native) if native else None,
        "per_proposal": per_proposal,
    }


def candidate_set_for_tile(dataset: NativeVectorDataset, tile_id: str, source: str = "native_vector"):
    """Build a Task 6J `CandidateSet` from native instances (masks from the label map)."""

    from buildreasonseg_mvp.structured_grounding import Candidate, CandidateSet

    label_map = dataset.label_map(tile_id)
    if label_map is None:
        return None
    instances = dataset.list_instances(tile_id)
    candidates = []
    for row in instances:
        instance_id = int(row["tile_instance_id"])
        mask = label_map == instance_id
        bbox = row["bbox_xyxy_px"]
        candidates.append(
            Candidate(
                candidate_id=instance_id,
                mask=mask,
                bbox_xyxy_px=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                centroid_px=(float(row["centroid_px"][0]), float(row["centroid_px"][1])),
                area_px=int(row["clipped_area_px"]),
                touches_image_border=bool(row["touches_tile_border"]),
                confidence=None,
                source=source,
            )
        )
    height, width = label_map.shape
    return CandidateSet(
        width=int(width), height=int(height), candidates=candidates, label_map=label_map, source=source
    )
