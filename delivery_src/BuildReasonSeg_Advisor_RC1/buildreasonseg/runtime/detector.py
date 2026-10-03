"""Detector runtime: 512-px source-space tiled detection, global merge and reference eligibility.

Frozen engineering policy (Task 8B sections 11-13):

* `<=512` both dimensions → reflection pad to 512, never resize;
* otherwise sliding 512 px tiles with 128 px overlap (stride 384), last row/column clamped to the border;
* every tile runs the frozen U-C1 detector (imgsz 640 / conf 0.05 / max_det 300 / default NMS / no TTA);
* masks map back to original source pixels, pad region is cropped away;
* duplicates are merged at global-mask IoU >= 0.50 by *keeping one proposal* (never a union);
* stable global ids are assigned by `centroid_y, centroid_x, mask_area`;
* eligibility is the frozen U-C1 rule (non-empty, not touching the **original image** border,
  bbox extent ratio <= 0.20) with tie-break area → confidence → global id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from buildreasonseg import paths
from buildreasonseg.errors import BuildReasonSegError
from buildreasonseg.runtime.imageio import pad_to_512, reflection_pad

TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP
IMGSZ = 640
CONF = 0.05
MAX_DET = 300
DUPLICATE_IOU = 0.50
MERGE_BBOX_EXTENT_RATIO_MAX = 0.20
SMALLEST_MIN_AREA_PX = 150
FROZEN_THRESHOLD = 0.5  # ultralytics mask binarisation for the frozen U-C1 detector


@dataclass
class TileWindow:
    source_tile_id: str
    top: int
    left: int
    size: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    @property
    def valid_height(self) -> int:
        return self.size - self.pad_top - self.pad_bottom

    @property
    def valid_width(self) -> int:
        return self.size - self.pad_left - self.pad_right


def plan_tiles(width: int, height: int, *, tile_size: int = TILE_SIZE,
               overlap: int = TILE_OVERLAP) -> list[TileWindow]:
    """Deterministic tile plan covering the whole image, last row/column clamped to the border."""

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    if width <= tile_size and height <= tile_size:
        padded, padding = pad_to_512(np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8),
                                     tile_size)
        return [TileWindow(source_tile_id="tile_0000_r000_c000", top=0, left=0, size=tile_size,
                           pad_top=padding["top"], pad_bottom=padding["bottom"],
                           pad_left=padding["left"], pad_right=padding["right"])]
    tops = list(range(0, max(height - tile_size, 0) + 1, stride)) or [0]
    lefts = list(range(0, max(width - tile_size, 0) + 1, stride)) or [0]
    if tops[-1] != max(height - tile_size, 0):
        tops.append(max(height - tile_size, 0))
    if lefts[-1] != max(width - tile_size, 0):
        lefts.append(max(width - tile_size, 0))
    windows = []
    for row, top in enumerate(sorted(set(tops))):
        for column, left in enumerate(sorted(set(lefts))):
            windows.append(TileWindow(source_tile_id=f"tile_{row * 100 + column:04d}_r{row:03d}_c{column:03d}",
                                      top=int(top), left=int(left), size=tile_size, pad_top=0,
                                      pad_bottom=0, pad_left=0, pad_right=0))
    return windows


def extract_tile(rgb: np.ndarray, window: TileWindow) -> tuple[np.ndarray, dict]:
    """Extract the tile window in source pixels, reflection-padding outside the image."""

    height, width = rgb.shape[:2]
    top, left, size = window.top, window.left, window.size
    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, (top + size) - height)
    pad_right = max(0, (left + size) - width)
    window_array = rgb[max(0, top):min(height, top + size), max(0, left):min(width, left + size)]
    if pad_top or pad_bottom or pad_left or pad_right:
        window_array = reflection_pad(window_array, pad_top, pad_bottom, pad_left, pad_right)
    if window_array.shape[0] != size or window_array.shape[1] != size:
        window_array = np.pad(window_array, [(0, max(0, size - window_array.shape[0])),
                                             (0, max(0, size - window_array.shape[1])), (0, 0)],
                              mode="edge")
    padding = {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right,
               "applied": bool(pad_top or pad_bottom or pad_left or pad_right)}
    return np.ascontiguousarray(window_array[:size, :size]), padding


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    top, bottom = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    left, right = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return top, left, bottom, right


def centroid_of(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return (0.0, 0.0)
    return (float(rows.mean()), float(cols.mean()))


@dataclass
class GlobalProposal:
    proposal_id: int
    source_tile_id: str
    tile_index: int
    confidence: float
    global_mask: np.ndarray
    global_bbox: tuple[int, int, int, int]
    mask_area: int
    touches_image_border: bool
    border_clearance: float
    centroid: tuple[float, float]
    raw_index: int
    pad_mask_empty: bool = False

    @property
    def bbox_extent_ratio(self) -> float:
        top, left, bottom, right = self.global_bbox
        return max((bottom - top + 1), (right - left + 1)) / float(TILE_SIZE)

    def to_dict(self) -> dict:
        top, left, bottom, right = self.global_bbox
        return {"proposal_id": self.proposal_id, "source_tile_id": self.source_tile_id,
                "confidence": float(self.confidence), "mask_area": int(self.mask_area),
                "global_bbox": [int(top), int(left), int(bottom), int(right)],
                "centroid": [float(self.centroid[0]), float(self.centroid[1])],
                "touches_image_border": bool(self.touches_image_border),
                "border_clearance": float(self.border_clearance),
                "bbox_extent_ratio": float(self.bbox_extent_ratio),
                "raw_index": int(self.raw_index)}


def border_clearance(mask: np.ndarray) -> float:
    """Distance in px from the mask to the nearest image border (min over the four sides)."""

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return 0.0
    height, width = mask.shape
    return float(min(rows.min(), cols.min(), height - 1 - rows.max(), width - 1 - cols.max()))


def touches_original_border(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def iou_of(first: np.ndarray, second: np.ndarray) -> float:
    intersection = float(np.logical_and(first, second).sum())
    union = float(np.logical_or(first, second).sum())
    if union <= 0:
        return 0.0
    return intersection / union


class DetectorRuntime:
    """Frozen U-C1 YOLO26m-seg runtime loaded from the delivery model package."""

    def __init__(self, checkpoint: Path | None = None, device: str = "cuda", *,
                 imgsz: int = IMGSZ, conf: float = CONF, max_det: int = MAX_DET) -> None:
        self.checkpoint = Path(checkpoint or (paths.default_model_dir() / "detector.pt"))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.max_det = max_det
        self._model = None
        self.load_seconds: float | None = None

    def load(self):
        if self._model is not None:
            return self._model
        import time

        from ultralytics import YOLO

        if not self.checkpoint.is_file():
            raise BuildReasonSegError("E301", detail=f"detector 权重缺失: {self.checkpoint}")
        started = time.time()
        model = YOLO(str(self.checkpoint))
        try:
            model.to(self.device)
        except Exception:
            pass
        self._model = model
        self.load_seconds = round(time.time() - started, 2)
        return model

    def detect_tile(self, tile_rgb: np.ndarray) -> list[dict]:
        """One tile → raw proposals with tile-local masks at 512×512."""

        import torch

        model = self.load()
        with torch.no_grad():
            results = model.predict(source=tile_rgb, imgsz=self.imgsz, conf=self.conf,
                                    max_det=self.max_det, verbose=False, retina_masks=False,
                                    device=self.device if self.device != "cpu" else "cpu")
        output = []
        if not results:
            return output
        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return output
        masks = result.masks.data.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        for index in range(masks.shape[0]):
            mask = masks[index] > FROZEN_THRESHOLD
            if mask.shape != (TILE_SIZE, TILE_SIZE):
                mask = _resize_bool(mask, TILE_SIZE)
            output.append({"index": index, "confidence": float(confidences[index]),
                           "mask": mask, "box": [float(value) for value in boxes[index]]})
        # ultralytics returns detections in confidence order already; keep the order stable anyway
        output.sort(key=lambda entry: (-entry["confidence"], entry["index"]))
        return output

    def detect_global(self, rgb: np.ndarray) -> dict:
        """Tiled global detection → merged proposals in original source-pixel coordinates."""

        import time

        height, width = rgb.shape[:2]
        windows = plan_tiles(width, height)
        height_px, width_px = height, width
        accumulated: list[dict] = []
        raw_count = 0
        started = time.time()
        for window in windows:
            tile_rgb, padding = extract_tile(rgb, window)
            detections = self.detect_tile(tile_rgb)
            raw_count += len(detections)
            for detection in detections:
                mask = detection["mask"]
                if padding["applied"]:
                    keep_top = padding["top"]
                    keep_left = padding["left"]
                    mask = mask[keep_top:keep_top + window.size - padding["top"] - padding["bottom"],
                                keep_left:keep_left + window.size - padding["left"] - padding["right"]]
                    if mask.size == 0:
                        continue
                global_mask = np.zeros((height_px, width_px), dtype=bool)
                top = max(0, window.top)
                left = max(0, window.left)
                bottom = min(height_px, top + mask.shape[0])
                right = min(width_px, left + mask.shape[1])
                if bottom <= top or right <= left:
                    continue
                global_mask[top:bottom, left:right] = mask[:bottom - top, :right - left]
                if not global_mask.any():
                    continue
                accumulated.append({"source_tile_id": window.source_tile_id, "tile_index": window.top
                                    * 10000 + window.left, "confidence": detection["confidence"],
                                    "mask": global_mask, "raw_index": detection["index"],
                                    "tile_top": window.top, "tile_left": window.left})
        merged, groups = merge_proposals(accumulated)
        return {"windows": windows, "raw_count": raw_count, "merged": merged, "groups": groups,
                "detector_seconds": round(time.time() - started, 3),
                "tile_size": TILE_SIZE, "overlap": TILE_OVERLAP, "stride": TILE_STRIDE,
                "imgsz": self.imgsz, "conf": self.conf, "max_det": self.max_det,
                "image_size": [width, height]}


def _resize_bool(mask: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    resized = Image.fromarray((mask.astype(np.uint8) * 255)).resize((size, size), Image.NEAREST)
    return np.asarray(resized) > 127


def merge_proposals(entries: list[dict]) -> tuple[list[GlobalProposal], list[list[int]]]:
    """Duplicate groups at global-mask IoU >= 0.50; keep exactly one proposal per group (no union)."""

    proposals: list[GlobalProposal] = []
    for entry in entries:
        mask = entry["mask"]
        box = bbox_of(mask)
        if box is None:
            continue
        proposals.append(GlobalProposal(
            proposal_id=-1, source_tile_id=entry["source_tile_id"], tile_index=entry["tile_index"],
            confidence=float(entry["confidence"]), global_mask=mask, global_bbox=box,
            mask_area=int(mask.sum()), touches_image_border=touches_original_border(mask),
            border_clearance=border_clearance(mask), centroid=centroid_of(mask),
            raw_index=int(entry["raw_index"])))

    parent = list(range(len(proposals)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)

    for first in range(len(proposals)):
        for second in range(first + 1, len(proposals)):
            if iou_of(proposals[first].global_mask, proposals[second].global_mask) >= DUPLICATE_IOU:
                union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(len(proposals)):
        groups.setdefault(find(index), []).append(index)

    def priority(index: int) -> tuple:
        proposal = proposals[index]
        return (proposal.touches_image_border, -proposal.border_clearance, -proposal.mask_area,
                -proposal.confidence, proposal.source_tile_id, proposal.raw_index)

    winners: list[int] = []
    group_lists: list[list[int]] = []
    for members in groups.values():
        ordered = sorted(members, key=priority)
        winners.append(ordered[0])
        group_lists.append([proposals[index].source_tile_id for index in ordered])

    selected = [proposals[index] for index in winners]
    selected.sort(key=lambda proposal: (round(proposal.centroid[0], 6), round(proposal.centroid[1], 6),
                                        proposal.mask_area))
    for stable_id, proposal in enumerate(selected):
        proposal.proposal_id = stable_id
    return selected, group_lists


def eligible(proposal: GlobalProposal, *, family: str = "largest") -> bool:
    """Frozen U-C1 eligibility: non-empty, not touching the original image border, extent <= 0.20."""

    if proposal.mask_area <= 0 or not proposal.global_mask.any():
        return False
    if proposal.touches_image_border:
        return False
    if proposal.bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX:
        return False
    if family == "smallest" and proposal.mask_area < SMALLEST_MIN_AREA_PX:
        return False
    return True


def eligible_proposals(proposals: list[GlobalProposal], *, family: str = "largest"
                       ) -> list[GlobalProposal]:
    return [proposal for proposal in proposals if eligible(proposal, family=family)]


def select_reference(proposals: list[GlobalProposal], *, family: str = "largest") -> GlobalProposal | None:
    """Largest eligible building: area desc → confidence desc → global id asc."""

    candidates = eligible_proposals(proposals, family=family)
    if not candidates:
        return None
    return sorted(candidates, key=lambda proposal: (-proposal.mask_area, -proposal.confidence,
                                                    proposal.proposal_id))[0]


def proposal_by_id(proposals: list[GlobalProposal], reference_id: int) -> GlobalProposal | None:
    for proposal in proposals:
        if proposal.proposal_id == reference_id:
            return proposal
    return None


__all__ = ["CONF", "DUPLICATE_IOU", "DetectorRuntime", "FROZEN_THRESHOLD", "GlobalProposal", "IMGSZ",
           "MAX_DET", "MERGE_BBOX_EXTENT_RATIO_MAX", "SMALLEST_MIN_AREA_PX", "TILE_OVERLAP",
           "TILE_SIZE", "TILE_STRIDE", "TileWindow", "bbox_of", "border_clearance", "centroid_of",
           "eligible", "eligible_proposals", "extract_tile", "iou_of", "merge_proposals",
           "plan_tiles", "proposal_by_id", "select_reference", "touches_original_border"]
