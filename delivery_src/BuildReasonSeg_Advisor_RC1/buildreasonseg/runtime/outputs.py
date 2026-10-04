"""Output writing: mask / overlay / diagnostics / result.json (Task 8B sections 23-25).

Frozen rules:

* mask → `inference/output/masks/<stem>_mask.png`, original width×height, uint8, background 0, target 255;
* overlay → `inference/output/overlays/<stem>_overlay.png`, original size, red `(255,0,0)`, alpha default 0.45;
* diagnostics → `inference/output/diagnostics/<sample>/` with the frozen file names;
* existing outputs are never overwritten silently: `_001`, `_002`, … suffixes are used and mask/overlay/
  diagnostics share the same run suffix;
* no ground truth is ever written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from buildreasonseg import paths
from buildreasonseg.runtime.imageio import save_png

DIAGNOSTIC_NAMES = ("prompt.txt", "parsed_program.json", "global_proposals.png", "proposals.json",
                    "selected_reference.png", "reasoning_context.png", "reference_context_mask.png",
                    "direction_field.png", "nearest_field.png", "relation_weight.png",
                    "prototype_similarity.png", "maps.npz", "result.json")
PREVIEW_MAX_DIMENSION = 2048


def output_dirs() -> dict:
    base = paths.inference_dir()
    return {"masks": base / "output" / "masks", "overlays": base / "output" / "overlays",
            "diagnostics": base / "output" / "diagnostics"}


def sample_slug(image: Path) -> str:
    return image.stem


def allocate_run_suffix(directory: Path, stem: str, suffix: str = "_mask") -> int:
    """Return the first free `_NNN` index (0 means the unsuffixed name is free)."""

    if not (directory / f"{stem}{suffix}.png").exists():
        return 0
    index = 1
    while (directory / f"{stem}{suffix}_{index:03d}.png").exists():
        index += 1
    return index


def suffix_of(index: int) -> str:
    return "" if index == 0 else f"_{index:03d}"


def _normalise_for_preview(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-12:
        return np.zeros(array.shape, dtype=np.uint8)
    return ((array - low) / (high - low) * 255.0).astype(np.uint8)


@dataclass
class SampleOutputs:
    stem: str
    suffix_index: int
    mask_path: str | None = None
    overlay_path: str | None = None
    diagnostics_dir: str | None = None
    written: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"mask": self.mask_path, "overlay": self.overlay_path,
                "diagnostics": self.diagnostics_dir, "written": list(self.written)}


def allocate_outputs(image: Path) -> SampleOutputs:
    stem = sample_slug(image)
    dirs = output_dirs()
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    index = allocate_run_suffix(dirs["masks"], stem)
    suffix = suffix_of(index)
    diagnostics = dirs["diagnostics"] / f"{stem}{suffix}"
    return SampleOutputs(stem=stem, suffix_index=index,
                         mask_path=str(dirs["masks"] / f"{stem}_mask{suffix}.png"),
                         overlay_path=str(dirs["overlays"] / f"{stem}_overlay{suffix}.png"),
                         diagnostics_dir=str(diagnostics))


def save_final_outputs(outputs: SampleOutputs, rgb: np.ndarray, mask: np.ndarray, *,
                       alpha: float = 0.45) -> dict:
    """Write the final mask and overlay (only called when the sample succeeded)."""

    from buildreasonseg.runtime.imageio import overlay as overlay_fn

    height, width = rgb.shape[:2]
    if mask.shape != (height, width):
        raise ValueError(f"mask shape {mask.shape} does not match image {(height, width)}")
    mask_uint8 = (mask.astype(bool).astype(np.uint8) * 255)
    save_png(outputs.mask_path, mask_uint8)
    save_png(outputs.overlay_path, overlay_fn(rgb, mask, alpha=alpha))
    outputs.written.extend([outputs.mask_path, outputs.overlay_path])
    return {"mask": outputs.mask_path, "overlay": outputs.overlay_path,
            "mask_area": int(mask.sum()), "mask_foreground_values": [0, 255]}


def save_diagnostics(outputs: SampleOutputs, payload: dict, *, image: np.ndarray | None = None,
                     proposals_preview: np.ndarray | None = None,
                     reference_preview: np.ndarray | None = None,
                     context_rgb: np.ndarray | None = None,
                     reference_context_mask: np.ndarray | None = None,
                     direction_field: np.ndarray | None = None,
                     nearest_field: np.ndarray | None = None,
                     relation_weight: np.ndarray | None = None,
                     prototype_similarity: np.ndarray | None = None,
                     maps: dict | None = None) -> str:
    """Write every available diagnostics artifact; missing stages are simply skipped."""

    directory = Path(outputs.diagnostics_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs.written.append(str(directory))

    def _write_text(name: str, text: str) -> None:
        target = directory / name
        target.write_text(text, encoding="utf-8")
        outputs.written.append(str(target))

    _write_text("prompt.txt", str(payload.get("prompt", "")))
    _write_text("parsed_program.json",
                json.dumps(payload.get("parsed", {}), ensure_ascii=False, indent=1))
    _write_text("proposals.json",
                json.dumps(payload.get("proposals", {}), ensure_ascii=False, indent=1))
    _write_text("result.json", json.dumps(payload, ensure_ascii=False, indent=1))

    if proposals_preview is not None:
        save_png(directory / "global_proposals.png", proposals_preview)
        outputs.written.append(str(directory / "global_proposals.png"))
    if reference_preview is not None:
        save_png(directory / "selected_reference.png", reference_preview)
        outputs.written.append(str(directory / "selected_reference.png"))
    if context_rgb is not None:
        save_png(directory / "reasoning_context.png", context_rgb)
        outputs.written.append(str(directory / "reasoning_context.png"))
    if reference_context_mask is not None:
        save_png(directory / "reference_context_mask.png",
                 reference_context_mask.astype(np.uint8) * 255)
        outputs.written.append(str(directory / "reference_context_mask.png"))
    for name, array in (("direction_field.png", direction_field),
                        ("nearest_field.png", nearest_field),
                        ("relation_weight.png", relation_weight),
                        ("prototype_similarity.png", prototype_similarity)):
        if array is not None:
            save_png(directory / name, _normalise_for_preview(array))
            outputs.written.append(str(directory / name))
    if maps:
        target = directory / "maps.npz"
        np.savez_compressed(target, **{key: np.asarray(value, dtype=np.float32)
                                       for key, value in maps.items()})
        outputs.written.append(str(target))
    return str(directory)


def proposals_preview_image(rgb: np.ndarray, proposals, *, selected_id: int | None = None,
                            max_dimension: int = PREVIEW_MAX_DIMENSION) -> np.ndarray:
    """Preview with `#ID` markers, bbox and mask outline (diagnostics only, aspect preserved)."""

    from buildreasonseg.runtime.imageio import resize_preview

    preview = rgb.copy()
    for proposal in proposals:
        top, left, bottom, right = proposal.global_bbox
        colour = (0, 255, 0) if proposal.proposal_id == selected_id else (255, 255, 0)
        for row in (max(0, top), min(rgb.shape[0] - 1, bottom)):
            preview[row, max(0, left):min(rgb.shape[1], right + 1)] = colour
        for column in (max(0, left), min(rgb.shape[1] - 1, right)):
            preview[max(0, top):min(rgb.shape[0], bottom + 1), column] = colour
        outline_crop = proposal.mask_crop & ~_erode(proposal.mask_crop)
        if outline_crop.any():
            outline_rows, outline_cols = np.nonzero(outline_crop)
            global_rows = np.clip(outline_rows + top, 0, rgb.shape[0] - 1)
            global_cols = np.clip(outline_cols + left, 0, rgb.shape[1] - 1)
            preview[global_rows, global_cols] = (255, 0, 255)
        row = int(np.clip(proposal.centroid[0], 0, rgb.shape[0] - 1))
        column = int(np.clip(proposal.centroid[1], 0, rgb.shape[1] - 1))
        preview[max(0, row - 3):row + 4, max(0, column - 3):column + 4] = (0, 0, 255)
    return resize_preview(preview, max_dimension)


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
            & padded[1:-1, :-2] & padded[1:-1, 2:])


__all__ = ["DIAGNOSTIC_NAMES", "PREVIEW_MAX_DIMENSION", "SampleOutputs", "allocate_outputs",
           "allocate_run_suffix", "output_dirs", "proposals_preview_image", "sample_slug",
           "save_diagnostics", "save_final_outputs", "suffix_of"]
