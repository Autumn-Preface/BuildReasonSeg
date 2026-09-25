"""Safe, read-only access to the generated component representation.

All paths stored in the metadata are **relative to the repository root**, e.g.::

    datasets/whu/components/train/1_0.png
    ../WHU_Building_Segment/dataset/WHU_YOLO_dataset/images/train/1_0.tif

This module resolves them and refuses anything that escapes the shared
workspace, so an accidentally absolute or traversing path can never be read.

The legacy dataset is read-only. Nothing here writes to it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Repository root: .../BuildReasonSeg
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Shared workspace root, one level above this repository.
WORKSPACE_ROOT = REPO_ROOT.parent

#: Default location of the generated component representation.
DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "whu"

SPLITS = ("train", "val", "test")


class PathEscapeError(RuntimeError):
    """Raised when a stored path would resolve outside the shared workspace."""


def resolve(path_like: str | Path) -> Path:
    """Resolve a stored path (repository-relative) to an absolute path.

    Raises :class:`PathEscapeError` if the result leaves the workspace, and
    rejects absolute paths outright.
    """

    raw = str(path_like)
    if raw.startswith(("/", "\\")) or (len(raw) > 1 and raw[1] == ":"):
        raise PathEscapeError(f"absolute paths are not permitted in metadata: {raw}")

    candidate = (REPO_ROOT / raw).resolve()
    workspace = WORKSPACE_ROOT.resolve()
    if not candidate.is_relative_to(workspace):
        raise PathEscapeError(f"path escapes the workspace: {raw} -> {candidate}")
    return candidate


def read_component_map(dataset_root: Path, path_like: str | Path) -> np.ndarray:
    """Read a component-index map as a uint8 array.

    ``path_like`` is the ``component_map`` value from the metadata, i.e.
    repository-relative.
    """

    from PIL import Image

    path = resolve(path_like)
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array


def read_npz_archive(dataset_root: Path, path_like: str | Path) -> dict:
    """Load a polygon provenance archive."""

    path = resolve(path_like)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def polygon_for_component(
    archive: dict, component_index: int
) -> np.ndarray:
    """Extract one component's polygon vertices from a loaded archive."""

    offsets = archive["offsets"]
    lengths = archive["lengths"]
    start = int(offsets[component_index])
    length = int(lengths[component_index])
    return archive["vertices"][start:start + length]
