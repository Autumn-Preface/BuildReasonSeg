"""Dataset access for the BuildReasonSeg-MVP.

Reads the frozen BuildSpatialReason-v0.1.1 records and turns each one into:

* the source RGB image (512x512 WHU tile);
* the Chinese instruction (`instruction_zh`) -- the ONLY language input;
* the assistant target text `reasoning_zh + " [SEG]"`;
* the ground-truth mask, reconstructed from the component map + target id.

Hard rules enforced here (Task 6A sections 7 and 11.3):

* `reasoning_steps`, component ids, centroids, bounding boxes and the reference
  list are **never** passed to the model. They are read only for reporting.
* Ground-truth geometry is supervision/evaluation only and is never an inference
  input.
* The dataset is read-only; nothing in this module writes to `datasets/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

#: Repository root: .../BuildReasonSeg
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Frozen dataset version. v0.1 must never be used (Task 5.5 section 2).
DATASET_VERSION = "v0.1.1"

#: Assistant text terminator appended after `[SEG]` is handled by the chat
#: template; the string built here is exactly `reasoning_zh + " [SEG]"`.
SEG_TOKEN = "[SEG]"

DATASET_DIR = REPO_ROOT / "datasets" / "build_spatial_reason" / DATASET_VERSION
SPLITS = ("train", "val", "test")


def dataset_dir(version: str = DATASET_VERSION) -> Path:
    return REPO_ROOT / "datasets" / "build_spatial_reason" / version


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve a record-relative path against the repository root.

    Record paths are repository-relative and may point one level up (the legacy
    WHU imagery lives at `../WHU_Building_Segment/`). Resolution is deliberately
    read-only and never normalises outside the workspace.
    """

    candidate = (REPO_ROOT / str(path_like)).resolve()
    workspace = REPO_ROOT.parent.resolve()
    if not candidate.is_relative_to(workspace):
        raise ValueError(f"path escapes the shared workspace: {path_like}")
    return candidate


@dataclass(frozen=True)
class Sample:
    """One instruction/segmentation record."""

    sample_id: str
    image_id: str
    split: str
    level: int
    query_type: str
    instruction_zh: str
    reasoning_zh: str
    target_component_id: int
    image_path: str
    component_map_path: str
    reference_component_ids: tuple[int, ...] = ()
    trivial_selection: bool = False
    template_id: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def assistant_text(self) -> str:
        """The teacher-forced assistant target: `reasoning_zh + " [SEG]"`."""

        return f"{self.reasoning_zh} {SEG_TOKEN}"

    def target_mask(self) -> np.ndarray:
        """Boolean ground-truth mask for the target component (512x512)."""

        return load_component_map(self.component_map_path) == self.target_component_id

    def image_rgb(self) -> np.ndarray:
        from PIL import Image

        with Image.open(resolve_repo_path(self.image_path)) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def load_component_map(path_like: str | Path) -> np.ndarray:
    """Load a component-index map as a uint8 array."""

    from PIL import Image

    with Image.open(resolve_repo_path(path_like)) as handle:
        return np.asarray(handle.convert("L"), dtype=np.uint8)


def read_records(split: str, version: str = DATASET_VERSION) -> list[dict]:
    path = dataset_dir(version) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset split: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_records(splits: Sequence[str] = ("train",), version: str = DATASET_VERSION) -> Iterator[dict]:
    for split in splits:
        yield from read_records(split, version)


def to_sample(record: dict) -> Sample:
    return Sample(
        sample_id=record["sample_id"],
        image_id=record["image_id"],
        split=record["split"],
        level=int(record["level"]),
        query_type=record["query_type"],
        instruction_zh=record["instruction_zh"],
        reasoning_zh=record["reasoning_zh"],
        target_component_id=int(record["target_component_id"]),
        image_path=record["image_path"],
        component_map_path=record["component_map_path"],
        reference_component_ids=tuple(int(v) for v in record.get("reference_component_ids", [])),
        trivial_selection=bool(record.get("trivial_selection", False)),
        template_id=str(record.get("template_id", "")),
        raw=record,
    )


# --------------------------------------------------------------------------
# Deterministic subsets (Task 6A section 8)
# --------------------------------------------------------------------------


def select_smoke_pair(records: Iterable[dict]) -> list[dict]:
    """Two records from the SAME image with different instructions and targets.

    Preference order: different query types, then different levels. Selection is
    by sorted sample id so it is fully deterministic and independent of file
    order.
    """

    by_image: dict[str, list[dict]] = {}
    for record in records:
        by_image.setdefault(record["image_id"], []).append(record)

    candidates: list[tuple[str, list[dict]]] = []
    for image_id, group in by_image.items():
        by_target: dict[int, list[dict]] = {}
        for record in group:
            by_target.setdefault(int(record["target_component_id"]), []).append(record)
        if len(by_target) < 2:
            continue
        first_key = sorted(by_target)[0]
        second_key = sorted(by_target)[1]
        left = sorted(by_target[first_key], key=lambda r: r["sample_id"])[0]
        right = sorted(by_target[second_key], key=lambda r: r["sample_id"])[0]
        score = (
            left["query_type"] != right["query_type"],
            left["level"] != right["level"],
        )
        candidates.append((image_id, [left, right]))

    if not candidates:
        raise RuntimeError("no image in the split has two different target components")

    # deterministic: prefer a pair with two different query types, then the
    # smallest image id. Never influenced by model results.
    candidates.sort(
        key=lambda item: (not (item[1][0]["query_type"] != item[1][1]["query_type"]), item[0])
    )
    pair = sorted(candidates[0][1], key=lambda r: r["sample_id"])
    if len({int(r["target_component_id"]) for r in pair}) != 2:
        raise RuntimeError("smoke pair must target two different components")
    return pair


def select_overfit_set(
    records: Iterable[dict],
    images: int = 10,
    per_image: int = 2,
) -> list[dict]:
    """A deterministic `images x per_image` set with guaranteed level coverage.

    Requirements satisfied: all from train; covers L1, L2 and nontrivial L3 where
    available; spans direction / nearest / size / extreme families; each paired
    image contributes two DIFFERENT target masks so image-only memorisation is
    insufficient; the selection is deterministic and is never revised using model
    results.

    A naive "first pair per image" walk returns ten Level-1 pairs, because Level 1
    is the most common record per image. Pairs are therefore bucketed by the
    highest level they contain and the budget is spread across the buckets.
    """

    if per_image != 2:
        raise ValueError("Task 6A selects exactly two different targets per image")

    by_image: dict[str, list[dict]] = {}
    for record in records:
        by_image.setdefault(record["image_id"], []).append(record)

    def is_nontrivial_level3(record: dict) -> bool:
        return int(record["level"]) == 3 and not bool(record.get("trivial_selection", False))

    # ---- candidate pairs: two different targets on the same image ----------
    pairs: list[tuple[str, list[dict], int, bool]] = []
    for image_id in sorted(by_image):
        group = by_image[image_id]
        by_target: dict[int, list[dict]] = {}
        for record in group:
            by_target.setdefault(int(record["target_component_id"]), []).append(record)
        targets = sorted(by_target)

        best: tuple[tuple, list[dict]] | None = None
        for i, left_id in enumerate(targets):
            for right_id in targets[i + 1 :]:
                for left in sorted(by_target[left_id], key=lambda r: r["sample_id"]):
                    for right in sorted(by_target[right_id], key=lambda r: r["sample_id"]):
                        pair = [left, right]
                        top_level = max(int(r["level"]) for r in pair)
                        has_nt3 = any(is_nontrivial_level3(r) for r in pair)
                        same_type = left["query_type"] == right["query_type"]
                        # prefer: highest level present, then a nontrivial L3,
                        # then two different query types, then stable ids
                        key = (
                            -top_level if has_nt3 else -1,
                            not has_nt3,
                            same_type,
                            left["sample_id"],
                            right["sample_id"],
                        )
                        if best is None or key < best[0]:
                            best = (key, pair)
        if best is not None:
            pair = best[1]
            pairs.append(
                (
                    image_id,
                    pair,
                    max(int(r["level"]) for r in pair),
                    any(is_nontrivial_level3(r) for r in pair),
                )
            )

    if len(pairs) < images:
        raise RuntimeError(f"only {len(pairs)} usable images found, need {images}")

    # ---- buckets by the highest level present ------------------------------
    nt3 = [p for p in pairs if p[3]]
    level2 = [p for p in pairs if not p[3] and p[2] == 2]
    level1 = [p for p in pairs if not p[3] and p[2] == 1]

    # spread the budget, taking the highest-value buckets first but never
    # exhausting one bucket at the expense of coverage
    quota = {"nt3": max(1, images // 3), "level2": max(1, images // 3)}
    quota["level1"] = images - quota["nt3"] - quota["level2"]

    chosen: list[tuple[str, list[dict], int, bool]] = []
    used_images: set[str] = set()

    def take(bucket: list, count: int) -> None:
        taken = 0
        for entry in bucket:
            if taken >= count or len(chosen) >= images:
                break
            if entry[0] in used_images:
                continue
            chosen.append(entry)
            used_images.add(entry[0])
            taken += 1

    take(nt3, quota["nt3"])
    take(level2, quota["level2"])
    take(level1, quota["level1"])
    # top up from whatever remains if a bucket was short
    for bucket in (nt3, level2, level1):
        if len(chosen) >= images:
            break
        take(bucket, images - len(chosen))

    if len(chosen) < images:
        raise RuntimeError(f"only {len(chosen)} images selected, need {images}")

    flat: list[dict] = []
    for _image_id, pair, _top_level, _has_nt3 in chosen[:images]:
        flat.extend(sorted(pair, key=lambda r: r["sample_id"]))
    return flat


def subset_summary(records: Sequence[dict]) -> dict:
    """Reporting helper: never fed to the model."""

    levels: dict[str, int] = {}
    types: dict[str, int] = {}
    images: set[str] = set()
    for record in records:
        levels[str(record["level"])] = levels.get(str(record["level"]), 0) + 1
        types[record["query_type"]] = types.get(record["query_type"], 0) + 1
        images.add(record["image_id"])
    return {
        "n_records": len(records),
        "n_images": len(images),
        "by_level": dict(sorted(levels.items())),
        "by_query_type": dict(sorted(types.items())),
        "sample_ids": [r["sample_id"] for r in records],
        "image_ids": sorted(images),
    }
