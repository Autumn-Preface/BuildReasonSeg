#!/usr/bin/env python
"""Build the BuildSpatialReason-v0.1 validation visualization pack.

Draws, per sample:
  * source image;
  * reference outline;
  * target outline;
  * candidate ids as an AUDIT OVERLAY ONLY (they are not part of the dataset's
    natural-language task, and the source image does not display them);
  * the instruction and query_type.

Also emits ``contact_sheet.png`` tiling the whole pack.

Kept deliberately small (<10 MB) and never claims manual inspection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "spatial_reasoning"))

import geometry as G  # noqa: E402
import thresholds as T  # noqa: E402

DATASET_ROOT = _REPO_ROOT / "datasets" / "whu"
GEN_DIR = _REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.1"
SPLITS = ("train", "val", "test")

THUMB = 384
CONTACT_COLUMNS = 6


def _load_records(limit: int | None, version: str = "v0.1") -> list[dict]:
    records: list[dict] = []
    gen_dir = _REPO_ROOT / "datasets" / "build_spatial_reason" / version
    for split in SPLITS:
        path = gen_dir / f"{split}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if limit is not None:
        records = records[:limit]
    return records


def _select(records: list[dict], report: dict, composition: dict | None = None) -> list[dict]:
    """Pick the required pack composition, deterministically."""

    buckets: dict[str, list[dict]] = {
        "level1": [],
        "level2_nearest": [],
        "level2_direction": [],
        "level3_nontrivial": [],
        "level3_trivial": [],
    }
    for record in records:
        level = record["level"]
        query_type = record["query_type"]
        if level == 1:
            buckets["level1"].append(record)
        elif level == 2:
            if query_type.endswith("_to_nearest"):
                buckets["level2_nearest"].append(record)
            else:
                buckets["level2_direction"].append(record)
        else:
            key = "level3_trivial" if record.get("trivial_selection") else "level3_nontrivial"
            buckets[key].append(record)

    wanted = [
        ("level1", (composition or {}).get("level1", 6)),
        ("level2_nearest", (composition or {}).get("level2_nearest", 4)),
        ("level2_direction", (composition or {}).get("level2_direction", 4)),
        ("level3_nontrivial", (composition or {}).get("level3_nontrivial", 10)),
        ("level3_trivial", (composition or {}).get("level3_trivial", 4)),
    ]
    chosen: list[tuple[str, dict]] = []
    for name, count in wanted:
        # Deterministic spread: take evenly spaced picks rather than the first N.
        pool = buckets[name]
        if not pool:
            continue
        step = max(1, len(pool) // max(count, 1))
        picks = pool[::step][:count]
        for record in picks:
            chosen.append((name, record))

    # Up to 8 representative semantic failure cases.
    failures = report.get("issues", {}).get("examples", {})
    failure_ids: set[str] = set()
    for code in ("hidden_eligibility_largest", "hidden_eligibility_smallest",
                 "hidden_eligibility_nearest", "hidden_eligibility_level3_nearest",
                 "target_recompute_mismatch"):
        for entry in failures.get(code, []):
            if entry.get("sample_id"):
                failure_ids.add(entry["sample_id"])
    if failure_ids:
        by_id = {r["sample_id"]: r for r in records}
        for sample_id in sorted(failure_ids)[:8]:
            if sample_id in by_id:
                chosen.append(("semantic_failure", by_id[sample_id]))

    return chosen


def _render(name: str, record: dict, out_path: Path) -> bool:
    import cv2

    image_id = record["image_id"]
    split = record["split"]
    meta_path = DATASET_ROOT / "metadata" / f"{split}.jsonl"
    meta = None
    with meta_path.open(encoding="utf-8") as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate["image_id"] == image_id:
                meta = candidate
                break
    if meta is None:
        return False

    image = G.image_geometry_from_record(meta)
    config = T.load_config()
    quality = None
    try:
        component_map = image.load_map(DATASET_ROOT)
    except Exception:  # noqa: BLE001
        return False

    # Source image greyscale as background.
    source_path = (_REPO_ROOT / meta["image_path"]).resolve()
    canvas = None
    if source_path.is_file():
        source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if source is not None:
            canvas = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    if canvas is None:
        canvas = np.zeros((image.height, image.width, 3), np.uint8)
    canvas = (canvas * 0.55).astype(np.uint8)

    target_id = record["target_component_id"]

    # Candidates as a dim overlay (audit only).
    for component_id in record["candidate_component_ids"]:
        mask = component_map == component_id
        canvas[mask] = (canvas[mask] * 0.4 + np.array([60, 60, 60]) * 0.6).astype(np.uint8)

    # Reference outline.
    for reference_id in record["reference_component_ids"]:
        contour_mask = (component_map == reference_id).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 200, 255), 2)

    # Target outline and fill.
    target_mask = component_map == target_id
    canvas[target_mask] = (canvas[target_mask] * 0.35 + np.array([0, 255, 0]) * 0.65).astype(np.uint8)
    contours, _ = cv2.findContours(target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)

    # Audit-only id overlay.
    for component in image.components:
        x0, y0, _, _ = component.bbox_xyxy_px
        colour = (0, 255, 0) if component.component_id == target_id else (180, 180, 180)
        cv2.putText(canvas, str(component.component_id), (x0, max(10, y0 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    # Instruction banner.
    banner = np.zeros((96, image.width, 3), np.uint8)
    lines = [
        f"[{name}] {record['query_type']}  L{record['level']}"
        + ("  TRIVIAL" if record.get("trivial_selection") else ""),
        f"ref={record['reference_component_ids']} target={target_id} "
        f"cand={record['candidate_component_ids'][:6]}",
        record["instruction_en"][:110],
    ]
    for i, text in enumerate(lines):
        cv2.putText(banner, text, (4, 22 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    combined = np.vstack([banner, canvas])

    scale = THUMB / combined.shape[0]
    resized = cv2.resize(combined, (int(combined.shape[1] * scale), THUMB), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), resized)
    return True


def build_contact_sheet(sample_dir: Path, out_path: Path, columns: int = CONTACT_COLUMNS) -> int:
    import cv2

    paths = sorted(p for p in sample_dir.glob("*.png") if p.name != "contact_sheet.png")
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        thumb = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, path.stem[:34], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    if not thumbs:
        return 0
    rows = (len(thumbs) + columns - 1) // columns
    sheet = np.zeros((rows * 256, columns * 256, 3), np.uint8)
    for index, thumb in enumerate(thumbs):
        r, c = divmod(index, columns)
        sheet[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = thumb
    cv2.imwrite(str(out_path), sheet)
    return len(thumbs)


def build_sample_pack(sample_dir: Path, report: dict, limit: int | None = None,
                      version: str = "v0.1", composition: dict | None = None) -> list[str]:
    """Write the pack and the contact sheet. Returns written file names.

    ``composition`` overrides the default (v0.1-shaped) pack composition, e.g.
    ``{"level1": 4, "level2_nearest": 3, "level2_direction": 3,
       "level3_nontrivial": 8, "level3_trivial": 4}``.
    """

    sample_dir.mkdir(parents=True, exist_ok=True)
    for stale in sample_dir.glob("*.png"):
        stale.unlink()

    records = _load_records(limit, version)
    chosen = _select(records, report, composition=composition)

    written: list[str] = []
    for index, (name, record) in enumerate(chosen):
        safe_id = record["sample_id"].replace("/", "-")
        out_path = sample_dir / f"{index:03d}_{name}_{safe_id}.png"
        try:
            if _render(name, record, out_path):
                written.append(out_path.name)
        except Exception:  # noqa: BLE001
            continue

    build_contact_sheet(sample_dir, sample_dir / "contact_sheet.png")
    return written
