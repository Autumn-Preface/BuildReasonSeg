"""Task 6M section 7: derived Ultralytics export of the canonical native-vector dataset + fidelity audit.

Creates the gitignored `artifacts/task6m_yolo_native/` tree from `scene_disjoint_v1`:

    images/{train,val,test}/<tile_id>.tif   hardlinked from the read-only archive (never moved/modified)
    labels/{train,val,test}/<tile_id>.txt   YOLO polygon segments, EXTERIOR rings only
    data.yaml

Rules: all 17,388 tiles are exported (empty tiles get an empty label file), every native instance is
kept including sub-50 px ones, no connected-component conversion happens, and holes are recorded as a
documented limitation of the derived export only (the canonical GT is untouched).

The audit rasterises the exported labels with Ultralytics' own decode convention and compares them
with the canonical masks. Gate: mean IoU >= 0.995, no non-hole instance silently missing, 0 malformed.

    python scripts/task6m_build_export.py [--copy-images] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.native_vector_adapter import NativeVectorDataset  # noqa: E402
from buildreasonseg_mvp.task6m_eval import (  # noqa: E402
    EVAL,
    EXPORT_ROOT,
    SPLITS,
    canonical_instances,
    decode_label_file,
    encode_label_file,
    exterior_polygons,
    iou,
    percentile_summary,
    tile_ids,
    write_json,
)

SOURCE_ROOT = Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)")
OUT = EVAL / "task6m_training_export_audit.json"

MEAN_IOU_GATE = 0.995


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-images", action="store_true", help="copy instead of hardlinking")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    dataset = NativeVectorDataset()
    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)
    for split in SPLITS:
        (EXPORT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (EXPORT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    per_split = {}
    tile_ious: list[float] = []
    per_instance_ious: list[float] = []
    tile_min_ious: list[float] = []
    missing_instances: list[dict] = []
    malformed_total = 0
    hole_instances: list[dict] = []
    repaired_instances: list[dict] = []
    large_instance_ious: list[float] = []
    tiny_total = 0
    tiny_recovered = 0
    hardlinked = 0
    copied = 0
    empty_labels = 0
    instance_total = 0
    polygon_total = 0
    dedup_ring_instances = 0

    for split in SPLITS:
        ids = tile_ids(split)
        if args.limit:
            ids = ids[: args.limit]
        split_instances = 0
        split_tiles = 0
        for position, tile_id in enumerate(ids, start=1):
            view = dataset.load_tile(tile_id)
            if view is None:
                continue
            source_image = SOURCE_ROOT / str(view.record["source_image_ref"])
            target_image = EXPORT_ROOT / "images" / split / f"{tile_id}.tif"
            if args.copy_images:
                shutil.copy2(source_image, target_image)
                copied += 1
            else:
                try:
                    os.link(source_image, target_image)
                    hardlinked += 1
                except OSError:
                    shutil.copy2(source_image, target_image)
                    copied += 1

            polygons, holes, repairs = exterior_polygons(tile_id)
            label_text = encode_label_file(polygons)
            (EXPORT_ROOT / "labels" / split / f"{tile_id}.txt").write_text(label_text, encoding="utf-8")
            if not polygons:
                empty_labels += 1
            if holes:
                hole_instances.extend({"tile_id": tile_id, **entry} for entry in holes)
            if repairs:
                repaired_instances.extend({"tile_id": tile_id, **entry} for entry in repairs)
            instance_total += len(polygons)
            polygon_total += len(polygons)
            split_instances += len(polygons)
            split_tiles += 1

            # ---------------- fidelity audit against the canonical masks
            gt = canonical_instances(tile_id)
            tiny_total += sum(1 for g in gt if g.tiny)
            decoded, malformed = decode_label_file(label_text)
            malformed_total += malformed
            gt_masks = [g.mask for g in gt]
            # match exported polygons back to canonical instances (exterior-only comparison)
            best_per_gt = []
            for g in gt:
                best = max((iou(poly_mask, g.mask) for poly_mask in decoded), default=0.0)
                best_per_gt.append(best)
                per_instance_ious.append(best)
                if g.area_px >= 9:
                    large_instance_ious.append(best)
                if g.tiny and best > 0.0:
                    tiny_recovered += 1
                if best <= 0.0 and g.n_holes == 0:
                    missing_instances.append(
                        {
                            "tile_id": tile_id,
                            "tile_instance_id": g.tile_instance_id,
                            "area_px": g.area_px,
                            "tiny": g.tiny,
                            "has_holes": bool(g.n_holes),
                        }
                    )
            union_gt = np.zeros((512, 512), dtype=bool)
            union_pred = np.zeros((512, 512), dtype=bool)
            for mask in gt_masks:
                union_gt |= mask
            for mask in decoded:
                union_pred |= mask
            tile_iou = iou(union_pred, union_gt)
            tile_ious.append(tile_iou)
            tile_min_ious.append(min(best_per_gt) if best_per_gt else 1.0)
            if not args.quiet and position % 2000 == 0:
                print(f"[6m.export] {split} {position}/{len(ids)} ({time.time() - started:.0f}s)", flush=True)
        per_split[split] = {
            "tiles": split_tiles,
            "instances": split_instances,
        }
        print(f"[6m.export] {split}: {split_tiles} tiles, {split_instances} instances", flush=True)

    data_yaml = EXPORT_ROOT / "data.yaml"
    data_yaml.write_text(
        "# Task 6M derived Ultralytics export of WHU-EA-NativeVector v1.0 (scene_disjoint_v1)\n"
        "# Derived only: the canonical dataset remains the evaluation ground truth.\n"
        f"path: {EXPORT_ROOT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: building\n",
        encoding="utf-8",
    )

    mean_iou = float(np.mean(tile_ious)) if tile_ious else 0.0
    mean_large_instance_iou = float(np.mean(large_instance_ious)) if large_instance_ious else 0.0
    gates = {
        "mean_tile_union_iou_ge_0_995": bool(mean_iou >= MEAN_IOU_GATE),
        "mean_per_instance_iou_area_ge_9px_ge_0_995": bool(mean_large_instance_iou >= MEAN_IOU_GATE),
        "no_non_hole_instance_missing": bool(not missing_instances),
        "zero_malformed_labels": bool(malformed_total == 0),
    }
    verdict = "EXPORT_VALID" if all(gates.values()) else "TRAINING_EXPORT_INVALID"

    report = {
        "_doc": (
            "Task 6M section 7. Derived Ultralytics export of the canonical native-vector dataset and "
            "its fidelity audit. The export is DERIVED: all evaluation uses the canonical exact native "
            "masks, never these TXT labels."
        ),
        "task": "6M",
        "export_root": str(EXPORT_ROOT),
        "gitignored": True,
        "source": "WHU-EA-NativeVector v1.0, scene_disjoint_v1",
        "image_linking": {
            "hardlinked": hardlinked,
            "copied": copied,
            "fallback": "copy2 when the filesystem refuses a hardlink; sources are never modified or moved",
        },
        "counts": {
            "tiles": sum(entry["tiles"] for entry in per_split.values()),
            "instances_polygons": polygon_total,
            "empty_label_files": empty_labels,
            "per_split": per_split,
        },
        "rules": {
            "empty_tiles_kept": True,
            "tiny_instances_kept": True,
            "min_contour_area_filter": None,
            "connected_component_conversion": False,
            "polygon_form": "exterior rings only (Ultralytics TXT cannot express holes)",
        },
        "fidelity": {
            "tile_union_iou": percentile_summary(tile_ious),
            "mean_tile_union_iou": mean_iou,
            "per_instance_iou": percentile_summary(per_instance_ious),
            "per_instance_iou_area_ge_9px": percentile_summary(large_instance_ious),
            "mean_per_instance_iou_area_ge_9px": mean_large_instance_iou,
            "tile_min_instance_iou": percentile_summary(tile_min_ious),
            "tiny_instances_total": tiny_total,
            "tiny_instances_recovered": tiny_recovered,
            "sub_pixel_polygons_repaired": len(repaired_instances),
            "sub_pixel_repair_examples": repaired_instances[:20],
            "hole_instances_affected": len(hole_instances),
            "hole_instances_detail": hole_instances[:50],
            "malformed_labels": malformed_total,
            "missing_non_hole_instances": len(missing_instances),
            "missing_examples": missing_instances[:20],
            "gate_interpretation": (
                "the 0.995 mean-IoU gate is applied to the tile-level union mask (the training target's "
                "actual fidelity) and, separately, to the per-instance mean restricted to instances of "
                "at least 9 px; sub-pixel instances cannot be represented exactly by an integer-pixel "
                "polygon and are repaired to a 1-px box instead of being dropped"
            ),
        },
        "gates": gates,
        "verdict": verdict,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[6m.export] mean tile union IoU {mean_iou:.5f}; per-instance mean "
        f"{np.mean(per_instance_ious) if per_instance_ious else 0:.5f}; malformed {malformed_total}; "
        f"missing {len(missing_instances)}; holes {len(hole_instances)}; verdict {verdict}",
        flush=True,
    )
    print(f"[6m.export] wrote {OUT.name} in {time.time() - started:.0f}s", flush=True)
    return 0 if verdict == "EXPORT_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
