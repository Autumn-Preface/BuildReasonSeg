"""Task 6N section 13 — Stage N0, the parameter-free field sanity check.

No training. On MiniVal240, for every record:

1. mean ``P_rel`` over the target mask;
2. mean over the reference mask;
3. mean over every other native building of the tile;
4. target rank among non-reference native instances by mean field score.

Reports top-1 rate, top-3 rate, mean target score, mean best distractor score and the per-relation
breakdown. Diagnostic only: no threshold is tuned here.

    python scripts/task6n_field_sanity.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.geometric_relation_field import (  # noqa: E402
    DEFAULT_FIELD_CONFIG,
    geometric_relation_field,
)
from buildreasonseg_mvp.task6m_eval import canonical_instances, write_json  # noqa: E402
from buildreasonseg_mvp.task6n_relation_decoder import read_pack  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6n_field_sanity.json"
FEATURE_SIZE = (64, 64)


def field_scores(record, instances, label_masks) -> dict:
    masks = [instance.mask for instance in instances]
    reference_index = next(
        index for index, instance in enumerate(instances)
        if instance.source_feature_id == record.reference_source_feature_id
    )
    target_index = next(
        index for index, instance in enumerate(instances)
        if instance.source_feature_id == record.target_source_feature_id
    )
    reference_mask = masks[reference_index]
    field = geometric_relation_field(reference_mask, record.relation, FEATURE_SIZE, DEFAULT_FIELD_CONFIG)
    upsampled = torch.nn.functional.interpolate(
        field, size=masks[0].shape, mode="bilinear", align_corners=False
    )[0, 0].numpy()

    scores = np.asarray(
        [float(upsampled[mask].mean()) if mask.any() else 0.0 for mask in masks], dtype=np.float64
    )
    distractors = [index for index in range(len(masks)) if index != reference_index]
    ranking = sorted(distractors, key=lambda index: (-scores[index], index))
    rank = ranking.index(target_index) + 1 if target_index in ranking else None
    # item 3: every other native building (the reference excluded, the target included)
    other_buildings = [scores[index] for index in distractors]
    # true competitors: neither the reference nor the target
    true_distractors = [index for index in distractors if index != target_index]
    best_true_distractor = max((scores[index] for index in true_distractors), default=None)
    target_mask = masks[target_index]
    reference_area = int(reference_mask.sum())
    return {
        "tile_id": record.tile_id,
        "sample_id": record.sample_id,
        "program_id": record.program_id,
        "relation": record.relation,
        "reference_family": record.program_id.split("_", 1)[0],
        "native_instances": len(masks),
        "distractors": len(distractors),
        "target_score": float(scores[target_index]),
        "reference_score": float(scores[reference_index]),
        "mean_other_building_score": float(np.mean(other_buildings)) if other_buildings else None,
        "best_distractor_score": float(best_true_distractor) if best_true_distractor is not None else None,
        "target_rank": rank,
        "target_area_px": int(target_mask.sum()),
        "reference_area_px": reference_area,
        "target_touches_border": bool(record.metadata.get("target_touches_border")),
        "target_tiny": bool(record.metadata.get("target_tiny")),
    }


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ranks = [row["target_rank"] for row in rows if row["target_rank"] is not None]
    target_scores = [row["target_score"] for row in rows]
    distractors = [row["best_distractor_score"] for row in rows if row["best_distractor_score"] is not None]
    margins = [
        row["target_score"] - row["best_distractor_score"]
        for row in rows if row["best_distractor_score"] is not None
    ]
    return {
        "records": len(rows),
        "top_1_rate": float(np.mean([rank == 1 for rank in ranks])) if ranks else None,
        "top_3_rate": float(np.mean([rank <= 3 for rank in ranks])) if ranks else None,
        "mean_target_score": float(np.mean(target_scores)) if target_scores else None,
        "mean_other_building_score": float(
            np.mean([row["mean_other_building_score"] for row in rows if row["mean_other_building_score"] is not None])
        ),
        "mean_best_distractor_score": float(np.mean(distractors)) if distractors else None,
        "mean_target_minus_best_distractor": float(np.mean(margins)) if margins else None,
        "mean_target_rank": float(np.mean(ranks)) if ranks else None,
        "mean_reference_score": float(np.mean([row["reference_score"] for row in rows])),
        "mean_native_instances": float(np.mean([row["native_instances"] for row in rows])),
        "mean_target_area_px": float(np.mean([row["target_area_px"] for row in rows])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    started = time.time()
    pack_path = args.pack or (REPO_ROOT / "artifacts" / "task6n" / "packs" / "mini_val_240.json")
    samples = read_pack(pack_path)
    print(f"[6n.n0] field sanity on {len(samples)} MiniVal240 records", flush=True)

    rows = []
    tile_cache: dict[str, list] = {}
    label_cache: dict[str, np.ndarray] = {}
    for index, record in enumerate(samples, start=1):
        if record.tile_id not in tile_cache:
            tile_cache[record.tile_id] = canonical_instances(record.tile_id)
            label_cache[record.tile_id] = _label_map(record.tile_id)
        instances = tile_cache[record.tile_id]
        rows.append(field_scores(record, instances, label_cache))
        if index % 60 == 0:
            print(f"[6n.n0] {index}/{len(samples)}", flush=True)

    overall = summarise(rows)
    per_relation = {}
    for relation in ("left_of", "right_of", "above", "below"):
        per_relation[relation] = summarise([row for row in rows if row["relation"] == relation])
    per_family = {}
    for family in ("largest", "smallest"):
        per_family[family] = summarise([row for row in rows if row["reference_family"] == family])

    payload = {
        "_doc": (
            "Task 6N section 13. Stage N0 diagnostic: how well the parameter-free "
            "GeometricRelationField v0.1 ranks the true target among the tile's native instances, "
            "given an oracle reference mask. No training, no threshold tuning."
        ),
        "task": "6N",
        "stage": "N0",
        "reference_source": "oracle_native_gt",
        "pack": {"path": str(pack_path), "count": len(samples)},
        "field": {
            "name": "GeometricRelationField v0.1",
            "s_axis": DEFAULT_FIELD_CONFIG.s_axis,
            "s_margin": DEFAULT_FIELD_CONFIG.s_margin,
            "alpha": DEFAULT_FIELD_CONFIG.alpha,
            "tau": DEFAULT_FIELD_CONFIG.tau,
            "softness": DEFAULT_FIELD_CONFIG.softness,
            "learned_parameters": 0,
        },
        "overall": overall,
        "per_relation": per_relation,
        "per_reference_family": per_family,
        "records": rows,
        "diagnostic_only": True,
        "threshold_tuning": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(Path(args.out), payload)
    print(
        f"[6n.n0] top-1 {overall['top_1_rate']:.4f} | top-3 {overall['top_3_rate']:.4f} | "
        f"mean target score {overall['mean_target_score']:.4f} | mean best distractor "
        f"{overall['mean_best_distractor_score']:.4f}",
        flush=True,
    )
    return 0


def _label_map(tile_id: str) -> np.ndarray:
    from buildreasonseg_mvp.whu_native_vector import read_tile_cache

    cache = read_tile_cache(tile_id)
    return cache["label_map"] if cache is not None else np.zeros((512, 512), dtype=np.int32)


if __name__ == "__main__":
    raise SystemExit(main())
