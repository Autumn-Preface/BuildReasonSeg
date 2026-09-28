"""Task 6M section 12: native J1-v2 — oracle program + PREDICTED proposals on validation.

Runs the frozen structured executor over YOLO26m-seg predictions for the whole v0.2 val split, the
frozen val fixed120 pack and the frozen val paired20 pack, and reports the development gate.

    python scripts/task6m_j1v2.py [--limit N] [--skip-full]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import (  # noqa: E402
    EVAL,
    canonical_instances,
    tile_ids,
    write_json,
)
from buildreasonseg_mvp.task6m_structured import (  # noqa: E402
    evaluate_fixed120,
    evaluate_full_split,
    evaluate_paired20,
    load_frozen_config,
    load_pack,
    predict_tiles,
    relation_config,
)

OUT = EVAL / "task6m_j1v2_val.json"
PACK_FIXED = "task6m_val_fixed120.json"
PACK_PAIRED = "task6m_val_paired20.json"

#: development gate (Task 6M section 12)
GATE = {
    "overall_recall_at_0_50_min": 0.92,
    "tiny_recall_at_0_50_min": 0.60,
    "fixed120_miou_min": 0.50,
    "paired_pass_min": 14,
    "abstentions_max": 20,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposal_recall(predictions: dict, ids: list[str]) -> dict:
    """Recall @0.25/0.50/0.75 and the tiny subset, computed with the fast label-map path."""

    from buildreasonseg_mvp.task6m_eval import build_label_map, fast_best_iou

    hits = {0.25: 0, 0.50: 0, 0.75: 0}
    total = 0
    tiny_hits = tiny_total = 0
    border_hits = border_total = 0
    best_ious: list[float] = []
    for tile_id in ids:
        gt = canonical_instances(tile_id)
        if not gt:
            continue
        masks = predictions.get(tile_id, {}).get("masks", [])
        label_map = build_label_map(masks)
        areas = np.asarray([int(np.asarray(m).astype(bool).sum()) for m in masks], dtype=np.int64)
        for g in gt:
            total += 1
            best, _index = fast_best_iou(g.mask, label_map, areas)
            best_ious.append(best)
            for threshold in hits:
                if best >= threshold:
                    hits[threshold] += 1
            if g.tiny:
                tiny_total += 1
                tiny_hits += int(best >= 0.50)
            if g.touches_border:
                border_total += 1
                border_hits += int(best >= 0.50)
    return {
        "gt_instances": total,
        "recall_at_0_25": hits[0.25] / total if total else None,
        "recall_at_0_50": hits[0.50] / total if total else None,
        "recall_at_0_75": hits[0.75] / total if total else None,
        "tiny_recall_at_0_50": tiny_hits / tiny_total if tiny_total else None,
        "tiny_instances": tiny_total,
        "border_recall_at_0_50": border_hits / border_total if border_total else None,
        "border_instances": border_total,
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    from ultralytics import YOLO

    frozen = load_frozen_config()
    checkpoint = Path(frozen["checkpoint"]["path"])
    if not checkpoint.is_file():
        print(f"error: checkpoint missing: {checkpoint}", file=sys.stderr)
        return 2
    config = relation_config()

    ids = tile_ids("val")
    if args.limit:
        ids = ids[: args.limit]
    model = YOLO(str(checkpoint))
    predictions = predict_tiles(
        model, ids, conf=float(frozen["conf"]), max_det=int(frozen["max_det"]),
        imgsz=int(frozen.get("imgsz", 640)), device=args.device, verbose=not args.quiet,
    )

    recall = proposal_recall(predictions, ids)
    fixed_pack = load_pack(PACK_FIXED)
    paired_pack = load_pack(PACK_PAIRED)
    fixed = evaluate_fixed120(predictions, fixed_pack, config, program_of=lambda record: str(record["query_type"]))
    paired = evaluate_paired20(predictions, paired_pack, config, program_of=lambda record: str(record["query_type"]))
    full = None if args.skip_full else evaluate_full_split(predictions, ids, config)

    checks = {
        "overall_recall_at_0_50": bool(recall["recall_at_0_50"] is not None
                                      and recall["recall_at_0_50"] >= GATE["overall_recall_at_0_50_min"]),
        "tiny_recall_at_0_50": bool(recall["tiny_recall_at_0_50"] is not None
                                    and recall["tiny_recall_at_0_50"] >= GATE["tiny_recall_at_0_50_min"]),
        "fixed120_miou": bool(fixed["miou"] is not None and fixed["miou"] >= GATE["fixed120_miou_min"]),
        "paired_pass": bool(paired["passed"] >= GATE["paired_pass_min"]),
        "abstentions": bool(fixed["abstentions"] <= GATE["abstentions_max"]),
    }

    report = {
        "_doc": (
            "Task 6M section 12. Native J1-v2 on the validation split: oracle program + PREDICTED "
            "YOLO26m-seg proposals. GT is used for scoring only; no GT enters the executor."
        ),
        "task": "6M",
        "split": "val",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "inference_config": {
            "conf": frozen["conf"],
            "max_det": frozen["max_det"],
            "imgsz": frozen.get("imgsz", 640),
            "frozen_before_test": frozen.get("frozen_before_test"),
        },
        "proposal_recall": recall,
        "fixed120": {key: value for key, value in fixed.items() if key != "rows"},
        "paired20": {key: value for key, value in paired.items() if key != "rows"},
        "full_val": None if full is None else {key: value for key, value in full.items() if key != "rows_sample"},
        "gate": {"thresholds": GATE, "checks": checks, "passed": all(checks.values())},
        "verdict": "J1V2_GATE_PASS" if all(checks.values()) else "J1V2_GATE_FAIL",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[6m.j1v2] recall@0.50 {recall['recall_at_0_50']:.4f} (tiny {recall['tiny_recall_at_0_50']}); "
        f"fixed120 mIoU {fixed['miou']:.4f}; paired {paired['passed']}/{paired['pairs']}; "
        f"abstentions {fixed['abstentions']}; gate {'PASS' if all(checks.values()) else 'FAIL'}",
        flush=True,
    )
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
