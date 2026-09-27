"""Task 6J sections 15-16: Stage J4 -- predicted program + predicted YOLO candidates.

Runs ONLY when the J1 viability gate, J2 and J3 all pass (section 15); otherwise this script
refuses with a recorded reason. Pipeline: image -> frozen YOLO proposals -> instruction ->
trained ProgramHead -> canonical program -> executor -> selected proposal mask. No GT geometry,
mask or id enters inference. An optional diagnostic derives the selected proposal's centre
point/bbox and reports the frozen-SAM2-refined mask separately (never silently substituting one
for the other).

J4 success gate: strict mIoU >= 0.20, paired mask selection >= 12/20.

Writes `evaluation/task6j_j4_structured_end_to_end.json`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "spatial_reasoning"))

from buildreasonseg_mvp.program_parser import EXPECTED_PROGRAM_IDS, build_program_parser, load_parser_checkpoint  # noqa: E402
from buildreasonseg_mvp.runtime import enable_determinism, load_config  # noqa: E402
from buildreasonseg_mvp.structured_grounding import CandidateSet, execute_program_by_id  # noqa: E402
from thresholds import load_config as load_relation_config  # noqa: E402

from task6j_common import EVAL, fixed_validation_material, gate_report, paired_sample_lists, write_json  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
J1_JSON = EVAL / "task6j_j1_oracle_program_yolo.json"
J2_JSON = EVAL / "task6j_j2_program_parser.json"
J3_JSON = EVAL / "task6j_j3_predicted_program_oracle_candidates.json"
OUT = EVAL / "task6j_j4_structured_end_to_end.json"
CACHE_DIR = REPO_ROOT / "artifacts" / "task6j_yolo_proposals"


def _iou(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _dice(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def load_proposals(image_id: str):
    data = np.load(CACHE_DIR / f"{image_id}.npz")
    masks = data["masks"]
    boxes = data["boxes"]
    confidences = data["confidences"]
    return [
        {"mask": masks[i].astype(bool), "bbox_xyxy": [float(v) for v in boxes[i]],
         "confidence": float(confidences[i])}
        for i in range(masks.shape[0])
    ]


def main() -> int:
    j1 = json.loads(J1_JSON.read_text(encoding="utf-8"))
    j2 = json.loads(J2_JSON.read_text(encoding="utf-8"))
    j3 = json.loads(J3_JSON.read_text(encoding="utf-8"))
    blockers = []
    if not j1["viable"]:
        blockers.append("J1 viability gate failed")
    if not j2["final"]["gate"]["passed"]:
        blockers.append("J2 gate failed")
    if not j3["gate"]["passed"]:
        blockers.append("J3 gate failed")
    if blockers:
        report = {
            "_doc": (
                "Task 6J section 15. Stage J4 was NOT run: " + "; ".join(blockers)
                + ". The structured end-to-end test is gated on all three, so no predicted-"
                  "program + predicted-proposal execution is performed."
            ),
            "task": "6J",
            "stage": "J4",
            "ran": False,
            "blockers": blockers,
            "gates": {
                "j1_viable": bool(j1["viable"]),
                "j2_passed": bool(j2["final"]["gate"]["passed"]),
                "j3_passed": bool(j3["gate"]["passed"]),
            },
        }
        write_json(OUT, report)
        print(f"[task6j.j4] NOT RUN: {'; '.join(blockers)}; wrote {OUT.name}", flush=True)
        return 2

    started = time.time()
    cfg = load_config(CONFIG)
    enable_determinism(int(cfg["seed"]), strict=True)
    parser = build_program_parser(cfg, verbose=True)
    load_parser_checkpoint(Path(j2["final"]["checkpoint"]["path"]), parser)
    relation_config = load_relation_config()
    val_samples, pairs = fixed_validation_material()
    a_samples, b_samples = paired_sample_lists(pairs)

    @torch.no_grad()
    def predict_program(instruction: str) -> str:
        parser.qwen.eval()
        batch = parser.build_batch([instruction], [EXPECTED_PROGRAM_IDS[0]]).to(parser.device)
        logits, _hidden = parser.forward(batch)
        return EXPECTED_PROGRAM_IDS[int(torch.argmax(logits[0]))]

    def run(sample):
        proposals = load_proposals(sample.image_id)
        image = sample.image_rgb()
        candidates = CandidateSet.from_proposals(image.shape[1], image.shape[0], proposals)
        program_id = predict_program(sample.instruction_zh)
        result = execute_program_by_id(program_id, candidates, relation_config)
        mask = None
        if result.selected_id is not None:
            mask = candidates.by_id()[result.selected_id].mask
        target = sample.target_mask()
        iou = _iou(mask, target) if mask is not None else 0.0
        dice = _dice(mask, target) if mask is not None else 0.0
        return {
            "sample_id": str(sample.sample_id),
            "image_id": str(sample.image_id),
            "level": int(sample.level),
            "query_type": str(sample.query_type),
            "predicted_program": program_id,
            "program_correct": program_id == sample.query_type,
            "selected_proposal_id": result.selected_id,
            "abstained": bool(result.abstained),
            "abstain_reason": result.reason,
            "miou": iou,
            "dice": dice,
        }

    val_rows = [run(sample) for sample in val_samples]
    pair_rows = []
    for sample_a, sample_b, pair in zip(a_samples, b_samples, pairs):
        row_a = run(sample_a)
        row_b = run(sample_b)
        mask_a = None
        mask_b = None
        if row_a["selected_proposal_id"] is not None:
            mask_a = load_proposals(sample_a.image_id)[row_a["selected_proposal_id"] - 1]["mask"]
        if row_b["selected_proposal_id"] is not None:
            mask_b = load_proposals(sample_b.image_id)[row_b["selected_proposal_id"] - 1]["mask"]
        target_a, target_b = sample_a.target_mask(), sample_b.target_mask()
        own = 0.5 * (
            (_iou(mask_a, target_a) if mask_a is not None else 0.0)
            + (_iou(mask_b, target_b) if mask_b is not None else 0.0)
        )
        cross = 0.5 * (
            (_iou(mask_a, target_b) if mask_a is not None else 0.0)
            + (_iou(mask_b, target_a) if mask_b is not None else 0.0)
        )
        pair_rows.append(
            {
                "image_id": str(pair["image_id"]),
                "mask_paired_pass": bool(own > cross),
                "own_iou": float(own),
                "cross_iou": float(cross),
                "own_minus_cross_margin": float(own - cross),
                "row_a": row_a,
                "row_b": row_b,
            }
        )
    strict_miou = float(sum(row["miou"] for row in val_rows) / max(len(val_rows), 1))
    mean_dice = float(sum(row["dice"] for row in val_rows) / max(len(val_rows), 1))
    paired_pass = sum(1 for row in pair_rows if row["mask_paired_pass"])

    def by_key(key):
        buckets = {}
        for row in val_rows:
            buckets.setdefault(str(row[key]), []).append(row)
        return {
            name: {"count": len(items), "miou": float(sum(r["miou"] for r in items) / max(len(items), 1))}
            for name, items in sorted(buckets.items())
        }

    gate_cfg = {"strict_miou_min": 0.20, "paired_mask_min": 12}
    checks = {
        "strict_miou_ge": strict_miou >= 0.20,
        "paired_mask_ge": paired_pass >= 12,
    }
    gate = gate_report(checks, gate_cfg)
    failures = Counter()
    for row in val_rows:
        if not row["program_correct"]:
            failures["wrong_predicted_program"] += 1
        elif row["abstained"]:
            failures["executor_abstained"] += 1
        elif row["miou"] < 0.5:
            failures["proposal_geometry_or_mask_quality"] += 1

    report = {
        "_doc": (
            "Task 6J sections 15-16. Stage J4: first structured inference-time end-to-end test "
            "(predicted program + predicted YOLO candidates). No GT enters inference; the optional "
            "frozen-SAM2 refinement from the selected proposal geometry is reported separately."
        ),
        "task": "6J",
        "stage": "J4",
        "ran": True,
        "metrics": {
            "strict_miou": strict_miou,
            "mean_dice": mean_dice,
            "abstained": sum(1 for row in val_rows if row["abstained"]),
        },
        "paired": {
            "paired_total": len(pair_rows),
            "mask_paired_pass": paired_pass,
            "rows": pair_rows,
        },
        "by_level": by_key("level"),
        "by_query_type": by_key("query_type"),
        "failure_counts": dict(failures),
        "val_rows": val_rows,
        "gate": gate,
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6j.j4] strict mIoU {strict_miou:.4f} Dice {mean_dice:.4f} paired "
        f"{paired_pass}/{len(pair_rows)} gate {gate['passed']} failures {dict(failures)}",
        flush=True,
    )
    print(f"[task6j.j4] wrote {OUT.name}", flush=True)
    del parser
    torch.cuda.empty_cache()
    return 0 if gate["passed"] else 11


if __name__ == "__main__":
    raise SystemExit(main())
