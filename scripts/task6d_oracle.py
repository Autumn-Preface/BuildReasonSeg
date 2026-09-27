#!/usr/bin/env python
"""Task 6D sections 4-5: oracle point/box spatial-prompt diagnostic.

    python scripts/task6d_oracle.py

Answers one question before any training happens: **can SAM2 recover the target building
when it is given correct geometry?**

* Oracle Point — one deterministic interior point per GT target mask (maximum of the
  Euclidean distance transform, ties broken by row-major argmax), fed through the
  official SAM2 prompt encoder as a single positive point.
* Oracle Box — the tight GT bounding box, fed through the official SAM2 box path.

Both use the frozen SAM2 encoder/decoder state that the Task 6D candidate will start
from: a clean runtime, no training, the Task 6C.7 accepted runtime settings. The fixed
120-record validation subset and the 20 paired validation images are used; nothing is
trained on validation and no GT geometry is available to any predicted path.

Writes `evaluation/task6d_oracle_prompt_diagnostic.json` and applies section 5's geometry
selection rule, which is recorded verbatim in the artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.grounding import (  # noqa: E402
    GEOMETRY_BOX,
    GEOMETRY_KINDS,
    GEOMETRY_POINT,
    decode_mask_from_geometry,
    geometry_is_valid,
    mask_contains_point,
    target_geometry,
)
from buildreasonseg_mvp.metrics import dice_from_logits, mask_iou_from_logits  # noqa: E402
from task6c6_common import EVAL, build_variant_runtime, write_json  # noqa: E402
from task6c_train import validation_material  # noqa: E402

OUT = EVAL / "task6d_oracle_prompt_diagnostic.json"

#: Section 5 gates.
PAIRED_GATE = 18
MIOU_GATE = 0.50
BOX_BETTER_BY = 0.10


def _iou(predicted_logits: torch.Tensor, target: np.ndarray) -> float:
    target_tensor = torch.as_tensor(target).float()
    value = mask_iou_from_logits(
        predicted_logits, target_tensor, tuple(target.shape), threshold=0.0
    )
    return float(value)


def _dice(predicted_logits: torch.Tensor, target: np.ndarray) -> float:
    target_tensor = torch.as_tensor(target).float()
    value = dice_from_logits(predicted_logits, target_tensor, tuple(target.shape), threshold=0.0)
    return float(value)


def run_oracle(runtime, samples, pairs, kind: str) -> dict:
    """Decode every validation target from its own oracle geometry."""

    records = []
    for sample in samples:
        image = sample.image_rgb()
        features, _cached = runtime.features_for(sample, image)
        target = sample.target_mask()
        geometry = target_geometry(target, kind)
        geometry_tensor = torch.as_tensor(geometry, dtype=torch.float32, device=runtime.device).reshape(1, -1)
        result = decode_mask_from_geometry(
            runtime.model.sam, features, geometry_tensor, kind, multimask_output=False
        )
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "level": sample.level,
                "query_type": sample.query_type,
                "target_component_id": sample.target_component_id,
                "geometry_normalized": list(geometry),
                "geometry_valid": geometry_is_valid(geometry, kind),
                "iou": _iou(result.low_res_logits, target),
                "dice": _dice(result.low_res_logits, target),
                "image_shape": list(image.shape[:2]) if hasattr(image, "shape") else None,
                "prompt": result.prompt_diagnostics,
            }
        )
        del result, features

    ious = [record["iou"] for record in records]
    dices = [record["dice"] for record in records]

    # Paired probe: each image's two instructions get their own oracle geometry.
    paired_records = []
    for index, pair in enumerate(pairs):
        sample_a = data_mod.to_sample(pair["a"])
        sample_b = data_mod.to_sample(pair["b"])
        if sample_a.image_id != sample_b.image_id:
            raise ValueError(f"paired probe is not same-image: {sample_a.sample_id} vs {sample_b.sample_id}")
        image = sample_a.image_rgb()
        features, _cached = runtime.features_for(sample_a, image)
        target_a = sample_a.target_mask()
        target_b = sample_b.target_mask()
        geometry_a = target_geometry(target_a, kind)
        geometry_b = target_geometry(target_b, kind)
        tensor_a = torch.as_tensor(geometry_a, dtype=torch.float32, device=runtime.device).reshape(1, -1)
        tensor_b = torch.as_tensor(geometry_b, dtype=torch.float32, device=runtime.device).reshape(1, -1)
        result_a = decode_mask_from_geometry(runtime.model.sam, features, tensor_a, kind)
        result_b = decode_mask_from_geometry(runtime.model.sam, features, tensor_b, kind)

        own_a = _iou(result_a.low_res_logits, target_a)
        own_b = _iou(result_b.low_res_logits, target_b)
        cross_a = _iou(result_a.low_res_logits, target_b)
        cross_b = _iou(result_b.low_res_logits, target_a)
        prediction_iou = _iou(result_a.low_res_logits, target_b)  # placeholder, replaced below

        mask_a = (
            torch.nn.functional.interpolate(
                result_a.low_res_logits, size=tuple(target_a.shape), mode="bilinear", align_corners=False
            )
            > 0.0
        )
        mask_b = (
            torch.nn.functional.interpolate(
                result_b.low_res_logits, size=tuple(target_b.shape), mode="bilinear", align_corners=False
            )
            > 0.0
        )
        intersection = float((mask_a & mask_b).sum())
        union = float((mask_a | mask_b).sum())
        prediction_iou = intersection / union if union else 0.0

        geometry_contains = {
            "a_point_in_target_a": mask_contains_point(target_a, geometry_a[:2]),
            "a_point_in_target_b": mask_contains_point(target_b, geometry_a[:2]),
            "b_point_in_target_b": mask_contains_point(target_b, geometry_b[:2]),
            "b_point_in_target_a": mask_contains_point(target_a, geometry_b[:2]),
        }
        paired_records.append(
            {
                "image_id": sample_a.image_id,
                "sample_a": sample_a.sample_id,
                "sample_b": sample_b.sample_id,
                "geometry_a": list(geometry_a),
                "geometry_b": list(geometry_b),
                "geometry_distance": float(
                    np.linalg.norm(np.asarray(geometry_a) - np.asarray(geometry_b))
                ),
                "geometry_differs": bool(
                    not np.allclose(np.asarray(geometry_a), np.asarray(geometry_b))
                ),
                "own_iou_a": own_a,
                "own_iou_b": own_b,
                "cross_iou_a": cross_a,
                "cross_iou_b": cross_b,
                "mean_own_iou": (own_a + own_b) / 2.0,
                "mean_cross_iou": (cross_a + cross_b) / 2.0,
                "own_minus_cross_margin": ((own_a - cross_a) + (own_b - cross_b)) / 2.0,
                "iou_prediction_a_vs_prediction_b": prediction_iou,
                "paired_pass": bool(own_a > cross_a and own_b > cross_b),
                "point_containment": geometry_contains,
                "oracle_geometry_hit": (
                    geometry_contains["a_point_in_target_a"] and geometry_contains["b_point_in_target_b"]
                    if kind == GEOMETRY_POINT
                    else None
                ),
            }
        )
        del result_a, result_b, features

    paired_passes = sum(1 for record in paired_records if record["paired_pass"])
    return {
        "kind": kind,
        "records": records,
        "records_count": len(records),
        "strict_mask_miou": float(np.mean(ious)) if ious else None,
        "strict_mask_dice": float(np.mean(dices)) if dices else None,
        "min_iou": float(np.min(ious)) if ious else None,
        "max_iou": float(np.max(ious)) if ious else None,
        "geometry_valid_rate": float(np.mean([record["geometry_valid"] for record in records])),
        "paired_records": paired_records,
        "paired_pass": paired_passes,
        "paired_total": len(paired_records),
        "mean_own_iou": float(np.mean([record["mean_own_iou"] for record in paired_records])),
        "mean_cross_iou": float(np.mean([record["mean_cross_iou"] for record in paired_records])),
        "mean_own_minus_cross_margin": float(
            np.mean([record["own_minus_cross_margin"] for record in paired_records])
        ),
        "mean_iou_pred_a_vs_b": float(
            np.mean([record["iou_prediction_a_vs_prediction_b"] for record in paired_records])
        ),
        "all_pairs_have_different_geometry": bool(
            all(record["geometry_differs"] for record in paired_records)
        ),
    }


def choose_geometry(point: dict, box: dict) -> dict:
    """Section 5, applied verbatim and recorded."""

    point_passes = point["paired_pass"] >= PAIRED_GATE and (point["strict_mask_miou"] or 0) >= MIOU_GATE
    box_passes = box["paired_pass"] >= PAIRED_GATE and (box["strict_mask_miou"] or 0) >= MIOU_GATE
    rule = (
        "1) if Oracle Point paired >= 18/20 and mIoU >= 0.50 choose POINT unless Oracle Box "
        "improves mIoU by >= 0.10; 2) otherwise if Oracle Box paired >= 18/20 and mIoU >= 0.50 "
        "choose BOX; 3) otherwise stop and report ORACLE_SPATIAL_PROMPT_INSUFFICIENT"
    )
    mIoU_gain = (box["strict_mask_miou"] or 0) - (point["strict_mask_miou"] or 0)
    if point_passes:
        chosen = GEOMETRY_BOX if mIoU_gain >= BOX_BETTER_BY else GEOMETRY_POINT
        reason = (
            f"point passes the gates and box improves mIoU by {mIoU_gain:+.4f} "
            f"({'>= ' if mIoU_gain >= BOX_BETTER_BY else '< '}{BOX_BETTER_BY}), so {chosen} is chosen"
        )
    elif box_passes:
        chosen = GEOMETRY_BOX
        reason = f"point does not pass the gates but box does (mIoU gain {mIoU_gain:+.4f})"
    else:
        chosen = None
        reason = "neither oracle reaches paired >= 18/20 with mIoU >= 0.50"
    return {
        "rule": rule,
        "point_passes_gates": bool(point_passes),
        "box_passes_gates": bool(box_passes),
        "box_minus_point_miou": mIoU_gain,
        "chosen_geometry": chosen,
        "reason": reason,
        "verdict_if_none": None if chosen else "ORACLE_SPATIAL_PROMPT_INSUFFICIENT",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-val", type=int, default=None, help="smoke only")
    parser.add_argument("--limit-pairs", type=int, default=None, help="smoke only")
    args = parser.parse_args(argv)

    val_samples, pairs, lookup, lookup_audit = validation_material()
    if args.limit_val:
        val_samples = val_samples[: args.limit_val]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]

    baseline = json.loads((EVAL / "task6c_comparison.json").read_text(encoding="utf-8"))
    runtime = build_variant_runtime()
    report: dict = {
        "_doc": (
            "Task 6D sections 4-5. Oracle point/box spatial-prompt diagnostic on the fixed 120-record "
            "validation subset and the 20 paired validation images. Geometry comes from the GT target "
            "mask and is used ONLY here (oracle) and as training supervision; no predicted path in this "
            "script sees GT. Decoder state is the frozen SAM2 decoder of a clean Task 6D runtime."
        ),
        "task": "6D",
        "baseline": {
            "source": "evaluation/task6c_comparison.json",
            "arm": "P_C",
            "strict_end_to_end_miou": baseline.get("arms", {}).get("P_C", {}).get("teacher_forced", {}).get(
                "miou"
            ),
            "paired_probe": "0/20 (Task 6C)",
            "note": (
                "Task 6C P_C is the frozen failure baseline: the projected language vector as SAM's "
                "sparse prompt selects the same target regardless of instruction (~0.999 same-image "
                "prediction IoU). Task 6D does not rerun the 6C arms."
            ),
        },
        "validation_material": {
            "val_records": len(val_samples),
            "paired_images": len(pairs),
            "lookup_audit": lookup_audit,
        },
        "runtime": {
            "visual_cache": runtime.visual_cache_stats(),
            "quantization_note": "frozen SAM2.1 Hiera Base+ encoder/decoder, no training",
        },
        "geometry_convention": {
            "mask_resolution": "original 512x512 target mask",
            "normalized": "[0,1]^2 point / [0,1]^4 box, resolution independent",
            "sam_prompt_space": "input image pixels (1024x1024 frame); official prompt encoder",
            "verified_against": "sam2/modeling/sam/prompt_encoder.py::_embed_points/_embed_boxes",
        },
    }

    for kind in GEOMETRY_KINDS:
        report[kind] = run_oracle(runtime, val_samples, pairs, kind)

    report["selection"] = choose_geometry(report[GEOMETRY_POINT], report[GEOMETRY_BOX])
    report["verdict"] = (
        "PROCEED_WITH_" + report["selection"]["chosen_geometry"].upper()
        if report["selection"]["chosen_geometry"]
        else "ORACLE_SPATIAL_PROMPT_INSUFFICIENT"
    )
    write_json(OUT, report)

    for kind in GEOMETRY_KINDS:
        entry = report[kind]
        print(
            f"[task6d:oracle] {kind}: mIoU={entry['strict_mask_miou']:.4f} dice={entry['strict_mask_dice']:.4f} "
            f"paired={entry['paired_pass']}/{entry['paired_total']} "
            f"own={entry['mean_own_iou']:.4f} cross={entry['mean_cross_iou']:.4f} "
            f"margin={entry['mean_own_minus_cross_margin']:+.4f} "
            f"IoU(predAB)={entry['mean_iou_pred_a_vs_b']:.4f}",
            flush=True,
        )
    print(f"[task6d:oracle] selection: {json.dumps(report['selection'], ensure_ascii=False)}", flush=True)
    print(f"[task6d:oracle] verdict: {report['verdict']}", flush=True)
    print(f"[task6d:oracle] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if report["selection"]["chosen_geometry"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
