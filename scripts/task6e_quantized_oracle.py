#!/usr/bin/env python
"""Task 6E sections 3-4: quantized-box oracle diagnostic and bin selection.

    python scripts/task6e_quantized_oracle.py [--bins 32,64,128,256]

For each candidate B, GT boxes are quantized with the deterministic enclosing rule,
dequantized, and fed through the **official frozen SAM2 box prompt path** on the fixed
120-record validation set and the 20 paired validation images. This measures whether
quantization itself destroys the oracle pathway before any tokenizer work happens.

Reference points: continuous Oracle BOX mIoU 0.7506 / paired 20/20 (Task 6D), Task 6C `P_C`
mIoU 0.10604 / paired 0/20.

Selection (section 5): the **smallest** B whose quantized oracle keeps paired 20/20 and
mIoU >= 0.7206 (within 4 % of the continuous oracle). If no B qualifies, the verdict is
`QUANTIZED_BOX_REPRESENTATION_INADEQUATE` and Task 6E stops.

Writes `evaluation/task6e_quantized_oracle.json`.
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
from buildreasonseg_mvp.grounding import GEOMETRY_BOX, decode_mask_from_geometry, target_geometry  # noqa: E402
from buildreasonseg_mvp.metrics import dice_from_logits, mask_iou_from_logits  # noqa: E402
from buildreasonseg_mvp.spatial_tokens import BIN_CANDIDATES, QuantizedBoxCodec, box_iou_values  # noqa: E402
from task6c6_common import EVAL, build_variant_runtime, write_json  # noqa: E402
from task6c_train import validation_material  # noqa: E402

OUT = EVAL / "task6e_quantized_oracle.json"
CONTINUOUS_ORACLE_MIOU = 0.7506
CONTINUOUS_ORACLE_PAIRED = 20
MIOU_TOLERANCE = 0.03  # section 5: strict mIoU >= 0.7206 == continuous oracle minus 0.03


def _iou(predicted_logits: torch.Tensor, target: np.ndarray) -> float:
    target_tensor = torch.as_tensor(target).float()
    value = mask_iou_from_logits(predicted_logits, target_tensor, tuple(target.shape), threshold=0.0)
    return float(value)


def _dice(predicted_logits: torch.Tensor, target: np.ndarray) -> float:
    target_tensor = torch.as_tensor(target).float()
    value = dice_from_logits(predicted_logits, target_tensor, tuple(target.shape), threshold=0.0)
    return float(value)


def run_bins(runtime, samples, pairs, bins: int) -> dict:
    codec = QuantizedBoxCodec(bins)
    records = []
    for sample in samples:
        image = sample.image_rgb()
        features, _cached = runtime.features_for(sample, image)
        mask = sample.target_mask()
        continuous = target_geometry(mask, GEOMETRY_BOX)
        codes = codec.encode(continuous)
        box = codec.decode(codes)
        if not codec.is_valid(codes):
            raise RuntimeError(f"quantizer produced an invalid box for {sample.sample_id}: {codes}")
        tensor = torch.as_tensor(box, dtype=torch.float32, device=runtime.device).reshape(1, -1)
        result = decode_mask_from_geometry(runtime.model.sam, features, tensor, GEOMETRY_BOX)
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "level": sample.level,
                "codes": codes,
                "continuous_box": list(continuous),
                "quantized_box": list(box),
                "quantization_box_iou": box_iou_values(continuous, box),
                "max_coordinate_error": max(
                    abs(a - b) for a, b in zip(continuous, box)
                ),
                "iou": _iou(result.low_res_logits, mask),
                "dice": _dice(result.low_res_logits, mask),
            }
        )
        del result, features

    paired = []
    for pair in pairs:
        sample_a = data_mod.to_sample(pair["a"])
        sample_b = data_mod.to_sample(pair["b"])
        image = sample_a.image_rgb()
        features, _cached = runtime.features_for(sample_a, image)
        target_a, target_b = sample_a.target_mask(), sample_b.target_mask()
        box_a = codec.decode(codec.encode(target_geometry(target_a, GEOMETRY_BOX)))
        box_b = codec.decode(codec.encode(target_geometry(target_b, GEOMETRY_BOX)))
        tensor_a = torch.as_tensor(box_a, dtype=torch.float32, device=runtime.device).reshape(1, -1)
        tensor_b = torch.as_tensor(box_b, dtype=torch.float32, device=runtime.device).reshape(1, -1)
        result_a = decode_mask_from_geometry(runtime.model.sam, features, tensor_a, GEOMETRY_BOX)
        result_b = decode_mask_from_geometry(runtime.model.sam, features, tensor_b, GEOMETRY_BOX)
        own_a = _iou(result_a.low_res_logits, target_a)
        own_b = _iou(result_b.low_res_logits, target_b)
        cross_a = _iou(result_a.low_res_logits, target_b)
        cross_b = _iou(result_b.low_res_logits, target_a)
        mask_a = (
            torch.nn.functional.interpolate(
                result_a.low_res_logits.reshape(1, 1, *result_a.low_res_logits.shape[-2:]),
                size=tuple(target_a.shape), mode="bilinear", align_corners=False
            ).reshape(tuple(target_a.shape)) > 0.0
        )
        mask_b = (
            torch.nn.functional.interpolate(
                result_b.low_res_logits.reshape(1, 1, *result_b.low_res_logits.shape[-2:]),
                size=tuple(target_b.shape), mode="bilinear", align_corners=False
            ).reshape(tuple(target_b.shape)) > 0.0
        )
        union = float((mask_a | mask_b).sum())
        paired.append(
            {
                "image_id": sample_a.image_id,
                "own_a": own_a,
                "own_b": own_b,
                "cross_a": cross_a,
                "cross_b": cross_b,
                "paired_pass": bool(own_a > cross_a and own_b > cross_b),
                "iou_pred_a_vs_b": float((mask_a & mask_b).sum()) / union if union else 0.0,
                "quantization_box_iou_a": box_iou_values(target_geometry(target_a, GEOMETRY_BOX), box_a),
                "quantization_box_iou_b": box_iou_values(target_geometry(target_b, GEOMETRY_BOX), box_b),
            }
        )
        del result_a, result_b, features

    ious = [record["iou"] for record in records]
    dices = [record["dice"] for record in records]
    return {
        "bins": bins,
        "codec": codec.as_dict(),
        "records": records,
        "strict_mask_miou": float(np.mean(ious)),
        "strict_mask_dice": float(np.mean(dices)),
        "min_iou": float(np.min(ious)),
        "max_iou": float(np.max(ious)),
        "mean_quantization_box_iou": float(np.mean([r["quantization_box_iou"] for r in records])),
        "max_coordinate_error": float(np.max([r["max_coordinate_error"] for r in records])),
        "paired": paired,
        "paired_pass": sum(1 for record in paired if record["paired_pass"]),
        "paired_total": len(paired),
        "mean_own_iou": float(np.mean([(r["own_a"] + r["own_b"]) / 2 for r in paired])),
        "mean_cross_iou": float(np.mean([(r["cross_a"] + r["cross_b"]) / 2 for r in paired])),
        "mean_own_minus_cross_margin": float(
            np.mean([((r["own_a"] - r["cross_a"]) + (r["own_b"] - r["cross_b"])) / 2 for r in paired])
        ),
        "mean_iou_pred_a_vs_b": float(np.mean([r["iou_pred_a_vs_b"] for r in paired])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", default=",".join(str(value) for value in BIN_CANDIDATES))
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args(argv)

    candidate_bins = [int(token) for token in args.bins.split(",") if token.strip()]
    val_samples, pairs, _lookup, lookup_audit = validation_material()
    if args.limit_val:
        val_samples = val_samples[: args.limit_val]
    if args.limit_pairs:
        pairs = pairs[: args.limit_pairs]

    runtime = build_variant_runtime()
    report: dict = {
        "_doc": (
            "Task 6E sections 3-4. Quantized-box oracle: GT boxes quantized with the deterministic "
            "enclosing rule and dequantized before the official frozen SAM2 box prompt path, on the "
            "fixed 120-record validation set and the 20 paired validation images. GT geometry is used "
            "only here (oracle) and as training supervision."
        ),
        "task": "6E",
        "continuous_oracle_reference": {
            "source": "evaluation/task6d_oracle_prompt_diagnostic.json",
            "miou": CONTINUOUS_ORACLE_MIOU,
            "paired": f"{CONTINUOUS_ORACLE_PAIRED}/20",
        },
        "task6c_reference": {"arm": "P_C", "miou": 0.10604, "paired": "0/20"},
        "validation_material": {"val_records": len(val_samples), "paired_images": len(pairs), "lookup_audit": lookup_audit},
        "candidates": {},
    }

    for bins in candidate_bins:
        entry = run_bins(runtime, val_samples, pairs, bins)
        report["candidates"][str(bins)] = entry
        print(
            f"[task6e:oracle] B={bins}: mIoU {entry['strict_mask_miou']:.4f} dice "
            f"{entry['strict_mask_dice']:.4f} paired {entry['paired_pass']}/{entry['paired_total']} "
            f"quant-box IoU {entry['mean_quantization_box_iou']:.4f} margin "
            f"{entry['mean_own_minus_cross_margin']:+.4f}",
            flush=True,
        )

    threshold = CONTINUOUS_ORACLE_MIOU - MIOU_TOLERANCE
    qualifying = [
        bins
        for bins in candidate_bins
        if report["candidates"][str(bins)]["paired_pass"] == report["candidates"][str(bins)]["paired_total"]
        and report["candidates"][str(bins)]["strict_mask_miou"] >= threshold
    ]
    selected = min(qualifying) if qualifying else None
    report["selection"] = {
        "rule": (
            "smallest B whose quantized oracle keeps paired 20/20 and strict mIoU >= "
            f"{threshold:.4f} (continuous oracle {CONTINUOUS_ORACLE_MIOU} minus {MIOU_TOLERANCE})"
        ),
        "miou_threshold": threshold,
        "qualifying_bins": qualifying,
        "selected_bins": selected,
        "verdict": (
            f"PROCEED_WITH_B{selected}" if selected else "QUANTIZED_BOX_REPRESENTATION_INADEQUATE"
        ),
    }
    write_json(OUT, report)
    print(f"[task6e:oracle] selection: {json.dumps(report['selection'], ensure_ascii=False)}", flush=True)
    print(f"[task6e:oracle] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if selected else 4


if __name__ == "__main__":
    raise SystemExit(main())
