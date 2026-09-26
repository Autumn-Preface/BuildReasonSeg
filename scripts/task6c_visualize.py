#!/usr/bin/env python
"""Task 6C section 19: compact four-arm visualization on fixed paired val images.

    python scripts/task6c_visualize.py

For the first four Task 6B paired validation images it renders one PNG per image with

    source | GT A | GT B | U_C | U_L | P_C | P_L

Each prediction panel shows the generated instruction's IoU against both ground
truths, so instruction conditioning is visible directly. A small singular-value
figure summarising the projected prompt spectrum of the four arms is added.

Writes `evaluation/task6c_samples/`.
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

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.metrics import mask_iou_from_logits, upsample_logits  # noqa: E402
from buildreasonseg_mvp.qwen_seg import generate_with_seg, seg_hidden_from_full_sequence  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_phase_trainables  # noqa: E402
from buildreasonseg_mvp.sam2_bridge import decode_mask  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
SAMPLES = EVAL / "task6c_samples"
TASK6B_SUBSETS = EVAL / "task6b_subset_ids.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6c_2b_ablation.yaml"
ARMS = (("U_C", "U", "centre"), ("U_L", "U", "language"), ("P_C", "P", "centre"), ("P_L", "P", "language"))
PANEL = 210
BANNER = 54
FONT = 0.42
THICKNESS = 1


def _iou_vs(pred_mask, gt) -> float:
    gt_bool = np.asarray(gt).astype(bool)
    inter = int((pred_mask & gt_bool).sum())
    union = int((pred_mask | gt_bool).sum())
    return inter / union if union else 0.0


def _tile(image_rgb, mask=None, colour=(0, 0, 255)) -> np.ndarray:
    import cv2

    tile = cv2.resize(np.asarray(image_rgb), (PANEL, PANEL), interpolation=cv2.INTER_AREA).copy()
    if mask is not None:
        # the mask is at the original 512x512 resolution; bring it to the tile size first
        resized = cv2.resize(np.asarray(mask).astype(np.uint8), (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
        outline = cv2.morphologyEx(resized, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        tile[outline] = colour
    return tile


def _label(tile: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    banner = np.full((BANNER, PANEL, 3), 24, np.uint8)
    for index, text in enumerate(lines[:3]):
        cv2.putText(
            banner,
            text[:34],
            (4, 14 + index * 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT,
            (240, 240, 240),
            THICKNESS,
            cv2.LINE_AA,
        )
    return np.vstack([banner, tile])


def collect_arm(arm: str, subset: str, bridge: str, samples_pairs: list[dict]) -> dict:
    cfg = load_config(CONFIG)
    cfg["bridge"] = {"mode": bridge}
    runtime = build_runtime(cfg, device="cuda", verbose=False)
    set_phase_trainables(runtime.model, "B")
    checkpoint = REPO_ROOT / cfg["paths"]["checkpoints"] / arm / "last.pt"
    info = load_checkpoint(checkpoint, runtime.model) if checkpoint.is_file() else None
    print(f"[viz] {arm}: checkpoint {'loaded' if info else 'MISSING'} {checkpoint}", flush=True)
    runtime.model.eval()
    if hasattr(runtime.qwen, "gradient_checkpointing_disable"):
        runtime.qwen.gradient_checkpointing_disable()

    out: dict[str, dict] = {}
    max_new_tokens = int(cfg["inference"]["validation_max_new_tokens"])
    with torch.no_grad():
        for pair in samples_pairs:
            for label in ("a", "b"):
                record = pair[label]
                sample = data_mod.to_sample(record)
                image = sample.image_rgb()
                gt_a = np.asarray(data_mod.to_sample(pair["a"]).target_mask()).astype(bool)
                gt_b = np.asarray(data_mod.to_sample(pair["b"]).target_mask()).astype(bool)
                generation = generate_with_seg(
                    runtime.model.qwen,
                    runtime.processor,
                    runtime.tokenizer,
                    image,
                    sample.instruction_zh,
                    max_new_tokens=max_new_tokens,
                    seg_token_id=runtime.model.seg_token_id,
                )
                entry = {
                    "text": generation["text"],
                    "seg_count": generation["seg_count"],
                    "iou_a": 0.0,
                    "iou_b": 0.0,
                    "mask": None,
                }
                if generation["seg_count"] == 1:
                    position = generation["seg_positions"][0]
                    hidden = seg_hidden_from_full_sequence(
                        runtime.model.qwen,
                        runtime.processor,
                        image,
                        generation["token_ids"],
                        position,
                        instruction=sample.instruction_zh,
                    )
                    projected = runtime.projection(hidden)
                    features, _cached = runtime.features_for(sample, image)
                    decoded = decode_mask(
                        runtime.model.sam, features, projected, multimask_output=False, bridge=bridge
                    )
                    mask = (
                        upsample_logits(decoded.low_res_logits.detach().float(), gt_a.shape).squeeze() > 0.0
                    ).cpu().numpy()
                    entry["mask"] = mask
                    entry["iou_a"] = _iou_vs(mask, gt_a)
                    entry["iou_b"] = _iou_vs(mask, gt_b)
                out[f"{pair['a']['image_id']}:{label}"] = entry
    del runtime
    torch.cuda.empty_cache()
    return out


def spectrum_figure(payloads: dict[str, dict], out_path: Path) -> bool:
    import cv2

    width, height = 760, 380
    canvas = np.full((height, width, 3), 18, np.uint8)
    cv2.putText(
        canvas,
        "projected prompt singular-value spectrum (10 same-image pairs)",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    colours = {"U_C": (60, 76, 231), "U_L": (113, 204, 46), "P_C": (219, 152, 52), "P_L": (182, 89, 155)}
    for index, (arm, payload) in enumerate(payloads.items()):
        values = ((payload.get("representation") or {}).get("projected") or {}).get("top10_singular_values")
        if not values:
            continue
        top = max(values) or 1.0
        points = [
            (40 + i * 68, height - 40 - int((value / top) * (height - 110))) for i, value in enumerate(values)
        ]
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, colours.get(arm, (200, 200, 200)), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 3, colours.get(arm, (200, 200, 200)), -1)
        cv2.putText(
            canvas,
            f"{arm}  top1={((payload['representation']['projected']).get('variance_explained_top1'))}  "
            f"effrank={((payload['representation']['projected']).get('effective_rank'))}",
            (16, 60 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            colours.get(arm, (200, 200, 200)),
            1,
            cv2.LINE_AA,
        )
    return bool(cv2.imwrite(str(out_path), canvas))


def main(argv: list[str] | None = None) -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=4)
    args = parser.parse_args(argv)

    payload = json.loads(TASK6B_SUBSETS.read_text(encoding="utf-8"))
    val = {r["sample_id"]: r for r in data_mod.read_records("val")}
    paired_ids = payload["paired_probe"]["sample_ids"]
    pairs = []
    for index in range(0, min(len(paired_ids), args.images * 2), 2):
        pairs.append({"a": val[paired_ids[index]], "b": val[paired_ids[index + 1]]})

    predictions: dict[str, dict] = {}
    for arm, _subset, bridge in ARMS:
        predictions[arm] = collect_arm(arm, _subset, bridge, pairs)

    SAMPLES.mkdir(parents=True, exist_ok=True)
    for stale in SAMPLES.glob("*.png"):
        stale.unlink()

    written: list[str] = []
    for index, pair in enumerate(pairs):
        sample_a = data_mod.to_sample(pair["a"])
        sample_b = data_mod.to_sample(pair["b"])
        image = sample_a.image_rgb()
        gt_a = np.asarray(sample_a.target_mask()).astype(bool)
        gt_b = np.asarray(sample_b.target_mask()).astype(bool)

        columns = [
            _label(_tile(image), [f"image {pair['a']['image_id']}", "source"]),
            _label(_tile(image, gt_a, (0, 215, 255)), [f"A: {sample_a.query_type}", "ground truth A"]),
            _label(_tile(image, gt_b, (0, 215, 255)), [f"B: {sample_b.query_type}", "ground truth B"]),
        ]
        for arm, _subset, _bridge in ARMS:
            for label in ("a", "b"):
                entry = predictions[arm][f"{pair['a']['image_id']}:{label}"]
                mask = entry["mask"]
                colour = (60, 76, 231) if arm.endswith("C") else (113, 204, 46)
                columns.append(
                    _label(
                        _tile(image, mask, colour) if mask is not None else _tile(image),
                        [
                            f"{arm}  [{label}]",
                            f"IoU own {entry['iou_a' if label == 'a' else 'iou_b']:.3f}",
                            f"other {entry['iou_b' if label == 'a' else 'iou_a']:.3f}"
                            + ("" if entry["seg_count"] == 1 else "  no [SEG]"),
                        ],
                    )
                )
        strip = np.hstack(columns)
        header = np.full((46, strip.shape[1], 3), 12, np.uint8)
        cv2.putText(
            header,
            f"image {pair['a']['image_id']}   A: {sample_a.instruction_zh[:46]}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            header,
            f"                  B: {sample_b.instruction_zh[:46]}",
            (8, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        name = f"{index:02d}_paired_{pair['a']['image_id']}.png"
        cv2.imwrite(str(SAMPLES / name), np.vstack([header, strip]))
        written.append(name)

    prompt_payloads = {}
    for arm, _subset, _bridge in ARMS:
        path = EVAL / f"task6c_prompt_{arm}.json"
        if path.is_file():
            prompt_payloads[arm] = json.loads(path.read_text(encoding="utf-8"))
    if prompt_payloads:
        spectrum_figure(prompt_payloads, SAMPLES / "projected_prompt_spectrum.png")
        written.append("projected_prompt_spectrum.png")

    print(f"[viz] wrote {len(written)} files to {SAMPLES.relative_to(REPO_ROOT).as_posix()}: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
