#!/usr/bin/env python
"""Task 6B: diagnose WHY the mask is (or is not) instruction-dependent.

    python scripts/task6b_diagnose_prompt.py [--config ...] [--checkpoint PATH] [--pairs 8]

The paired probe asks whether two different instructions on the same image
produce different masks. When it fails, the cause can sit in three different
places and the fix is different in each case, so this script separates them:

1. **`[SEG]` hidden state** -- is the 2048-d hidden state at the `[SEG]` position
   different for two instructions on the same image? Compared against the same
   quantity measured across *different* images, which gives the scale.
2. **projection** -- how far apart are the two 256-d sparse prompts after the
   projection MLP?
3. **mask decoder** -- how different are the two decoded masks, and how does each
   compare with the two ground-truth masks?

Only the teacher-forced path is used, so these numbers are not confounded by
generation quality. Ground truth is used for scoring only.

Writes `evaluation/task6b_prompt_diagnosis.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint, sha256_file, write_json  # noqa: E402
from buildreasonseg_mvp.metrics import mask_iou_from_logits  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, set_phase_trainables  # noqa: E402

SUBSET_JSON = REPO_ROOT / "evaluation" / "task6b_subset_ids.json"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def _iou(logits_a, logits_b, shape) -> float:
    """IoU between two predicted logit maps (thresholded at 0)."""

    mask_a = (torch.nn.functional.interpolate(
        logits_a.float(), size=shape, mode="bilinear", align_corners=False
    ) > 0.0)
    mask_b = (torch.nn.functional.interpolate(
        logits_b.float(), size=shape, mode="bilinear", align_corners=False
    ) > 0.0)
    intersection = (mask_a & mask_b).sum().item()
    union = (mask_a | mask_b).sum().item()
    return intersection / union if union else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--out-tag", default="original_recipe")
    args = parser.parse_args(argv)

    cfg = load_config(args.config or (REPO_ROOT / "configs" / "mvp" / "task6b_2b_minitrain_adjusted.yaml"))
    payload = json.loads(SUBSET_JSON.read_text(encoding="utf-8"))
    records = {r["sample_id"]: r for r in data_mod.read_records("val")}
    pairs = payload["paired_probe"]["pairs"][: args.pairs]

    runtime = build_runtime(cfg, device="cuda", verbose=False)
    set_phase_trainables(runtime.model, "B")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(cfg["paths"]["checkpoints"]) / "best_joint.pt"
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
    checkpoint_path = Path(checkpoint_path)
    restore = load_checkpoint(checkpoint_path, runtime.model) if checkpoint_path.is_file() else None
    print(f"[diagnose] checkpoint {checkpoint_path} restore={restore}", flush=True)

    runtime.model.eval()

    entries: list[dict] = []
    with torch.no_grad():
        for pair in pairs:
            left = records.get(pair["a"])
            right = records.get(pair["b"])
            if left is None or right is None:
                continue
            measured = {}
            for label, record in (("a", left), ("b", right)):
                sample = data_mod.to_sample(record)
                image = sample.image_rgb()
                batch, _image = runtime.prepare(sample)
                features, _cached = runtime.features_for(sample, image)
                output = runtime.model(batch.to(runtime.device), features)
                measured[label] = {
                    "sample_id": sample.sample_id,
                    "query_type": sample.query_type,
                    "seg_hidden": output.seg_hidden.detach().float().flatten().cpu(),
                    "projected": output.projected.detach().float().flatten().cpu(),
                    "logits": output.mask_logits.detach().float().cpu(),
                    "gt": torch.as_tensor(sample.target_mask()).float(),
                    "gt_shape": sample.target_mask().shape,
                }

            a, b = measured["a"], measured["b"]
            shape = a["gt_shape"]
            entries.append(
                {
                    "image_id": pair["image_id"],
                    "a": a["sample_id"],
                    "b": b["sample_id"],
                    "a_query_type": a["query_type"],
                    "b_query_type": b["query_type"],
                    "seg_hidden_cosine_same_image": _cosine(a["seg_hidden"], b["seg_hidden"]),
                    "seg_hidden_l2_same_image": float((a["seg_hidden"] - b["seg_hidden"]).norm()),
                    "projected_cosine_same_image": _cosine(a["projected"], b["projected"]),
                    "projected_l2_same_image": float((a["projected"] - b["projected"]).norm()),
                    "prediction_iou_a_vs_b": _iou(a["logits"], b["logits"], shape),
                    "a_iou_on_a": mask_iou_from_logits(a["logits"], a["gt"], shape),
                    "a_iou_on_b": mask_iou_from_logits(a["logits"], b["gt"], shape),
                    "b_iou_on_a": mask_iou_from_logits(b["logits"], a["gt"], shape),
                    "b_iou_on_b": mask_iou_from_logits(b["logits"], b["gt"], shape),
                }
            )

    # Reference scale: the same measurements between DIFFERENT images.
    cross_hidden: list[float] = []
    cross_projected: list[float] = []
    sample_records = [records[p["a"]] for p in pairs if p["a"] in records]
    with torch.no_grad():
        hiddens = []
        projected = []
        for record in sample_records[:12]:
            sample = data_mod.to_sample(record)
            image = sample.image_rgb()
            batch, _image = runtime.prepare(sample)
            features, _cached = runtime.features_for(sample, image)
            output = runtime.model(batch.to(runtime.device), features)
            hiddens.append(output.seg_hidden.detach().float().flatten().cpu())
            projected.append(output.projected.detach().float().flatten().cpu())
    for i in range(len(hiddens)):
        for j in range(i + 1, len(hiddens)):
            cross_hidden.append(_cosine(hiddens[i], hiddens[j]))
            cross_projected.append(_cosine(projected[i], projected[j]))

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def mm(values: list[float]) -> dict:
        return {
            "mean": mean(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    report = {
        "_doc": (
            "Task 6B prompt-pathway diagnosis. Teacher-forced only; ground truth is used for "
            "scoring only."
        ),
        "task": "6B",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
            "restore": restore,
        },
        "n_pairs": len(entries),
        "same_image_different_instruction": {
            "seg_hidden_cosine": mm([e["seg_hidden_cosine_same_image"] for e in entries]),
            "seg_hidden_l2": mm([e["seg_hidden_l2_same_image"] for e in entries]),
            "projected_cosine": mm([e["projected_cosine_same_image"] for e in entries]),
            "projected_l2": mm([e["projected_l2_same_image"] for e in entries]),
            "prediction_iou_a_vs_b": mm([e["prediction_iou_a_vs_b"] for e in entries]),
        },
        "different_image_reference": {
            "seg_hidden_cosine": mm(cross_hidden),
            "projected_cosine": mm(cross_projected),
        },
        "pairs": entries,
    }
    out = REPO_ROOT / "evaluation" / f"task6b_prompt_diagnosis_{args.out_tag}.json"
    write_json(out, report)

    same = report["same_image_different_instruction"]
    cross = report["different_image_reference"]
    print(f"[diagnose] [SEG] hidden cosine  same-image {same['seg_hidden_cosine']['mean']:.6f}  "
          f"across-image {cross['seg_hidden_cosine']['mean']:.6f}")
    print(f"[diagnose] projected cosine     same-image {same['projected_cosine']['mean']:.6f}  "
          f"across-image {cross['projected_cosine']['mean']:.6f}")
    print(f"[diagnose] prediction IoU a vs b (same image, different instruction): "
          f"{same['prediction_iou_a_vs_b']['mean']:.6f}")
    print(f"[diagnose] wrote {out.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
