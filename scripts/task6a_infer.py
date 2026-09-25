#!/usr/bin/env python
"""Task 6A inference: image + instruction only -> `[SEG]` -> mask.

    python scripts/task6a_infer.py [--checkpoint artifacts/checkpoints/last.pt]
                                   [--split train] [--limit 4] [--tag infer]

This is the free-generation path, with no ground-truth reasoning, no target id
and no geometry supplied at any point:

    image + instruction
        -> Qwen3-VL generates assistant text
        -> locate the generated `[SEG]`
        -> re-forward the COMPLETE generated sequence to read that token's hidden state
        -> projection -> SAM2.1 mask decoder -> mask

Ground-truth masks are used only afterwards, to score and to draw the panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.metrics import collapse_flags, mask_iou_from_logits, upsample_logits  # noqa: E402
from buildreasonseg_mvp.qwen_seg import generate_with_seg, seg_hidden_from_full_sequence  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config, vram  # noqa: E402
from buildreasonseg_mvp.sam2_bridge import decode_mask  # noqa: E402
from buildreasonseg_mvp.visualize import render_panel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--tag", default="infer")
    parser.add_argument(
        "--from-subset",
        default=None,
        choices=["smoke_pair", "overfit_set"],
        help="run the recorded deterministic subset instead of the first N records",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    runtime = build_runtime(cfg, device="cuda", verbose=True)

    if args.checkpoint:
        info = load_checkpoint(Path(args.checkpoint), runtime.model)
        print(f"[infer] restored {args.checkpoint}: step={info['step']} metrics={info['metrics']}")

    runtime.model.eval()
    if hasattr(runtime.qwen, "gradient_checkpointing_disable"):
        runtime.qwen.gradient_checkpointing_disable()

    records = data_mod.read_records(args.split)[: args.limit]
    out_dir = REPO_ROOT / "evaluation" / "task6a_samples"
    if args.from_subset:
        import json

        payload = json.loads(
            (REPO_ROOT / "evaluation" / "task6a_subset_ids.json").read_text(encoding="utf-8")
        )
        wanted = payload[args.from_subset]["sample_ids"]
        by_id = {r["sample_id"]: r for r in data_mod.read_records("train")}
        records = [by_id[sample_id] for sample_id in wanted]
    results = []
    with torch.no_grad():
        for record in records:
            sample = data_mod.to_sample(record)
            image = sample.image_rgb()
            generation = generate_with_seg(
                runtime.model.qwen,
                runtime.processor,
                runtime.tokenizer,
                image,
                sample.instruction_zh,
                max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
                seg_token_id=runtime.model.seg_token_id,
            )
            entry = {
                "sample_id": sample.sample_id,
                "query_type": sample.query_type,
                "level": sample.level,
                "instruction_zh": sample.instruction_zh,
                "generated_text": generation["text"],
                "seg_count": generation["seg_count"],
                "emitted_exactly_once": generation["seg_count"] == 1,
                "ground_truth_used_at_inference": False,
            }
            if generation["seg_count"] >= 1:
                position = generation["seg_positions"][0]
                hidden = seg_hidden_from_full_sequence(
                    runtime.model.qwen, runtime.processor, image, generation["token_ids"], position
                )
                projected = runtime.projection(hidden)
                features = runtime.sam_encoder.encode(image)
                decoded = decode_mask(runtime.model.sam, features, projected, multimask_output=False)
                gt = sample.target_mask()
                iou = mask_iou_from_logits(decoded.low_res_logits, torch.as_tensor(gt).float(), gt.shape)
                entry["iou"] = iou
                entry["collapse"] = collapse_flags(decoded.low_res_logits.detach().float())
                predicted = (
                    upsample_logits(decoded.low_res_logits.detach().float(), gt.shape).squeeze() > 0.0
                ).cpu().numpy()
                render_panel(
                    out_dir / f"{args.tag}_{sample.sample_id[-12:]}.png",
                    image,
                    gt,
                    predicted,
                    [sample.instruction_zh, sample.query_type, sample.sample_id],
                    iou,
                )
            results.append(entry)
            print(f"[infer] {sample.sample_id}: [SEG]x{generation['seg_count']} iou={entry.get('iou')}")

    report = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "limit": args.limit,
        "seg_emission_rate": sum(1 for r in results if r["emitted_exactly_once"]) / max(len(results), 1),
        "mean_iou": (
            sum(r["iou"] for r in results if "iou" in r) / max(1, sum(1 for r in results if "iou" in r))
        ),
        "vram": vram(),
        "results": results,
    }
    out = REPO_ROOT / "artifacts" / "stage_outputs" / f"task6a_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"[infer] emission {report['seg_emission_rate']:.2f} mean IoU {report['mean_iou']:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
