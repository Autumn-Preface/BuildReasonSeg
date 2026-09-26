#!/usr/bin/env python
"""Task 6B: build the validation visualization pack.

    python scripts/task6b_visualize.py [--config ...] [--checkpoint PATH] [--per-category N]

Reads `evaluation/task6b_validation.json` to pick representative records, then
re-runs the free-generation path for them from the best_joint checkpoint and
renders panels into `evaluation/task6b_samples/`:

* successful L1 / L2 / nontrivial L3 examples;
* hard or failed examples (missing or multiple `[SEG]`);
* at least four paired-image examples.

Panels show the source image, the instruction, the generated reasoning, `[SEG]`
validity, the ground-truth outline, the predicted outline and the IoU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.checkpointing import load_checkpoint  # noqa: E402
from buildreasonseg_mvp.metrics import mask_iou_from_logits, upsample_logits  # noqa: E402
from buildreasonseg_mvp.qwen_seg import generate_with_seg, seg_hidden_from_full_sequence  # noqa: E402
from buildreasonseg_mvp.runtime import build_runtime, load_config  # noqa: E402
from buildreasonseg_mvp.sam2_bridge import decode_mask  # noqa: E402
from buildreasonseg_mvp.visualize import build_contact_sheet, render_task6b_panel  # noqa: E402

VALIDATION_JSON = REPO_ROOT / "evaluation" / "task6b_validation.json"
SUBSET_JSON = REPO_ROOT / "evaluation" / "task6b_subset_ids.json"
SAMPLES_DIR = REPO_ROOT / "evaluation" / "task6b_samples"


def select_records(validation: dict, per_category: int) -> list[tuple[str, dict]]:
    free = validation.get("per_record_free_generation", [])
    success = [r for r in free if r["seg_valid"]]
    failure = [r for r in free if not r["seg_valid"]]
    by_level: dict[str, list[dict]] = defaultdict(list)
    for record in sorted(success, key=lambda r: -r["iou"]):
        by_level[record["level_key"]].append(record)

    chosen: list[tuple[str, dict]] = []
    for level_key in ("L1", "L2", "L3_nontrivial"):
        for record in by_level.get(level_key, [])[:per_category]:
            chosen.append((f"success_{level_key}", record))
    for record in sorted(failure, key=lambda r: r["sample_id"])[:per_category]:
        chosen.append(("failure", record))
    # anything remaining that is still worth showing
    for record in sorted(success, key=lambda r: r["iou"])[: max(1, per_category // 2)]:
        if record not in [entry for _name, entry in chosen]:
            chosen.append(("hard_success", record))
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args(argv)

    cfg = load_config(args.config or (REPO_ROOT / "configs" / "mvp" / "task6b_2b_minitrain.yaml"))
    validation = json.loads(VALIDATION_JSON.read_text(encoding="utf-8")) if VALIDATION_JSON.is_file() else {}
    subsets = json.loads(SUBSET_JSON.read_text(encoding="utf-8"))

    runtime = build_runtime(cfg, device="cuda", verbose=False)
    # The checkpoint directory is part of the config, so the adjusted run renders
    # its own best_joint checkpoint rather than the original run's.
    default_checkpoint = Path(cfg["paths"]["checkpoints"]) / "best_joint.pt"
    if not default_checkpoint.is_absolute():
        default_checkpoint = REPO_ROOT / default_checkpoint
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    if checkpoint_path.is_file():
        info = load_checkpoint(checkpoint_path, runtime.model)
        print(f"[viz] restored {checkpoint_path.name}: step={info['step']} metrics={info['metrics']}")
    else:
        print(f"[viz] WARNING: {checkpoint_path} not found; rendering the untrained model")

    runtime.model.eval()
    if hasattr(runtime.qwen, "gradient_checkpointing_disable"):
        runtime.qwen.gradient_checkpointing_disable()

    records_by_id = {r["sample_id"]: r for r in data_mod.read_records("val")}
    # `select_records` works on the per-record *validation* entries (they carry
    # `iou` / `seg_valid`); rendering needs the dataset records, so translate by
    # sample id rather than passing validation entries to `to_sample`.
    chosen = [
        (tag, records_by_id[entry["sample_id"]])
        for tag, entry in select_records(validation, args.per_category)
        if entry["sample_id"] in records_by_id
    ]

    # paired examples: reuse the recorded pairs, take the first few
    pairs = subsets["paired_probe"]["pairs"][:4]
    paired_entries = []
    for pair in pairs:
        for label in ("a", "b"):
            record = records_by_id.get(pair[label])
            if record is not None:
                paired_entries.append((f"paired_{pair['image_id']}_{label}", record))

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for stale in SAMPLES_DIR.glob("*.png"):
        stale.unlink()

    written: list[str] = []
    max_new_tokens = int(cfg["inference"]["validation_max_new_tokens"])

    def render(tag: str, record: dict, index: int) -> None:
        sample = data_mod.to_sample(record)
        image = sample.image_rgb()
        gt = sample.target_mask()
        generation = generate_with_seg(
            runtime.model.qwen,
            runtime.processor,
            runtime.tokenizer,
            image,
            sample.instruction_zh,
            max_new_tokens=max_new_tokens,
            seg_token_id=runtime.model.seg_token_id,
        )
        valid = generation["seg_count"] == 1
        predicted = None
        iou = None
        if valid:
            position = generation["seg_positions"][0]
            with torch.no_grad():
                hidden = seg_hidden_from_full_sequence(
                    runtime.model.qwen,
                    runtime.processor,
                    image,
                    generation["token_ids"],
                    position,
                    instruction=sample.instruction_zh,
                )
                projected = runtime.projection(hidden)
                features, _ = runtime.features_for(sample, image)
                decoded = decode_mask(runtime.model.sam, features, projected, multimask_output=False)
                iou = mask_iou_from_logits(decoded.low_res_logits, torch.as_tensor(gt).float(), gt.shape)
                predicted = (
                    upsample_logits(decoded.low_res_logits.detach().float(), gt.shape).squeeze() > 0.0
                ).cpu().numpy()

        header = (
            f"L{sample.level} {sample.query_type} | target {sample.target_component_id} "
            f"| [SEG]x{generation['seg_count']}"
        )
        lines = [
            f"instruction: {sample.instruction_zh}",
            f"generated  : {generation['text'][:110]}",
            f"expected   : {sample.assistant_text[:110]}",
        ]
        name = f"{index:02d}_{tag}_{sample.sample_id[-14:]}.png"
        if render_task6b_panel(
            SAMPLES_DIR / name, image, gt, predicted, header, lines, valid, iou
        ):
            written.append(name)

    index = 0
    for tag, record in chosen:
        render(tag, record, index)
        index += 1
    for tag, record in paired_entries:
        render(tag, record, index)
        index += 1

    if written:
        build_contact_sheet(SAMPLES_DIR, SAMPLES_DIR / "contact_sheet.png")
    print(f"[viz] wrote {len(written)} panels (+ contact sheet) to {SAMPLES_DIR.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
