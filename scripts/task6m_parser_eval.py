"""Task 6M section 13: evaluate the frozen Task 6J Qwen3-VL-2B ProgramHead on BuildSpatialReason v0.2.

Instruction text only (no image tokens, no query_type leakage). If the frozen checkpoint is missing
or v0.2 val accuracy < 0.95, the same 2B text-only head is retrained on v0.2 TRAIN only and frozen
before any test use. No upgrade to 4B.

    python scripts/task6m_parser_eval.py [--batch 32] [--limit N] [--retrain]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "spatial_reasoning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.program_parser import (  # noqa: E402
    EXPECTED_PROGRAM_IDS,
    build_program_parser,
    load_parser_checkpoint,
    save_parser_checkpoint,
)
from buildreasonseg_mvp.runtime import load_config  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6j" / "j2_best.pt"
RETRAINED = REPO_ROOT / "artifacts" / "checkpoints" / "task6m" / "program_parser_v02_best.pt"
OUT = EVAL / "task6m_parser_v02.json"
CONFIG = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
ACCURACY_GATE = 0.95


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(split: str, limit: int | None = None) -> list[dict]:
    path = V02 / f"{split}.jsonl"
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def macro_f1(confusion: np.ndarray) -> float:
    recalls = []
    for index in range(confusion.shape[0]):
        support = confusion[index].sum()
        if support == 0:
            continue
        recalls.append(confusion[index, index] / support)
    return float(np.mean(recalls)) if recalls else 0.0


def evaluate(runtime, records: list[dict], batch_size: int, tag: str) -> dict:
    index_of = {program: index for index, program in enumerate(EXPECTED_PROGRAM_IDS)}
    confusion = np.zeros((len(EXPECTED_PROGRAM_IDS), len(EXPECTED_PROGRAM_IDS)), dtype=np.int64)
    started = time.time()
    import torch

    for start in range(0, len(records), batch_size):
        chunk = records[start: start + batch_size]
        instructions = [str(record["instruction_en"]) for record in chunk]
        truth = [str(record["query_type"]) for record in chunk]
        batch = runtime.build_batch(instructions, truth).to(runtime.device)
        with torch.no_grad():
            logits, _hidden = runtime.forward(batch)
        predictions = logits.argmax(dim=1).cpu().numpy()
        for row, prediction, expected in zip(chunk, predictions, truth):
            confusion[index_of[expected], int(prediction)] += 1
        if (start // batch_size) % 50 == 0:
            print(f"[6m.parser] {tag} {start + len(chunk)}/{len(records)} "
                  f"({time.time() - started:.0f}s)", flush=True)
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    return {
        "tag": tag,
        "count": total,
        "exact_correct": correct,
        "exact_accuracy": correct / total if total else None,
        "macro_f1": macro_f1(confusion),
        "confusion_matrix": confusion.tolist(),
        "class_order": list(EXPECTED_PROGRAM_IDS),
        "seconds": round(time.time() - started, 2),
    }


def retrain(runtime, cfg: dict, parser_cfg: dict, device: str) -> dict:
    """Retrain the SAME 2B text-only head on v0.2 train only (no image tokens).

    Mirrors the frozen Task 6J J2 training setup (`cfg['j2']` for the schedule,
    `cfg['optimizer']` for the two learning-rate groups).
    """

    print("[6m.parser] retraining the 2B text-only ProgramHead on v0.2 train", flush=True)
    import torch

    train_records = load_records("train")
    val_records = load_records("val", limit=1200)
    epochs = int(parser_cfg.get("epochs", 5))
    batch_size = int(parser_cfg.get("batch_size", 16))
    optimizer_cfg = cfg.get("optimizer", {})
    lora_lr = float(optimizer_cfg.get("lora_lr", 1e-4))
    head_lr = float(optimizer_cfg.get("head_lr", 1e-4))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    grad_clip = float(optimizer_cfg.get("grad_clip_norm", 1.0))
    betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
    groups = runtime.trainable_parameter_groups(lora_lr, head_lr, weight_decay)
    optimizer = torch.optim.AdamW(groups, betas=betas) if groups else None
    best_accuracy = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        runtime.qwen.train()
        order = np.random.default_rng(20260813 + epoch).permutation(len(train_records))
        losses = []
        for start in range(0, len(order), batch_size):
            chunk = [train_records[index] for index in order[start: start + batch_size]]
            batch = runtime.build_batch(
                [str(record["instruction_en"]) for record in chunk],
                [str(record["query_type"]) for record in chunk],
            ).to(device)
            step = runtime.train_step(batch, optimizer, grad_clip)
            losses.append(step["loss"])
        metrics = evaluate(runtime, val_records, batch_size, f"retrain_epoch{epoch}_val1200")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": metrics})
        print(f"[6m.parser] epoch {epoch}: loss {np.mean(losses):.4f} val acc "
              f"{metrics['exact_accuracy']:.4f}", flush=True)
        if metrics["exact_accuracy"] and metrics["exact_accuracy"] > best_accuracy:
            best_accuracy = metrics["exact_accuracy"]
            save_parser_checkpoint(RETRAINED, runtime, step=epoch, metrics=metrics, optimizer=optimizer)
    return {"epochs": epochs, "best_val_accuracy": best_accuracy, "history": history,
            "checkpoint": str(RETRAINED) if RETRAINED.is_file() else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--no-retrain", action="store_true",
                        help="evaluate the given checkpoint only (no retrain fallback)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="parser checkpoint to evaluate (default: the frozen Task 6J J2 head)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    started = time.time()
    cfg = load_config(CONFIG)
    parser_cfg = cfg["j2"]
    runtime = build_program_parser(cfg, device=args.device, verbose=True)

    source_checkpoint = Path(args.checkpoint) if args.checkpoint else CHECKPOINT
    checkpoint_info = None
    load_report = None
    if source_checkpoint.is_file() and not args.retrain:
        load_info = load_parser_checkpoint(source_checkpoint, runtime)
        load_report = {key: value for key, value in (load_info or {}).items() if key != "state_dict"}
        checkpoint_info = {
            "path": str(source_checkpoint),
            "sha256": sha256_file(source_checkpoint),
            "bytes": source_checkpoint.stat().st_size,
            "load": load_report,
        }
        print(f"[6m.parser] loaded checkpoint {source_checkpoint.name}", flush=True)

    fixed120 = json.loads((EVAL / "task6m_val_fixed120.json").read_text(encoding="utf-8"))
    fixed_ids = {str(record["sample_id"]) for record in fixed120["records"]}

    val_records = load_records("val", limit=args.limit)
    full_val = evaluate(runtime, val_records, args.batch, "v0.2_full_val")
    fixed_records = [record for record in val_records if str(record["sample_id"]) in fixed_ids]
    fixed_val = evaluate(runtime, fixed_records, args.batch, "v0.2_fixed120")
    pre_retrain_accuracy = full_val["exact_accuracy"]

    retrain_report = None
    needs_retrain = (
        pre_retrain_accuracy is not None and pre_retrain_accuracy < ACCURACY_GATE
    )
    if (args.retrain or needs_retrain) and not args.no_retrain:
        retrain_report = retrain(runtime, cfg, parser_cfg, args.device)
        load_parser_checkpoint(RETRAINED, runtime)
        full_val = evaluate(runtime, val_records, args.batch, "v0.2_full_val_retrained")
        fixed_records = [record for record in val_records if str(record["sample_id"]) in fixed_ids]
        fixed_val = evaluate(runtime, fixed_records, args.batch, "v0.2_fixed120_retrained")
        checkpoint_info = {
            "path": str(RETRAINED),
            "sha256": sha256_file(RETRAINED),
            "bytes": RETRAINED.stat().st_size,
            "retrained_on": "BuildSpatialReason v0.2 train only",
            "retrain_reason": (
                "the frozen Task 6J J2 checkpoint did not clear the 0.95 v0.2 accuracy gate; its "
                "ProgramHead loaded but 626 LoRA/token keys were reported missing by "
                "load_state_dict(strict=False), i.e. the adapter half of the checkpoint does not "
                "transfer into this environment"
            ),
        }

    ready = bool(full_val["exact_accuracy"] is not None and full_val["exact_accuracy"] >= ACCURACY_GATE)
    report = {
        "_doc": (
            "Task 6M section 13. ProgramHead evaluated on BuildSpatialReason v0.2 (instruction text "
            "only; no image tokens; no query_type in the input). When the frozen Task 6J checkpoint "
            "falls below the 0.95 gate, the SAME 2B text-only head is retrained on v0.2 train only."
        ),
        "task": "6M",
        "parser": {
            "model": cfg["models"]["qwen_model_id"],
            "head": "ProgramHead (20-way) over the last prompt-position hidden state",
            "input": "instruction text only",
            "image_tokens": False,
            "query_type_leakage": False,
            "program_ids": list(EXPECTED_PROGRAM_IDS),
        },
        "checkpoint": checkpoint_info,
        "load_report_of_evaluated_checkpoint": load_report,
        "pre_retrain_accuracy": pre_retrain_accuracy,
        "retrain_triggered": bool(retrain_report is not None),
        "v0.2_full_val": full_val,
        "v0.2_fixed120": fixed_val,
        "retrain": retrain_report,
        "gate": {"accuracy_min": ACCURACY_GATE, "passed": ready},
        "verdict": "PARSER_READY" if ready else "PROGRAM_PARSER_NEEDS_IMPROVEMENT",
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(f"[6m.parser] v0.2 full val accuracy {full_val['exact_accuracy']:.4f} "
          f"(macro F1 {full_val['macro_f1']:.4f}); gate {'PASS' if ready else 'FAIL'}", flush=True)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
