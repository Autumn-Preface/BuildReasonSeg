"""Task 7C Parts H-I — reduced-adaptation training of the 20-class rehearsal ProgramHead.

Initialization is **only** the stable Task 6T checkpoint (never the failed Task 7B one). Architecture, text
input, tokenizer/prompt format, LoRA parameters and the ProgramHead are preserved exactly; the base backbone
stays frozen outside the existing trainables.

Internal split (section 13, seed 20261001): per class 90 % train / 10 % internal holdout by normalized
prompt hash, so every one of the 20 classes has holdout support (L3 80/class, non-L3 30/class) and
train↔holdout normalized overlap is zero.

Frozen protocol (section 14): AdamW, ProgramHead lr **1e-4**, LoRA lr **2e-5**, weight decay 1e-4,
effective batch 32, seed 20261001, max 5 epochs, early-stopping patience 2, bfloat16 AMP, grad clip 1.0,
no scheduler, no sweep. Selection (section 15) uses the internal holdout only, ranking
`selection_primary = min(macro_f1_20, l3_macro_recall)`, then minimum class recall, macro F1, accuracy and
finally the earlier epoch.

    python scripts/task7c_train_parser.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
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
from scripts.task7b_build_parser_data import sha256_file  # noqa: E402
from scripts.task7c_build_rehearsal import L3_PROGRAMS, normalize_prompt  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CONFIG_PATH = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
BASELINE = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
BASELINE_SHA256 = "4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e"
TASK7B_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7b" \
    / "program_parser_l3_hardened_v1.pt"
ROWS = REPO_ROOT / "artifacts" / "task7c" / "parser_rehearsal" / "parser_rehearsal_rows.jsonl"
AUDIT = EVAL / "task7c_training_data_audit.json"
OUT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7c" / "program_parser_l3_rehearsal_v1.pt"
OUT = EVAL / "task7c_training_summary.json"
SEED = 20261001
HOLDOUT_FRACTION = 0.10
PROTOCOL = {"optimizer": "AdamW", "program_head_lr": 1.0e-4, "lora_lr": 2.0e-5,
            "weight_decay": 1.0e-4, "effective_batch_size": 32, "max_epochs": 5,
            "early_stopping_patience": 2, "amp": "bfloat16", "grad_clip_norm": 1.0,
            "scheduler": "none", "sweep": False, "seed": SEED}


def macro_f1(confusion: np.ndarray) -> float:
    scores = []
    for index in range(confusion.shape[0]):
        tp = confusion[index, index]
        fp = confusion[:, index].sum() - tp
        fn = confusion[index, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(scores))


def load_rows() -> list[dict]:
    rows = []
    with ROWS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per class 90/10 by normalized-prompt group; every class keeps holdout support."""

    by_program: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_program[row["program"]][normalize_prompt(row["prompt"])].append(row)
    train, holdout = [], []
    for program in EXPECTED_PROGRAM_IDS:
        groups = by_program.get(program, {})
        ordered = sorted(groups.items(), key=lambda item: hashlib.sha256(
            f"{SEED}:{item[0]}".encode("utf-8")).hexdigest())
        cut = max(1, int(round(len(ordered) * HOLDOUT_FRACTION))) if ordered else 0
        for _key, members in ordered[:cut]:
            holdout.extend(members)
        for _key, members in ordered[cut:]:
            train.extend(members)
    return train, holdout


def evaluate(runtime, rows: list[dict], batch_size: int = 32) -> dict:
    import torch

    index_of = {program: index for index, program in enumerate(EXPECTED_PROGRAM_IDS)}
    confusion = np.zeros((len(EXPECTED_PROGRAM_IDS), len(EXPECTED_PROGRAM_IDS)), dtype=np.int64)
    runtime.qwen.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start: start + batch_size]
            batch = runtime.build_batch([row["prompt"] for row in chunk],
                                        [row["program"] for row in chunk]).to(runtime.device)
            logits, _hidden = runtime.forward(batch)
            for row, prediction in zip(chunk, logits.argmax(dim=1).cpu().numpy()):
                confusion[index_of[row["program"]], int(prediction)] += 1
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    supports = confusion.sum(axis=1)
    recalls = {program: (float(confusion[index, index] / supports[index])
                         if supports[index] else None)
               for index, program in enumerate(EXPECTED_PROGRAM_IDS)}
    represented = {program: value for program, value in recalls.items() if value is not None}
    l3_recalls = [recalls[program] for program in L3_PROGRAMS if recalls[program] is not None]
    l3_macro = float(np.mean(l3_recalls)) if l3_recalls else 0.0
    accuracy = correct / total if total else 0.0
    macro = macro_f1(confusion)
    return {
        "count": total, "correct": correct, "accuracy": accuracy,
        "macro_f1_20": macro, "l3_macro_recall": l3_macro,
        "min_class_recall": min(represented.values()) if represented else 0.0,
        "per_class_recall": recalls,
        "per_class_support": {program: int(supports[index])
                              for index, program in enumerate(EXPECTED_PROGRAM_IDS)},
        "confusion_matrix": confusion.tolist(),
        "selection_primary": min(macro, l3_macro),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=PROTOCOL["max_epochs"])
    args = parser.parse_args(argv)

    started = time.time()
    baseline_sha = sha256_file(BASELINE) if BASELINE.is_file() else None
    if baseline_sha != BASELINE_SHA256:
        write_json(OUT, {"_doc": "Task 7C section 2.", "task": "7C",
                         "verdict": "BASELINE_PARSER_UNAVAILABLE",
                         "baseline": {"path": str(BASELINE), "sha256": baseline_sha,
                                      "expected": BASELINE_SHA256}})
        print("[7c.train] STOP BASELINE_PARSER_UNAVAILABLE", flush=True)
        return 2
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["verdict"] != "REHEARSAL_CLEAN":
        write_json(OUT, {"_doc": "Task 7C section 12.", "task": "7C",
                         "verdict": "INVALID_EXPERIMENT", "reason": "rehearsal data audit not clean"})
        print("[7c.train] STOP INVALID_EXPERIMENT (rehearsal audit)", flush=True)
        return 2

    import torch

    rows = load_rows()
    train_rows, holdout_rows = stratified_split(rows)
    holdout_support = Counter(row["program"] for row in holdout_rows)
    train_normalized = {normalize_prompt(row["prompt"]) for row in train_rows}
    holdout_normalized = {normalize_prompt(row["prompt"]) for row in holdout_rows}
    overlap = train_normalized & holdout_normalized
    missing_support = [program for program in EXPECTED_PROGRAM_IDS if not holdout_support[program]]
    print(f"[7c.train] rows {len(rows)} train {len(train_rows)} holdout {len(holdout_rows)} "
          f"overlap {len(overlap)} missing_support {missing_support}", flush=True)

    cfg = load_config(CONFIG_PATH)
    optimizer_cfg = dict(cfg.get("optimizer", {}))
    runtime = build_program_parser(cfg, device=args.device, verbose=True)
    load_parser_checkpoint(BASELINE, runtime)
    trainable = int(sum(parameter.numel() for parameter in runtime.qwen.parameters()
                        if parameter.requires_grad))
    total = int(sum(parameter.numel() for parameter in runtime.qwen.parameters()))
    groups = runtime.trainable_parameter_groups(PROTOCOL["lora_lr"], PROTOCOL["program_head_lr"],
                                                PROTOCOL["weight_decay"])
    optimizer = torch.optim.AdamW(groups, betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))))
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()

    history = []
    best = {"key": (-1.0, -1.0, -1.0, -1.0, 0), "epoch": 0, "holdout": None}
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        runtime.qwen.train()
        generator = np.random.default_rng(SEED + epoch)
        order = generator.permutation(len(train_rows))
        losses = []
        for step, start in enumerate(range(0, len(order), args.microbatch)):
            chunk = [train_rows[int(index)] for index in order[start: start + args.microbatch]]
            batch = runtime.build_batch([row["prompt"] for row in chunk],
                                        [row["program"] for row in chunk]).to(runtime.device)
            result = runtime.train_step(batch, None, PROTOCOL["grad_clip_norm"])
            losses.append(result["loss"])
            if (step + 1) % args.accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in runtime.parameters() if parameter.requires_grad],
                    PROTOCOL["grad_clip_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if len(losses) % args.accumulation != 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in runtime.parameters() if parameter.requires_grad],
                PROTOCOL["grad_clip_norm"])
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(runtime, holdout_rows)
        key = (metrics["selection_primary"], metrics["min_class_recall"], metrics["macro_f1_20"],
               metrics["accuracy"], -epoch)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "holdout_selection_primary": metrics["selection_primary"],
                        "holdout_macro_f1_20": metrics["macro_f1_20"],
                        "holdout_l3_macro_recall": metrics["l3_macro_recall"],
                        "holdout_min_class_recall": metrics["min_class_recall"],
                        "holdout_accuracy": metrics["accuracy"], "microbatches": len(losses)})
        print(f"[7c.train] epoch {epoch}: loss {np.mean(losses):.4f} | holdout primary "
              f"{metrics['selection_primary']:.4f} macroF1 {metrics['macro_f1_20']:.4f} L3 "
              f"{metrics['l3_macro_recall']:.4f} minRecall {metrics['min_class_recall']:.4f} acc "
              f"{metrics['accuracy']:.4f}", flush=True)
        if key > best["key"]:
            best = {"key": key, "epoch": epoch, "holdout": metrics}
            OUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            save_parser_checkpoint(OUT_CHECKPOINT, runtime, step=epoch,
                                   metrics={"holdout_selection_primary": metrics["selection_primary"],
                                            "holdout_macro_f1_20": metrics["macro_f1_20"],
                                            "holdout_l3_macro_recall": metrics["l3_macro_recall"],
                                            "holdout_min_class_recall": metrics["min_class_recall"]},
                                   optimizer=None)
            stale = 0
        else:
            stale += 1
            if stale >= PROTOCOL["early_stopping_patience"]:
                print(f"[7c.train] early stop at epoch {epoch}", flush=True)
                break

    ok = OUT_CHECKPOINT.is_file()
    payload = {
        "_doc": ("Task 7C sections 13-15. 20-class rehearsal training of the same Qwen3-VL-2B text-only "
                 "20-class ProgramHead, initialized ONLY from the stable Task 6T checkpoint, with the fixed "
                 "differential learning rates (ProgramHead 1e-4, LoRA 2e-5) and section-14 protocol. "
                 "Checkpoint selection uses only the internal holdout."),
        "task": "7C", "stage": "I-training",
        "baseline": {"path": str(BASELINE), "sha256": baseline_sha, "expected_sha256": BASELINE_SHA256,
                     "matches": baseline_sha == BASELINE_SHA256, "initialized_from_task7b": False,
                     "task7b_checkpoint": str(TASK7B_CHECKPOINT),
                     "task7b_checkpoint_used": False},
        "architecture": {"model_family": "Qwen3-VL-2B", "text_only": True, "image_input": False,
                         "classes": len(EXPECTED_PROGRAM_IDS),
                         "trainable_policy": "Task 6M/6T policy: LoRA adapters + ProgramHead (backbone "
                                             "frozen)",
                         "trainable_parameters": trainable, "total_parameters": total,
                         "backbone_unfrozen": False},
        "protocol": {**PROTOCOL, "microbatch": args.microbatch, "accumulation": args.accumulation,
                     "actual_effective_batch": args.microbatch * args.accumulation,
                     "selection": [
                         "min(macro_f1_20, l3_macro_recall)", "minimum class recall", "macro F1",
                         "accuracy", "earlier epoch"]},
        "data": {"rows": len(rows), "train_rows": len(train_rows), "holdout_rows": len(holdout_rows),
                 "normalized_overlap": len(overlap),
                 "holdout_support": {program: holdout_support.get(program, 0)
                                     for program in EXPECTED_PROGRAM_IDS},
                 "classes_without_holdout_support": missing_support,
                 "external_eval_used_for_selection": False,
                 "original_v02_train_text_used": False},
        "history": history, "selected_epoch": best["epoch"],
        "selection_metrics": {"selection_primary": best["key"][0], "min_class_recall": best["key"][1],
                              "macro_f1_20": best["key"][2], "accuracy": best["key"][3]},
        "holdout": best["holdout"],
        "checkpoint": {"path": str(OUT_CHECKPOINT), "exists": ok,
                       "sha256": sha256_file(OUT_CHECKPOINT) if ok else None,
                       "bytes": OUT_CHECKPOINT.stat().st_size if ok else None, "committed": False},
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if args.device != "cpu" else None,
        "training_performed": True, "test_split_used": False, "hyperparameter_sweep": False,
    }
    payload["verdict"] = "PARSER_TRAINED" if ok else "PARSER_TRAINING_FAILED"
    write_json(OUT, payload)
    print(f"[7c.train] selected epoch {best['epoch']} primary {best['key'][0]:.4f} macroF1 "
          f"{best['key'][2]:.4f} L3 {best['holdout']['l3_macro_recall']:.4f} minRecall "
          f"{best['key'][1]:.4f} -> {payload['verdict']}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
