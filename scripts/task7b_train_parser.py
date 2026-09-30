"""Task 7B Part E — train the L3 compositional-semantic hardened ProgramHead.

Initialization is the authoritative Task 6T hardened checkpoint
(`4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e`); the architecture, text-only input,
tokenizer/prompt formatting, LoRA/trainable policy and the 20-class ProgramHead are preserved exactly.

Training data (Part D): the cleaned BuildSpatialReason v0.2 train rows that survive the evaluation-overlap
removal plus exactly 4,800 deterministic Task 7B augmentations.

Frozen protocol (section 13): AdamW, learning rate 2e-4, weight decay 1e-4, effective batch size 32,
seed 20261001, max 8 epochs, early-stopping patience 2, bfloat16 AMP, no hyperparameter sweep.
Internal selection (section 12) is a deterministic 90/10 group-disjoint, class-stratified split of the
training data only; checkpoint selection ranks internal-holdout macro F1, then exact accuracy, then the
L3-four-class macro recall.

    python scripts/task7b_train_parser.py
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
from scripts.task7b_build_parser_data import normalize_prompt, sha256_file  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CONFIG_PATH = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
BASELINE = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
BASELINE_SHA256 = "4cbba36b1364b0a85dce7272b967ede1e9b4a3138d330def1a8aab0ec9d44a5e"
TRAIN_DATA = REPO_ROOT / "artifacts" / "task7b" / "parser_train_augmented" \
    / "parser_train_combined.jsonl"
OUT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7b" / "program_parser_l3_hardened_v1.pt"
OUT = EVAL / "task7b_training_summary.json"
SPEC = EVAL / "task7b_train_augmentation_spec.json"
LEAKAGE = EVAL / "task7b_parser_leakage_audit.json"
SEED = 20261001
HOLDOUT_FRACTION = 0.10
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")
PROTOCOL = {"optimizer": "AdamW", "learning_rate": 2.0e-4, "weight_decay": 1.0e-4,
            "effective_batch_size": 32, "max_epochs": 8, "early_stopping_patience": 2,
            "amp": "bfloat16", "sweep": False, "seed": SEED}


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
    with TRAIN_DATA.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_holdout(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic 90/10 split by normalized-prompt groups, stratified by canonical class.

    Every row sharing a normalized prompt is assigned to the same side, so no normalized prompt can appear
    in both the internal train and the internal holdout.
    """

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
    l3_indices = [index_of[program] for program in L3_PROGRAMS]
    l3_recalls = []
    for index in l3_indices:
        support = confusion[index, :].sum()
        l3_recalls.append(float(confusion[index, index] / support) if support else 0.0)
    return {
        "count": total, "correct": correct,
        "accuracy": correct / total if total else None,
        "macro_f1": macro_f1(confusion), "l3_macro_recall": float(np.mean(l3_recalls)),
        "confusion_matrix": confusion.tolist(),
        "per_class_support": {program: int(confusion[index, :].sum())
                              for program, index in index_of.items()},
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
        write_json(OUT, {"_doc": "Task 7B section 2.", "task": "7B",
                         "verdict": "BASELINE_PARSER_UNAVAILABLE",
                         "baseline": {"path": str(BASELINE), "sha256": baseline_sha,
                                      "expected": BASELINE_SHA256}})
        print("[7b.train] STOP BASELINE_PARSER_UNAVAILABLE", flush=True)
        return 2
    leakage = json.loads(LEAKAGE.read_text(encoding="utf-8"))
    if leakage["verdict"] != "LEAKAGE_FREE":
        write_json(OUT, {"_doc": "Task 7B section 11.", "task": "7B",
                         "verdict": "INVALID_EXPERIMENT", "reason": "leakage audit not clean"})
        print("[7b.train] STOP INVALID_EXPERIMENT (leakage)", flush=True)
        return 2
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    import torch

    rows = load_rows()
    train_rows, holdout_rows = stratified_holdout(rows)
    overlap = {normalize_prompt(row["prompt"]) for row in train_rows} & \
        {normalize_prompt(row["prompt"]) for row in holdout_rows}
    print(f"[7b.train] rows {len(rows)} -> train {len(train_rows)} holdout {len(holdout_rows)} "
          f"(normalized overlap {len(overlap)})", flush=True)

    cfg = load_config(CONFIG_PATH)
    optimizer_cfg = dict(cfg.get("optimizer", {}))
    grad_clip = float(optimizer_cfg.get("grad_clip_norm", 1.0))
    runtime = build_program_parser(cfg, device=args.device, verbose=True)
    load_parser_checkpoint(BASELINE, runtime)
    lora_report = runtime.lora_report() if hasattr(runtime, "lora_report") else {}
    trainable = int(sum(parameter.numel() for parameter in runtime.qwen.parameters()
                        if parameter.requires_grad))
    total = int(sum(parameter.numel() for parameter in runtime.qwen.parameters()))

    groups = runtime.trainable_parameter_groups(PROTOCOL["learning_rate"],
                                                PROTOCOL["learning_rate"],
                                                PROTOCOL["weight_decay"])
    optimizer = torch.optim.AdamW(groups, betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))))
    if args.device != "cpu":
        torch.cuda.reset_peak_memory_stats()

    history = []
    best = {"key": (-1.0, -1.0, -1.0), "epoch": 0, "holdout": None}
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        runtime.qwen.train()
        generator = np.random.default_rng(SEED + epoch)
        order = generator.permutation(len(train_rows))
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, start in enumerate(range(0, len(order), args.microbatch)):
            chunk = [train_rows[int(index)] for index in order[start: start + args.microbatch]]
            batch = runtime.build_batch([row["prompt"] for row in chunk],
                                        [row["program"] for row in chunk]).to(runtime.device)
            # gradient accumulation: backward only for every microbatch, step once per effective batch
            result = runtime.train_step(batch, None, grad_clip)
            losses.append(result["loss"])
            if (step + 1) % args.accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in runtime.parameters() if parameter.requires_grad],
                    grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if len(losses) % args.accumulation != 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in runtime.parameters() if parameter.requires_grad],
                grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics = evaluate(runtime, holdout_rows)
        key = (metrics["macro_f1"], metrics["accuracy"] or 0.0, metrics["l3_macro_recall"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "holdout_macro_f1": metrics["macro_f1"],
                        "holdout_accuracy": metrics["accuracy"],
                        "holdout_l3_macro_recall": metrics["l3_macro_recall"],
                        "microbatches": len(losses)})
        print(f"[7b.train] epoch {epoch}: loss {np.mean(losses):.4f} holdout macroF1 "
              f"{metrics['macro_f1']:.4f} acc {metrics['accuracy']:.4f} L3 recall "
              f"{metrics['l3_macro_recall']:.4f}", flush=True)
        if key > best["key"]:
            best = {"key": key, "epoch": epoch, "holdout": metrics}
            OUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            save_parser_checkpoint(OUT_CHECKPOINT, runtime, step=epoch,
                                   metrics={"holdout_macro_f1": metrics["macro_f1"],
                                            "holdout_accuracy": metrics["accuracy"],
                                            "holdout_l3_macro_recall": metrics["l3_macro_recall"]},
                                   optimizer=None)
            stale = 0
        else:
            stale += 1
            if stale >= PROTOCOL["early_stopping_patience"]:
                print(f"[7b.train] early stop at epoch {epoch}", flush=True)
                break
    if OUT_CHECKPOINT.is_file():
        load_parser_checkpoint(OUT_CHECKPOINT, runtime)

    payload = {
        "_doc": ("Task 7B sections 12-13. L3 compositional-semantic hardening of the same Qwen3-VL-2B "
                 "text-only 20-class ProgramHead, initialized from the Task 6T hardened checkpoint with "
                 "the frozen Task 6M trainable policy (LoRA + ProgramHead, backbone frozen) and the "
                 "section-13 optimization protocol. Checkpoint selection uses only the internal holdout "
                 "(macro F1, then exact accuracy, then L3-four-class macro recall)."),
        "task": "7B", "stage": "E-training",
        "baseline": {"path": str(BASELINE), "sha256": baseline_sha, "expected_sha256": BASELINE_SHA256,
                     "matches": baseline_sha == BASELINE_SHA256},
        "architecture": {"model_family": "Qwen3-VL-2B", "text_only": True, "image_input": False,
                         "classes": len(EXPECTED_PROGRAM_IDS), "program_head": "Task 6M ProgramHead",
                         "trainable_policy": "Task 6M policy: LoRA adapters + ProgramHead (backbone "
                                             "frozen)",
                         "trainable_parameters": trainable, "total_parameters": total,
                         "lora": lora_report},
        "protocol": {**PROTOCOL, "microbatch": args.microbatch, "accumulation": args.accumulation,
                     "actual_effective_batch": args.microbatch * args.accumulation,
                     "grad_clip_norm": grad_clip,
                     "selection": ["internal-holdout macro F1", "exact accuracy",
                                   "L3-four-class macro recall"]},
        "data": {"rows": len(rows), "train_rows": len(train_rows), "holdout_rows": len(holdout_rows),
                 "normalized_overlap": len(overlap),
                 "augmentations": spec["augmentation_total"],
                 "cleaned_v02_train_rows": spec["cleaned_train_rows"],
                 "dropped_v02_train_records": spec["dropped_train_records"],
                 "drop_reason": ("every BuildSpatialReason v0.2 train instruction template also appears "
                                 "in the val split (120 distinct normalized texts each, 100 % overlap), "
                                 "so the section-8 rule removes all original train records"),
                 "class_counts": {program: sum(1 for row in rows if row["program"] == program)
                                  for program in EXPECTED_PROGRAM_IDS},
                 "val_or_test_used_for_selection": False},
        "history": history,
        "selected_epoch": best["epoch"],
        "selection_metrics": {"holdout_macro_f1": best["key"][0], "holdout_accuracy": best["key"][1],
                              "holdout_l3_macro_recall": best["key"][2]},
        "holdout": best["holdout"],
        "checkpoint": {"path": str(OUT_CHECKPOINT), "exists": OUT_CHECKPOINT.is_file(),
                       "sha256": sha256_file(OUT_CHECKPOINT) if OUT_CHECKPOINT.is_file() else None,
                       "bytes": OUT_CHECKPOINT.stat().st_size if OUT_CHECKPOINT.is_file() else None,
                       "committed": False},
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if args.device != "cpu" else None,
        "training_performed": True, "test_split_used": False,
        "hyperparameter_sweep": False,
    }
    payload["verdict"] = "PARSER_TRAINED" if payload["checkpoint"]["exists"] else "PARSER_TRAINING_FAILED"
    write_json(OUT, payload)
    print(f"[7b.train] selected epoch {best['epoch']} holdout macroF1 {best['key'][0]:.4f} acc "
          f"{best['key'][1]:.4f} L3 recall {best['key'][2]:.4f} -> {payload['verdict']}", flush=True)
    return 0 if payload["verdict"] == "PARSER_TRAINED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
