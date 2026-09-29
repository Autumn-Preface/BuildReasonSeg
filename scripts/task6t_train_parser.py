"""Task 6T section 7 — ProgramHead semantic hardening on the no-leakage augmented training set.

Initialises from the **authoritative Task 6S checkpoint** (`artifacts/checkpoints/task6m/
program_parser_v02_best.pt`, SHA256 `eb50b021…d028a3`) into the same Qwen3-VL-2B text-only 20-class
ProgramHead, and continues training with exactly the Task 6M trainable-parameter policy (LoRA adapters +
the ProgramHead; the backbone stays frozen).

Training data: `artifacts/task6t/parser_train_augmented/parser_train_combined.jsonl`
(BuildSpatialReason v0.2 **train** split text in both languages + the deterministic augmented
paraphrases, after the collision filter).

Model selection uses **only** the internal train-only holdout (deterministic 90/10 split by normalized
prompt hash, stratified by canonical program, seed 20260930): macro F1 first, exact accuracy as
tie-break. MiniVal240, the fixed-24 pack, the minimal pairs, the stress pack and the test split are
never used for optimisation or checkpoint selection.

    python scripts/task6t_train_parser.py [--configs C1,C2,C3] [--max-epochs 2]
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
from scripts.task6t_build_parser_data import normalize_prompt  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CONFIG_PATH = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
BASELINE = REPO_ROOT / "artifacts" / "checkpoints" / "task6m" / "program_parser_v02_best.pt"
BASELINE_SHA256 = "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3"
TRAIN_DATA = REPO_ROOT / "artifacts" / "task6t" / "parser_train_augmented" / "parser_train_combined.jsonl"
AUGMENT_SPEC = EVAL / "task6t_parser_train_augmentation_spec.json"
OUT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
OUT = EVAL / "task6t_training_summary.json"
SEED = 20260930
HOLDOUT_FRACTION = 0.10

#: Declared before training; selected only on the train-internal holdout.
CONFIGS = {
    "C1": {"name": "task6m_recipe", "lora_lr": 1.0e-4, "head_lr": 3.0e-4, "max_epochs": 2},
    "C2": {"name": "half_lr", "lora_lr": 5.0e-5, "head_lr": 1.5e-4, "max_epochs": 2},
    "C3": {"name": "quarter_lr", "lora_lr": 2.5e-5, "head_lr": 7.5e-5, "max_epochs": 2},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(confusion: np.ndarray) -> float:
    recalls = []
    for index in range(confusion.shape[0]):
        support = confusion[index].sum()
        if support:
            recalls.append(confusion[index, index] / support)
    return float(np.mean(recalls)) if recalls else 0.0


def load_rows() -> list[dict]:
    rows = []
    with TRAIN_DATA.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_holdout(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic 90/10 split by normalized prompt hash, stratified by canonical program."""

    by_program: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_program[row["program"]].append(row)
    train, holdout = [], []
    for program in EXPECTED_PROGRAM_IDS:
        group = by_program.get(program, [])
        ordered = sorted(group, key=lambda row: hashlib.sha256(
            f"{SEED}:{normalize_prompt(row['prompt'])}".encode("utf-8")).hexdigest())
        cut = max(1, int(round(len(ordered) * HOLDOUT_FRACTION))) if ordered else 0
        holdout.extend(ordered[:cut])
        train.extend(ordered[cut:])
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
            predictions = logits.argmax(dim=1).cpu().numpy()
            for row, prediction in zip(chunk, predictions):
                confusion[index_of[row["program"]], int(prediction)] += 1
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    return {
        "count": total, "correct": correct,
        "accuracy": correct / total if total else None,
        "macro_f1": macro_f1(confusion),
        "confusion_matrix": confusion.tolist(),
    }


def train_config(runtime, rows: list[dict], config: dict, optimizer_cfg: dict, device: str,
                 batch_size: int, grad_clip: float, max_epochs: int, patience: int = 1,
                 candidate_path: Path | None = None) -> dict:
    import torch

    train_rows, holdout_rows = stratified_holdout(rows)
    groups = runtime.trainable_parameter_groups(config["lora_lr"], config["head_lr"],
                                                float(optimizer_cfg.get("weight_decay", 0.01)))
    optimizer = torch.optim.AdamW(groups, betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))))
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    history = []
    best = {"key": (-1.0, -1.0), "epoch": 0, "holdout": None, "path": None}
    stale = 0
    for epoch in range(1, max_epochs + 1):
        runtime.qwen.train()
        generator = np.random.default_rng(SEED + epoch)
        order = generator.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), batch_size):
            chunk = [train_rows[int(index)] for index in order[start: start + batch_size]]
            batch = runtime.build_batch([row["prompt"] for row in chunk],
                                        [row["program"] for row in chunk]).to(runtime.device)
            losses.append(runtime.train_step(batch, optimizer, grad_clip)["loss"])
        metrics = evaluate(runtime, holdout_rows)
        key = (metrics["macro_f1"], metrics["accuracy"] or 0.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "holdout_macro_f1": metrics["macro_f1"],
                        "holdout_accuracy": metrics["accuracy"], "steps": len(losses)})
        print(f"[6t.train] {config['name']} epoch {epoch}: loss {np.mean(losses):.4f} holdout "
              f"macroF1 {metrics['macro_f1']:.4f} acc {metrics['accuracy']:.4f}", flush=True)
        if key > best["key"]:
            stale = 0
            if candidate_path is not None:
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                save_parser_checkpoint(candidate_path, runtime, step=epoch,
                                       metrics={"holdout_macro_f1": metrics["macro_f1"],
                                                "holdout_accuracy": metrics["accuracy"]},
                                       optimizer=None)
            best = {"key": key, "epoch": epoch, "holdout": metrics, "path": candidate_path}
        else:
            stale += 1
            if stale >= patience:
                print(f"[6t.train] {config['name']}: early stop at epoch {epoch}", flush=True)
                break
    if best["path"] is not None and Path(best["path"]).is_file():
        load_parser_checkpoint(best["path"], runtime)
    return {
        "config": config,
        "holdout_records": len(holdout_rows),
        "train_records": len(train_rows),
        "history": history,
        "selected_epoch": best["epoch"],
        "selection_metric": {"holdout_macro_f1": best["key"][0], "holdout_accuracy": best["key"][1]},
        "holdout": best["holdout"],
        "candidate_checkpoint": str(best["path"]) if best["path"] is not None else None,
        "wall_seconds": round(time.time() - started, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if device != "cpu" else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="C1,C2,C3")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    started = time.time()
    baseline_sha = sha256_file(BASELINE) if BASELINE.is_file() else None
    if baseline_sha != BASELINE_SHA256:
        write_json(OUT, {"_doc": "Task 6T section 7.", "task": "6T",
                         "verdict": "BASELINE_PARSER_UNAVAILABLE",
                         "baseline": {"path": str(BASELINE), "sha256": baseline_sha,
                                      "expected": BASELINE_SHA256}})
        print("[6t.train] STOP BASELINE_PARSER_UNAVAILABLE", flush=True)
        return 2

    spec = json.loads(AUGMENT_SPEC.read_text(encoding="utf-8"))
    rows = load_rows()
    cfg = load_config(CONFIG_PATH)
    optimizer_cfg = dict(cfg.get("optimizer", {}))
    batch_size = int(cfg["j2"].get("batch_size", 16))
    grad_clip = float(optimizer_cfg.get("grad_clip_norm", 1.0))
    selected = [name.strip() for name in args.configs.split(",") if name.strip()]
    print(f"[6t.train] training rows {len(rows)}; configs {selected}", flush=True)

    runs = {}
    sweep_terminated_early = None
    for name in selected:
        config = dict(CONFIGS[name])
        if args.max_epochs is not None:
            config["max_epochs"] = args.max_epochs
        runtime = build_program_parser(cfg, device=args.device, verbose=False)
        load_parser_checkpoint(BASELINE, runtime)
        trainable = sum(parameter.numel() for parameter in runtime.parameters()
                        if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in runtime.parameters())
        candidate_path = OUT_CHECKPOINT.parent / f"_candidate_{name}.pt"
        result = train_config(runtime, rows, config, optimizer_cfg, args.device, batch_size,
                              grad_clip, int(config["max_epochs"]), candidate_path=candidate_path)
        result["trainable_parameters"] = int(trainable)
        result["total_parameters"] = int(total)
        runs[name] = result
        if (result["selection_metric"]["holdout_macro_f1"] >= 1.0
                and result["selection_metric"]["holdout_accuracy"] >= 1.0
                and name != selected[-1]):
            sweep_terminated_early = {
                "config": name,
                "reason": ("the declared configuration reached the selection ceiling on the internal "
                           "holdout (macro F1 1.0 and accuracy 1.0); the remaining declared "
                           "configurations were not run"),
                "declared_but_not_run": selected[selected.index(name) + 1:],
            }
            print(f"[6t.train] sweep terminated early after {name} (holdout ceiling reached)",
                  flush=True)
            break

    best_name = max(runs, key=lambda key: (runs[key]["selection_metric"]["holdout_macro_f1"],
                                           runs[key]["selection_metric"]["holdout_accuracy"]))
    best_candidate = Path(runs[best_name]["candidate_checkpoint"])
    if best_candidate.is_file():
        OUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        OUT_CHECKPOINT.write_bytes(best_candidate.read_bytes())
    checkpoint_sha = sha256_file(OUT_CHECKPOINT) if OUT_CHECKPOINT.is_file() else None
    payload = {
        "_doc": (
            "Task 6T section 7. ProgramHead semantic hardening: same Qwen3-VL-2B text-only 20-class "
            "ProgramHead and the same Task 6M trainable-parameter policy, initialised from the "
            "authoritative Task 6S checkpoint and continued on the no-leakage augmented training set. "
            "Checkpoint selection uses only the train-internal stratified holdout (macro F1, then "
            "accuracy); MiniVal240, fixed24, minimal pairs, stress and the test split are never used "
            "for selection."
        ),
        "task": "6T", "stage": "training",
        "baseline": {"path": str(BASELINE), "sha256": baseline_sha,
                     "expected_sha256": BASELINE_SHA256, "verified": True, "retrained_from_scratch": False},
        "architecture": {
            "model": cfg["models"]["qwen_model_id"], "text_only": True, "image_tokens": False,
            "program_ids": list(EXPECTED_PROGRAM_IDS),
            "trainable_policy": "Task 6M policy: LoRA adapters + ProgramHead (backbone frozen)",
        },
        "data": {
            "path": str(TRAIN_DATA), "rows": len(rows),
            "sources": dict(Counter(row["source"] for row in rows)),
            "languages": dict(Counter(row["language"] for row in rows)),
            "augmentation_spec": "evaluation/task6t_parser_train_augmentation_spec.json",
            "augmented_rows": spec["counts"]["augmented_kept"],
            "v02_train_rows_kept": spec["counts"]["v02_train_examples_kept"],
            "collision_filter_dropped": spec["deterministic_collision_filter"],
        },
        "holdout": {
            "fraction": HOLDOUT_FRACTION, "seed": SEED,
            "stratified_by": "canonical program id",
            "split_key": "sha256(seed:normalized_prompt)",
            "selection_metric": ["internal-holdout macro F1", "tie-break exact accuracy"],
            "only_set_used_for_selection": True,
        },
        "optimizer": {
            "name": "AdamW", "betas": optimizer_cfg.get("betas", [0.9, 0.999]),
            "weight_decay": optimizer_cfg.get("weight_decay"),
            "grad_clip_norm": grad_clip, "batch_size": batch_size,
            "amp": "bfloat16 autocast", "seed": SEED,
            "scheduler": "none (constant group LRs)",
            "early_stopping_patience": 1,
        },
        "configs": {name: {"declared": True, **CONFIGS[name], **runs[name]} for name in runs},
        "selected_config": best_name,
        "selected": runs[best_name]["selection_metric"],
        "checkpoint": {
            "path": str(OUT_CHECKPOINT), "sha256": checkpoint_sha,
            "exists": OUT_CHECKPOINT.is_file(),
            "bytes": OUT_CHECKPOINT.stat().st_size if OUT_CHECKPOINT.is_file() else None,
            "selected_config": best_name,
            "selected_epoch": runs[best_name]["selected_epoch"],
            "trainable_parameters": runs[best_name]["trainable_parameters"],
            "total_parameters": runs[best_name]["total_parameters"],
            "committed": False,
        },
        "training_policy": {
            "declared_configurations": len(CONFIGS),
            "configurations_run": len(runs),
            "sweep_terminated_early": sweep_terminated_early,
            "sweep_larger_than_three": False,
            "val_test_tuning": False,
        },
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[6t.train] selected {best_name} (holdout macroF1 "
          f"{runs[best_name]['selection_metric']['holdout_macro_f1']:.4f}); checkpoint "
          f"{str(checkpoint_sha)[:16]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
