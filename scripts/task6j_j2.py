"""Task 6J sections 10-13: Stage J2 -- ProgramHead training and evaluation.

The parser branch maps an instruction TEXT (chat formatting, no image tokens) to one of the 20
canonical program ids via a text-only Qwen LoRA + ProgramHead (LayerNorm -> Linear(2048, 20))
over the last prompt position's hidden state. Training uses a deterministic query-type-stratified
train-only subset (100 records per program, 2000 total); query_type appears ONLY as the CE
target, never as input. Per-epoch validation runs on the fixed 120 validation records and the
fixed 20 paired validation images; the best epoch is then re-scored on the full val split.

J2 gate: program accuracy >= 0.90, macro F1 >= 0.85 (fixed 120), paired program correctness
>= 18/20. Failure is `PROGRAM_PARSER_NOT_READY`.

Writes `evaluation/task6j_parser_setup.json`, `evaluation/task6j_j2_program_parser.json` and the
checkpoint manifest.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp import data as data_mod  # noqa: E402
from buildreasonseg_mvp.program_parser import (  # noqa: E402
    NUM_PROGRAMS,
    EXPECTED_PROGRAM_IDS,
    PROGRAM_ID_TO_INDEX,
    build_program_parser,
    load_parser_checkpoint,
    save_parser_checkpoint,
)
from buildreasonseg_mvp.runtime import enable_determinism, load_config, set_seed  # noqa: E402

from task6j_common import EVAL, fixed_validation_material, paired_sample_lists, write_json  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
OUT_SETUP = EVAL / "task6j_parser_setup.json"
OUT_J2 = EVAL / "task6j_j2_program_parser.json"
MANIFEST = EVAL / "task6j_checkpoint_manifest.json"


def stratified_subset(train_records: list[dict], per_program: int) -> list[dict]:
    """Section 12: query-type-stratified, deterministic (sorted by sample id), train-only."""

    buckets: dict[str, list[dict]] = {}
    for record in train_records:
        buckets.setdefault(str(record["query_type"]), []).append(record)
    chosen = []
    for query_type in sorted(buckets):
        bucket = sorted(buckets[query_type], key=lambda r: r["sample_id"])
        chosen.extend(bucket[:per_program])
    return sorted(chosen, key=lambda r: r["sample_id"])


@torch.no_grad()
def evaluate(runtime, records: list[dict], tag: str) -> dict:
    was_training = bool(runtime.qwen.training)
    runtime.qwen.eval()
    try:
        predictions = []
        for start in range(0, len(records), 32):
            chunk = records[start:start + 32]
            batch = runtime.build_batch(
                [r["instruction_zh"] for r in chunk],
                [r["query_type"] for r in chunk],
            ).to(runtime.device)
            logits, _hidden = runtime.forward(batch)
            predictions.extend(torch.argmax(logits, dim=1).tolist())
    finally:
        if was_training:
            runtime.qwen.train()

    rows = []
    for record, index in zip(records, predictions):
        target = PROGRAM_ID_TO_INDEX[str(record["query_type"])]
        rows.append(
            {
                "sample_id": str(record["sample_id"]),
                "image_id": str(record["image_id"]),
                "level": int(record["level"]),
                "query_type": str(record["query_type"]),
                "predicted_index": int(index),
                "correct": bool(index == target),
            }
        )
    correct = sum(1 for row in rows if row["correct"])
    confusion = [[0] * NUM_PROGRAMS for _ in range(NUM_PROGRAMS)]
    for row in rows:
        confusion[PROGRAM_ID_TO_INDEX[row["query_type"]]][row["predicted_index"]] += 1
    per_class = Counter(row["query_type"] for row in rows if row["correct"])
    per_class_total = Counter(row["query_type"] for row in rows)
    f1_per_class = {}
    for program_id in EXPECTED_PROGRAM_IDS:
        tp = confusion[PROGRAM_ID_TO_INDEX[program_id]][PROGRAM_ID_TO_INDEX[program_id]]
        fp = sum(confusion[i][PROGRAM_ID_TO_INDEX[program_id]] for i in range(NUM_PROGRAMS)) - tp
        fn = sum(confusion[PROGRAM_ID_TO_INDEX[program_id]][j] for j in range(NUM_PROGRAMS)) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_per_class[program_id] = 2 * precision * recall / max(precision + recall, 1e-12)
    levels = {}
    for level in sorted({row["level"] for row in rows}):
        level_rows = [row for row in rows if row["level"] == level]
        levels[str(level)] = {
            "count": len(level_rows),
            "accuracy": sum(1 for row in level_rows if row["correct"]) / max(len(level_rows), 1),
        }
    return {
        "tag": tag,
        "count": len(rows),
        "exact_correct": correct,
        "exact_accuracy": correct / max(len(rows), 1),
        "macro_f1": sum(f1_per_class.values()) / max(len(f1_per_class), 1),
        "confusion_matrix": confusion,
        "per_query_type_accuracy": {
            query_type: per_class[query_type] / max(per_class_total.get(query_type, 0), 1)
            for query_type in EXPECTED_PROGRAM_IDS
        },
        "f1_per_class": f1_per_class,
        "by_level": levels,
        "rows": rows,
    }


def main() -> int:
    started = time.time()
    cfg = load_config(CONFIG)
    j2_cfg = cfg["j2"]
    determinism = enable_determinism(int(cfg["seed"]), strict=True)

    train_records = data_mod.read_records("train")
    subset = stratified_subset(train_records, int(j2_cfg["records_per_program"]))
    assert len(subset) <= int(j2_cfg["max_records"]), len(subset)
    val_samples, pairs = fixed_validation_material()
    a_samples, b_samples = paired_sample_lists(pairs)
    val_records = [sample.raw for sample in val_samples]
    paired_records = [sample.raw for sample in a_samples + b_samples]

    print(
        f"[task6j.j2] subset {len(subset)} records ({len(set(r['query_type'] for r in subset))} "
        f"programs), epochs {j2_cfg['epochs']}, batch {j2_cfg['batch_size']}",
        flush=True,
    )

    runtime = build_program_parser(cfg, verbose=True)
    groups = runtime.trainable_parameter_groups(
        lora_lr=float(cfg["optimizer"]["lora_lr"]),
        head_lr=float(cfg["optimizer"]["head_lr"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
    )
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg["optimizer"]["betas"]))
    steps_per_epoch = max(1, (len(subset) + int(j2_cfg["batch_size"]) - 1) // int(j2_cfg["batch_size"]))
    total_steps = steps_per_epoch * int(j2_cfg["epochs"])
    warmup = int(cfg["optimizer"]["warmup_steps"])

    def factor(step):
        if step < warmup:
            return max(1e-3, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    setup = {
        "_doc": (
            "Task 6J section 11. ProgramHead setup: text-only Qwen LoRA + LayerNorm->Linear "
            "classifier over the last prompt position (assistant-prefix representation). "
            "Instruction text only; no image tokens; query_type appears only as the CE target."
        ),
        "task": "6J",
        "determinism": determinism,
        "runtime": {
            **runtime.reports,
            "token_trainable_ids": runtime.reports["lora"].get("trainable_token_ids"),
        },
        "optimizer_groups": [
            {
                "name": group["name"],
                "lr": group["lr"],
                "tensors": len(group["params"]),
                "parameters": sum(p.numel() for p in group["params"]),
            }
            for group in groups
        ],
        "training_subset": {
            "records": len(subset),
            "per_program": int(j2_cfg["records_per_program"]),
            "programs_covered": len({r["query_type"] for r in subset}),
            "split": "train only",
        },
        "budget": {
            "epochs": int(j2_cfg["epochs"]),
            "batch_size": int(j2_cfg["batch_size"]),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup,
        },
    }
    write_json(OUT_SETUP, setup)

    report = {
        "_doc": (
            "Task 6J sections 10-13. Stage J2: instruction -> canonical program classification. "
            "Per-epoch validation on the fixed 120 records and the fixed 20 paired images; the "
            "best epoch by fixed-120 accuracy is re-scored on the full val split."
        ),
        "task": "6J",
        "stage": "J2",
        "history": [],
        "epochs": [],
    }

    global_step = 0
    for epoch in range(1, int(j2_cfg["epochs"]) + 1):
        set_seed(int(cfg["seed"]) + epoch)
        epoch_losses = []
        for start in range(0, len(subset), int(j2_cfg["batch_size"])):
            chunk = subset[start:start + int(j2_cfg["batch_size"])]
            batch = runtime.build_batch(
                [r["instruction_zh"] for r in chunk], [r["query_type"] for r in chunk]
            ).to(runtime.device)
            result = runtime.train_step(batch, optimizer, float(cfg["optimizer"]["grad_clip_norm"]))
            scheduler.step()
            global_step += 1
            epoch_losses.append(result["loss"])
            if global_step % int(j2_cfg["log_every_steps"]) == 0:
                print(
                    f"[task6j.j2] epoch {epoch} step {global_step}/{total_steps} "
                    f"loss {result['loss']:.4f} lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        val_eval = evaluate(runtime, val_records, "fixed_120_val")
        paired_eval = evaluate(runtime, paired_records, "paired_40")
        n_pairs = len(paired_eval["rows"]) // 2  # rows are A-side x n then B-side x n
        paired_pairs = []
        for i in range(n_pairs):
            row_a, row_b = paired_eval["rows"][i], paired_eval["rows"][n_pairs + i]
            paired_pairs.append(
                {
                    "image_id": str(pairs[i]["image_id"]),
                    "a": row_a["sample_id"],
                    "b": row_b["sample_id"],
                    "a_correct": row_a["correct"],
                    "b_correct": row_b["correct"],
                    "pair_correct": bool(row_a["correct"] and row_b["correct"]),
                }
            )
        paired_correct = sum(1 for row in paired_pairs if row["pair_correct"])
        epoch_entry = {
            "epoch": epoch,
            "train_mean_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
            "val": val_eval,
            "paired": {
                "paired_total": len(paired_pairs),
                "paired_program_correct": paired_correct,
                "rows": paired_pairs,
            },
        }
        report["epochs"].append(epoch_entry)
        report["history"].append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_mean_loss": epoch_entry["train_mean_loss"],
                "val_accuracy": val_eval["exact_accuracy"],
                "val_macro_f1": val_eval["macro_f1"],
                "paired_correct": paired_correct,
            }
        )
        print(
            f"[task6j.j2] epoch {epoch}: val acc {val_eval['exact_accuracy']:.4f} "
            f"macroF1 {val_eval['macro_f1']:.4f} paired {paired_correct}/{len(paired_pairs)}",
            flush=True,
        )

    best = max(report["epochs"], key=lambda entry: entry["val"]["exact_accuracy"])
    checkpoint = save_parser_checkpoint(
        REPO_ROOT / cfg["paths"]["checkpoints"] / "j2_best.pt",
        runtime,
        global_step,
        metrics={
            "epoch": best["epoch"],
            "val_accuracy": best["val"]["exact_accuracy"],
            "macro_f1": best["val"]["macro_f1"],
            "paired_correct": best["paired"]["paired_program_correct"],
        },
        optimizer=optimizer,
    )
    manifest = {
        "_doc": "Task 6J: hashes of every Task 6J checkpoint. Checkpoint bytes stay gitignored.",
        "task": "6J",
        "checkpoints": {"J2/j2_best": checkpoint},
    }
    write_json(MANIFEST, manifest)

    full_val = evaluate(runtime, data_mod.read_records("val"), "full_val")
    gate_cfg = j2_cfg["gate"]
    checks = {
        "program_accuracy_ge": float(best["val"]["exact_accuracy"]) >= float(gate_cfg["program_accuracy_min"]),
        "macro_f1_ge": float(best["val"]["macro_f1"]) >= float(gate_cfg["macro_f1_min"]),
        "paired_program_correct_ge": int(best["paired"]["paired_program_correct"])
        >= int(gate_cfg["paired_program_correct_min"]),
    }
    gate = {"checks": checks, "passed": all(checks.values()), "gate": dict(gate_cfg)}
    report.update(
        {
            "selection": {
                "selected_epoch": int(best["epoch"]),
                "checkpoint": checkpoint,
                "criterion": "fixed-120 exact accuracy",
            },
            "full_val": full_val,
            "final": {
                "epoch": int(best["epoch"]),
                "val": best["val"],
                "paired": best["paired"],
                "gate": gate,
                "checkpoint": checkpoint,
            },
            "verdict": "J2_PASS" if gate["passed"] else "PROGRAM_PARSER_NOT_READY",
            "seconds": round(time.time() - started, 2),
        }
    )
    write_json(OUT_J2, report)
    print(
        f"[task6j.j2] {report['verdict']}: epoch {best['epoch']} val acc "
        f"{best['val']['exact_accuracy']:.4f} macroF1 {best['val']['macro_f1']:.4f} paired "
        f"{best['paired']['paired_program_correct']}/{best['paired']['paired_total']} | "
        f"full-val acc {full_val['exact_accuracy']:.4f} gate {gate['passed']}",
        flush=True,
    )
    print(f"[task6j.j2] wrote {OUT_SETUP.name} + {OUT_J2.name} + manifest", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if gate["passed"] else 9


if __name__ == "__main__":
    raise SystemExit(main())
