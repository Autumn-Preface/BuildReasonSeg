"""Task 7C Part J — final frozen-parser evaluation on the six reused evaluation sets.

The evaluation packs are **reused byte-for-byte** from Task 7A/7B (no regeneration). After the checkpoint SHA
is frozen each set is evaluated exactly once:

* full BuildSpatialReason v0.2 val — accuracy >= 0.995, macro F1 >= 0.995, every class recall >= 0.98;
* Z-MiniVal240 — 240/240;
* Z-PairedVal20 member queries — 40/40;
* Task 7A fixed24 — >= 22/24, all 8 compact prompts correct, every L3 class >= 5/6;
* Task 7B minimal96 — 96/96;
* Task 7B stress192 — accuracy >= 0.97, macro F1 >= 0.97, every represented class recall >= 0.90,
  L3-four-class macro recall >= 0.95.

    python scripts/task7c_eval_parser.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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
)
from buildreasonseg_mvp.runtime import load_config  # noqa: E402
from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES  # noqa: E402
from buildreasonseg_mvp.task6s_directional_pipeline import parse_instruction  # noqa: E402
from scripts.task7b_build_parser_data import sha256_file  # noqa: E402
from scripts.task7b_eval_parser import confusion_for, metrics_from  # noqa: E402
from scripts.task7c_build_rehearsal import DATA, L3_PROGRAMS  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
CONFIG_PATH = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7c" / "program_parser_l3_rehearsal_v1.pt"
TRAINING = EVAL / "task7c_training_summary.json"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
MINIMAL = EVAL / "task7b_compositional_minimal_pairs.json"
STRESS = EVAL / "task7b_l3_stress_v1.json"
FIXED24 = EVAL / "task7a_l3_paraphrase_pack.json"
OUT_FULL = EVAL / "task7c_full_val.json"
OUT_ZMINI = EVAL / "task7c_z_minival240.json"
OUT_ZPAIRED = EVAL / "task7c_z_paired_parser.json"
OUT_FIXED24 = EVAL / "task7c_task7a_fixed24.json"
OUT_MINIMAL = EVAL / "task7c_minimal_pairs_result.json"
OUT_STRESS = EVAL / "task7c_stress_result.json"
GATES = {"full_val_accuracy": 0.995, "full_val_macro_f1": 0.995, "class_recall": 0.98,
         "z_minival": 240, "z_paired": 40, "fixed24_total": 22, "fixed24_compact": 8,
         "fixed24_class": 5, "minimal_pairs": 96, "stress_accuracy": 0.97, "stress_macro_f1": 0.97,
         "stress_class_recall": 0.90, "stress_l3_recall": 0.95}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    if not CHECKPOINT.is_file():
        write_json(OUT_FULL, {"_doc": "Task 7C section 16.", "task": "7C",
                              "verdict": "PARSER_TRAINING_FAILED"})
        print("[7c.eval] STOP PARSER_TRAINING_FAILED", flush=True)
        return 2
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(CHECKPOINT)
    cfg = load_config(CONFIG_PATH)
    runtime = build_program_parser(cfg, device=args.device, verbose=False)
    load_parser_checkpoint(CHECKPOINT, runtime)
    vocabulary = tuple(EXPECTED_QUERY_TYPES)
    parse = lambda query: parse_instruction(runtime, query, vocabulary)  # noqa: E731
    frozen = {"path": str(CHECKPOINT), "sha256": checkpoint_sha,
              "selected_epoch": training["selected_epoch"],
              "initialized_from": training["baseline"]["sha256"],
              "architecture": training["architecture"]["model_family"], "text_only": True,
              "classes": len(EXPECTED_PROGRAM_IDS)}

    # ---------------- full v0.2 val
    val_rows = []
    with (DATA / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for key, language in (("instruction_en", "en"), ("instruction_zh", "zh")):
                if record.get(key):
                    val_rows.append({"prompt": str(record[key]),
                                     "expected_program": str(record["query_type"]),
                                     "language": language, "sample_id": str(record["sample_id"])})
    full = metrics_from(confusion_for(val_rows, parse))
    full_gate = {
        "accuracy": {"required": GATES["full_val_accuracy"], "measured": full["accuracy"],
                     "passed": (full["accuracy"] or 0.0) >= GATES["full_val_accuracy"]},
        "macro_f1": {"required": GATES["full_val_macro_f1"], "measured": full["macro_f1"],
                     "passed": full["macro_f1"] >= GATES["full_val_macro_f1"]},
        "every_class_recall": {"required": GATES["class_recall"],
                               "measured": full["min_represented_class_recall"],
                               "passed": (full["min_represented_class_recall"] or 0.0)
                               >= GATES["class_recall"]},
    }
    write_json(OUT_FULL, {
        "_doc": ("Task 7C section 16. Full BuildSpatialReason v0.2 val evaluation of the frozen Task 7C "
                 "checkpoint with the section-16 gate."),
        "task": "7C", "stage": "J-full-val", "checkpoint": frozen,
        "records": len(val_rows), "languages": dict(Counter(row["language"] for row in val_rows)),
        "metrics": full, "gate_constants": GATES, "gate": full_gate,
        "all_gates_passed": all(entry["passed"] for entry in full_gate.values()),
        "test_split_used": False, "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7c.eval] full val {full['correct']}/{full['count']} acc {full['accuracy']:.4f} macroF1 "
          f"{full['macro_f1']:.4f} minRecall {full['min_represented_class_recall']:.4f}", flush=True)

    # ---------------- Z-MiniVal240 and Z-PairedVal members (exact reuse)
    instructions = {}
    with (DATA / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            instructions[str(record["sample_id"])] = record
    mini = json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]
    mini_rows = [{"prompt": str(instructions[record["sample_id"]]["instruction_en"]),
                  "expected_program": record["program_id"], "sample_id": record["sample_id"]}
                 for record in mini]
    mini_metrics = metrics_from(confusion_for(mini_rows, parse))
    write_json(OUT_ZMINI, {
        "_doc": "Task 7C section 17. Z-MiniVal240 natural-language queries, reused byte-for-byte.",
        "task": "7C", "stage": "J-z-minival240", "checkpoint": frozen, "pack": "z_mini_val_240",
        "metrics": mini_metrics, "required_correct": 240,
        "passed": mini_metrics["correct"] == 240 and mini_metrics["count"] == 240,
        "test_split_used": False,
    })
    paired_records = json.loads((PACK_ROOT / "z_paired_val20.json")
                                .read_text(encoding="utf-8"))["records"]
    paired_rows = [{"prompt": str(instructions[record["sample_id"]]["instruction_en"]),
                    "expected_program": record["program_id"], "sample_id": record["sample_id"]}
                   for record in paired_records]
    paired_metrics = metrics_from(confusion_for(paired_rows, parse))
    write_json(OUT_ZPAIRED, {
        "_doc": "Task 7C section 18. Z-PairedVal20 member queries, reused byte-for-byte.",
        "task": "7C", "stage": "J-z-paired-members", "checkpoint": frozen, "pack": "z_paired_val20",
        "metrics": paired_metrics, "required_correct": 40,
        "passed": paired_metrics["correct"] == 40 and paired_metrics["count"] == 40,
        "test_split_used": False,
    })
    print(f"[7c.eval] Z-MiniVal240 {mini_metrics['correct']}/{mini_metrics['count']} | Z-Paired members "
          f"{paired_metrics['correct']}/{paired_metrics['count']}", flush=True)

    # ---------------- Task 7A fixed24 (exact reuse)
    fixed24 = json.loads(FIXED24.read_text(encoding="utf-8"))
    fixed_results = []
    for entry in fixed24["prompts"]:
        prediction = str(parse(str(entry["text"]))["program"])
        fixed_results.append({"id": entry["id"], "language": entry["language"],
                              "prompt": entry["text"], "expected_program": entry["program"],
                              "required_compact": bool(entry.get("required_compact")),
                              "parsed_program": prediction,
                              "correct": prediction == str(entry["program"])})
    fixed_correct = sum(1 for row in fixed_results if row["correct"])
    per_program = {program: {"correct": sum(1 for row in fixed_results
                                            if row["expected_program"] == program and row["correct"]),
                             "total": sum(1 for row in fixed_results
                                          if row["expected_program"] == program)}
                   for program in L3_PROGRAMS}
    compact = [row for row in fixed_results if row["required_compact"]]
    fixed_gate = {
        "total": {"required": f">= {GATES['fixed24_total']}/24", "measured": fixed_correct,
                  "passed": fixed_correct >= GATES["fixed24_total"]},
        "all_compact_correct": {"required": "8/8",
                                "measured": sum(1 for row in compact if row["correct"]),
                                "passed": all(row["correct"] for row in compact)},
        "every_l3_class": {"required": f">= {GATES['fixed24_class']}/6", "measured": per_program,
                           "passed": all(entry["correct"] >= GATES["fixed24_class"]
                                         for entry in per_program.values())},
    }
    write_json(OUT_FIXED24, {
        "_doc": ("Task 7C section 19. Exact Task 7A fixed 24 L3 paraphrases, reused byte-for-byte "
                 "(the pack was frozen before Task 7A and is unchanged)."),
        "task": "7C", "stage": "J-task7a-fixed24", "checkpoint": frozen,
        "correct": fixed_correct, "total": len(fixed_results),
        "accuracy": fixed_correct / len(fixed_results), "per_program": per_program,
        "compact": {"correct": sum(1 for row in compact if row["correct"]), "total": len(compact)},
        "gate_constants": GATES, "gate": fixed_gate,
        "all_gates_passed": all(entry["passed"] for entry in fixed_gate.values()),
        "failures": [{key: row[key] for key in ("id", "language", "expected_program",
                                                "parsed_program", "prompt")}
                     for row in fixed_results if not row["correct"]],
        "rows": fixed_results, "test_split_used": False,
    })
    print(f"[7c.eval] fixed24 {fixed_correct}/24 compact {sum(1 for row in compact if row['correct'])}/8",
          flush=True)

    # ---------------- Task 7B minimal96 and stress192 (exact reuse)
    minimal_rows = json.loads(MINIMAL.read_text(encoding="utf-8"))["rows"]
    minimal_results = []
    for row in minimal_rows:
        prediction = str(parse(str(row["prompt"]))["program"])
        minimal_results.append({"id": row["id"], "contrast_group": row["contrast_group"],
                                "language": row["language"], "prompt": row["prompt"],
                                "expected_program": row["expected_program"],
                                "parsed_program": prediction,
                                "correct": prediction == row["expected_program"]})
    minimal_correct = sum(1 for row in minimal_results if row["correct"])
    by_group = {group: {"correct": sum(1 for row in minimal_results
                                       if row["contrast_group"] == group and row["correct"]),
                        "total": sum(1 for row in minimal_results
                                     if row["contrast_group"] == group)}
                for group in sorted({row["contrast_group"] for row in minimal_results})}
    write_json(OUT_MINIMAL, {
        "_doc": ("Task 7C section 20. Task 7B compositional minimal-pair pack (96 prompts) reused "
                 "byte-for-byte."),
        "task": "7C", "stage": "J-minimal96", "checkpoint": frozen,
        "correct": minimal_correct, "total": len(minimal_results),
        "accuracy": minimal_correct / len(minimal_results), "by_contrast_group": by_group,
        "passed": minimal_correct == len(minimal_results),
        "failures": [{key: row[key] for key in ("id", "contrast_group", "language",
                                                "expected_program", "parsed_program", "prompt")}
                     for row in minimal_results if not row["correct"]],
        "rows": minimal_results, "test_split_used": False,
    })
    stress_rows = json.loads(STRESS.read_text(encoding="utf-8"))["rows"]
    stress_inputs = [{"prompt": str(row["prompt"]), "expected_program": str(row["expected_program"])}
                     for row in stress_rows]
    stress = metrics_from(confusion_for(stress_inputs, parse))
    stress_predictions = [{"language": str(row["language"]),
                           "correct": str(parse(str(row["prompt"]))["program"])
                           == str(row["expected_program"])}
                          for row in stress_rows]
    per_language = {language: {"correct": sum(1 for row in stress_predictions
                                              if row["language"] == language and row["correct"]),
                               "total": sum(1 for row in stress_predictions
                                            if row["language"] == language)}
                    for language in ("zh", "en")}
    l3_recalls = [stress["per_class_recall"][program] for program in L3_PROGRAMS
                  if stress["per_class_recall"][program] is not None]
    l3_macro = float(np.mean(l3_recalls)) if l3_recalls else 0.0
    stress_gate = {
        "accuracy": {"required": GATES["stress_accuracy"], "measured": stress["accuracy"],
                     "passed": (stress["accuracy"] or 0.0) >= GATES["stress_accuracy"]},
        "macro_f1": {"required": GATES["stress_macro_f1"], "measured": stress["macro_f1"],
                     "passed": stress["macro_f1"] >= GATES["stress_macro_f1"]},
        "every_class_recall": {"required": GATES["stress_class_recall"],
                               "measured": stress["min_represented_class_recall"],
                               "passed": (stress["min_represented_class_recall"] or 0.0)
                               >= GATES["stress_class_recall"]},
        "l3_macro_recall": {"required": GATES["stress_l3_recall"], "measured": l3_macro,
                            "passed": l3_macro >= GATES["stress_l3_recall"]},
    }
    write_json(OUT_STRESS, {
        "_doc": ("Task 7C section 21. Task 7B held-out L3 stress v1 pack (192 prompts) reused "
                 "byte-for-byte, with the section-21 gate."),
        "task": "7C", "stage": "J-stress192", "checkpoint": frozen,
        "metrics": stress, "l3_macro_recall": l3_macro, "l3_classes": list(L3_PROGRAMS),
        "per_language": per_language,
        "gate_constants": GATES, "gate": stress_gate,
        "all_gates_passed": all(entry["passed"] for entry in stress_gate.values()),
        "test_split_used": False, "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7c.eval] minimal96 {minimal_correct}/96 | stress {stress['correct']}/{stress['count']} acc "
          f"{stress['accuracy']:.4f} macroF1 {stress['macro_f1']:.4f} minRecall "
          f"{stress['min_represented_class_recall']:.4f} L3 {l3_macro:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
