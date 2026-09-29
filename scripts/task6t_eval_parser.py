"""Task 6T sections 7.1 and 8 — frozen baseline and final ProgramHead evaluations.

* `--stage baseline` (section 7.1): evaluate the authoritative Task 6S checkpoint on MiniVal240,
  PairedVal20 members, the exact Task 6S fixed-24 paraphrases, the Task 6T minimal pairs and the Task 6T
  stress pack → `evaluation/task6t_parser_baseline.json`.
* `--stage final` (section 8): the same evaluation with the hardened Task 6T checkpoint, plus the full
  BuildSpatialReason v0.2 val audit (exact accuracy, macro F1, per-class recall, confusion matrix,
  language breakdown) → `task6t_parser_full_val.json`, `task6t_parser_minival240.json`,
  `task6t_parser_pairedval20.json`, `task6t_parser_fixed24.json`,
  `task6t_parser_minimal_pairs_result.json`, `task6t_parser_stress_result.json`.

Instruction text only; the pack queries come from the frozen BuildSpatialReason v0.2 val records. The
test split is never read.

    python scripts/task6t_eval_parser.py --stage baseline
    python scripts/task6t_eval_parser.py --stage final
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
)
from buildreasonseg_mvp.runtime import load_config  # noqa: E402
from scripts.task6s_cli_audit import OOD_PROMPTS, OUT_OF_SCOPE_PROMPTS, SUPPORTED_PROMPTS  # noqa: E402
from scripts.task6t_build_parser_data import FIXED24, normalize_prompt  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6n" / "packs"
V02 = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
CONFIG_PATH = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
BASELINE = REPO_ROOT / "artifacts" / "checkpoints" / "task6m" / "program_parser_v02_best.pt"
BASELINE_SHA256 = "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3"
HARDENED = REPO_ROOT / "artifacts" / "checkpoints" / "task6t" / "program_parser_hardened_v1.pt"
TRAINING_SUMMARY = EVAL / "task6t_training_summary.json"
MINIMAL = EVAL / "task6t_parser_minimal_pairs.json"
STRESS = EVAL / "task6t_parser_stress_v1.json"
OUT_BASELINE = EVAL / "task6t_parser_baseline.json"
OUT_FULL_VAL = EVAL / "task6t_parser_full_val.json"
OUT_MINIVAL = EVAL / "task6t_parser_minival240.json"
OUT_PAIRED = EVAL / "task6t_parser_pairedval20.json"
OUT_FIXED24 = EVAL / "task6t_parser_fixed24.json"
OUT_MINIMAL_RESULT = EVAL / "task6t_parser_minimal_pairs_result.json"
OUT_STRESS_RESULT = EVAL / "task6t_parser_stress_result.json"
BATCH = 32
FIXED24_MUST_PASS = (
    "找出最大建筑左边的建筑物。",
    "找出最大建筑右边的建筑物。",
    "找出最大建筑下面的建筑物。",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def val_records() -> dict[str, dict]:
    records = {}
    with (V02 / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["sample_id"]] = record
    return records


def mini_val_items(records: dict[str, dict]) -> list[dict]:
    pack = json.loads((PACK_ROOT / "mini_val_240.json").read_text(encoding="utf-8"))
    items = []
    for sample in pack["records"]:
        record = records[sample["sample_id"]]
        for language in ("en", "zh"):
            items.append({"prompt": str(record[f"instruction_{language}"]),
                          "expected_program": sample["program_id"], "language": language,
                          "sample_id": sample["sample_id"]})
    return items


def paired_items(records: dict[str, dict]) -> list[dict]:
    pack = json.loads((PACK_ROOT / "paired_val_20.json").read_text(encoding="utf-8"))
    items = []
    for pair in pack["pairs"]:
        for side in ("a", "b"):
            member = pair[side]
            record = records[member["sample_id"]]
            for language in ("en", "zh"):
                items.append({"prompt": str(record[f"instruction_{language}"]),
                              "expected_program": member["program_id"], "language": language,
                              "sample_id": member["sample_id"]})
    return items


def fixed24_items() -> list[dict]:
    expected = {}
    for program, prompt in SUPPORTED_PROMPTS:
        expected[prompt] = program
    return [{"prompt": prompt, "expected_program": expected[prompt],
             "language": "zh" if not prompt.isascii() else "en", "index": index}
            for index, prompt in enumerate(FIXED24, start=1)]


def pack_items(path: Path, key: str = "prompts") -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [{"prompt": row["prompt"], "expected_program": row["expected_program"],
             "language": row["language"], "contrast_group": row.get("contrast_group")}
            for row in payload[key]]


def full_val_items(records: dict[str, dict], language: str) -> list[dict]:
    return [{"prompt": str(record[f"instruction_{language}"]),
             "expected_program": str(record["query_type"]), "language": language,
             "sample_id": sample_id}
            for sample_id, record in records.items()]


def evaluate(runtime, items: list[dict], batch_size: int = BATCH) -> dict:
    import torch

    index_of = {program: index for index, program in enumerate(EXPECTED_PROGRAM_IDS)}
    confusion = np.zeros((len(EXPECTED_PROGRAM_IDS), len(EXPECTED_PROGRAM_IDS)), dtype=np.int64)
    rows = []
    runtime.qwen.eval()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start: start + batch_size]
            batch = runtime.build_batch([item["prompt"] for item in chunk],
                                        [item["expected_program"] for item in chunk]).to(runtime.device)
            logits, _hidden = runtime.forward(batch)
            probabilities = torch.softmax(logits.float(), dim=-1)
            predictions = logits.argmax(dim=1).cpu().numpy()
            confidences = probabilities.max(dim=1).values.cpu().numpy()
            for item, prediction, confidence in zip(chunk, predictions, confidences):
                predicted = EXPECTED_PROGRAM_IDS[int(prediction)]
                confusion[index_of[item["expected_program"]], int(prediction)] += 1
                rows.append({**item, "predicted_program": predicted,
                             "correct": predicted == item["expected_program"],
                             "confidence": float(confidence)})
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    recalls = {}
    for index, program in enumerate(EXPECTED_PROGRAM_IDS):
        support = int(confusion[index].sum())
        recalls[program] = (float(confusion[index, index] / support) if support else None)
    with_support = [value for value in recalls.values() if value is not None]
    per_language = {}
    for language in ("en", "zh"):
        subset = [row for row in rows if row["language"] == language]
        if subset:
            per_language[language] = {
                "records": len(subset),
                "exact_accuracy": sum(1 for row in subset if row["correct"]) / len(subset),
            }
    return {
        "count": total,
        "exact_correct": correct,
        "exact_accuracy": correct / total if total else None,
        "macro_f1": float(np.mean(with_support)) if with_support else 0.0,
        "per_class_recall": recalls,
        "min_class_recall": min(with_support) if with_support else None,
        "confusion_matrix": confusion.tolist(),
        "class_order": list(EXPECTED_PROGRAM_IDS),
        "per_language": per_language,
        "rows": rows,
    }


def summarise(result: dict, drop_rows: bool = True) -> dict:
    return {key: value for key, value in result.items() if not (drop_rows and key == "rows")}


def confusion_pairs(result: dict, top: int = 12) -> list[dict]:
    matrix = np.asarray(result["confusion_matrix"])
    order = result["class_order"]
    entries = []
    for i, expected in enumerate(order):
        for j, predicted in enumerate(order):
            if i != j and matrix[i, j] > 0:
                entries.append({"expected": expected, "predicted": predicted,
                                "count": int(matrix[i, j])})
    return sorted(entries, key=lambda entry: -entry["count"])[:top]


def checkpoint_for(stage: str) -> Path | None:
    if stage == "baseline":
        return BASELINE if BASELINE.is_file() else None
    if not HARDENED.is_file():
        return None
    return HARDENED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "final"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    started = time.time()
    checkpoint = checkpoint_for(args.stage)
    if checkpoint is None:
        write_json(OUT_BASELINE if args.stage == "baseline" else OUT_FULL_VAL,
                   {"_doc": f"Task 6T {args.stage}.", "task": "6T",
                    "verdict": "BASELINE_PARSER_UNAVAILABLE" if args.stage == "baseline"
                    else "PARSER_TRAINING_FAILED"})
        print(f"[6t.eval] STOP: checkpoint unavailable for stage {args.stage}", flush=True)
        return 2
    if args.stage == "baseline" and sha256_file(checkpoint) != BASELINE_SHA256:
        write_json(OUT_BASELINE, {"_doc": "Task 6T section 7.1.", "task": "6T",
                                  "verdict": "BASELINE_PARSER_UNAVAILABLE",
                                  "baseline_sha256": sha256_file(checkpoint)})
        return 2

    cfg = load_config(CONFIG_PATH)
    runtime = build_program_parser(cfg, device=args.device, verbose=False)
    load_parser_checkpoint(checkpoint, runtime)
    checkpoint_sha = sha256_file(checkpoint)

    records = val_records()
    if args.limit:
        records = dict(list(records.items())[: args.limit])
    items = {
        "minival240": mini_val_items(records),
        "pairedval20": paired_items(records),
        "fixed24": fixed24_items(),
        "minimal_pairs": pack_items(MINIMAL),
        "stress_v1": pack_items(STRESS),
        "full_val_en": full_val_items(records, "en"),
        "full_val_zh": full_val_items(records, "zh"),
    }
    results = {}
    for name, subset in items.items():
        results[name] = evaluate(runtime, subset)
        print(f"[6t.eval] {args.stage} {name}: {results[name]['exact_correct']}/"
              f"{results[name]['count']} = {results[name]['exact_accuracy']:.4f} "
              f"macroF1 {results[name]['macro_f1']:.4f}", flush=True)

    common = {
        "task": "6T", "stage": args.stage,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha,
                       "authoritative_baseline": checkpoint == BASELINE},
        "parser": {"model": cfg["models"]["qwen_model_id"], "text_only": True, "image_tokens": False,
                   "program_ids": list(EXPECTED_PROGRAM_IDS)},
        "test_split_used": False,
    }

    if args.stage == "baseline":
        payload = {
            "_doc": (
                "Task 6T section 7.1. Frozen baseline ProgramHead (authoritative Task 6S checkpoint) "
                "evaluated BEFORE training on MiniVal240, PairedVal20 members, the exact Task 6S fixed-24 "
                "paraphrases, the Task 6T frozen minimal-pair pack and the Task 6T frozen stress pack. "
                "Both languages are evaluated for the pack-derived sets."
            ),
            **common,
            "minival240": summarise(results["minival240"]),
            "pairedval20": summarise(results["pairedval20"]),
            "fixed24": {**summarise(results["fixed24"]),
                        "failed_prompts": [row["prompt"] for row in results["fixed24"]["rows"]
                                           if not row["correct"]],
                        "previously_failing_prompts": {
                            prompt: next(row["predicted_program"]
                                         for row in results["fixed24"]["rows"]
                                         if row["prompt"] == prompt) for prompt in FIXED24_MUST_PASS},
            },
            "minimal_pairs": summarise(results["minimal_pairs"]),
            "stress_v1": summarise(results["stress_v1"]),
            "smoke": {"note": "the exhaustive v0.2 val audit is written by the `final` stage only"},
            "runtime_seconds": round(time.time() - started, 1),
        }
        write_json(OUT_BASELINE, payload)
        print(f"[6t.eval] baseline: MiniVal240 {payload['minival240']['exact_correct']}/"
              f"{payload['minival240']['count']}, fixed24 {payload['fixed24']['exact_correct']}/24, "
              f"minimal pairs {payload['minimal_pairs']['exact_accuracy']:.4f}, stress "
              f"{payload['stress_v1']['exact_accuracy']:.4f}", flush=True)
        return 0

    # ---------------- final stage artifacts
    training = json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8")) if TRAINING_SUMMARY.is_file() else {}
    full_val_en, full_val_zh = results["full_val_en"], results["full_val_zh"]
    combined_confusion = (np.asarray(full_val_en["confusion_matrix"])
                          + np.asarray(full_val_zh["confusion_matrix"]))
    combined_recall = {}
    for program in EXPECTED_PROGRAM_IDS:
        english = full_val_en["per_class_recall"].get(program) or 0.0
        chinese = full_val_zh["per_class_recall"].get(program) or 0.0
        combined_recall[program] = (english + chinese) / 2
    full_val = {
        "_doc": (
            "Task 6T section 8.1. Full BuildSpatialReason v0.2 val ProgramHead audit (all records, text "
            "only) for the hardened Task 6T checkpoint, in both languages, with exact accuracy, macro "
            "F1, per-class recall and the confusion matrix. No test split."
        ),
        **common, "stage": "final-full-val",
        "val_records": len(records),
        "english": summarise(full_val_en),
        "chinese": summarise(full_val_zh),
        "combined": {
            "count": full_val_en["count"] + full_val_zh["count"],
            "exact_correct": full_val_en["exact_correct"] + full_val_zh["exact_correct"],
            "exact_accuracy": (full_val_en["exact_correct"] + full_val_zh["exact_correct"])
            / (full_val_en["count"] + full_val_zh["count"]),
            "macro_f1": float(np.mean(list(combined_recall.values()))),
            "per_class_recall": combined_recall,
            "confusion_matrix": combined_confusion.tolist(),
            "class_order": list(EXPECTED_PROGRAM_IDS),
        },
        "training_summary_selected_config": training.get("selected_config"),
        "gate": {"accuracy_min": 0.995},
        "runtime_seconds": round(time.time() - started, 1),
    }
    full_val["gate"]["measured"] = full_val["combined"]["exact_accuracy"]
    full_val["gate"]["passed"] = full_val["combined"]["exact_accuracy"] >= 0.995
    full_val["english"]["top_confusions"] = confusion_pairs(full_val_en)
    full_val["chinese"]["top_confusions"] = confusion_pairs(full_val_zh)
    write_json(OUT_FULL_VAL, full_val)

    mini = {
        "_doc": "Task 6T section 8.2. MiniVal240 parser regression (240 English + 240 Chinese queries).",
        **common, "stage": "final-minival240",
        "pack_sha256": sha256_file(PACK_ROOT / "mini_val_240.json"),
        "english": summarise(results["minival240"]),
        "combined": {"count": len(results["minival240"]["rows"]),
                     "exact_correct": sum(1 for row in results["minival240"]["rows"] if row["correct"]),
                     "exact_accuracy": sum(1 for row in results["minival240"]["rows"] if row["correct"])
                     / len(results["minival240"]["rows"])},
        "english_240_of_240": all(row["correct"] for row in results["minival240"]["rows"]
                                  if row["language"] == "en"),
        "failed_prompts": [row["prompt"] for row in results["minival240"]["rows"] if not row["correct"]],
    }
    mini["gate"] = {"required_correct": 240, "measured_correct_english": sum(
        1 for row in results["minival240"]["rows"] if row["language"] == "en" and row["correct"]),
        "passed": mini["english_240_of_240"]}
    write_json(OUT_MINIVAL, mini)

    paired = {
        "_doc": "Task 6T section 8.3. PairedVal20 parser regression (40 members, both languages).",
        **common, "stage": "final-pairedval20",
        "pack_sha256": sha256_file(PACK_ROOT / "paired_val_20.json"),
        "english": summarise(results["pairedval20"]),
        "members_correct_english": sum(1 for row in results["pairedval20"]["rows"]
                                       if row["language"] == "en" and row["correct"]),
        "members_total_english": sum(1 for row in results["pairedval20"]["rows"]
                                     if row["language"] == "en"),
        "per_pair": {},
        "failed_prompts": [row["prompt"] for row in results["pairedval20"]["rows"] if not row["correct"]],
    }
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in results["pairedval20"]["rows"]:
        by_sample[row["sample_id"]].append(row)
    paired["per_pair"] = {sample_id: {"prompts": [row["prompt"] for row in rows],
                                      "correct": [row["correct"] for row in rows]}
                          for sample_id, rows in sorted(by_sample.items())}
    paired["gate"] = {"required_members": 40,
                      "measured_members_english": paired["members_correct_english"],
                      "passed": paired["members_correct_english"] == 40}
    write_json(OUT_PAIRED, paired)

    fixed = {
        "_doc": (
            "Task 6T section 8.4. The exact Task 6S fixed-24 paraphrase list re-run through the hardened "
            "ProgramHead (text only). PASS requires >= 23/24 AND all three previously failing short "
            "Chinese prompts to be correct."
        ),
        **common, "stage": "final-fixed24",
        "result": summarise(results["fixed24"]),
        "failed_prompts": [row["prompt"] for row in results["fixed24"]["rows"] if not row["correct"]],
        "previously_failing_prompts": {
            prompt: {"expected": next(row["expected_program"] for row in results["fixed24"]["rows"]
                                      if row["prompt"] == prompt),
                     "predicted": next(row["predicted_program"] for row in results["fixed24"]["rows"]
                                       if row["prompt"] == prompt),
                     "correct": next(row["correct"] for row in results["fixed24"]["rows"]
                                     if row["prompt"] == prompt)}
            for prompt in FIXED24_MUST_PASS},
    }
    fixed["gate"] = {
        "required_correct_min": 23,
        "measured_correct": results["fixed24"]["exact_correct"],
        "all_three_previously_failing_correct": all(
            entry["correct"] for entry in fixed["previously_failing_prompts"].values()),
    }
    fixed["gate"]["passed"] = bool(fixed["gate"]["measured_correct"] >= 23
                                   and fixed["gate"]["all_three_previously_failing_correct"])
    write_json(OUT_FIXED24, fixed)

    minimal = {
        "_doc": "Task 6T section 8.5. Frozen semantic minimal-pair pack result (requires 100 %).",
        **common, "stage": "final-minimal-pairs",
        "pack_sha256": sha256_file(MINIMAL),
        "result": summarise(results["minimal_pairs"]),
        "per_contrast_group": {},
        "failed_prompts": [row["prompt"] for row in results["minimal_pairs"]["rows"] if not row["correct"]],
    }
    for group in sorted({row["contrast_group"] for row in results["minimal_pairs"]["rows"]}):
        subset = [row for row in results["minimal_pairs"]["rows"] if row["contrast_group"] == group]
        minimal["per_contrast_group"][group] = {
            "prompts": len(subset),
            "correct": sum(1 for row in subset if row["correct"]),
            "accuracy": sum(1 for row in subset if row["correct"]) / len(subset),
        }
    minimal["gate"] = {"required_accuracy": 1.0,
                       "measured_accuracy": results["minimal_pairs"]["exact_accuracy"],
                       "every_group_1_0": all(entry["accuracy"] == 1.0
                                              for entry in minimal["per_contrast_group"].values())}
    minimal["gate"]["passed"] = bool(minimal["gate"]["measured_accuracy"] == 1.0
                                     and minimal["gate"]["every_group_1_0"])
    write_json(OUT_MINIMAL_RESULT, minimal)

    stress = {
        "_doc": (
            "Task 6T section 8.6. Frozen held-out stress pack result: exact accuracy >= 0.95, macro "
            "F1 >= 0.95 and every class recall >= 0.90."
        ),
        **common, "stage": "final-stress-v1",
        "pack_sha256": sha256_file(STRESS),
        "result": summarise(results["stress_v1"]),
        "failed_prompts": [row["prompt"] for row in results["stress_v1"]["rows"] if not row["correct"]],
        "per_language": results["stress_v1"]["per_language"],
    }
    classes = [value for value in results["stress_v1"]["per_class_recall"].values()
               if value is not None]
    stress["classes_below_0_90"] = {program: value for program, value
                                    in results["stress_v1"]["per_class_recall"].items()
                                    if value is not None and value < 0.90}
    stress["gate"] = {
        "exact_accuracy_min": 0.95, "macro_f1_min": 0.95, "class_recall_min": 0.90,
        "measured_accuracy": results["stress_v1"]["exact_accuracy"],
        "measured_macro_f1": results["stress_v1"]["macro_f1"],
        "measured_min_class_recall": min(classes) if classes else None,
    }
    stress["gate"]["passed"] = bool(stress["gate"]["measured_accuracy"] >= 0.95
                                    and stress["gate"]["measured_macro_f1"] >= 0.95
                                    and stress["gate"]["measured_min_class_recall"] is not None
                                    and stress["gate"]["measured_min_class_recall"] >= 0.90)
    write_json(OUT_STRESS_RESULT, stress)

    print(f"[6t.eval] final: full val en {full_val_en['exact_accuracy']:.4f} / zh "
          f"{full_val_zh['exact_accuracy']:.4f} / combined {full_val['combined']['exact_accuracy']:.4f}"
          f" | MiniVal240 {mini['english_240_of_240']} | paired "
          f"{paired['members_correct_english']}/40 | fixed24 {results['fixed24']['exact_correct']}/24 | "
          f"minimal {minimal['gate']['passed']} | stress {stress['gate']['passed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
