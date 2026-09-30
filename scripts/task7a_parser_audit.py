"""Task 7A Parts F-G — hardened ProgramHead audit on Z-MiniVal240 and the fixed 24-prompt paraphrase pack.

Part F (section 12): run the frozen Task 6T hardened ProgramHead on the **actual natural-language queries**
stored in the exact Z-MiniVal240 records (join by `sample_id` against the frozen v0.2 val split) and report
exact canonical-program accuracy, per-class recall, the confusion matrix and how many queries were predicted
into non-Task-7A canonical classes.

Part G (sections 15-16): freeze exactly 24 paraphrase prompts (6 per L3 class: 3 Chinese + 3 English) —
including the eight exact compact/contrast forms required by the task file — **before** running the parser,
then report total exact accuracy, per program, per language and every failure with its predicted canonical id.

No parser retraining, no threshold tuning.

    python scripts/task7a_parser_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import (  # noqa: E402
    SUPPORTED_L3_PROGRAMS,
    default_parser_checkpoint,
    sha256_file,
)

EVAL = REPO_ROOT / "evaluation"
PACK_ROOT = REPO_ROOT / "artifacts" / "task6z" / "packs"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
OUT_PARSER = EVAL / "task7a_parser_l3_val.json"
OUT_PARAPHRASE_PACK = EVAL / "task7a_l3_paraphrase_pack.json"
OUT_PARAPHRASE_RESULT = EVAL / "task7a_l3_paraphrase_result.json"
TASK6T = EVAL / "task6t_training_summary.json"

#: The eight exact compact/contrast prompts required by section 15 (must all be present).
REQUIRED_COMPACT_PROMPTS: tuple[tuple[str, str, str], ...] = (
    ("largest_to_right_of_to_nearest", "zh", "分割面积最大的建筑物右侧最近的建筑物。"),
    ("largest_to_right_of_to_nearest", "en",
     "segment the building nearest to the right of the largest building"),
    ("largest_to_left_of_to_nearest", "zh", "找出最大建筑左边最近的建筑。"),
    ("largest_to_left_of_to_nearest", "en",
     "find the closest building to the left of the largest building"),
    ("largest_to_above_to_nearest", "zh", "找出最大建筑上方最近的建筑。"),
    ("largest_to_above_to_nearest", "en", "find the nearest building above the largest building"),
    ("largest_to_below_to_nearest", "zh", "找出最大建筑下方最近的建筑。"),
    ("largest_to_below_to_nearest", "en", "find the closest building below the largest building"),
)
#: The remaining 16 prompts: 2 Chinese + 2 English per program, all unambiguous paraphrases.
EXTRA_PROMPTS: tuple[tuple[str, str, str], ...] = (
    ("largest_to_left_of_to_nearest", "zh", "请分割位于最大建筑左侧、且与它最近的那栋建筑。"),
    ("largest_to_left_of_to_nearest", "zh", "在最大建筑的左侧找距离最近的一栋建筑并分割出来。"),
    ("largest_to_left_of_to_nearest", "en",
     "please segment the building on the left side of the largest building that is closest to it"),
    ("largest_to_left_of_to_nearest", "en",
     "segment the nearest building on the left-hand side of the largest building"),
    ("largest_to_right_of_to_nearest", "zh", "请分割位于最大建筑右侧、且与它最近的那栋建筑。"),
    ("largest_to_right_of_to_nearest", "zh", "在最大建筑的右侧找距离最近的一栋建筑并分割出来。"),
    ("largest_to_right_of_to_nearest", "en",
     "please segment the building on the right side of the largest building that is closest to it"),
    ("largest_to_right_of_to_nearest", "en",
     "segment the nearest building on the right-hand side of the largest building"),
    ("largest_to_above_to_nearest", "zh", "请分割位于最大建筑上方、且与它最近的那栋建筑。"),
    ("largest_to_above_to_nearest", "zh", "在最大建筑的上方找距离最近的一栋建筑并分割出来。"),
    ("largest_to_above_to_nearest", "en",
     "please segment the building above the largest building that is closest to it"),
    ("largest_to_above_to_nearest", "en",
     "segment the nearest building located above the largest building"),
    ("largest_to_below_to_nearest", "zh", "请分割位于最大建筑下方、且与它最近的那栋建筑。"),
    ("largest_to_below_to_nearest", "zh", "在最大建筑的下方找距离最近的一栋建筑并分割出来。"),
    ("largest_to_below_to_nearest", "en",
     "please segment the building below the largest building that is closest to it"),
    ("largest_to_below_to_nearest", "en",
     "segment the nearest building located below the largest building"),
)


def instruction_by_sample_id() -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    with (DATA / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            lookup[str(record["sample_id"])] = record
    return lookup


def load_parser(device: str):
    from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint
    from buildreasonseg_mvp.runtime import load_config
    from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES
    from buildreasonseg_mvp.task6s_directional_pipeline import parse_instruction

    training = json.loads(TASK6T.read_text(encoding="utf-8"))
    checkpoint = default_parser_checkpoint()
    expected = training["checkpoint"]["sha256"]
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
        return None, None, None, expected, training
    config = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(config, device=device, verbose=False)
    load_parser_checkpoint(checkpoint, runtime)
    vocabulary = tuple(EXPECTED_QUERY_TYPES)
    return (lambda query: parse_instruction(runtime, query, vocabulary)), checkpoint, vocabulary, \
        expected, training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    started = time.time()

    # ---------------- Part G first: freeze the paraphrase pack BEFORE any parser run
    prompts = []
    for program, language, text in (*REQUIRED_COMPACT_PROMPTS, *EXTRA_PROMPTS):
        prompts.append({"id": f"p{len(prompts) + 1:02d}", "program": program, "language": language,
                        "text": text,
                        "required_compact": any(entry[:3] == (program, language, text)
                                                for entry in REQUIRED_COMPACT_PROMPTS)})
    counts = Counter((entry["program"], entry["language"]) for entry in prompts)
    pack = {
        "_doc": ("Task 7A section 15. Fixed 24-prompt L3 paraphrase audit pack, frozen before any parser "
                 "run in Task 7A: 6 prompts per L3 program (3 Chinese, 3 English), including the eight "
                 "exact compact/contrast forms required by the task file. No prompt was used for training."),
        "task": "7A", "stage": "G-paraphrase-pack",
        "programs": list(SUPPORTED_L3_PROGRAMS),
        "prompt_count": len(prompts),
        "per_program": {program: sum(1 for entry in prompts if entry["program"] == program)
                        for program in SUPPORTED_L3_PROGRAMS},
        "per_program_language": {f"{program}|{language}": counts[(program, language)]
                                 for program in SUPPORTED_L3_PROGRAMS
                                 for language in ("zh", "en")},
        "required_compact_present": all(any(entry["program"] == program and entry["language"] == language
                                            and entry["text"] == text for entry in prompts)
                                        for program, language, text in REQUIRED_COMPACT_PROMPTS),
        "required_compact_count": len(REQUIRED_COMPACT_PROMPTS),
        "training_use": False,
        "prompts": prompts,
    }
    write_json(OUT_PARAPHRASE_PACK, pack)
    print(f"[7a.parser] paraphrase pack frozen: {len(prompts)} prompts, per program "
          f"{pack['per_program']}, compact present {pack['required_compact_present']}", flush=True)

    parse, checkpoint, vocabulary, expected, training = load_parser(args.device)
    if parse is None:
        write_json(OUT_PARSER, {"_doc": "Task 7A section 4.", "task": "7A",
                                "verdict": "PARSER_CHECKPOINT_UNAVAILABLE",
                                "expected_sha256": expected})
        write_json(OUT_PARAPHRASE_RESULT, {"_doc": "Task 7A section 16.", "task": "7A",
                                           "verdict": "PARSER_CHECKPOINT_UNAVAILABLE"})
        print("[7a.parser] STOP PARSER_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2

    # ---------------- Part F: canonical-query parser audit on Z-MiniVal240
    records = json.loads((PACK_ROOT / "z_mini_val_240.json").read_text(encoding="utf-8"))["records"]
    instructions = instruction_by_sample_id()
    rows = []
    missing = 0
    for record in records:
        source = instructions.get(record["sample_id"])
        if source is None:
            missing += 1
            continue
        query = str(source["instruction_en"])
        parsed = parse(query)
        rows.append({"sample_id": record["sample_id"], "expected_program": record["program_id"],
                     "parsed_program": parsed["program"],
                     "correct": parsed["program"] == record["program_id"],
                     "direction": record["direction"],
                     "query_en": query, "query_zh": str(source.get("instruction_zh", ""))})
    programs = list(SUPPORTED_L3_PROGRAMS)
    confusion = {expected_program: Counter(row["parsed_program"] for row in rows
                                           if row["expected_program"] == expected_program)
                 for expected_program in programs}
    per_class_recall = {
        program: (sum(1 for row in rows if row["expected_program"] == program
                      and row["correct"])
                  / max(1, sum(1 for row in rows if row["expected_program"] == program)))
        for program in programs}
    accuracy = sum(1 for row in rows if row["correct"]) / max(1, len(rows))
    in_scope = set(programs)
    payload = {
        "_doc": ("Task 7A section 12. Frozen Task 6T hardened ProgramHead run on the actual natural-language "
                 "queries (`instruction_en`) of the exact Z-MiniVal240 records, joined by sample_id against "
                 "the frozen v0.2 val split. Text-only parser; no parser training in Task 7A."),
        "task": "7A", "stage": "F-parser-l3-val",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                       "expected_sha256": expected,
                       "matches": sha256_file(checkpoint) == expected},
        "parser": {"text_only": True, "vocabulary_size": len(vocabulary),
                   "selected_config": training["selected_config"],
                   "holdout_accuracy": training.get("selected", {}).get("holdout_accuracy"),
                   "holdout_macro_f1": training.get("selected", {}).get("holdout_macro_f1")},
        "pack": {"name": "z_mini_val_240", "records": len(records),
                 "joined_records": len(rows), "missing_instruction": missing},
        "exact_accuracy": accuracy,
        "exact_correct": sum(1 for row in rows if row["correct"]),
        "per_class_recall": per_class_recall,
        "confusion_matrix": {program: dict(confusion[program]) for program in programs},
        "predicted_out_of_scope_count": sum(1 for row in rows
                                            if row["parsed_program"] not in in_scope),
        "predicted_out_of_scope_examples": sorted({row["parsed_program"] for row in rows
                                                   if row["parsed_program"] not in in_scope}),
        "rows": rows,
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_PARSER, payload)
    print(f"[7a.parser] canonical queries: {payload['exact_correct']}/{len(rows)} exact "
          f"({accuracy:.4f}); out-of-scope {payload['predicted_out_of_scope_count']}", flush=True)

    # ---------------- Part G result
    paraphrase_rows = []
    for entry in prompts:
        parsed = parse(str(entry["text"]))
        paraphrase_rows.append({**entry, "parsed_program": parsed["program"],
                                "correct": parsed["program"] == entry["program"]})
    total_correct = sum(1 for row in paraphrase_rows if row["correct"])
    compact_rows = [row for row in paraphrase_rows if row["required_compact"]]
    result = {
        "_doc": ("Task 7A section 16. Parser-only result of the fixed 24-prompt paraphrase audit (pack "
                 "frozen before this run). No parser retraining, no threshold tuning."),
        "task": "7A", "stage": "G-paraphrase-result",
        "pack": {"path": str(OUT_PARAPHRASE_PACK), "prompt_count": len(prompts)},
        "checkpoint": {"sha256": sha256_file(checkpoint), "expected_sha256": expected},
        "total_correct": total_correct, "total": len(paraphrase_rows),
        "exact_accuracy": total_correct / max(1, len(paraphrase_rows)),
        "per_program": {program: {"correct": sum(1 for row in paraphrase_rows
                                                 if row["program"] == program and row["correct"]),
                                  "total": sum(1 for row in paraphrase_rows
                                               if row["program"] == program)}
                        for program in programs},
        "per_language": {language: {"correct": sum(1 for row in paraphrase_rows
                                                   if row["language"] == language and row["correct"]),
                                    "total": sum(1 for row in paraphrase_rows
                                                 if row["language"] == language)}
                         for language in ("zh", "en")},
        "required_compact": {"correct": sum(1 for row in compact_rows if row["correct"]),
                             "total": len(compact_rows),
                             "all_correct": all(row["correct"] for row in compact_rows),
                             "failures": [row for row in compact_rows if not row["correct"]]},
        "failures": [{key: row[key] for key in ("id", "program", "language", "text",
                                                "parsed_program")}
                     for row in paraphrase_rows if not row["correct"]],
        "rows": paraphrase_rows,
        "parser_retrained": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_PARAPHRASE_RESULT, result)
    print(f"[7a.parser] paraphrases: {total_correct}/{len(paraphrase_rows)} exact; compact "
          f"{result['required_compact']['correct']}/{result['required_compact']['total']} "
          f"(all correct: {result['required_compact']['all_correct']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
