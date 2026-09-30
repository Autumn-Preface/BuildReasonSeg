"""Task 7A Part F — natural-language L3 end-to-end chain (A2) on Z-MiniVal240 and Z-PairedVal20.

The hardened Task 6T ProgramHead turns the natural-language query into a canonical program, a Task 7A scope
check runs, and only then does the frozen predicted-reference chain execute:

```text
natural language -> hardened ProgramHead -> canonical program -> Task 7A scope check
                 -> predicted largest reference -> P_dir + P_near -> frozen Z-B3 -> target mask
```

Rules (section 13): parser wrong relative to the expected program -> strict target IoU = 0; parser valid but
out-of-scope -> status `unsupported_l3_program`, no downstream work, strict IoU = 0; reference abstention ->
strict IoU = 0.

Writes `evaluation/task7a_natural_language_val.json` and
`evaluation/task7a_natural_language_paired.json`.

    python scripts/task7a_evaluate_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import (  # noqa: E402
    SUPPORTED_L3_PROGRAMS,
    L3Pipeline,
    default_parser_checkpoint,
    default_target_checkpoint,
    sha256_file,
)
from scripts.task7a_evaluate_reference import load_pack, image_path_of  # noqa: E402
from scripts.task6u_common import iou  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT_VAL = EVAL / "task7a_natural_language_val.json"
OUT_PAIRED = EVAL / "task7a_natural_language_paired.json"
TASK6Z_TRAINING = EVAL / "task6z_training.json"
TASK6T = EVAL / "task6t_training_summary.json"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
TOLERANCE = 1.0e-6


def instruction_lookup() -> dict[str, dict]:
    lookup = {}
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

    config = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(config, device=device, verbose=False)
    load_parser_checkpoint(default_parser_checkpoint(), runtime)
    vocabulary = tuple(EXPECTED_QUERY_TYPES)
    return lambda query: parse_instruction(runtime, query, vocabulary), vocabulary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    training = json.loads(TASK6Z_TRAINING.read_text(encoding="utf-8"))
    expected_sha = training["results"]["Z-B3"]["checkpoint"]["sha256"]
    target_path = default_target_checkpoint()
    if not target_path.is_file() or sha256_file(target_path) != expected_sha:
        write_json(OUT_VAL, {"_doc": "Task 7A section 3.", "task": "7A",
                             "verdict": "L3_CHECKPOINT_UNAVAILABLE"})
        print("[7a.a2] STOP L3_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2
    parser_training = json.loads(TASK6T.read_text(encoding="utf-8"))
    parser_sha = sha256_file(default_parser_checkpoint())
    if parser_sha != parser_training["checkpoint"]["sha256"]:
        write_json(OUT_VAL, {"_doc": "Task 7A section 4.", "task": "7A",
                             "verdict": "PARSER_CHECKPOINT_UNAVAILABLE"})
        print("[7a.a2] STOP PARSER_CHECKPOINT_UNAVAILABLE", flush=True)
        return 2

    pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device,
                          target_checkpoint=target_path, expected_target_sha256=expected_sha)
    parse, vocabulary = load_parser(args.device)
    from task6n_train import MaskStore

    masks = MaskStore()
    instructions = instruction_lookup()

    records = load_pack("z_mini_val_240")
    rows = []
    for record in records:
        source = instructions[record["sample_id"]]
        query = str(source["instruction_en"])
        parsed = parse(query)
        parsed_program = str(parsed["program"])
        truth_target = np.asarray(masks.mask(record["tile_id"],
                                            record["target_source_feature_id"]), dtype=bool)
        row = {"sample_id": record["sample_id"], "direction": record["direction"],
               "expected_program": record["program_id"], "parsed_program": parsed_program,
               "parser_correct": parsed_program == record["program_id"],
               "query_en": query}
        if parsed_program not in SUPPORTED_L3_PROGRAMS:
            row.update({"status": "unsupported_l3_program", "miou": 0.0, "dice": 0.0,
                        "precision_at_0_5": 0.0, "reference_abstained": None})
            rows.append(row)
            continue
        outcome = pipeline.infer(image_path=image_path_of(record), tile_id=record["tile_id"],
                                 program_id=parsed_program)
        if outcome["abstained"]:
            row.update({"status": "reference_abstained", "miou": 0.0, "dice": 0.0,
                        "precision_at_0_5": 0.0, "reference_abstained": True,
                        "abstention_reason": outcome["abstention_reason"]})
            rows.append(row)
            continue
        prediction = outcome["mask"]
        intersection = float((prediction & truth_target).sum())
        row.update({"status": "ok", "reference_abstained": False,
                    "miou": iou(prediction, truth_target),
                    "dice": (2.0 * intersection + TOLERANCE)
                    / (float(prediction.sum()) + float(truth_target.sum()) + TOLERANCE),
                    "precision_at_0_5": (intersection + TOLERANCE)
                    / (float(prediction.sum()) + TOLERANCE)})
        rows.append(row)

    strict_miou = float(np.mean([row["miou"] for row in rows]))
    strict_dice = float(np.mean([row["dice"] for row in rows]))
    answered = [row for row in rows if row["status"] == "ok"]
    parser_correct = [row for row in rows if row["parser_correct"]]
    per_direction = {}
    for direction in ("above", "below", "left", "right"):
        subset = [row for row in rows if row["direction"] == direction]
        per_direction[direction] = {"records": len(subset),
                                    "miou": float(np.mean([row["miou"] for row in subset]))}

    payload = {
        "_doc": ("Task 7A section 13. A2 natural-language L3 end-to-end integration on Z-MiniVal240: the "
                 "source image and the natural-language query are the only inputs. Parser errors, "
                 "out-of-scope programs and reference abstentions all score strict target IoU = 0. No "
                 "training, no test split."),
        "task": "7A", "stage": "A2-natural-language-val",
        "pack": {"name": "z_mini_val_240", "records": len(records)},
        "checkpoints": {"parser": {"path": str(default_parser_checkpoint()), "sha256": parser_sha,
                                   "expected_sha256": parser_training["checkpoint"]["sha256"]},
                        "target": {"path": str(target_path), "sha256": sha256_file(target_path),
                                   "expected_sha256": expected_sha}},
        "parser": {"vocabulary_size": len(vocabulary),
                   "exact_accuracy": sum(1 for row in rows if row["parser_correct"]) / len(rows),
                   "exact_correct": sum(1 for row in rows if row["parser_correct"]),
                   "records": len(rows),
                   "out_of_scope_predictions": sum(1 for row in rows
                                                   if row["status"] == "unsupported_l3_program")},
        "strict_all_miou": strict_miou, "strict_all_dice": strict_dice,
        "precision_at_0_5": float(np.mean([row["precision_at_0_5"] for row in rows])),
        "answered_records": len(answered),
        "answered_only_miou": float(np.mean([row["miou"] for row in answered]))
        if answered else None,
        "answered_only_dice": float(np.mean([row["dice"] for row in answered]))
        if answered else None,
        "abstentions": sum(1 for row in rows if row["status"] == "reference_abstained"),
        "parser_failures": sum(1 for row in rows if not row["parser_correct"]),
        "parser_correct_subset_miou": float(np.mean([row["miou"] for row in parser_correct]))
        if parser_correct else None,
        "per_direction": per_direction,
        "rows": rows,
        "training_performed": False, "test_split_used": False, "ground_truth_used_in_inference": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_VAL, payload)
    print(f"[7a.a2] parser {payload['parser']['exact_correct']}/{len(rows)} | strict {strict_miou:.4f} "
          f"answered {payload['answered_only_miou']:.4f} abstain {payload['abstentions']} "
          f"out-of-scope {payload['parser']['out_of_scope_predictions']}", flush=True)

    # ---------------- natural-language PairedVal20
    paired_records = load_pack("z_paired_val20")
    pairs = [{"tile_id": paired_records[index]["tile_id"], "a": paired_records[index],
              "b": paired_records[index + 1]} for index in range(0, len(paired_records), 2)]
    pair_rows = []
    for entry in pairs:
        members = [entry["a"], entry["b"]]
        own, cross = [], []
        parser_errors = 0
        abstentions = 0
        for index, member in enumerate(members):
            query = str(instructions[member["sample_id"]]["instruction_en"])
            parsed_program = str(parse(query)["program"])
            if parsed_program != member["program_id"]:
                parser_errors += 1
            if parsed_program not in SUPPORTED_L3_PROGRAMS:
                own.append(0.0)
                cross.append(0.0)
                continue
            outcome = pipeline.infer(image_path=image_path_of(member), tile_id=member["tile_id"],
                                     program_id=parsed_program)
            if outcome["abstained"]:
                abstentions += 1
                own.append(0.0)
                cross.append(0.0)
                continue
            prediction = outcome["mask"]
            own_mask = np.asarray(masks.mask(member["tile_id"], member["target_source_feature_id"]),
                                 dtype=bool)
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other["tile_id"], other["target_source_feature_id"]),
                                   dtype=bool)
            own.append(float((prediction & own_mask).sum())
                       / (float((prediction | own_mask).sum()) + TOLERANCE))
            cross.append(float((prediction & other_mask).sum())
                         / (float((prediction | other_mask).sum()) + TOLERANCE))
        pair_rows.append({"tile_id": entry["tile_id"], "own": own, "cross": cross,
                          "passes": bool(own[0] > cross[0] and own[1] > cross[1]),
                          "parser_errors": parser_errors, "reference_abstentions": abstentions})
    passed = sum(1 for row in pair_rows if row["passes"])
    own_mean = float(np.mean([value for row in pair_rows for value in row["own"]]))
    cross_mean = float(np.mean([value for row in pair_rows for value in row["cross"]]))
    paired_payload = {
        "_doc": ("Task 7A section 14. A2 natural-language PairedVal20: each pair member's own "
                 "natural-language query is parsed and executed separately."),
        "task": "7A", "stage": "A2-natural-language-paired",
        "pack": {"name": "z_paired_val20", "pairs": len(pair_rows)},
        "parser_correct_members": sum(2 - row["parser_errors"] for row in pair_rows),
        "members": 2 * len(pair_rows),
        "passed": passed, "pairs": len(pair_rows), "pass_rate": passed / max(1, len(pair_rows)),
        "mean_own_iou": own_mean, "mean_cross_iou": cross_mean,
        "own_cross_margin": own_mean - cross_mean,
        "parser_error_pairs": sum(1 for row in pair_rows if row["parser_errors"] > 0),
        "reference_abstention_pairs": sum(1 for row in pair_rows
                                          if row["reference_abstentions"] > 0),
        "rows": pair_rows,
        "training_performed": False, "test_split_used": False,
    }
    write_json(OUT_PAIRED, paired_payload)
    print(f"[7a.a2] paired {passed}/{len(pair_rows)} margin "
          f"{paired_payload['own_cross_margin']:+.4f} parser-correct members "
          f"{paired_payload['parser_correct_members']}/40", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
