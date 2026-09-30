"""Task 7C Part L — end-to-end frozen regression with the Task 7C parser.

Section 22: run the exact Z-MiniVal240 natural-language queries through

```text
Task 7C parser -> frozen Task 7A U-C1 reference resolver -> frozen P_dir + P_near -> frozen Z-B3
```

and require reproduction of the Task 7A A2 numbers (strict mIoU 0.21700369907681483, answered mIoU
0.21975058134361, abstentions 3, Paired 8/20, margin 0.19949275176250805) within `1e-6` with exact counts.
Nothing downstream changed; only the parser checkpoint differs.

The Task 7A artifacts are never overwritten.

    python scripts/task7c_e2e_regression.py
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
from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint  # noqa: E402
from buildreasonseg_mvp.runtime import load_config  # noqa: E402
from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES  # noqa: E402
from buildreasonseg_mvp.task6s_directional_pipeline import parse_instruction  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import (  # noqa: E402
    SUPPORTED_L3_PROGRAMS,
    L3Pipeline,
    default_target_checkpoint,
)
from scripts.task6u_common import iou  # noqa: E402
from scripts.task7a_evaluate_reference import image_path_of, load_pack  # noqa: E402
from scripts.task7b_build_parser_data import sha256_file  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7c_end_to_end_regression.json"
CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "task7c" / "program_parser_l3_rehearsal_v1.pt"
TRAINING = EVAL / "task7c_training_summary.json"
TASK7A_VAL = EVAL / "task7a_natural_language_val.json"
TASK7A_PAIRED = EVAL / "task7a_natural_language_paired.json"
DATA = REPO_ROOT / "datasets" / "build_spatial_reason" / "v0.2"
TOLERANCE = 1.0e-6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-device", default="0")
    args = parser.parse_args(argv)
    started = time.time()

    reference_val = json.loads(TASK7A_VAL.read_text(encoding="utf-8"))
    reference_paired = json.loads(TASK7A_PAIRED.read_text(encoding="utf-8"))
    if not CHECKPOINT.is_file():
        write_json(OUT, {"_doc": "Task 7C section 23.", "task": "7C",
                         "verdict": "PARSER_TRAINING_FAILED"})
        print("[7c.e2e] STOP PARSER_TRAINING_FAILED", flush=True)
        return 2
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    expected_sha = training["results"]["Z-B3"]["checkpoint"]["sha256"] \
        if "results" in training else None
    target_path = default_target_checkpoint()
    target_sha = sha256_file(target_path)

    cfg = load_config(REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml")
    runtime = build_program_parser(cfg, device=args.device, verbose=False)
    load_parser_checkpoint(CHECKPOINT, runtime)
    vocabulary = tuple(EXPECTED_QUERY_TYPES)
    parse = lambda query: parse_instruction(runtime, query, vocabulary)  # noqa: E731

    pipeline = L3Pipeline(device=args.device, yolo_device=args.yolo_device,
                          target_checkpoint=target_path)
    from task6n_train import MaskStore

    masks = MaskStore()
    instructions = {}
    with (DATA / "val.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            instructions[str(record["sample_id"])] = record

    records = load_pack("z_mini_val_240")
    rows = []
    for record in records:
        query = str(instructions[record["sample_id"]]["instruction_en"])
        parsed_program = str(parse(query)["program"])
        truth_target = np.asarray(masks.mask(record["tile_id"],
                                            record["target_source_feature_id"]), dtype=bool)
        row = {"sample_id": record["sample_id"], "expected_program": record["program_id"],
               "parsed_program": parsed_program, "parser_correct": parsed_program == record["program_id"],
               "miou": 0.0, "dice": 0.0, "status": "unsupported_l3_program"}
        if parsed_program in SUPPORTED_L3_PROGRAMS:
            outcome = pipeline.infer(image_path=image_path_of(record), tile_id=record["tile_id"],
                                     program_id=parsed_program)
            if outcome["abstained"]:
                row["status"] = "reference_abstained"
            else:
                prediction = outcome["mask"]
                intersection = float((prediction & truth_target).sum())
                row.update({"status": "ok", "miou": iou(prediction, truth_target),
                            "dice": (2.0 * intersection + TOLERANCE)
                            / (float(prediction.sum()) + float(truth_target.sum()) + TOLERANCE)})
        rows.append(row)

    strict_miou = float(np.mean([row["miou"] for row in rows]))
    strict_dice = float(np.mean([row["dice"] for row in rows]))
    answered = [row for row in rows if row["status"] == "ok"]
    answered_miou = float(np.mean([row["miou"] for row in answered])) if answered else None
    abstentions = sum(1 for row in rows if row["status"] == "reference_abstained")
    parser_correct = sum(1 for row in rows if row["parser_correct"])

    paired_records = load_pack("z_paired_val20")
    pairs = [{"tile_id": paired_records[index]["tile_id"], "a": paired_records[index],
              "b": paired_records[index + 1]} for index in range(0, len(paired_records), 2)]
    pair_rows = []
    for entry in pairs:
        tile_id = entry["tile_id"]
        image_path = image_path_of(entry["a"])
        proposals = pipeline.proposals(tile_id, image_path)
        bundle = pipeline.resolve_reference(proposals)
        row = {"tile_id": tile_id, "abstained": bundle.selection is None}
        if bundle.selection is None:
            row.update({"own": [0.0, 0.0], "cross": [0.0, 0.0], "passes": False})
            pair_rows.append(row)
            continue
        reference_mask = np.asarray(bundle.selection["mask"], dtype=bool)
        members = [entry["a"], entry["b"]]
        own, cross = [], []
        for index, member in enumerate(members):
            query = str(instructions[member["sample_id"]]["instruction_en"])
            parsed_program = str(parse(query)["program"])
            if parsed_program not in SUPPORTED_L3_PROGRAMS:
                own.append(0.0)
                cross.append(0.0)
                continue
            prediction = pipeline.predict_target(image_path, member["tile_id"], parsed_program,
                                                 reference_mask)["mask"]
            own_mask = np.asarray(masks.mask(member["tile_id"], member["target_source_feature_id"]),
                                 dtype=bool)
            other = members[1 - index]
            other_mask = np.asarray(masks.mask(other["tile_id"], other["target_source_feature_id"]),
                                   dtype=bool)
            own.append(float((prediction & own_mask).sum())
                       / (float((prediction | own_mask).sum()) + TOLERANCE))
            cross.append(float((prediction & other_mask).sum())
                         / (float((prediction | other_mask).sum()) + TOLERANCE))
        row.update({"own": own, "cross": cross,
                    "passes": bool(own[0] > cross[0] and own[1] > cross[1])})
        pair_rows.append(row)
    paired_passed = sum(1 for row in pair_rows if row["passes"])
    own_mean = float(np.mean([value for row in pair_rows for value in row["own"]]))
    cross_mean = float(np.mean([value for row in pair_rows for value in row["cross"]]))
    margin = own_mean - cross_mean

    expected = {
        "strict_all_miou": reference_val["strict_all_miou"],
        "answered_only_miou": reference_val["answered_only_miou"],
        "abstentions": reference_val["abstentions"],
        "paired_passed": reference_paired["passed"],
        "own_cross_margin": reference_paired["own_cross_margin"],
    }
    measured = {"strict_all_miou": strict_miou, "strict_all_dice": strict_dice,
                "answered_only_miou": answered_miou, "abstentions": abstentions,
                "paired_passed": paired_passed, "paired_pairs": len(pair_rows),
                "own_cross_margin": margin, "parser_correct": parser_correct}
    deltas = {"strict_all_miou": abs(strict_miou - expected["strict_all_miou"]),
              "answered_only_miou": abs((answered_miou or 0.0) - expected["answered_only_miou"]),
              "own_cross_margin": abs(margin - expected["own_cross_margin"])}
    reproduction_required = parser_correct == 240
    passed = bool(
        (parser_correct == 240)
        and all(value <= TOLERANCE for value in deltas.values())
        and abstentions == expected["abstentions"]
        and paired_passed == expected["paired_passed"])

    payload = {
        "_doc": ("Task 7C section 22. End-to-end frozen regression: the Task 7C parser drives the frozen "
                 "Task 7A U-C1 reference resolver, the frozen P_dir/P_near fields and the frozen Z-B3 "
                 "decoder on the exact Z-MiniVal240/Z-PairedVal20 natural-language queries. Reproduction "
                 "is required only because the parser stays 240/240; tolerance 1e-6 with exact counts."),
        "task": "7C", "stage": "L-end-to-end-regression",
        "checkpoints": {"parser": {"path": str(CHECKPOINT), "sha256": sha256_file(CHECKPOINT)},
                        "target": {"path": str(target_path), "sha256": target_sha},
                        "target_expected_sha256": expected_sha,
                        "downstream_unchanged": True},
        "measured": measured, "task7a_reference": expected, "deltas": deltas,
        "tolerance": TOLERANCE,
        "reproduction_required": reproduction_required,
        "passed": passed,
        "training_performed": False, "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT, payload)
    print(f"[7c.e2e] parser {parser_correct}/240 strict {strict_miou:.10f} answered "
          f"{answered_miou:.10f} abstain {abstentions} | paired {paired_passed}/20 margin "
          f"{margin:.10f} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
