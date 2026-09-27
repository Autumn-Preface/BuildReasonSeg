"""Task 6J section 14: Stage J3 -- predicted program + oracle candidates.

Gated on J2. Pipeline: instruction -> trained ProgramHead -> predicted canonical program ->
ORACLE candidate set (diagnostic-only, frozen component maps) -> executor -> selected candidate.
GT is used only for scoring; there is NO fallback to the GT program.

Reports exact selected-target accuracy, paired selection /20, and program-error vs
executor-error attribution. Gate: selected-target accuracy >= 0.85, paired >= 17/20.

Writes `evaluation/task6j_j3_predicted_program_oracle_candidates.json`.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from buildreasonseg_mvp.program_parser import build_program_parser, load_parser_checkpoint  # noqa: E402
from buildreasonseg_mvp.runtime import enable_determinism, load_config  # noqa: E402
from buildreasonseg_mvp.structured_grounding import execute_program_by_id  # noqa: E402

from task6j_common import (  # noqa: E402
    EVAL,
    fixed_validation_material,
    gate_report,
    oracle_candidate_set,
    paired_sample_lists,
    write_json,
)

CONFIG = REPO_ROOT / "configs" / "mvp" / "task6j_program_parser.yaml"
J2_JSON = EVAL / "task6j_j2_program_parser.json"
OUT = EVAL / "task6j_j3_predicted_program_oracle_candidates.json"


def main() -> int:
    started = time.time()
    j2 = json.loads(J2_JSON.read_text(encoding="utf-8"))
    if not j2["final"]["gate"]["passed"]:
        raise SystemExit("J2 gate did not pass; section 14 forbids J3")
    cfg = load_config(CONFIG)
    enable_determinism(int(cfg["seed"]), strict=True)
    runtime = build_program_parser(cfg, verbose=True)
    load_report = load_parser_checkpoint(Path(j2["final"]["checkpoint"]["path"]), runtime)

    from thresholds import load_config as load_relation_config

    relation_config = load_relation_config()
    val_samples, pairs = fixed_validation_material()
    a_samples, b_samples = paired_sample_lists(pairs)
    val_records = [sample.raw for sample in val_samples]
    paired_records = [sample.raw for sample in a_samples + b_samples]

    @torch.no_grad()
    def predicted_programs(records):
        runtime.qwen.eval()
        predictions = []
        for start in range(0, len(records), 32):
            chunk = records[start:start + 32]
            batch = runtime.build_batch(
                [r["instruction_zh"] for r in chunk], [r["query_type"] for r in chunk]
            ).to(runtime.device)
            logits, _hidden = runtime.forward(batch)
            predictions.extend(torch.argmax(logits, dim=1).tolist())
        from buildreasonseg_mvp.program_parser import EXPECTED_PROGRAM_IDS

        return [EXPECTED_PROGRAM_IDS[index] for index in predictions]

    def run(sample, predicted_program):
        record = sample.raw
        candidates = oracle_candidate_set(sample)
        result = execute_program_by_id(predicted_program, candidates, relation_config)
        program_correct = predicted_program == record["query_type"]
        selected_correct = (
            (not result.abstained) and result.selected_id == int(record["target_component_id"])
        )
        if not program_correct:
            failure = "wrong_predicted_program"
        elif result.abstained:
            failure = "executor_abstained"
        elif not selected_correct:
            failure = "executor_wrong_selection"
        else:
            failure = None
        return {
            "sample_id": str(sample.sample_id),
            "image_id": str(sample.image_id),
            "level": int(sample.level),
            "query_type": str(sample.query_type),
            "predicted_program": predicted_program,
            "program_correct": bool(program_correct),
            "selected_id": result.selected_id,
            "target_id": int(record["target_component_id"]),
            "correct": bool(selected_correct),
            "abstained": bool(result.abstained),
            "abstain_reason": result.reason,
            "failure": failure,
            "n_candidates": len(candidates.candidates),
        }

    val_programs = predicted_programs(val_records)
    paired_programs = predicted_programs(paired_records)  # A-side x 20 then B-side x 20
    val_rows = [run(sample, program) for sample, program in zip(val_samples, val_programs)]
    paired_rows = []
    n_pairs = len(pairs)
    for i, (sample_a, sample_b) in enumerate(zip(a_samples, b_samples)):
        row_a = run(sample_a, paired_programs[i])
        row_b = run(sample_b, paired_programs[n_pairs + i])
        paired_rows.append(
            {
                "image_id": str(pairs[i]["image_id"]),
                "a": row_a,
                "b": row_b,
                "pair_correct": bool(row_a["correct"] and row_b["correct"]),
            }
        )
    paired_pass = sum(1 for row in paired_rows if row["pair_correct"])

    def accuracy(rows):
        return float(sum(1 for row in rows if row["correct"]) / max(len(rows), 1))

    failure_counts = Counter(row["failure"] for row in val_rows if row["failure"])
    program_accuracy = float(sum(1 for row in val_rows if row["program_correct"]) / max(len(val_rows), 1))
    gate_cfg = {"selected_target_accuracy_min": 0.85, "paired_min": 17}
    checks = {
        "selected_target_accuracy_ge": accuracy(val_rows) >= 0.85,
        "paired_ge": paired_pass >= 17,
    }
    gate = gate_report(checks, gate_cfg)

    report = {
        "_doc": (
            "Task 6J section 14. Stage J3: predicted program (trained ProgramHead) + oracle "
            "candidates (diagnostic-only). No fallback to the GT program anywhere."
        ),
        "task": "6J",
        "stage": "J3",
        "j2_checkpoint": j2["final"]["checkpoint"],
        "checkpoint_load": load_report,
        "metrics": {
            "selected_target_accuracy": accuracy(val_rows),
            "selected_target_correct": sum(1 for row in val_rows if row["correct"]),
            "count": len(val_rows),
            "program_accuracy": program_accuracy,
            "paired_selection_pass": paired_pass,
            "paired_total": len(paired_rows),
            "failure_counts": dict(failure_counts),
        },
        "val_rows": val_rows,
        "paired_rows": paired_rows,
        "gate": gate,
        "verdict": "J3_PASS" if gate["passed"] else "J3_FAIL",
        "no_gt_leakage": {
            "program_source": "trained ProgramHead on instruction text",
            "candidates": "oracle component maps (diagnostic-only)",
            "gt_used_for": ["scoring only"],
        },
        "seconds": round(time.time() - started, 2),
    }
    write_json(OUT, report)
    print(
        f"[task6j.j3] {report['verdict']}: selected-target accuracy "
        f"{report['metrics']['selected_target_accuracy']:.4f}, program accuracy "
        f"{program_accuracy:.4f}, paired {paired_pass}/{len(paired_rows)}, failures "
        f"{dict(failure_counts)}, gate {gate['passed']}",
        flush=True,
    )
    print(f"[task6j.j3] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    del runtime
    torch.cuda.empty_cache()
    return 0 if gate["passed"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
