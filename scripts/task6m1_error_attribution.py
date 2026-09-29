"""Task 6M.1 Part H (section 16): compare the Task 6M epoch-18 state with the Task 6M.1 continuation.

Only meaningful when the continuation reached normal completion and the validation evaluation ran
(section 10). Reports every required delta and one of the three allowed diagnoses:

* `CONVERGENCE_PASSES_GATE`
* `CONVERGENCE_HELPED_BUT_GATE_STILL_FAILS`
* `CONVERGENCE_DID_NOT_HELP_MATERIALLY`

    python scripts/task6m1_error_attribution.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6m1_error_attribution.json"
HELPFUL_RECALL_DELTA = 0.05
HELPFUL_MIOU_DELTA = 0.05


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def delta(new, old):
    if new is None or old is None:
        return None
    return round(float(new) - float(old), 6)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    training = load("task6m1_training_summary.json")
    proposal_6m = load("task6m_proposal_val.json")
    proposal_6m1 = load("task6m1_proposal_val.json")
    j1_6m = load("task6m_j1v2_val.json")
    j1_6m1 = load("task6m1_j1v2_val.json")

    if training is None or proposal_6m1 is None or j1_6m1 is None:
        write_json(
            OUT,
            {
                "_doc": "Task 6M.1 section 16 failure attribution.",
                "task": "6M.1",
                "status": "NOT_APPLICABLE",
                "reason": (
                    "the continuation did not reach normal completion (or the validation evaluation "
                    "did not run), so per section 10 there is no converged proposal result to compare"
                ),
                "training_stop": (training or {}).get("stop"),
            },
        )
        print("[6m1.attrib] NOT_APPLICABLE (no normal completion / no validation evaluation)", flush=True)
        return 0

    metrics_6m = (proposal_6m or {}).get("metrics", {})
    metrics_6m1 = proposal_6m1.get("metrics", {})
    size_6m = metrics_6m.get("size_breakdown", {})
    size_6m1 = metrics_6m1.get("size_breakdown", {})

    def size_recall(source, key):
        return (source.get(key) or {}).get("recall_at_0_50")

    fix_6m = (j1_6m or {}).get("fixed120", {})
    fix_6m1 = j1_6m1.get("fixed120", {})
    pair_6m = (j1_6m or {}).get("paired20", {})
    pair_6m1 = j1_6m1.get("paired20", {})

    deltas = {
        "overall_recall_at_0_50": delta(metrics_6m1.get("recall_at", {}).get("0.5"),
                                        metrics_6m.get("recall_at", {}).get("0.5")),
        "recall_at_0_25": delta(metrics_6m1.get("recall_at", {}).get("0.25"),
                                metrics_6m.get("recall_at", {}).get("0.25")),
        "recall_at_0_75": delta(metrics_6m1.get("recall_at", {}).get("0.75"),
                                metrics_6m.get("recall_at", {}).get("0.75")),
        "tiny_recall_at_0_50": delta(metrics_6m1.get("tiny_recall_at_0_50"),
                                     metrics_6m.get("tiny_recall_at_0_50")),
        "border_recall_at_0_50": delta(metrics_6m1.get("border_recall_at_0_50"),
                                       metrics_6m.get("border_recall_at_0_50")),
        "dense_recall_at_0_50": delta(metrics_6m1.get("dense_recall_at_0_50"),
                                      metrics_6m.get("dense_recall_at_0_50")),
        "small_recall_at_0_50": delta(size_recall(size_6m1, "small(50-200)"),
                                      size_recall(size_6m, "small(50-200)")),
        "medium_recall_at_0_50": delta(size_recall(size_6m1, "medium(200-1000)"),
                                       size_recall(size_6m, "medium(200-1000)")),
        "large_recall_at_0_50": delta(size_recall(size_6m1, "large(>=1000)"),
                                      size_recall(size_6m, "large(>=1000)")),
        "mask_map50": delta((proposal_6m1.get("validator_metrics") or {}).get("mask_map50"),
                            (proposal_6m.get("validator_metrics") or {}).get("mask_map50")),
        "mask_map50_95": delta((proposal_6m1.get("validator_metrics") or {}).get("mask_map50_95"),
                               (proposal_6m.get("validator_metrics") or {}).get("mask_map50_95")),
        "empty_tile_false_proposal_rate": delta(metrics_6m1.get("empty_tile_false_proposal_rate"),
                                                metrics_6m.get("empty_tile_false_proposal_rate")),
        "j1_fixed120_miou": delta(fix_6m1.get("miou"), fix_6m.get("miou")),
        "j1_paired_pass": delta(pair_6m1.get("passed"), pair_6m.get("passed")),
        "j1_abstentions": delta(fix_6m1.get("abstentions"), fix_6m.get("abstentions")),
        "j1_full_val_miou": delta(((j1_6m1.get("full_val") or {}).get("miou")),
                                  ((j1_6m or {}).get("full_val") or {}).get("miou")),
    }

    gate_passed = bool((j1_6m1.get("gate") or {}).get("passed"))
    recall_delta = deltas["overall_recall_at_0_50"] or 0.0
    miou_delta = deltas["j1_fixed120_miou"] or 0.0
    if gate_passed:
        diagnosis = "CONVERGENCE_PASSES_GATE"
    elif recall_delta >= HELPFUL_RECALL_DELTA or miou_delta >= HELPFUL_MIOU_DELTA:
        diagnosis = "CONVERGENCE_HELPED_BUT_GATE_STILL_FAILS"
    else:
        diagnosis = "CONVERGENCE_DID_NOT_HELP_MATERIALLY"

    payload = {
        "_doc": (
            "Task 6M.1 section 16. Task 6M (epoch 18) vs Task 6M.1 (converged) comparison on the "
            "validation split: every required recall / structured-IoU / abstention delta."
        ),
        "task": "6M.1",
        "status": "COMPARED",
        "epochs": {
            "task6m_last_epoch": ((training.get("source_stage") or {}).get("last_epoch")),
            "task6m1_last_epoch": ((training.get("continuation_stage") or {}).get("last_epoch")),
            "combined_best_epoch": ((training.get("combined_lineage") or {}).get("best_epoch")),
        },
        "task6m_baseline": {
            "recall_at": metrics_6m.get("recall_at"),
            "tiny_recall_at_0_50": metrics_6m.get("tiny_recall_at_0_50"),
            "size_breakdown": {key: value.get("recall_at_0_50") for key, value in size_6m.items()},
            "mask_map50": (proposal_6m or {}).get("validator_metrics", {}).get("mask_map50"),
            "mask_map50_95": (proposal_6m or {}).get("validator_metrics", {}).get("mask_map50_95"),
            "j1_fixed120_miou": fix_6m.get("miou"),
            "j1_paired_pass": pair_6m.get("passed"),
            "j1_abstentions": fix_6m.get("abstentions"),
        },
        "task6m1_result": {
            "recall_at": metrics_6m1.get("recall_at"),
            "tiny_recall_at_0_50": metrics_6m1.get("tiny_recall_at_0_50"),
            "size_breakdown": {key: value.get("recall_at_0_50") for key, value in size_6m1.items()},
            "mask_map50": proposal_6m1.get("validator_metrics", {}).get("mask_map50"),
            "mask_map50_95": proposal_6m1.get("validator_metrics", {}).get("mask_map50_95"),
            "j1_fixed120_miou": fix_6m1.get("miou"),
            "j1_paired_pass": pair_6m1.get("passed"),
            "j1_abstentions": fix_6m1.get("abstentions"),
            "j1_gate": (j1_6m1.get("gate") or {}).get("checks"),
        },
        "deltas": deltas,
        "diagnosis": diagnosis,
        "diagnosis_rule": {
            "helpful_recall_delta": HELPFUL_RECALL_DELTA,
            "helpful_miou_delta": HELPFUL_MIOU_DELTA,
            "note": (
                "convergence is called helpful when overall recall@0.50 or J1 fixed120 mIoU improved by "
                "at least the declared delta; the architecture was NOT changed and no new strategy is "
                "chosen in this task"
            ),
        },
    }
    write_json(OUT, payload)
    print(f"[6m1.attrib] diagnosis {diagnosis}; recall@0.50 delta {recall_delta:+.4f}; "
          f"fixed120 mIoU delta {miou_delta:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
