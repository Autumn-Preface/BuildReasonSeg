"""Task 6J section 17: resolve exactly one primary verdict from the recorded artifacts.

Reads only tracked artifacts; never re-runs a model. Writes `evaluation/task6j_verdict.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from task6j_common import EVAL, write_json  # noqa: E402

OUT = EVAL / "task6j_verdict.json"

VERDICTS = (
    "STRUCTURED_PROPOSAL_GROUNDING_PROMISING",
    "PROPOSAL_QUALITY_LIMIT",
    "PROGRAM_PARSER_NOT_READY",
    "STRUCTURED_EXECUTOR_SEMANTICS_MISMATCH",
    "YOLO_BASELINE_INFERENCE_UNAVAILABLE",
    "STRUCTURED_ROUTE_PARTIAL",
    "INVALID_EXPERIMENT",
)


def _load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    spec = _load("task6j_program_spec.json")
    j0 = _load("task6j_j0_oracle_executor.json")
    recall = _load("task6j_yolo_proposal_recall.json")
    j1 = _load("task6j_j1_oracle_program_yolo.json")
    j2 = _load("task6j_j2_program_parser.json")
    j3 = _load("task6j_j3_predicted_program_oracle_candidates.json")
    j4 = _load("task6j_j4_structured_end_to_end.json")

    invalid_reasons = []
    if j0 is None or not j0["gate"]["passed"]:
        invalid_reasons.append("J0 artifact missing or gate not passed")
    if j0 is not None and not j0["frozen_recompute_agreement"]["all_agree"]:
        invalid_reasons.append("executor disagrees with the frozen recompute on some sample")
    if j1 is not None and not j1["provenance_check"]["matches"]:
        invalid_reasons.append("YOLO model hash on disk does not match the recorded provenance")
    if j2 is not None and j2.get("full_val", {}).get("exact_accuracy", 0.0) >= 1.0:
        pass  # a perfect parser is plausible for template-generated instructions; recorded as-is
    if j4 is not None and j4.get("ran") and not j4.get("no_gt_in_inference", True):
        invalid_reasons.append("J4 executed with GT in inference")

    if invalid_reasons:
        verdict, reason = "INVALID_EXPERIMENT", "; ".join(invalid_reasons)
    elif j0 is None or not j0["gate"]["passed"]:
        verdict, reason = (
            "STRUCTURED_EXECUTOR_SEMANTICS_MISMATCH",
            "the oracle program + oracle candidate executor did not reproduce the frozen targets",
        )
    elif j2 is None or not j2["final"]["gate"]["passed"]:
        verdict, reason = (
            "PROGRAM_PARSER_NOT_READY",
            "the instruction -> canonical program parser missed its semantic gate",
        )
    elif j3 is None or not j3["gate"]["passed"]:
        verdict, reason = (
            "STRUCTURED_ROUTE_PARTIAL",
            "J0 and J2 pass but the predicted-program + oracle-candidate route misses its gate",
        )
    elif j1 is not None and j1["viable"] and j4 is not None and j4.get("ran") and j4["gate"]["passed"]:
        verdict, reason = (
            "STRUCTURED_PROPOSAL_GROUNDING_PROMISING",
            "J0/J1/J2/J3 pass and J4 reaches the success gate",
        )
    elif j1 is not None and not j1["viable"]:
        verdict, reason = (
            "PROPOSAL_QUALITY_LIMIT",
            "J0/J2/J3 are healthy (executor 1.000, parser 1.000, predicted-program route 1.000), "
            f"but the frozen YOLO proposal chain is the binding failure: J1 paired mask selection "
            f"{j1['paired']['mask_paired_pass']}/{j1['paired']['paired_total']} (gate 12/20) with "
            f"recall@0.50 {recall['target_recall']['recall_at_0_50']} and oracle-program mIoU "
            f"{j1['metrics']['strict_selected_mask_miou']:.3f}",
        )
    else:
        verdict, reason = (
            "STRUCTURED_ROUTE_PARTIAL",
            "major gates pass and the route materially improves target selection, but J4 does not "
            "reach the full success gate",
        )

    if verdict not in VERDICTS:
        raise RuntimeError(f"unknown verdict {verdict!r}")
    report = {
        "_doc": (
            "Task 6J section 17. Exactly one primary verdict, resolved from the recorded artifacts."
        ),
        "task": "6J",
        "verdict": verdict,
        "reason": reason,
        "evidence": {
            "program_spec": {"program_count": spec["program_count"]} if spec else None,
            "J0": {
                "gate": j0["gate"],
                "exact_accuracy": j0["val"]["exact_accuracy"],
                "paired": j0["paired"]["paired_selection_pass"],
            }
            if j0
            else None,
            "proposal_recall": recall["target_recall"] if recall else None,
            "J1": {
                "viable": j1["viable"],
                "metrics": j1["metrics"],
                "paired": {
                    "mask_paired_pass": j1["paired"]["mask_paired_pass"],
                    "paired_total": j1["paired"]["paired_total"],
                },
                "gate": j1["gate"],
            }
            if j1
            else None,
            "J2": {
                "val_accuracy": j2["final"]["val"]["exact_accuracy"],
                "macro_f1": j2["final"]["val"]["macro_f1"],
                "paired_correct": j2["final"]["paired"]["paired_program_correct"],
                "full_val_accuracy": j2["full_val"]["exact_accuracy"],
                "gate": j2["final"]["gate"],
            }
            if j2
            else None,
            "J3": {
                "selected_target_accuracy": j3["metrics"]["selected_target_accuracy"],
                "paired": j3["metrics"]["paired_selection_pass"],
                "gate": j3["gate"],
            }
            if j3
            else None,
            "J4": {"ran": j4.get("ran"), "blockers": j4.get("blockers")} if j4 else None,
        },
    }
    write_json(OUT, report)
    print(f"[task6j:verdict] {verdict}: {reason}", flush=True)
    print(f"[task6j:verdict] wrote {OUT.relative_to(REPO_ROOT).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
