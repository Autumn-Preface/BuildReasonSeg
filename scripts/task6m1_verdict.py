"""Task 6M.1 Part I (section 17): exactly one final verdict.

Allowed: `STRUCTURED_DEMO_READY`, `PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE`,
`CONTINUATION_INTERRUPTED`, `SOURCE_CHECKPOINT_MISMATCH`, `SAFE_RESUME_UNAVAILABLE`,
`INVALID_EXPERIMENT`.

`STRUCTURED_DEMO_READY` requires normal completion/early stop AND all five validation J1 gates AND the
Task 6M.1 test gates (fixed120 mIoU >= 0.40, paired >= 14/20) AND the corrected CLI gate.

    python scripts/task6m1_verdict.py
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

from buildreasonseg_mvp.task6m_eval import CHECKPOINT_ROOT, EVAL, write_json  # noqa: E402

OUT = EVAL / "task6m1_verdict.json"
ALLOWED = (
    "STRUCTURED_DEMO_READY",
    "PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE",
    "CONTINUATION_INTERRUPTED",
    "SOURCE_CHECKPOINT_MISMATCH",
    "SAFE_RESUME_UNAVAILABLE",
    "INVALID_EXPERIMENT",
)
EXPECTED_HASHES = {
    "best.pt": "fd407db634a8a7ef83f09f8096686e73407095105f1b45c70c623d18dbf4ea44",
    "last.pt": "ea998bda37dd2dcb2cc7bb5e19f6d15b7a205a137c9cc2866c508a45513f860e",
}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    source = load("task6m1_source_checkpoint_audit.json")
    preflight = load("task6m1_resume_preflight.json")
    training = load("task6m1_training_summary.json")
    proposal = load("task6m1_proposal_val.json")
    frozen = load("task6m1_inference_config_frozen.json")
    j1 = load("task6m1_j1v2_val.json")
    j4 = load("task6m1_j4v2_test.json")
    demo = load("task6m1_demo_cli_audit.json")
    attribution = load("task6m1_error_attribution.json")

    source_ok = bool(source and source.get("verdict") == "SOURCE_CHECKPOINT_VERIFIED")
    resume_ok = bool(preflight and preflight.get("verdict") in ("SAFE_RESUME_READY", "SAFE_RESUME_COMPLETED"))
    normal_completion = bool(training and training.get("stop", {}).get("normal_completion"))
    j1_gate = bool(j1 and (j1.get("gate") or {}).get("passed"))
    j4_fixed = (j4 or {}).get("fixed120", {}).get("miou")
    j4_paired = (j4 or {}).get("paired20", {}).get("passed")
    demo_gate = bool(demo and (demo.get("gate") or {}).get("passed"))

    if not source_ok:
        verdict = "SOURCE_CHECKPOINT_MISMATCH"
    elif not resume_ok:
        verdict = "SAFE_RESUME_UNAVAILABLE"
    elif not normal_completion:
        verdict = "CONTINUATION_INTERRUPTED"
    elif j1 is None or frozen is None or proposal is None:
        verdict = "INVALID_EXPERIMENT"
    elif not j1_gate:
        verdict = "PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE"
    elif j4 is None:
        verdict = "INVALID_EXPERIMENT"
    elif j4_fixed is None or j4_fixed < 0.40 or (j4_paired or 0) < 14:
        verdict = "PROPOSAL_MODEL_NEEDS_IMPROVEMENT_AFTER_CONVERGENCE"
    elif not demo_gate:
        verdict = "INVALID_EXPERIMENT"
    else:
        verdict = "STRUCTURED_DEMO_READY"

    gates = {
        "source_checkpoint_verified": source_ok,
        "safe_resume_verified": resume_ok,
        "normal_training_completion": normal_completion,
        "validation_evaluation_ran": bool(proposal is not None and frozen is not None and j1 is not None),
        "validation_j1_gate_passed": j1_gate,
        "test_fixed120_miou_ge_0_40": bool(j4_fixed is not None and j4_fixed >= 0.40),
        "test_paired_pass_ge_14": bool((j4_paired or 0) >= 14),
        "demo_cli_gate_passed": demo_gate,
    }
    payload = {
        "_doc": (
            "Task 6M.1 section 17. Exactly one verdict from the allowed set with every gate measured. "
            "An interrupted continuation preserves state, does not claim convergence and does not run "
            "the downstream graded evaluation."
        ),
        "task": "6M.1",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "measured": {
            "training_stop_reason": (training or {}).get("stop", {}).get("stop_reason"),
            "start_epoch": ((training or {}).get("continuation") or {}).get("start_epoch"),
            "end_epoch": ((training or {}).get("continuation") or {}).get("end_epoch"),
            "combined_best_epoch": ((training or {}).get("combined_lineage") or {}).get("best_epoch"),
            "combined_best_mask_mAP50_95": ((training or {}).get("combined_lineage") or {}).get("best_mAP50_95"),
            "proposal_recall_at_0_50": ((proposal or {}).get("metrics") or {}).get("recall_at", {}).get("0.5"),
            "proposal_tiny_recall_at_0_50": ((proposal or {}).get("metrics") or {}).get("tiny_recall_at_0_50"),
            "frozen_conf": (frozen or {}).get("conf"),
            "frozen_max_det": (frozen or {}).get("max_det"),
            "j1_fixed120_miou": ((j1 or {}).get("fixed120") or {}).get("miou"),
            "j1_paired_pass": ((j1 or {}).get("paired20") or {}).get("passed"),
            "j1_abstentions": ((j1 or {}).get("fixed120") or {}).get("abstentions"),
            "j4_fixed120_miou": j4_fixed,
            "j4_paired_pass": j4_paired,
            "demo_supported_runs": ((demo or {}).get("supported_summary") or {}).get("runs"),
            "demo_ood_rejected": ((demo or {}).get("out_of_domain_summary") or {}).get(
                "all_rejected_with_exit_4"
            ),
        },
        "attribution_diagnosis": (attribution or {}).get("diagnosis"),
        "task6m_evidence_preserved": {
            "recorded_hashes": EXPECTED_HASHES,
            "source_audit": (source or {}).get("verdict"),
            "snapshot_dir": str(CHECKPOINT_ROOT.parent / "task6m1" / "source_epoch18_snapshot"),
        },
        "keep_flags": {
            "keep_task6m_artifacts_frozen": True,
            "keep_frozen_packs_and_parser": True,
            "no_architecture_change": True,
            "no_imgsz_change": True,
            "no_new_dataset": True,
            "no_gui": True,
        },
    }
    write_json(OUT, payload)
    print(f"[6m1.verdict] {verdict}; failed gates {payload['failed_gates']}", flush=True)
    return 0 if verdict in ALLOWED else 2


if __name__ == "__main__":
    raise SystemExit(main())
