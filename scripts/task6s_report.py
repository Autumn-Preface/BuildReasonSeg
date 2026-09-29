"""Task 6S Parts J-M — architecture-freeze checkpoint summary and the single verdict.

Section 25 writes `evaluation/task6s_hardening_checkpoint.json` (primary chain, proven positive modules,
negative/non-primary evidence, known technical debt, the dominant bottleneck and
`next_research_decision: WAIT_FOR_CHATGPT`).

Section 24 writes `evaluation/task6s_verdict.json` using the fixed priority order with the section 22
integration gates and the section 23 regression check. DSH does not prescribe a repair.

    python scripts/task6s_report.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT_CHECKPOINT = EVAL / "task6s_hardening_checkpoint.json"
OUT_VERDICT = EVAL / "task6s_verdict.json"
BASE_COMMIT = "a3d59da8c8698f793357887762dbc95a08a254aa"

ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "PARSER_CHECKPOINT_UNAVAILABLE",
    "END_TO_END_INTEGRATION_REGRESSION",
    "DIRECTIONAL_PARSER_HARDENING_REQUIRED",
    "DIRECTIONAL_CHAIN_BELOW_GATE",
    "DIRECTIONAL_END_TO_END_CHAIN_READY_FOR_HARDENING",
)

GATES = {
    "parser_accuracy_min": 0.95,
    "strict_all_240_miou_min": 0.28,
    "answered_only_miou_min": 0.2893521927137638,
    "paired_min": 9,
    "own_cross_margin_min": 0.22,
    "paraphrase_min": 22,
}
TASK6Q_ANSWERED_MIOU = 0.3045812554881724
REGRESSION_TOLERANCE = 1e-6

#: Task 6S sections 2-3 — the recorded Task 6R evidence and the gate erratum.
TASK6R_EVIDENCE = {
    "r0_b3_miou": 0.4299680351479113,
    "r1_b3_grcl_miou": 0.41185668634454997,
    "r0_relation_accuracy": 0.95,
    "r1_relation_accuracy": 0.9541666666666667,
    "paired_r0": 14,
    "paired_r1": 12,
    "proposal_reference_transfer_miou": 0.3045812554881724,
    "proposal_reference_transfer_r1_miou": 0.283782,
    "paired_transfer_r0": 10,
    "paired_transfer_r1": 2,
    "gate_erratum": (
        "the predeclared Task 6R criterion `R1 relation_accuracy >= R0 relation_accuracy + 0.08` was "
        "ill-posed: the frozen baseline measured R0 = 0.95, leaving only 0.05 of headroom below the "
        "metric ceiling 1.0. This is recorded, not mutated; Task 6R artifacts and verdict are unchanged."
    ),
}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changed() -> str:
    return subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", "evaluation/task6m_", "evaluation/task6n_",
         "evaluation/task6o_", "evaluation/task6p_", "evaluation/task6q_", "evaluation/task6r_",
         "buildreasonseg_mvp/geometric_relation_field.py",
         "buildreasonseg_mvp/geometric_relation_field_v02.py",
         "buildreasonseg_mvp/task6q_reference_resolver.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    assets = load("task6s_frozen_asset_audit.json")
    parser_audit = load("task6s_parser_val.json")
    e2e = load("task6s_end_to_end_val.json")
    paired = load("task6s_end_to_end_paired_val.json")
    cli = load("task6s_cli_prompt_audit.json")
    attribution = load("task6s_failure_attribution.json")
    required = {"task6s_frozen_asset_audit.json": assets, "task6s_parser_val.json": parser_audit,
                "task6s_end_to_end_val.json": e2e, "task6s_end_to_end_paired_val.json": paired,
                "task6s_cli_prompt_audit.json": cli, "task6s_failure_attribution.json": attribution}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT_VERDICT, {"_doc": "Task 6S section 24.", "task": "6S",
                                 "verdict": "INVALID_EXPERIMENT",
                                 "reason": f"missing artifacts: {missing}"})
        print(f"[6s.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changed = frozen_changed()
    protocol = {
        "frozen_artifacts_unchanged": changed == "",
        "changed_paths": changed,
        "test_split_used": any(payload.get("test_split_used", False) for payload in required.values()),
        "ground_truth_required_by_cli": bool(cli.get("ground_truth_required_by_cli", False)),
        "grcl_in_primary_chain": bool(e2e.get("grcl_used", False)),
        "oracle_reference_in_inference": bool(e2e.get("oracle_reference_used", False)),
    }
    protocol["clean"] = bool(protocol["frozen_artifacts_unchanged"]
                             and not protocol["test_split_used"]
                             and not protocol["ground_truth_required_by_cli"]
                             and not protocol["grcl_in_primary_chain"]
                             and not protocol["oracle_reference_in_inference"])

    parser_accuracy = parser_audit["gate"]["measured"]
    paraphrase = cli["supported_gate"]["measured"]
    answered_miou = e2e["answered_only"]["miou"]
    strict_miou = e2e["strict_all_240"]["miou"]
    paired_pass = paired["passed"]
    margin = paired["own_cross_margin"]
    regression_delta = answered_miou - TASK6Q_ANSWERED_MIOU
    regression = {
        "expected_task6q_answered_miou": TASK6Q_ANSWERED_MIOU,
        "measured_answered_miou": answered_miou,
        "absolute_delta": abs(regression_delta),
        "tolerance": REGRESSION_TOLERANCE,
        "parser_accuracy_is_1_0": parser_accuracy == 1.0,
        "answered_records": e2e["answered_only"]["records"],
        "task6q_answered_records": 234,
        "same_answered_set": e2e["answered_only"]["records"] == 234,
        "material_difference": abs(regression_delta) > REGRESSION_TOLERANCE,
    }

    gates = {
        "1_parser_accuracy": {"required": GATES["parser_accuracy_min"], "measured": parser_accuracy,
                              "passed": parser_accuracy >= GATES["parser_accuracy_min"]},
        "2_strict_all_240_miou": {"required": GATES["strict_all_240_miou_min"],
                                  "measured": strict_miou,
                                  "passed": strict_miou >= GATES["strict_all_240_miou_min"]},
        "3_answered_only_miou": {"required": GATES["answered_only_miou_min"],
                                 "measured": answered_miou,
                                 "passed": answered_miou >= GATES["answered_only_miou_min"]},
        "4_paired": {"required": GATES["paired_min"], "measured": paired_pass,
                     "passed": paired_pass >= GATES["paired_min"]},
        "5_own_cross_margin": {"required": GATES["own_cross_margin_min"], "measured": margin,
                               "passed": margin >= GATES["own_cross_margin_min"]},
        "6_paraphrase_pack": {"required": GATES["paraphrase_min"], "measured": paraphrase,
                              "passed": paraphrase >= GATES["paraphrase_min"]},
        "7_ood_exit_4": {"measured": cli["ood_all_exit_4"], "passed": bool(cli["ood_all_exit_4"])},
        "8_out_of_scope_exit_5": {"measured": cli["out_of_scope_all_exit_5"],
                                  "passed": bool(cli["out_of_scope_all_exit_5"])},
        "9_no_gt_dependency": {"measured": not cli["ground_truth_required_by_cli"],
                               "passed": not cli["ground_truth_required_by_cli"]},
        "10_no_test_split": {"measured": not protocol["test_split_used"],
                            "passed": not protocol["test_split_used"]},
    }
    all_gates = all(entry["passed"] for entry in gates.values())
    parser_gate_ok = gates["1_parser_accuracy"]["passed"] and gates["6_paraphrase_pack"]["passed"]

    if not protocol["clean"]:
        verdict = "INVALID_EXPERIMENT"
        reason = "protocol violation: leakage, test access, frozen mutation or GT inference dependency"
    elif not assets["program_head"]["matches_expected"]:
        verdict = "PARSER_CHECKPOINT_UNAVAILABLE"
        reason = "the frozen ProgramHead SHA256 does not match the Task 6S expected hash"
    elif regression["material_difference"]:
        verdict = "END_TO_END_INTEGRATION_REGRESSION"
        reason = (f"answered-only mIoU {answered_miou:.9f} differs from the frozen Task 6Q "
                  f"{TASK6Q_ANSWERED_MIOU:.9f} by {abs(regression_delta):.3e}")
    elif not parser_gate_ok:
        failing_parser = []
        if not gates["1_parser_accuracy"]["passed"]:
            failing_parser.append(f"canonical parser accuracy {parser_accuracy:.4f} < 0.95")
        if not gates["6_paraphrase_pack"]["passed"]:
            failing_parser.append(f"paraphrase accuracy {paraphrase}/24 < 22")
        verdict = "DIRECTIONAL_PARSER_HARDENING_REQUIRED"
        reason = "; ".join(failing_parser)
    elif not all_gates:
        failing = [name for name, entry in gates.items() if not entry["passed"]]
        verdict = "DIRECTIONAL_CHAIN_BELOW_GATE"
        reason = (f"failing integration gates: {failing} "
                  f"(paraphrase {paraphrase}/24, out-of-scope exit-5 "
                  f"{gates['8_out_of_scope_exit_5']['measured']})")
    else:
        verdict = "DIRECTIONAL_END_TO_END_CHAIN_READY_FOR_HARDENING"
        reason = "all section 22 integration gates pass"

    checkpoint = {
        "_doc": (
            "Task 6S section 25. Architecture-freeze checkpoint: the integrated primary chain, the "
            "modules with positive evidence, the negative/non-primary evidence, the known technical "
            "debt, the dominant bottleneck computed by the fixed rule and the next research decision "
            "(always WAIT_FOR_CHATGPT). DSH does not prescribe a repair."
        ),
        "task": "6S",
        "primary_chain": {
            "parser": "frozen Qwen3-VL-2B ProgramHead",
            "reference": "frozen Task6Q proposal resolver",
            "relation_field": "v0.2",
            "target_decoder": "frozen Task6O B3",
            "grcl_primary": False,
        },
        "proven_positive_modules": [
            "GeometricRelationField v0.2 + dense visual target decoder",
        ],
        "negative_or_non_primary_evidence": [
            "Task6P dense ReferenceMaskHead",
            "Task6R GRCL v0.1",
        ],
        "known_technical_debt": [
            "reference proposal coverage / extreme ranking, especially smallest",
            "tiny buildings",
            "directional-only current scope",
            "no nearest",
            "no L3 multi-hop",
            "no cross-dataset generalization",
        ],
        "dominant_bottleneck": attribution["dominant_bottleneck"],
        "dominant_bottleneck_detail": attribution["aggregation"],
        "next_research_decision": "WAIT_FOR_CHATGPT",
        "frozen_asset_hashes": {
            "program_head": assets["program_head"]["resolved_sha256"],
            "proposal_resolver": assets["proposal_resolver"]["sha256"],
            "target_decoder": assets["target_decoder"]["sha256"],
            "relation_field": assets["relation_field"]["sha256"],
        },
        "task6r_negative_evidence": TASK6R_EVIDENCE,
        "evidence_paths": {
            "frozen_asset_audit": "evaluation/task6s_frozen_asset_audit.json",
            "parser_audit": "evaluation/task6s_parser_val.json",
            "end_to_end": "evaluation/task6s_end_to_end_val.json",
            "paired": "evaluation/task6s_end_to_end_paired_val.json",
            "cli_prompt_audit": "evaluation/task6s_cli_prompt_audit.json",
            "failure_attribution": "evaluation/task6s_failure_attribution.json",
        },
    }
    verdict_payload = {
        "_doc": (
            "Task 6S section 24. Exactly one verdict by the fixed priority order, with the section 22 "
            "integration gates, the section 23 regression check and the protocol audit. "
            "READY_FOR_HARDENING does not mean the final model or paper is ready."
        ),
        "task": "6S",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "protocol": protocol,
        "gates": gates,
        "all_gates_passed": all_gates,
        "regression_check": regression,
        "gate_constants": GATES,
        "headline_metrics": {
            "parser_accuracy": parser_accuracy,
            "paraphrase_pack": f"{paraphrase}/24",
            "strict_all_240_miou": strict_miou,
            "strict_all_240_dice": e2e["strict_all_240"]["dice"],
            "answered_only_miou": answered_miou,
            "answered_only_dice": e2e["answered_only"]["dice"],
            "answered_records": e2e["answered_only"]["records"],
            "abstention_rate": e2e["abstention_rate"],
            "paired": f"{paired_pass}/20",
            "own_cross_margin": margin,
            "dominant_bottleneck": attribution["dominant_bottleneck"],
        },
        "failure_attribution_counts": attribution["counts"],
        "failure_attribution_aggregation": attribution["aggregation"],
        "hardening_checkpoint": "evaluation/task6s_hardening_checkpoint.json",
        "test_split_used": False,
        "no_module_or_threshold_changed_by_dsh": True,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT_CHECKPOINT, checkpoint)
    write_json(OUT_VERDICT, verdict_payload)
    print(f"[6s.report] parser {parser_accuracy:.4f} | paraphrase {paraphrase}/24 | strict "
          f"{strict_miou:.6f} | answered {answered_miou:.6f} | paired {paired_pass}/20 margin "
          f"{margin:+.6f} | dominant {attribution['dominant_bottleneck']} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
