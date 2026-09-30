"""Task 7A Parts J-K — predeclared gates, sample retention and the single verdict.

Section 20 canonical predicted-reference retention: strict mIoU >= 0.22, answered-only mIoU >= 0.24,
retention = A1 strict mIoU / frozen Task 6Z Z-B3 mIoU >= 0.68, PairedVal >= 10/20, own-cross margin >= 0.15,
reference abstention rate <= 0.10.

Section 21 natural-language integration: canonical-query parser accuracy >= 0.98, strict end-to-end
mIoU >= 0.21, answered-only >= 0.23, PairedVal >= 9/20, own-cross margin >= 0.14.

Section 22 `l3_paraphrase_ready`: fixed-24 accuracy >= 22/24 **and** all eight exact compact prompts correct.

Section 23 then selects exactly one verdict in the fixed priority order.

Writes `evaluation/task7a_verdict.json`. DSH reports measurements only and proposes no repair.

    python scripts/task7a_report.py
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

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.task7a_l3_pipeline import (  # noqa: E402
    SUPPORTED_L3_PROGRAMS,
    default_parser_checkpoint,
    default_target_checkpoint,
    pipeline_report,
    sha256_file,
)

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7a_verdict.json"
BASE_COMMIT = "7006d7d416382f775112541a448425da1535cfd1"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "L3_CHECKPOINT_UNAVAILABLE",
    "PARSER_CHECKPOINT_UNAVAILABLE",
    "TASK6Z_REPRODUCTION_FAIL",
    "L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE",
    "L3_LANGUAGE_HARDENING_REQUIRED",
    "L3_END_TO_END_DEVELOPMENT_CHAIN_READY",
)
FROZEN_PATHS = (
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/nearest_boundary_field.py",
    "buildreasonseg_mvp/task6z_field_composition.py",
    "buildreasonseg_mvp/task6z_l3_decoder.py",
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/program_parser.py",
    "buildreasonseg_mvp/task6s_directional_pipeline.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "configs/spatial_relations_v1.yaml",
    "evaluation/task6z_", "evaluation/task6y_",
)
GATES_REFERENCE = {"strict_miou_min": 0.22, "answered_miou_min": 0.24, "retention_min": 0.68,
                   "paired_min": 10, "margin_min": 0.15, "abstention_max": 0.10}
GATES_LANGUAGE = {"parser_accuracy_min": 0.98, "strict_miou_min": 0.21, "answered_miou_min": 0.23,
                  "paired_min": 9, "margin_min": 0.14}
GATES_PARAPHRASE = {"accuracy_min": 22, "total": 24}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def paraphrase_modes(result: dict) -> dict:
    """Failure-mode breakdown: dropped terminal `to_nearest` vs other confusions."""

    modes = {"terminal_to_nearest_dropped": 0, "direction_confused": 0, "family_confused": 0,
             "other": 0}
    dropped = []
    for row in result["rows"]:
        if row["correct"]:
            continue
        expected, parsed = row["program"], row["parsed_program"] or ""
        expected_direction = expected.split("_to_")[1].replace("_to_nearest", "")
        if parsed == f"largest_to_{expected_direction}":
            modes["terminal_to_nearest_dropped"] += 1
            dropped.append(row["id"])
        elif expected_direction not in parsed:
            modes["direction_confused"] += 1
        elif not parsed.startswith("largest"):
            modes["family_confused"] += 1
        else:
            modes["other"] += 1
    modes["terminal_dropped_prompt_ids"] = dropped
    return modes


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    a0 = load("task7a_oracle_reproduction.json")
    ref_quality = load("task7a_predicted_reference_quality.json")
    a1_val = load("task7a_canonical_predicted_reference_val.json")
    a1_paired = load("task7a_canonical_predicted_reference_paired.json")
    parser_audit = load("task7a_parser_l3_val.json")
    a2_val = load("task7a_natural_language_val.json")
    a2_paired = load("task7a_natural_language_paired.json")
    paraphrase_pack = load("task7a_l3_paraphrase_pack.json")
    paraphrase_result = load("task7a_l3_paraphrase_result.json")
    attribution = load("task7a_failure_attribution.json")
    cli = load("task7a_cli_audit.json")
    task6z = load("task6z_training.json")

    required = {"task7a_oracle_reproduction.json": a0,
                "task7a_predicted_reference_quality.json": ref_quality,
                "task7a_canonical_predicted_reference_val.json": a1_val,
                "task7a_canonical_predicted_reference_paired.json": a1_paired,
                "task7a_parser_l3_val.json": parser_audit,
                "task7a_natural_language_val.json": a2_val,
                "task7a_natural_language_paired.json": a2_paired,
                "task7a_l3_paraphrase_pack.json": paraphrase_pack,
                "task7a_l3_paraphrase_result.json": paraphrase_result,
                "task7a_failure_attribution.json": attribution,
                "task7a_cli_audit.json": cli}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 7A section 23.", "task": "7A",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[7a.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    expected_target_sha = task6z["results"]["Z-B3"]["checkpoint"]["sha256"]
    target_sha = sha256_file(default_target_checkpoint())
    parser_sha = sha256_file(default_parser_checkpoint())
    expected_parser_sha = json.loads((EVAL / "task6t_training_summary.json")
                                     .read_text(encoding="utf-8"))["checkpoint"]["sha256"]
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "target_checkpoint_sha256": target_sha, "expected_target_sha256": expected_target_sha,
        "target_checkpoint_matches": target_sha == expected_target_sha,
        "parser_checkpoint_sha256": parser_sha, "expected_parser_sha256": expected_parser_sha,
        "parser_checkpoint_matches": parser_sha == expected_parser_sha,
        "supported_programs": list(SUPPORTED_L3_PROGRAMS),
        "program_count": len(SUPPORTED_L3_PROGRAMS),
        "training_performed": False,
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (a0, ref_quality, a1_val, a1_paired, parser_audit,
                                               a2_val, a2_paired, paraphrase_result, attribution, cli)
                               if isinstance(payload, dict)),
        "ground_truth_used_in_inference": False,
        "ranker_or_quality_or_refinement_used": False,
        "attention_or_transformer_or_gnn": False,
        "grcl": False,
        "thresholds_tuned_after_results": False,
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and not protocol["test_split_used"]
                             and protocol["target_checkpoint_matches"]
                             and protocol["parser_checkpoint_matches"]
                             and not protocol["training_performed"])

    reference_gate = {
        "strict_miou": {"required": GATES_REFERENCE["strict_miou_min"],
                        "measured": a1_val["strict"]["miou"],
                        "passed": a1_val["strict"]["miou"] >= GATES_REFERENCE["strict_miou_min"]},
        "answered_only_miou": {"required": GATES_REFERENCE["answered_miou_min"],
                               "measured": a1_val["strict"]["answered_only_miou"],
                               "passed": a1_val["strict"]["answered_only_miou"]
                               >= GATES_REFERENCE["answered_miou_min"]},
        "retention": {"required": GATES_REFERENCE["retention_min"],
                      "measured": a1_val["retention"],
                      "passed": (a1_val["retention"] or 0.0) >= GATES_REFERENCE["retention_min"]},
        "paired": {"required": f">= {GATES_REFERENCE['paired_min']}/20",
                   "measured": f"{a1_paired['passed']}/{a1_paired['pairs']}",
                   "passed": a1_paired["passed"] >= GATES_REFERENCE["paired_min"]},
        "margin": {"required": GATES_REFERENCE["margin_min"],
                   "measured": a1_paired["own_cross_margin"],
                   "passed": a1_paired["own_cross_margin"] >= GATES_REFERENCE["margin_min"]},
        "abstention_rate": {"required": f"<= {GATES_REFERENCE['abstention_max']}",
                            "measured": ref_quality["reference"]["reference_abstention_rate"],
                            "passed": ref_quality["reference"]["reference_abstention_rate"]
                            <= GATES_REFERENCE["abstention_max"]},
    }
    reference_gate_passed = all(entry["passed"] for entry in reference_gate.values())

    language_gate = {
        "parser_accuracy": {"required": GATES_LANGUAGE["parser_accuracy_min"],
                            "measured": parser_audit["exact_accuracy"],
                            "passed": parser_audit["exact_accuracy"]
                            >= GATES_LANGUAGE["parser_accuracy_min"]},
        "strict_miou": {"required": GATES_LANGUAGE["strict_miou_min"],
                        "measured": a2_val["strict_all_miou"],
                        "passed": a2_val["strict_all_miou"] >= GATES_LANGUAGE["strict_miou_min"]},
        "answered_only_miou": {"required": GATES_LANGUAGE["answered_miou_min"],
                               "measured": a2_val["answered_only_miou"],
                               "passed": (a2_val["answered_only_miou"] or 0.0)
                               >= GATES_LANGUAGE["answered_miou_min"]},
        "paired": {"required": f">= {GATES_LANGUAGE['paired_min']}/20",
                   "measured": f"{a2_paired['passed']}/{a2_paired['pairs']}",
                   "passed": a2_paired["passed"] >= GATES_LANGUAGE["paired_min"]},
        "margin": {"required": GATES_LANGUAGE["margin_min"],
                   "measured": a2_paired["own_cross_margin"],
                   "passed": a2_paired["own_cross_margin"] >= GATES_LANGUAGE["margin_min"]},
    }
    language_gate_passed = all(entry["passed"] for entry in language_gate.values())

    paraphrase_ready = bool(paraphrase_result["total_correct"] >= GATES_PARAPHRASE["accuracy_min"]
                            and paraphrase_result["required_compact"]["all_correct"])

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, test use or training)"
    elif not protocol["target_checkpoint_matches"]:
        verdict, reason = "L3_CHECKPOINT_UNAVAILABLE", "the frozen Z-B3 checkpoint hash does not match"
    elif not protocol["parser_checkpoint_matches"]:
        verdict, reason = "PARSER_CHECKPOINT_UNAVAILABLE", "the hardened ProgramHead hash does not match"
    elif not a0["reproduction_passed"]:
        verdict, reason = ("TASK6Z_REPRODUCTION_FAIL",
                           f"oracle reproduction deltas {a0['deltas']} exceed {a0['tolerance']}")
    elif not reference_gate_passed:
        verdict = "L3_PREDICTED_REFERENCE_CHAIN_BELOW_GATE"
        reason = ("the canonical predicted-reference gate fails on "
                  + str([name for name, entry in reference_gate.items() if not entry["passed"]]))
    elif not language_gate_passed or not paraphrase_ready:
        verdict = "L3_LANGUAGE_HARDENING_REQUIRED"
        reason = ("the canonical predicted-reference gate passes but the natural-language gate or the "
                  f"paraphrase flag fails (language gate: "
                  f"{[name for name, entry in language_gate.items() if not entry['passed']]}, "
                  f"paraphrase_ready={paraphrase_ready})")
    else:
        verdict = "L3_END_TO_END_DEVELOPMENT_CHAIN_READY"
        reason = "canonical predicted-reference gate, natural-language gate and paraphrase flag all pass"

    payload = {
        "_doc": (
            "Task 7A sections 20-23. Integration/attribution audit with no training: A0 oracle "
            "reproduction, A1 canonical predicted-reference chain, A2 natural-language integration, the "
            "fixed 24-prompt paraphrase audit, the CMD entry-point audit and exclusive failure "
            "attribution. The verdict follows the fixed section-23 priority order; READY would mean "
            "development-chain ready only."
        ),
        "task": "7A", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "protocol": protocol,
        "pipeline": pipeline_report(),
        "a0_oracle_reproduction": a0,
        "reference_quality": ref_quality["reference"],
        "reference_gate": reference_gate, "reference_gate_passed": reference_gate_passed,
        "language_gate": language_gate, "language_gate_passed": language_gate_passed,
        "paraphrase": {
            "total_correct": paraphrase_result["total_correct"],
            "total": paraphrase_result["total"],
            "accuracy": paraphrase_result["exact_accuracy"],
            "per_program": paraphrase_result["per_program"],
            "per_language": paraphrase_result["per_language"],
            "required_compact": paraphrase_result["required_compact"],
            "failure_modes": paraphrase_modes(paraphrase_result),
            "l3_paraphrase_ready": paraphrase_ready,
        },
        "a1_canonical_predicted_reference": {
            "strict_miou": a1_val["strict"]["miou"],
            "strict_dice": a1_val["strict"]["dice"],
            "answered_only_miou": a1_val["strict"]["answered_only_miou"],
            "precision_at_0_5": a1_val["strict"]["precision_at_0_5"],
            "abstentions": a1_val["abstentions"],
            "retention": a1_val["retention"],
            "oracle_task6z_miou": a1_val["oracle_task6z_miou"],
            "reference_ok_subset_miou": a1_val["reference_ok_subset"].get("miou"),
            "reference_fail_subset_miou": a1_val["reference_fail_subset"].get("miou"),
            "per_direction": {direction: a1_val["by_direction"][direction]["miou"]
                              for direction in ("above", "below", "left", "right")},
            "paired": a1_paired["passed"], "paired_margin": a1_paired["own_cross_margin"],
            "reference_abstention_pairs": a1_paired["reference_abstention_pairs"],
        },
        "a2_natural_language": {
            "parser_accuracy": parser_audit["exact_accuracy"],
            "parser_correct": parser_audit["exact_correct"],
            "per_class_recall": parser_audit["per_class_recall"],
            "out_of_scope_predictions": parser_audit["predicted_out_of_scope_count"],
            "strict_miou": a2_val["strict_all_miou"],
            "strict_dice": a2_val["strict_all_dice"],
            "answered_only_miou": a2_val["answered_only_miou"],
            "abstentions": a2_val["abstentions"],
            "paired": a2_paired["passed"], "paired_margin": a2_paired["own_cross_margin"],
            "parser_correct_members": a2_paired["parser_correct_members"],
            "per_direction": {direction: a2_val["per_direction"][direction]["miou"]
                              for direction in ("above", "below", "left", "right")},
        },
        "failure_attribution": {
            "counts": attribution["counts"], "percentages": attribution["percentages"],
            "summary": attribution["summary"],
            "dominant_bottleneck": attribution["dominant_bottleneck"],
        },
        "cli_audit": {"all_checks_passed": cli["all_checks_passed"],
                      "success_exit_code": cli["success_path"]["exit_code"],
                      "required_files_present": cli["success_path"]["required_files_present"],
                      "ground_truth_used": cli["success_path"]["ground_truth_used"],
                      "unsupported_exit_code": cli["unsupported_program_path"]["exit_code"]},
        "gate_constants": {"reference": GATES_REFERENCE, "language": GATES_LANGUAGE,
                           "paraphrase": GATES_PARAPHRASE},
        "interpretation_boundary": {
            "task6z_turned_into_success_claim": False,
            "parser_retrained": False,
            "reference_hardening_reopened": False,
            "global_attention_added": False,
            "z_b3_changed": False,
            "fields_changed": False,
            "full_training_started": False,
            "test_evaluated": False,
            "repair_proposed": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 7A 的 L3 predicted-reference、ProgramHead 与 failure "
                           "attribution 结果决定下一步，不自行进行 parser 再训练、reference 再硬化、"
                           "attention/global competition 或正式全量训练。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7a.report] A0 {'PASS' if a0['reproduction_passed'] else 'FAIL'} | A1 strict "
          f"{a1_val['strict']['miou']:.4f} retention {a1_val['retention']:.4f} paired "
          f"{a1_paired['passed']}/20 | A2 strict {a2_val['strict_all_miou']:.4f} parser "
          f"{parser_audit['exact_accuracy']:.4f} | paraphrase "
          f"{paraphrase_result['total_correct']}/24 ready {paraphrase_ready} | bottleneck "
          f"{attribution['dominant_bottleneck']} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
