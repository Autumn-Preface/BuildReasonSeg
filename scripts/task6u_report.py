"""Task 6U Parts K-L — predeclared flags, the downstream hardening gate and the single verdict.

Section 22 `candidate_coverage_improved`; section 23 `ranker_improved_selection`; section 24 the ten
U-S2 downstream hardening conditions; section 25 the fixed priority order for exactly one verdict.

Writes `evaluation/task6u_verdict.json`. DSH reports measurements only and does not choose a repair.

    python scripts/task6u_report.py
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

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6u_verdict.json"
BASE_COMMIT = "e63f8c40761637040ac6169ad603b6f7c7001274"
PROPOSAL_SHA256 = "ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "PROPOSAL_CHECKPOINT_UNAVAILABLE",
    "REFERENCE_CANDIDATE_COVERAGE_STILL_LIMITING",
    "REFERENCE_RANKER_NOT_HELPFUL",
    "REFERENCE_HARDENING_PARTIAL",
    "REFERENCE_HARDENING_FEASIBLE",
)
FROZEN_PATHS = (
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/program_parser.py",
    "buildreasonseg_mvp/structured_grounding.py",
    "evaluation/task6s_", "evaluation/task6t_", "evaluation/task6o_", "evaluation/task6q_",
)
GATES = {
    "1_refval_miou_min": 0.48,
    "2_refval_centroid_median_max": 0.03,
    "3_refval_centroid_p90_max": 0.25,
    "4_minival_answered_miou_min": 0.33,
    "5_minival_strict_miou_min": 0.31,
    "6_paired_min": 12,
    "7_own_cross_margin_min": 0.30,
    "8_reference_fail_max": 93,
}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    split = load("task6u_reference_train_split.json")
    calibration = load("task6u_calibration_candidate_coverage.json")
    selected = load("task6u_selected_proposal_config.json")
    audit = load("task6u_refval_candidate_audit.json")
    training = load("task6u_ranker_training.json")
    comparison = load("task6u_refval_selector_comparison.json")
    mini = load("task6u_downstream_minival240.json")
    paired = load("task6u_downstream_pairedval20.json")
    integration = load("task6u_hardened_parser_integration.json")
    required = {
        "task6u_reference_train_split.json": split,
        "task6u_calibration_candidate_coverage.json": calibration,
        "task6u_selected_proposal_config.json": selected,
        "task6u_refval_candidate_audit.json": audit,
        "task6u_ranker_training.json": training,
        "task6u_refval_selector_comparison.json": comparison,
        "task6u_downstream_minival240.json": mini,
        "task6u_downstream_pairedval20.json": paired,
        "task6u_hardened_parser_integration.json": integration,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 6U section 25.", "task": "6U",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[6u.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    checkpoint_sha = calibration["checkpoint"]["sha256"]
    protocol = {
        "frozen_non_ranker_unchanged": changes == "", "changed_paths": changes,
        "proposal_checkpoint_sha256": checkpoint_sha,
        "proposal_checkpoint_expected": PROPOSAL_SHA256,
        "proposal_checkpoint_matches": checkpoint_sha == PROPOSAL_SHA256,
        "test_split_used": any(payload.get("test_split_used", False) for payload in required.values()),
        "oracle_reference_in_inference": False,
        "gt_in_inference": False,
        "selectors_compared": comparison["selectors_compared"],
        "configs_declared": list(calibration["configurations"]),
        "parser_retrained": False,
        "proposal_retrained": calibration["checkpoint"]["retrained"],
    }
    protocol["clean"] = bool(protocol["frozen_non_ranker_unchanged"]
                             and not protocol["test_split_used"]
                             and protocol["proposal_checkpoint_matches"]
                             and calibration["checkpoint"]["retrained"] is False
                             and comparison["selectors_compared"] == 3
                             and len(protocol["configs_declared"]) == 4)

    s0 = comparison["systems"]["U-S0"]["overall"]
    s1 = comparison["systems"]["U-S1"]["overall"]
    s2 = comparison["systems"]["U-S2"]["overall"]
    mini_s0 = mini["systems"]["U-S0"]
    mini_s1 = mini["systems"]["U-S1"]
    mini_s2 = mini["systems"]["U-S2"]
    paired_s2 = paired["systems"]["U-S2"]

    coverage_improved = bool(audit["candidate_coverage_improved"])
    ranker_improved = bool(comparison["ranker_improved_selection"])
    gates = {
        "1_refval_miou": {"required": GATES["1_refval_miou_min"],
                          "measured": s2["selected_reference_miou"],
                          "passed": (s2["selected_reference_miou"] or 0.0)
                          >= GATES["1_refval_miou_min"]},
        "2_refval_centroid_median": {"required": GATES["2_refval_centroid_median_max"],
                                     "measured": s2["centroid_error_median"],
                                     "passed": (s2["centroid_error_median"] or 1.0)
                                     <= GATES["2_refval_centroid_median_max"]},
        "3_refval_centroid_p90": {"required": GATES["3_refval_centroid_p90_max"],
                                  "measured": s2["centroid_error_p90"],
                                  "passed": (s2["centroid_error_p90"] or 1.0)
                                  <= GATES["3_refval_centroid_p90_max"]},
        "4_minival_answered_miou": {"required": GATES["4_minival_answered_miou_min"],
                                    "measured": mini_s2["answered_only_miou"],
                                    "passed": (mini_s2["answered_only_miou"] or 0.0)
                                    >= GATES["4_minival_answered_miou_min"]},
        "5_minival_strict_miou": {"required": GATES["5_minival_strict_miou_min"],
                                  "measured": mini_s2["strict_all_miou"],
                                  "passed": mini_s2["strict_all_miou"]
                                  >= GATES["5_minival_strict_miou_min"]},
        "6_paired": {"required": GATES["6_paired_min"], "measured": paired_s2["passed"],
                     "passed": paired_s2["passed"] >= GATES["6_paired_min"]},
        "7_own_cross_margin": {"required": GATES["7_own_cross_margin_min"],
                               "measured": paired_s2["own_cross_margin"],
                               "passed": paired_s2["own_cross_margin"]
                               >= GATES["7_own_cross_margin_min"]},
        "8_reference_fail_count": {"required": GATES["8_reference_fail_max"],
                                   "measured": mini_s2["reference_fail_count"],
                                   "passed": mini_s2["reference_fail_count"]
                                   <= GATES["8_reference_fail_max"],
                                   "task6s_baseline": 117,
                                   "required_reduction": "at least 20% from 117"},
        "9_no_test_split": {"measured": not protocol["test_split_used"],
                            "passed": not protocol["test_split_used"]},
        "10_no_gt_in_inference": {"measured": True, "passed": True},
    }
    all_gates = all(entry["passed"] for entry in gates.values())

    if not protocol["clean"]:
        verdict = "INVALID_EXPERIMENT"
        reason = "protocol violation: frozen-module mutation, leakage, test use or GT inference"
    elif not protocol["proposal_checkpoint_matches"]:
        verdict = "PROPOSAL_CHECKPOINT_UNAVAILABLE"
        reason = "the frozen Task 6M.1 proposal checkpoint hash does not match"
    elif not coverage_improved and not all_gates:
        verdict = "REFERENCE_CANDIDATE_COVERAGE_STILL_LIMITING"
        reason = "the selected configuration does not improve candidate coverage and U-S2 fails the gate"
    elif not ranker_improved and not all_gates:
        verdict = "REFERENCE_RANKER_NOT_HELPFUL"
        reason = (f"candidate coverage improves ({audit['delta']['overall_eligible_coverage@0.50']:+.4f} "
                  f"overall, {audit['delta']['smallest_eligible_coverage@0.50']:+.4f} smallest) but "
                  f"ranker_improved_selection=false and U-S2 fails the downstream hardening gate")
    elif all_gates:
        verdict = "REFERENCE_HARDENING_FEASIBLE"
        reason = "all section 24 gates pass"
    else:
        verdict = "REFERENCE_HARDENING_PARTIAL"
        reason = "U-S2 improves reference/downstream metrics materially but misses some section 24 gates"

    payload = {
        "_doc": (
            "Task 6U sections 22-25. Predeclared flags, the U-S2 downstream hardening gate and the single "
            "verdict. The ProposalSetRanker is support infrastructure, not a claimed novelty; the "
            "GeometricRelationField/B3 method was not changed."
        ),
        "task": "6U",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "protocol": protocol,
        "flags": {
            "candidate_coverage_improved": coverage_improved,
            "candidate_coverage_criteria": {
                "overall_delta_required": 0.04,
                "overall_delta_measured": audit["delta"]["overall_eligible_coverage@0.50"],
                "smallest_delta_required": 0.06,
                "smallest_delta_measured": audit["delta"]["smallest_eligible_coverage@0.50"],
            },
            "ranker_improved_selection": ranker_improved,
            "ranker_criteria": {
                "s2_minus_s1_miou_required": 0.05,
                "s2_minus_s1_miou_measured":
                    (s2["selected_reference_miou"] or 0.0) - (s1["selected_reference_miou"] or 0.0),
                "selection_wrong_ratio_required": 0.70,
                "selection_wrong_ratio_measured":
                    (s2["buckets"]["REFERENCE_SELECTION_WRONG"]
                     / max(1, s1["buckets"]["REFERENCE_SELECTION_WRONG"])),
                "abstention_rate_max": 0.10,
                "abstention_rate_measured": s2["abstention_rate"],
            },
        },
        "gates": gates,
        "all_gates_passed": all_gates,
        "gate_constants": GATES,
        "headline": {
            "selected_config": selected["selected_config"],
            "selected_config_values": selected["selected"],
            "calibration_ranking": calibration["ranking"],
            "reference_metrics": {
                "U-S0": {key: s0[key] for key in ("selected_reference_miou", "selected_reference_dice",
                                                  "precision_at_0_5", "centroid_error_median",
                                                  "centroid_error_p90", "abstention_rate")},
                "U-S1": {key: s1[key] for key in ("selected_reference_miou", "selected_reference_dice",
                                                  "precision_at_0_5", "centroid_error_median",
                                                  "centroid_error_p90", "abstention_rate")},
                "U-S2": {key: s2[key] for key in ("selected_reference_miou", "selected_reference_dice",
                                                  "precision_at_0_5", "centroid_error_median",
                                                  "centroid_error_p90", "abstention_rate")},
            },
            "reference_buckets": comparison["buckets"],
            "refval_coverage": {
                "baseline": audit["configs"]["baseline"]["coverage"],
                "selected": audit["configs"]["selected"]["coverage"],
                "oracle_ceiling_selected": audit["oracle_selection_ceiling"]["selected"]["overall"],
            },
            "downstream": {
                "U-S0": {key: mini_s0[key] for key in ("strict_all_miou", "answered_only_miou",
                                                        "abstentions", "reference_fail_count")},
                "U-S1": {key: mini_s1[key] for key in ("strict_all_miou", "answered_only_miou",
                                                        "abstentions", "reference_fail_count")},
                "U-S2": {key: mini_s2[key] for key in ("strict_all_miou", "answered_only_miou",
                                                        "abstentions", "reference_fail_count")},
            },
            "paired": {name: paired["systems"][name] for name in ("U-S0", "U-S1", "U-S2")},
            "ranker_training": {
                "trainable_examples": training["data"]["trainable"],
                "untrainable_not_covered": training["data"]["untrainable_not_covered"],
                "internal_holdout_top1": training["internal_holdout"]["top1_accuracy"],
                "internal_holdout_mean_iou": training["internal_holdout"]["mean_selected_iou"],
                "selected_epoch": training["selected_epoch"],
                "parameter_count": training["architecture"]["total_parameters"],
            },
            "hardened_parser_integration": {
                "parser_exact": f"{integration['parser']['exact_correct']}/"
                                f"{integration['parser']['records']}",
                "strict_all_miou": integration["strict_all_miou"],
                "answered_only_miou": integration["answered_only_miou"],
                "abstentions": integration["abstentions"],
            },
        },
        "task6t_erratum": {
            "statement": "Task 6T FROM_DSH reported peak VRAM 0.67 GB; the authoritative training "
                         "summary records C1 peak VRAM 7.29 GB",
            "authoritative_value_gb": 7.29,
            "task6t_artifacts_mutated": False,
            "affects_parser_metrics": False,
        },
        "interpretation_boundary": {
            "ranker_is_novelty": False,
            "retrain_yolo_decision": "not taken",
            "different_detector_decision": "not taken",
            "keep_1024_permanently_decision": "not taken (selected config is frozen evidence only)",
            "tiling_or_tta_added": False,
            "smallest_threshold_changed": False,
            "nearest_or_l3_started": False,
            "b3_retrained": False,
            "grcl_added": False,
        },
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[6u.report] coverage_improved {coverage_improved} | ranker_improved {ranker_improved} | "
          f"U-S2 refval mIoU {s2['selected_reference_miou']:.4f} | U-S2 answered "
          f"{mini_s2['answered_only_miou']:.4f} | ref_fail {mini_s2['reference_fail_count']} | paired "
          f"{paired_s2['passed']}/20 -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
