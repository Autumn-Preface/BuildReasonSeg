"""Task 6V Parts H-I — predeclared flags, the directional hardening gate and the single verdict.

Section 13 `family_policy_improved` (versus the frozen Task 6U U-S1); section 14 the ten
`DIRECTIONAL_REFERENCE_HARDENING_PASS` conditions; section 15 the fixed priority order.

Writes `evaluation/task6v_verdict.json`. DSH reports measurements only.

    python scripts/task6v_report.py
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
OUT = EVAL / "task6v_verdict.json"
BASE_COMMIT = "b8b5224ec88d32af7d086e337016f6e7d2ada1a2"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "FROZEN_RESOLVER_ASSET_UNAVAILABLE",
    "FAMILY_POLICY_NOT_BETTER",
    "FAMILY_POLICY_PARTIAL",
    "FAMILY_CONDITIONED_REFERENCE_HARDENING_PASS",
)
FROZEN_PATHS = (
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/task6u_reference_ranker.py",
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/program_parser.py",
    "buildreasonseg_mvp/task6s_directional_pipeline.py",
    "evaluation/task6s_", "evaluation/task6t_", "evaluation/task6u_",
)
GATES = {
    "1_refval_miou_min": 0.45,
    "2_refval_centroid_median_max": 0.03,
    "3_refval_centroid_p90_max": 0.32,
    "4_minival_answered_miou_min": 0.325,
    "5_minival_strict_miou_min": 0.305,
    "6_paired_min": 12,
    "7_own_cross_margin_min": 0.30,
    "8_reference_fail_max": 105,
}
IMPROVEMENT = {"miou_delta_min": 0.015, "reference_ok_delta_min": 5,
               "selection_wrong_delta_min": 5, "abstention_rate_max": 0.05}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    calibration = load("task6v_calibration_family_policy.json")
    frozen = load("task6v_frozen_family_policy.json")
    refval = load("task6v_refval_family_policy.json")
    mini = load("task6v_downstream_minival240.json")
    paired = load("task6v_downstream_pairedval20.json")
    integration = load("task6v_hardened_parser_integration.json")
    task6u = load("task6u_refval_selector_comparison.json")
    required = {
        "task6v_calibration_family_policy.json": calibration,
        "task6v_frozen_family_policy.json": frozen,
        "task6v_refval_family_policy.json": refval,
        "task6v_downstream_minival240.json": mini,
        "task6v_downstream_pairedval20.json": paired,
        "task6v_hardened_parser_integration.json": integration,
        "task6u_refval_selector_comparison.json": task6u,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 6V section 15.", "task": "6V",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[6v.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    u_s1 = task6u["systems"]["U-S1"]["overall"]
    protocol = {
        "frozen_non_task6v_unchanged": changes == "", "changed_paths": changes,
        "training_performed": False,
        "ranker_retrained": False,
        "yolo_retrained": False,
        "parser_retrained": False,
        "b3_retrained": False,
        "test_split_used": any(payload.get("test_split_used", False) for payload in required.values()),
        "gt_in_inference": False,
        "policy_changed_after_refval": refval["policy_changed_after_refval"],
        "policy_frozen_before_refval": True,
        "options_compared": calibration["options_compared"],
        "fourth_option_added": frozen["fourth_option"] is not None,
        "selected_policy": frozen["policy"],
    }
    protocol["clean"] = bool(protocol["frozen_non_task6v_unchanged"]
                             and not protocol["test_split_used"]
                             and not protocol["policy_changed_after_refval"]
                             and protocol["options_compared"] == 3
                             and not protocol["fourth_option_added"]
                             and not protocol["training_performed"])

    improvement = {
        "miou": {"required": (u_s1["selected_reference_miou"] or 0.0) + IMPROVEMENT["miou_delta_min"],
                 "measured": refval["overall"]["selected_reference_miou"],
                 "passed": (refval["overall"]["selected_reference_miou"] or 0.0)
                 >= (u_s1["selected_reference_miou"] or 0.0) + IMPROVEMENT["miou_delta_min"]},
        "reference_ok": {"required": u_s1["buckets"]["REFERENCE_OK"] + IMPROVEMENT["reference_ok_delta_min"],
                         "measured": refval["overall"]["buckets"]["REFERENCE_OK"],
                         "passed": refval["overall"]["buckets"]["REFERENCE_OK"]
                         >= u_s1["buckets"]["REFERENCE_OK"] + IMPROVEMENT["reference_ok_delta_min"]},
        "selection_wrong": {
            "required": u_s1["buckets"]["REFERENCE_SELECTION_WRONG"]
            - IMPROVEMENT["selection_wrong_delta_min"],
            "measured": refval["overall"]["buckets"]["REFERENCE_SELECTION_WRONG"],
            "passed": refval["overall"]["buckets"]["REFERENCE_SELECTION_WRONG"]
            <= u_s1["buckets"]["REFERENCE_SELECTION_WRONG"]
            - IMPROVEMENT["selection_wrong_delta_min"]},
        "abstention_rate": {"required": IMPROVEMENT["abstention_rate_max"],
                            "measured": refval["overall"]["abstention_rate"],
                            "passed": refval["overall"]["abstention_rate"]
                            <= IMPROVEMENT["abstention_rate_max"]},
    }
    family_policy_improved = all(entry["passed"] for entry in improvement.values())

    gates = {
        "1_refval_miou": {"required": GATES["1_refval_miou_min"],
                          "measured": refval["overall"]["selected_reference_miou"],
                          "passed": (refval["overall"]["selected_reference_miou"] or 0.0)
                          >= GATES["1_refval_miou_min"]},
        "2_refval_centroid_median": {"required": GATES["2_refval_centroid_median_max"],
                                     "measured": refval["overall"]["centroid_error_median"],
                                     "passed": (refval["overall"]["centroid_error_median"] or 1.0)
                                     <= GATES["2_refval_centroid_median_max"]},
        "3_refval_centroid_p90": {"required": GATES["3_refval_centroid_p90_max"],
                                  "measured": refval["overall"]["centroid_error_p90"],
                                  "passed": (refval["overall"]["centroid_error_p90"] or 1.0)
                                  <= GATES["3_refval_centroid_p90_max"]},
        "4_minival_answered_miou": {"required": GATES["4_minival_answered_miou_min"],
                                    "measured": mini["answered_only_miou"],
                                    "passed": (mini["answered_only_miou"] or 0.0)
                                    >= GATES["4_minival_answered_miou_min"]},
        "5_minival_strict_miou": {"required": GATES["5_minival_strict_miou_min"],
                                  "measured": mini["strict_all_miou"],
                                  "passed": mini["strict_all_miou"] >= GATES["5_minival_strict_miou_min"]},
        "6_paired": {"required": GATES["6_paired_min"], "measured": paired["passed"],
                     "passed": paired["passed"] >= GATES["6_paired_min"]},
        "7_own_cross_margin": {"required": GATES["7_own_cross_margin_min"],
                               "measured": paired["own_cross_margin"],
                               "passed": paired["own_cross_margin"] >= GATES["7_own_cross_margin_min"]},
        "8_reference_fail_count": {"required": GATES["8_reference_fail_max"],
                                   "measured": mini["reference_fail_count"],
                                   "passed": mini["reference_fail_count"] <= GATES["8_reference_fail_max"]},
        "9_parser_integration_240": {"required": "240/240",
                                     "measured": integration["parser"]["exact_correct"],
                                     "passed": integration["parser"]["exact_correct"] == 240},
        "10_no_test_no_gt": {"measured": {"no_test_split": not protocol["test_split_used"],
                                          "no_gt_in_inference": True},
                             "passed": (not protocol["test_split_used"])},
    }
    all_gates = all(entry["passed"] for entry in gates.values())

    if not protocol["clean"]:
        verdict = "INVALID_EXPERIMENT"
        reason = ("protocol violation: test use, GT inference, post-RefVal policy change, frozen-module "
                  "mutation or a fourth option")
    elif not (calibration["frozen_assets"]["yolo_checkpoint"]["sha256"]
              == calibration["frozen_assets"]["yolo_checkpoint"]["expected"]
              and calibration["frozen_assets"]["ranker_checkpoint"]["sha256"]
              == calibration["frozen_assets"]["ranker_checkpoint"]["expected"]):
        verdict = "FROZEN_RESOLVER_ASSET_UNAVAILABLE"
        reason = "a frozen resolver asset hash does not match its recorded value"
    elif not family_policy_improved and not all_gates:
        verdict = "FAMILY_POLICY_NOT_BETTER"
        reason = ("the section 13 improvement flag fails (mIoU delta "
                  f"{(refval['overall']['selected_reference_miou'] or 0.0) - (u_s1['selected_reference_miou'] or 0.0):+.4f} "
                  f"< +0.015; REFERENCE_OK {refval['overall']['buckets']['REFERENCE_OK']} < "
                  f"{u_s1['buckets']['REFERENCE_OK'] + 5}) and the section 14 hardening gate fails on "
                  f"{sum(1 for entry in gates.values() if not entry['passed'])} of 10 conditions")
    elif all_gates:
        verdict = "FAMILY_CONDITIONED_REFERENCE_HARDENING_PASS"
        reason = "all section 14 gates pass"
    else:
        verdict = "FAMILY_POLICY_PARTIAL"
        reason = ("measurable improvement but "
                  f"{[name for name, entry in gates.items() if not entry['passed']]} fail")

    payload = {
        "_doc": (
            "Task 6V sections 13-15. Predeclared family-policy improvement flag, the ten directional "
            "reference-hardening conditions and the single verdict. No model was trained in Task 6V; the "
            "per-family policy was frozen on train-only U-Calib200 before any RefValUnique evaluation. "
            "Family routing is support infrastructure, not a claimed novelty."
        ),
        "task": "6V",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "protocol": protocol,
        "improvement_flag": {"family_policy_improved": family_policy_improved, "criteria": improvement,
                             "reference_baseline": "Task 6U U-S1",
                             "u_s1_reference": {key: u_s1[key] for key in (
                                 "selected_reference_miou", "precision_at_0_5", "abstention_rate")},
                             "u_s1_buckets": u_s1["buckets"]},
        "gates": gates,
        "all_gates_passed": all_gates,
        "gate_constants": GATES,
        "headline": {
            "frozen_policy": frozen["policy"],
            "calibration_ranking": calibration["ranking"],
            "calibration_selected_metrics": frozen["selected_metrics"],
            "refval": {
                "overall": {key: refval["overall"][key] for key in (
                    "selected_reference_miou", "selected_reference_dice", "precision_at_0_5",
                    "centroid_error_mean", "centroid_error_median", "centroid_error_p90",
                    "area_ratio_median", "abstention_rate")},
                "largest": {key: refval["largest"][key] for key in (
                    "selected_reference_miou", "precision_at_0_5")},
                "smallest": {key: refval["smallest"][key] for key in (
                    "selected_reference_miou", "precision_at_0_5")},
                "buckets": refval["buckets"],
                "option_usage": refval["option_usage"],
            },
            "comparison_task6u": {
                "U-S0_miou": task6u["systems"]["U-S0"]["overall"]["selected_reference_miou"],
                "U-S1_miou": u_s1["selected_reference_miou"],
                "U-S2_miou": task6u["systems"]["U-S2"]["overall"]["selected_reference_miou"],
                "U-S0_buckets": task6u["systems"]["U-S0"]["overall"]["buckets"],
                "U-S1_buckets": u_s1["buckets"],
                "U-S2_buckets": task6u["systems"]["U-S2"]["overall"]["buckets"],
                "oracle_ceiling_miou": refval["comparison_task6u"]["U-C1_oracle_selection_ceiling"][
                    "selected_reference_miou"],
            },
            "downstream": {
                "strict_all_miou": mini["strict_all_miou"],
                "answered_only_miou": mini["answered_only_miou"],
                "abstentions": mini["abstentions"],
                "reference_fail_count": mini["reference_fail_count"],
                "target_fail_with_reference_ok_count": mini["target_fail_with_reference_ok_count"],
                "largest_miou": mini["largest"]["largest"]["miou"],
                "smallest_miou": mini["largest"]["smallest"]["miou"],
                "per_direction": mini["per_direction"],
                "border_target": mini["border_target"],
                "tiny_target": mini["tiny_target"],
            },
            "paired": {"passed": paired["passed"], "mean_own_iou": paired["mean_own_iou"],
                       "mean_cross_iou": paired["mean_cross_iou"],
                       "own_cross_margin": paired["own_cross_margin"],
                       "reference_abstention_pairs": paired["reference_abstention_pairs"]},
            "parser_integration": {
                "exact": f"{integration['parser']['exact_correct']}/{integration['parser']['records']}",
                "strict_all_miou": integration["strict_all_miou"],
                "answered_only_miou": integration["answered_only_miou"],
                "abstentions": integration["abstentions"],
                "reference_fail_count": integration["reference_fail_count"],
            },
        },
        "interpretation_boundary": {
            "family_routing_claimed_as_novelty": False,
            "ranker_redesigned": False,
            "smallest_only_ranker_trained": False,
            "proposal_quality_classifier_trained": False,
            "candidate_config_changed": False,
            "confidence_threshold_added": False,
            "tta_or_tiling_added": False,
            "yolo_retrained": False,
            "nearest_or_l3_started": False,
            "task6w_chosen": False,
        },
        "training_performed": False,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[6v.report] policy {frozen['policy']} | improved {family_policy_improved} | refval mIoU "
          f"{refval['overall']['selected_reference_miou']:.4f} | answered "
          f"{mini['answered_only_miou']:.4f} | strict {mini['strict_all_miou']:.4f} | paired "
          f"{paired['passed']}/20 margin {paired['own_cross_margin']:+.4f} | ref_fail "
          f"{mini['reference_fail_count']} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
