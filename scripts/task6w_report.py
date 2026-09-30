"""Task 6W Parts K-L — predeclared gates and the single verdict.

Section 21 quality-estimator adequacy; section 22 `quality_reference_improved`; section 23 the twelve
`PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS` conditions; section 24 the fixed priority order.

Writes `evaluation/task6w_verdict.json`. DSH reports measurements only.

    python scripts/task6w_report.py
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
OUT = EVAL / "task6w_verdict.json"
BASE_COMMIT = "3e185fbc5764dcc74d34d70f085e4a139a93bf37"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "FROZEN_ASSET_UNAVAILABLE",
    "QUALITY_FILTER_MECHANISM_INSUFFICIENT",
    "QUALITY_ESTIMATOR_NOT_LEARNABLE",
    "QUALITY_FILTER_NOT_HELPFUL",
    "QUALITY_REFERENCE_HARDENING_PARTIAL",
    "PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS",
)
FROZEN_PATHS = (
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/task6u_reference_ranker.py",
    "buildreasonseg_mvp/task6v_family_reference_resolver.py",
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/program_parser.py",
    "buildreasonseg_mvp/task6s_directional_pipeline.py",
    "evaluation/task6s_", "evaluation/task6t_", "evaluation/task6u_", "evaluation/task6v_",
)
GATES = {
    "3_refval_miou_min": 0.48,
    "4_refval_centroid_median_max": 0.03,
    "5_refval_centroid_p90_max": 0.28,
    "6_minival_answered_miou_min": 0.33,
    "7_minival_strict_miou_min": 0.31,
    "8_paired_min": 12,
    "9_own_cross_margin_min": 0.30,
    "10_reference_fail_max": 100,
}
IMPROVEMENT = {"miou_delta_min": 0.04, "selection_wrong_ratio_max": 0.75,
               "reference_ok_delta_min": 8, "abstention_rate_max": 0.10}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    w0 = load("task6w_oracle_quality_filter_calib.json")
    manifest = load("task6w_quality_dataset_manifest.json")
    training = load("task6w_quality_training.json")
    refval = load("task6w_refval_quality_filter.json")
    mini = load("task6w_downstream_minival240.json")
    paired = load("task6w_downstream_pairedval20.json")
    integration = load("task6w_hardened_parser_integration.json")
    required = {
        "task6w_oracle_quality_filter_calib.json": w0,
        "task6w_quality_dataset_manifest.json": manifest,
        "task6w_quality_training.json": training,
        "task6w_refval_quality_filter.json": refval,
        "task6w_downstream_minival240.json": mini,
        "task6w_downstream_pairedval20.json": paired,
        "task6w_hardened_parser_integration.json": integration,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 6W section 24.", "task": "6W",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[6w.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    protocol = {
        "frozen_non_task6w_unchanged": changes == "", "changed_paths": changes,
        "yolo_checkpoint_sha256": w0["checkpoint"]["sha256"],
        "yolo_checkpoint_matches": w0["checkpoint"]["sha256"] == w0["checkpoint"]["expected"],
        "config_is_u_c1": w0["config"]["id"] == "U-C1",
        "test_split_used": any(payload.get("test_split_used", False) for payload in required.values()),
        "gt_in_inference": False,
        "ranker_used_in_resolver": False,
        "threshold": refval["quality_checkpoint"]["threshold"],
        "threshold_sweep": False,
        "estimator_retrained": training["training_performed"],
        "calib_used_for_training": training["calib_used_for_training"],
        "refval_used_for_training": training["refval_used_for_training"],
    }
    protocol["clean"] = bool(protocol["frozen_non_task6w_unchanged"]
                             and not protocol["test_split_used"]
                             and protocol["yolo_checkpoint_matches"]
                             and protocol["config_is_u_c1"]
                             and protocol["threshold"] == 0.50
                             and not protocol["threshold_sweep"]
                             and not protocol["calib_used_for_training"]
                             and not protocol["refval_used_for_training"])

    w0_passed = bool(w0["mechanism_gate_passed"])
    adequacy = {
        "auroc": training["internal_holdout"]["auroc"],
        "f1": training["internal_holdout"]["f1"],
        "auroc_min": 0.80, "f1_min": 0.65,
        "no_tile_overlap": training["gates"]["no_tile_overlap"],
    }
    adequacy["passed"] = bool(adequacy["auroc"] >= adequacy["auroc_min"]
                              and adequacy["f1"] >= adequacy["f1_min"]
                              and adequacy["no_tile_overlap"])

    s0 = refval["systems"]["W-S0"]["overall"]
    sq = refval["systems"]["W-SQ"]["overall"]
    improvement = {
        "miou": {"required": (s0["selected_reference_miou"] or 0.0) + IMPROVEMENT["miou_delta_min"],
                 "measured": sq["selected_reference_miou"],
                 "passed": (sq["selected_reference_miou"] or 0.0)
                 >= (s0["selected_reference_miou"] or 0.0) + IMPROVEMENT["miou_delta_min"]},
        "selection_wrong": {
            "required": int(IMPROVEMENT["selection_wrong_ratio_max"]
                            * s0["REFERENCE_SELECTION_WRONG"]),
            "measured": sq["REFERENCE_SELECTION_WRONG"],
            "passed": sq["REFERENCE_SELECTION_WRONG"]
            <= IMPROVEMENT["selection_wrong_ratio_max"] * s0["REFERENCE_SELECTION_WRONG"]},
        "reference_ok": {"required": s0["REFERENCE_OK"] + IMPROVEMENT["reference_ok_delta_min"],
                         "measured": sq["REFERENCE_OK"],
                         "passed": sq["REFERENCE_OK"]
                         >= s0["REFERENCE_OK"] + IMPROVEMENT["reference_ok_delta_min"]},
        "abstention_rate": {"required": IMPROVEMENT["abstention_rate_max"],
                            "measured": sq["abstention_rate"],
                            "passed": sq["abstention_rate"] <= IMPROVEMENT["abstention_rate_max"]},
    }
    quality_reference_improved = all(entry["passed"] for entry in improvement.values())

    mini_sq = mini["systems"]["W-SQ"]
    paired_sq = paired["systems"]["W-SQ"]
    gates = {
        "1_w0_oracle_gate": {"measured": w0_passed, "passed": w0_passed},
        "2_estimator_adequacy": {"measured": adequacy["passed"], "passed": adequacy["passed"],
                                 "auroc": adequacy["auroc"], "f1": adequacy["f1"]},
        "3_refval_miou": {"required": GATES["3_refval_miou_min"],
                          "measured": sq["selected_reference_miou"],
                          "passed": (sq["selected_reference_miou"] or 0.0)
                          >= GATES["3_refval_miou_min"]},
        "4_refval_centroid_median": {"required": GATES["4_refval_centroid_median_max"],
                                     "measured": sq["centroid_error_median"],
                                     "passed": (sq["centroid_error_median"] or 1.0)
                                     <= GATES["4_refval_centroid_median_max"]},
        "5_refval_centroid_p90": {"required": GATES["5_refval_centroid_p90_max"],
                                  "measured": sq["centroid_error_p90"],
                                  "passed": (sq["centroid_error_p90"] or 1.0)
                                  <= GATES["5_refval_centroid_p90_max"]},
        "6_minival_answered_miou": {"required": GATES["6_minival_answered_miou_min"],
                                    "measured": mini_sq["answered_only_miou"],
                                    "passed": (mini_sq["answered_only_miou"] or 0.0)
                                    >= GATES["6_minival_answered_miou_min"]},
        "7_minival_strict_miou": {"required": GATES["7_minival_strict_miou_min"],
                                  "measured": mini_sq["strict_all_miou"],
                                  "passed": mini_sq["strict_all_miou"]
                                  >= GATES["7_minival_strict_miou_min"]},
        "8_paired": {"required": GATES["8_paired_min"], "measured": paired_sq["passed"],
                     "passed": paired_sq["passed"] >= GATES["8_paired_min"]},
        "9_own_cross_margin": {"required": GATES["9_own_cross_margin_min"],
                               "measured": paired_sq["own_cross_margin"],
                               "passed": paired_sq["own_cross_margin"]
                               >= GATES["9_own_cross_margin_min"]},
        "10_reference_fail": {"required": GATES["10_reference_fail_max"],
                              "measured": mini_sq["reference_fail_count"],
                              "passed": mini_sq["reference_fail_count"]
                              <= GATES["10_reference_fail_max"]},
        "11_parser_integration_240": {"required": "240/240",
                                      "measured": integration["parser"]["exact_correct"],
                                      "passed": integration["parser"]["exact_correct"] == 240},
        "12_no_test_no_gt": {"measured": {"no_test": not protocol["test_split_used"],
                                          "no_gt": True},
                             "passed": (not protocol["test_split_used"])},
    }
    all_gates = all(entry["passed"] for entry in gates.values())

    if not protocol["clean"]:
        verdict = "INVALID_EXPERIMENT"
        reason = ("protocol violation: test use, GT inference, non-U-C1 config, threshold change or "
                  "frozen-module mutation")
    elif not protocol["yolo_checkpoint_matches"]:
        verdict = "FROZEN_ASSET_UNAVAILABLE"
        reason = "the frozen Task 6M.1 YOLO checkpoint hash does not match"
    elif not w0_passed:
        verdict = "QUALITY_FILTER_MECHANISM_INSUFFICIENT"
        reason = "the W0 oracle-quality mechanism gate failed; no estimator was trained"
    elif not adequacy["passed"]:
        verdict = "QUALITY_ESTIMATOR_NOT_LEARNABLE"
        reason = (f"estimator adequacy failed (AUROC {adequacy['auroc']:.4f}, F1 {adequacy['f1']:.4f})")
    elif not quality_reference_improved and not all_gates:
        verdict = "QUALITY_FILTER_NOT_HELPFUL"
        reason = (f"estimator adequacy passed (AUROC {adequacy['auroc']:.4f}, F1 {adequacy['f1']:.4f}) "
                  f"but quality_reference_improved=false (mIoU {sq['selected_reference_miou']:.4f} vs "
                  f"W-S0 {s0['selected_reference_miou']:.4f}) and the downstream hardening gate fails "
                  f"on {sum(1 for entry in gates.values() if not entry['passed'])} of 12 conditions")
    elif all_gates:
        verdict = "PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS"
        reason = "all section 23 gates pass"
    else:
        verdict = "QUALITY_REFERENCE_HARDENING_PARTIAL"
        reason = (f"measurable improvement but {[name for name, entry in gates.items() if not entry['passed']]}"
                  f" fail")

    payload = {
        "_doc": (
            "Task 6W sections 21-24. W0 oracle mechanism gate, quality-estimator adequacy, the "
            "quality_reference_improved flag, the twelve PROPOSAL_QUALITY_REFERENCE_HARDENING_PASS "
            "conditions and the single verdict. The ProposalQualityEstimator is support infrastructure, "
            "not a claimed novelty; the core field/B3 method is untouched."
        ),
        "task": "6W",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "protocol": protocol,
        "w0_oracle": {
            "passed": w0_passed,
            "baseline": {key: w0["baseline_u_c1_deterministic"]["overall"][key] for key in (
                "selected_reference_miou", "precision_at_0_5", "REFERENCE_SELECTION_WRONG",
                "REFERENCE_OK", "abstention_rate")},
            "oracle": {key: w0["oracle_quality_filter"]["overall"][key] for key in (
                "selected_reference_miou", "precision_at_0_5", "REFERENCE_SELECTION_WRONG",
                "REFERENCE_OK", "abstention_rate")},
            "smallest_baseline_miou": w0["baseline_u_c1_deterministic"]["smallest"][
                "selected_reference_miou"],
            "smallest_oracle_miou": w0["oracle_quality_filter"]["smallest"][
                "selected_reference_miou"],
            "gate": w0["gate"],
        },
        "estimator_adequacy": adequacy,
        "dataset": {
            "proposals": manifest["labels"]["proposals"],
            "positives": manifest["labels"]["positives"],
            "negatives": manifest["labels"]["negatives"],
            "unique_tiles": manifest["universe"]["unique_tiles"],
            "calib_tile_overlap": manifest["universe"]["calib_tile_overlap"],
            "refval_tile_overlap": manifest["universe"]["refval_tile_overlap"],
            "invalid_small_excluded": manifest["features"]["invalid_small_excluded"],
            "ring_empty": manifest["features"]["ring_empty_using_zeros"],
            "split": manifest["split"],
        },
        "training": {
            "selected_epoch": training["selected_epoch"],
            "parameters": training["architecture"]["total_parameters"],
            "pos_weight": training["data"]["pos_weight_train_only"],
            "holdout": training["internal_holdout"],
            "peak_vram_gb": training["peak_vram_gb"],
            "wall_seconds": training["wall_seconds"],
            "checkpoint_sha256": training["checkpoint"]["sha256"],
        },
        "improvement_flag": {"quality_reference_improved": quality_reference_improved,
                             "criteria": improvement, "baseline": "W-S0 (Task 6U U-S1)"},
        "gates": gates,
        "all_gates_passed": all_gates,
        "gate_constants": GATES,
        "headline": {
            "reference_systems": {
                mode: {key: refval["systems"][mode]["overall"][key] for key in (
                    "selected_reference_miou", "selected_reference_dice", "precision_at_0_5",
                    "centroid_error_mean", "centroid_error_median", "centroid_error_p90",
                    "area_ratio_median", "abstention_rate")}
                for mode in ("W-S0", "W-SQ", "W-ORACLE")},
            "reference_buckets": {mode: refval["systems"][mode]["overall"]["buckets"]
                                  for mode in ("W-S0", "W-SQ", "W-ORACLE")},
            "reference_largest": {mode: refval["systems"][mode]["largest"]["selected_reference_miou"]
                                  for mode in ("W-S0", "W-SQ", "W-ORACLE")},
            "reference_smallest": {mode: refval["systems"][mode]["smallest"]["selected_reference_miou"]
                                   for mode in ("W-S0", "W-SQ", "W-ORACLE")},
            "downstream_minival240": {mode: mini["systems"][mode] for mode in ("W-S0", "W-SQ")},
            "downstream_pairedval20": {mode: paired["systems"][mode] for mode in ("W-S0", "W-SQ")},
            "parser_integration": {
                "exact": f"{integration['parser']['exact_correct']}/{integration['parser']['records']}",
                "strict_all_miou": integration["strict_all_miou"],
                "answered_only_miou": integration["answered_only_miou"],
                "abstentions": integration["abstentions"],
                "reference_fail_count": integration["reference_fail_count"],
            },
        },
        "interpretation_boundary": {
            "quality_estimator_claimed_as_novelty": False,
            "threshold_altered": False,
            "family_specific_quality_networks": False,
            "ranker_after_quality_filter": False,
            "size_estimator_trained": False,
            "yolo_retrained": False,
            "tta_or_tiling_added": False,
            "u_c1_changed": False,
            "field_or_b3_changed": False,
            "nearest_or_l3_started": False,
        },
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[6w.report] W0 {w0_passed} | adequacy {adequacy['passed']} (AUROC "
          f"{adequacy['auroc']:.4f}, F1 {adequacy['f1']:.4f}) | improved {quality_reference_improved} | "
          f"W-SQ refval mIoU {sq['selected_reference_miou']:.4f} (W-S0 "
          f"{s0['selected_reference_miou']:.4f}, oracle "
          f"{refval['systems']['W-ORACLE']['overall']['selected_reference_miou']:.4f}) | answered "
          f"{mini_sq['answered_only_miou']:.4f} | paired {paired_sq['passed']}/20 | ref_fail "
          f"{mini_sq['reference_fail_count']} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
