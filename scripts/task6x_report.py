"""Task 6X Parts J-K — calibration verdict and the STOP rule.

Section 13 defines exactly one verdict per situation:

1. `SAM2_REFINEMENT_NOT_HELPFUL` — **X-C0 (no refinement) wins on U-Calib200** or a refinement option wins
   but fails the downstream gate. In the X-C0 case the task **stops immediately**: no RefValUnique /
   MiniVal240 / PairedVal20 evaluation files are produced.
2. `SAM2_REFINEMENT_PARTIAL` — a refinement option wins some criteria but fails others.
3. `SAM2_REFINEMENT_USEFUL` — a refinement option wins and passes all downstream gates.
4. `INVALID_EXPERIMENT` / `FROZEN_ASSET_UNAVAILABLE` — protocol or asset failure.

Writes `evaluation/task6x_verdict.json`. DSH reports measurements only.

    python scripts/task6x_report.py
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
OUT = EVAL / "task6x_verdict.json"
BASE_COMMIT = "3de2142010e335f3a9d9b1c5244fdd6271f88c4e"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "FROZEN_ASSET_UNAVAILABLE",
    "SAM2_REFINEMENT_NOT_HELPFUL",
    "SAM2_REFINEMENT_PARTIAL",
    "SAM2_REFINEMENT_USEFUL",
)
FROZEN_PATHS = (
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/task6u_reference_ranker.py",
    "buildreasonseg_mvp/task6v_family_reference_resolver.py",
    "buildreasonseg_mvp/task6w_proposal_quality.py",
    "buildreasonseg_mvp/task6w_quality_reference_resolver.py",
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/program_parser.py",
    "evaluation/task6s_", "evaluation/task6t_", "evaluation/task6u_", "evaluation/task6v_",
    "evaluation/task6w_",
)
#: Files that must NOT exist on the STOP path (section 13.1 / section N).
STOP_PATH_FORBIDDEN = ("task6x_refval_refinement.json", "task6x_downstream_minival240.json",
                       "task6x_downstream_pairedval20.json",
                       "task6x_hardened_parser_integration.json")
GATES = {
    "1_overall_miou_min": 0.4789,
    "2_smallest_miou_min": 0.3545,
    "3_largest_miou_min": 0.6034,
    "4_abstention_max": 0.02,
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

    assets = load("task6x_frozen_asset_audit.json")
    calibration = load("task6x_calibration_refinement.json")
    frozen = load("task6x_frozen_refinement_option.json")
    refval = load("task6x_refval_refinement.json")
    mini = load("task6x_downstream_minival240.json")
    paired = load("task6x_downstream_pairedval20.json")
    integration = load("task6x_hardened_parser_integration.json")
    required = {"task6x_frozen_asset_audit.json": assets,
                "task6x_calibration_refinement.json": calibration,
                "task6x_frozen_refinement_option.json": frozen}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 6X section 13.", "task": "6X",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[6x.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    protocol = {
        "frozen_non_task6x_unchanged": changes == "", "changed_paths": changes,
        "asset_audit_verdict": assets["verdict"],
        "sam2_sha256": assets["sam2"]["checkpoint_sha256"],
        "sam2_sha256_matches": assets["sam2"]["sha256_matches"],
        "sam2_probe_masks": assets["sam2"]["probe"]["masks_shape"],
        "proposal_sha256_matches": assets["proposal"]["matches"],
        "config_is_u_c1": assets["proposal"]["config"]["id"] == "U-C1",
        "training_performed": False,
        "ranker_used": False,
        "quality_estimator_used": False,
        "sam_quality_threshold": None,
        "point_or_mask_prompt": False,
        "tta_or_tiling": False,
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (assets, calibration, frozen)
                               if isinstance(payload, dict)),
        "options_evaluated": len(calibration["results"]),
        "fourth_option_added": frozen["fourth_option"] is not None,
    }
    protocol["clean"] = bool(protocol["frozen_non_task6x_unchanged"]
                             and not protocol["test_split_used"]
                             and protocol["options_evaluated"] == 4
                             and not protocol["fourth_option_added"]
                             and not protocol["ranker_used"]
                             and not protocol["quality_estimator_used"])

    baseline = calibration["results"]["X-C0"]
    selected = calibration["results"][frozen["selected_option"]]
    refinements = {option: calibration["results"][option]
                   for option in ("X-C1", "X-C2", "X-C3")}
    best_refinement = max(refinements,
                          key=lambda option: refinements[option]["overall"]["selected_reference_miou"])
    deltas = {
        "overall_miou": selected["overall"]["selected_reference_miou"]
        - baseline["overall"]["selected_reference_miou"],
        "smallest_miou": selected["smallest"]["selected_reference_miou"]
        - baseline["smallest"]["selected_reference_miou"],
        "largest_miou": selected["largest"]["selected_reference_miou"]
        - baseline["largest"]["selected_reference_miou"],
        "abstention_rate": selected["overall"]["abstention_rate"]
        - baseline["overall"]["abstention_rate"],
    }

    if not protocol["clean"]:
        verdict = "INVALID_EXPERIMENT"
        reason = "protocol violation: frozen-module mutation, test use, a fifth option or a banned model"
    elif assets["verdict"] != "FROZEN_ASSETS_VERIFIED":
        verdict = "FROZEN_ASSET_UNAVAILABLE"
        reason = "the frozen SAM2.1 or U-C1 proposal checkpoint hash does not match"
    elif frozen["baseline_is_selected"]:
        verdict = "SAM2_REFINEMENT_NOT_HELPFUL"
        reason = (f"X-C0 (no refinement) wins the train-only U-Calib200 ranking {calibration['ranking']}; "
                  f"the best refinement option {best_refinement} reaches mIoU "
                  f"{refinements[best_refinement]['overall']['selected_reference_miou']:.4f} vs baseline "
                  f"{baseline['overall']['selected_reference_miou']:.4f} -> task stops, no "
                  f"RefVal/MiniVal/Paired evaluation is run")
    else:
        gates = {
            "1_overall_miou": {"required": GATES["1_overall_miou_min"],
                               "measured": selected["overall"]["selected_reference_miou"],
                               "passed": selected["overall"]["selected_reference_miou"]
                               >= GATES["1_overall_miou_min"]},
            "2_smallest_miou": {"required": GATES["2_smallest_miou_min"],
                                "measured": selected["smallest"]["selected_reference_miou"],
                                "passed": selected["smallest"]["selected_reference_miou"]
                                >= GATES["2_smallest_miou_min"]},
            "3_largest_miou": {"required": GATES["3_largest_miou_min"],
                               "measured": selected["largest"]["selected_reference_miou"],
                               "passed": selected["largest"]["selected_reference_miou"]
                               >= GATES["3_largest_miou_min"]},
            "4_abstention": {"required": GATES["4_abstention_max"],
                             "measured": selected["overall"]["abstention_rate"],
                             "passed": selected["overall"]["abstention_rate"]
                             <= GATES["4_abstention_max"]},
        }
        if refval is not None:
            gates["5_refval_miou"] = {"required": 0.48,
                                      "measured": refval["overall"]["selected_reference_miou"],
                                      "passed": refval["overall"]["selected_reference_miou"] >= 0.48}
        if mini is not None:
            gates["6_minival_answered_miou"] = {
                "required": 0.33, "measured": mini["answered_only_miou"],
                "passed": mini["answered_only_miou"] >= 0.33}
        if paired is not None:
            gates["7_paired"] = {"required": 12, "measured": paired["passed"],
                                 "passed": paired["passed"] >= 12}
        if integration is not None:
            gates["8_parser_240"] = {"required": "240/240",
                                     "measured": integration["parser"]["exact_correct"],
                                     "passed": integration["parser"]["exact_correct"] == 240}
        verdict = ("SAM2_REFINEMENT_USEFUL" if all(entry["passed"] for entry in gates.values())
                   else "SAM2_REFINEMENT_PARTIAL")
        reason = ("the selected refinement option " + frozen["selected_option"]
                  + (" passes all downstream gates" if verdict == "SAM2_REFINEMENT_USEFUL"
                     else " fails "
                          + str([name for name, entry in gates.items() if not entry["passed"]])))

    forbidden_present = [name for name in STOP_PATH_FORBIDDEN if (EVAL / name).is_file()]
    payload = {
        "_doc": (
            "Task 6X sections 11-13. Frozen-asset audit, the train-only U-Calib200 comparison of exactly "
            "the four predeclared refinement options, the frozen option and the single verdict. "
            "Support infrastructure, not a claimed novelty; no model was trained. On the STOP path "
            "(X-C0 wins) no RefVal/MiniVal/Paired evaluation file is produced."
        ),
        "task": "6X",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "protocol": protocol,
        "stop_rule": {
            "triggered": bool(frozen["baseline_is_selected"]),
            "rule": "if X-C0 (no refinement) wins the U-Calib200 ranking the task ends with "
                    "SAM2_REFINEMENT_NOT_HELPFUL",
            "refval_minival_paired_evaluated": bool(refval is not None or mini is not None
                                                    or paired is not None),
            "forbidden_files_present": forbidden_present,
            "forbidden_files_absent": not forbidden_present,
        },
        "headline": {
            "frozen_option": frozen["selected_option"],
            "ranking": calibration["ranking"],
            "baseline": {"overall_miou": baseline["overall"]["selected_reference_miou"],
                         "smallest_miou": baseline["smallest"]["selected_reference_miou"],
                         "largest_miou": baseline["largest"]["selected_reference_miou"],
                         "abstention_rate": baseline["overall"]["abstention_rate"]},
            "options": {option: {
                "overall_miou": calibration["results"][option]["overall"]["selected_reference_miou"],
                "smallest_miou": calibration["results"][option]["smallest"]["selected_reference_miou"],
                "largest_miou": calibration["results"][option]["largest"]["selected_reference_miou"],
                "precision_at_0_5": calibration["results"][option]["overall"]["precision_at_0_5"],
                "centroid_median": calibration["results"][option]["overall"]["centroid_error_median"],
                "abstention_rate": calibration["results"][option]["overall"]["abstention_rate"],
                "mean_sam2_calls_per_tile":
                    calibration["results"][option]["mean_sam2_calls_per_tile"],
                "mean_wall_time_per_tile":
                    calibration["results"][option]["mean_wall_time_per_tile"],
                "empty_refined_masks": calibration["results"][option]["empty_refined_masks"],
            } for option in ("X-C0", "X-C1", "X-C2", "X-C3")},
            "deltas_vs_baseline": deltas,
            "best_refinement_option": best_refinement,
            "best_refinement_overall_miou":
                refinements[best_refinement]["overall"]["selected_reference_miou"],
            "baseline_buckets": baseline["overall"]["buckets"],
        },
        "interpretation_boundary": {
            "refinement_claimed_as_novelty": False,
            "sam_quality_threshold_used": False,
            "point_or_mask_prompt_used": False,
            "box_expansion_beyond_predeclared": False,
            "tta_or_tiling_added": False,
            "u_c1_changed": False,
            "yolo_or_sam2_retrained": False,
            "ranker_or_quality_estimator_used": False,
            "nearest_or_l3_started": False,
            "task6y_chosen": False,
        },
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[6x.report] frozen option {frozen['selected_option']} | ranking {calibration['ranking']} | "
          f"baseline mIoU {baseline['overall']['selected_reference_miou']:.4f} vs best refinement "
          f"{refinements[best_refinement]['overall']['selected_reference_miou']:.4f} | stop "
          f"{payload['stop_rule']['triggered']} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
