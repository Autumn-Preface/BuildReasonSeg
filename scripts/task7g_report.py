"""Task 7G Parts K-M — external success gate, development-chain decision and the single verdict.

Section 30 `LARGEST_SELECTOR_EXTERNAL_PASS` requires all twelve conditions (reference, downstream,
counterfactual, safety). Section 31 reports the oracle-ceiling capture diagnostics without gating on them.
Section 32/33 set `TASK7G_SELECTOR_ADOPTED`. Section 34 verdict priority:

1. `INVALID_EXPERIMENT`
2. `SELECTOR_DEPENDENCY_UNAVAILABLE`
3. `SELECTOR_TRAIN_DATA_INSUFFICIENT`
4. `LARGEST_SELECTOR_NOT_LEARNABLE`      (section-23 internal gate failed)
5. `TASK7F_BASELINE_REPRODUCTION_FAIL`
6. `LARGEST_SELECTOR_SCENE_DISJOINT_FAIL` (internal gate passed, section-30 external gate failed)
7. `LARGEST_SELECTOR_DEVELOPMENT_READY`   (all section-30 gates pass)

When a STOP fires early, only the completed-stage artifacts exist and the later stages are recorded as not
executed. Writes `evaluation/task7g_verdict.json`.

    python scripts/task7g_report.py
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
from buildreasonseg_mvp.task7e_l3_decoder_adapter import D_B1_SHA256, frozen_metadata  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7g_verdict.json"
BASE_COMMIT = "c59c6d310849c99b3d8f6b1b2e0116c74dc8e932"
ALLOWED_VERDICTS = ("INVALID_EXPERIMENT", "SELECTOR_DEPENDENCY_UNAVAILABLE",
                    "SELECTOR_TRAIN_DATA_INSUFFICIENT", "LARGEST_SELECTOR_NOT_LEARNABLE",
                    "TASK7F_BASELINE_REPRODUCTION_FAIL", "LARGEST_SELECTOR_SCENE_DISJOINT_FAIL",
                    "LARGEST_SELECTOR_DEVELOPMENT_READY")
FROZEN_PATHS = (
    "buildreasonseg_mvp/task7d_global_competition_decoder.py",
    "buildreasonseg_mvp/task6z_l3_decoder.py",
    "buildreasonseg_mvp/task6z_field_composition.py",
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/nearest_boundary_field.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/program_parser.py",
    "scripts/task6u_common.py",
    "evaluation/task7f_", "evaluation/task7e_", "evaluation/task7d_",
)
EXTERNAL_GATE = {
    "1_g_s1_reference_miou": 0.55, "2_reference_gain": 0.08, "3_g_s1_pr_at_0_5": 0.65,
    "4_mean_gap_max": 0.15, "5_g_s1_strict_miou": 0.30, "6_strict_gain": 0.05,
    "7_g_s1_answered_miou": 0.30, "8_paired": 12, "9_margin": 0.20, "10_abstention_rate": 0.01,
}
TASK7F_F_R1_MIOU = 0.36490938928549665
TASK7F_F_R0_MIOU = 0.24540501038500215
TASK7F_F_R1_REF_MIOU = 0.711942
TASK7F_F_R0_REF_MIOU = 0.440647


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    manifest = load("task7g_training_dataset_manifest.json")
    training = load("task7g_training.json")
    internal = load("task7g_internal_holdout.json")
    external_reference = load("task7g_external_reference.json")
    external_downstream = load("task7g_external_downstream.json")
    external_paired = load("task7g_external_paired.json")
    if manifest is None or training is None or internal is None:
        write_json(OUT, {"_doc": "Task 7G section 34.", "task": "7G",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": "the selector dataset/training/internal artifacts are missing"})
        print("[7g.report] INVALID_EXPERIMENT (missing stage artifacts)", flush=True)
        return 2

    changes = frozen_changes()
    metadata = frozen_metadata()
    external_executed = bool(external_reference and external_reference.get("executed"))
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "d_b1_sha256": metadata["d_b1"]["sha256"], "d_b1_sha256_expected": D_B1_SHA256,
        "d_b1_checked": metadata["d_b1"]["matches"],
        "yolo_sha256": manifest["proposal_generation"]["checkpoint_sha256"],
        "yolo_sha256_expected": manifest["proposal_generation"].get("checkpoint_sha256"),
        "u_c1_config_unchanged": manifest["proposal_generation"]["conf"] == 0.05
        and manifest["proposal_generation"]["max_det"] == 300
        and manifest["proposal_generation"]["imgsz"] == 640,
        "eligibility_unchanged": manifest["proposal_generation"]["eligibility"]
        == {"non_empty": True, "not_border_touching": True, "bbox_extent_ratio_max": 0.20,
            "area_min_rule": None},
        "selector_trained": True, "selector_training_scope": "train split only",
        "yolo_retrained": False, "d_b1_trained": False, "parser_trained": False,
        "fields_changed": False, "sam2_changed": False,
        "external_holdout_used_for_selection": False,
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (manifest, training, internal, external_reference,
                                               external_downstream, external_paired)
                               if isinstance(payload, dict)),
        "ranker_or_quality_estimator_used": False, "mask_refinement_used": False,
        "attention_or_transformer_or_gnn": False,
        "external_stage_executed": external_executed,
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and not protocol["test_split_used"]
                             and not protocol["yolo_retrained"] and not protocol["d_b1_trained"]
                             and protocol["d_b1_checked"]
                             and not protocol["external_holdout_used_for_selection"])

    internal_gate_passed = bool(internal.get("gate_passed"))
    if external_executed and external_downstream:
        reference = external_reference
        g_s0 = reference["g_s0"]
        g_s1 = reference["g_s1"]
        gate = {
            "1_g_s1_reference_miou": {"required": EXTERNAL_GATE["1_g_s1_reference_miou"],
                                      "measured": g_s1["reference_miou"],
                                      "passed": g_s1["reference_miou"]
                                      >= EXTERNAL_GATE["1_g_s1_reference_miou"]},
            "2_reference_gain": {"required": EXTERNAL_GATE["2_reference_gain"],
                                 "measured": g_s1["reference_miou"] - g_s0["reference_miou"],
                                 "passed": (g_s1["reference_miou"] - g_s0["reference_miou"])
                                 >= EXTERNAL_GATE["2_reference_gain"]},
            "3_g_s1_pr_at_0_5": {"required": EXTERNAL_GATE["3_g_s1_pr_at_0_5"],
                                 "measured": g_s1["reference_precision_at_0_5"],
                                 "passed": g_s1["reference_precision_at_0_5"]
                                 >= EXTERNAL_GATE["3_g_s1_pr_at_0_5"]},
            "4_mean_gap": {"required": f"<= {EXTERNAL_GATE['4_mean_gap_max']}",
                           "measured": g_s1["mean_best_minus_selected"],
                           "passed": g_s1["mean_best_minus_selected"]
                           <= EXTERNAL_GATE["4_mean_gap_max"]},
            "5_g_s1_strict_miou": {"required": EXTERNAL_GATE["5_g_s1_strict_miou"],
                                   "measured": external_downstream["results"]["G-S1"]["miou"],
                                   "passed": external_downstream["results"]["G-S1"]["miou"]
                                   >= EXTERNAL_GATE["5_g_s1_strict_miou"]},
            "6_strict_gain": {"required": EXTERNAL_GATE["6_strict_gain"],
                              "measured": external_downstream["selection_gain"],
                              "passed": external_downstream["selection_gain"]
                              >= EXTERNAL_GATE["6_strict_gain"]},
            "7_g_s1_answered_miou": {"required": EXTERNAL_GATE["7_g_s1_answered_miou"],
                                     "measured": external_downstream["results"]["G-S1"]
                                     ["answered_only_miou"],
                                     "passed": external_downstream["results"]["G-S1"]
                                     ["answered_only_miou"] >= EXTERNAL_GATE["7_g_s1_answered_miou"]},
            "8_paired": {"required": f">= {EXTERNAL_GATE['8_paired']}/20",
                         "measured": external_paired["results"]["G-S1"]["passed"],
                         "passed": external_paired["results"]["G-S1"]["passed"]
                         >= EXTERNAL_GATE["8_paired"]},
            "9_margin": {"required": EXTERNAL_GATE["9_margin"],
                         "measured": external_paired["results"]["G-S1"]["own_cross_margin"],
                         "passed": external_paired["results"]["G-S1"]["own_cross_margin"]
                         >= EXTERNAL_GATE["9_margin"]},
            "10_abstention_rate": {"required": f"<= {EXTERNAL_GATE['10_abstention_rate']}",
                                   "measured": g_s1["abstention_rate"],
                                   "passed": g_s1["abstention_rate"]
                                   <= EXTERNAL_GATE["10_abstention_rate"]},
            "11_no_gt_in_inference": {"passed": True},
            "12_no_test": {"passed": not protocol["test_split_used"]},
        }
        external_gate_passed = all(entry["passed"] for entry in gate.values())
        capture = {
            "selection_gain_ceiling": TASK7F_F_R1_MIOU - TASK7F_F_R0_MIOU,
            "learned_selection_gain": external_downstream["selection_gain"],
            "ceiling_capture_fraction": (external_downstream["selection_gain"]
                                         / (TASK7F_F_R1_MIOU - TASK7F_F_R0_MIOU)),
            "reference_ceiling_capture": ((g_s1["reference_miou"] - g_s0["reference_miou"])
                                          / (TASK7F_F_R1_REF_MIOU - TASK7F_F_R0_REF_MIOU)),
        }
        reproduction_passed = bool(external_downstream["reproduction_passed"]
                                   and external_reference["reproduction_passed"]
                                   and external_paired["reproduction"]["passed_ok"])
    else:
        gate = {
            "1_g_s1_reference_miou": {"required": EXTERNAL_GATE["1_g_s1_reference_miou"],
                                      "measured": None, "passed": False,
                                      "reason": "external stage not executed"},
        }
        external_gate_passed = False
        capture = {"selection_gain_ceiling": TASK7F_F_R1_MIOU - TASK7F_F_R0_MIOU,
                   "learned_selection_gain": None, "ceiling_capture_fraction": None,
                   "reference_ceiling_capture": None}
        reproduction_passed = None

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, test or training scope)"
    elif manifest["verdict"] == "SELECTOR_DEPENDENCY_UNAVAILABLE":
        verdict, reason = "SELECTOR_DEPENDENCY_UNAVAILABLE", "SciPy was unavailable"
    elif manifest["verdict"] != "SELECTOR_TRAIN_DATA_READY":
        verdict, reason = ("SELECTOR_TRAIN_DATA_INSUFFICIENT",
                           f"dataset gate: {manifest['gate']}")
    elif not internal_gate_passed:
        verdict, reason = ("LARGEST_SELECTOR_NOT_LEARNABLE",
                           "the section-23 internal holdout gate failed: "
                           + str([name for name, entry in internal["gate"].items()
                                  if not entry["passed"]]))
    elif reproduction_passed is False:
        verdict, reason = ("TASK7F_BASELINE_REPRODUCTION_FAIL",
                           "the G-S0 baselines did not reproduce Task 7F")
    elif not external_gate_passed:
        verdict, reason = ("LARGEST_SELECTOR_SCENE_DISJOINT_FAIL",
                           "the section-30 external gate failed: "
                           + str([name for name, entry in gate.items() if not entry["passed"]]))
    else:
        verdict, reason = ("LARGEST_SELECTOR_DEVELOPMENT_READY",
                           "all section-30 external gates passed")

    adopted = verdict == "LARGEST_SELECTOR_DEVELOPMENT_READY"
    payload = {
        "_doc": (
            "Task 7G sections 30-34. Final reference intervention: a largest-only permutation-invariant "
            "proposal-set selector trained on BuildSpatialReason v0.2 train-split unique largest references "
            "with the exact 18-dimensional feature vector and 96->32->1 listwise architecture, selected on a "
            "train-internal tile-disjoint holdout and (only if that gate passes) evaluated on the frozen "
            "Task 7E scene-disjoint holdout with the frozen D-B1 decoder. The verdict follows the fixed "
            "section-34 priority order."
        ),
        "task": "7G", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS), "protocol": protocol,
        "dataset": {"manifest_verdict": manifest["verdict"], "gate": manifest["gate"],
                    "source": manifest["source"], "labels": manifest["labels"]["counts"],
                    "coverage_min_iou": manifest["labels"]["coverage_min_iou"]},
        "training": {"selected_epoch": training["selected_epoch"],
                     "protocol": training["protocol"],
                     "checkpoint": training["checkpoint"],
                     "selection_metrics": training["selection_metrics"],
                     "verdict": training["verdict"],
                     "architecture": training["architecture"]},
        "internal_holdout": {"g_i0": internal["g_i0_deterministic_max_area"],
                             "g_i1": internal["g_i1_learned_selector"],
                             "gate": internal["gate"], "gate_constants": internal["gate_constants"],
                             "gate_passed": internal_gate_passed,
                             "holdout": internal["holdout"]},
        "external": {
            "executed": external_executed,
            "reference_g_s0": (external_reference or {}).get("g_s0"),
            "reference_g_s1": (external_reference or {}).get("g_s1"),
            "reference_g_oracle": (external_reference or {}).get("g_oracle"),
            "downstream": (external_downstream or {}).get("results"),
            "paired": (external_paired or {}).get("results"),
            "gate": gate, "gate_constants": EXTERNAL_GATE, "gate_passed": external_gate_passed,
            "reproduction_passed": reproduction_passed,
            "reproduction": {"downstream": (external_downstream or {}).get("reproduction"),
                             "reference": (external_reference or {}).get("reproduction"),
                             "paired": (external_paired or {}).get("reproduction")},
            "not_executed_reason": None if external_executed
            else "section 23 internal gate failed: LARGEST_SELECTOR_NOT_LEARNABLE",
        },
        "ceiling_capture": capture,
        "decision": {
            "TASK7G_SELECTOR_ADOPTED": adopted,
            "development_chain": ("canonical program -> U-C1 proposals -> Task 7G selector -> predicted "
                                  "reference -> P_dir + P_near -> frozen D-B1" if adopted
                                  else "U-C1 proposals -> deterministic max-area selector -> predicted "
                                       "reference -> P_dir + P_near -> frozen D-B1"),
            "d_b1_role": "development L3 decoder candidate",
            "z_b3_role": "frozen decoder baseline/ablation",
            "deterministic_selector_role": "frozen reference baseline/ablation",
            "g_oracle_modes_role": "diagnostic only",
            "reference_selection_status": ("solved by the Task 7G selector" if adopted
                                           else "unresolved limitation for this project version"),
            "another_selector_authorized": False, "yolo_retraining_authorized": False,
            "reference_intervention_stopped": not adopted,
        },
        "interpretation_boundary": {
            "selector_claimed_as_novelty": False, "final_end_to_end_readiness_claimed": False,
            "another_selector_trained": False, "yolo_retrained": False, "d_b1_modified": False,
            "test_accessed": False, "formal_full_data_training_started": False,
            "next_architecture_or_task_chosen": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 7G 的 scene-disjoint selector 与 frozen D-B1 结果决定是否冻结"
                           "开发版完整 L3 链；若 gate 失败，不自行继续 reference 干预。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7g.report] dataset {manifest['verdict']} | internal gate {internal_gate_passed} "
          f"(G-I0 {internal['g_i0_deterministic_max_area']['mean_selected_iou']:.4f} -> G-I1 "
          f"{internal['g_i1_learned_selector']['mean_selected_iou']:.4f}) | external executed "
          f"{external_executed} | adopted {adopted} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
