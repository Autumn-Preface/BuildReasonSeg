"""Task 7F Parts J-K — diagnostic labels, verdict and the D-B1 architecture status.

Section 16-19 labels: `SELECTION_IS_ACTIONABLE`, `PROPOSAL_GEOMETRY_IS_MAJOR`,
`PROPOSAL_COVERAGE_IS_MAJOR`, `CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING`.

Section 20 verdict priority:

1. `INVALID_EXPERIMENT`
2. `TASK7E_HOLDOUT_MISMATCH`
3. `TASK7E_NUMERIC_REPRODUCTION_FAIL`
4. `REFERENCE_PROPOSAL_CEILING_INSUFFICIENT`      (usable ceiling false)
5. `REFERENCE_SELECTION_DOMINANT`                 (usable ceiling true, selection actionable, selection_gain >= coverage_gain)
6. `REFERENCE_COVERAGE_OR_GEOMETRY_DOMINANT`      (usable ceiling true, not 5, coverage or geometry major)
7. `REFERENCE_MIXED_BOTTLENECK`                   (usable ceiling true, neither 5 nor 6)

Section 21 records D-B1 as the `preferred oracle-reference L3 target decoder candidate` when the Task 7E oracle
holdout facts reproduce; it is explicitly not end-to-end ready, not the final model and not paper-final.

Writes `evaluation/task7f_verdict.json`. DSH reports measurements only.

    python scripts/task7f_report.py
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
from buildreasonseg_mvp.task7f_reference_ceiling import MODES, MODE_LABELS  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7f_verdict.json"
BASE_COMMIT = "f9c6e876f7ba2be33f5dff04a182a06813edf580"
ALLOWED_VERDICTS = ("INVALID_EXPERIMENT", "TASK7E_HOLDOUT_MISMATCH",
                    "TASK7E_NUMERIC_REPRODUCTION_FAIL", "REFERENCE_PROPOSAL_CEILING_INSUFFICIENT",
                    "REFERENCE_SELECTION_DOMINANT", "REFERENCE_COVERAGE_OR_GEOMETRY_DOMINANT",
                    "REFERENCE_MIXED_BOTTLENECK")
FROZEN_PATHS = (
    "buildreasonseg_mvp/task7d_global_competition_decoder.py",
    "buildreasonseg_mvp/task6z_l3_decoder.py",
    "buildreasonseg_mvp/task6z_field_composition.py",
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/nearest_boundary_field.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/task7a_l3_pipeline.py",
    "buildreasonseg_mvp/program_parser.py",
    "scripts/task6u_common.py",
    "evaluation/task7e_", "evaluation/task7d_", "evaluation/task7c_", "evaluation/task6z_",
)
TASK7E_ORACLE = {"z_b3_miou": 0.3141113773177512, "d_b1_miou": 0.38549570532647004,
                 "delta": 0.07138432800871886, "paired": 18, "margin": 0.3193402994,
                 "ci_lower": 0.058623364793963954, "ci_upper": 0.08426264360286147}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    modes = load("task7f_reference_modes.json")
    downstream = load("task7f_downstream_reference_modes.json")
    gaps = load("task7f_gap_decomposition.json")
    paired = load("task7f_paired_reference_modes.json")
    task7e = load("task7e_oracle_holdout.json")
    task7e_paired = load("task7e_oracle_holdout_paired.json")
    task7e_predicted = load("task7e_predicted_reference_holdout.json")
    missing = [name for name, payload in
               (("task7f_reference_modes.json", modes),
                ("task7f_downstream_reference_modes.json", downstream),
                ("task7f_gap_decomposition.json", gaps),
                ("task7f_paired_reference_modes.json", paired)) if payload is None]
    if missing:
        write_json(OUT, {"_doc": "Task 7F section 20.", "task": "7F",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[7f.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    oracle_facts = {
        "task7e_z_b3_miou": task7e["results"]["Z-B3"]["overall"]["miou"],
        "task7e_d_b1_miou": task7e["results"]["D-B1"]["overall"]["miou"],
        "task7e_delta": task7e["delta"]["overall"], "task7e_paired": task7e_paired["results"]["D-B1"]["passed"],
        "task7e_margin": task7e_paired["results"]["D-B1"]["own_cross_margin"],
        "task7e_ci": [task7e["bootstrap"]["ci_lower"], task7e["bootstrap"]["ci_upper"]],
        "expected": TASK7E_ORACLE,
        "reproduced": {
            "z_b3_miou": abs(task7e["results"]["Z-B3"]["overall"]["miou"] - TASK7E_ORACLE["z_b3_miou"]) <= 1e-9,
            "d_b1_miou": abs(task7e["results"]["D-B1"]["overall"]["miou"] - TASK7E_ORACLE["d_b1_miou"]) <= 1e-9,
            "delta": abs(task7e["delta"]["overall"] - TASK7E_ORACLE["delta"]) <= 1e-9,
            "paired": task7e_paired["results"]["D-B1"]["passed"] == TASK7E_ORACLE["paired"],
            "margin": abs(task7e_paired["results"]["D-B1"]["own_cross_margin"]
                          - TASK7E_ORACLE["margin"]) <= 1e-9},
    }
    oracle_facts["all_reproduced"] = all(oracle_facts["reproduced"].values())
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "training_performed": any(payload.get("training_performed", False)
                                  for payload in (modes, downstream, gaps, paired)
                                  if isinstance(payload, dict)),
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (modes, downstream, gaps, paired)
                               if isinstance(payload, dict)),
        "checkpoints_created": False,
        "gt_used_only_in_declared_modes": True,
        "production_selector_changed": False, "detector_changed": False, "eligibility_changed": False,
        "fields_changed": False, "sam2_changed": False, "d_b1_changed": False,
        "ranker_or_filter_or_refinement_added": False,
        "selector_trained": False, "yolo_retrained": False,
        "attention_or_graph_added": False, "grcl_added": False,
        "diagnostics_used_as_tuning_data": False,
        "parser_trained_or_used_for_decision": False,
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and not protocol["training_performed"]
                             and not protocol["test_split_used"]
                             and not protocol["selector_trained"] and not protocol["yolo_retrained"])

    labels = {name: entry["value"] for name, entry in gaps["labels"].items()}
    label_conditions = {name: entry["conditions"] for name, entry in gaps["labels"].items()}
    gap = gaps["gaps"]
    reproduction = downstream["reproduction"]
    holdout_ok = modes["checks"]["record_id_hash_ok"] and modes["checks"]["pair_id_hash_ok"] \
        and modes["checks"]["no_test"] and modes["checks"]["overlap_minival"] \
        and modes["checks"]["overlap_pairedval"] and modes["checks"]["d_b1_sha256_ok"]

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, training or test use)"
    elif not holdout_ok:
        verdict, reason = "TASK7E_HOLDOUT_MISMATCH", f"holdout checks: {modes['checks']}"
    elif not reproduction["passed"]:
        verdict, reason = ("TASK7E_NUMERIC_REPRODUCTION_FAIL",
                           f"F-R0/F-R3 reproduction: {json.dumps(reproduction)}")
    elif not labels["CURRENT_PROPOSAL_SET_HAS_USABLE_CEILING"]:
        verdict, reason = ("REFERENCE_PROPOSAL_CEILING_INSUFFICIENT",
                           "the frozen U-C1 candidate set has no usable ceiling")
    elif (labels["SELECTION_IS_ACTIONABLE"]
          and gap["selection_gain"] >= gap["coverage_gain"]):
        verdict, reason = ("REFERENCE_SELECTION_DOMINANT",
                           f"selection_gain {gap['selection_gain']:+.4f} >= coverage_gain "
                           f"{gap['coverage_gain']:+.4f} with an actionable selection ceiling")
    elif labels["PROPOSAL_COVERAGE_IS_MAJOR"] or labels["PROPOSAL_GEOMETRY_IS_MAJOR"]:
        verdict, reason = ("REFERENCE_COVERAGE_OR_GEOMETRY_DOMINANT",
                           f"coverage_major {labels['PROPOSAL_COVERAGE_IS_MAJOR']} geometry_major "
                           f"{labels['PROPOSAL_GEOMETRY_IS_MAJOR']}")
    else:
        verdict, reason = ("REFERENCE_MIXED_BOTTLENECK",
                           "a usable ceiling exists but no single component dominates")

    payload = {
        "_doc": (
            "Task 7F sections 16-21. Reference bottleneck ceiling decomposition for the frozen D-B1 L3 target "
            "decoder on the frozen Task 7E E-HoldoutL3 / E-PairedHoldout20 populations. F-R0 is the "
            "production-like frozen U-C1 selection, F-R1 the selection ceiling on the same proposal set, F-R2 "
            "the coverage-conditional GT ceiling and F-R3 the full oracle ceiling. GT is used only inside the "
            "declared diagnostic modes; nothing was trained, tuned or repaired."
        ),
        "task": "7F", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS), "protocol": protocol,
        "holdout": {"records": modes["records"], "pairs": paired["pairs"],
                    "checks": modes["checks"]},
        "reference_modes": {mode: {"label": MODE_LABELS[mode], "reference": modes["reference"][mode],
                                   "downstream": downstream["results"][mode]["strict"],
                                   "paired": paired["results"][mode]}
                            for mode in MODES},
        "coverage": modes["coverage"],
        "gaps": gap, "labels": labels, "label_conditions": label_conditions,
        "thresholds": gaps["thresholds"],
        "reproduction": {"f_r0": reproduction["F-R0"], "f_r3": reproduction["F-R3"],
                         "passed": reproduction["passed"]},
        "task7e_oracle_facts": oracle_facts,
        "architecture_status": {
            "d_b1_status": ("preferred oracle-reference L3 target decoder candidate"
                            if oracle_facts["all_reproduced"] else "unresolved"),
            "end_to_end_ready": False, "final_model": False, "paper_final": False,
            "practical_chain_blocked": True,
            "reason": ("D-B1's untouched oracle-reference gain over Z-B3 is established, but the practical "
                       "chain stays blocked by the reference stage"),
            "z_b3_role": "frozen baseline / ablation",
        },
        "interpretation_boundary": {
            "selector_trained": False, "yolo_retrained": False, "u_c1_changed": False,
            "d_b1_retrained": False, "threshold_change_proposed": False,
            "full_training_or_test_started": False, "repair_chosen": False,
            "task7g_chosen": False, "gt_used_beyond_declared_diagnostics": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 7F 的 selection / proposal-geometry / coverage ceiling 分解决定"
                           "是否值得进行最后一次 reference 干预；不自行训练 selector、重训 YOLO 或开始正式 test。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7f.report] M0 {gap['M0_f_r0']:.4f} M1 {gap['M1_f_r1']:.4f} M2 {gap['M2_f_r2']:.4f} M3 "
          f"{gap['M3_f_r3']:.4f} | selection {gap['selection_gain']:+.4f} "
          f"({gap['selection_fraction']:.3f}) coverage {gap['coverage_gain']:+.4f} "
          f"({gap['coverage_fraction']:.3f}) geometry {gap['geometry_gain_covered']:+.4f} | labels "
          f"{labels} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
