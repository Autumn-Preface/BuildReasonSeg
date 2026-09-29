"""Task 6R Parts J-K — predeclared criteria, the strong-signal flag and the single verdict.

Section 20 criteria (never altered):

1. R1 overfit gate passes;
2. mask retention `R1 mIoU >= R0 mIoU - 0.02` (numerical threshold `0.4099680351479113`);
3. relation gain `R1 relation_accuracy >= R0 relation_accuracy + 0.08`;
4. `R1 PairedVal >= 16/20`;
5. `R1 own-cross margin >= 0.38`;
6. field remains useful `R1 mIoU - R2 mIoU >= 0.10`.

Section 21 additionally reports `strong_mask_gain` = `R1 mIoU >= R0 mIoU + 0.02` (not required for
feasibility). Section 22 applies the fixed priority order.

Writes `evaluation/task6r_verdict.json`.

    python scripts/task6r_report.py
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
OUT = EVAL / "task6r_verdict.json"
BASE_COMMIT = "7c19bec577282029d0eb0eac7187940262c1a63a"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "GRCL_IMPLEMENTATION_INVALID",
    "GRCL_OVERFIT_FAIL",
    "GRCL_MASK_RELATION_TRADEOFF",
    "GRCL_NO_MEANINGFUL_RELATION_GAIN",
    "GRCL_FIELD_REDUNDANCY_RISK",
    "GRCL_DIRECTIONAL_FEASIBLE",
)
CRITERIA_CONSTANTS = {
    "mask_retention_numerical_threshold": 0.4099680351479113,
    "relation_gain_min": 0.08,
    "paired_pass_min": 16,
    "own_cross_margin_min": 0.38,
    "field_usefulness_min": 0.10,
    "strong_mask_gain_min": 0.02,
}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_artifacts_changed() -> str:
    return subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", "evaluation/task6n_",
         "evaluation/task6o_", "evaluation/task6p_", "evaluation/task6q_",
         "buildreasonseg_mvp/geometric_relation_field.py",
         "buildreasonseg_mvp/geometric_relation_field_v02.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    audit = load("task6r_grcl_audit.json")
    baseline = load("task6r_b3_relation_baseline.json")
    overfit = load("task6r_overfit20.json")
    training = load("task6r_training.json")
    mini = load("task6r_mini_val.json")
    paired = load("task6r_paired_val.json")
    transfer = load("task6r_proposal_reference_transfer.json")
    required = {
        "task6r_grcl_audit.json": audit, "task6r_b3_relation_baseline.json": baseline,
        "task6r_overfit20.json": overfit, "task6r_training.json": training,
        "task6r_mini_val.json": mini, "task6r_paired_val.json": paired,
        "task6r_proposal_reference_transfer.json": transfer,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 6R section 22 verdict.", "task": "6R",
                         "verdict": "INVALID_EXPERIMENT", "reason": f"missing artifacts: {missing}"})
        print(f"[6r.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changed = frozen_artifacts_changed()
    protocol = {
        "frozen_artifacts_unchanged": changed == "",
        "changed_paths": changed,
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in required.values()),
        "b3_retrained": baseline["checkpoint"]["retrained"],
        "target_as_input": False,
    }
    protocol["clean"] = bool(protocol["frozen_artifacts_unchanged"]
                             and not protocol["test_split_used"]
                             and not protocol["b3_retrained"])

    criteria = mini["predeclared_criteria"]
    criteria["1_r1_overfit_gate"] = {"measured": overfit["gate"]["passed"],
                                     "passed": bool(overfit["gate"]["passed"])}
    mask_retention = criteria["2_mask_retention"]["passed"]
    relation_gain = criteria["3_relation_gain"]["gain"]
    relation_gain_ok = criteria["3_relation_gain"]["passed"]
    paired_ok = criteria["4_paired_discrimination"]["passed"]
    margin_ok = criteria["5_own_cross_margin"]["passed"]
    field_ok = criteria["6_field_remains_useful"]["passed"]
    r0_miou = criteria["2_mask_retention"]["r0_miou"]
    r1_miou = criteria["2_mask_retention"]["r1_miou"]
    all_pass = all(entry["passed"] for entry in criteria.values())
    strong_mask_gain = bool(r1_miou >= r0_miou + CRITERIA_CONSTANTS["strong_mask_gain_min"])

    # section 22 priority order, applied literally
    if not protocol["clean"]:
        verdict = "INVALID_EXPERIMENT"
        reason = "protocol violation: leakage, test access, frozen mutation or target-as-input"
    elif audit["verdict"] != "GRCL_VALID":
        verdict = "GRCL_IMPLEMENTATION_INVALID"
        reason = "the GRCL gradient/directional audit failed"
    elif not overfit["gate"]["passed"]:
        verdict = "GRCL_OVERFIT_FAIL"
        reason = "R1 failed the Overfit20 gate"
    elif relation_gain_ok and not mask_retention:
        verdict = "GRCL_MASK_RELATION_TRADEOFF"
        reason = "relation accuracy improves by >= 0.08 but mask retention fails"
    elif mask_retention and (not relation_gain_ok or not paired_ok):
        verdict = "GRCL_NO_MEANINGFUL_RELATION_GAIN"
        reason = ("mask retention passes but the relation-accuracy gain is below 0.08 or PairedVal "
                  f"is below 16/20 (gain {relation_gain:+.4f}, paired "
                  f"{criteria['4_paired_discrimination']['measured']}/20)")
    elif not mask_retention:
        verdict = "GRCL_MASK_RELATION_TRADEOFF"
        reason = "mask retention fails"
    elif not (margin_ok and field_ok):
        if not field_ok:
            verdict = "GRCL_FIELD_REDUNDANCY_RISK"
            reason = f"R1 - R2 mIoU {criteria['6_field_remains_useful']['delta']:+.4f} < 0.10"
        else:
            verdict = "GRCL_NO_MEANINGFUL_RELATION_GAIN"
            reason = f"own-cross margin {criteria['5_own_cross_margin']['measured']:.6f} < 0.38"
    elif all_pass:
        verdict = "GRCL_DIRECTIONAL_FEASIBLE"
        reason = "all section-20 criteria pass"
    else:
        verdict = "GRCL_NO_MEANINGFUL_RELATION_GAIN"
        reason = "a valid but non-positive GRCL outcome"

    payload = {
        "_doc": (
            "Task 6R sections 20-22. Predeclared GRCL feasibility criteria, the strong-signal flag and "
            "the single verdict. DSH reports measurements only and does not change lambda or design "
            "another loss. Oracle reference for training/evaluation; GT target is a label/score only; "
            "test split never used."
        ),
        "task": "6R",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "criteria_constants": CRITERIA_CONSTANTS,
        "protocol": protocol,
        "criteria": criteria,
        "all_criteria_passed": all_pass,
        "strong_mask_gain": strong_mask_gain,
        "grcl_audit": {"verdict": audit["verdict"],
                       "gradient_l1_min": min(row["gradient_l1"] for row in audit["gradient_rows"]),
                       "directional_sanity_passed": audit["directional_sanity_passed"],
                       "constants": audit["constants"]},
        "b3_baseline": {
            "reproduction": baseline["reproduction"],
            "mini_val_240": baseline["mini_val_240"],
            "paired_val_20": baseline["paired_val_20"],
        },
        "overfit20": {
            "gate": overfit["gate"],
            "r1_final": overfit["variants"]["R1"]["final"],
            "r2_final": overfit["variants"]["R2"]["final"],
        },
        "variants": {
            "R0": {"role": "frozen Task 6O B3 (BCE + Dice only)", "miou": r0_miou,
                   "relation_accuracy": baseline["mini_val_240"]["relation_accuracy"],
                   "paired": paired["variants"]["R0"]["passed"],
                   "own_cross_margin": paired["variants"]["R0"]["own_cross_margin"]},
            "R1": {"role": "B3 + GRCL", "miou": mini["variants"]["R1"]["miou"],
                   "dice": mini["variants"]["R1"]["dice"],
                   "precision_at_0_5": mini["variants"]["R1"]["precision_at_0_5"],
                   "relation_accuracy": mini["variants"]["R1"]["relation_accuracy"],
                   "relation_per_direction": mini["variants"]["R1"]["relation_per_direction"],
                   "per_relation_miou": mini["variants"]["R1"]["per_relation_miou"],
                   "per_reference_family_miou": mini["variants"]["R1"]["per_reference_family_miou"],
                   "mean_grcl": mini["variants"]["R1"]["mean_grcl"],
                   "mean_positive_signed_margin":
                       mini["variants"]["R1"]["mean_positive_signed_margin"],
                   "axis_violation_rate": mini["variants"]["R1"]["axis_violation_rate"],
                   "paired": paired["variants"]["R1"]["passed"],
                   "own_cross_margin": paired["variants"]["R1"]["own_cross_margin"],
                   "parameters": mini["variants"]["R1"]["parameters"],
                   "selected_epoch": mini["variants"]["R1"]["selected_epoch"],
                   "wall_seconds": mini["variants"]["R1"]["wall_seconds"],
                   "peak_vram_gb": mini["variants"]["R1"]["peak_vram_gb"]},
            "R2": {"role": "no-field + GRCL (B0 control)", "miou": mini["variants"]["R2"]["miou"],
                   "dice": mini["variants"]["R2"]["dice"],
                   "precision_at_0_5": mini["variants"]["R2"]["precision_at_0_5"],
                   "relation_accuracy": mini["variants"]["R2"]["relation_accuracy"],
                   "per_relation_miou": mini["variants"]["R2"]["per_relation_miou"],
                   "per_reference_family_miou": mini["variants"]["R2"]["per_reference_family_miou"],
                   "mean_grcl": mini["variants"]["R2"]["mean_grcl"],
                   "mean_positive_signed_margin":
                       mini["variants"]["R2"]["mean_positive_signed_margin"],
                   "axis_violation_rate": mini["variants"]["R2"]["axis_violation_rate"],
                   "paired": paired["variants"]["R2"]["passed"],
                   "own_cross_margin": paired["variants"]["R2"]["own_cross_margin"],
                   "parameters": mini["variants"]["R2"]["parameters"],
                   "selected_epoch": mini["variants"]["R2"]["selected_epoch"],
                   "wall_seconds": mini["variants"]["R2"]["wall_seconds"],
                   "peak_vram_gb": mini["variants"]["R2"]["peak_vram_gb"]},
        },
        "proposal_reference_transfer": {
            "target": transfer["target"], "paired": transfer["paired"],
            "comparison_task6q_b3_chain": transfer["comparison_task6q_b3_chain"],
            "diagnostic_only": True, "training_performed": False, "lambda_tuning": False,
        },
        "training": {
            "variant": {name: {"epochs_run": training["variants"][name]["epochs_run"],
                               "best_epoch": training["variants"][name]["best_epoch"]}
                        for name in ("R1", "R2")},
            "config": training["config"],
        },
        "test_split_used": False,
        "no_loss_or_gate_change_by_dsh": True,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(
        f"[6r.report] R0 mIoU {r0_miou:.6f} rel-acc "
        f"{baseline['mini_val_240']['relation_accuracy']:.4f} | R1 mIoU {r1_miou:.6f} rel-acc "
        f"{mini['variants']['R1']['relation_accuracy']:.4f} gain {relation_gain:+.4f} | R2 mIoU "
        f"{mini['variants']['R2']['miou']:.6f} | paired R1 "
        f"{paired['variants']['R1']['passed']}/20 margin "
        f"{paired['variants']['R1']['own_cross_margin']:+.6f} -> {verdict}",
        flush=True,
    )
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
