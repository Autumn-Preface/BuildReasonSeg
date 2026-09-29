"""Task 6P Parts I-J — gates and the single allowed verdict.

Predeclared gates (fixed, never altered):

* section 17 reference-head adequacy — RefValUnique mIoU >= 0.35, median normalized centroid error
  <= 0.05, p90 <= 0.12;
* section 18 predicted-reference chain retention — section 17 passes, predicted-reference target mIoU
  >= 0.3009776246035379 (= 70 % of the oracle B3 mIoU), PairedVal >= 10/20, own-cross margin >= 0.20.

Writes `evaluation/task6p_verdict.json`.

    python scripts/task6p_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.task6m_eval import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6p_verdict.json"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "DIFFERENTIABLE_FIELD_INVALID",
    "TASK6O_B3_REPRODUCTION_FAIL",
    "REFERENCE_HEAD_NOT_LEARNABLE",
    "REFERENCE_HEAD_INSUFFICIENT",
    "REFERENCE_ERROR_PROPAGATION_SEVERE",
    "PREDICTED_REFERENCE_CHAIN_FEASIBLE",
)

#: section 17/18 constants, fixed by the task file
GATES = {
    "reference_head_miou_min": 0.35,
    "centroid_median_max": 0.05,
    "centroid_p90_max": 0.12,
    "predicted_target_miou_min": 0.3009776246035379,
    "paired_pass_min": 10,
    "own_cross_margin_min": 0.20,
}
ORACLE_B3_MIOU = 0.4299680351479113
ORACLE_B3_PAIRED = 14
ORACLE_B3_MARGIN = 0.39719566349802166


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    field_audit = load("task6p_field_v02_audit.json")
    reproduction = load("task6p_b3_reproduction.json")
    reference_packs = load("task6p_reference_pack_manifest.json")
    overfit = load("task6p_reference_overfit20.json")
    reference_val = load("task6p_reference_val.json")
    target = load("task6p_predicted_reference_target_val.json")
    diagnostics = load("task6p_field_propagation_diagnostics.json")
    paired = load("task6p_predicted_reference_paired_val.json")

    required = {
        "task6p_field_v02_audit.json": field_audit,
        "task6p_b3_reproduction.json": reproduction,
        "task6p_reference_pack_manifest.json": reference_packs,
        "task6p_reference_overfit20.json": overfit,
        "task6p_reference_val.json": reference_val,
        "task6p_predicted_reference_target_val.json": target,
        "task6p_field_propagation_diagnostics.json": diagnostics,
        "task6p_predicted_reference_paired_val.json": paired,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 6P section 19 verdict.", "task": "6P",
                         "verdict": "INVALID_EXPERIMENT", "reason": f"missing artifacts: {missing}"})
        print(f"[6p.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    centroid = reference_val["selected_model"]["centroid_error_normalized"]
    adequacy = {
        "name": "17 reference-head adequacy",
        "miou": {"required": GATES["reference_head_miou_min"],
                 "measured": reference_val["selected_model"]["miou"],
                 "passed": reference_val["selected_model"]["miou"] >= GATES["reference_head_miou_min"]},
        "centroid_median": {"required": GATES["centroid_median_max"], "measured": centroid["median"],
                            "passed": centroid["median"] <= GATES["centroid_median_max"]},
        "centroid_p90": {"required": GATES["centroid_p90_max"], "measured": centroid["p90"],
                         "passed": centroid["p90"] <= GATES["centroid_p90_max"]},
    }
    adequacy["passed"] = bool(adequacy["miou"]["passed"] and adequacy["centroid_median"]["passed"]
                              and adequacy["centroid_p90"]["passed"])

    predicted_miou = target["predicted_target"]["overall"]["miou"]
    retention = {
        "name": "18 predicted-reference chain retention",
        "section_17_passed": {"required": True, "measured": adequacy["passed"],
                              "passed": adequacy["passed"]},
        "predicted_target_miou": {"required": GATES["predicted_target_miou_min"],
                                  "measured": predicted_miou,
                                  "passed": predicted_miou >= GATES["predicted_target_miou_min"]},
        "paired_pass": {"required": GATES["paired_pass_min"], "measured": paired["passed"],
                        "passed": paired["passed"] >= GATES["paired_pass_min"]},
        "own_cross_margin": {"required": GATES["own_cross_margin_min"],
                             "measured": paired["own_cross_margin"],
                             "passed": paired["own_cross_margin"] >= GATES["own_cross_margin_min"]},
    }
    retention["passed"] = all(entry["passed"] for key, entry in retention.items() if isinstance(entry, dict))

    # section 19 priority order, applied literally
    if field_audit["verdict"] != "FIELD_V02_VALID":
        verdict = "DIFFERENTIABLE_FIELD_INVALID"
        reason = "the v0.2 equivalence/gradient audit failed"
    elif reproduction["verdict"] != "B3_REPRODUCED":
        verdict = "TASK6O_B3_REPRODUCTION_FAIL"
        reason = "the frozen Task 6O B3 baseline could not be reproduced within 1e-6"
    elif overfit["verdict"] != "P1_PASS":
        verdict = "REFERENCE_HEAD_NOT_LEARNABLE"
        reason = "the reference head failed the P1 Overfit20 gate"
    elif not adequacy["passed"]:
        verdict = "REFERENCE_HEAD_INSUFFICIENT"
        reason = "section 17 reference-head adequacy gates fail"
    elif not retention["passed"]:
        verdict = "REFERENCE_ERROR_PROPAGATION_SEVERE"
        reason = "the reference head is adequate but the predicted-reference chain loses too much"
    else:
        verdict = "PREDICTED_REFERENCE_CHAIN_FEASIBLE"
        reason = "sections 17 and 18 pass"

    payload = {
        "_doc": (
            "Task 6P sections 17-19. Predicted-reference substitution gates and the single verdict. "
            "DSH reports measurements only. Oracle-reference only in Tasks 6N/6O; here the reference "
            "mask is predicted by a frozen head and the error is propagated through the "
            "differentiable v0.2 field into the frozen B3 target decoder."
        ),
        "task": "6P",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "priority_order_applied": list(ALLOWED_VERDICTS),
        "gate_constants": GATES,
        "criteria": {"17": adequacy, "18": retention},
        "field_v02": {
            "verdict": field_audit["verdict"],
            "max_abs_error_vs_v01": field_audit["equivalence"]["max_abs_error"],
            "mean_abs_error_vs_v01": field_audit["equivalence"]["mean_abs_error"],
            "gradient_min_l1": min(row["gradient_l1"] for row in field_audit["gradient_check"]["rows"]),
            "masks_equivalence": field_audit["equivalence"]["masks"],
            "masks_gradient": field_audit["gradient_check"]["masks"],
        },
        "b3_reproduction": {
            "verdict": reproduction["verdict"],
            "comparisons": reproduction["comparisons"],
        },
        "reference_packs": {
            "counts": reference_packs["counts"],
            "family_counts": reference_packs["family_counts"],
            "unique_key": reference_packs["unique_key"],
        },
        "reference_head": {
            "p1": {"verdict": overfit["verdict"], "best_miou": overfit["gate"]["best_miou"],
                   "best_dice": overfit["gate"]["best_dice"]},
            "p2": {"selected_epoch": reference_val["best_epoch"],
                   "miou": reference_val["selected_model"]["miou"],
                   "dice": reference_val["selected_model"]["dice"],
                   "precision_at_0_5": reference_val["selected_model"]["precision_at_0_5"],
                   "per_family": reference_val["selected_model"]["per_family"],
                   "centroid_error_normalized": centroid,
                   "median_area_ratio": reference_val["selected_model"]["median_area_ratio"],
                   "border_reference": reference_val["selected_model"]["border_reference"],
                   "non_border_reference": reference_val["selected_model"]["non_border_reference"],
                   "tiny_reference": reference_val["selected_model"]["tiny_reference"],
                   "parameters": reference_val["parameters"],
                   "wall_seconds": reference_val["wall_seconds"],
                   "peak_vram_gb": reference_val["peak_vram_gb"]},
        },
        "chain": {
            "predicted_target": target["predicted_target"]["overall"],
            "predicted_target_per_relation": target["predicted_target"]["per_relation"],
            "predicted_target_per_family": target["predicted_target"]["per_reference_family"],
            "paired_pass": paired["passed"], "paired_pairs": paired["pairs"],
            "paired_mean_own_iou": paired["mean_own_iou"],
            "paired_mean_cross_iou": paired["mean_cross_iou"],
            "paired_own_cross_margin": paired["own_cross_margin"],
            "oracle_b3_reference": {"miou": ORACLE_B3_MIOU, "paired_pass": ORACLE_B3_PAIRED,
                                    "own_cross_margin": ORACLE_B3_MARGIN},
            "miou_drop_vs_oracle": predicted_miou - ORACLE_B3_MIOU,
            "paired_drop_vs_oracle": paired["passed"] - ORACLE_B3_PAIRED,
        },
        "field_propagation": diagnostics["overall"],
        "field_propagation_per_family": diagnostics["per_family"],
        "field_propagation_per_relation": diagnostics["per_relation"],
        "test_split_used": False,
        "end_to_end_inference": False,
        "joint_training": False,
        "mllm_hidden_state_fusion": False,
        "no_architecture_decision_by_dsh": True,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(
        f"[6p.report] ref head mIoU {reference_val['selected_model']['miou']:.6f} (>= "
        f"{GATES['reference_head_miou_min']}) centroid median {centroid['median']:.6f} p90 "
        f"{centroid['p90']:.6f} | predicted target mIoU {predicted_miou:.6f} paired "
        f"{paired['passed']}/{paired['pairs']} margin {paired['own_cross_margin']:+.6f} -> {verdict}",
        flush=True,
    )
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
