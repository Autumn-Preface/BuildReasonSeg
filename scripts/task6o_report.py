"""Task 6O Parts F-G — predeclared causal comparisons and the single allowed verdict.

Reproduced B2 (section 7) is the baseline. Section 14 defines the comparisons and the three
predeclared criteria; section 15 defines the verdict priority order. No threshold is changed and no
new criterion is invented here.

    python scripts/task6o_report.py
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
OUT_SUMMARY = EVAL / "task6o_causal_summary.json"
OUT_VERDICT = EVAL / "task6o_verdict.json"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE",
    "GEOMETRY_ONLY_BENCHMARK_CONFOUND",
    "DIRECT_REFERENCE_CHANNEL_MATTERS",
    "FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED",
    "FIELD_CAUSAL_SIGNAL_PARTIAL",
)

#: Section 14/15 constants — fixed by the task file and never altered.
CRITERIA_CONSTANTS = {
    "direct_reference_retention_miou_slack": 0.03,
    "direct_reference_retention_paired_min": 14,
    "direct_reference_retention_margin_min": 0.10,
    "visual_contribution_min": 0.10,
    "geometry_only_miou_slack": 0.05,
    "geometry_only_paired_min": 12,
    "direct_reference_channel_matters_drop": 0.05,
}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    reproduction = load("task6o_b2_reproduction.json")
    overfit = load("task6o_overfit20.json")
    mini = load("task6o_mini_val.json")
    paired = load("task6o_paired_val.json")
    task6n = load("task6n_mini_val.json")
    if not all([reproduction, overfit, mini, paired, task6n]):
        missing = [name for name, value in (
            ("task6o_b2_reproduction.json", reproduction), ("task6o_overfit20.json", overfit),
            ("task6o_mini_val.json", mini), ("task6o_paired_val.json", paired),
            ("task6n_mini_val.json", task6n),
        ) if value is None]
        write_json(OUT_VERDICT, {
            "_doc": "Task 6O section 15 verdict.",
            "task": "6O",
            "verdict": "INVALID_EXPERIMENT",
            "reason": f"missing artifacts: {missing}",
        })
        print(f"[6o.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    b2_miou = reproduction["reproduced"]["mini_val"]["miou"]
    b2_dice = reproduction["reproduced"]["mini_val"]["dice"]
    b2_paired = reproduction["reproduced"]["paired_val"]
    b1_miou = task6n["variants"]["B1"]["selected_model"]["miou"]
    b3_miou = mini["variants"]["B3"]["overall"]["miou"]
    b3_dice = mini["variants"]["B3"]["overall"]["dice"]
    b4_miou = mini["variants"]["B4"]["overall"]["miou"]
    b4_dice = mini["variants"]["B4"]["overall"]["dice"]
    b3_paired = paired["variants"]["B3"]
    b4_paired = paired["variants"]["B4"]

    comparisons = {
        "delta_B3_B2_miou": b3_miou - b2_miou,
        "delta_B3_B1_miou": b3_miou - b1_miou,
        "delta_B3_B4_miou": b3_miou - b4_miou,
        "delta_B3_B2_dice": b3_dice - b2_dice,
        "delta_B3_B4_dice": b3_dice - b4_dice,
        "delta_B3_B2_paired_pass": b3_paired["passed"] - b2_paired["passed"],
        "delta_B3_B4_paired_pass": b3_paired["passed"] - b4_paired["passed"],
        "delta_B3_B2_own_cross_margin": b3_paired["own_cross_margin"] - b2_paired["own_cross_margin"],
        "delta_B3_B4_own_cross_margin": b3_paired["own_cross_margin"] - b4_paired["own_cross_margin"],
    }

    c = CRITERIA_CONSTANTS
    retention = {
        "name": "14.1 direct reference-channel retention",
        "condition": "B3_mIoU >= B2_mIoU - 0.03 AND B3 paired >= 14/20 AND B3 own-cross >= 0.10",
        "miou_ok": b3_miou >= b2_miou - c["direct_reference_retention_miou_slack"],
        "paired_ok": b3_paired["passed"] >= c["direct_reference_retention_paired_min"],
        "margin_ok": b3_paired["own_cross_margin"] >= c["direct_reference_retention_margin_min"],
        "measured": {"b3_miou": b3_miou, "b2_miou": b2_miou, "b2_miou_minus_slack":
                     b2_miou - c["direct_reference_retention_miou_slack"],
                     "b3_paired": b3_paired["passed"], "b3_own_cross_margin":
                     b3_paired["own_cross_margin"]},
    }
    retention["passed"] = bool(retention["miou_ok"] and retention["paired_ok"] and retention["margin_ok"])

    visual = {
        "name": "14.2 visual contribution",
        "condition": "B3_mIoU - B4_mIoU >= 0.10",
        "measured": {"delta_B3_B4_miou": comparisons["delta_B3_B4_miou"]},
        "required": c["visual_contribution_min"],
    }
    visual["passed"] = comparisons["delta_B3_B4_miou"] >= c["visual_contribution_min"]

    geometry_only = {
        "name": "14.3 geometry-only confound",
        "condition": "B4_mIoU >= B3_mIoU - 0.05 AND B4 paired >= 12/20",
        "miou_ok": b4_miou >= b3_miou - c["geometry_only_miou_slack"],
        "paired_ok": b4_paired["passed"] >= c["geometry_only_paired_min"],
        "measured": {"b4_miou": b4_miou, "b3_miou_minus_slack": b3_miou - c["geometry_only_miou_slack"],
                     "b4_paired": b4_paired["passed"]},
    }
    geometry_only["passed"] = bool(geometry_only["miou_ok"] and geometry_only["paired_ok"])

    o1_passed = bool(overfit["gate"]["passed"])
    b3_drop_ok = b3_miou < b2_miou - c["direct_reference_channel_matters_drop"]

    # section 15 priority order, applied literally
    if reproduction["verdict"] != "B2_REPRODUCED":
        verdict = "INVALID_EXPERIMENT"
        reason = "the frozen Task 6N B2 baseline could not be reproduced"
    elif not o1_passed:
        verdict = "FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE"
        reason = "B3 failed the O1 Overfit20 gate"
    elif geometry_only["passed"]:
        verdict = "GEOMETRY_ONLY_BENCHMARK_CONFOUND"
        reason = "section 14.3 passes: the handcrafted field alone nearly matches B3"
    elif b3_drop_ok:
        verdict = "DIRECT_REFERENCE_CHANNEL_MATTERS"
        reason = f"B3 mIoU is more than 0.05 below B2 ({comparisons['delta_B3_B2_miou']:+.6f})"
    elif retention["passed"] and visual["passed"] and not geometry_only["passed"]:
        verdict = "FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED"
        reason = "sections 14.1 and 14.2 pass and 14.3 does not"
    else:
        verdict = "FIELD_CAUSAL_SIGNAL_PARTIAL"
        reason = "a valid but partial causal outcome"

    variants = {
        "B2_reproduced": {
            "role": "task 6N full model (visual + M_ref_down + field + relation), reproduced here",
            "mini_val_miou": b2_miou, "mini_val_dice": b2_dice,
            "paired_pass": b2_paired["passed"], "paired_pairs": b2_paired["pairs"],
            "own_cross_margin": b2_paired["own_cross_margin"],
            "first_conv_in_channels": 146,
            "parameters": task6n["variants"]["B2"]["parameters"]["total_parameters"],
        },
        "B1_frozen_task6n": {
            "role": "reference-mask baseline (no field)",
            "mini_val_miou": b1_miou,
            "parameters": task6n["variants"]["B1"]["parameters"]["total_parameters"],
        },
        "B3": {
            "role": "task 6O: visual + field + relation, NO direct reference channel",
            "mini_val_miou": b3_miou, "mini_val_dice": b3_dice,
            "precision_at_0_5": mini["variants"]["B3"]["overall"]["precision_at_0_5"],
            "per_relation_miou": {relation: entry["miou"]
                                  for relation, entry in mini["variants"]["B3"]["per_relation"].items()},
            "per_reference_family_miou": {family: entry["miou"] for family, entry
                                          in mini["variants"]["B3"]["per_reference_family"].items()},
            "border_target": mini["variants"]["B3"]["overall"]["border_target"],
            "tiny_target": mini["variants"]["B3"]["overall"]["tiny_target"],
            "paired_pass": b3_paired["passed"], "paired_pairs": b3_paired["pairs"],
            "mean_own_iou": b3_paired["mean_own_iou"], "mean_cross_iou": b3_paired["mean_cross_iou"],
            "own_cross_margin": b3_paired["own_cross_margin"],
            "first_conv_in_channels": mini["variants"]["B3"]["parameters"]["first_conv_in_channels"],
            "parameters": mini["variants"]["B3"]["parameters"]["total_parameters"],
            "trainable_parameters": mini["variants"]["B3"]["parameters"]["trainable_parameters"],
            "peak_vram_gb": mini["variants"]["B3"]["training"]["peak_vram_gb"],
            "wall_seconds": mini["variants"]["B3"]["training"]["wall_seconds"],
            "best_epoch": mini["variants"]["B3"]["training"]["selected_epoch"],
            "o1_best_miou": overfit["variants"]["B3"]["best"]["miou"],
            "o1_best_dice": overfit["variants"]["B3"]["best"]["dice"],
            "o1_paired_own_cross": overfit["variants"]["B3"]["paired_own_cross"],
        },
        "B4": {
            "role": "task 6O geometry-only control: field + relation, NO visual, NO direct reference",
            "mini_val_miou": b4_miou, "mini_val_dice": b4_dice,
            "precision_at_0_5": mini["variants"]["B4"]["overall"]["precision_at_0_5"],
            "per_relation_miou": {relation: entry["miou"]
                                  for relation, entry in mini["variants"]["B4"]["per_relation"].items()},
            "per_reference_family_miou": {family: entry["miou"] for family, entry
                                          in mini["variants"]["B4"]["per_reference_family"].items()},
            "border_target": mini["variants"]["B4"]["overall"]["border_target"],
            "tiny_target": mini["variants"]["B4"]["overall"]["tiny_target"],
            "paired_pass": b4_paired["passed"], "paired_pairs": b4_paired["pairs"],
            "mean_own_iou": b4_paired["mean_own_iou"], "mean_cross_iou": b4_paired["mean_cross_iou"],
            "own_cross_margin": b4_paired["own_cross_margin"],
            "first_conv_in_channels": mini["variants"]["B4"]["parameters"]["first_conv_in_channels"],
            "parameters": mini["variants"]["B4"]["parameters"]["total_parameters"],
            "trainable_parameters": mini["variants"]["B4"]["parameters"]["trainable_parameters"],
            "peak_vram_gb": mini["variants"]["B4"]["training"]["peak_vram_gb"],
            "wall_seconds": mini["variants"]["B4"]["training"]["wall_seconds"],
            "best_epoch": mini["variants"]["B4"]["training"]["selected_epoch"],
            "o1_best_miou": overfit["variants"]["B4"]["best"]["miou"],
            "o1_best_dice": overfit["variants"]["B4"]["best"]["dice"],
            "o1_paired_own_cross": overfit["variants"]["B4"]["paired_own_cross"],
        },
    }

    summary = {
        "_doc": (
            "Task 6O sections 14-15. Causal decomposition of the Task 6N result: is the direct "
            "reference channel still needed with the field present (B3), and is the gain really "
            "field-guided visual segmentation or does the handcrafted field alone solve the benchmark "
            "(B4)? Oracle-reference only; test split never used."
        ),
        "task": "6O",
        "reference_source": "oracle_native_gt",
        "base_commit": "90f3735cf57115cd67e15b1b3457752e1fb049a6",
        "criteria_constants": CRITERIA_CONSTANTS,
        "b2_reproduction": {
            "verdict": reproduction["verdict"],
            "comparisons": reproduction["comparisons"],
            "tolerance": reproduction["tolerance"],
        },
        "o1": {
            "gate": overfit["gate"],
            "verdict": overfit["verdict"],
            "b3_best": overfit["variants"]["B3"]["best"],
            "b4_best": overfit["variants"]["B4"]["best"],
        },
        "variants": variants,
        "comparisons": comparisons,
        "criteria": {"14.1": retention, "14.2": visual, "14.3": geometry_only},
        "verdict": verdict,
        "verdict_reason": reason,
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT_SUMMARY, summary)

    verdict_payload = {
        "_doc": (
            "Task 6O section 15. Exactly one verdict in the fixed priority order, with every "
            "predeclared criterion measured. DSH reports measurements only and takes no architecture "
            "decision."
        ),
        "task": "6O",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "priority_order_applied": [
            "INVALID_EXPERIMENT", "FIELD_WITHOUT_DIRECT_REFERENCE_NOT_LEARNABLE",
            "GEOMETRY_ONLY_BENCHMARK_CONFOUND", "DIRECT_REFERENCE_CHANNEL_MATTERS",
            "FIELD_GUIDED_VISUAL_SEGMENTATION_SUPPORTED", "FIELD_CAUSAL_SIGNAL_PARTIAL",
        ],
        "criteria": {"14.1": retention, "14.2": visual, "14.3": geometry_only},
        "comparisons": comparisons,
        "mini_val_miou": {"B1_frozen": b1_miou, "B2_reproduced": b2_miou, "B3": b3_miou, "B4": b4_miou},
        "paired_pass": {"B2_reproduced": b2_paired["passed"], "B3": b3_paired["passed"],
                        "B4": b4_paired["passed"]},
        "reference_source": "oracle_native_gt",
        "test_split_used": False,
        "end_to_end_inference": False,
        "no_architecture_decision_by_dsh": True,
    }
    write_json(OUT_VERDICT, verdict_payload)
    print(
        f"[6o.report] B1 {b1_miou:.6f} | B2 {b2_miou:.6f} | B3 {b3_miou:.6f} | B4 {b4_miou:.6f} | "
        f"B3-B2 {comparisons['delta_B3_B2_miou']:+.6f} B3-B4 {comparisons['delta_B3_B4_miou']:+.6f} | "
        f"14.1 {retention['passed']} 14.2 {visual['passed']} 14.3 {geometry_only['passed']} -> {verdict}",
        flush=True,
    )
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
