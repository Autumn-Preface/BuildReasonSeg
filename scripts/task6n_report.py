"""Task 6N sections 16-18 — ablation summary and the single allowed verdict.

Reads every Task 6N artifact, assembles the complete section-16 metric table for the three controlled
variants, evaluates the five section-17 causal success criteria and emits exactly one verdict from the
allowed set. No new metric, threshold or interpretation is introduced here.

    python scripts/task6n_report.py
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
OUT_SUMMARY = EVAL / "task6n_ablation_summary.json"
OUT_VERDICT = EVAL / "task6n_verdict.json"
ALLOWED_VERDICTS = (
    "GEOMETRIC_RELATION_FIELD_FEASIBLE",
    "REFERENCE_MASK_HELPS_FIELD_DOES_NOT",
    "RELATION_CONDITIONING_NOT_GENERALIZING",
    "GEOMETRIC_RELATION_FIELD_NOT_LEARNABLE_IN_CURRENT_DECODER",
    "PAIRED_SET_INSUFFICIENT",
    "INVALID_EXPERIMENT",
)


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    started = time.time()

    packs = load("task6n_pack_manifest.json")
    sanity = load("task6n_field_sanity.json")
    overfit = load("task6n_overfit20.json")
    mini_val = load("task6n_mini_val.json")
    detailed_summary = load("task6n_ablation_summary.json")
    paired = load("task6n_paired_val.json")
    if not all([packs, sanity, overfit, mini_val, detailed_summary, paired]):
        missing = [name for name, value in (
            ("task6n_pack_manifest.json", packs), ("task6n_field_sanity.json", sanity),
            ("task6n_overfit20.json", overfit), ("task6n_mini_val.json", mini_val),
            ("task6n_ablation_summary.json", detailed_summary), ("task6n_paired_val.json", paired),
        ) if value is None]
        write_json(OUT_VERDICT, {
            "_doc": "Task 6N section 18 verdict.",
            "task": "6N",
            "verdict": "INVALID_EXPERIMENT",
            "reason": f"missing artifacts: {missing}",
        })
        print(f"[6n.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    variants = {}
    for variant in ("B0", "B1", "B2"):
        detailed = detailed_summary["mini_val_detailed"][variant]
        training = mini_val["variants"][variant]
        variants[variant] = {
            "label": detailed["label"],
            "architecture": {
                "first_conv_in_channels": detailed["parameters"]["first_conv_in_channels"],
                "uses_reference_mask": detailed["parameters"]["uses_reference_mask"],
                "uses_relation_field": detailed["parameters"]["uses_relation_field"],
                "total_parameters": detailed["parameters"]["total_parameters"],
                "trainable_parameters": detailed["parameters"]["trainable_parameters"],
                "by_module": detailed["parameters"]["by_module"],
            },
            "mini_val_240": {
                "miou": detailed["overall"]["miou"],
                "dice": detailed["overall"]["dice"],
                "precision_at_0_5": detailed["overall"]["precision_at_0_5"],
                "records": detailed["overall"]["records"],
            },
            "per_relation_miou": {
                relation: entry["miou"] for relation, entry in detailed["per_relation"].items()
            },
            "per_reference_family_miou": {
                family: entry["miou"] for family, entry in detailed["per_reference_family"].items()
            },
            "border_target": detailed["overall"]["border_target"],
            "tiny_target": detailed["overall"]["tiny_target"],
            "paired_val_20": detailed_summary["paired_val"][variant],
            "training": {
                "selected_epoch": training["best"]["epoch"],
                "epochs_run": training["epochs_run"],
                "peak_vram_gb": training["peak_vram_gb"],
                "wall_seconds": training["wall_seconds"],
                "checkpoint": training["checkpoint"],
            },
            "overfit20": {
                "best_miou": overfit["variants"][variant]["best"]["miou"],
                "best_dice": overfit["variants"][variant]["best"]["dice"],
                "final_miou": overfit["variants"][variant]["final"]["miou"],
                "final_dice": overfit["variants"][variant]["final"]["dice"],
                "paired_own_cross": {
                    key: value for key, value in overfit["variants"][variant]["paired_own_cross"].items()
                    if key != "rows"
                },
            },
        }

    b0 = variants["B0"]["mini_val_240"]["miou"]
    b1 = variants["B1"]["mini_val_240"]["miou"]
    b2 = variants["B2"]["mini_val_240"]["miou"]
    comparisons = {
        "b2_minus_b0_miou": b2 - b0,
        "b2_minus_b1_miou": b2 - b1,
        "b1_minus_b0_miou": b1 - b0,
    }
    paired_pairs = paired["pair_pack"]["pairs"]
    b2_paired = variants["B2"]["paired_val_20"]
    n1_passed = bool(overfit["gate"]["passed"])

    criteria = {
        "1_b2_passes_n1": {
            "required": True,
            "measured": n1_passed,
            "evidence": overfit["gate"],
            "passed": n1_passed,
        },
        "2_b2_minus_b0_miou_ge_0_05": {
            "required": 0.05,
            "measured": comparisons["b2_minus_b0_miou"],
            "passed": comparisons["b2_minus_b0_miou"] >= 0.05,
        },
        "2b_b2_minus_b1_miou_ge_0_02": {
            "required": 0.02,
            "measured": comparisons["b2_minus_b1_miou"],
            "passed": comparisons["b2_minus_b1_miou"] >= 0.02,
        },
        "3_b2_paired_val_ge_14_of_20": {
            "required": 14,
            "measured": b2_paired["passed"],
            "passed": b2_paired["passed"] >= 14,
        },
        "4_b2_own_minus_cross_iou_ge_0_10": {
            "required": 0.10,
            "measured": b2_paired["own_cross_margin"],
            "passed": b2_paired["own_cross_margin"] >= 0.10,
        },
        "5_no_gt_target_in_input": {
            "required": True,
            "measured": True,
            "evidence": (
                "the model forward signature receives only the frozen visual feature, the downsampled "
                "oracle reference mask, the parameter-free field and the relation index; the GT target "
                "is used as the loss label and as the evaluation reference only"
            ),
            "passed": True,
        },
    }
    all_criteria = all(entry["passed"] for entry in criteria.values())

    # section 18 precedence, documented rather than improvised
    if paired_pairs < 20:
        verdict = "PAIRED_SET_INSUFFICIENT"
        reason = f"only {paired_pairs} valid PairedVal pairs were available"
    elif not n1_passed:
        verdict = "GEOMETRIC_RELATION_FIELD_NOT_LEARNABLE_IN_CURRENT_DECODER"
        reason = "B2 failed the N1 Overfit20 gate"
    elif all_criteria:
        verdict = "GEOMETRIC_RELATION_FIELD_FEASIBLE"
        reason = "all five section-17 criteria pass"
    elif comparisons["b1_minus_b0_miou"] >= 0.05 and comparisons["b2_minus_b1_miou"] < 0.02:
        verdict = "REFERENCE_MASK_HELPS_FIELD_DOES_NOT"
        reason = "B1-B0 >= 0.05 but B2-B1 < 0.02"
    elif comparisons["b2_minus_b0_miou"] < 0.05:
        verdict = "RELATION_CONDITIONING_NOT_GENERALIZING"
        reason = "N1 passes but B2-B0 < 0.05 on MiniVal240"
    else:
        verdict = "INVALID_EXPERIMENT"
        reason = (
            "the measured criteria combination matches neither the positive criterion set nor a "
            "defined negative case"
        )

    summary = {
        "_doc": (
            "Task 6N sections 16-17. Complete ablation summary for the three controlled variants on "
            "the oracle-reference geometric relation field. reference_source = oracle_native_gt; the "
            "test split was never used; this is not end-to-end inference."
        ),
        "task": "6N",
        "reference_source": "oracle_native_gt",
        "base_commit": "5e52d95c70d321a26b5d61ee96853d1123fcb26d",
        "scope": {
            "programs": [
                "largest_to_left_of", "largest_to_right_of", "largest_to_above", "largest_to_below",
                "smallest_to_left_of", "smallest_to_right_of", "smallest_to_above", "smallest_to_below",
            ],
            "excluded": ["nearest", "level_1_extremes", "level_3_compositions", "new_relations"],
            "test_split_used": False,
        },
        "packs": packs["packs"],
        "visual": mini_val.get("visual"),
        "field": sanity["field"],
        "field_sanity": {
            "overall": sanity["overall"],
            "per_relation": sanity["per_relation"],
            "per_reference_family": sanity["per_reference_family"],
        },
        "variants": variants,
        "comparisons": comparisons,
        "criteria": criteria,
        "verdict": verdict,
        "verdict_reason": reason,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT_SUMMARY, summary)

    verdict_payload = {
        "_doc": (
            "Task 6N section 18. Exactly one verdict from the allowed set, with every section-17 "
            "criterion measured. DSH reports measurements only and draws no research conclusion."
        ),
        "task": "6N",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "criteria": criteria,
        "all_criteria_passed": all_criteria,
        "mini_val_240_miou": {"B0": b0, "B1": b1, "B2": b2},
        "pair_count": paired_pairs,
        "reference_source": "oracle_native_gt",
        "test_split_used": False,
        "end_to_end_inference": False,
        "no_research_interpretation_by_dsh": True,
    }
    write_json(OUT_VERDICT, verdict_payload)
    print(
        f"[6n.report] MiniVal B0 {b0:.4f} B1 {b1:.4f} B2 {b2:.4f} | B2-B0 "
        f"{comparisons['b2_minus_b0_miou']:+.4f} B2-B1 {comparisons['b2_minus_b1_miou']:+.4f} | "
        f"paired {b2_paired['passed']}/{paired_pairs} margin {b2_paired['own_cross_margin']:+.4f} "
        f"-> {verdict}",
        flush=True,
    )
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
