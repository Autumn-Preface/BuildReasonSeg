"""Task 7D Parts M-N — predeclared criteria, diagnostic flags and the single verdict.

Section 28 primary D-B2 feasibility (all eleven conditions), section 29 diagnostic flags
(`global_competition_mask_gain`, `prototype_gain`, `field_guidance_gain`,
`competition_localizes_target`), section 30 priority order:

1. `INVALID_EXPERIMENT`
2. `TASK6Z_PACK_MISMATCH`
3. `TASK6Z_BASELINE_REPRODUCTION_FAIL`
4. `GLOBAL_COMPETITION_NOT_LEARNABLE`
5. `GLOBAL_COMPETITION_VISUAL_ONLY`              (D-B2-D-B3 < 0.08 and D-B3 >= D-B0 + 0.05)
6. `GLOBAL_COMPETITION_PROTOTYPE_NOT_HELPFUL`    (D-B2-D-B4 < 0.03 and D-B4 >= D-B0 + 0.05)
7. `GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN`       (D-B2 < 0.38 or D-B2-D-B0 < 0.05 or D-B2-D-B1 < 0.03)
8. `GLOBAL_COMPETITION_COUNTERFACTUAL_WEAK`      (mask criteria pass but paired/margin fail)
9. `RELATION_GUIDED_GLOBAL_COMPETITION_FEASIBLE` (all section-28 criteria pass)

Writes `evaluation/task7d_verdict.json`. DSH reports measurements only.

    python scripts/task7d_report.py
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
from buildreasonseg_mvp.task7d_global_competition_decoder import (  # noqa: E402
    ALL_VARIANTS,
    PRIMARY_VARIANT,
    TRAINABLE_VARIANTS,
    variant_report,
)
from scripts.task7d_train import verify_packs  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7d_verdict.json"
BASE_COMMIT = "632c9c02a373ba8eea7153aafad69281946bae26"
ALLOWED_VERDICTS = ("INVALID_EXPERIMENT", "TASK6Z_PACK_MISMATCH",
                    "TASK6Z_BASELINE_REPRODUCTION_FAIL", "GLOBAL_COMPETITION_NOT_LEARNABLE",
                    "GLOBAL_COMPETITION_VISUAL_ONLY", "GLOBAL_COMPETITION_PROTOTYPE_NOT_HELPFUL",
                    "GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN", "GLOBAL_COMPETITION_COUNTERFACTUAL_WEAK",
                    "RELATION_GUIDED_GLOBAL_COMPETITION_FEASIBLE")
FROZEN_PATHS = (
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/nearest_boundary_field.py",
    "buildreasonseg_mvp/task6z_field_composition.py",
    "buildreasonseg_mvp/task6z_l3_decoder.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/program_parser.py",
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "evaluation/task6z_", "evaluation/task7c_",
)
GATES = {"d_b2_miou_min": 0.38, "vs_b0_min": 0.05, "vs_b1_min": 0.03, "vs_b3_min": 0.08,
         "vs_b4_min": 0.03, "paired_min": 16, "margin_min": 0.30, "target_mass_min": 0.10,
         "argmax_in_target_min": 0.45, "mask_gain_flag": 0.03, "prototype_gain_flag": 0.02,
         "field_guidance_flag": 0.05, "localize_mass_flag": 0.08, "localize_argmax_flag": 0.35,
         "visual_only_vs_b3": 0.08, "visual_only_b3_gain": 0.05,
         "prototype_vs_b4": 0.03, "prototype_b4_gain": 0.05}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    baseline = load("task7d_baseline_reproduction.json")
    overfit = load("task7d_overfit20.json")
    training = load("task7d_training.json")
    mini = load("task7d_mini_val.json")
    paired = load("task7d_paired_val.json")
    diagnostics = load("task7d_competition_diagnostics.json")
    required = {"task7d_baseline_reproduction.json": baseline, "task7d_overfit20.json": overfit,
                "task7d_training.json": training, "task7d_mini_val.json": mini,
                "task7d_paired_val.json": paired,
                "task7d_competition_diagnostics.json": diagnostics}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 7D section 30.", "task": "7D",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[7d.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    packs = verify_packs()
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "packs": packs, "packs_match": all(entry["matches"] for entry in packs.values()),
        "l3_programs": list(mini["pack"]["by_direction"]) or None,
        "variants": list(TRAINABLE_VARIANTS), "variant_count": len(TRAINABLE_VARIANTS),
        "primary_variant": PRIMARY_VARIANT,
        "oracle_reference": "oracle_native_gt",
        "predicted_reference_used": False,
        "parser_used": False,
        "target_gt_used_as_input": False,
        "competition_supervised": False,
        "attention_or_transformer_or_gnn": False,
        "grcl": False,
        "target_proposals": False,
        "training_performed": True,
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (baseline, overfit, training, mini, paired, diagnostics)
                               if isinstance(payload, dict)),
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and protocol["packs_match"]
                             and not protocol["test_split_used"]
                             and not protocol["predicted_reference_used"]
                             and not protocol["competition_supervised"]
                             and not protocol["attention_or_transformer_or_gnn"]
                             and not protocol["grcl"])

    miou = {variant: mini["results"][variant]["overall"]["miou"] for variant in ALL_VARIANTS}
    dice = {variant: mini["results"][variant]["overall"]["dice"] for variant in ALL_VARIANTS}
    d_b2 = miou["D-B2"]
    diagnostics_d_b2 = diagnostics["variants"]["D-B2"]["summary"]
    criteria = {
        "1_overfit_passed": {"passed": bool(overfit["d_b2_overfit_passed"]),
                             "measured": overfit["gate"]},
        "2_d_b2_miou": {"required": GATES["d_b2_miou_min"], "measured": d_b2,
                        "passed": d_b2 >= GATES["d_b2_miou_min"]},
        "3_d_b2_minus_b0": {"required": GATES["vs_b0_min"], "measured": d_b2 - miou["D-B0"],
                            "passed": d_b2 - miou["D-B0"] >= GATES["vs_b0_min"]},
        "4_d_b2_minus_b1": {"required": GATES["vs_b1_min"], "measured": d_b2 - miou["D-B1"],
                            "passed": d_b2 - miou["D-B1"] >= GATES["vs_b1_min"]},
        "5_d_b2_minus_b3": {"required": GATES["vs_b3_min"], "measured": d_b2 - miou["D-B3"],
                            "passed": d_b2 - miou["D-B3"] >= GATES["vs_b3_min"]},
        "6_d_b2_minus_b4": {"required": GATES["vs_b4_min"], "measured": d_b2 - miou["D-B4"],
                            "passed": d_b2 - miou["D-B4"] >= GATES["vs_b4_min"]},
        "7_paired": {"required": f">= {GATES['paired_min']}/20",
                     "measured": paired["results"]["D-B2"]["passed"],
                     "passed": paired["results"]["D-B2"]["passed"] >= GATES["paired_min"]},
        "8_margin": {"required": GATES["margin_min"],
                     "measured": paired["results"]["D-B2"]["own_cross_margin"],
                     "passed": paired["results"]["D-B2"]["own_cross_margin"] >= GATES["margin_min"]},
        "9_target_mass": {"required": GATES["target_mass_min"],
                          "measured": diagnostics_d_b2["target_mass"],
                          "passed": diagnostics_d_b2["target_mass"] >= GATES["target_mass_min"]},
        "10_argmax_in_target": {"required": GATES["argmax_in_target_min"],
                                "measured": diagnostics_d_b2["argmax_in_target_rate"],
                                "passed": diagnostics_d_b2["argmax_in_target_rate"]
                                >= GATES["argmax_in_target_min"]},
        "11_no_target_gt_input": {"passed": not protocol["target_gt_used_as_input"]},
    }
    criteria_passed = all(entry["passed"] for entry in criteria.values())
    mask_criteria = [name for name in criteria if name not in ("7_paired", "8_margin")]
    mask_passed = all(criteria[name]["passed"] for name in mask_criteria)

    flags = {
        "global_competition_mask_gain": bool(d_b2 >= miou["D-B0"] + GATES["mask_gain_flag"]),
        "prototype_gain": bool(d_b2 >= miou["D-B4"] + GATES["prototype_gain_flag"]),
        "field_guidance_gain": bool(d_b2 >= miou["D-B3"] + GATES["field_guidance_flag"]),
        "competition_localizes_target": bool(diagnostics_d_b2["target_mass"]
                                             >= GATES["localize_mass_flag"]
                                             and diagnostics_d_b2["argmax_in_target_rate"]
                                             >= GATES["localize_argmax_flag"]),
    }
    visual_only = {"d_b2_minus_d_b3": d_b2 - miou["D-B3"],
                   "d_b3_minus_d_b0": miou["D-B3"] - miou["D-B0"],
                   "applies": bool((d_b2 - miou["D-B3"]) < GATES["visual_only_vs_b3"]
                                   and (miou["D-B3"] - miou["D-B0"])
                                   >= GATES["visual_only_b3_gain"])}
    prototype_not_helpful = {
        "d_b2_minus_d_b4": d_b2 - miou["D-B4"],
        "d_b4_minus_d_b0": miou["D-B4"] - miou["D-B0"],
        "applies": bool((d_b2 - miou["D-B4"]) < GATES["prototype_vs_b4"]
                        and (miou["D-B4"] - miou["D-B0"]) >= GATES["prototype_b4_gain"])}
    no_gain = {"d_b2_miou": d_b2, "d_b2_minus_d_b0": d_b2 - miou["D-B0"],
               "d_b2_minus_d_b1": d_b2 - miou["D-B1"],
               "applies": bool(d_b2 < GATES["d_b2_miou_min"]
                               or (d_b2 - miou["D-B0"]) < GATES["vs_b0_min"]
                               or (d_b2 - miou["D-B1"]) < GATES["vs_b1_min"])}

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, pack or test use)"
    elif not protocol["packs_match"]:
        verdict, reason = "TASK6Z_PACK_MISMATCH", "a frozen Task 6Z pack hash does not match"
    elif not baseline["reproduction_passed"]:
        verdict, reason = ("TASK6Z_BASELINE_REPRODUCTION_FAIL",
                           f"deltas {baseline['deltas']} exceed {baseline['tolerance']}")
    elif not overfit["d_b2_overfit_passed"]:
        verdict, reason = ("GLOBAL_COMPETITION_NOT_LEARNABLE",
                           "the D-B2 Overfit20 gate failed")
    elif visual_only["applies"]:
        verdict, reason = ("GLOBAL_COMPETITION_VISUAL_ONLY",
                           "D-B2-D-B3 < 0.08 while D-B3 >= D-B0 + 0.05")
    elif prototype_not_helpful["applies"]:
        verdict, reason = ("GLOBAL_COMPETITION_PROTOTYPE_NOT_HELPFUL",
                           "D-B2-D-B4 < 0.03 while D-B4 >= D-B0 + 0.05")
    elif no_gain["applies"]:
        verdict, reason = ("GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN",
                           f"D-B2 mIoU {d_b2:.4f} and its gains over D-B0 "
                           f"({d_b2 - miou['D-B0']:+.4f}) / D-B1 ({d_b2 - miou['D-B1']:+.4f}) miss the "
                           f"section-28 bars")
    elif mask_passed and not criteria_passed:
        verdict, reason = ("GLOBAL_COMPETITION_COUNTERFACTUAL_WEAK",
                           "the mask/ablation criteria pass but the paired/margin criteria fail")
    elif criteria_passed:
        verdict, reason = "RELATION_GUIDED_GLOBAL_COMPETITION_FEASIBLE", \
            "all section-28 criteria pass"
    else:
        verdict, reason = "GLOBAL_COMPETITION_NO_MEANINGFUL_GAIN", \
            "not all section-28 criteria pass and no other section-30 condition applies"

    payload = {
        "_doc": (
            "Task 7D sections 28-30. Oracle-reference relation-guided global competition: D-B0 is the "
            "frozen Task 6Z Z-B3 baseline (reproduced exactly) and D-B1/D-B2/D-B3/D-B4 are trained from "
            "fresh initialization on the exact frozen Task 6Z packs. The competition map is never "
            "supervised; the verdict follows the fixed section-30 priority order."
        ),
        "task": "7D", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS), "protocol": protocol,
        "variants": variant_report(),
        "baseline_reproduction": baseline,
        "overfit20": {"gate_passed": overfit["d_b2_overfit_passed"], "gate": overfit["gate"],
                      "results": {variant: {"best_miou": overfit["results"][variant]["best"]["miou"],
                                            "best_dice": overfit["results"][variant]["best"]["dice"],
                                            "params": overfit["results"][variant]["params"]}
                                  for variant in TRAINABLE_VARIANTS}},
        "training": {"results": {variant: {"best_miou": training["results"][variant]["best"]["miou"],
                                           "best_dice": training["results"][variant]["best"]["dice"],
                                           "best_epoch": training["results"][variant]["best"]["epoch"],
                                           "params": training["results"][variant]["params"],
                                           "wall_seconds": training["results"][variant]["wall_seconds"],
                                           "peak_vram_gb": training["results"][variant]["peak_vram_gb"]}
                                 for variant in TRAINABLE_VARIANTS}},
        "mini_val_240": {"miou": miou, "dice": dice,
                         "results": {variant: mini["results"][variant] for variant in ALL_VARIANTS}},
        "paired_val20": paired["results"],
        "competition_diagnostics": {variant: diagnostics["variants"][variant]["summary"]
                                    for variant in TRAINABLE_VARIANTS},
        "deltas": {"d_b2_minus_b0": d_b2 - miou["D-B0"], "d_b2_minus_b1": d_b2 - miou["D-B1"],
                   "d_b2_minus_b3": d_b2 - miou["D-B3"], "d_b2_minus_b4": d_b2 - miou["D-B4"],
                   "d_b1_minus_b0": miou["D-B1"] - miou["D-B0"],
                   "d_b3_minus_b0": miou["D-B3"] - miou["D-B0"],
                   "d_b4_minus_b0": miou["D-B4"] - miou["D-B0"]},
        "criteria": criteria, "criteria_passed": criteria_passed,
        "diagnostic_flags": flags,
        "visual_only_check": visual_only, "prototype_check": prototype_not_helpful,
        "no_gain_check": no_gain, "gate_constants": GATES,
        "interpretation_boundary": {
            "global_competition_claimed_as_novelty": False,
            "final_architecture_claimed": False,
            "predicted_reference_integrated": False,
            "parser_or_reference_changed": False,
            "graph_or_attention_added": False,
            "supervision_added_to_competition": False,
            "formal_full_data_training_started": False,
            "test_accessed": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 7D 的 oracle-reference global competition 因果结果决定是否替换 "
                           "Z-B3，不自行加入 attention/graph、predicted reference 或正式全量训练。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7d.report] D-B0 {miou['D-B0']:.4f} D-B1 {miou['D-B1']:.4f} D-B2 {miou['D-B2']:.4f} D-B3 "
          f"{miou['D-B3']:.4f} D-B4 {miou['D-B4']:.4f} | criteria "
          f"{sum(1 for entry in criteria.values() if entry['passed'])}/11 | flags {flags} -> {verdict}",
          flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
