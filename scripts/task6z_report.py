"""Task 6Z Parts K-L — causal criteria and the single verdict.

Section 28 learned two-field composition criteria (Z-B3): MiniVal Z-B3 mIoU >= 0.35; B3-B0 >= +0.10;
B3-B1 >= +0.05; B3-B2 >= +0.05; B3-B5 >= +0.10; B3 paired pass rate >= 0.70; B3 own-cross margin >= 0.15.

Section 29 deterministic product comparator criteria (Z-B4): the analogous seven conditions;
`deterministic_product_pass` is reported.

Section 30 learned-vs-product comparison: `delta_learned_vs_product = B3_mIoU - B4_mIoU` with the labels
`learned_not_worse` (B3 >= B4 - 0.03) and `product_materially_stronger` (B4 >= B3 + 0.05).

Section 31 then selects exactly one verdict in the fixed priority order.

Writes `evaluation/task6z_verdict.json`. DSH reports measurements only.

    python scripts/task6z_report.py
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
from buildreasonseg_mvp.task6z_field_composition import field_report  # noqa: E402
from buildreasonseg_mvp.task6z_l3_decoder import ALL_VARIANTS  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6z_verdict.json"
BASE_COMMIT = "24be95494a36e1a21d1f3566590c06ef118c7f28"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "L3_PAIRED_SET_INSUFFICIENT",
    "L3_COMPOSITION_FIELD_SEMANTIC_MISMATCH",
    "L3_LEARNED_COMPOSITION_NOT_LEARNABLE",
    "L3_GEOMETRY_ONLY_CONFOUND",
    "L3_DETERMINISTIC_COMPOSITION_ONLY",
    "L3_COMPOSITION_NO_MEANINGFUL_GAIN",
    "L3_COMPOSITION_COUNTERFACTUAL_WEAK",
    "L3_LEARNED_COMPOSITION_FEASIBLE",
)
FROZEN_PATHS = (
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/nearest_boundary_field.py",
    "buildreasonseg_mvp/task6y_nearest_decoder.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/program_parser.py",
    "spatial_reasoning/relations.py",
    "configs/spatial_relations_v1.yaml",
    "evaluation/task6y_", "evaluation/task6x_", "evaluation/task6w_",
)
CRITERIA = {"miou_min": 0.35, "vs_b0_min": 0.10, "vs_b1_min": 0.05, "vs_b2_min": 0.05,
            "vs_b5_min": 0.10, "paired_rate_min": 0.70, "margin_min": 0.15,
            "geometry_confound_vs_b5": 0.10, "geometry_confound_paired": 0.60,
            "learned_not_worse_slack": 0.03, "product_stronger_margin": 0.05}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def mask_criteria(results: dict, paired: dict, variant: str) -> dict:
    miou = {name: results[name]["overall"]["miou"] for name in ALL_VARIANTS}
    return {
        f"1_{variant}_miou": {"required": CRITERIA["miou_min"], "measured": miou[variant],
                              "passed": miou[variant] >= CRITERIA["miou_min"]},
        f"2_{variant}_minus_b0": {"required": CRITERIA["vs_b0_min"],
                                  "measured": miou[variant] - miou["Z-B0"],
                                  "passed": miou[variant] - miou["Z-B0"] >= CRITERIA["vs_b0_min"]},
        f"3_{variant}_minus_b1": {"required": CRITERIA["vs_b1_min"],
                                  "measured": miou[variant] - miou["Z-B1"],
                                  "passed": miou[variant] - miou["Z-B1"] >= CRITERIA["vs_b1_min"]},
        f"4_{variant}_minus_b2": {"required": CRITERIA["vs_b2_min"],
                                  "measured": miou[variant] - miou["Z-B2"],
                                  "passed": miou[variant] - miou["Z-B2"] >= CRITERIA["vs_b2_min"]},
        f"5_{variant}_minus_b5": {"required": CRITERIA["vs_b5_min"],
                                  "measured": miou[variant] - miou["Z-B5"],
                                  "passed": miou[variant] - miou["Z-B5"] >= CRITERIA["vs_b5_min"]},
        f"6_{variant}_paired_rate": {"required": CRITERIA["paired_rate_min"],
                                     "measured": paired[variant]["pass_rate"],
                                     "passed": paired[variant]["pass_rate"]
                                     >= CRITERIA["paired_rate_min"]},
        f"7_{variant}_margin": {"required": CRITERIA["margin_min"],
                                "measured": paired[variant]["own_cross_margin"],
                                "passed": paired[variant]["own_cross_margin"]
                                >= CRITERIA["margin_min"]},
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    manifest = load("task6z_pack_manifest.json")
    sanity = load("task6z_composition_sanity.json")
    overfit = load("task6z_overfit20.json")
    training = load("task6z_training.json")
    mini = load("task6z_mini_val.json")
    paired = load("task6z_paired_val.json")
    if manifest is None or sanity is None:
        write_json(OUT, {"_doc": "Task 6Z section 31.", "task": "6Z",
                         "verdict": "INVALID_EXPERIMENT", "reason": "pack manifest or sanity missing"})
        print("[6z.report] INVALID_EXPERIMENT (missing packs/sanity)", flush=True)
        return 2

    changes = frozen_changes()
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "programs": manifest["programs"], "program_count": len(manifest["programs"]),
        "only_l3_programs": manifest["integrity"]["only_l3_programs"],
        "l1_records": manifest["integrity"]["l1_records"],
        "l2_records": manifest["integrity"]["l2_records"],
        "smallest_l3_programs": manifest["integrity"]["smallest_l3_programs"],
        "canonical_operation_order": manifest["canonical_semantics"]["operation_order"],
        "directional_alpha": manifest["canonical_semantics"]["directional_alpha"],
        "directional_tau": manifest["canonical_semantics"]["directional_tau"],
        "distance_metric": manifest["canonical_semantics"]["distance_metric"],
        "nearest_margin_px_floor": manifest["canonical_semantics"]["nearest_margin_px_floor"],
        "nearest_margin_diag_fraction": manifest["canonical_semantics"]["nearest_margin_diag_fraction"],
        "labels_regenerated": manifest["canonical_semantics"]["labels_regenerated"],
        "sigma_diag": sanity["sigma_diag"],
        "constants_tuned": False,
        "variants": list(ALL_VARIANTS), "variant_count": len(ALL_VARIANTS),
        "n_pair": manifest["paired"]["n_pair"],
        "reference_source": "oracle_native_gt",
        "predicted_reference_used": False,
        "candidate_masks_as_input": False,
        "attention_or_transformer_or_gnn": False,
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (manifest, sanity, mini, paired)
                               if isinstance(payload, dict)),
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and not protocol["test_split_used"]
                             and protocol["only_l3_programs"] and protocol["variant_count"] == 6
                             and protocol["sigma_diag"] == 0.05
                             and protocol["distance_metric"] == "boundary_distance"
                             and not protocol["labels_regenerated"]
                             and protocol["canonical_operation_order"]
                             == ["argmax_area", "filter_relation", "argmin_boundary_distance"])

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, scope or semantics)"
    elif protocol["n_pair"] < manifest["paired"]["minimum"]:
        verdict, reason = ("L3_PAIRED_SET_INSUFFICIENT",
                           f"only {protocol['n_pair']} pairs exist (minimum "
                           f"{manifest['paired']['minimum']})")
    elif not sanity["sanity_passed"]:
        verdict, reason = ("L3_COMPOSITION_FIELD_SEMANTIC_MISMATCH",
                           "direction-valid composition sanity gate failed; constants were not tuned")
    elif overfit is None or not overfit.get("z_b3_overfit_passed"):
        verdict, reason = ("L3_LEARNED_COMPOSITION_NOT_LEARNABLE",
                           "the Z-B3 Overfit20 gate failed; stage Z2 never ran")
    elif training is None or mini is None or paired is None:
        write_json(OUT, {"_doc": "Task 6Z section 31.", "task": "6Z",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": "training/evaluation artifacts missing after a passing Z1 gate"})
        print("[6z.report] INVALID_EXPERIMENT (missing Z2 artifacts)", flush=True)
        return 2
    else:
        verdict, reason = None, None

    if verdict is not None:
        payload = {
            "_doc": ("Task 6Z sections 28-31. Oracle-reference L3 direction x nearest composition audit; "
                     "the verdict follows the fixed section 31 priority order."),
            "task": "6Z", "verdict": verdict, "reason": reason,
            "allowed_verdicts": list(ALLOWED_VERDICTS), "protocol": protocol,
            "composition_sanity": {"passed": sanity["sanity_passed"],
                                   "direction_valid": sanity["direction_valid_subset"]},
            "packs": {"overfit20": manifest["packs"]["z_overfit20"]["records"],
                      "mini_train_1200": manifest["packs"]["z_mini_train_1200"]["records"],
                      "mini_val_240": manifest["packs"]["z_mini_val_240"]["records"],
                      "n_pair": protocol["n_pair"]},
            "test_split_used": False, "runtime_seconds": round(time.time() - started, 2),
        }
        write_json(OUT, payload)
        print(f"[6z.report] {verdict} ({reason})", flush=True)
        return 0

    results = {variant: mini["results"][variant] for variant in ALL_VARIANTS}
    paired_systems = {variant: paired["systems"][variant] for variant in ALL_VARIANTS}
    b3_criteria = mask_criteria(results, paired_systems, "Z-B3")
    b4_criteria = mask_criteria(results, paired_systems, "Z-B4")
    b3_mask_passed = all(entry["passed"] for entry in b3_criteria.values())
    b4_passed = all(entry["passed"] for entry in b4_criteria.values())
    miou = {variant: results[variant]["overall"]["miou"] for variant in ALL_VARIANTS}
    geometry_confound = {
        "b3_minus_b5": miou["Z-B3"] - miou["Z-B5"],
        "b5_paired_pass_rate": paired_systems["Z-B5"]["pass_rate"],
        "applies": bool((miou["Z-B3"] - miou["Z-B5"]) < CRITERIA["geometry_confound_vs_b5"]
                        and paired_systems["Z-B5"]["pass_rate"]
                        >= CRITERIA["geometry_confound_paired"]),
    }
    delta_learned_vs_product = miou["Z-B3"] - miou["Z-B4"]
    comparison = {
        "delta_learned_vs_product": delta_learned_vs_product,
        "learned_not_worse": bool(miou["Z-B3"] >= miou["Z-B4"]
                                  - CRITERIA["learned_not_worse_slack"]),
        "product_materially_stronger": bool(miou["Z-B4"] >= miou["Z-B3"]
                                            + CRITERIA["product_stronger_margin"]),
    }
    b3_mask_only = all(entry["passed"] for name, entry in b3_criteria.items()
                       if not name.startswith(("6_", "7_")))
    b3_paired_only = all(entry["passed"] for name, entry in b3_criteria.items()
                         if name.startswith(("6_", "7_")))

    if b3_mask_passed:
        verdict = "L3_LEARNED_COMPOSITION_FEASIBLE"
        reason = "all section-28 criteria pass (learned two-field composition)"
    elif geometry_confound["applies"]:
        verdict = "L3_GEOMETRY_ONLY_CONFOUND"
        reason = (f"B3-B5 = {geometry_confound['b3_minus_b5']:+.4f} < +0.10 while B5's paired pass rate "
                  f"{geometry_confound['b5_paired_pass_rate']:.2f} >= 0.60")
    elif b4_passed:
        verdict = "L3_DETERMINISTIC_COMPOSITION_ONLY"
        reason = ("the learned two-field composition fails its criteria while the deterministic product "
                  "field passes all section-29 criteria")
    elif b3_mask_only and not b3_paired_only:
        verdict = "L3_COMPOSITION_COUNTERFACTUAL_WEAK"
        reason = (f"B3 mask criteria 1-5 pass but the paired pass rate "
                  f"{paired_systems['Z-B3']['pass_rate']:.2f} < 0.70 or the margin "
                  f"{paired_systems['Z-B3']['own_cross_margin']:+.4f} < 0.15")
    else:
        verdict = "L3_COMPOSITION_NO_MEANINGFUL_GAIN"
        reason = ("B3 mask criteria 1-5 fail and the deterministic product comparator does not pass the "
                  "section-29 criteria")

    payload = {
        "_doc": (
            "Task 6Z sections 28-31. Oracle-reference L3 direction x nearest composition audit: the "
            "deterministic product field is clamp(P_dir * P_near) with no renormalization, and the six "
            "predeclared variants are trained on the frozen L3 packs. The verdict follows the fixed "
            "section 31 priority order. Support infrastructure, not a novelty claim."
        ),
        "task": "6Z", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS), "protocol": protocol,
        "fields": field_report(),
        "composition_sanity": {
            "passed": sanity["sanity_passed"],
            "gate": sanity["gate"],
            "direction_valid_subset": sanity["direction_valid_subset"],
            "canonical_direction_valid_subset": sanity["canonical_direction_valid_subset"],
            "all_candidates": sanity["all_candidates"],
            "engine": sanity["dynamic_relation_engine"],
        },
        "packs": {
            "overfit20": {"records": manifest["packs"]["z_overfit20"]["records"],
                          "by_direction": manifest["packs"]["z_overfit20"]["by_direction"],
                          "tiles": manifest["packs"]["z_overfit20"]["unique_tiles"],
                          "sha256": manifest["packs"]["z_overfit20"]["sha256"]},
            "mini_train_1200": {"records": manifest["packs"]["z_mini_train_1200"]["records"],
                                "by_direction": manifest["packs"]["z_mini_train_1200"]["by_direction"],
                                "sha256": manifest["packs"]["z_mini_train_1200"]["sha256"]},
            "mini_val_240": {"records": manifest["packs"]["z_mini_val_240"]["records"],
                             "by_direction": manifest["packs"]["z_mini_val_240"]["by_direction"],
                             "sha256": manifest["packs"]["z_mini_val_240"]["sha256"]},
            "n_pair": protocol["n_pair"],
        },
        "overfit20": {"gate_passed": overfit["z_b3_overfit_passed"], "gate": overfit["gate"],
                      "results": {variant: {"best_miou": overfit["results"][variant]["best"]["miou"],
                                            "best_dice": overfit["results"][variant]["best"]["dice"],
                                            "params": overfit["results"][variant]["params"]}
                                  for variant in ALL_VARIANTS}},
        "training": {"selection_metric": training["selection_metric"],
                     "results": {variant: {"best_miou": training["results"][variant]["best"]["miou"],
                                           "best_dice": training["results"][variant]["best"]["dice"],
                                           "best_epoch": training["results"][variant]["best"]["epoch"],
                                           "params": training["results"][variant]["params"],
                                           "wall_seconds": training["results"][variant]["wall_seconds"],
                                           "peak_vram_gb": training["results"][variant]["peak_vram_gb"]}
                                 for variant in ALL_VARIANTS}},
        "mini_val_240": results,
        "paired_val": {"systems": paired_systems, "pairs": paired["pack"]["pairs"]},
        "deltas": {"b1_minus_b0": miou["Z-B1"] - miou["Z-B0"],
                   "b2_minus_b0": miou["Z-B2"] - miou["Z-B0"],
                   "b3_minus_b0": miou["Z-B3"] - miou["Z-B0"],
                   "b3_minus_b1": miou["Z-B3"] - miou["Z-B1"],
                   "b3_minus_b2": miou["Z-B3"] - miou["Z-B2"],
                   "b3_minus_b4": delta_learned_vs_product,
                   "b3_minus_b5": miou["Z-B3"] - miou["Z-B5"],
                   "b4_minus_b5": miou["Z-B4"] - miou["Z-B5"]},
        "criteria_z_b3": b3_criteria, "z_b3_criteria_passed": b3_mask_passed,
        "criteria_z_b4": b4_criteria, "deterministic_product_pass": b4_passed,
        "learned_vs_product": comparison,
        "geometry_confound_check": geometry_confound,
        "criteria_constants": CRITERIA,
        "interpretation_boundary": {
            "global_novelty_claimed": False,
            "final_l3_end_to_end_capability_claimed": False,
            "reference_subsystem_modified": False,
            "field_constants_tuned": False,
            "attention_or_global_competition_added": False,
            "predicted_reference_used": False,
            "program_head_retrained": False,
            "target_loss_changed": False,
            "learned_or_deterministic_selection_decided": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 6Z 的 L3 direction×nearest composition 因果结果决定下一步，"
                           "不自行进行 predicted-reference L3 集成、attention/global competition 改造或正式全量训练。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[6z.report] B0 {miou['Z-B0']:.4f} B1 {miou['Z-B1']:.4f} B2 {miou['Z-B2']:.4f} B3 "
          f"{miou['Z-B3']:.4f} B4 {miou['Z-B4']:.4f} B5 {miou['Z-B5']:.4f} | B3 criteria "
          f"{sum(1 for e in b3_criteria.values() if e['passed'])}/7 B4 pass {b4_passed} | "
          f"delta_learned_vs_product {delta_learned_vs_product:+.4f} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
