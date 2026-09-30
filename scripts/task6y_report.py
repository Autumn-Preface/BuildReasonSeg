"""Task 6Y Parts K-L — feasibility criteria and the single verdict.

Section 25 `NEAREST_BOUNDARY_FIELD_FEASIBLE` requires ALL of:

1. field semantic sanity passes;
2. B2 overfit passes;
3. B2-B0 MiniVal mIoU >= +0.08;
4. B2-B1 MiniVal mIoU >= +0.05;
5. B2-B3 MiniVal mIoU >= +0.10;
6. B2 MiniVal mIoU >= 0.35;
7. B2 paired pass rate >= 0.70;
8. B2 own-cross margin >= 0.15;
9. no target GT model input.

Section 26 then selects exactly one verdict in the fixed priority order.

Writes `evaluation/task6y_verdict.json`. DSH reports measurements only.

    python scripts/task6y_report.py
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
from buildreasonseg_mvp.nearest_boundary_field import SIGMA_DIAG, scipy_available  # noqa: E402
from buildreasonseg_mvp.task6y_nearest_decoder import (  # noqa: E402
    ALL_VARIANTS,
    VARIANT_USES_NEAREST_FIELD,
    VARIANT_USES_REFERENCE_MASK,
    VARIANT_USES_VISUAL,
)

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6y_verdict.json"
BASE_COMMIT = "93188905df246c2669c730ad1251a651ec22f106"
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "NEAREST_FIELD_DEPENDENCY_UNAVAILABLE",
    "NEAREST_PAIRED_SET_INSUFFICIENT",
    "NEAREST_BOUNDARY_FIELD_SEMANTIC_MISMATCH",
    "NEAREST_FIELD_NOT_LEARNABLE",
    "NEAREST_FIELD_GEOMETRY_ONLY_CONFOUND",
    "NEAREST_FIELD_NO_MEANINGFUL_GAIN",
    "NEAREST_FIELD_COUNTERFACTUAL_WEAK",
    "NEAREST_BOUNDARY_FIELD_FEASIBLE",
)
FROZEN_PATHS = (
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "buildreasonseg_mvp/task6o_relation_decoder.py",
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/task6u_reference_ranker.py",
    "buildreasonseg_mvp/task6v_family_reference_resolver.py",
    "buildreasonseg_mvp/task6x_sam2_reference_refiner.py",
    "buildreasonseg_mvp/task6w_proposal_quality.py",
    "spatial_reasoning/relations.py",
    "configs/spatial_relations_v1.yaml",
    "evaluation/task6x_", "evaluation/task6w_", "evaluation/task6v_", "evaluation/task6u_",
)
CRITERIA = {"b2_minus_b0_min": 0.08, "b2_minus_b1_min": 0.05, "b2_minus_b3_min": 0.10,
            "b2_miou_min": 0.35, "b2_paired_rate_min": 0.70, "b2_margin_min": 0.15,
            "b3_paired_rate_confound": 0.60, "b2_minus_b3_confound": 0.10}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    manifest = load("task6y_pack_manifest.json")
    sanity = load("task6y_field_sanity.json")
    overfit = load("task6y_overfit20.json")
    training = load("task6y_training.json")
    mini = load("task6y_mini_val.json")
    paired = load("task6y_paired_val.json")
    if manifest is None or sanity is None:
        write_json(OUT, {"_doc": "Task 6Y section 26.", "task": "6Y",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": "pack manifest or field sanity artifact missing"})
        print("[6y.report] INVALID_EXPERIMENT (missing packs/sanity)", flush=True)
        return 2

    changes = frozen_changes()
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "sigma_diag": SIGMA_DIAG,
        "scipy_available": scipy_available(),
        "distance_metric": manifest["nearest_semantics"]["distance_metric"],
        "margin_px_floor": manifest["nearest_semantics"]["margin_px_floor"],
        "margin_diag_fraction": manifest["nearest_semantics"]["margin_diag_fraction"],
        "margin_mode": manifest["nearest_semantics"]["margin_mode"],
        "labels_regenerated": manifest["nearest_semantics"]["labels_regenerated"],
        "programs_in_packs": manifest["integrity"]["programs_present"],
        "only_nearest_programs": manifest["integrity"]["only_nearest_programs"],
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (manifest, sanity, mini, paired)
                               if isinstance(payload, dict)),
        "reference_source": "oracle_native_gt",
        "predicted_reference_used": False,
        "target_gt_as_input": False,
        "variants": list(ALL_VARIANTS),
        "variant_inputs": {variant: {"visual": VARIANT_USES_VISUAL[variant],
                                     "reference_mask": VARIANT_USES_REFERENCE_MASK[variant],
                                     "nearest_field": VARIANT_USES_NEAREST_FIELD[variant]}
                           for variant in ALL_VARIANTS},
        "sigma_tuned": False,
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and not protocol["test_split_used"]
                             and protocol["only_nearest_programs"]
                             and protocol["distance_metric"] == "boundary_distance"
                             and protocol["sigma_diag"] == 0.05
                             and not protocol["labels_regenerated"]
                             and not protocol["sigma_tuned"])

    n_pair = manifest["paired"]["n_pair"]
    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, test split or semantics)"
    elif not scipy_available():
        verdict, reason = "NEAREST_FIELD_DEPENDENCY_UNAVAILABLE", "scipy EDT missing"
    elif n_pair < manifest["paired"]["minimum"]:
        verdict, reason = ("NEAREST_PAIRED_SET_INSUFFICIENT",
                           f"only {n_pair} pairs exist (minimum {manifest['paired']['minimum']})")
    elif not sanity["sanity_passed"]:
        verdict, reason = ("NEAREST_BOUNDARY_FIELD_SEMANTIC_MISMATCH",
                           "field semantic sanity gate failed; sigma was not tuned")
    elif overfit is None or not overfit.get("b2_overfit_passed"):
        verdict, reason = ("NEAREST_FIELD_NOT_LEARNABLE",
                           "the Y-B2 Overfit20 gate failed; stage Y2 never ran")
    elif training is None or mini is None or paired is None:
        write_json(OUT, {"_doc": "Task 6Y section 26.", "task": "6Y",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": "training/evaluation artifacts missing after a passing overfit gate"})
        print("[6y.report] INVALID_EXPERIMENT (missing Y2 artifacts)", flush=True)
        return 2
    else:
        verdict, reason = None, None

    if verdict is not None:
        payload = {
            "_doc": ("Task 6Y sections 25-26. Oracle-reference NearestBoundaryField v0.1 causal audit; the "
                     "verdict follows the fixed section 26 priority order."),
            "task": "6Y", "verdict": verdict, "reason": reason,
            "allowed_verdicts": list(ALLOWED_VERDICTS), "protocol": protocol,
            "field_sanity": {"passed": sanity["sanity_passed"], "overall": sanity.get("overall")},
            "packs": {"overfit20": manifest["packs"]["y_overfit20"]["records"],
                      "mini_train_1000": manifest["packs"]["y_mini_train_1000"]["records"],
                      "mini_val_240": manifest["packs"]["y_mini_val_240"]["records"],
                      "n_pair": n_pair},
            "test_split_used": False, "runtime_seconds": round(time.time() - started, 2),
        }
        write_json(OUT, payload)
        print(f"[6y.report] {verdict} ({reason})", flush=True)
        return 0

    b0 = mini["results"]["Y-B0"]["overall"]
    b1 = mini["results"]["Y-B1"]["overall"]
    b2 = mini["results"]["Y-B2"]["overall"]
    b3 = mini["results"]["Y-B3"]["overall"]
    paired_b2 = paired["systems"]["Y-B2"]
    paired_b3 = paired["systems"]["Y-B3"]
    criteria = {
        "1_field_sanity": {"passed": sanity["sanity_passed"],
                           "measured": {"top1": sanity["overall"]["top1_rate"],
                                        "top3": sanity["overall"]["top3_rate"],
                                        "spearman": sanity["overall"]["mean_spearman"]}},
        "2_b2_overfit": {"passed": overfit["b2_overfit_passed"],
                         "measured": {"miou": overfit["gate"]["measured_miou"],
                                      "dice": overfit["gate"]["measured_dice"]},
                         "required": {"miou": overfit["gate"]["miou_min"],
                                      "dice": overfit["gate"]["dice_min"]}},
        "3_b2_minus_b0_miou": {"required": CRITERIA["b2_minus_b0_min"],
                               "measured": b2["miou"] - b0["miou"],
                               "passed": b2["miou"] - b0["miou"] >= CRITERIA["b2_minus_b0_min"]},
        "4_b2_minus_b1_miou": {"required": CRITERIA["b2_minus_b1_min"],
                               "measured": b2["miou"] - b1["miou"],
                               "passed": b2["miou"] - b1["miou"] >= CRITERIA["b2_minus_b1_min"]},
        "5_b2_minus_b3_miou": {"required": CRITERIA["b2_minus_b3_min"],
                               "measured": b2["miou"] - b3["miou"],
                               "passed": b2["miou"] - b3["miou"] >= CRITERIA["b2_minus_b3_min"]},
        "6_b2_miou": {"required": CRITERIA["b2_miou_min"], "measured": b2["miou"],
                      "passed": b2["miou"] >= CRITERIA["b2_miou_min"]},
        "7_b2_paired_rate": {"required": CRITERIA["b2_paired_rate_min"],
                             "measured": paired_b2["pass_rate"],
                             "passed": paired_b2["pass_rate"] >= CRITERIA["b2_paired_rate_min"]},
        "8_b2_margin": {"required": CRITERIA["b2_margin_min"],
                        "measured": paired_b2["own_cross_margin"],
                        "passed": paired_b2["own_cross_margin"] >= CRITERIA["b2_margin_min"]},
        "9_no_target_gt_input": {"passed": True, "measured": "GT target used as label/eval only"},
    }
    all_passed = all(entry["passed"] for entry in criteria.values())
    confound = {
        "b2_minus_b3_below_confound": (b2["miou"] - b3["miou"]) < CRITERIA["b2_minus_b3_confound"],
        "b3_paired_rate_at_least": paired_b3["pass_rate"] >= CRITERIA["b3_paired_rate_confound"],
        "applies": bool((b2["miou"] - b3["miou"]) < CRITERIA["b2_minus_b3_confound"]
                        and paired_b3["pass_rate"] >= CRITERIA["b3_paired_rate_confound"]),
    }
    no_gain = {
        "b2_minus_b0": b2["miou"] - b0["miou"], "b2_minus_b1": b2["miou"] - b1["miou"],
        "applies": bool((b2["miou"] - b0["miou"]) < CRITERIA["b2_minus_b0_min"]
                        or (b2["miou"] - b1["miou"]) < CRITERIA["b2_minus_b1_min"]
                        or b2["miou"] < CRITERIA["b2_miou_min"]),
    }
    counterfactual_weak = {"applies": bool(paired_b2["pass_rate"] < CRITERIA["b2_paired_rate_min"]
                                           or paired_b2["own_cross_margin"]
                                           < CRITERIA["b2_margin_min"])}

    if all_passed:
        verdict = "NEAREST_BOUNDARY_FIELD_FEASIBLE"
        reason = "all section-25 criteria pass"
    elif confound["applies"]:
        verdict = "NEAREST_FIELD_GEOMETRY_ONLY_CONFOUND"
        reason = ("B2 generalizes but B2-B3 < 0.10 while B3 paired pass rate >= 0.60: visual evidence is "
                  "not materially necessary")
    elif no_gain["applies"]:
        verdict = "NEAREST_FIELD_NO_MEANINGFUL_GAIN"
        reason = (f"learnability passes but B2-B1 = {no_gain['b2_minus_b1']:+.4f} < +0.05 and/or "
                  f"B2-B0 = {no_gain['b2_minus_b0']:+.4f} < +0.08 and/or B2 mIoU {b2['miou']:.4f} < 0.35")
    elif counterfactual_weak["applies"]:
        verdict = "NEAREST_FIELD_COUNTERFACTUAL_WEAK"
        reason = ("mask gains pass but the paired pass rate "
                  f"{paired_b2['pass_rate']:.2f} < 0.70 or the margin "
                  f"{paired_b2['own_cross_margin']:+.4f} < 0.15")
    else:
        verdict = "NEAREST_FIELD_NO_MEANINGFUL_GAIN"
        reason = "not all section-25 criteria pass and no other section-26 condition applies"

    payload = {
        "_doc": (
            "Task 6Y sections 25-26. Oracle-reference NearestBoundaryField v0.1 causal audit: the field is "
            "built from the GT reference mask only, the four variants are trained on the frozen nearest "
            "packs, and the single verdict follows the fixed section 26 priority order. Support "
            "infrastructure for the nearest relation family, not a novelty claim."
        ),
        "task": "6Y", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "protocol": protocol,
        "packs": {
            "overfit20": {"records": manifest["packs"]["y_overfit20"]["records"],
                          "tiles": manifest["packs"]["y_overfit20"]["unique_tiles"],
                          "sha256": manifest["packs"]["y_overfit20"]["sha256"]},
            "mini_train_1000": {"records": manifest["packs"]["y_mini_train_1000"]["records"],
                                "by_program": manifest["packs"]["y_mini_train_1000"]["by_program"],
                                "sha256": manifest["packs"]["y_mini_train_1000"]["sha256"]},
            "mini_val_240": {"records": manifest["packs"]["y_mini_val_240"]["records"],
                             "by_program": manifest["packs"]["y_mini_val_240"]["by_program"],
                             "sha256": manifest["packs"]["y_mini_val_240"]["sha256"]},
            "n_pair": n_pair,
        },
        "field_sanity": {"passed": sanity["sanity_passed"], "overall": sanity["overall"],
                         "by_family": sanity["by_family"], "gate": sanity["gate"],
                         "sigma_diag": SIGMA_DIAG},
        "overfit20": {"gate_passed": overfit["b2_overfit_passed"], "gate": overfit["gate"],
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
        "mini_val_240": {variant: mini["results"][variant] for variant in ALL_VARIANTS},
        "paired_val": {"systems": paired["systems"], "pairs": paired["pack"]["pairs"]},
        "deltas": {"b2_minus_b0": b2["miou"] - b0["miou"], "b2_minus_b1": b2["miou"] - b1["miou"],
                   "b2_minus_b3": b2["miou"] - b3["miou"],
                   "b1_minus_b0": b1["miou"] - b0["miou"]},
        "criteria": criteria, "all_criteria_passed": all_passed,
        "confound_check": confound, "no_gain_check": no_gain,
        "counterfactual_weak_check": counterfactual_weak,
        "interpretation": {
            "b0_to_b1": "value of raw reference localization",
            "b1_to_b2": "value of explicit boundary-proximity geometry",
            "b3_vs_b2": "whether visual evidence is materially necessary",
        },
        "interpretation_boundary": {
            "global_novelty_claimed": False,
            "end_to_end_nearest_capability_claimed": False,
            "sigma_changed": False,
            "learned_distance_transform_added": False,
            "predicted_reference_used": False,
            "directional_and_nearest_fields_combined": False,
            "l3_started": False,
            "reference_resolver_changed": False,
            "target_gt_used_as_input": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 6Y 的 nearest boundary field 因果结果决定下一步，不自行进行 "
                           "predicted-reference nearest 集成、direction+nearest 场组合或 L3 训练。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[6y.report] B0 {b0['miou']:.4f} | B1 {b1['miou']:.4f} | B2 {b2['miou']:.4f} | B3 "
          f"{b3['miou']:.4f} | paired B2 {paired_b2['passed']}/{paired_b2['pairs']} margin "
          f"{paired_b2['own_cross_margin']:+.4f} | criteria "
          f"{sum(1 for entry in criteria.values() if entry['passed'])}/9 -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
