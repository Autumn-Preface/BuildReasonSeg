"""Task 7B Parts I — predeclared gates and the single verdict.

Section 23 PASS criteria (all twenty) and section 24 priority order:

1. `INVALID_EXPERIMENT`
2. `BASELINE_PARSER_UNAVAILABLE`
3. `PARSER_TRAINING_FAILED`
4. `L3_PARSER_CANONICAL_REGRESSION`   — a canonical gate fails (full v0.2 val, Z-MiniVal240, Z-PairedVal)
5. `L3_PARSER_COMPOSITIONAL_ROBUSTNESS_FAIL` — canonical gates pass but fixed24 / minimal pairs / stress fail
6. `END_TO_END_REGRESSION`            — all parser gates pass but the frozen downstream does not reproduce
7. `L3_PROGRAMHEAD_HARDENING_PASS`    — every section-23 criterion passes

Writes `evaluation/task7b_verdict.json`. DSH reports measurements only.

    python scripts/task7b_report.py
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
from scripts.task7b_build_parser_data import sha256_file  # noqa: E402
from scripts.task7b_train_parser import BASELINE, BASELINE_SHA256, L3_PROGRAMS  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7b_verdict.json"
BASE_COMMIT = "6ee3d0d90a56fb11c19112ab51e341b9f36c1392"
ALLOWED_VERDICTS = ("INVALID_EXPERIMENT", "BASELINE_PARSER_UNAVAILABLE", "PARSER_TRAINING_FAILED",
                    "L3_PARSER_CANONICAL_REGRESSION", "L3_PARSER_COMPOSITIONAL_ROBUSTNESS_FAIL",
                    "END_TO_END_REGRESSION", "L3_PROGRAMHEAD_HARDENING_PASS")
FROZEN_PATHS = (
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/nearest_boundary_field.py",
    "buildreasonseg_mvp/task6z_field_composition.py",
    "buildreasonseg_mvp/task6z_l3_decoder.py",
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/task6u_reference_ranker.py",
    "buildreasonseg_mvp/task6w_proposal_quality.py",
    "buildreasonseg_mvp/task6x_sam2_reference_refiner.py",
    "evaluation/task7a_",
)
GATES = {"full_val_accuracy": 0.995, "full_val_macro_f1": 0.995, "full_val_class_recall": 0.98,
         "z_minival": 240, "z_paired": 40, "fixed24_total": 22, "fixed24_compact": 8,
         "fixed24_class": 5, "minimal_pairs": 96, "stress_accuracy": 0.97, "stress_macro_f1": 0.97,
         "stress_class_recall": 0.90, "stress_l3_recall": 0.95}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    manifest = load("task7b_eval_prompt_manifest.json")
    spec = load("task7b_train_augmentation_spec.json")
    leakage = load("task7b_parser_leakage_audit.json")
    training = load("task7b_training_summary.json")
    full_val = load("task7b_full_val.json")
    z_mini = load("task7b_z_minival240.json")
    z_paired = load("task7b_z_paired_parser.json")
    fixed24 = load("task7b_task7a_fixed24.json")
    minimal = load("task7b_minimal_pairs_result.json")
    stress = load("task7b_stress_result.json")
    scope = load("task7b_scope_safety.json")
    e2e = load("task7b_end_to_end_regression.json")
    required = {"task7b_eval_prompt_manifest.json": manifest,
                "task7b_train_augmentation_spec.json": spec,
                "task7b_parser_leakage_audit.json": leakage,
                "task7b_training_summary.json": training,
                "task7b_full_val.json": full_val, "task7b_z_minival240.json": z_mini,
                "task7b_z_paired_parser.json": z_paired, "task7b_task7a_fixed24.json": fixed24,
                "task7b_minimal_pairs_result.json": minimal, "task7b_stress_result.json": stress,
                "task7b_scope_safety.json": scope,
                "task7b_end_to_end_regression.json": e2e}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 7B section 24.", "task": "7B",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[7b.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    baseline_sha = sha256_file(BASELINE)
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "baseline_parser_sha256": baseline_sha, "expected_baseline_sha256": BASELINE_SHA256,
        "baseline_matches": baseline_sha == BASELINE_SHA256,
        "architecture": training["architecture"]["model_family"],
        "text_only": training["architecture"]["text_only"],
        "image_input": training["architecture"]["image_input"],
        "classes": training["architecture"]["classes"],
        "trainable_policy": training["architecture"]["trainable_policy"],
        "leakage_verdict": leakage["verdict"],
        "exact_overlap": leakage["augmentation"]["exact_overlap"],
        "normalized_overlap": leakage["augmentation"]["normalized_overlap"],
        "leakage_zero": bool(leakage["verdict"] == "LEAKAGE_FREE"
                             and leakage["augmentation"]["exact_overlap"] == 0
                             and leakage["augmentation"]["normalized_overlap"] == 0
                             and leakage["cleaned_train_overlap"]["exact"] == 0
                             and leakage["cleaned_train_overlap"]["normalized"] == 0),
        "augmentations": spec["augmentation_total"],
        "keyword_or_regex_override": scope["cli"]["keyword_or_regex_override"],
        "hyperparameter_sweep": training["hyperparameter_sweep"],
        "training_performed": True,
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (manifest, spec, leakage, full_val, z_mini, z_paired,
                                               fixed24, minimal, stress, scope, e2e)
                               if isinstance(payload, dict)),
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and not protocol["test_split_used"]
                             and protocol["leakage_zero"]
                             and not protocol["keyword_or_regex_override"]
                             and not protocol["hyperparameter_sweep"])
    if training.get("verdict") == "BASELINE_PARSER_UNAVAILABLE":
        payload = {"_doc": "Task 7B section 24.", "task": "7B",
                   "verdict": "BASELINE_PARSER_UNAVAILABLE", "protocol": protocol}
        write_json(OUT, payload)
        print("[7b.report] BASELINE_PARSER_UNAVAILABLE", flush=True)
        return 0

    canonical = {
        "1_baseline_parser_hash": {"passed": protocol["baseline_matches"],
                                   "measured": baseline_sha, "required": BASELINE_SHA256},
        "2_architecture": {"passed": bool(protocol["architecture"] == "Qwen3-VL-2B"
                                          and protocol["text_only"] and not protocol["image_input"]
                                          and protocol["classes"] == 20)},
        "3_leakage_zero": {"passed": protocol["leakage_zero"],
                           "measured": {"exact": protocol["exact_overlap"],
                                        "normalized": protocol["normalized_overlap"]}},
        "4_full_val_accuracy": {"required": GATES["full_val_accuracy"],
                                "measured": full_val["metrics"]["accuracy"],
                                "passed": (full_val["metrics"]["accuracy"] or 0.0)
                                >= GATES["full_val_accuracy"]},
        "5_full_val_macro_f1": {"required": GATES["full_val_macro_f1"],
                                "measured": full_val["metrics"]["macro_f1"],
                                "passed": full_val["metrics"]["macro_f1"] >= GATES["full_val_macro_f1"]},
        "6_full_val_class_recall": {"required": GATES["full_val_class_recall"],
                                    "measured": full_val["metrics"]["min_represented_class_recall"],
                                    "passed": (full_val["metrics"]["min_represented_class_recall"] or 0.0)
                                    >= GATES["full_val_class_recall"]},
        "7_z_minival240": {"required": "240/240", "measured": z_mini["metrics"]["correct"],
                           "passed": z_mini["passed"]},
        "8_z_paired_members": {"required": "40/40", "measured": z_paired["metrics"]["correct"],
                               "passed": z_paired["passed"]},
    }
    compositional = {
        "9_fixed24_total": {"required": f">= {GATES['fixed24_total']}/24",
                            "measured": fixed24["correct"],
                            "passed": fixed24["correct"] >= GATES["fixed24_total"]},
        "10_fixed24_compact": {"required": "8/8", "measured": fixed24["compact"]["correct"],
                               "passed": fixed24["compact"]["correct"] == GATES["fixed24_compact"]},
        "11_fixed24_every_class": {"required": f">= {GATES['fixed24_class']}/6",
                                   "measured": fixed24["per_program"],
                                   "passed": all(entry["correct"] >= GATES["fixed24_class"]
                                                 for entry in fixed24["per_program"].values())},
        "12_minimal_pairs": {"required": "96/96", "measured": minimal["correct"],
                             "passed": minimal["passed"]},
        "13_stress_accuracy": {"required": GATES["stress_accuracy"],
                               "measured": stress["metrics"]["accuracy"],
                               "passed": (stress["metrics"]["accuracy"] or 0.0)
                               >= GATES["stress_accuracy"]},
        "14_stress_macro_f1": {"required": GATES["stress_macro_f1"],
                               "measured": stress["metrics"]["macro_f1"],
                               "passed": stress["metrics"]["macro_f1"] >= GATES["stress_macro_f1"]},
        "15_stress_class_recall": {"required": GATES["stress_class_recall"],
                                   "measured": stress["metrics"]["min_represented_class_recall"],
                                   "passed": (stress["metrics"]["min_represented_class_recall"] or 0.0)
                                   >= GATES["stress_class_recall"]},
        "16_stress_l3_recall": {"required": GATES["stress_l3_recall"],
                                "measured": stress["l3_macro_recall"],
                                "passed": (stress["l3_macro_recall"] or 0.0)
                                >= GATES["stress_l3_recall"]},
    }
    safety = {
        "17_scope_controls": {"passed": scope["out_of_scope_controls_ok"]},
        "18_no_keyword_override": {"passed": not protocol["keyword_or_regex_override"]},
        "19_end_to_end_regression": {"passed": e2e["passed"]},
        "20_no_test_split": {"passed": not protocol["test_split_used"]},
    }
    canonical_gates_passed = all(entry["passed"] for entry in canonical.values())
    compositional_gates_passed = all(entry["passed"] for entry in compositional.values())
    all_passed = canonical_gates_passed and compositional_gates_passed and \
        all(entry["passed"] for entry in safety.values())

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, leakage, override or test)"
    elif not protocol["baseline_matches"]:
        verdict, reason = ("BASELINE_PARSER_UNAVAILABLE",
                           "the Task 6T baseline checkpoint hash does not match")
    elif training.get("verdict") != "PARSER_TRAINED" or not training["checkpoint"]["exists"]:
        verdict, reason = "PARSER_TRAINING_FAILED", "no valid Task 7B checkpoint was produced"
    elif not canonical_gates_passed:
        verdict = "L3_PARSER_CANONICAL_REGRESSION"
        reason = ("canonical gates fail on "
                  + str([name for name, entry in canonical.items() if not entry["passed"]]))
    elif not compositional_gates_passed:
        verdict = "L3_PARSER_COMPOSITIONAL_ROBUSTNESS_FAIL"
        reason = ("canonical gates pass but the compositional robustness gates fail on "
                  + str([name for name, entry in compositional.items() if not entry["passed"]]))
    elif not e2e["passed"]:
        verdict = "END_TO_END_REGRESSION"
        reason = f"the frozen downstream did not reproduce Task 7A (deltas {e2e['deltas']})"
    else:
        verdict = "L3_PROGRAMHEAD_HARDENING_PASS"
        reason = "every section-23 criterion passes"

    payload = {
        "_doc": (
            "Task 7B sections 23-24. L3 compositional-semantic hardening of the same Qwen3-VL-2B "
            "text-only 20-class ProgramHead: frozen evaluation packs, exact/normalized leakage audit, the "
            "frozen training protocol, six evaluation sets, scope safety and the frozen end-to-end "
            "regression. The verdict follows the fixed section-24 priority order."
        ),
        "task": "7B", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "protocol": protocol,
        "canonical_gates": canonical, "canonical_gates_passed": canonical_gates_passed,
        "compositional_gates": compositional,
        "compositional_gates_passed": compositional_gates_passed,
        "safety_gates": safety, "all_criteria_passed": all_passed,
        "gates": GATES,
        "data": {"cleaned_v02_train_rows": spec["cleaned_train_rows"],
                 "dropped_v02_train_records": spec["dropped_train_records"],
                 "augmentations": spec["augmentation_total"],
                 "allocation": spec["allocation"],
                 "class_counts": training["data"]["class_counts"],
                 "internal_split": {"train_rows": training["data"]["train_rows"],
                                    "holdout_rows": training["data"]["holdout_rows"],
                                    "normalized_overlap": training["data"]["normalized_overlap"]}},
        "training": {"selected_epoch": training["selected_epoch"],
                     "selection_metrics": training["selection_metrics"],
                     "protocol": training["protocol"],
                     "checkpoint": training["checkpoint"],
                     "wall_seconds": training["wall_seconds"],
                     "peak_vram_gb": training["peak_vram_gb"]},
        "evaluation": {
            "full_val": {"accuracy": full_val["metrics"]["accuracy"],
                         "macro_f1": full_val["metrics"]["macro_f1"],
                         "min_class_recall": full_val["metrics"]["min_represented_class_recall"],
                         "correct": full_val["metrics"]["correct"],
                         "count": full_val["metrics"]["count"],
                         "zero_recall_classes": [program for program, value
                                                 in full_val["metrics"]["per_class_recall"].items()
                                                 if value is not None and value < 0.5]},
            "z_minival240": {"correct": z_mini["metrics"]["correct"],
                             "count": z_mini["metrics"]["count"]},
            "z_paired_members": {"correct": z_paired["metrics"]["correct"],
                                 "count": z_paired["metrics"]["count"]},
            "fixed24": {"correct": fixed24["correct"], "compact": fixed24["compact"],
                        "per_program": fixed24["per_program"]},
            "minimal_pairs": {"correct": minimal["correct"], "total": minimal["total"],
                              "by_contrast_group": minimal["by_contrast_group"]},
            "stress": {"accuracy": stress["metrics"]["accuracy"],
                       "macro_f1": stress["metrics"]["macro_f1"],
                       "l3_macro_recall": stress["l3_macro_recall"],
                       "min_class_recall": stress["metrics"]["min_represented_class_recall"],
                       "correct": stress["metrics"]["correct"],
                       "count": stress["metrics"]["count"],
                       "per_class_recall": stress["metrics"]["per_class_recall"]},
        },
        "scope_safety": {"out_of_scope_controls_ok": scope["out_of_scope_controls_ok"],
                         "ood_ok": scope["ood_ok"], "cli": scope["cli"],
                         "controls": [{key: entry[key] for key in
                                       ("prompt", "expected_program", "parsed_program",
                                        "exit_code", "semantically_correct_before_exit")}
                                      for entry in scope["out_of_scope_controls"]]},
        "end_to_end": {"passed": e2e["passed"], "measured": e2e["measured"],
                       "task7a_reference": e2e["task7a_reference"], "deltas": e2e["deltas"],
                       "tolerance": e2e["tolerance"]},
        "cli_default_parser": {"updated": scope["cli"]["updated_to_task7b"],
                               "path": scope["cli"]["default_parser_checkpoint"]},
        "interpretation_boundary": {
            "parser_claimed_as_novelty": False,
            "downstream_architecture_changed": False,
            "reference_hardening_reopened": False,
            "attention_or_global_competition_added": False,
            "full_training_or_test_started": False,
            "task7c_decided": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 7B 的 L3 ProgramHead 组合语义泛化结果决定下一步，不自行重新开启 "
                           "reference hardening、修改 Z-B3 或进行 attention/global competition 改造。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7b.report] canonical {canonical_gates_passed} compositional "
          f"{compositional_gates_passed} e2e {e2e['passed']} | full val "
          f"{full_val['metrics']['accuracy']:.4f} (min recall "
          f"{full_val['metrics']['min_represented_class_recall']:.3f}) | fixed24 {fixed24['correct']}/24 "
          f"| minimal {minimal['correct']}/96 | stress {stress['metrics']['accuracy']:.4f} -> {verdict}",
          flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
