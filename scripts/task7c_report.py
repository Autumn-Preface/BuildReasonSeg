"""Task 7C Parts M-N — canonical gate check, CLI default decision and the single verdict.

Section 23 PASS criteria (all 26) and section 24 priority order:

1. `INVALID_EXPERIMENT`
2. `BASELINE_PARSER_UNAVAILABLE`
3. `PARSER_TRAINING_FAILED`
4. `L3_20CLASS_CANONICAL_REGRESSION`      — full val / Z-MiniVal240 / Z-Paired gate fails
5. `L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL` — canonical gates pass, compositional gates fail
6. `END_TO_END_REGRESSION`                — parser gates pass but the frozen downstream does not reproduce
7. `L3_20CLASS_REHEARSAL_PASS`            — every section-23 criterion passes

Section 24 CLI default: move the L3 default parser to Task 7C only when the canonical gates pass.

Writes `evaluation/task7c_verdict.json`.

    python scripts/task7c_report.py
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
from scripts.task7c_build_rehearsal import ALL_CLASSES, L3_PROGRAMS  # noqa: E402
from scripts.task7c_train_parser import BASELINE, BASELINE_SHA256, TASK7B_CHECKPOINT  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7c_verdict.json"
BASE_COMMIT = "b325585c18ddef1ee80cf05d4fc971c5c19b477a"
ALLOWED_VERDICTS = ("INVALID_EXPERIMENT", "BASELINE_PARSER_UNAVAILABLE", "PARSER_TRAINING_FAILED",
                    "L3_20CLASS_CANONICAL_REGRESSION", "L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL",
                    "END_TO_END_REGRESSION", "L3_20CLASS_REHEARSAL_PASS")
FROZEN_PATHS = (
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/nearest_boundary_field.py",
    "buildreasonseg_mvp/task6z_field_composition.py",
    "buildreasonseg_mvp/task6z_l3_decoder.py",
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/task6u_reference_ranker.py",
    "buildreasonseg_mvp/task6w_proposal_quality.py",
    "buildreasonseg_mvp/task6x_sam2_reference_refiner.py",
    "evaluation/task7a_", "evaluation/task7b_",
)
GATES = {"full_val_accuracy": 0.995, "full_val_macro_f1": 0.995, "full_val_class_recall": 0.98,
         "z_minival": 240, "z_paired": 40, "fixed24_total": 22, "fixed24_compact": 8,
         "fixed24_class": 5, "minimal_pairs": 96, "stress_accuracy": 0.97, "stress_macro_f1": 0.97,
         "stress_class_recall": 0.90, "stress_l3_recall": 0.95, "rehearsal_rows": 8000}


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    reused = load("task7c_reused_eval_manifest.json")
    spec = load("task7c_rehearsal_spec.json")
    audit = load("task7c_training_data_audit.json")
    training = load("task7c_training_summary.json")
    full_val = load("task7c_full_val.json")
    z_mini = load("task7c_z_minival240.json")
    z_paired = load("task7c_z_paired_parser.json")
    fixed24 = load("task7c_task7a_fixed24.json")
    minimal = load("task7c_minimal_pairs_result.json")
    stress = load("task7c_stress_result.json")
    scope = load("task7c_scope_safety.json")
    e2e = load("task7c_end_to_end_regression.json")
    required = {"task7c_reused_eval_manifest.json": reused, "task7c_rehearsal_spec.json": spec,
                "task7c_training_data_audit.json": audit,
                "task7c_training_summary.json": training, "task7c_full_val.json": full_val,
                "task7c_z_minival240.json": z_mini, "task7c_z_paired_parser.json": z_paired,
                "task7c_task7a_fixed24.json": fixed24,
                "task7c_minimal_pairs_result.json": minimal, "task7c_stress_result.json": stress,
                "task7c_scope_safety.json": scope,
                "task7c_end_to_end_regression.json": e2e}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 7C section 26.", "task": "7C",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[7c.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    baseline_sha = sha256_file(BASELINE)
    task7b_sha = sha256_file(TASK7B_CHECKPOINT) if TASK7B_CHECKPOINT.is_file() else None
    class_counts = audit["rehearsal"]["by_class"]
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "baseline_sha256": baseline_sha, "expected_baseline_sha256": BASELINE_SHA256,
        "baseline_matches": baseline_sha == BASELINE_SHA256,
        "initialized_from_task6t": bool(training["baseline"]["sha256"] == BASELINE_SHA256),
        "initialized_from_task7b": bool(training["checkpoint"]["sha256"] == task7b_sha)
        if task7b_sha else False,
        "task7b_checkpoint_sha256": task7b_sha,
        "new_checkpoint_sha256": training["checkpoint"]["sha256"],
        "checkpoint_differs_from_task7b": bool(task7b_sha is None
                                               or training["checkpoint"]["sha256"] != task7b_sha),
        "architecture": training["architecture"]["model_family"],
        "text_only": training["architecture"]["text_only"],
        "image_input": training["architecture"]["image_input"],
        "classes": training["architecture"]["classes"],
        "backbone_unfrozen": training["architecture"]["backbone_unfrozen"],
        "rehearsal_rows": audit["rehearsal"]["rows"],
        "all_classes_represented": all(class_counts.get(program, 0) > 0 for program in ALL_CLASSES),
        "exact_overlap": audit["rehearsal"]["exact_overlap"],
        "normalized_overlap": audit["rehearsal"]["normalized_overlap"],
        "normalized_duplicates": audit["rehearsal"]["normalized_duplicates"],
        "per_class_counts_exact": audit["rehearsal"]["per_class_counts_exact"],
        "holdout_classes_without_support": training["data"]["classes_without_holdout_support"],
        "holdout_normalized_overlap": training["data"]["normalized_overlap"],
        "external_eval_used_for_selection": training["data"]["external_eval_used_for_selection"],
        "original_v02_train_text_used": training["data"]["original_v02_train_text_used"],
        "lora_lr": training["protocol"]["lora_lr"],
        "program_head_lr": training["protocol"]["program_head_lr"],
        "sweep": training["protocol"]["sweep"],
        "keyword_or_regex_override": scope["cli"]["keyword_or_regex_override"],
        "test_split_used": any(payload.get("test_split_used", False)
                               for payload in (reused, spec, audit, full_val, z_mini, z_paired,
                                               fixed24, minimal, stress, scope, e2e)
                               if isinstance(payload, dict)),
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and not protocol["test_split_used"]
                             and protocol["audit_clean"] if False else
                             (protocol["frozen_paths_unchanged"] and not protocol["test_split_used"]
                              and audit["verdict"] == "REHEARSAL_CLEAN"
                              and not protocol["keyword_or_regex_override"]
                              and not protocol["sweep"]
                              and not protocol["external_eval_used_for_selection"]
                              and not protocol["original_v02_train_text_used"]
                              and not protocol["initialized_from_task7b"]
                              and not protocol["backbone_unfrozen"]))

    canonical = {
        "1_baseline_parser_hash": {"passed": protocol["baseline_matches"], "measured": baseline_sha,
                                   "required": BASELINE_SHA256},
        "2_initialized_from_task6t": {"passed": protocol["initialized_from_task6t"]
                                      and not protocol["initialized_from_task7b"]},
        "3_architecture": {"passed": bool(protocol["architecture"] == "Qwen3-VL-2B"
                                          and protocol["text_only"] and not protocol["image_input"]
                                          and protocol["classes"] == 20)},
        "4_rehearsal_rows": {"required": GATES["rehearsal_rows"], "measured": protocol["rehearsal_rows"],
                             "passed": protocol["rehearsal_rows"] == GATES["rehearsal_rows"]},
        "5_all_classes_represented": {"passed": protocol["all_classes_represented"],
                                      "measured": class_counts},
        "6_exact_zh_en_allocation": {"passed": protocol["per_class_counts_exact"]},
        "7_exact_overlap_zero": {"passed": protocol["exact_overlap"] == 0},
        "8_normalized_overlap_zero": {"passed": protocol["normalized_overlap"] == 0},
        "9_holdout_all_classes": {"passed": not protocol["holdout_classes_without_support"]
                                  and protocol["holdout_normalized_overlap"] == 0},
        "10_full_val_accuracy": {"required": GATES["full_val_accuracy"],
                                 "measured": full_val["metrics"]["accuracy"],
                                 "passed": (full_val["metrics"]["accuracy"] or 0.0)
                                 >= GATES["full_val_accuracy"]},
        "11_full_val_macro_f1": {"required": GATES["full_val_macro_f1"],
                                 "measured": full_val["metrics"]["macro_f1"],
                                 "passed": full_val["metrics"]["macro_f1"] >= GATES["full_val_macro_f1"]},
        "12_full_val_class_recall": {"required": GATES["full_val_class_recall"],
                                     "measured": full_val["metrics"]["min_represented_class_recall"],
                                     "passed": (full_val["metrics"]["min_represented_class_recall"] or 0.0)
                                     >= GATES["full_val_class_recall"]},
        "13_z_minival240": {"required": "240/240", "measured": z_mini["metrics"]["correct"],
                            "passed": z_mini["passed"]},
        "14_z_paired_members": {"required": "40/40", "measured": z_paired["metrics"]["correct"],
                                "passed": z_paired["passed"]},
    }
    compositional = {
        "15_fixed24_total": {"required": f">= {GATES['fixed24_total']}/24",
                             "measured": fixed24["correct"],
                             "passed": fixed24["correct"] >= GATES["fixed24_total"]},
        "16_fixed24_compact": {"required": "8/8", "measured": fixed24["compact"]["correct"],
                               "passed": fixed24["compact"]["correct"] == GATES["fixed24_compact"]},
        "17_fixed24_every_class": {"required": f">= {GATES['fixed24_class']}/6",
                                   "measured": fixed24["per_program"],
                                   "passed": all(entry["correct"] >= GATES["fixed24_class"]
                                                 for entry in fixed24["per_program"].values())},
        "18_minimal96": {"required": "96/96", "measured": minimal["correct"],
                         "passed": minimal["passed"]},
        "19_stress_accuracy": {"required": GATES["stress_accuracy"],
                               "measured": stress["metrics"]["accuracy"],
                               "passed": (stress["metrics"]["accuracy"] or 0.0)
                               >= GATES["stress_accuracy"]},
        "20_stress_macro_f1": {"required": GATES["stress_macro_f1"],
                               "measured": stress["metrics"]["macro_f1"],
                               "passed": stress["metrics"]["macro_f1"] >= GATES["stress_macro_f1"]},
        "21_stress_class_recall": {"required": GATES["stress_class_recall"],
                                   "measured": stress["metrics"]["min_represented_class_recall"],
                                   "passed": (stress["metrics"]["min_represented_class_recall"] or 0.0)
                                   >= GATES["stress_class_recall"]},
        "22_stress_l3_recall": {"required": GATES["stress_l3_recall"],
                                "measured": stress["l3_macro_recall"],
                                "passed": (stress["l3_macro_recall"] or 0.0)
                                >= GATES["stress_l3_recall"]},
    }
    safety = {
        "23_scope_safety": {"passed": scope["passed"]},
        "24_no_keyword_override": {"passed": not protocol["keyword_or_regex_override"]},
        "25_downstream_reproduction": {"passed": e2e["passed"]},
        "26_no_test": {"passed": not protocol["test_split_used"]},
    }
    canonical_passed = all(entry["passed"] for entry in canonical.values())
    compositional_passed = all(entry["passed"] for entry in compositional.values())
    all_passed = canonical_passed and compositional_passed and \
        all(entry["passed"] for entry in safety.values())

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", "protocol violation (frozen path, leakage, override or test)"
    elif not protocol["baseline_matches"]:
        verdict, reason = ("BASELINE_PARSER_UNAVAILABLE",
                           "the Task 6T baseline checkpoint hash does not match")
    elif training.get("verdict") != "PARSER_TRAINED" or not training["checkpoint"]["exists"]:
        verdict, reason = "PARSER_TRAINING_FAILED", "no valid Task 7C checkpoint was produced"
    elif not canonical_passed:
        verdict = "L3_20CLASS_CANONICAL_REGRESSION"
        reason = ("canonical gates fail on "
                  + str([name for name, entry in canonical.items() if not entry["passed"]]))
    elif not compositional_passed:
        verdict = "L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL"
        reason = ("canonical gates pass but the compositional robustness gates fail on "
                  + str([name for name, entry in compositional.items() if not entry["passed"]]))
    elif not e2e["passed"]:
        verdict = "END_TO_END_REGRESSION"
        reason = f"the frozen downstream did not reproduce Task 7A (deltas {e2e['deltas']})"
    else:
        verdict = "L3_20CLASS_REHEARSAL_PASS"
        reason = "every section-23 criterion passes"

    payload = {
        "_doc": (
            "Task 7C sections 23-26. 20-class rehearsal hardening of the same Qwen3-VL-2B text-only "
            "20-class ProgramHead, restarted from the stable Task 6T checkpoint with fixed differential "
            "learning rates, evaluated once on the reused frozen sets, with scope safety and the frozen "
            "end-to-end regression. The verdict follows the fixed section-26 priority order."
        ),
        "task": "7C", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS), "protocol": protocol,
        "canonical_gates": canonical, "canonical_gates_passed": canonical_passed,
        "compositional_gates": compositional, "compositional_gates_passed": compositional_passed,
        "safety_gates": safety, "all_criteria_passed": all_passed, "gates": GATES,
        "rehearsal": {"rows": spec["total_rows"], "allocation": spec["allocation"],
                      "by_class": class_counts,
                      "by_class_language": audit["rehearsal"]["by_class_language"],
                      "unique_normalized": audit["rehearsal"]["unique_normalized"],
                      "normalized_duplicates": audit["rehearsal"]["normalized_duplicates"],
                      "exact_overlap": audit["rehearsal"]["exact_overlap"],
                      "normalized_overlap": audit["rehearsal"]["normalized_overlap"],
                      "original_v02_train_text_used": False},
        "training": {"selected_epoch": training["selected_epoch"],
                     "selection_metrics": training["selection_metrics"],
                     "protocol": training["protocol"], "checkpoint": training["checkpoint"],
                     "wall_seconds": training["wall_seconds"],
                     "peak_vram_gb": training["peak_vram_gb"],
                     "internal_split": {"train_rows": training["data"]["train_rows"],
                                        "holdout_rows": training["data"]["holdout_rows"],
                                        "normalized_overlap": training["data"]["normalized_overlap"],
                                        "holdout_support": training["data"]["holdout_support"]}},
        "evaluation": {
            "full_val": {"accuracy": full_val["metrics"]["accuracy"],
                         "macro_f1": full_val["metrics"]["macro_f1"],
                         "min_class_recall": full_val["metrics"]["min_represented_class_recall"],
                         "correct": full_val["metrics"]["correct"],
                         "count": full_val["metrics"]["count"],
                         "per_class_recall": full_val["metrics"]["per_class_recall"]},
            "z_minival240": {"correct": z_mini["metrics"]["correct"],
                             "count": z_mini["metrics"]["count"]},
            "z_paired_members": {"correct": z_paired["metrics"]["correct"],
                                 "count": z_paired["metrics"]["count"]},
            "fixed24": {"correct": fixed24["correct"], "compact": fixed24["compact"],
                        "per_program": fixed24["per_program"]},
            "minimal96": {"correct": minimal["correct"], "total": minimal["total"],
                          "by_contrast_group": minimal["by_contrast_group"]},
            "stress192": {"accuracy": stress["metrics"]["accuracy"],
                          "macro_f1": stress["metrics"]["macro_f1"],
                          "l3_macro_recall": stress["l3_macro_recall"],
                          "min_class_recall": stress["metrics"]["min_represented_class_recall"],
                          "correct": stress["metrics"]["correct"],
                          "count": stress["metrics"]["count"],
                          "per_language": stress["per_language"]},
        },
        "scope_safety": {"passed": scope["passed"], "l2_controls_ok": scope["l2_controls_ok"],
                         "nearest_only_controls_ok": scope["nearest_only_controls_ok"],
                         "ood_ok": scope["ood_ok"]},
        "end_to_end": {"passed": e2e["passed"], "measured": e2e["measured"],
                       "task7a_reference": e2e["task7a_reference"], "deltas": e2e["deltas"]},
        "cli_default": {"canonical_gates_passed": canonical_passed,
                        "updated_to_task7c": canonical_passed,
                        "note": ("section 24: the L3 default parser moves to the Task 7C checkpoint only "
                                 "when the canonical gates pass, otherwise Task 6T stays the default")},
        "interpretation_boundary": {
            "parser_claimed_as_novelty": False,
            "another_parser_repair_run": False,
            "reference_hardening_reopened": False,
            "downstream_modified": False,
            "attention_or_global_competition_added": False,
            "formal_full_training_or_test_started": False,
            "task7d_decided": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 根据 Task 7C 的 20-class rehearsal 与 L3 组合语义结果决定 parser 是否正式"
                           "冻结，不自行继续 parser 调参、reference hardening 或下游架构改造。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7c.report] canonical {canonical_passed} compositional {compositional_passed} e2e "
          f"{e2e['passed']} | full val {full_val['metrics']['accuracy']:.4f} (min recall "
          f"{full_val['metrics']['min_represented_class_recall']:.3f}) | fixed24 {fixed24['correct']}/24 "
          f"minimal {minimal['correct']}/96 stress {stress['metrics']['accuracy']:.4f} -> {verdict}",
          flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
