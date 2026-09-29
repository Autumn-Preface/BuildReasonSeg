"""Task 6T sections 12-13 — predeclared gates and the single verdict.

Reads the frozen Task 6T artifacts, evaluates the 19 section-12 PASS gates and applies the section-13
priority order to produce exactly one verdict. DSH does not repair the REFERENCE bottleneck and does not
extend the parser scope.

    python scripts/task6t_report.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from buildreasonseg_mvp.structured_grounding import EXPECTED_QUERY_TYPES  # noqa: E402
from scripts.task6t_build_parser_data import FIXED24  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task6t_verdict.json"
BASE_COMMIT = "dc8544f161535d7321ee4c874184348c10192105"
BASELINE_SHA256 = "eb50b02163ec5e8f3305321ee47a6d52a799235b68730b67f539d962a6d028a3"
PREVIOUSLY_FAILING = (
    "找出最大建筑左边的建筑物。", "找出最大建筑右边的建筑物。", "找出最大建筑下面的建筑物。")
ALLOWED_VERDICTS = (
    "INVALID_EXPERIMENT",
    "BASELINE_PARSER_UNAVAILABLE",
    "PARSER_TRAINING_FAILED",
    "PARSER_CANONICAL_REGRESSION",
    "PARSER_SEMANTIC_CONTRAST_FAIL",
    "END_TO_END_REGRESSION",
    "PARSER_HARDENING_PASS",
)
PARSER_SOURCES = ("task6t_build_parser_data.py", "task6t_train_parser.py", "task6t_eval_parser.py",
                  "task6t_scope_audit.py")
FROZEN_PATHS = (
    "buildreasonseg_mvp/task6q_reference_resolver.py",
    "buildreasonseg_mvp/geometric_relation_field_v02.py",
    "buildreasonseg_mvp/task6n_relation_decoder.py",
    "evaluation/task6q_", "evaluation/task6s_",
)

#: The classification/scope path that must contain no keyword or regex semantic override. Evaluation and
#: audit scripts may legitimately filter prompts by text for *reporting* (e.g. selecting the nearest
#: controls), which is not semantic remapping and is therefore not scanned here.
CLASSIFICATION_PATHS = (
    "predict_buildreasonseg_directional.py",
    "buildreasonseg_mvp/task6s_directional_pipeline.py",
    "buildreasonseg_mvp/program_parser.py",
)

#: patterns that would indicate a forbidden keyword/regex semantic override in the classification path
FORBIDDEN_OVERRIDE_PATTERNS = (
    r"最近['\"]\s*in\s+", r"if\s+['\"][^'\"]*['\"]\s+in\s+prompt", r"if\s+['\"][^'\"]*['\"]\s+in\s+text",
    r"re\.search\(\s*['\"][^'\"]*(最大|最小|最近|nearest|largest)",
    r"prompt\.replace\(", r"post_process_program", r"SECONDARY_KEYWORDS",
)


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PATHS],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def keyword_override_findings() -> list[dict]:
    findings = []
    for relative in CLASSIFICATION_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_OVERRIDE_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text[:match.start()].count("\n") + 1
                findings.append({"file": relative, "line": line,
                                 "pattern": pattern, "text": match.group(0)[:60]})
    return findings


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    leakage = load("task6t_parser_leakage_audit.json")
    baseline = load("task6t_parser_baseline.json")
    training = load("task6t_training_summary.json")
    full_val = load("task6t_parser_full_val.json")
    minival = load("task6t_parser_minival240.json")
    paired = load("task6t_parser_pairedval20.json")
    fixed24 = load("task6t_parser_fixed24.json")
    minimal = load("task6t_parser_minimal_pairs_result.json")
    stress = load("task6t_parser_stress_result.json")
    scope = load("task6t_cli_scope_safety.json")
    e2e = load("task6t_end_to_end_regression.json")
    required = {
        "task6t_parser_leakage_audit.json": leakage, "task6t_parser_baseline.json": baseline,
        "task6t_training_summary.json": training, "task6t_parser_full_val.json": full_val,
        "task6t_parser_minival240.json": minival, "task6t_parser_pairedval20.json": paired,
        "task6t_parser_fixed24.json": fixed24,
        "task6t_parser_minimal_pairs_result.json": minimal,
        "task6t_parser_stress_result.json": stress, "task6t_cli_scope_safety.json": scope,
        "task6t_end_to_end_regression.json": e2e,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        write_json(OUT, {"_doc": "Task 6T section 13.", "task": "6T",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[6t.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    override_findings = keyword_override_findings()
    vocab_ok = len(EXPECTED_QUERY_TYPES) == 20
    nearest_present = sum(1 for program in EXPECTED_QUERY_TYPES if "nearest" in program) == 6
    protocol = {
        "frozen_non_parser_unchanged": changes == "", "changed_paths": changes,
        "leakage_exact": leakage["exact_overlap_count"],
        "leakage_normalized": leakage["normalized_overlap_count"],
        "test_split_used": any(payload.get("test_split_used", False) for payload in required.values()),
        "keyword_or_regex_override_findings": override_findings,
        "vocabulary_size": len(EXPECTED_QUERY_TYPES),
        "nearest_classes_present": nearest_present,
        "baseline_sha_verified": training["baseline"]["verified"],
        "baseline_sha256": training["baseline"]["sha256"],
    }
    protocol["clean"] = bool(protocol["frozen_non_parser_unchanged"]
                             and leakage["exact_overlap_count"] == 0
                             and leakage["normalized_overlap_count"] == 0
                             and not protocol["test_split_used"]
                             and not override_findings
                             and protocol["baseline_sha_verified"]
                             and protocol["baseline_sha256"] == BASELINE_SHA256)

    training_ok = bool(training["checkpoint"]["exists"] and training["checkpoint"]["sha256"]
                       and training["selected"]["holdout_macro_f1"] is not None)
    full_val_accuracy = full_val["combined"]["exact_accuracy"]
    minival_ok = bool(minival["english_240_of_240"])
    paired_ok = bool(paired["gate"]["passed"])
    fixed_ok = bool(fixed24["gate"]["passed"])
    all_three = all(entry["correct"] for entry in fixed24["previously_failing_prompts"].values())
    minimal_ok = bool(minimal["gate"]["passed"])
    stress_classes = [value for value in stress["result"]["per_class_recall"].values()
                      if value is not None]
    stress_ok = bool(stress["gate"]["passed"])

    gates = {
        "1_baseline_hash_verified": {"measured": protocol["baseline_sha_verified"],
                                     "passed": protocol["baseline_sha_verified"]},
        "2_zero_leakage": {"measured": {"exact": leakage["exact_overlap_count"],
                                        "normalized": leakage["normalized_overlap_count"]},
                           "passed": leakage["verdict"] == "NO_LEAKAGE"},
        "3_same_architecture": {
            "measured": {"model": training["architecture"]["model"],
                         "text_only": training["architecture"]["text_only"],
                         "classes": len(training["architecture"]["program_ids"])},
            "passed": bool(training["architecture"]["text_only"] and vocab_ok
                           and not training["architecture"]["image_tokens"]),
            "trainable_parameters": training["checkpoint"]["trainable_parameters"],
            "total_parameters": training["checkpoint"]["total_parameters"],
        },
        "4_full_val_accuracy": {"required": 0.995, "measured": full_val_accuracy,
                                "passed": full_val_accuracy >= 0.995},
        "5_minival240": {"required": "240/240", "measured": minival["english_240_of_240"],
                         "passed": minival_ok},
        "6_pairedval_members": {"required": "40/40",
                                "measured": paired["members_correct_english"], "passed": paired_ok},
        "7_fixed24": {"required": ">= 23/24", "measured": fixed24["gate"]["measured_correct"],
                      "passed": fixed24["gate"]["measured_correct"] >= 23},
        "8_previously_failing_prompts": {
            "measured": {prompt: entry["correct"]
                         for prompt, entry in fixed24["previously_failing_prompts"].items()},
            "passed": all_three},
        "9_minimal_pairs": {"required": 1.0, "measured": minimal["gate"]["measured_accuracy"],
                            "passed": minimal_ok},
        "10_stress_accuracy": {"required": 0.95, "measured": stress["gate"]["measured_accuracy"],
                               "passed": stress["gate"]["measured_accuracy"] >= 0.95},
        "11_stress_macro_f1": {"required": 0.95, "measured": stress["gate"]["measured_macro_f1"],
                               "passed": stress["gate"]["measured_macro_f1"] >= 0.95},
        "12_stress_class_recall": {"required": 0.90,
                                   "measured": min(stress_classes) if stress_classes else None,
                                   "passed": bool(stress_classes and min(stress_classes) >= 0.90)},
        "13_ood_exit_4": {"measured": scope["ood_all_exit_4"], "passed": bool(scope["ood_all_exit_4"])},
        "14_scope_exit_5": {"measured": {"exit_5": scope["out_of_scope_all_exit_5"],
                                         "semantic": scope["out_of_scope_all_semantically_correct"],
                                         "no_downstream": scope["out_of_scope_no_downstream"]},
                            "passed": bool(scope["out_of_scope_all_exit_5"]
                                           and scope["out_of_scope_all_semantically_correct"]
                                           and scope["out_of_scope_no_downstream"])},
        "15_e2e_miou": {"required": 1e-6, "measured": e2e["deltas"]["answered_only_miou"],
                        "passed": bool(e2e["within_tolerance"]["answered_only_miou"])},
        "16_paired_exact": {"required": 10, "measured": e2e["measured"]["paired"],
                            "passed": bool(e2e["within_tolerance"]["paired"])},
        "17_no_test_split": {"measured": not protocol["test_split_used"],
                             "passed": not protocol["test_split_used"]},
        "18_no_downstream_changes": {"measured": changes == "", "passed": changes == ""},
        "19_no_keyword_override": {"measured": override_findings, "passed": not override_findings},
    }
    all_gates = all(entry["passed"] for entry in gates.values())
    canonical_ok = (gates["4_full_val_accuracy"]["passed"] and gates["5_minival240"]["passed"]
                    and gates["6_pairedval_members"]["passed"])
    semantic_ok = (gates["7_fixed24"]["passed"] and gates["8_previously_failing_prompts"]["passed"]
                   and gates["9_minimal_pairs"]["passed"] and gates["10_stress_accuracy"]["passed"]
                   and gates["11_stress_macro_f1"]["passed"]
                   and gates["12_stress_class_recall"]["passed"]
                   and gates["13_ood_exit_4"]["passed"] and gates["14_scope_exit_5"]["passed"])

    if not protocol["clean"]:
        verdict = "INVALID_EXPERIMENT"
        reason = ("protocol violation: leakage, test use, frozen non-parser mutation, keyword/regex "
                  "semantic override or baseline hash mismatch")
    elif not training_ok:
        verdict = "PARSER_TRAINING_FAILED"
        reason = "no valid finite hardened checkpoint was produced"
    elif not canonical_ok:
        failing = [name for name in ("4_full_val_accuracy", "5_minival240", "6_pairedval_members")
                   if not gates[name]["passed"]]
        verdict = "PARSER_CANONICAL_REGRESSION"
        reason = f"canonical gates failed: {failing}"
    elif not semantic_ok:
        failing = [name for name in ("7_fixed24", "8_previously_failing_prompts", "9_minimal_pairs",
                                     "10_stress_accuracy", "11_stress_macro_f1",
                                     "12_stress_class_recall", "13_ood_exit_4", "14_scope_exit_5")
                   if not gates[name]["passed"]]
        verdict = "PARSER_SEMANTIC_CONTRAST_FAIL"
        reason = f"semantic contrast / scope-safety gates failed: {failing}"
    elif not (gates["15_e2e_miou"]["passed"] and gates["16_paired_exact"]["passed"]):
        verdict = "END_TO_END_REGRESSION"
        reason = "parsed programs are correct but the frozen downstream no longer reproduces Task 6S"
    elif all_gates:
        verdict = "PARSER_HARDENING_PASS"
        reason = "all section 12 gates pass"
    else:
        verdict = "INVALID_EXPERIMENT"
        reason = "unclassified gate configuration"

    payload = {
        "_doc": (
            "Task 6T sections 12-13. Predeclared PASS gates and the single verdict. The Task 6T scope is "
            "limited to ProgramHead training data/checkpoint and parser-evaluation code; the REFERENCE "
            "bottleneck is explicitly out of scope and was not touched."
        ),
        "task": "6T",
        "verdict": verdict,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "reason": reason,
        "protocol": protocol,
        "gates": gates,
        "all_gates_passed": all_gates,
        "headline_metrics": {
            "baseline": {
                "minival240_en": baseline["minival240"]["exact_accuracy"],
                "fixed24": f"{baseline['fixed24']['exact_correct']}/24",
                "minimal_pairs_accuracy": baseline["minimal_pairs"]["exact_accuracy"],
                "stress_accuracy": baseline["stress_v1"]["exact_accuracy"],
                "stress_macro_f1": baseline["stress_v1"]["macro_f1"],
                "previously_failing_predictions": baseline["fixed24"]["previously_failing_prompts"],
            },
            "hardened": {
                "full_val_accuracy": full_val_accuracy,
                "full_val_english": full_val["english"]["exact_accuracy"],
                "full_val_chinese": full_val["chinese"]["exact_accuracy"],
                "minival240_en": minival["english"]["exact_accuracy"],
                "pairedval20_members": paired["members_correct_english"],
                "fixed24": f"{fixed24['gate']['measured_correct']}/24",
                "minimal_pairs_accuracy": minimal["gate"]["measured_accuracy"],
                "stress_accuracy": stress["gate"]["measured_accuracy"],
                "stress_macro_f1": stress["gate"]["measured_macro_f1"],
                "stress_min_class_recall": stress["gate"]["measured_min_class_recall"],
                "previously_failing_predictions": {
                    prompt: entry["predicted"]
                    for prompt, entry in fixed24["previously_failing_prompts"].items()},
                "scope_out_of_scope_programs": scope["nearest_controls_parsed_program"],
                "e2e_answered_miou": e2e["measured"]["answered_only_miou"],
                "e2e_paired": e2e["measured"]["paired"],
                "e2e_abstentions": e2e["measured"]["abstentions"],
            },
            "training": {
                "selected_config": training["selected_config"],
                "holdout_macro_f1": training["selected"]["holdout_macro_f1"],
                "checkpoint_sha256": training["checkpoint"]["sha256"],
                "trainable_parameters": training["checkpoint"]["trainable_parameters"],
                "total_parameters": training["checkpoint"]["total_parameters"],
                "wall_seconds": {name: training["configs"][name]["wall_seconds"]
                                 for name in training["configs"]},
            },
        },
        "erratum_recorded": {
            "task6s_claim": "the frozen 20-program vocabulary has no nearest program",
            "actual": "the vocabulary contains 6 nearest classes",
            "nearest_classes": [program for program in EXPECTED_QUERY_TYPES if "nearest" in program],
            "cause_of_task6s_scope_failures": (
                "the frozen ProgramHead misclassified nearest-containing instructions as direction-only "
                "programs, so the Task 6S scope check could not reject them"),
            "task6s_artifacts_mutated": False,
        },
        "out_of_scope_bottleneck": "REFERENCE (deliberately untouched by Task 6T)",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[6t.report] full-val {full_val_accuracy:.4f} | MiniVal240 {minival_ok} | paired "
          f"{paired['members_correct_english']}/40 | fixed24 {fixed24['gate']['measured_correct']}/24 | "
          f"minimal {minimal['gate']['measured_accuracy']:.4f} | stress "
          f"{stress['gate']['measured_accuracy']:.4f} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
