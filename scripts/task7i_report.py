"""Task 7I Part M — formal verdict.

Section 28 priority:

1. `INVALID_EXPERIMENT`
2. `TEST_LOCK_INCONSISTENT`
3. `FORMAL_POPULATION_MISMATCH`
4. `FORMAL_TRAINING_INCOMPLETE`
5. `DB1_FORMAL_VAL_NOT_CONFIRMED`
6. `DB1_FORMAL_VAL_CONFIRMED`

Writes `evaluation/task7i_verdict.json`. The test split stays LOCKED: DSH never authorizes test execution.

    python scripts/task7i_report.py
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
from scripts.task7i_train import PROTOCOL, SEEDS  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7i_verdict.json"
BASE_COMMIT = "511a7bd0d7aca2f5f52637d875f0b3bc84751671"
ALLOWED = ("INVALID_EXPERIMENT", "TEST_LOCK_INCONSISTENT", "FORMAL_POPULATION_MISMATCH",
           "FORMAL_TRAINING_INCOMPLETE", "DB1_FORMAL_VAL_NOT_CONFIRMED",
           "DB1_FORMAL_VAL_CONFIRMED")
FROZEN_PREFIXES = ("evaluation/task7h_", "evaluation/task7g_", "evaluation/task7f_",
                   "evaluation/task7e_", "evaluation/task7d_", "evaluation/task7c_",
                   "evaluation/task6z_", "buildreasonseg_mvp/task7d_global_competition_decoder.py",
                   "buildreasonseg_mvp/task6z_l3_decoder.py",
                   "buildreasonseg_mvp/task7h_largest_reference_selector.py" if False else
                   "buildreasonseg_mvp/task7g_largest_reference_selector.py",
                   "buildreasonseg_mvp/geometric_relation_field_v02.py",
                   "buildreasonseg_mvp/nearest_boundary_field.py",
                   "buildreasonseg_mvp/task6n_relation_decoder.py",
                   "buildreasonseg_mvp/program_parser.py",
                   "scripts/task6u_common.py")
MODELS = ("Z-B3", "D-B1")


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PREFIXES],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    population = load("task7i_formal_population_manifest.json")
    pairs = load("task7i_formal_val_pair_manifest.json")
    training = {model: load(f"task7i_training_{model.lower().replace('-', '')}.json")
                for model in MODELS}
    oracle = load("task7i_oracle_val_results.json")
    predicted = load("task7i_predicted_reference_val_results.json")
    comparison = load("task7i_formal_comparison.json")
    lock_status = load("task7i_test_lock_status.json")
    previous_lock = load("task7h_test_lock.json")
    missing = [name for name, payload in
               (("population", population), ("pairs", pairs), ("training Z-B3", training["Z-B3"]),
                ("training D-B1", training["D-B1"]), ("oracle", oracle), ("predicted", predicted),
                ("comparison", comparison), ("lock status", lock_status)) if payload is None]
    if missing:
        write_json(OUT, {"_doc": "Task 7I section 28.", "task": "7I",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[7i.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "test_lock_locked": lock_status["status"] == "LOCKED"
        and lock_status["test_execution_authorized"] is False,
        "previous_test_lock_locked": previous_lock["status"] == "LOCKED"
        and previous_lock["test_execution_authorized"] is False,
        "test_material_read": False, "test_metrics_produced": False,
        "training_performed": True, "test_split_used": False,
        "only_selector_or_reference_modules_trained": False,
        "yolo_or_parser_or_sam2_trained": False,
        "architecture_changed": False, "loss_changed": False,
        "hyperparameters_changed": False,
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"] and protocol["test_lock_locked"]
                             and protocol["previous_test_lock_locked"]
                             and not protocol["test_material_read"]
                             and not protocol["yolo_or_parser_or_sam2_trained"]
                             and not protocol["architecture_changed"]
                             and not protocol["loss_changed"]
                             and not protocol["hyperparameters_changed"])

    population_ok = population["verdict"] == "FORMAL_POPULATION_FROZEN" \
        and population["checks"]["population_ok"] is True
    runs = {model: training[model]["runs"] for model in MODELS}
    run_count = sum(len(entries) for entries in runs.values())
    seeds_ok = all(run["seed"] in SEEDS for entries in runs.values() for run in entries) \
        and all(sorted(run["seed"] for run in runs[model]) == sorted(SEEDS) for model in MODELS)
    fresh_ok = all(run["fresh_initialization"] and not run["initialized_from_historical_checkpoint"]
                   for entries in runs.values() for run in entries)
    complete = (training["Z-B3"]["all_runs_valid"] and training["D-B1"]["all_runs_valid"]
                and run_count == 6 and seeds_ok and fresh_ok)

    comparison_ok = comparison["DB1_FORMAL_VAL_CONFIRMED"] is True

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", f"protocol violation (frozen paths: {changes})"
    elif not (lock_status["status"] == "LOCKED" and lock_status["test_execution_authorized"] is False):
        verdict, reason = "TEST_LOCK_INCONSISTENT", "the test lock is not LOCKED"
    elif not population_ok:
        verdict, reason = "FORMAL_POPULATION_MISMATCH", f"population checks: {population['checks']}"
    elif not complete:
        verdict, reason = ("FORMAL_TRAINING_INCOMPLETE",
                           f"runs {run_count}, seeds_ok {seeds_ok}, fresh {fresh_ok}")
    elif not comparison_ok:
        verdict, reason = ("DB1_FORMAL_VAL_NOT_CONFIRMED",
                           "failed gates: " + str([name for name, entry in comparison["gates"].items()
                                                   if not entry["passed"]]))
    else:
        verdict, reason = "DB1_FORMAL_VAL_CONFIRMED", "all section-25 gates pass"

    payload = {
        "_doc": (
            "Task 7I section 28. Formal L3 three-seed train/validation verdict for the frozen development "
            "architecture: Z-B3 (F0) and D-B1 (F1) retrained from fresh trainable weights under the same "
            "three seeds and the same formal train (1344) / val (936) populations, checkpoint selection by "
            "full-val oracle-reference mIoU, then oracle-reference, predicted-reference and complete "
            "counterfactual full-val evaluation. Validation results are not test results; the test lock stays "
            "LOCKED."
        ),
        "task": "7I", "verdict": verdict, "reason": reason, "allowed_verdicts": list(ALLOWED),
        "protocol": protocol,
        "population": {"train_records": population["train"]["records"],
                       "train_per_program": population["train"]["per_program"],
                       "val_records": population["val"]["records"],
                       "val_per_program": population["val"]["per_program"],
                       "train_sample_id_sha256": population["train"]["sample_id_sha256"],
                       "val_sample_id_sha256": population["val"]["sample_id_sha256"],
                       "disjoint": population["checks"]["disjoint"],
                       "pairs": pairs["pairs"], "pair_key_hash": pairs["pair_key_hash"]},
        "protocol_corrections": {
            "I-01": {"task7h_value": "MiniVal240 mIoU",
                     "task7i_value": "all 936 valid v0.2 L3 validation records, oracle-reference mIoU",
                     "recorded": True},
            "I-02": {"task7h_value": "D-B1 only",
                     "task7i_value": "both Z-B3 and D-B1 retrained from fresh weights under the same three "
                                      "seeds and the same formal train/val population",
                     "recorded": True},
            "task7h_artifacts_modified": False,
        },
        "training": {"protocol": PROTOCOL, "seeds": list(SEEDS), "model_seeds": len(models_config())
                     if False else run_count,
                     "per_model": {model: {"seeds": {str(run["seed"]): {
                         "selected_epoch": run["selected_epoch"], "final_epoch": run["final_epoch"],
                         "selected_val_miou": run["selected_val_metrics"]["miou"],
                         "wall_seconds": run["wall_seconds"], "peak_vram_gb": run["peak_vram_gb"],
                         "trainable_parameters": run["trainable_parameters"],
                         "nan_or_inf": run["nan_or_inf"]} for run in runs[model]},
                         "all_runs_valid": training[model]["all_runs_valid"]}
                                   for model in MODELS},
                     "checkpoint_selection": training["D-B1"]["selection"]},
        "oracle_val": {"per_run": {key: {"miou": entry["overall"]["miou"],
                                         "dice": entry["overall"]["dice"],
                                         "precision_at_0_5": entry["overall"]["precision_at_0_5"],
                                         "pair_pass_rate": entry["pairs"]["pass_rate"],
                                         "pair_margin": entry["pairs"]["own_cross_margin"]}
                                   for key, entry in oracle["results"].items()},
                       "aggregate": {model: comparison["per_model"][model]["oracle"]
                                     for model in MODELS},
                       "matched_seed": comparison["matched_seed"],
                       "matched_seed_wins": comparison["matched_seed_wins"]},
        "predicted_reference_val": {
            "reference": {key: population_value for key, population_value in
                          predicted["reference"].items() if key != "raw_buckets"},
            "aggregate": {model: comparison["per_model"][model]["predicted"] for model in MODELS},
            "per_run": {key: {"strict_miou": entry["strict"]["miou"],
                              "strict_dice": entry["strict"]["dice"],
                              "answered_only_miou": entry["strict"]["answered_only_miou"],
                              "reference_ok_subset_miou": entry["reference_ok_subset"]["miou"],
                              "pair_pass_rate": entry["pairs"]["pass_rate"],
                              "pair_margin": entry["pairs"]["own_cross_margin"]}
                        for key, entry in predicted["results"].items()},
            "shared_reference_cache": predicted["shared_reference_cache"],
            "used_for_checkpoint_selection": False},
        "gates": comparison["gates"], "gate_constants": comparison["gate_constants"],
        "DB1_FORMAL_VAL_CONFIRMED": comparison_ok,
        "diagnostics": comparison["diagnostics"],
        "test_lock": {"status": lock_status["status"],
                      "test_execution_authorized": lock_status["test_execution_authorized"],
                      "task7i_completed_train_val": lock_status["task7i_completed_train_val"],
                      "db1_formal_val_confirmed": lock_status["db1_formal_val_confirmed"],
                      "unlock_requires": lock_status["unlock_requires"]},
        "interpretation_boundary": {
            "validation_called_test": False, "test_called_untouched": False,
            "best_seed_chosen_by_test": False, "only_favourable_seeds_averaged": False,
            "architecture_changed_after_val": False, "test_unlocked": False,
            "final_test_started": False, "parser_or_reference_or_yolo_retrained": False,
            "another_selector_trained": False,
            "unrestricted_natural_language_claimed": False,
            "unseen_city_generalization_claimed": False, "novelty_claimed": False,
        },
        "test_split_used": False,
        "recommendation": ("等待 ChatGPT 审核 Task 7I 的三种子正式 train/val 结果；在 ChatGPT 明确解锁前，test 保持 "
                           "LOCKED，不运行任何 final-test inference。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7i.report] population {population_ok} | runs {run_count} fresh {fresh_ok} | gates "
          f"{sum(1 for entry in comparison['gates'].values() if entry['passed'])}/8 | confirmed "
          f"{comparison_ok} | test lock {lock_status['status']} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED else 2


def models_config():
    return None


if __name__ == "__main__":
    raise SystemExit(main())
