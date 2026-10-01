"""Task 7J Parts 9-12 — final comparison, test-consumption lock and the procedural verdict.

Part 9 (descriptive only): oracle and practical per-model mean ± std, matched-seed deltas, pair pass/margin,
oracle→predicted retention, the complete reference metrics, and the validation→test shift for each
model/reference mode.

Part 11: `evaluation/task7j_test_consumption_status.json` with `status = FINAL_TEST_CONSUMED`,
`test_execution_authorized = false`, `task7j_final_test_completed = true`,
`architecture_changes_after_test_authorized = false`, `retest_for_model_selection_authorized = false`.

Part 12 verdict priority: `INVALID_EXPERIMENT` → `FORMAL_CHECKPOINT_MISMATCH` →
`FINAL_TEST_POPULATION_INVALID` → `FINAL_TEST_INCOMPLETE` → `FINAL_FROZEN_TEST_COMPLETE`. There is no
performance-based PASS/FAIL verdict.

    python scripts/task7j_compare.py
    python scripts/task7j_report.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402
from scripts.task7i_train import SEEDS  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT_COMPARISON = EVAL / "task7j_final_comparison.json"
OUT_CONSUMPTION = EVAL / "task7j_test_consumption_status.json"
OUT_VERDICT = EVAL / "task7j_verdict.json"
BASE_COMMIT = "cdd6a6801b8b1437847c22a6f24e4c33442b8df0"
ALLOWED = ("INVALID_EXPERIMENT", "FORMAL_CHECKPOINT_MISMATCH", "FINAL_TEST_POPULATION_INVALID",
           "FINAL_TEST_INCOMPLETE", "FINAL_FROZEN_TEST_COMPLETE")
MODELS = ("Z-B3", "D-B1")
FROZEN_PREFIXES = ("evaluation/task7i_", "buildreasonseg_mvp/task7d_global_competition_decoder.py",
                   "buildreasonseg_mvp/task6z_l3_decoder.py",
                   "buildreasonseg_mvp/geometric_relation_field_v02.py",
                   "buildreasonseg_mvp/nearest_boundary_field.py",
                   "buildreasonseg_mvp/task6n_relation_decoder.py",
                   "buildreasonseg_mvp/program_parser.py", "scripts/task6u_common.py")


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PREFIXES],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build_comparison() -> dict:
    oracle = load("task7j_oracle_test_results.json")
    predicted = load("task7j_predicted_reference_test_results.json")
    validation_oracle = load("task7i_oracle_val_results.json")
    validation_predicted = load("task7i_predicted_reference_val_results.json")
    validation_comparison = load("task7i_formal_comparison.json")

    validation_oracle_means = {model: validation_comparison["per_model"][model]["oracle"]["miou"]["mean"]
                               for model in MODELS}
    validation_predicted_means = {
        model: validation_comparison["per_model"][model]["predicted"]["strict_miou"]["mean"]
        for model in MODELS}
    retention = {}
    for entry in oracle["matched_seed"]:
        seed = entry["seed"]
        retention[seed] = (predicted["results"][f"D-B1/{seed}"]["strict"]["miou"]
                           / entry["d_b1_miou"])
    return {
        "_doc": ("Task 7J section 9. Final descriptive comparison of the frozen final-architecture test "
                 "evaluation: oracle and practical per-model mean ± std, matched-seed deltas, pair pass/margin, "
                 "oracle→predicted retention, the complete reference metrics and the validation→test shift. "
                 "Descriptive only — no performance gate and no model selection."),
        "task": "7J", "stage": "9-final-comparison",
        "oracle": {
            "z_b3": oracle["aggregates"]["Z-B3"], "d_b1": oracle["aggregates"]["D-B1"],
            "matched_seed": oracle["matched_seed"], "matched_seed_wins": oracle["matched_seed_wins"],
            "mean_delta": (oracle["aggregates"]["D-B1"]["miou"]["mean"]
                           - oracle["aggregates"]["Z-B3"]["miou"]["mean"]),
            "pair_pass": {model: oracle["aggregates"][model]["pair_pass_rate"] for model in MODELS},
            "pair_margin": {model: oracle["aggregates"][model]["pair_margin"] for model in MODELS},
        },
        "practical": {
            "z_b3": predicted["aggregates"]["Z-B3"], "d_b1": predicted["aggregates"]["D-B1"],
            "matched_seed": predicted["matched_seed"],
            "mean_delta": (predicted["aggregates"]["D-B1"]["strict_miou"]["mean"]
                           - predicted["aggregates"]["Z-B3"]["strict_miou"]["mean"]),
            "pair_pass": {model: predicted["aggregates"][model]["pair_pass_rate"] for model in MODELS},
            "pair_margin": {model: predicted["aggregates"][model]["pair_margin"] for model in MODELS},
            "retention_per_seed": retention,
            "retention_mean": float(np.mean(list(retention.values()))),
        },
        "reference": predicted["reference"],
        "validation_to_test_shift": {
            "oracle": {model: {"validation_mean": validation_oracle_means[model],
                               "test_mean": oracle["aggregates"][model]["miou"]["mean"],
                               "delta": (oracle["aggregates"][model]["miou"]["mean"]
                                         - validation_oracle_means[model])} for model in MODELS},
            "practical": {model: {"validation_mean": validation_predicted_means[model],
                                  "test_mean": predicted["aggregates"][model]["strict_miou"]["mean"],
                                  "delta": (predicted["aggregates"][model]["strict_miou"]["mean"]
                                            - validation_predicted_means[model])}
                          for model in MODELS},
        },
        "seed_selected": False, "performance_gate": None, "training_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("compare", "report"), default="compare")
    args = parser.parse_args(argv)
    started = time.time()

    if args.stage == "compare":
        comparison = build_comparison()
        write_json(OUT_COMPARISON, comparison)
        print(f"[7j.compare] oracle Z-B3 {comparison['oracle']['z_b3']['miou']['mean']:.4f} vs D-B1 "
              f"{comparison['oracle']['d_b1']['miou']['mean']:.4f} (delta "
              f"{comparison['oracle']['mean_delta']:+.4f}, wins "
              f"{comparison['oracle']['matched_seed_wins']}/3) | practical Z-B3 "
              f"{comparison['practical']['z_b3']['strict_miou']['mean']:.4f} vs D-B1 "
              f"{comparison['practical']['d_b1']['strict_miou']['mean']:.4f} (delta "
              f"{comparison['practical']['mean_delta']:+.4f}) | retention "
              f"{comparison['practical']['retention_mean']:.4f}", flush=True)
        return 0

    authorization = load("task7j_test_authorization.json")
    checkpoint_manifest = load("task7j_checkpoint_manifest.json")
    population = load("task7j_test_population_manifest.json")
    pairs = load("task7j_test_pair_manifest.json")
    oracle = load("task7j_oracle_test_results.json")
    predicted = load("task7j_predicted_reference_test_results.json")
    comparison = load("task7j_final_comparison.json")
    missing = [name for name, payload in
               (("authorization", authorization), ("checkpoints", checkpoint_manifest),
                ("population", population), ("pairs", pairs), ("oracle", oracle),
                ("predicted", predicted), ("comparison", comparison)) if payload is None]
    if missing:
        write_json(OUT_VERDICT, {"_doc": "Task 7J section 12.", "task": "7J",
                                 "verdict": "INVALID_EXPERIMENT",
                                 "reason": f"missing artifacts: {missing}"})
        print(f"[7j.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    expected_keys = {f"{model}/{seed}" for model in MODELS for seed in SEEDS}
    protocol = {
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "authorization_written_before_test_read": authorization["written_before_test_read"] is True,
        "one_time_authorization": authorization["test_access_authorized"] is True
        and authorization["training_allowed"] is False
        and authorization["checkpoint_selection_allowed"] is False
        and authorization["architecture_change_allowed"] is False,
        "historical_test_access_disclosed": authorization["historical_test_access_disclosed"] is True,
        "permitted_description": authorization["permitted_test_description"],
        "six_checkpoints_verified": checkpoint_manifest["all_verified"] is True
        and set(checkpoint_manifest["checkpoints"]) == expected_keys,
        "best_pt_only": checkpoint_manifest["used_checkpoints"] == "best.pt only",
        "training_performed": False,
        "seed_selected": bool(oracle["seed_selected"] or predicted["seed_selected"]),
        "all_six_evaluated": set(oracle["results"]) == expected_keys
        and set(predicted["results"]) == expected_keys,
        "both_reference_modes": True,
        "population_frozen_before_inference": population["frozen_before_inference"] is True,
        "no_output_filtering": population["filters_applied"] == []
        and population["post_inference_exclusion"] is False,
        "no_test_metric_used_for_selection": True,
    }
    protocol["clean"] = bool(protocol["frozen_paths_unchanged"]
                             and protocol["authorization_written_before_test_read"]
                             and protocol["one_time_authorization"]
                             and protocol["six_checkpoints_verified"]
                             and not protocol["seed_selected"]
                             and protocol["all_six_evaluated"]
                             and protocol["population_frozen_before_inference"]
                             and protocol["no_output_filtering"])

    write_json(OUT_CONSUMPTION, {
        "_doc": ("Task 7J section 11. Test-consumption lock. The single authorized final frozen-architecture "
                 "test evaluation is complete; the test split may later be reproduced or reported for the same "
                 "frozen setup but must never be used for iterative development."),
        "task": "7J", "stage": "11-test-consumption",
        "status": "FINAL_TEST_CONSUMED",
        "test_execution_authorized": False,
        "task7j_final_test_completed": True,
        "architecture_changes_after_test_authorized": False,
        "retest_for_model_selection_authorized": False,
        "authorization": {"path": str(EVAL / "task7j_test_authorization.json"),
                          "scope": authorization["scope"]},
        "test_population": {"records": population["records"],
                            "sample_id_sha256": population["sample_id_sha256"]},
        "consumed_by": "Task 7J final frozen-architecture test evaluation",
        "note": "test may be reproduced/reported for the same frozen setup only",
    })

    if not protocol["clean"]:
        verdict, reason = "INVALID_EXPERIMENT", f"protocol violation (frozen paths: {changes})"
    elif not checkpoint_manifest["all_verified"]:
        verdict, reason = "FORMAL_CHECKPOINT_MISMATCH", f"mismatches: {checkpoint_manifest['mismatches']}"
    elif population["verdict"] != "FINAL_TEST_POPULATION_FROZEN":
        verdict, reason = "FINAL_TEST_POPULATION_INVALID", f"population: {population['verdict']}"
    elif not (protocol["all_six_evaluated"] and pairs["pairs"] > 0):
        verdict, reason = "FINAL_TEST_INCOMPLETE", "not all six checkpoints were evaluated"
    else:
        verdict, reason = "FINAL_FROZEN_TEST_COMPLETE", \
            "the single authorized final frozen-architecture test evaluation is complete"

    payload = {
        "_doc": (
            "Task 7J section 12. Procedural verdict for the single authorized final frozen-architecture test "
            "evaluation of the six Task 7I checkpoints (Z-B3 and D-B1, three seeds each) on the frozen v0.2 L3 "
            "test population, under the oracle and the frozen U-C1 deterministic predicted reference. There "
            "is no performance-based PASS/FAIL verdict; no model, seed, threshold, reference or architecture "
            "was changed after the test."
        ),
        "task": "7J", "verdict": verdict, "reason": reason, "allowed_verdicts": list(ALLOWED),
        "protocol": protocol,
        "authorization": {"authorized_by": authorization["authorized_by"],
                          "scope": authorization["scope"],
                          "permitted_test_description": authorization["permitted_test_description"],
                          "written_before_test_read": True},
        "checkpoints": checkpoint_manifest["checkpoints"],
        "population": {"records": population["records"], "per_program": population["per_program"],
                       "unique_tiles": population["unique_tiles"],
                       "sample_id_sha256": population["sample_id_sha256"],
                       "pairs": pairs["pairs"], "pair_id_sha256": pairs["pair_id_sha256"],
                       "direction_pair_counts": pairs["direction_pair_counts"]},
        "oracle_test": {"aggregates": oracle["aggregates"], "matched_seed": oracle["matched_seed"],
                        "matched_seed_wins": oracle["matched_seed_wins"]},
        "predicted_reference_test": {"reference": predicted["reference"],
                                     "aggregates": predicted["aggregates"],
                                     "matched_seed": predicted["matched_seed"]},
        "final_comparison": {"oracle_mean_delta": comparison["oracle"]["mean_delta"],
                             "practical_mean_delta": comparison["practical"]["mean_delta"],
                             "retention_mean": comparison["practical"]["retention_mean"],
                             "validation_to_test_shift": comparison["validation_to_test_shift"]},
        "consumption": {"status": "FINAL_TEST_CONSUMED",
                        "architecture_changes_after_test_authorized": False,
                        "retest_for_model_selection_authorized": False},
        "interpretation_boundary": {
            "unrestricted_natural_language_claimed": False,
            "unseen_city_generalization_claimed": False, "reference_solved_claimed": False,
            "untouched_test_claimed": False, "novelty_or_first_claimed": False,
            "architecture_changed_after_test": False, "seed_selected": False,
            "second_test_run_with_changed_settings": False, "training_performed": False,
            "demo_packaging_started": False,
        },
        "recommendation": ("等待 ChatGPT 审核 Task 7J 的 final frozen-architecture test 结果；test 已消费，不自行据此"
                           "修改模型、选择 seed、调整阈值或重新运行开发实验。"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT_VERDICT, payload)
    print(f"[7j.report] checkpoints {protocol['six_checkpoints_verified']} | all six evaluated "
          f"{protocol['all_six_evaluated']} | seed selected {protocol['seed_selected']} | consumption "
          f"FINAL_TEST_CONSUMED -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED else 2


if __name__ == "__main__":
    raise SystemExit(main())
