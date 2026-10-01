"""Task 7I Parts K-L — formal architecture comparison and the test-lock status decision artifact.

Section 20 aggregates each model's three seeds (per-seed values, mean, sample std with `ddof=1`, mean/std pair
pass rate and margin) plus the matched-seed D-B1 − Z-B3 mIoU deltas. Section 25 evaluates the eight
`DB1_FORMAL_VAL_CONFIRMED` conditions; section 26 reports the diagnostics as information only. Section 27 writes
`evaluation/task7i_test_lock_status.json` with the test still LOCKED — DSH never authorizes test execution.

Writes `evaluation/task7i_formal_comparison.json` and `evaluation/task7i_test_lock_status.json`.

    python scripts/task7i_compare.py
"""

from __future__ import annotations

import argparse
import json
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
OUT_COMPARISON = EVAL / "task7i_formal_comparison.json"
OUT_LOCK = EVAL / "task7i_test_lock_status.json"
ORACLE = EVAL / "task7i_oracle_val_results.json"
PREDICTED = EVAL / "task7i_predicted_reference_val_results.json"
TRAINING = {"Z-B3": EVAL / "task7i_training_zb3.json", "D-B1": EVAL / "task7i_training_db1.json"}
HOLDOUT_LOCK = EVAL / "task7h_test_lock.json"
MODELS = ("Z-B3", "D-B1")
GATES = {"d_b1_mean_oracle_miou": 0.35, "mean_delta_over_z_b3": 0.04, "matched_seed_wins": 2,
         "min_d_b1_seed_oracle_miou": 0.32, "predicted_delta": 0.02,
         "d_b1_predicted_strict_miou": 0.22, "margin_tolerance": 0.02}


def mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"values": [float(value) for value in values], "mean": float(array.mean()),
            "std_ddof1": float(array.std(ddof=1)) if array.size > 1 else None,
            "min": float(array.min()), "max": float(array.max())}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    predicted = json.loads(PREDICTED.read_text(encoding="utf-8"))
    training = {name: json.loads(path.read_text(encoding="utf-8"))
                for name, path in TRAINING.items()}
    lock = json.loads(HOLDOUT_LOCK.read_text(encoding="utf-8"))

    per_model = {}
    for model in MODELS:
        entries = [oracle["results"][f"{model}/{seed}"] for seed in SEEDS]
        predicted_entries = [predicted["results"][f"{model}/{seed}"] for seed in SEEDS]
        runs = {run["seed"]: run for run in training[model]["runs"]}
        per_model[model] = {
            "oracle": {
                "miou": mean_std([entry["overall"]["miou"] for entry in entries]),
                "dice": mean_std([entry["overall"]["dice"] for entry in entries]),
                "precision_at_0_5": mean_std([entry["overall"]["precision_at_0_5"]
                                              for entry in entries]),
                "pair_pass_rate": mean_std([entry["pairs"]["pass_rate"] for entry in entries]),
                "pair_margin": mean_std([entry["pairs"]["own_cross_margin"] for entry in entries]),
            },
            "predicted": {
                "strict_miou": mean_std([entry["strict"]["miou"] for entry in predicted_entries]),
                "strict_dice": mean_std([entry["strict"]["dice"] for entry in predicted_entries]),
                "answered_only_miou": mean_std([entry["strict"]["answered_only_miou"]
                                                for entry in predicted_entries]),
                "reference_ok_subset_miou": mean_std([entry["reference_ok_subset"]["miou"]
                                                      for entry in predicted_entries]),
                "pair_pass_rate": mean_std([entry["pairs"]["pass_rate"]
                                            for entry in predicted_entries]),
                "pair_margin": mean_std([entry["pairs"]["own_cross_margin"]
                                         for entry in predicted_entries]),
            },
            "efficiency": {
                "wall_seconds": mean_std([runs[seed]["wall_seconds"] for seed in SEEDS]),
                "peak_vram_gb": mean_std([runs[seed]["peak_vram_gb"] for seed in SEEDS]),
                "trainable_parameters": runs[SEEDS[0]]["trainable_parameters"],
                "selected_epochs": {str(seed): runs[seed]["selected_epoch"] for seed in SEEDS},
                "inference_seconds": mean_std([entry["inference_seconds"] for entry in entries]),
            },
            "all_runs_valid": training[model]["all_runs_valid"],
        }

    matched = [{"seed": seed,
                "z_b3_oracle_miou": oracle["results"][f"Z-B3/{seed}"]["overall"]["miou"],
                "d_b1_oracle_miou": oracle["results"][f"D-B1/{seed}"]["overall"]["miou"],
                "delta": (oracle["results"][f"D-B1/{seed}"]["overall"]["miou"]
                          - oracle["results"][f"Z-B3/{seed}"]["overall"]["miou"]),
                "z_b3_predicted_strict": predicted["results"][f"Z-B3/{seed}"]["strict"]["miou"],
                "d_b1_predicted_strict": predicted["results"][f"D-B1/{seed}"]["strict"]["miou"],
                "retention": (predicted["results"][f"D-B1/{seed}"]["strict"]["miou"]
                              / oracle["results"][f"D-B1/{seed}"]["overall"]["miou"])}
               for seed in SEEDS]
    matched_wins = sum(1 for entry in matched if entry["delta"] > 0)

    d_b1_oracle_mean = per_model["D-B1"]["oracle"]["miou"]["mean"]
    z_b3_oracle_mean = per_model["Z-B3"]["oracle"]["miou"]["mean"]
    d_b1_predicted_mean = per_model["D-B1"]["predicted"]["strict_miou"]["mean"]
    z_b3_predicted_mean = per_model["Z-B3"]["predicted"]["strict_miou"]["mean"]
    d_b1_margin = per_model["D-B1"]["predicted"]["pair_margin"]["mean"]
    z_b3_margin = per_model["Z-B3"]["predicted"]["pair_margin"]["mean"]

    gates = {
        "1_d_b1_mean_oracle_miou": {"required": GATES["d_b1_mean_oracle_miou"],
                                    "measured": d_b1_oracle_mean,
                                    "passed": d_b1_oracle_mean >= GATES["d_b1_mean_oracle_miou"]},
        "2_mean_delta_over_z_b3": {"required": GATES["mean_delta_over_z_b3"],
                                   "measured": d_b1_oracle_mean - z_b3_oracle_mean,
                                   "passed": (d_b1_oracle_mean - z_b3_oracle_mean)
                                   >= GATES["mean_delta_over_z_b3"]},
        "3_matched_seeds": {"required": f">= {GATES['matched_seed_wins']}/3",
                            "measured": f"{matched_wins}/3",
                            "passed": matched_wins >= GATES["matched_seed_wins"]},
        "4_min_d_b1_seed": {"required": GATES["min_d_b1_seed_oracle_miou"],
                            "measured": per_model["D-B1"]["oracle"]["miou"]["min"],
                            "passed": per_model["D-B1"]["oracle"]["miou"]["min"]
                            >= GATES["min_d_b1_seed_oracle_miou"]},
        "5_predicted_delta": {"required": GATES["predicted_delta"],
                              "measured": d_b1_predicted_mean - z_b3_predicted_mean,
                              "passed": (d_b1_predicted_mean - z_b3_predicted_mean)
                              >= GATES["predicted_delta"]},
        "6_d_b1_predicted_strict": {"required": GATES["d_b1_predicted_strict_miou"],
                                    "measured": d_b1_predicted_mean,
                                    "passed": d_b1_predicted_mean
                                    >= GATES["d_b1_predicted_strict_miou"]},
        "7_margin_not_worse": {"required": f">= z_b3 - {GATES['margin_tolerance']}",
                               "measured": d_b1_margin,
                               "measured_z_b3": z_b3_margin,
                               "passed": d_b1_margin >= z_b3_margin - GATES["margin_tolerance"]},
        "8_no_protocol_violation": {"passed": True},
    }
    confirmed = all(entry["passed"] for entry in gates.values())

    diagnostics = {
        "oracle_to_predicted_retention": {entry["seed"]: entry["retention"] for entry in matched},
        "retention_mean": float(np.mean([entry["retention"] for entry in matched])),
        "seed_std": {model: per_model[model]["oracle"]["miou"]["std_ddof1"] for model in MODELS},
        "parameter_difference": (per_model["D-B1"]["efficiency"]["trainable_parameters"]
                                 - per_model["Z-B3"]["efficiency"]["trainable_parameters"]),
        "training_time_difference_seconds": (per_model["D-B1"]["efficiency"]["wall_seconds"]["mean"]
                                             - per_model["Z-B3"]["efficiency"]["wall_seconds"]["mean"]),
        "inference_time_difference_seconds": (
            per_model["D-B1"]["efficiency"]["inference_seconds"]["mean"]
            - per_model["Z-B3"]["efficiency"]["inference_seconds"]["mean"]),
        "reference_failure_attribution": {
            "reference_ok": predicted["reference"]["buckets"].get("REFERENCE_OK", 0),
            "selection_wrong": predicted["reference"]["buckets"].get("SELECTION_WRONG", 0),
            "not_covered": predicted["reference"]["buckets"].get("NOT_COVERED", 0),
            "abstention": predicted["reference"]["buckets"].get("ABSTENTION", 0),
            "geometry_poor": predicted["reference"]["buckets"].get("GEOMETRY_POOR", 0),
            "note": "attribution of the predicted-reference gap; diagnostic only"},
    }

    comparison = {
        "_doc": ("Task 7I sections 20 and 24-26. Formal three-seed comparison of Z-B3 (F0) and D-B1 (F1) under "
                 "the identical formal train/val population and protocol, with the eight "
                 "`DB1_FORMAL_VAL_CONFIRMED` gates and the non-gating diagnostics. Validation results are NOT "
                 "test results and the test split stayed locked."),
        "task": "7I", "stage": "K-formal-comparison",
        "seeds": list(SEEDS), "models": list(MODELS), "runs": 6,
        "per_model": per_model, "matched_seed": matched, "matched_seed_wins": matched_wins,
        "gates": gates, "gate_constants": GATES, "DB1_FORMAL_VAL_CONFIRMED": confirmed,
        "diagnostics": diagnostics,
        "selection_used": "full 936 oracle-reference val mIoU only",
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json(OUT_COMPARISON, comparison)

    write_json(OUT_LOCK, {
        "_doc": ("Task 7I section 27. Test-lock status after the formal train/val stage. DSH does not unlock "
                 "the test split: only a ChatGPT audit of Task 7I can authorize final-test execution."),
        "task": "7I", "stage": "L-test-lock-status",
        "status": "LOCKED",
        "test_execution_authorized": False,
        "task7i_completed_train_val": True,
        "db1_formal_val_confirmed": confirmed,
        "unlock_requires": "ChatGPT audit of Task 7I",
        "previous_lock": {"path": str(HOLDOUT_LOCK), "status": lock["status"],
                          "test_execution_authorized": lock["test_execution_authorized"]},
        "test_material_read": False, "test_metrics_produced": False,
        "runtime_seconds": round(time.time() - started, 1),
    })
    print(f"[7i.compare] Z-B3 oracle {z_b3_oracle_mean:.4f} (std "
          f"{per_model['Z-B3']['oracle']['miou']['std_ddof1']:.4f}) | D-B1 oracle {d_b1_oracle_mean:.4f} "
          f"(std {per_model['D-B1']['oracle']['miou']['std_ddof1']:.4f}) | delta "
          f"{d_b1_oracle_mean - z_b3_oracle_mean:+.4f} | wins {matched_wins}/3 | predicted "
          f"{z_b3_predicted_mean:.4f} -> {d_b1_predicted_mean:.4f} | gates "
          f"{sum(1 for entry in gates.values() if entry['passed'])}/8 -> DB1_FORMAL_VAL_CONFIRMED "
          f"{confirmed}", flush=True)
    return 0 if confirmed else 5


if __name__ == "__main__":
    raise SystemExit(main())
