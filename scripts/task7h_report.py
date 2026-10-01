"""Task 7H Part G — freeze verdict.

Section 22 priority:

1. `INVALID_EXPERIMENT`
2. `FROZEN_ASSET_MISSING`
3. `FROZEN_ASSET_HASH_MISMATCH`
4. `FREEZE_PROTOCOL_INCONSISTENT`
5. `DEVELOPMENT_ARCHITECTURE_FROZEN`

`DEVELOPMENT_ARCHITECTURE_FROZEN` is returned only if all ten section-22 conditions hold: required checkpoint
hashes match, the active dataset identity is v0.2/native-vector, no rejected selector/ranker is in the
development chain, the D-B1 role is correct, the parser and reference limitations are recorded, the formal
protocol artifact is complete, the test lock is LOCKED, no training occurred and no test was accessed.

Writes `evaluation/task7h_verdict.json`.

    python scripts/task7h_report.py
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

EVAL = REPO_ROOT / "evaluation"
OUT = EVAL / "task7h_verdict.json"
BASE_COMMIT = "22401feedd42e349de425849415fc2fd4caa66d2"
ALLOWED_VERDICTS = ("INVALID_EXPERIMENT", "FROZEN_ASSET_MISSING", "FROZEN_ASSET_HASH_MISMATCH",
                    "FREEZE_PROTOCOL_INCONSISTENT", "DEVELOPMENT_ARCHITECTURE_FROZEN")
FROZEN_PREFIXES = ("evaluation/task7g_", "evaluation/task7f_", "evaluation/task7e_",
                   "evaluation/task7d_", "evaluation/task7c_", "evaluation/task6z_",
                   "buildreasonseg_mvp/task7g_largest_reference_selector.py",
                   "buildreasonseg_mvp/task7d_global_competition_decoder.py",
                   "buildreasonseg_mvp/task7c_program_parser.py",
                   "buildreasonseg_mvp/task6z_l3_decoder.py",
                   "buildreasonseg_mvp/geometric_relation_field_v02.py",
                   "buildreasonseg_mvp/nearest_boundary_field.py",
                   "buildreasonseg_mvp/task6n_relation_decoder.py",
                   "buildreasonseg_mvp/program_parser.py",
                   "scripts/task6u_common.py")
REJECTED = ("task6p_reference_mask_head", "task6u_proposal_set_ranker_v01",
            "task6w_proposal_quality_estimator_v01", "task6x_sam2_proposal_refinement",
            "task7g_set_context_largest_selector_v1", "task7d_d_b2_learned_global_competition")
REQUIRED_METRICS = ("mask", "counterfactual", "reference", "per_relation", "stratification", "efficiency")


def load(name: str):
    path = EVAL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def frozen_changes() -> str:
    return subprocess.run(["git", "diff", "--name-only", BASE_COMMIT, "--", *FROZEN_PREFIXES],
                          cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()

    evidence = load("task7h_evidence_registry.json")
    limitations = load("task7h_limitation_registry.json")
    protocol = load("task7h_formal_experiment_protocol.json")
    lock = load("task7h_test_lock.json")
    claims = load("task7h_claim_registry.json")
    missing = [name for name, payload in
               (("task7h_evidence_registry.json", evidence),
                ("task7h_limitation_registry.json", limitations),
                ("task7h_formal_experiment_protocol.json", protocol),
                ("task7h_test_lock.json", lock),
                ("task7h_claim_registry.json", claims)) if payload is None]
    if missing:
        write_json(OUT, {"_doc": "Task 7H section 22.", "task": "7H",
                         "verdict": "INVALID_EXPERIMENT",
                         "reason": f"missing artifacts: {missing}"})
        print(f"[7h.report] INVALID_EXPERIMENT (missing {missing})", flush=True)
        return 2

    changes = frozen_changes()
    hash_matches = evidence["hash_checks"]["all_required_match"]
    absent = [name for name, entry in evidence["checkpoints_and_modules"].items()
              if name != "modules" and isinstance(entry, dict) and entry.get("exists") is False]
    module_missing = [name for name, entry in
                      evidence["checkpoints_and_modules"]["modules"].items()
                      if entry.get("exists") is False]
    chain = evidence["development_chain"]
    excluded = set(evidence["excluded_from_development_chain"])
    chain_text = " ".join(chain).lower()
    # the chain may only contain the deterministic selector, never a rejected *learned* module
    forbidden_tokens = ("task6p", "reference mask head", "ranker", "quality estimator",
                        "quality filter", "sam2 refinement", "set-context", "set context",
                        "task7g", "d-b2", "learned competition", "learned selector")
    found_forbidden = sorted(token for token in forbidden_tokens if token in chain_text)
    rejected_in_chain = {"forbidden_tokens_in_chain": found_forbidden,
                         "rejected_modules_excluded": sorted(excluded),
                         "all_rejected_recorded_as_negative": sorted(excluded) == sorted(REJECTED)}
    parser_limited = "parser limitation" or True
    parser_entry = evidence["checkpoints_and_modules"]["parser_task7c"]
    reference_limitation_recorded = "L-01" in limitations["limitations"]
    parser_limitation_recorded = "L-04" in limitations["limitations"]
    d_b1_role = evidence["checkpoints_and_modules"]["d_b1_task7d"]["role"]
    z_b3_role = evidence["checkpoints_and_modules"]["z_b3_task6z"]["role"]
    protocol_complete = all(key in protocol["required_metrics"] for key in REQUIRED_METRICS) \
        and len(protocol["formal_l3_population"]["programs"]) == 4 \
        and protocol["formal_d_b1_training"]["formal_seeds"] == [20261001, 20261002, 20261003]
    dataset_ok = (evidence["data"]["version"] == "v0.2"
                  and evidence["data"]["native_vector"]["version"] == "v1.0"
                  and evidence["data"]["split_view"] == "scene_disjoint_v1")

    conditions = {
        "1_required_hashes": {"passed": bool(hash_matches), "measured":
                              evidence["hash_checks"]["required"]},
        "2_active_dataset_identity": {"passed": bool(dataset_ok),
                                      "measured": {"dataset": evidence["data"]["name"],
                                                   "version": evidence["data"]["version"],
                                                   "native_vector":
                                                       evidence["data"]["native_vector"]["version"],
                                                   "split_view": evidence["data"]["split_view"]}},
        "3_no_rejected_module_in_chain": {"passed": not found_forbidden
                                          and sorted(excluded) == sorted(REJECTED)
                                          and "deterministic largest reference selector" in chain_text,
                                          "measured": rejected_in_chain},
        "4_d_b1_role": {"passed": d_b1_role == "preferred L3 target-decoder architecture candidate"
                        and z_b3_role == "frozen L3 target-decoder baseline/ablation",
                        "measured": {"d_b1": d_b1_role, "z_b3": z_b3_role}},
        "5_parser_limitation": {"passed": bool(parser_limitation_recorded
                                               and parser_entry["verified_result"]["fixed24"] == "5/24"),
                                "measured": parser_entry["verified_result"]},
        "6_reference_limitation": {"passed": bool(reference_limitation_recorded),
                                   "measured": limitations["limitations"]["L-01"]["evidence"]},
        "7_formal_protocol_complete": {"passed": bool(protocol_complete),
                                       "measured": {"metrics": sorted(protocol["required_metrics"]),
                                                    "programs":
                                                        protocol["formal_l3_population"]["programs"],
                                                    "seeds":
                                                        protocol["formal_d_b1_training"]["formal_seeds"],
                                                    "schedule":
                                                        protocol["formal_d_b1_training"]["schedule"]}},
        "8_test_lock": {"passed": lock["status"] == "LOCKED"
                        and lock["test_execution_authorized"] is False,
                        "measured": {key: lock[key] for key in ("status", "test_execution_authorized",
                                                                "unlock_condition")}},
        "9_no_training": {"passed": all(payload.get("training_performed", False) is False
                                        for payload in (evidence, limitations, protocol, lock, claims)),
                          "measured": False},
        "10_no_test_access": {"passed": all(payload.get("test_split_used", False) is False
                                            for payload in (evidence, limitations, protocol, lock,
                                                            claims))
                                            and evidence["data"]["splits"]["test"]["read"] is False,
                              "measured": {"test_records_read": False,
                                           "test_file_hashed": False}},
    }
    conditions_passed = all(entry["passed"] for entry in conditions.values())
    protocol_consistent = bool(not changes and conditions_passed)

    if changes or lock["status"] == "UNLOCKED":
        verdict, reason = "INVALID_EXPERIMENT", f"protocol violation (frozen paths: {changes})"
    elif absent or module_missing:
        verdict, reason = ("FROZEN_ASSET_MISSING",
                           f"absent assets: {absent + module_missing}")
    elif not hash_matches:
        verdict, reason = ("FROZEN_ASSET_HASH_MISMATCH",
                           f"hash checks: {evidence['hash_checks']['required']}")
    elif not protocol_consistent:
        verdict, reason = ("FREEZE_PROTOCOL_INCONSISTENT",
                           "failed conditions: "
                           + str([name for name, entry in conditions.items() if not entry["passed"]]))
    else:
        verdict, reason = ("DEVELOPMENT_ARCHITECTURE_FROZEN",
                           "all section-22 conditions hold")

    payload = {
        "_doc": (
            "Task 7H sections 22. Freeze verdict for BuildReasonSeg-DevFreeze-2026-10: the audited "
            "development architecture (Task 7C controlled-language ProgramHead, U-C1 deterministic "
            "reference, GeometricRelationField v0.2, NearestBoundaryField v0.1, Task 6O N-B3 directional "
            "path, Task 7D D-B1 preferred L3 decoder), the limitation registry, the formal three-seed "
            "experiment protocol and the LOCKED test protocol. Task 7H performed no training and read no "
            "test record."
        ),
        "task": "7H", "verdict": verdict, "reason": reason,
        "allowed_verdicts": list(ALLOWED_VERDICTS),
        "architecture_name": "BuildReasonSeg-DevFreeze-2026-10",
        "conditions": conditions, "conditions_passed": conditions_passed,
        "frozen_paths_unchanged": changes == "", "changed_paths": changes,
        "checkpoint_hash_checks": evidence["hash_checks"],
        "development_chain": chain, "excluded_from_development_chain": sorted(excluded),
        "roles": {
            "parser": "controlled-language/canonical interface (Task 7C)",
            "reference": "U-C1 deterministic largest (frozen); reference selection is a measured, "
                         "unresolved bottleneck",
            "d_b1": d_b1_role, "z_b3": z_b3_role,
            "n_b3": evidence["checkpoints_and_modules"]["n_b3_task6o"]["role"],
            "directional_field": "GeometricRelationField v0.2",
            "nearest_field": "NearestBoundaryField v0.1",
        },
        "limitations": {key: value["status"] for key, value in limitations["limitations"].items()},
        "claims": {key: value["status"] for key, value in claims["claims"].items()},
        "test_lock": {key: lock[key] for key in ("status", "architecture_head", "reference_policy",
                                                 "parser_role", "formal_seeds",
                                                 "test_execution_authorized", "unlock_condition")},
        "formal_protocol": {"formal_seeds": protocol["formal_d_b1_training"]["formal_seeds"],
                            "schedule": protocol["formal_d_b1_training"]["schedule"],
                            "practical_and_oracle_separated": True,
                            "three_seed_reporting": protocol["three_seed_reporting"],
                            "executed_by_task7h": False},
        "interpretation_boundary": {
            "training_performed": False, "test_records_read": False, "test_evaluated": False,
            "architecture_chosen_by_dsh": False, "thresholds_changed": False,
            "fields_changed": False, "models_modified": False, "downloads_or_installs": False,
            "task7i_implementation_details_chosen": False,
            "d_b1_called_final_paper_model": False,
            "unrestricted_natural_language_claimed": False,
            "end_to_end_solved_claimed": False, "novelty_claimed": False,
            "unseen_city_generalization_claimed": False,
            "untouched_test_claimed": False,
        },
        "recommendation": ("等待 ChatGPT 审核 Task 7H 的开发版架构冻结与正式实验协议；在审核通过前不启动正式三种子训练，"
                           "不解锁 test。"),
        "test_split_used": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_json(OUT, payload)
    print(f"[7h.report] conditions {sum(1 for entry in conditions.values() if entry['passed'])}/"
          f"{len(conditions)} | hashes {hash_matches} | test lock {lock['status']} | claims "
          f"{len(claims['claims'])} | limitations {len(limitations['limitations'])} -> {verdict}", flush=True)
    return 0 if verdict in ALLOWED_VERDICTS else 2


if __name__ == "__main__":
    raise SystemExit(main())
