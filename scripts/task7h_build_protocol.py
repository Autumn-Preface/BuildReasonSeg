"""Task 7H Parts D-F — freeze the limitations, the formal experiment protocol, the test lock and the claims.

Builds, from the audited evidence only:

* `evaluation/task7h_limitation_registry.json` — L-01 ... L-06;
* `evaluation/task7h_formal_experiment_protocol.json` — data policy, formal L3 population, the D-B1 schedule
  copied verbatim from the authoritative `evaluation/task7d_training.json`, the three formal seeds, the
  reference policy, baselines/ablations, required metrics and reporting rules;
* `evaluation/task7h_test_lock.json` — the exact LOCKED state;
* `evaluation/task7h_claim_registry.json` — C1 ... C9 with fixed statuses (no novelty/"first" claim);
* `docs/task7h_formal_experiment_protocol.md`.

Task 7H does not execute the protocol and does not unlock the test split.

    python scripts/task7h_build_protocol.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from buildreasonseg_mvp.checkpointing import write_json  # noqa: E402

EVAL = REPO_ROOT / "evaluation"
OUT_LIMITATIONS = EVAL / "task7h_limitation_registry.json"
OUT_PROTOCOL = EVAL / "task7h_formal_experiment_protocol.json"
OUT_TEST_LOCK = EVAL / "task7h_test_lock.json"
OUT_CLAIMS = EVAL / "task7h_claim_registry.json"
OUT_PROTOCOL_DOC = REPO_ROOT / "docs" / "task7h_formal_experiment_protocol.md"
EVIDENCE = EVAL / "task7h_evidence_registry.json"
TASK7D_TRAINING = EVAL / "task7d_training.json"
FORMAL_SEEDS = [20261001, 20261002, 20261003]
L3_PROGRAMS = ("largest_to_left_of_to_nearest", "largest_to_right_of_to_nearest",
               "largest_to_above_to_nearest", "largest_to_below_to_nearest")


def limitation_registry() -> dict:
    return {
        "_doc": ("Task 7H section 11. Frozen limitation registry for the BuildReasonSeg-DevFreeze-2026-10 "
                 "development architecture. Every limitation is recorded with its measured evidence and its "
                 "status; none of them is hidden or repaired by this task."),
        "task": "7H", "stage": "D-limitation-registry",
        "limitations": {
            "L-01": {
                "title": "Reference selection",
                "status": "unresolved practical bottleneck",
                "evidence": {"task7f_f_r0_strict_miou": 0.24540501038500215,
                             "task7f_f_r1_oracle_selection_miou": 0.36490938928549665,
                             "selection_gain": 0.1195043789004945,
                             "selection_fraction_of_total_gap": 0.8530487001786086,
                             "task7g_selector_adopted": False,
                             "task7g_internal_gate_passed": False,
                             "task7g_external_scene_disjoint_stage": "not executed"},
                "claim_boundary": "the current project version contains no learned selector that passed its "
                                  "adoption protocol; the deterministic U-C1 selector stays in the chain",
            },
            "L-02": {
                "title": "Proposal coverage",
                "status": "secondary unresolved bottleneck",
                "evidence": {"u_c1_best_eligible_coverage_at_0_50": 0.8460388639760837,
                             "coverage_gain": 0.054316262681561533,
                             "task7f_f_r2_coverage_rate": 0.8460388639760837,
                             "uncovered_records": 103},
                "claim_boundary": "103/669 holdout records have no eligible U-C1 proposal at IoU >= 0.50",
            },
            "L-03": {
                "title": "Proposal-mask geometry",
                "status": "not a major current bottleneck",
                "evidence": {"task7f_geometry_gain_covered": 0.004253166957657317,
                             "covered_subset_records": 566},
                "claim_boundary": "where coverage exists, the predicted proposal mask is nearly as good as "
                                  "the GT reference mask",
            },
            "L-04": {
                "title": "Free-form L3 language",
                "status": "controlled-language interface only",
                "evidence": {"task7c_canonical_full_val_accuracy": 1.0,
                             "task7c_canonical_macro_f1": 1.0,
                             "task7c_fixed24": "5/24", "task7c_fixed24_compact": "0/8",
                             "task7c_minimal96": "60/96",
                             "task7c_stress_accuracy": 0.6666666666666666,
                             "task7c_stress_macro_f1": 0.5924464770230274,
                             "task7c_stress_l3_macro_recall": 0.44791666666666663,
                             "task7c_verdict": "L3_20CLASS_COMPOSITIONAL_ROBUSTNESS_FAIL"},
                "claim_boundary": "canonical/program-template behaviour is verified; unrestricted free-form "
                                  "language is NOT verified and must not be claimed",
            },
            "L-05": {
                "title": "nearest-only L2",
                "status": "not validated as standalone final capability",
                "evidence": {"task6y_b2_minival_miou": 0.2864, "task6y_b2_paired": "6/20",
                             "task6y_verdict": "NEAREST_FIELD_NO_MEANINGFUL_GAIN"},
                "claim_boundary": "field semantics verified and a clear gain over the visual baseline was "
                                  "measured, but the absolute/generalization/counterfactual feasibility gate "
                                  "failed; nearest-only stays experimental/limited",
            },
            "L-06": {
                "title": "unseen-city / domain generalization",
                "status": "not established",
                "evidence": {"split_view": "scene_disjoint_v1",
                             "note": "the native split demonstrates raster/scene separation only"},
                "claim_boundary": "no unseen-city or cross-domain generalization result exists; it must not "
                                  "be claimed",
            },
        },
        "count": 6, "training_performed": False, "test_split_used": False,
    }


def formal_protocol() -> dict:
    task7d = json.loads(TASK7D_TRAINING.read_text(encoding="utf-8"))
    schedule = dict(task7d["training"])
    schedule["amp"] = "bfloat16 autocast + GradScaler (Task 7D D2 trainer)"
    schedule["checkpoint_selection"] = task7d["selection_metric"]
    schedule["fresh_initialisation"] = True
    schedule["sam2_frozen"] = True
    schedule["source_artifact"] = "evaluation/task7d_training.json"
    return {
        "_doc": ("Task 7H sections 12-19. Formal experiment protocol for BuildReasonSeg-DevFreeze-2026-10. "
                 "The D-B1 training schedule is copied verbatim from the authoritative Task 7D training "
                 "artifact; no value was invented. The protocol is NOT executed by Task 7H."),
        "task": "7H", "stage": "E-formal-experiment-protocol",
        "architecture_name": "BuildReasonSeg-DevFreeze-2026-10",
        "executed_by_task7h": False,
        "data_policy": {
            "active_canonical_only": True,
            "datasets": ["WHU-EA-NativeVector v1.0", "BuildSpatialReason v0.2", "scene_disjoint_v1"],
            "train": "gradient updates only",
            "val": "checkpoint selection / early stopping / frozen threshold-free model selection only",
            "test": "only after every architecture and training choice is frozen",
            "test_access_disclosure": {
                "historically_accessed": True,
                "when": "Task 6M, for an earlier proposal-baseline J4-v2 audit",
                "consequence": "future reporting must NOT call the project test split 'never previously "
                               "viewed' or 'completely untouched'; the permitted phrasing is 'final "
                               "frozen-architecture test evaluation'",
                "post_6m_architecture_selection_used_test_metrics": False,
            },
        },
        "formal_l3_population": {
            "programs": list(L3_PROGRAMS),
            "train": {"source": "BuildSpatialReason v0.2 train split, all valid records of the four "
                                "programs",
                      "expected_total": 1344,
                      "per_program": {"largest_to_left_of_to_nearest": 323,
                                      "largest_to_right_of_to_nearest": 347,
                                      "largest_to_above_to_nearest": 338,
                                      "largest_to_below_to_nearest": 336},
                      "verified_from": "tracked v0.2 train split (Task 7H audit, no test access)"},
            "val": {"source": "all valid v0.2 val L3 records", "expected_total": 936,
                    "verified_from": "tracked v0.2 val split"},
        },
        "formal_d_b1_training": {
            "architecture": "exact Task 7D D-B1 (deterministic field-weighted prototype)",
            "initialize_from": "fresh random trainable decoder/projection weights",
            "backbone": "frozen SAM2.1 Hiera Base+ features",
            "fields": "exact GeometricRelationField v0.2 (P_dir) and NearestBoundaryField v0.1 (P_near)",
            "loss": "BCE + Dice only", "new_losses": [], "architecture_tuning": False,
            "schedule": schedule,
            "formal_seeds": list(FORMAL_SEEDS),
            "run_all_three_in": "a later training task (not Task 7H)",
            "checkpoint_selection": "val mIoU only; each seed selected independently; no seed chosen by test",
        },
        "reference_policy": {
            "report_both_separately": True,
            "practical_predicted_reference_chain": ["U-C1 proposals",
                                                    "deterministic largest selector",
                                                    "D-B1 target mask"],
            "oracle_reference_diagnostic": ["GT reference", "D-B1 target mask"],
            "mixing_allowed": False,
            "task7f_f_r1_oracle_selected_proposal_as_production_result": False,
        },
        "baselines_and_ablations": {
            "B-L3-0": {"name": "Z-B3", "definition": "visual + P_dir + P_near + direction embedding",
                       "role": "baseline/ablation"},
            "B-L3-1": {"name": "D-B1", "definition": "deterministic relation-conditioned prototype",
                       "role": "main target-decoder architecture candidate"},
            "historical_causal_controls": [
                "Task 6Z visual-only / directional-only / nearest-only / geometry-only",
                "Task 7D learned competition D-B2", "Task 7D map-only D-B4"],
            "new_retrospective_ablation_in_task7h": False,
            "documentation_must_distinguish": ["historical development ablations",
                                               "future formally retrained seeds",
                                               "diagnostic oracle-reference values"],
        },
        "required_metrics": {
            "mask": ["mIoU", "Dice", "Pr@0.5"],
            "counterfactual": ["pair pass rate", "own IoU", "cross IoU", "own-cross margin"],
            "reference": ["selected-reference mIoU", "reference Pr@0.5", "abstention rate",
                          "NO_PROPOSALS", "NO_ELIGIBLE", "NOT_COVERED", "SELECTION_WRONG",
                          "GEOMETRY_POOR", "REFERENCE_OK"],
            "per_relation": ["left", "right", "above", "below"],
            "stratification": ["target area quartiles", "reference quality bins",
                               "boundary-distance quartiles"],
            "efficiency": ["train wall time", "peak VRAM", "inference time/tile",
                           "trainable parameter count"],
        },
        "three_seed_reporting": {
            "report_each_seed_separately": True,
            "report_mean_and_std_on": ["val", "final test"],
            "best_test_seed_only": False,
            "crash_policy": "document it and resume only from that run's own checkpoint/state; never replace "
                            "the seed",
        },
        "test_lock": {"status": "LOCKED", "test_execution_authorized": False,
                      "artifact": "evaluation/task7h_test_lock.json",
                      "unlock_condition": "ChatGPT audit after formal train/val completion"},
        "training_performed": False, "test_split_used": False,
    }


def test_lock() -> dict:
    return {
        "_doc": ("Task 7H section 20. Test lock for BuildReasonSeg-DevFreeze-2026-10. Any future DSH task "
                 "must read this file before executing a test evaluation; Task 7H does not unlock it."),
        "task": "7H", "stage": "E-test-lock",
        "status": "LOCKED",
        "architecture_head": "D-B1",
        "reference_policy": "U-C1 deterministic largest",
        "parser_role": "controlled-language/canonical interface",
        "formal_seeds": list(FORMAL_SEEDS),
        "test_execution_authorized": False,
        "unlock_condition": "ChatGPT audit after formal train/val completion",
        "checked_before_test_execution": True,
        "test_access_disclosure": "the scene_disjoint_v1 test split was accessed once in Task 6M for an "
                                  "earlier proposal-baseline J4-v2 audit; post-6M architecture selection did "
                                  "not use test metrics, so future reporting must call the evaluation "
                                  "'final frozen-architecture test evaluation'",
        "task7h_test_records_read": False,
        "training_performed": False, "test_split_used": False,
    }


def claim_registry() -> dict:
    return {
        "_doc": ("Task 7H section 21. Claim registry for BuildReasonSeg-DevFreeze-2026-10. Every claim is "
                 "tagged SUPPORTED / SUPPORTED_WITH_LIMITATION / NOT_SUPPORTED / DIAGNOSTIC_ONLY. No "
                 "novelty or 'first' claim appears anywhere in this registry."),
        "task": "7H", "stage": "F-claim-registry",
        "allowed_statuses": ["SUPPORTED", "SUPPORTED_WITH_LIMITATION", "NOT_SUPPORTED",
                             "DIAGNOSTIC_ONLY"],
        "novelty_or_first_claim": False,
        "claims": {
            "C1": {"statement": "Explicit reference-conditioned directional geometric fields improve "
                                "same-class building target segmentation.",
                   "status": "SUPPORTED", "evidence": ["Task 6N", "Task 6O"]},
            "C2": {"statement": "Directional and nearest geometric priors can be composed for L3 "
                                "direction-to-nearest reasoning.",
                   "status": "SUPPORTED_WITH_LIMITATION",
                   "evidence": ["Task 6Z paired strength", "Task 7E D-B1 oracle holdout gain",
                                "practical performance remains reference-limited"]},
            "C3": {"statement": "Relation-conditioned deterministic spatial weighting can extract a useful "
                                "target visual prototype.",
                   "status": "SUPPORTED",
                   "evidence": ["Task 7D D-B1", "Task 7E untouched oracle-reference holdout"]},
            "C4": {"statement": "Learned global competition is superior.",
                   "status": "NOT_SUPPORTED", "evidence": ["Task 7D D-B2"]},
            "C5": {"statement": "The practical end-to-end system solves unrestricted natural-language L3 "
                                "segmentation.",
                   "status": "NOT_SUPPORTED",
                   "evidence": ["Task 7C language robustness", "Task 7E practical chain"]},
            "C6": {"statement": "Reference selection is the dominant practical bottleneck.",
                   "status": "SUPPORTED_WITH_LIMITATION",
                   "evidence": ["Task 7F ceilings", "no learned replacement passed the Task 7G adoption "
                                                      "gate"]},
            "C7": {"statement": "Proposal-mask geometry is the main reference problem.",
                   "status": "NOT_SUPPORTED",
                   "evidence": ["Task 7F geometry gain +0.004253"]},
            "C8": {"statement": "Performance generalizes to unseen cities.",
                   "status": "NOT_SUPPORTED", "evidence": ["no cross-city evaluation exists"]},
            "C9": {"statement": "D-B1 is the final paper-ready architecture.",
                   "status": "NOT_SUPPORTED",
                   "evidence": ["D-B1 is a frozen development architecture candidate pending formal "
                                "retraining and the locked test evaluation"]},
        },
        "count": 9, "training_performed": False, "test_split_used": False,
    }


def protocol_doc(protocol: dict, limitations: dict, lock: dict, claims: dict) -> str:
    schedule = protocol["formal_d_b1_training"]["schedule"]
    return f"""# Task 7H — Formal Experiment Protocol (BuildReasonSeg-DevFreeze-2026-10)

> Frozen by Task 7H · **not executed by Task 7H** · machine-readable form:
> `evaluation/task7h_formal_experiment_protocol.json` · test lock: `evaluation/task7h_test_lock.json`
> Architecture freeze: `docs/task7h_development_architecture_freeze.md`

## 1. Data policy

Only the active canonical assets: **WHU-EA-NativeVector v1.0**, **BuildSpatialReason v0.2**,
**`scene_disjoint_v1`**.

* **train** — gradient updates only;
* **val** — checkpoint selection / early stopping / frozen threshold-free model selection only;
* **test** — only after every architecture and training choice is frozen.

**Historical access disclosure.** The `scene_disjoint_v1` test split was accessed once in **Task 6M** for an
earlier proposal-baseline J4-v2 audit. Future reporting must therefore **not** call the project test split
"never previously viewed" or "completely untouched"; the permitted phrasing is
**`final frozen-architecture test evaluation`**, because post-6M architecture selection did not use test
metrics. Task 7H itself read no test record.

## 2. Formal L3 population

The four canonical L3 programs `largest_to_{{left_of,right_of,above,below}}_to_nearest`.

| Split | Population | Expected |
|---|---|---|
| train | all valid v0.2 train records of the four programs | **1344** (left 323 / right 347 / above 338 / below 336) |
| val | all valid v0.2 val L3 records | **936** |

Both totals were re-verified in Task 7H from the tracked train/val splits (no test access).

## 3. Formal D-B1 training

* architecture: the exact frozen Task 7D D-B1 (deterministic field-weighted prototype), initialized from
  **fresh random trainable weights**, SAM2 frozen, exact field formulas, **BCE + Dice only**, no new loss and
  no architecture tuning;
* schedule copied verbatim from the authoritative `evaluation/task7d_training.json`:
  **optimizer AdamW · lr {schedule['lr']} · weight decay {schedule['weight_decay']} · batch {schedule['batch']}
  · max epochs {schedule['max_epochs']} · early-stopping patience {schedule['patience']} ·
  AMP {schedule['amp']} · checkpoint selection `{schedule['checkpoint_selection']}`**;
* formal seeds **{FORMAL_SEEDS}** — all three run only in a later training task;
* checkpoint selection: **val mIoU only**, each seed selected independently, never by test.

## 4. Reference policy in formal evaluation

Two results must be reported **separately** and never mixed into one headline number:

```text
practical predicted-reference chain:  U-C1 -> deterministic largest selector -> D-B1
oracle-reference diagnostic:          GT reference -> D-B1
```

The Task 7F F-R1 oracle-selected proposal is **not** a production result.

## 5. Baselines and ablations

| ID | System | Definition | Role |
|---|---|---|---|
| B-L3-0 | Z-B3 | visual + P_dir + P_near + direction embedding | baseline/ablation |
| B-L3-1 | D-B1 | deterministic relation-conditioned prototype | main architecture candidate |
| historical | Task 6Z visual-only / directional-only / nearest-only / geometry-only · Task 7D D-B2 · Task 7D D-B4 | frozen development controls | historical ablations only |

Documentation must keep **historical development ablations**, **future formally retrained seeds** and
**diagnostic oracle-reference values** clearly apart. Task 7H invents no new retrospective ablation.

## 6. Required metrics

* **mask**: mIoU, Dice, Pr@0.5;
* **counterfactual**: pair pass rate, own IoU, cross IoU, own-cross margin;
* **reference**: selected-reference mIoU, reference Pr@0.5, abstention rate, `NO_PROPOSALS`, `NO_ELIGIBLE`,
  `NOT_COVERED`, `SELECTION_WRONG`, `GEOMETRY_POOR`, `REFERENCE_OK`;
* **per relation**: left, right, above, below;
* **stratification**: target-area quartiles, reference-quality bins, boundary-distance quartiles;
* **efficiency**: train wall time, peak VRAM, inference time/tile, trainable parameter count.

## 7. Three-seed reporting

Each seed is reported separately, with mean ± standard deviation on val and on the final test; reporting only
the best test seed is forbidden. A crashed run is documented and resumed only from that run's own
checkpoint/state — the seed is never replaced.

## 8. Test lock

```text
status                     = {lock['status']}
architecture_head          = {lock['architecture_head']}
reference_policy           = {lock['reference_policy']}
parser_role                = {lock['parser_role']}
formal_seeds               = {lock['formal_seeds']}
test_execution_authorized  = {str(lock['test_execution_authorized']).lower()}
unlock_condition           = {lock['unlock_condition']}
```

Any future DSH task must read `evaluation/task7h_test_lock.json` before executing a test evaluation. Task 7H
does not unlock it.

## 9. Frozen limitations carried into the formal protocol

| ID | Limitation | Status |
|---|---|---|
| L-01 | Reference selection | {limitations['limitations']['L-01']['status']} |
| L-02 | Proposal coverage | {limitations['limitations']['L-02']['status']} |
| L-03 | Proposal-mask geometry | {limitations['limitations']['L-03']['status']} |
| L-04 | Free-form L3 language | {limitations['limitations']['L-04']['status']} |
| L-05 | nearest-only L2 | {limitations['limitations']['L-05']['status']} |
| L-06 | unseen-city/domain generalization | {limitations['limitations']['L-06']['status']} |

## 10. Claim statuses (no novelty or "first" claim)

| Claim | Status |
|---|---|
""" + "\n".join(f"| {cid} | {entry['status']} |" for cid, entry in claims["claims"].items()) + """

Final recommendation (Task 7H, Part L):

`等待 ChatGPT 审核 Task 7H 的开发版架构冻结与正式实验协议；在审核通过前不启动正式三种子训练，不解锁 test。`
"""


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    started = time.time()
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if not evidence["hash_checks"]["all_required_match"]:
        write_json(OUT_LIMITATIONS, {"_doc": "Task 7H section 22.", "task": "7H",
                                     "verdict": "FROZEN_ASSET_HASH_MISMATCH"})
        print("[7h.protocol] STOP FROZEN_ASSET_HASH_MISMATCH", flush=True)
        return 3

    limitations = limitation_registry()
    protocol = formal_protocol()
    lock = test_lock()
    claims = claim_registry()
    write_json(OUT_LIMITATIONS, limitations)
    write_json(OUT_PROTOCOL, protocol)
    write_json(OUT_TEST_LOCK, lock)
    write_json(OUT_CLAIMS, claims)
    OUT_PROTOCOL_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROTOCOL_DOC.write_text(protocol_doc(protocol, limitations, lock, claims), encoding="utf-8")
    print(f"[7h.protocol] limitations {limitations['count']} | protocol {'complete' if protocol else ''} | "
          f"test lock {lock['status']} authorized={lock['test_execution_authorized']} | claims "
          f"{claims['count']} statuses "
          f"{sorted({entry['status'] for entry in claims['claims'].values()})} | doc written "
          f"({OUT_PROTOCOL_DOC.name})", flush=True)
    del started
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
