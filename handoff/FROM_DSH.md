<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task MASK01_D1_R1E2_FULL_SUITE_FAILURE_TRIAGE Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1E2_FULL_SUITE_FAILURE_TRIAGE` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1e1` / `3320212a6992316196bc5b2cfacaeaf027fd070e` |
| Task branch | `audit/task8b3-mask01-r1e2-full-suite-failure-triage` |
| Missing canonical assets | 7 (see report inventory) |
| Group exits | G1=1 · G2=1 · G3=1 · G4=1 · G5=1 |
| Classified nodes | 32 |
| Classification counts | {"D": 12, "A": 15, "B": 5} |
| Validity decision | **FULL_SUITE_CANONICAL_GATE = VALID_AND_HAS_TRUE_REGRESSION** |
| Recommendation | keep the complete-suite gate on the canonical source tree |
| full suite re-executed / repairs / canonical modifications | false / none / none |
| assets created / dependencies installed / inference / external write | false / false / false / false |
| Evidence | `evaluation\task8b3_mask01_d1_r1e2_full_suite_failure_triage.json` |
| Report | `docs\task8b3_mask01_d1_r1e2_full_suite_failure_triage.md` |
| Next gate | `CHATGPT_R1E2_TRUE_REGRESSION_TRIAGE` |

Watt was not needed for Task MASK01_D1_R1E2_FULL_SUITE_FAILURE_TRIAGE (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
