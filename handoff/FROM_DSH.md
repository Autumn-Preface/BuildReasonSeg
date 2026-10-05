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

# FROM_DSH — Task MASK01_D1_SUCCESS_SEMANTICS_HARDENING Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_SUCCESS_SEMANTICS_HARDENING` |
| Status | **STOP** |
| Starting branch / head | `audit/task8b3-mask01-f2-locked-success-artifacts` / `a98ccecce20585dc37520523cd32b49b0a684248` |
| Task branch | `fix/task8b3-mask01-success-semantics` |
| Design | `SUCCESS_SEMANTICS_HARDENING_V1` |
| validity_scope / semantic_status | `RUNTIME_STRUCTURAL_ONLY` / `NOT_EVALUATED` |
| normal success status changed | NO |
| PipelineResult.ok / exit code / inspect mode changed | NO / NO / NO |
| New numeric thresholds | NONE |
| Manifest entries updated | [] |
| Manifest all-entry validation | FAIL |
| Targeted tests | exit 1 · 1 failed, 61 passed in 17.60s |
| Full canonical suite | exit 1 · 17 failed, 103 passed, 6 errors in 38.42s |
| External RC1 written / inference | NO / NO |
| Evidence | `evaluation\task8b3_mask01_d1_success_semantics_hardening.json` |
| Report | `docs\task8b3_mask01_d1_success_semantics_hardening.md` |
| next_gate | `MASK01_D2_EXTERNAL_SYNC_AND_REGRESSION` (NOT executed) |
| Next action | Awaiting ChatGPT review; D2 not entered |

Watt was not needed for Task MASK01_D1_SUCCESS_SEMANTICS_HARDENING (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; the local and remote heads are printed after the push.
