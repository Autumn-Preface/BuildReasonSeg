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

# FROM_DSH — Task MASK01_D1_R1A_R1_CLI_CONTRACT_CORRECTION Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1A_R1_CLI_CONTRACT_CORRECTION` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1a` / `1e2282151aa8c0c3cbbdc63526d8e449dc7b655d` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1a-r1` |
| Single-image SUCCESS lines | Result / Validity / Semantic / Note (four exact lines) |
| Single-image failure lines | only `Result       : <status>` |
| Batch summary | `Runtime success: <n>` |
| Tests | `pytest tests/test_cli_contract.py -q -k "not test_predict_inspect_proposals_does_not_require_prompt"` → exit 0 · 23 passed, 1 deselected in 12.73s |
| Source-string checks replaced | YES (behaviour tests over fake payloads) |
| pipeline.py / manifest / test_task8b_runtime.py / docs | UNCHANGED |
| inspect-proposals 30-vs-20 defect | NOT fixed (out of scope) |
| Model inference / training / external sync | NONE |
| Evidence | `evaluation\task8b3_mask01_d1_r1a_r1_cli_contract_correction.json` |
| Report | `docs\task8b3_mask01_d1_r1a_r1_cli_contract_correction.md` |
| Next action | Awaiting ChatGPT review; next task not entered |

Watt was not needed for Task MASK01_D1_R1A_R1_CLI_CONTRACT_CORRECTION (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
