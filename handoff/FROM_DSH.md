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

# FROM_DSH — Task MASK01_D1_R1A_STATE_PERSISTENCE Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1A_STATE_PERSISTENCE` |
| Status | **STOP** (R1-A gate not green; real state now persisted for independent review) |
| Branch | `fix/task8b3-mask01-success-semantics-r1a` |
| predict.py implemented | single-image annotation True · note line True · batch annotation True |
| tests/test_cli_contract.py | weak D1 test removed True · 19 real contract tests |
| R1-A gate | `pytest tests/test_cli_contract.py -q` → exit 1 · 1 failed, 22 passed |
| Failing test | `test_predict_inspect_proposals_does_not_require_prompt` (30 != 20, E202 missing) |
| Attribution | PRE_EXISTING, not introduced by R1-A |
| pipeline.py / test_task8b_runtime.py / source_manifest.json / docs | UNCHANGED |
| Model inference / training / external sync / full suite | NONE / NONE / NONE / NOT RUN |
| Evidence | `evaluation\task8b3_mask01_d1_r1a_state_persistence.json` |
| Report | `docs\task8b3_mask01_d1_r1a_state_persistence.md` |
| Next action | Awaiting ChatGPT independent review; next task not entered |

Watt was not needed for Task MASK01_D1_R1A_STATE_PERSISTENCE (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; the local and remote heads are printed after the push.
