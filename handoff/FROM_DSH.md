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

# FROM_DSH — Task 8B.3-P1D11A-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D11A-R1` |
| Status | **STOP** (canonical manifest is mixed-convention) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `9f91d953b9ef5bb3993be4b7434b174de9856648` |
| Four previously flagged entries | **CRLF false positive CONFIRMED** — they match the Git index bytes exactly |
| Full manifest under Git index convention | 97/135 match |
| Full manifest under working-tree convention | 131/135 match |
| Entries matching both conventions | 93 |
| Entries matching neither | 0 |
| New finding | the manifest mixes Git-index (LF) and working-tree (CRLF) byte conventions, so no single convention reaches 135/135 |
| Policy doc / README / model card / manifest | NOT CREATED / NOT MODIFIED / NOT MODIFIED / NOT MODIFIED |
| Supported-domain policy | NOT IMPLEMENTED (blocked) |
| A2 policy status | unchanged (`DOCUMENTED_PERSISTENT_NON_DETECTION` as recorded in the P1D10-R1 report) |
| Model / test / runtime changes | NONE |
| External write sync | NOT PERFORMED |
| PROP-01 status | OPEN (not closed) |
| Locked candidates | 4 immutable v0.2 TEST sample IDs (unchanged) |
| Report | `docs/task8b3_p1d11a_policy_implementation.md` |
| STOP reason | the mandated pre-edit self-check cannot be 135/135 under any single byte convention because the manifest mixes index-blob and working-tree identities |
| Next action | Awaiting ChatGPT audit; canonical manifest normalisation needs a dedicated task book before the policy implementation is retried |

Watt was not needed for Task 8B.3-P1D11A-R1 (no downloads, no transfers).

Only this report and `handoff/FROM_DSH.md` changed; no canonical document, manifest entry, runtime file, model or
candidate was touched and nothing was synced to the external delivery.
