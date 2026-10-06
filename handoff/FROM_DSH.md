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

# FROM_DSH — Governance V1 Executor-Neutral Handoff

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history. Governance V1 adds the executor-neutral protocol files; this legacy file is retained._

| item | value |
|---|---|
| Task | `GOVERNANCE_V1_EXECUTOR_NEUTRAL_HANDOFF` |
| Status | **COMPLETE** |
| Starting branch / head | `delivery/task8b3-mask01-d2-external-sync-regression` / `1e9ac792d901b44db8ed33a09ee1552e11a865d9` |
| Task branch | `docs/governance-v1-executor-neutral-handoff` |
| New protocol files | `AGENTS.md`, `governance/PROJECT_STATE.yaml`, `governance/DECISIONS.md`, `handoff/CURRENT_TASK.md`, `handoff/EXECUTOR_STATE.yaml` |
| YAML parse pass / files present / state consistency | True / True / True |
| Product paths touched | NONE |
| Legacy `TO_DSH.md` / `FROM_DSH.md` | KEPT (not deleted, not renamed) |
| Scientific state | MASK-01 CLOSED · PROP-01 OPEN · REF-01 residual — no task auto-started |
| CURRENT_TASK status | `AWAITING_SUPERVISOR` (authorized executor = ANY) |
| Evidence | `evaluation\governance_v1_executor_neutral_handoff.json` |
| Report | `docs\governance_v1_executor_neutral_handoff.md` |
| Next gate | `CHATGPT_GOVERNANCE_V1_REMOTE_AUDIT` |

Watt was not needed for GOVERNANCE_V1_EXECUTOR_NEUTRAL_HANDOFF (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
