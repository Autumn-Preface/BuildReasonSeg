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

# FROM_DSH — Task 8B.3-REF01-E3C0 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3C0` |
| Status | **COMPLETE** (read-only saved-artifact audit) |
| Base branch / head | `fix/task8b3-ref01-eligibility-repair-sync` / `1197d860b8b1a3503c6cba730eda33ccbff4e1c5` |
| Audit branch | `audit/task8b3-ref01-locked-replay-artifacts` |
| Detector / model inference | NONE |
| Proposal regeneration / real replay | NONE / NONE |
| External write | NONE |
| Product / test / manifest / helper modified | NO |
| Per-case verdicts | right: SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU, left: SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU, above: SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU, below: SAVED_ARTIFACT_REPLAY_READY_FOR_REFERENCE_IOU |
| Reference-IoU replay ready (4/4) | True |
| End-to-end replay ready | False |
| Replay gaps | language parse output, relation-field output, SAM2 reference mask, D-B1 decoder output, final composite mask |
| Evidence | `evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json` |
| Report | `docs/task8b3_ref01_locked_replay_artifact_audit.md` |
| Overall outcome | **REFERENCE_IOU_REPLAY_READY_END_TO_END_NOT_READY** |
| Next gate | `REF01_LOCKED_REPLAY_ARTIFACT_COMPLETION` (not executed) |
| Next action | Awaiting ChatGPT audit |

Watt was not needed for Task 8B.3-REF01-E3C0 (no downloads, no transfers).

The audit read, hashed and inventoried saved artifacts only; nothing was executed, regenerated, replayed or written
outside the repository.
