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

# FROM_DSH — Task 8B.3-REF01-E3C0-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3C0-R1` |
| Status | **COMPLETE** (bounded read-only audit correction) |
| Branch | `audit/task8b3-ref01-locked-replay-artifacts` |
| Starting HEAD | `bde13bd15bbab3e455ea1d3ef10b3f6740fc110d` |
| Discarded forbidden roots | `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\artifacts`, `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\inference`, `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs` |
| Allowed search roots | repo `evaluation/docs/handoff/scripts` · external `inference/output`, `docs`, root-level text files |
| Candidate artifacts | 16 |
| Detector / model inference | NONE |
| Proposal regeneration / actual replay | NO / NO |
| External write / product change | NO / NO |
| Per-case scalar fields complete | right:True, left:True, above:True, below:True |
| Per-case exact mask material present | right:False, left:False, above:False, below:False |
| Overall readiness (frozen enum) | **LOCKED_REPLAY_ARTIFACTS_PARTIAL** |
| Production object replay possible | False |
| Record-level replay possible | False |
| Historical consistency mismatches | [{'relation': 'right', 'tile': '1010', 'detail': {'counts_match_ref01_evidence': False, 'stored_counts_match_live': True}}, {'relation': 'left', 'tile': '1003', 'detail': {'counts_match_ref01_evidence': False, 'stored_counts_match_live': True}}, {'relation': 'above', 'tile': '1008', 'detail': {'counts_match_ref01_evidence': False, 'stored_counts_match_live': True}}, {'relation': 'below', 'tile': '1009', 'detail': {'counts_match_ref01_evidence': False, 'stored_counts_match_live': True}}] |
| Evidence | `evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json` |
| Report | `docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md` |
| Wrong report path removed | YES (`docs/task8b3_ref01_locked_replay_artifact_audit.md`) |
| Overall outcome | **REF01_LOCKED_REPLAY_ARTIFACT_AUDIT_CORRECTED** |
| Next gate | `REF01_LOCKED_REPLAY_INPUT_RECOVERY_DESIGN` (not executed) |
| Next action | Awaiting ChatGPT audit |

Watt was not needed for Task 8B.3-REF01-E3C0-R1 (no downloads, no transfers).

The corrected audit read, hashed and inventoried saved artifacts only inside the allowed roots; no forbidden root was
consulted, nothing was regenerated or replayed, and no file outside the repository was written.
