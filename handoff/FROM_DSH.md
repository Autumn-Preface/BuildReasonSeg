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

# FROM_DSH — Task 8B.3-P1D12-R2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D12-R2` |
| Status | **COMPLETE** (read-only threshold + timing evidence) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `728932094a9f86f7c1607abc6a2beff12d2d2d70` |
| Candidate / predict / detector / model re-runs | NONE |
| `FROZEN_THRESHOLD` (canonical detector source) | `0.5  # ultralytics mask binarisation for the frozen U-C1 detector` |
| Sam2 / relation / D-B1 timing entries in the four result.json | NONE (key sets recorded per candidate in the report) |
| Inspect-only conclusion | SAM2 = 0 · relation fields = 0 · D-B1 = 0 (stages absent from the inspect-mode result contract) |
| External delivery / canonical RC1 modified | NO / NO |
| Locked candidate identities | UNCHANGED |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Outcome | **PROP01_LOCKED_DEMO_PROPOSAL_GATE_EVIDENCE_CLOSED** (retained) |
| Next gate (recommended, not executed) | `NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS` |
| Report | `docs\task8b3_p1d12_r2_final_evidence_closure.md` |
| Next action | Awaiting ChatGPT audit; the next gate needs its own task book |

Watt was not needed for Task 8B.3-P1D12-R2 (no downloads, no transfers).

No model, detector or candidate was executed and no delivery file was modified; only this report and the handoff
changed.
