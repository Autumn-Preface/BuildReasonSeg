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

# FROM_DSH — Task 8B.3-M1A.2B-R3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1A.2B-R3` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `872d45b3ad6adee3b3d9f0fa5bdfa95fd058a37c` |
| Files changed | `tests/test_task8b_runtime.py` and `source_manifest.json` |
| Runtime files | UNCHANGED |
| Guard test | two independent 50×60 detections on a fake 5000×5000 shape, one `proposal_iou` call, one merged proposal, area 3000; no `logical_and`/`logical_or` patching |
| Dedicated run | exactly once → **PASS** |
| Manifest | 135/135 verified (path/size/sha256) |
| External delivery / real inference | NOT touched / NOT RUN |
| Report | `docs/task8b3_m1a_compact_proposal_masks.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit; do not sync delivery. |

Watt was not needed for Task 8B.3-M1A.2B-R3 (no downloads, no transfers).
