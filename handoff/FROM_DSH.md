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

# FROM_DSH — Task 8B.3-M1A.2B-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1A.2B-R1` |
| Status | **PARTIAL / STOP** |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `e5229d60a869c1fc02536d41a039229f2936090b` |
| Files changed | `tests/test_task8b_runtime.py` only |
| Runtime files | UNCHANGED |
| Legacy entries converted | 10 |
| Dedicated run | exactly once → **FAIL (STOP)** |
| Manifest | UNCHANGED (dedicated gate failed) |
| External delivery / real inference | NOT touched / NOT RUN |
| Report | `docs/task8b3_m1a_compact_proposal_masks.md` |
| STOP reason | the single dedicated run failed; per the task book no runtime change, no rerun and no manifest update were performed |
| Next action | Awaiting ChatGPT audit; do not sync delivery. |

Watt was not needed for Task 8B.3-M1A.2B-R1 (no downloads, no transfers).
