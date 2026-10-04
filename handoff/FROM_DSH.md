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

# FROM_DSH — Task 8B.3-M1B.3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1B.3` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `f73b4294667219f093aa6333c3fd5d4a36832814` |
| External manifest | 135/135 PASS |
| B1.tif | exists, [5000, 5000], 75040370 bytes |
| B2.tif | exists, [5000, 5000], 75080895 bytes |
| Real predict invocations | B1 = 1, B2 = 1 |
| B1 result | exit 0 · SUCCESS · tiles 169 · raw 6578 · merged 3066 |
| B2 result | exit 0 · SUCCESS · tiles 169 · raw 7864 · merged 3740 |
| Proposal stage completed | B1 YES · B2 YES |
| Allocation signature found | NO (B1, B2) |
| E502 found | NO (B1, B2) |
| MEM gate verdict | **MEM01_REAL_GATE_PASS** |
| `RC1-DEMO-MEM-01` | **CLOSED** |
| Real inference | RUN — external `predict.py` only, one invocation per image, no rerun |
| Runtime / tests / manifest modified | NO |
| PROP-01 / REF-01 / MASK-01 | UNCHANGED / UNCHANGED / UNCHANGED |
| Report | `docs/task8b3_m1b3_large_image_memory_gate.md` |
| Next action | Awaiting ChatGPT audit; no visual verdict claimed; Task 8B.4 / 8C not entered. |

Watt was not needed for Task 8B.3-M1B.3 (no downloads, no transfers).

No visual or semantic quality judgement of the B1/B2 outputs is expressed. No runtime, test, manifest, model,
threshold, tiling, merge, Reference or validity code was modified.
