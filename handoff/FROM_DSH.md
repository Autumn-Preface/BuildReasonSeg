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

# FROM_DSH — Task 8B.3-R4A.1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.3-R4A.1` |
| Status | **COMPLETE** |
| Branch | `eval/task8b3-six-image-demo-suite` |
| Starting HEAD | `d4b7d36c68be7cee6970b9f3d58cc1cd4f47caf1` |
| Harness code | UNCHANGED |
| Harness tests | UNCHANGED |
| Repository tests | **1555 passed** (`pytest tests/ -q`, single run, exit 0) |
| Formal suite | NOT RUN BY DESIGN |
| A1–B2 | NOT RUN |
| Review pack | NOT CREATED / unchanged |
| Output-layout proposal | ACCEPTED / DEFERRED TO TASK 8B.4 |
| Report | `docs/task8b3_six_image_demo_suite.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit. |

Watt was not needed for Task 8B.3-R4A.1 (no downloads, no transfers); the pre-existing Watt instance, when present,
remains transport-only and is not owned by this project.

No harness, test, RC1 canonical/product/delivery source, ProgramHead, Qwen suggestion, detector, Reference, SAM2,
D-B1, threshold, config, checkpoint or model asset was modified; no real predict, no six-image suite and no review
pack action occurred; no Assisted Mode and no package installation.
