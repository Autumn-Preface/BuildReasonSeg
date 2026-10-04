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

# FROM_DSH — Task 8B.3-P1D10-R2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D10-R2` |
| Status | **COMPLETE** (v0.2 audit + metadata-only candidate lock) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `a3a0d01f5f3f49f249c4a56b45d47fc42d6cbd48` |
| Model/test execution | NONE |
| Functional files modified | NO |
| v0.2 test JSONL | AVAILABLE (14 415 287 bytes, 6 219 records) |
| v0.2 test classification | **USABLE_WITH_DISCLOSURE** (frozen final-test split; metadata-only deterministic locking, no reported result changed) |
| v0.2 val classification | **MODEL_SELECTION_LEAKAGE_RISK** |
| Best Demo pool | `datasets/build_spatial_reason/v0.2` (TEST split) |
| R1 `task6m1_demo` primary-pool statement | **SUPERSEDED** (pool is model-output-contaminated) |
| Locked candidate count | **4** |
| Locked right candidate | `buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91` |
| Locked left candidate | `buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3` |
| Locked above candidate | `buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314` |
| Locked below candidate | `buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450` |
| Eligibility fields used | canonical generator metadata only (`level`, `target_component_id`, `reference_component_ids`, `native_vector.target/references`, `target_geometry_ref`) |
| Forbidden fields consulted | NONE (no detector/parser/runtime/mask/visual/confidence/historical-success field) |
| Deterministic ordering | immutable `sample_id`; distinct `image_id` enforced across relations |
| Source image resolution | via frozen `image_metadata_ref` component maps (the v0.2 `image_path` field is a placeholder sentence) |
| Image copied into RC1 / detector run / visual inspection | NO / NO / NO |
| Primary resolution (from R1, unchanged) | `PROP01_RESOLUTION_DEMO_POLICY` |
| PROP-01 status (from R1, unchanged) | `PROP01_OPEN_ENGINEERING_DEFECT` |
| Scientific freeze preserved | YES |
| Next gate | `PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION` (recommended, not executed) |
| Report | `docs/task8b3_p1d10_prop01_resolution_decision.md` |
| Next action | Awaiting ChatGPT audit; locked candidates must not be run or replaced. |

Locked candidate IDs are frozen on commit and may not be replaced after future runtime results without a new ChatGPT
decision that explicitly acknowledges the failed locked candidate.

Watt was not needed for Task 8B.3-P1D10-R2 (no downloads, no transfers).

No model, test, training or inference execution occurred; only metadata and frozen documentation were read.
