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
| Status | **STOP** — v0.2 audited, four candidates locked by metadata, but §7 raster-resolution proof is incomplete |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `b36407862477565f1cf228df54b1aac648347bcc` |
| Model/test execution | NONE |
| Functional files modified | NO |
| v0.2 test JSONL | AVAILABLE (14 415 287 bytes, 6 219 records) |
| v0.2 test classification | **USABLE_WITH_DISCLOSURE** (frozen final-test split; metadata-only deterministic locking, no reported result changed) |
| v0.2 val classification | **MODEL_SELECTION_LEAKAGE_RISK** |
| Best Demo pool | `datasets/build_spatial_reason/v0.2` (TEST split) — pending raster resolution |
| R1 `task6m1_demo` primary-pool statement | **SUPERSEDED** (model-output-contaminated) |
| Locked candidate count | 4 (provisional metadata locks; not approved for running) |
| Locked right candidate | `buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91` |
| Locked left candidate | `buildsr_test_1003_3_largest_to_left_of_nearest_f3fcb14e14c3` |
| Locked above candidate | `buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314` |
| Locked below candidate | `buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450` |
| Eligibility fields used | canonical generator metadata only; no forbidden field consulted |
| Component-map resolution | RESOLVED (512×512 PNG present and decodable for all four) |
| Original raster resolution | **NOT ESTABLISHED** (v0.2 `image_path` is a placeholder sentence) |
| Selection policy | **NOT READY** (§11 source-image clause unsatisfied) |
| Image copied into RC1 / detector run / visual inspection | NO / NO / NO |
| Primary resolution / PROP-01 status (from R1) | `PROP01_RESOLUTION_DEMO_POLICY` / `PROP01_OPEN_ENGINEERING_DEFECT` (unchanged) |
| Scientific freeze preserved | YES |
| Next gate | none executed; `PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION` remains recommended but blocked by the raster-resolution gap |
| Report | `docs/task8b3_p1d10_prop01_resolution_decision.md` |
| STOP reason | §11 eligibility requires the referenced source image to exist and decode; the frozen v0.2 metadata resolves only component maps, not the original rasters, and §12 forbids loosening criteria |
| Next action | Awaiting ChatGPT audit; locked candidates must not be run or replaced. |

Watt was not needed for Task 8B.3-P1D10-R2 (no downloads, no transfers).

No model, test, training or inference execution occurred; no image was copied into RC1 and no detector or parser
output was consulted.
