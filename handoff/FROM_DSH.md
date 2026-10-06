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

# FROM_DSH — Task MASK01_D1_R1D_SOURCE_MANIFEST_CANONICALIZATION Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1D_SOURCE_MANIFEST_CANONICALIZATION` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1c` / `3bf9dafac69175d36a1abb6a83d019ef118ca72c` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1d` |
| Only file modified | `source_manifest.json` |
| Identity source | HEAD Git canonical blob bytes (working tree not used) |
| Entries updated / already canonical | 8 / 127 |
| Entry count preserved | True (135 → 135) |
| Path order preserved | True |
| All-entry `entry_source_bytes()` validation | PASS |
| Manifest contract validation | True |
| product source / tests / docs / sync script | UNCHANGED |
| pytest / model inference / training / external sync | not run / false / false / false |
| Evidence | `evaluation\task8b3_mask01_d1_r1d_source_manifest_canonicalization.json` |
| Report | `docs\task8b3_mask01_d1_r1d_source_manifest_canonicalization.md` |
| Next gate | `CHATGPT_R1D_REMOTE_AUDIT` (R1-E not entered) |

Watt was not needed for Task MASK01_D1_R1D_SOURCE_MANIFEST_CANONICALIZATION (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
