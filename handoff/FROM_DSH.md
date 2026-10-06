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

# FROM_DSH — Task MASK01_D1_R1D2_POST_INSPECT_MANIFEST_RECANONICALIZATION Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1D2_POST_INSPECT_MANIFEST_RECANONICALIZATION` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/rc1-inspect-proposals-preflight-order` / `efc48eb5ee8b0851a84e8a3d95533f1acd23cacf` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1d2` |
| Only file modified | `source_manifest.json` |
| ordered_path_digest_before | `7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd` |
| ordered_path_digest_after | `7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd` |
| digest / count / order preserved | True / True / True |
| Entries updated / already canonical | 2 / 133 |
| All-entry `entry_source_bytes()` validation | PASS |
| Manifest contract validation | True |
| product source / tests / docs / sync script | UNCHANGED |
| pytest / model inference / training / external sync | not run / false / false / false |
| Evidence | `evaluation\task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.json` |
| Report | `docs\task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.md` |
| Next gate | `CHATGPT_R1D2_REMOTE_AUDIT` (R1-E not entered) |

Watt was not needed for Task MASK01_D1_R1D2_POST_INSPECT_MANIFEST_RECANONICALIZATION (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
