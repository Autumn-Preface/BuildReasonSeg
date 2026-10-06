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

# FROM_DSH — Task MASK01_D2_EXTERNAL_RC1_SYNC_AND_FULL_REGRESSION Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D2_EXTERNAL_RC1_SYNC_AND_FULL_REGRESSION` |
| Status | **COMPLETE** |
| Starting branch / head | `audit/task8b3-mask01-r1e2-r2-remaining-failure-disambiguation` / `8ab53f3df2664a9ad8f9f6bc8b5659a2251b0e03` |
| Task branch | `delivery/task8b3-mask01-d2-external-sync-regression` |
| External source identity | checked=135 match=135 missing=0 mismatch=0 |
| Preserved assets/fixtures unchanged | True |
| `check_setup.py` READY | True |
| MASK-01 targeted contracts 7/7 | True |
| External complete suite | 130 passed in 54.05s |
| Sync helper (only writer) | `scripts\sync_advisor_rc1_delivery.py` |
| Manual external edits | NONE |
| Real inference / training / param tuning | NOT executed |
| Evidence | `evaluation\task8b3_mask01_d2_external_sync_and_full_regression.json` |
| Report | `docs\task8b3_mask01_d2_external_sync_and_full_regression.md` |
| Next gate | `CHATGPT_D2_REMOTE_AUDIT` |

Watt was not needed for Task MASK01_D2_EXTERNAL_RC1_SYNC_AND_FULL_REGRESSION (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
