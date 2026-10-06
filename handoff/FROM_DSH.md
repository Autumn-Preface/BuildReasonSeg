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

# FROM_DSH — Task RC1_INSPECT01_PREFLIGHT_ORDER_FIX Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `RC1_INSPECT01_PREFLIGHT_ORDER_FIX` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1d` / `5808e64ed43fd383be9e08bf623a9bfdc27f62b4` |
| Task branch | `fix/rc1-inspect-proposals-preflight-order` |
| Design | `INSPECT_IMAGE_PREFLIGHT_BEFORE_MODEL_V1` |
| Fix | inspect-only `load_image()` preflight before `_resolve_and_report` / `PredictRuntime` |
| Contract | `--inspect-proposals` + unreadable image → **E202 / exit 20** |
| Regression test | `test_inspect_unreadable_image_precedes_model_resolution` |
| Gate A | exit 0 · 2 passed in 3.24s |
| Gate B | exit 0 · 4 passed in 6.12s |
| runtime pipeline / imageio / detector / manifest / docs | UNCHANGED |
| normal inference ordering changed | NO |
| model inference / training / external sync | false / false / false |
| Evidence | `evaluation\rc1_inspect01_preflight_order_fix.json` |
| Report | `docs\rc1_inspect01_preflight_order_fix.md` |
| Next gate | `CHATGPT_INSPECT01_REMOTE_AUDIT` (R1-D2 / R1-E not entered) |

Watt was not needed for Task RC1_INSPECT01_PREFLIGHT_ORDER_FIX (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
