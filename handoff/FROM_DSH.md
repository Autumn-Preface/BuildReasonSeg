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

# FROM_DSH — Task 8B.3-R4B Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git history._

| item | value |
|---|---|
| Task | `8B.3-R4B` |
| Status | **COMPLETE** (procedure) |
| Branch | `eval/task8b3-six-image-demo-suite` |
| Starting HEAD | `b579148df7aac9c979265b2e4da5b91c9d41d34b` |
| check_setup | READY (live `ultralytics==8.4.164`) |
| Inputs | 6/6 with exact frozen SHA256 |
| Formal suite invocations | **1** |
| A1 | language=`DIRECT_CORRECT` runtime=`SUCCESS` |
| A2 | language=`FALLBACK_CORRECT` runtime=`FAILED` `E401` |
| A3 | language=`DIRECT_CORRECT` runtime=`SUCCESS` |
| A4 | language=`FALLBACK_CORRECT` runtime=`SUCCESS` |
| B1 | language=`FALLBACK_CORRECT` runtime=`FAILED` `E502` |
| B2 | language=`DIRECT_CORRECT` runtime=`FAILED` `E502` |
| Runtime successes pending visual review | 3/6 (A1, A3, A4) |
| Runtime failures | 3/6 (A2, B1, B2) |
| Review pack | `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\review_task8b3` |
| Visual verdict | PENDING CHATGPT/USER REVIEW |
| Output-layout proposal | ACCEPTED / DEFERRED TO TASK 8B.4 |
| Report | `docs/task8b3_six_image_demo_suite.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT/user visual review. |

Watt was not needed for Task 8B.3-R4B (no downloads, no transfers).

No RC1 product/canonical/delivery source/config, harness, test, prompt, program, threshold, config or checkpoint was modified; the formal suite was invoked exactly once and no sample was retried.
