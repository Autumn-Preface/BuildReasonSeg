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

# FROM_DSH — Task 8B.3-REF01-E3A-R5 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3A-R5` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-eligibility-repair-impl` |
| Starting HEAD | `34d1a263ac91ada6ef517fc54a99c19463e9ddd3` |
| Attachment | `handoff/E3A_R5_apply_exact.py` (10418 bytes) |
| Prescribed / attachment SHA256 | `1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f` / `1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f` |
| Copied verbatim to (outside repo) | `C:\D\DeepSeekHarness\E3A_R5_apply_exact.py` |
| Destination SHA256 | `1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f` |
| Patcher reconstructed from the task book | NO |
| Patcher compile gate | NONE |
| Patcher runs / exit | 1 / 0 |
| detector.py before / after | `a6fa4bdd76db6f50` / `bc5aed885aa5f27b` |
| Changed entries | ['??', 'M', 'delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py', 'delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py', 'handoff/E3A_R5_apply_exact.py', 'handoff/TO_DSH.md'] |
| source_manifest updated (canonical / external) | NO / NO |
| External RC1 synced | NO |
| Detector / model inference | NONE |
| Manual code modification | NONE |
| Retry / self-repair | NONE |
| Intermediate commit or push | NONE (one task commit records this result) |
| NEXT executed | NO |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Report | `docs\task8b3_ref01_eligibility_repair_impl.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed. |

Watt was not needed for Task 8B.3-REF01-E3A-R5 (no downloads, no transfers).

The attachment patcher was verified against its published SHA256, copied byte-for-byte to the prescribed location,
re-verified there, and executed exactly once; exactly one task commit records the result and no manifest, external file
or model asset was modified.
