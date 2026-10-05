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

# FROM_DSH — Task MASK01_F1_VALIDITY_FORENSICS Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_F1_VALIDITY_FORENSICS` |
| Status | **COMPLETE_WITH_EVIDENCE_GAPS** |
| Starting branch / head | `audit/task8b3-ref01-locked-replay-artifacts` / `43c24de59625dfb97dcc605435fa83b88be46035` |
| Task branch | `audit/task8b3-mask01-validity-forensics` |
| Source files audited | 104 |
| SUCCESS occurrences / mask-validity gates / failure-path lines | 6 / 2 / 237 |
| min_mask_pixels / min_mask_frac | NOT_FOUND_IN_AUDITED_CONFIG / NOT_FOUND_IN_AUDITED_CONFIG |
| Padding helper files | ['delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\box_query.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\program_parser.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\prompt_diagnostics.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\qwen_seg.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6n_relation_decoder.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6p_reference_head.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6u_reference_ranker.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6w_proposal_quality.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6y_nearest_decoder.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task6z_l3_decoder.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\_frozen\\mvp\\task7d_global_competition_decoder.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\context.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\detector.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\imageio.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\outputs.py', 'delivery_src\\BuildReasonSeg_Advisor_RC1\\buildreasonseg\\runtime\\pipeline.py'] |
| Test files covering mask/SUCCESS | 8 |
| Historical artifact availability | **PARTIAL** (10 referenced paths, 6 existing) |
| Evidence gaps | ['historical referenced artifacts availability = PARTIAL'] |
| Product source modified / inference / validity repair | NO / NO / NO |
| repair_decision | `DEFER_TO_CHATGPT` |
| next_gate | `CHATGPT_MASK01_DESIGN_REVIEW` |
| Evidence | `evaluation/task8b3_mask01_f1_validity_forensics.json` |
| Report | `docs/task8b3_mask01_f1_validity_forensics.md` |
| Next action | Awaiting ChatGPT mask-01 design review |

Watt was not needed for Task MASK01_F1_VALIDITY_FORENSICS (no downloads, no transfers).

This task performed a read-only audit only: no product source, threshold, mask-validity rule or model asset was changed,
and no model inference was executed.
