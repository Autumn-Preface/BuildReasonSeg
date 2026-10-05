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

# FROM_DSH — Task 8B.3-REF01-E3C3 Report

```text
Task: 8B.3-REF01-E3C3
Status: COMPLETE
Branch: audit/task8b3-ref01-locked-replay-artifacts
Starting HEAD: 6e33b88ecf9c78ef9fc62a9c79ec0a8f0c55c1bd
Decision owner: CHATGPT
Decision: REJECT_FURTHER_SCALAR_RANK_REPAIR
Detector/model calls: 0
Proposal regeneration performed: NO
Selector repair implementation performed: NO
Product source changed: NO
External write performed: NO
Frozen automatic selector repair: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
Right: 1->1 correct stable
Left: 14->14 residual selection limitation
Above: 4->5 eligibility blocker resolved
Below: 1->1 residual selection limitation
Engineering fallback: ASSISTED_REFERENCE_OVERRIDE
Fallback pipeline parameter: reference_id
REF-01 status: ACTIVE_RESIDUAL_SELECTION_LIMITATION
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Evidence: evaluation/task8b3_ref01_e3c3_selection_repair_decision.json
Report: docs/task8b3_ref01_e3c3_selection_repair_decision.md
Outcome: REF01_SCALAR_REPAIR_REJECTED_ASSISTED_FALLBACK_FROZEN
Next gate: MASK01_VALIDITY_FORENSICS_DESIGN
Next executed: NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```
