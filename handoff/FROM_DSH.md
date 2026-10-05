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

# FROM_DSH — Task 8B.3-REF01-E3C1 Report

```text
Task: 8B.3-REF01-E3C1
Status: COMPLETE
Branch: audit/task8b3-ref01-locked-replay-artifacts
Starting HEAD: 6d1e8d1e69f3adefb4301948cae8918d4aec7ee1
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
Detector/model calls: 0
Proposal regeneration performed: NO
Detector pipeline replay performed: NO
Full production-object replay performed: NO
Record-level selector replay performed: YES
External write performed: NO
Product source changed: NO
Frozen transition right: 1->1
Frozen transition left: 14->14
Frozen transition above: 4->5
Frozen transition below: 1->1
Eligibility repair validation: PASS
Eligibility blocker resolved cases: above
Selection blockers remaining: left; below
Evidence: evaluation/task8b3_ref01_e3c1_locked_record_replay.json
Report: docs/task8b3_ref01_e3c1_locked_record_replay.md
REF-01 status: ACTIVE
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Outcome: REF01_ELIGIBILITY_REPAIR_LOCKED_RECORD_REPLAY_PASS
Next gate: REF01_SELECTION_REPAIR_DESIGN
Next executed: NO
Next action: Awaiting ChatGPT audit.
```
