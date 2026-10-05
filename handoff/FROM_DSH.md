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

# FROM_DSH — Task 8B.3-REF01-E3B2-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

```text
Task: 8B.3-REF01-E3B2-R1
Status: COMPLETE
Branch: fix/task8b3-ref01-eligibility-repair-sync
Starting HEAD: 5b8edd742981f1128c851c262ec576665fa0aa1f
Technical source task: 8B.3-REF01-E3B2
Technical execution starting HEAD: 300cdad5629b945ffe10480351081f9c8befe7f4
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Canonical source changed in R1: NO
Canonical manifest changed in R1: NO
External pre-sync: 133 match / 0 missing / 2 mismatch
External mismatch paths: buildreasonseg/runtime/detector.py; tests/test_task8b_runtime.py
External sync runs: 1
External sync: 135 copied / 135 verified / 0 failures
External post-sync: 135 match / 0 missing / 0 mismatch
External source_manifest copied from Git object: YES
External source_manifest identity rechecked in R1: YES
External source_manifest bytes: 24375
External source_manifest SHA256: 5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c
Setup checker runs: 1
Setup checker: BuildReasonSeg environment: READY
External targeted test runs: 1
External targeted regression: 40 passed in 1.13s
External full suite runs: 1
External full suite: 124 passed in 112.81s (0:01:52)
External technical commands rerun in R1: NO
External write performed in R1: NO
Model inference performed: NO
Wrong E3B2 evidence removed: YES
Evidence: evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
Report: docs/task8b3_ref01_e3b2_external_sync_full_suite.md
Outcome: REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED
Next gate: REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects: UNRESOLVED
Next action: Awaiting ChatGPT audit; do not execute NEXT/E3C.
```

Watt was not needed for Task 8B.3-REF01-E3B2-R1 (no downloads, no transfers).
