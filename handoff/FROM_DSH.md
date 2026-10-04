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

# FROM_DSH — Task 8B.3-REF01-F1-R4 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R4` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `35105fb8b4255375923ff3174d3efd2356717333` |
| Preflight 1/3 compile | PASS |
| Preflight 2/3 import-only | PASS (detector calls 0) |
| Preflight 3/3 contract | PASS (external RC1 default `DetectorRuntime`, `eligible_proposals`, `select_reference`) |
| Runtime construction | external RC1 default `DetectorRuntime()` |
| Prior detector passes | 0 |
| Detector calls | 4 |
| P1D12 metadata reproduction | right 6/6 eligible4 MATCH · left 66/53 eligible42 MATCH · above 9/9 eligible4 MATCH · below 7/6 eligible3 MATCH |
| right selected/bestEligible/bestAny IoU | 0.5589 / 0.5589 / 0.5589 |
| left selected/bestEligible/bestAny IoU | 0.0000 / 0.6501 / 0.6501 |
| above selected/bestEligible/bestAny IoU | 0.0000 / 0.0000 / 0.9033 |
| below selected/bestEligible/bestAny IoU | 0.0000 / 0.6165 / 0.6165 |
| right / left / above / below class | REFERENCE_SELECTED_CORRECT · REFERENCE_SELECTION_WRONG_COVERED · REFERENCE_ELIGIBILITY_BLOCKED · REFERENCE_SELECTION_WRONG_COVERED |
| Class counts | {"REFERENCE_SELECTED_CORRECT": 1, "REFERENCE_SELECTION_WRONG_COVERED": 2, "REFERENCE_ELIGIBILITY_BLOCKED": 1} |
| Qwen / SAM2 / relation fields / D-B1 / target segmentation | NONE / NONE / NONE / NONE / NONE |
| Manual visual inspection / candidate replacement / product repair | NO / NO / NO |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| REF-01 status | FORENSICS_COMPLETE |
| Outcome | **REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE** |
| Dominant next blocker | ELIGIBILITY |
| Next gate | `NEXT = REF01_ELIGIBILITY_FORENSICS` (not executed) |
| Evidence | `evaluation\task8b3_ref01_locked_reference_forensics.json` |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed |

Watt was not needed for Task 8B.3-REF01-F1-R4 (no downloads, no transfers).

Exactly one detector pass per locked candidate ran through the external RC1 default `DetectorRuntime()`; no Qwen, SAM2,
relation-field, D-B1 or target-segmentation stage executed, no mask/PNG artifact was written, and no delivery file was
modified.
