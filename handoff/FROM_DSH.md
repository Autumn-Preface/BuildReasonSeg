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

# FROM_DSH — Task 8B.3-REF01-E1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E1` |
| Status | **COMPLETE** |
| Base branch / head | `fix/task8b3-ref01-reference-forensics` / `798a2fcb50054a67b8101581c5007cbbfa46c36a` |
| New task branch | `fix/task8b3-ref01-eligibility-forensics` |
| Detector / model calls | 0 |
| Proposal metadata source | EXISTING_P1D12_JSON_ONLY |
| above best-any id | 5 |
| above best-any IoU | 0.903250 |
| above best-any border flag | False |
| above best-any extent ratio | 0.296875 |
| above best-any fail reason | BBOX_EXTENT_CAP |
| above P0 selected/id IoU | 4 / 0.000000 |
| above P1 selected/id IoU | 4 / 0.000000 |
| above P2 selected/id IoU | 5 / 0.903250 |
| above P3 selected/id IoU | 5 / 0.903250 |
| right / left / below GT-best fail reasons | right:['NONE'], left:['NONE'], below:['NONE'] |
| Threshold / rules / repair changed | NO / NO / NO |
| Manual visual inspection / candidate replacement | NO / NO |
| Outcome | **REF01_ELIGIBILITY_BLOCKER_ISOLATED_BBOX_EXTENT_CAP** |
| Next gate | `NEXT = REF01_ELIGIBILITY_REPAIR_DESIGN` (not executed) |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| REF-01 status | FORENSICS_COMPLETE (eligibility attribution added) |
| Evidence | `evaluation\task8b3_ref01_eligibility_forensics.json` |
| Report | `docs/task8b3_ref01_eligibility_forensics.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed |

Watt was not needed for Task 8B.3-REF01-E1 (no downloads, no transfers).

No model, detector, Qwen, SAM2, D-B1 or target-segmentation execution occurred; the canonical REF01 evidence and the
external delivery were read only and not modified.
