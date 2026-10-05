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

# FROM_DSH — Task 8B.3-REF01-E1-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E1-R1` |
| Status | **COMPLETE** (E1 selection-rule defect corrected) |
| Branch | `fix/task8b3-ref01-eligibility-forensics` |
| Starting HEAD | `3c497bf3d427366e00b5890fdcfab1a35e68b28c` |
| Detector / model calls | 0 |
| Production selection rule | `argmin(-mask_area, -confidence, proposal_id)` over each policy eligible set |
| GT coverage | recorded separately as `argmin(-iou_to_gt, -confidence, proposal_id)` |
| Fail-reason enum | `MASK_AREA_ZERO` / `TOUCHES_IMAGE_BORDER` / `BBOX_EXTENT_CAP` (empty = `ELIGIBLE`) |
| Tracked replay script | `scripts/task8b3_ref01_eligibility_forensics.py` |
| above GT-best id / IoU | 5 / 0.903250 |
| above GT-best border / extent / margin | False / 0.296875 / 0.096875 |
| above GT-best fail reason | BBOX_EXTENT_CAP |
| above P0/P1/P2/P3 production selected | P0_FROZEN:4 / P1_BORDER_RELAXED_ONLY:4 / P2_EXTENT_RELAXED_ONLY:5 / P3_BOTH_RELAXED:3 |
| above P0/P1/P2/P3 selected IoU | 0.0 / 0.0 / 0.90325 / 0.0 |
| bbox-extent-cap isolation | True |
| right / left / below GT-best fail reasons | right:['ELIGIBLE'], left:['ELIGIBLE'], below:['ELIGIBLE'] |
| Threshold / rules / repair changed | NO / NO / NO |
| Manual visual inspection / candidate replacement | NO / NO |
| Outcome | **REF01_ELIGIBILITY_BLOCKER_ISOLATED_BBOX_EXTENT_CAP** |
| Next gate | `NEXT = REF01_ELIGIBILITY_REPAIR_DESIGN` (not executed) |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Evidence | `evaluation\task8b3_ref01_eligibility_forensics.json` |
| Report | `docs/task8b3_ref01_eligibility_forensics.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed |

Watt was not needed for Task 8B.3-REF01-E1-R1 (no downloads, no transfers).

The canonical REF01 evidence and the external delivery were read only and not modified; no model, detector, Qwen, SAM2,
D-B1 or target-segmentation execution occurred.
