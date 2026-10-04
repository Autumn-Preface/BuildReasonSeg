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

# FROM_DSH — Task 8B.3-M1A.2A Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git history._

| item | value |
|---|---|
| Task | 8B.3-M1A.2A |
| Status | COMPLETE (canonical runtime only) |
| Branch | fix/task8b3-mem01-compact-proposals |
| Starting HEAD | 2c51c48252fcc705da5d9ee914299775d985a725 |
| Compact proposal representation | IMPLEMENTED (mask_crop + inclusive global_bbox; no global_mask field) |
| Full-frame proposal masks retained | NO |
| Full-frame pairwise IoU temporaries | NO (proposal_iou intersects the two bboxes; iou_of retained untouched as legacy) |
| Detector parameters / tiling / DUPLICATE_IOU / winner / stable ID | UNCHANGED |
| Reference eligibility / selection / MERGE_BBOX_EXTENT_RATIO_MAX | UNCHANGED |
| Reasoning context / ProgramHead / SAM2 / D-B1 / SUCCESS validity | UNCHANGED |
| py_compile gate | PASS |
| Synthetic smoke gate | PASS (compactness, IoU equivalence, merge equivalence, 5000x5000 no-full-frame-bool guard, core crop, preview) |
| tests / source_manifest.json | NOT modified |
| External delivery modified | NO |
| Real inference executed | NO |
| Scientific model/checkpoint changed | NO |
| PROP-01 / REF-01 / MASK-01 | UNCHANGED / UNCHANGED / UNCHANGED |
| MEM-01 | canonical implementation DONE (external delivery sync still pending a later task) |
| Report | docs/task8b3_m1a_compact_proposal_masks.md |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit; do not sync delivery. |

Watt was not needed for Task 8B.3-M1A.2A (no downloads, no transfers).

No test file, manifest, external-delivery file, package, checkpoint or dataset was modified; no pytest suite, predict run or six-image Demo was executed.