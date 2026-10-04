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

# FROM_DSH — Task 8B.3-REF01-F1-R3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R3` |
| Status | **STOP** (harness import-time failure; single run consumed; no classification asserted) |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `ac1e4a9f8f452581c509ab0d73024c44c5de1be0` |
| STATIC_HARNESS_GATE | PASS |
| Harness runs consumed | 1 |
| Detector calls executed | 0 |
| Evidence json | NOT PRODUCED |
| Blocking defect | one-line operator-precedence bug in the harness raster-root construction (`Path / str + str`); the fix is to parenthesise the concatenation |
| Proposal mask source (as implemented) | RERUN_GLOBALPROPOSAL_MASK_CROP |
| proposals.json role | METADATA_REPRODUCTION_ONLY |
| Coverage threshold | 0.50 (TASK7F_FROZEN) |
| Qwen / SAM2 / relation fields / D-B1 / target segmentation | NOT EXECUTED |
| Manual visual inspection / candidate replacement / repair | NO / NO / NO |
| External delivery / canonical RC1 modified | NO / NO |
| Classification / outcome enum | NOT ASSERTED |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| REF-01 status | FORENSICS_INCOMPLETE |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| STOP reason | the repaired harness passed the static gate but crashed at import on the parenthesisation bug, consuming the single allowed execution; per the task book the harness may not be run again in this task |
| Next action | Awaiting ChatGPT audit — next task should apply the one-line fix and grant one fresh harness execution |

Watt was not needed for Task 8B.3-REF01-F1-R3 (no downloads, no transfers).

No detector pass executed, no delivery file was modified, and no Qwen/SAM2/D-B1/relation/target-segmentation stage ran.
