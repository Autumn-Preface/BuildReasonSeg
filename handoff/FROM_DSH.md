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

# FROM_DSH — Task 8B.3-REF01-F1-R5 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R5` |
| Status | **COMPLETE** (deterministic verification) |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `d1c7f6374f325fec71bce6a350609b2d23865c0a` |
| Detector calls | 4 (one additional pass per locked candidate) |
| Raster SHA identity | 4/4 PASS |
| External detector identity | identical to canonical = False |
| imageio / pillow / numpy | IMPORT_FAILED: No module named 'imageio' / 12.3.0 / 2.4.6 |
| Runtime | external RC1 default `DetectorRuntime()` |
| Frozen constants | TILE_SIZE 512 · TILE_OVERLAP 128 · stride 384 · IMGSZ 640 · CONF 0.05 · MAX_DET 300 · DUPLICATE_IOU 0.50 · MERGE_BBOX_EXTENT_RATIO_MAX 0.20 · FROZEN_THRESHOLD 0.5 |
| P1D12 metadata reproduction | counts 4/4 MATCH · per-proposal metadata 4/4 MATCH |
| R5 vs R4 IoU | within 1e-6 for all four candidates = True |
| DETERMINISTIC_VERIFICATION | **PASS** |
| Outcome (unchanged from R4) | REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE |
| Dominant next blocker | ELIGIBILITY |
| Next gate | `NEXT = REF01_ELIGIBILITY_FORENSICS` (not executed) |
| Qwen / SAM2 / relation fields / D-B1 / target segmentation | NONE / NONE / NONE / NONE / NONE |
| Manual visual inspection / candidate replacement / repair | NO / NO / NO |
| External delivery / canonical RC1 modified | NO / NO |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| REF-01 status | FORENSICS_COMPLETE |
| Evidence | `evaluation\task8b3_ref01_locked_reference_forensics_r5.json` |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed |

Watt was not needed for Task 8B.3-REF01-F1-R5 (no downloads, no transfers).

Only the harness's authorised additional pass per locked candidate ran; no Qwen/SAM2/D-B1/relation/target-segmentation
stage executed and no delivery file was modified.
