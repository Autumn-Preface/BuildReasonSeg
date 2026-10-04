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

# FROM_DSH — Task 8B.3-REF01-F1-R6 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R6` |
| Status | **COMPLETE** (authoritative closure) |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `53de75ae4f5246b4e18685b78bd5a9f913447984` |
| Detector calls | 4 (final deterministic pass, one per locked candidate) |
| Manifest basis / entries | GIT_CANONICAL_BLOB_BYTES / 135 |
| External detector == Git canonical blob == manifest identity | True |
| External control manifest == Git canonical control | True |
| imageio / pillow / numpy | IMPORT_FAILED: No module named 'imageio' / 12.3.0 / 2.4.6 |
| Runtime | external RC1 default `DetectorRuntime()` |
| Frozen constants | TILE_SIZE 512 · TILE_OVERLAP 128 · stride 384 · IMGSZ 640 · CONF 0.05 · MAX_DET 300 · DUPLICATE_IOU 0.50 · MERGE 0.20 · FROZEN_THRESHOLD 0.5 |
| Raster SHA identity | 4/4 PASS |
| 10-field proposal reproduction | 4/4 all ten fields match for every merged proposal = True |
| R6 vs R4 vs R5 IoU | within 1e-6 for all four candidates = True |
| AUTHORITATIVE_CLOSURE | **PASS** |
| Outcome | **REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE** |
| Dominant next blocker | ELIGIBILITY |
| Next gate | `NEXT = REF01_ELIGIBILITY_FORENSICS` (not executed) |
| Qwen / SAM2 / relation fields / D-B1 / target segmentation | NONE / NONE / NONE / NONE / NONE |
| Manual visual inspection / candidate replacement / repair | NO / NO / NO |
| External delivery / canonical RC1 modified | NO / NO |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| REF-01 status | FORENSICS_COMPLETE |
| Evidence | `evaluation\task8b3_ref01_locked_reference_forensics_r6.json` |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed |

Watt was not needed for Task 8B.3-REF01-F1-R6 (no downloads, no transfers).

Only the authorised final pass per locked candidate ran; no Qwen/SAM2/D-B1/relation/target-segmentation stage executed
and no delivery file was modified.
