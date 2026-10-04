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

# FROM_DSH — Task 8B.3-REF01-F1-R7 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R7` |
| Status | **COMPLETE** (zero-call authoritative replay) |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c` |
| Detector / model calls | 0 (no `DetectorRuntime()` instantiation, no `detect_global`, no predict) |
| `buildreasonseg.runtime.detector` Git-canonical identity | manifest = Git blob = external = `82531dc3b758cd8a...` (all three agree = True) |
| imageio Git-canonical identity | not declared in the manifest/requirements and not importable; delivery I/O = Pillow/ultralytics; `requirements.txt` Git-canonical sha256 `adfdd3b481a0fd21...` verified |
| Replay inputs | R4 canonical records · R5 verification · R6 closure · P1D12 diagnostics metadata |
| Classification reproduction | 4/4 reproduced mechanically from stored IoUs |
| R4 = R5 = R6 IoU agreement | 4/4 within 1e-6 |
| Canonical evidence | overwritten with the consolidated authoritative record |
| Temporary evidence `_r5` / `_r6` | DELETED |
| Qwen / SAM2 / relation fields / D-B1 / target segmentation | NONE |
| Manual visual inspection / candidate replacement / repair | NO / NO / NO |
| External delivery / canonical RC1 modified | NO / NO |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| REF-01 status | FORENSICS_COMPLETE |
| Outcome | **REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE** |
| Dominant next blocker | ELIGIBILITY |
| Next gate | `NEXT = REF01_ELIGIBILITY_FORENSICS` (not executed) |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed |

Watt was not needed for Task 8B.3-REF01-F1-R7 (no downloads, no transfers).

No model, detector, Qwen, SAM2, D-B1 or target-segmentation execution occurred; no delivery file was modified; the only
repository changes are the consolidated canonical evidence, the removal of the superseded temporary evidence files,
this report and the handoff.
