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

# FROM_DSH — Task 8B.3-REF01-F1-R12 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R12` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `d5341c5e45915e5b0877ebbd1ddbb48f467e7eeb` |
| Detector / model calls | 0 |
| Canonical evidence modified | NO (sha256 `f7495796577cb26a...` unchanged) |
| Exact disclosure asserted | YES (English verbatim + Chinese translation) |
| Dual manifest identity | PASS (Git-canonical == external == working, 135 entries, `GIT_CANONICAL_BLOB_BYTES`) |
| Module `__file__` identity | PASS (detector + imageio imported from the external delivery) |
| Pinned R6 `git show` ten-field evidence | PASS (revision `12d5fd9a92a5`, four candidates, four detector calls) |
| Live P1D12 counts | PASS (6/6/4, 66/53/42, 9/9/4, 7/6/3) |
| Live production selected replay | PASS (selected id present in live metadata with matching six fields) |
| bestEligible / bestAny replay | PASS (frozen tie-break `(-IoU, -confidence, proposal_id)`) |
| Six-field role facts | CHECKS 53/53 passed |
| Final verifier verdict | FINAL_INDEPENDENT_VERIFIER: PASS |
| Qwen / SAM2 / relation fields / D-B1 / target segmentation | NONE |
| Manual visual inspection / candidate replacement / repair | NO / NO / NO |
| External delivery / canonical RC1 modified | NO / NO |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| REF-01 status | FORENSICS_COMPLETE |
| Overall outcome | **REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE** |
| Dominant next blocker | ELIGIBILITY |
| Next gate | `NEXT = REF01_ELIGIBILITY_FORENSICS` (not executed) |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| Next action | Awaiting ChatGPT audit; NEXT is not executed |

Watt was not needed for Task 8B.3-REF01-F1-R12 (no downloads, no transfers).

No model, detector, Qwen, SAM2, D-B1 or target-segmentation execution occurred; the canonical evidence and the external
delivery were not modified.
