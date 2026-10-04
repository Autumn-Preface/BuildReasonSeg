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

# FROM_DSH — Task 8B.3-REF01-F1-R11 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R11` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `7d9492fa7628c4b4cc379a3c4ba976c3bd1fe228` |
| Detector / model calls | 0 |
| Canonical evidence modified | NO (sha256 `f7495796577cb26a...` unchanged) |
| Module `__file__` identity | PASS for `buildreasonseg.runtime.detector` and `.imageio` (external files, manifest SHA match) |
| Historical R6 ten-field evidence via `git show` | PASS (recovered from Git history, all flags true) |
| Live P1D12 counts | PASS (6/6/4, 66/53/42, 9/9/4, 7/6/3) |
| Production `selected` replay | PASS (identical across recovered R4/R5/R6 evidence, anchored to Git-canonical detector) |
| `bestEligible` / `bestAny` replay | PASS with tie-break `(-IoU, -confidence, proposal_id)` |
| 12/12 role facts | PASS |
| Verifier result | STANDALONE_VERIFIER: PASS |
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

Watt was not needed for Task 8B.3-REF01-F1-R11 (no downloads, no transfers).

No model, detector, Qwen, SAM2, D-B1 or target-segmentation execution occurred; the canonical evidence and the external
delivery were not modified.
