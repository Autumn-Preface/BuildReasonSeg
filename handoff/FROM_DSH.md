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

# FROM_DSH — Task 8B.3-REF01-F1-R8 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R8` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `65643f802a0a187b93160155f976689c0b50b8c6` |
| Detector / model calls | 0 |
| `detector.py` manifest identity | in manifest = True · manifest = Git blob = external = `82531dc3b758cd8a...` · all agree = True |
| `imageio.py` manifest identity | in manifest = True · Git blob present = True · external present = True (recorded verbatim) |
| Tracked forensic script | rewritten as a pure read-only replay verifier (no `DetectorRuntime`, no `detect_global`, no predict) |
| Verifier result | REPLAY_VERIFIER: PASS |
| Completed IDs | right: sel 1 / bestElig 1 / bestAny 1 · left: sel 14 / bestElig 30 / bestAny 30 · above: sel 4 / bestElig 2 / bestAny 5 · below: sel 1 / bestElig 2 / bestAny 2 |
| Canonical evidence | updated with the ID completion block, detector identity and imageio module identity |
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

Watt was not needed for Task 8B.3-REF01-F1-R8 (no downloads, no transfers).

No model, detector, Qwen, SAM2, D-B1 or target-segmentation execution occurred; no delivery file was modified.
