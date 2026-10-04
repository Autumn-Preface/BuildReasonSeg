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

# FROM_DSH — Task 8B.3-REF01-F1-R9 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R9` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `f268d03a9a70b494b7134c6b2f2647ed3468caa3` |
| Detector / model calls | 0 |
| Verification mode | READ_ONLY_HISTORICAL_EVIDENCE_REPLAY |
| Source identity basis | GIT_CANONICAL_BLOB_BYTES |
| External detector identity | `buildreasonseg/runtime/detector.py` · manifest = external = `82531dc3b758cd8a...` · match True |
| External imageio identity | `buildreasonseg/runtime/imageio.py` · manifest = external = `b6223be7cb2ab0e0...` · match True |
| Live P1D12 metadata replay | YES (read from the external diagnostics `proposals.json` for all four tiles) |
| Tie-break | `(-IoU, -confidence, proposal_id)` for best_eligible and best_any |
| Verifier result | REPLAY_VERIFIER: PASS |
| Canonical evidence schema replaced | YES (final schema, obsolete imageio structures removed) |
| Exact scientific reuse disclosure present | YES (English verbatim + Chinese translation) |
| Coverage threshold | 0.50 |
| Class counts | {"REFERENCE_SELECTED_CORRECT": 1, "REFERENCE_SELECTION_WRONG_COVERED": 2, "REFERENCE_ELIGIBILITY_BLOCKED": 1} |
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

Watt was not needed for Task 8B.3-REF01-F1-R9 (no downloads, no transfers).

No model, detector, Qwen, SAM2, D-B1 or target-segmentation execution occurred; no delivery file was modified.
