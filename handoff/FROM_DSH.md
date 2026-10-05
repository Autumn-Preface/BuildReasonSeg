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

# FROM_DSH — Task 8B.3-REF01-E3B1-R2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3B1-R2` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-eligibility-repair-sync` |
| Starting HEAD | `aa9da72ac840d04e5fe668a44935127dc4c1f1f0` |
| Canonical base | `f50404843f5189986f97633cd0edb6140b1d8034` |
| Authoritative evidence | `evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json` |
| Authoritative dedicated report | `docs/task8b3_ref01_e3b1_manifest_canonicalization.md` |
| Wrongly named R1 evidence removed | YES (`evaluation/task8b3_ref01_eligibility_repair_manifest_correction.json`) |
| Manifest semantic correction | PASS · Git-canonical validation PASS_135_OF_135 |
| External pre-sync check (recorded from R1) | exit 1 · match 133 · missing 0 · mismatch 2 |
| External mismatch paths | `buildreasonseg/runtime/detector.py`, `tests/test_task8b_runtime.py` |
| External write / real sync | NO / NO |
| Helper rerun / `--check` rerun / pytest / py_compile / inference | NONE |
| Manifest / detector / tests / pipeline / sync helper modified | NONE |
| detector_model_calls | 0 |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Overall outcome | **REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED** |
| Next gate | `REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE` (not executed) |
| Next action | Awaiting ChatGPT audit; E3B performs the external controlled sync and the external full suite |

Watt was not needed for Task 8B.3-REF01-E3B1-R2 (no downloads, no transfers).

This was an artifact-only closure: the wrongly named R1 evidence was removed, the exact final evidence and the
authoritative dedicated report were created, and the report/handoff were updated. No protected file was modified and no
verification command was re-run.
