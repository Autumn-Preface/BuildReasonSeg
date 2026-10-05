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

# FROM_DSH — Task 8B.3-REF01-E3B1-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3B1-R1` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-ref01-eligibility-repair-sync` |
| Starting HEAD | `8fdaf9c973af2b0b437d8892894e910c79cf2058` |
| Identity source | git object bytes of `f50404843f5189986f97633cd0edb6140b1d8034` (Windows working-tree bytes NOT used) |
| detector.py identity | 21257 bytes · `934bbb9c3fbdbd54...` |
| test_task8b_runtime.py identity | 28128 bytes · `71915e8abfbe96b3...` |
| Manifest changed lines | 4 |
| GIT_CANONICAL_135_135 | PASS |
| Helper `--check` | exit 1 · checked 135 · match 133 · missing 0 · mismatch 2 |
| External mismatch paths | ['buildreasonseg/runtime/detector.py', 'tests/test_task8b_runtime.py'] |
| External pre-sync exactly as required | True |
| External RC1 written / real sync | NO / NO |
| pytest / py_compile / inference | NONE / NONE / NONE |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Evidence | `evaluation\task8b3_ref01_eligibility_repair_manifest_correction.json` |
| Report | `docs/task8b3_ref01_eligibility_repair_impl.md` |
| Next gate | external RC1 controlled sync plus external full suite (E3B), not executed |
| Next action | Awaiting ChatGPT audit |

Watt was not needed for Task 8B.3-REF01-E3B1-R1 (no downloads, no transfers).

Only the two identity entries in the canonical manifest were corrected; no external file was written and no inference
ran.
