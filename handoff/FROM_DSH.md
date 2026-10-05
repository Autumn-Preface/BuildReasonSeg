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

# FROM_DSH — Task 8B.3-REF01-E3A-R5D Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3A-R5D` |
| Status | **COMPLETE** (canonical implementation closure reclassified) |
| Branch | `fix/task8b3-ref01-eligibility-repair-impl` |
| Starting HEAD | `06a031fdfb375f8ccd3689f23f4e384ebfa964ce` |
| pytest / py_compile this task | NONE / NONE (pure artifact closure) |
| Canonical runtime assets missing | 6 of 6 (`model/buildreasonseg_advisor/detector.pt, model/buildreasonseg_advisor/decoder.pt, model/components/sam2/sam2.1_hiera_base_plus.pt, model/components/sam2/sam2.1_hiera_b+.yaml, model/components/program_head/program_parser_l3_rehearsal_v1.pt, model/components/program_head/Qwen3-VL-2B-Instruct`) |
| External delivery assets present | 6 of 6 |
| Valid implementation regression evidence | targeted suite **40 passed in 0.69s** (exit 0, recorded from R5C) |
| Canonical full suite | **DEFERRED** until E3B external RC1 controlled sync (recorded R5C result: 17 failed / 101 passed / 6 errors) |
| Superseded R5C evidence | removed (`evaluation/task8b3_ref01_e3a_r5c_canonical_closure.json`) |
| Final E3A evidence | `evaluation\task8b3_ref01_e3a_canonical_implementation_closure.json` |
| detector.py / tests modified | NO / NO (`bc5aed885aa5f27b` / `8071fcc6a10e5f29`) |
| source_manifest updated | NO |
| External RC1 synced | NO |
| Detector / model inference | NONE |
| Evidence | `evaluation\task8b3_ref01_e3a_canonical_implementation_closure.json` |
| Report | `docs\task8b3_ref01_eligibility_repair_impl.md` |
| Next action | Awaiting ChatGPT audit; the canonical full suite runs after the E3B external RC1 controlled sync |

Watt was not needed for Task 8B.3-REF01-E3A-R5D (no downloads, no transfers).

Only artifact-level changes occurred: the superseded evidence was removed, the final E3A evidence was written, and the
report and handoff were updated.
