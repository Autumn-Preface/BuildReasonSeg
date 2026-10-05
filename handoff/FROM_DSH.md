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

# FROM_DSH — Task 8B.3-REF01-E3A-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-E3A-R1` |
| Status | **STOP** |
| Branch | `fix/task8b3-ref01-eligibility-repair-impl` |
| Starting HEAD | `628e9dedefcce6e2cba8a832de882d1df2b714cf` |
| Prescribed patcher SHA256 | `3c4323cb6e316c67238b9b775381faef9a6a474a56354afb550eb60c6db83604` |
| Verified patcher SHA256 | `3c4323cb6e316c67238b9b775381faef9a6a474a56354afb550eb60c6db83604` |
| Patcher path | `scripts\task8b3_ref01_eligibility_repair_patcher.py` |
| py_compile exit | 0 |
| Patcher runs | 1 |
| Patcher exit | 1 |
| detector.py before / after | `a6fa4bdd76db6f50` / `a6fa4bdd76db6f50` |
| Changed files | ['handoff/TO_DSH.md'] |
| source_manifest updated (canonical / external) | NO / NO |
| External RC1 synced | NO |
| Detector / model inference | NONE |
| Manual editing of detector or test files | NONE |
| Self-repair after failure | NONE |
| Manual visual inspection / candidate replacement | NO / NO |
| NEXT executed | NO |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Report | `docs\task8b3_ref01_eligibility_repair_impl.md` |
| Next action | Awaiting ChatGPT audit; the task book requires a stop without repair. |

Watt was not needed for Task 8B.3-REF01-E3A-R1 (no downloads, no transfers).

The prescribed patcher was copied verbatim, verified against its published SHA256 before execution, and run exactly once;
all repository changes originate from that patcher, and the source manifest, external delivery and model assets were
left unchanged with no inference executed.
