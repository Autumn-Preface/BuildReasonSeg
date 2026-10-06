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

# FROM_DSH — Task MASK01_D1_R1B_R2_EXACT_DOC_REPLACEMENT Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1B_R2_EXACT_DOC_REPLACEMENT` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1b-r1` / `79fc7e859cb0acbce2e86658cb65630026a20d1b` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1b-r2` |
| Documents modified | 4 (README, docs/model_card.md, docs/runtime_mapping.md, inference/README.md) |
| Global validation | README.md=PASS · docs/model_card.md=PASS · docs/runtime_mapping.md=PASS · inference/README.md=PASS |
| File-specific validation | README.md=PASS · docs/model_card.md=PASS · docs/runtime_mapping.md=PASS · inference/README.md=PASS |
| Contiguous phrase present in all four | True |
| scientific metrics / product source / tests / manifest changed | false / false / false / false |
| pytest / model inference / training / external write | not run / false / false / false |
| manifest_update | `DEFERRED_TO_R1_D` |
| github_persistence_policy | ALL_TASK_OUTCOMES_PUSHED |
| Evidence | `evaluation\task8b3_mask01_d1_r1b_r2_exact_doc_replacement.json` |
| Report | `docs\task8b3_mask01_d1_r1b_r2_exact_doc_replacement.md` |
| Next gate | `CHATGPT_R1B_R2_REMOTE_AUDIT` (R1-C not entered) |

Watt was not needed for Task MASK01_D1_R1B_R2_EXACT_DOC_REPLACEMENT (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local, remote and parent heads are printed after the
push.
