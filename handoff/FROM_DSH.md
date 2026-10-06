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

# FROM_DSH — Task MASK01_D1_R1E1_TARGETED_CANONICAL_GATES Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_D1_R1E1_TARGETED_CANONICAL_GATES` |
| Status | **COMPLETE** |
| Starting branch / head | `fix/task8b3-mask01-success-semantics-r1e` / `5c94163097605bb05d0373f12f7c6cb3d5681d9f` |
| Task branch | `fix/task8b3-mask01-success-semantics-r1e1` |
| Gate A (5 SUCCESS semantics tests) | exit 0 · 5 passed in 1.50s · passed = True |
| Gate B (2 inspect-proposals contract tests) | exit 0 · 2 passed in 3.61s · passed = True |
| Gate C (135-entry Git canonical manifest validation) | basis=GIT_CANONICAL_BLOB_BYTES · entries=135 · digest=7967127fcfadfe4f… · mismatch=0 · passed = True |
| Full `pytest -q` suite | NOT run |
| Canonical writes (product/tests/manifest/docs) | NONE |
| Repairs attempted | NONE |
| Evidence | `evaluation\task8b3_mask01_d1_r1e1_targeted_canonical_gates.json` |
| Report | `docs\task8b3_mask01_d1_r1e1_targeted_canonical_gates.md` |
| Next gate | `CHATGPT_R1E1_REMOTE_AUDIT` |

Watt was not needed for Task MASK01_D1_R1E1_TARGETED_CANONICAL_GATES (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; local and remote heads are printed after the push.
