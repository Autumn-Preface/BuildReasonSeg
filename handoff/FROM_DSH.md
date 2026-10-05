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

# FROM_DSH — Task MASK01_F2_LOCKED_SUCCESS_ARTIFACT_FORENSICS Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_F2_LOCKED_SUCCESS_ARTIFACT_FORENSICS` |
| Status | **COMPLETE** |
| Starting branch / head | `audit/task8b3-mask01-f1-r1-corrective-forensics` / `e8f1315c9afdaac818126db6a2ac76b758027f70` |
| Task branch | `audit/task8b3-mask01-f2-locked-success-artifacts` |
| Cases inspected | A1, A3, A4 only |
| Final mask PNG availability | **FOUND** (found: ['A1', 'A3', 'A4']) |
| foreground_pixel_count == result.json.mask_area | ['A1', 'A3', 'A4'] |
| Taxonomy | A1/A3/A4 = TAX_SEMANTIC_TARGET_MISMATCH; A1/A4 additionally TAX_RUNTIME_VALID_QUALITY_POOR |
| Connected-component signals | NOT_CURRENTLY_AVAILABLE (DERIVABLE_FROM_RUNTIME_MASK) |
| Numeric threshold justified by three cases | NO |
| Mask regenerated / model run / external write / repair | NO / NO / NO / NO |
| Evidence gaps | NONE |
| repair_decision | `DEFER_TO_CHATGPT` |
| next_gate | `CHATGPT_MASK01_F2_REVIEW` |
| Evidence | `evaluation/task8b3_mask01_f2_locked_success_artifacts.json` |
| Report | `docs/task8b3_mask01_f2_locked_success_artifacts.md` |
| Next action | Awaiting ChatGPT F2 review |

Watt was not needed for Task MASK01_F2_LOCKED_SUCCESS_ARTIFACT_FORENSICS (no downloads, no transfers).

The commit records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; the actual local and remote heads are printed after
the push, per the task book rule.
