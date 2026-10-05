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

# FROM_DSH — Task MASK01_F1_R1_CORRECTIVE_FORENSICS Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `MASK01_F1_R1_CORRECTIVE_FORENSICS` |
| Status | **COMPLETE_WITH_EVIDENCE_GAPS** |
| Starting branch / head | `audit/task8b3-mask01-validity-forensics` / `590d080648ef5ae1fc6891dd18b97972c643c3cc` |
| Task branch | `audit/task8b3-mask01-f1-r1-corrective-forensics` |
| Previous F1 disposition | `REJECTED_INCOMPLETE_FORENSICS` |
| predict_one SUCCESS contract verified | True (literal line evidence recorded) |
| Padding helper | `pipeline.py::_non_padding_mask` · references padding = False · all-True mask = True |
| padding_gate_conclusion | **PADDING_GATE_INEFFECTIVE** |
| padding_leakage_established | false |
| min_mask_* occurrences | 0 |
| Test files in the matrix | 6 |
| Known 8B.3 artifacts inspected | 3 |
| Additional bounded search (Stage A / Stage B) | 51 / 18 |
| Final-mask artifact classification | **SAVED_FINAL_MASK_MATERIAL_REFERENCED** |
| Evidence gaps | ['no min_mask_* setting exists in the audited canonical sources/configs'] |
| Product source modified / inference / training / repair | NO / NO / NO / NO |
| repair_decision | `DEFER_TO_CHATGPT` |
| next_gate | `CHATGPT_MASK01_F1_R1_REVIEW` |
| Evidence | `evaluation/task8b3_mask01_f1_r1_corrective_forensics.json` |
| Report | `docs/task8b3_mask01_f1_r1_corrective_forensics.md` |
| Next action | Awaiting ChatGPT F1-R1 review |

Watt was not needed for Task MASK01_F1_R1_CORRECTIVE_FORENSICS (no downloads, no transfers).

The commit deliberately records `final_commit_sha = POST_COMMIT_EXTERNAL_FACT`; the actual local and remote heads are
printed to the terminal after the push, per the task book rule.
