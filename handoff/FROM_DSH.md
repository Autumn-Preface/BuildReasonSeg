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

# FROM_DSH — Task 8B.3-M1A.2C Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1A.2C` |
| Status | **PARTIAL / STOP** |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `c432ec41f2d8bad9ddc67d65d0dbc033724e482a` |
| Files changed | `docs/task8b3_m1a_compact_proposal_masks.md` only |
| Canonical product / tests / manifest | UNCHANGED |
| Stale R3 failure wording | FIXED |
| Full canonical suite | exactly once → **FAIL (STOP)** (ERROR tests/test_setup_checker.py::test_missing_sam2_nonzero - FileNotFoundEr... | ERROR tests/test_setup_checker.py::test_invalid_model_yaml_nonzero - FileNotF... | 17 failed, 93 passed, 6 errors in 29.67s) |
| External delivery / real inference | NOT touched / NOT RUN |
| Report | `docs/task8b3_m1a_compact_proposal_masks.md` |
| STOP reason | the single canonical suite run failed; per the task book no code or test edit and no rerun were performed |
| Next action | Awaiting ChatGPT audit; do not sync delivery. |

Watt was not needed for Task 8B.3-M1A.2C (no downloads, no transfers).
