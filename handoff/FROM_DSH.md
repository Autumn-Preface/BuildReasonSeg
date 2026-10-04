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

# FROM_DSH — Task 8B.3-M1A.2C-D1.1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1A.2C-D1.1` |
| Status | **COMPLETE** (documentation normalization only) |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `6405c64c7bba40c398b16bd9f3050c3c95cc7db8` |
| New report | `docs/task8b3_m1a2c_canonical_suite_forensics.md` |
| Nodes listed | 23 (all recovered from `.pytest_cache/v/cache/lastfailed`) |
| Frozen classification | **A=10, B=13, C=0, D=0, E=0** |
| `tests/test_task8b_runtime.py` failing nodes | NONE |
| Frozen gate conclusion | **CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE** |
| Old M1A report | A=7/B=10/D=6 and `MIXED_FAILURES_REQUIRE_CODE_AUDIT` marked superseded |
| pytest / check_setup / predict / model runs | NONE |
| Canonical product / tests / manifest / external delivery | NOT modified |
| Report | `docs/task8b3_m1a2c_canonical_suite_forensics.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit; do not sync delivery. |

Watt was not needed for Task 8B.3-M1A.2C-D1.1 (no downloads, no transfers).
