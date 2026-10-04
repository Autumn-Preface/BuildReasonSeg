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

# FROM_DSH — Task 8B.3-M1B.2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1B.2` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `25d879845ef1508228067d4faa45e98266ed6aa5` |
| External preflight manifest | 135/135 PASS |
| `source_manifest` byte-identical before pytest | YES |
| External full pytest invocation count | 1 |
| External full suite | 116 passed in 65.12s (0:01:05) |
| Pytest exit code | 0 |
| Post-pytest external manifest | 135/135 PASS |
| Canonical product/tests/manifest modified | NO |
| External product/tests/manifest manually modified | NO |
| Real inference | NOT RUN |
| PROP-01 / REF-01 / MASK-01 | UNCHANGED / UNCHANGED / UNCHANGED |
| Gate result | **M1B_EXTERNAL_FULL_REGRESSION_PASS** |
| Report | `docs/task8b3_m1b2_external_full_regression.md` |
| Next action | Awaiting ChatGPT audit; do not run B1/B2. |

Watt was not needed for Task 8B.3-M1B.2 (no downloads, no transfers).

No canonical functional file and no external product/test/manifest file was modified; the external delivery changed
only through pytest's own transient cache files. No predict, Demo, model inference or PROP-01/REF-01/MASK-01 repair
was attempted.
