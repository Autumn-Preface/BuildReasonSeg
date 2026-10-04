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

# FROM_DSH — Task 8B.3-P1D10 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D10` |
| Status | **COMPLETE** (docs-only resolution decision audit) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `70e2e2361e9e16cca891138de5d2712318282b2b` |
| Model/test execution | NONE |
| Functional files modified | NO |
| Demo suite policy provenance | PARTIAL (suite definition/result documented; input provenance NOT ESTABLISHED) |
| Detector adaptation feasibility | **DETECTOR_ADAPTATION_NOT_READY** (no A2 GT, no adaptation spec, no held-out protocol, no acceptance metric) |
| Demo policy feasibility | **DEMO_POLICY_PATH_READY** |
| Replacement selection policy | **READY** (predeclared pool = frozen val split, deterministic filename order, detector outcome not consulted) |
| Primary resolution | **PROP01_RESOLUTION_DEMO_POLICY** |
| PROP-01 status | **PROP01_RECLASSIFIED_SUPPORTED_DOMAIN_FAILURE** (reclassification, not closure) |
| Scientific freeze preserved | YES (frozen research architecture and results untouched; only policy text proposed) |
| Next gate | **PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION** (recommended, not executed) |
| RC1-DEMO-MEM-01 | CLOSED |
| RC1-DEMO-PROP-01 | RECLASSIFIED (not closed) |
| RC1-DEMO-REF-01 | OPEN |
| RC1-DEMO-MASK-01 | OPEN |
| Report | `docs/task8b3_p1d10_prop01_resolution_decision.md` |
| Next action | Awaiting ChatGPT audit; do not execute the next gate. |

Release-language draft (≤120 Chinese characters): RC1 的 proposal detector 在 WHU 类航空建筑影像上验证；A2 属其有效域外的压力/失败样例，已如实记录，不计入成功演示；本版本不承诺任意航空影像的鲁棒性。

Watt was not needed for Task 8B.3-P1D10 (no downloads, no transfers).

No model, test, training or inference execution and no functional modification occurred; the six intervention impact
classifications use the task book's exact enums, and the four resolution options are compared explicitly.
