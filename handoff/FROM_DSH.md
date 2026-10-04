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

# FROM_DSH — Task 8B.3-P1D10-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D10-R1` |
| Status | **COMPLETE** (resolution-logic correction + candidate-pool audit, docs only) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `2edee1af99d89740cf5e75c99b08fe872b0bcef1` |
| Model/test execution | NONE |
| Functional files modified | NO |
| Corrected error 1 | original replacement pool was the detector's own validation split → **MODEL_SELECTION_LEAKAGE_RISK**, withdrawn |
| Corrected error 2 | original PROP-01 reclassification was unsupported (A2 provenance NOT ESTABLISHED) → withdrawn |
| Pool audit | A `images/val` MODEL_SELECTION_LEAKAGE_RISK · B `images/test` MISSING_RELATION_METADATA · C `whu_native_vector/v1.0` MISSING_RELATION_METADATA · D `reasoning_view/scene_disjoint_v1` POOL_STATUS_INCOMPLETE · E1 `task6m1_demo` (18 cases) **USABLE_WITH_DISCLOSURE** · E2 `task6m_demo` (13 cases) **USABLE_WITH_DISCLOSURE** |
| Best Demo candidate pool | `artifacts/task6m1_demo` (primary), `artifacts/task6m_demo` (secondary) |
| Replacement selection policy | READY — metadata-only contract declared in §19.3; detector outputs and manual visual quality may not be consulted; no image selected |
| Detector adaptation feasibility | **DETECTOR_ADAPTATION_NOT_READY** (unchanged; no readiness facts changed) |
| Demo policy feasibility | **DEMO_POLICY_PATH_READY** |
| Primary resolution | **PROP01_RESOLUTION_DEMO_POLICY** (retained on the corrected contract) |
| PROP-01 status | **PROP01_OPEN_ENGINEERING_DEFECT** (corrected; not closed) |
| Scientific freeze preserved | YES |
| Next gate | **PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION** (recommended, not executed) |
| RC1-DEMO-MEM-01 / PROP-01 / REF-01 / MASK-01 | CLOSED / OPEN / OPEN / OPEN |
| Report | `docs/task8b3_p1d10_prop01_resolution_decision.md` |
| Next action | Awaiting ChatGPT audit; do not execute the next gate and do not select any Demo case. |

Corrected release wording (≤120 Chinese characters): RC1 的 proposal detector 已在 WHU 类航空建筑实例域上完成验证；演示用例取自该域内既有资产，选择规则在检测结果之外预先确定；A2 为已如实记录的未检出样例，本版本不承诺任意航空影像的鲁棒性。

Watt was not needed for Task 8B.3-P1D10-R1 (no downloads, no transfers).

No model, test, training, inference or functional modification occurred; the audit read only local frozen assets and
documentation.
