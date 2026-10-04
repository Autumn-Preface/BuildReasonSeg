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

# FROM_DSH — Task 8B.3-M1A.2A-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.3-M1A.2A-R1` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `10d36117993f128c9a3d6bd2bea114dc63120121` |
| Fix | `core.reference_mask_from_proposal()` now uses original image ∩ 512 reasoning context ∩ `proposal.global_bbox` |
| Files changed | `buildreasonseg/runtime/core.py` only |
| py_compile gate | PASS |
| legacy-equivalence smokes | 3/3 PASS (context at proposal top-left; proposal clipped top-left; proposal clipped bottom-right) |
| `detector.py` / `outputs.py` / tests / manifest / external delivery | NOT modified |
| Real inference executed | NO |
| Scientific model/checkpoint changed | NO |
| PROP-01 / REF-01 / MASK-01 | UNCHANGED / UNCHANGED / UNCHANGED |
| Report | `docs/task8b3_m1a_compact_proposal_masks.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit; do not sync delivery. |

Watt was not needed for Task 8B.3-M1A.2A-R1 (no downloads, no transfers).

No detector/outputs/test/manifest/external-delivery file, package, threshold, merge rule, Reference semantic,
ProgramHead, SAM2, D-B1 or SUCCESS-validity code was modified; no pytest suite, predict run or Demo was executed.
