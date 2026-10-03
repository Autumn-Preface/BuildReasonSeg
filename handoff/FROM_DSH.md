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

# FROM_DSH — Task 8B.3-M1A Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.3-M1A` |
| Status | **PARTIAL / STOP** — frozen refactor not implemented in this turn; branch and dependency audit completed |
| Branch | `fix/task8b3-mem01-compact-proposals` (created from `7a9c576692abf82510d61fb1c814fd47d4b053a5`) |
| Starting HEAD | `7a9c576692abf82510d61fb1c814fd47d4b053a5` |
| MEM-01 implementation | NOT IMPLEMENTED (specified only) |
| Representation | tight global bbox + `mask_crop` — **planned, not applied** |
| Full-frame proposal masks retained | YES (unchanged; `detector.py:265` still present) |
| Full-frame pairwise IoU temporaries | YES (unchanged; `iou_of` still full-frame) |
| §4 dependency gate | PASS — `global_mask` usage confined to `detector.py`, `core.py`, `outputs.py`, `tests/test_task8b_runtime.py` (13 executable sites inventoried in the report) |
| Dedicated tests | NOT RUN |
| Canonical tests | NOT RUN |
| Manifest | 135/135 verified, no path or hash changed |
| External delivery modified | NO |
| Real inference executed | NO |
| Scientific model/checkpoint changed | NO |
| PROP-01 | UNCHANGED |
| REF-01 | UNCHANGED |
| MASK-01 | UNCHANGED |
| Report | `docs/task8b3_m1a_compact_proposal_masks.md` |
| STOP reason | execution budget exhausted after the branch/dependency gates; the refactor plus its seven test groups, manifest refresh and two test gates were not attempted, so no functional canonical file was modified |
| Next action | Awaiting ChatGPT audit; do not sync delivery. |

No functional canonical file (`detector.py`, `core.py`, `outputs.py`, `tests/test_task8b_runtime.py`,
`source_manifest.json`) was modified; no external-delivery file was touched; no inference, package change, training
or final-test access occurred; no detector model/threshold/tiling, merge threshold/winner/ID, Reference
eligibility/selection, language/SAM2/D-B1/relation/context/validity semantics were changed.

Watt was not needed for Task 8B.3-M1A (no downloads, no transfers).
