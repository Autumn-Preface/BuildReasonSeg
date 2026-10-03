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

# FROM_DSH — Task 8B.3-D1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in
git history._

| item | value |
|---|---|
| Task | `8B.3-D1` |
| Status | **COMPLETE** (read-only forensics) |
| Branch | `eval/task8b3-six-image-demo-suite` |
| Starting HEAD | `648dc9d48886f6da478f8e27b54116d48ca1d725` |
| Inference executed | NO |
| Delivery modified | NO |
| Product/harness/tests modified | NO |
| Reference forensics | A1/A3/A4 each correctly selected the largest **eligible** proposal (A1 #48 4853, A3 #1 680, A4 #30 3622), but the largest-area proposals (A1 #6 6470, A3 #4 2023, A4 #29 10404) were all rejected by the frozen extent/border rule → eligibility filtering **and** proposal fragmentation; implementation “largest” = largest eligible detected proposal, not the visually largest complete building |
| A2 forensics | detector/proposal-stage failure: 9 tiles, **0 raw** and 0 merged proposals, `WARNING NMS time limit 2.050s exceeded` recorded; NMS causality NOT established |
| B1/B2 memory signature | **EXACT_SIGNATURE_MATCH** — `Unable to allocate 23.8 MiB for an array with shape (5000, 5000) and data type bool` matches `detector.py:265 global_mask = np.zeros((height_px, width_px), dtype=bool)` (25,000,000 B = 23.84 MiB), allocated once per raw detection |
| Defects | `RC1-DEMO-REF-01` = CONFIRMED (blocker) · `RC1-DEMO-MASK-01` = CONFIRMED (major) · `RC1-DEMO-PROP-01` = CONFIRMED, NMS causality NOT CONFIRMED (major) · `RC1-DEMO-MEM-01` = CONFIRMED (blocker) |
| Report | `docs/task8b3_d1_demo_failure_forensics.md` |
| Output-layout proposal | ACCEPTED / STILL DEFERRED TO TASK 8B.4 |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit. |

Priority recommendation (factual, not implemented): RC1-DEMO-MEM-01 → RC1-DEMO-PROP-01 → RC1-DEMO-REF-01 →
RC1-DEMO-MASK-01 → packaging/output layout (Task 8B.4) → free manual Demo.

Watt was not needed for Task 8B.3-D1 (no downloads, no transfers).

No inference, no delivery modification and no source/test/harness change occurred in this task; the frozen R4B
baseline and the ChatGPT visual labels are reproduced unchanged.
