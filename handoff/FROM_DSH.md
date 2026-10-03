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

# FROM_DSH — Task 8B.3-D1.2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-D1.2` |
| Status | **COMPLETE** (documentation normalization only) |
| Branch | `eval/task8b3-six-image-demo-suite` |
| Starting HEAD | `e9bbfa4fc536335c8cbfd3f1afecdd8d6dedae6f` |
| Inference executed | NO |
| Delivery modified | NO |
| Product/harness/tests modified | NO |
| MEM-01 | CONFIRMED |
| Exception signature | EXACT_SIGNATURE_MATCH |
| Unique throwing allocation site | NOT CONFIRMED FROM EXISTING ARTIFACTS |
| Failure threshold | NOT ESTABLISHED |
| 5000×5000 path | BLOCKED IN B1/B2 BASELINE |
| >512 universal-failure claim | REMOVED |
| Report consistency | NORMALIZED |
| Encoding | UTF-8 WITHOUT BOM |
| Output-layout proposal | STILL DEFERRED TO TASK 8B.4 |
| Report | `docs/task8b3_d1_demo_failure_forensics.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit. |

Accepted D1 facts preserved (unchanged): **REF-01** — on A1/A3/A4 the implementation selects the largest *eligible*
proposal (A1 #48, A3 #1, A4 #30) while the user-facing “largest building” differs from that semantics
(fragmentation + eligibility filtering); **MASK-01** — SUCCESS validity is only non-empty + non-padding +
directional-centroid, so A1 (101 px) and A4 (359 px) returned SUCCESS with tiny fragments; **PROP-01** — A2 is a
detector/proposal-stage zero-proposal failure (9 tiles, 0 raw / 0 merged) with `WARNING NMS time limit 2.050s
exceeded` recorded but NMS causality NOT CONFIRMED. The frozen R4B baseline remains language 6/6, runtime 3/6 and
manual end-to-end semantic 0/6 — a qualitative six-sample Demo audit, not a research metric.

Watt was not needed for Task 8B.3-D1.2 (no downloads, no transfers).

No inference, no delivery modification, no defect fix and no source/test/harness change occurred in this task.
