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

# FROM_DSH — Task 8B.3-M1B.1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-M1B.1` |
| Status | **COMPLETE** |
| Branch | `fix/task8b3-mem01-compact-proposals` |
| Starting HEAD | `a215db142caec155e2f6787804378e442fbb55d5` |
| Canonical manifest | 135/135 PASS |
| Canonical `source_manifest.json` SHA256 | `1135d8b44d882e5462ed9cfb627322a5b2895c4874ec97fcdd61d6f2e6567497` |
| Pre-sync external setup | READY |
| Manifest-listed files copied | 135 / 135 |
| External manifest match | 135/135 PASS |
| `source_manifest` byte-identical | YES |
| Protected external assets changed | NO |
| Post-sync external setup | READY |
| Pytest | NOT RUN BY DESIGN |
| Real inference | NOT RUN |
| External delivery modified | YES — CONTROLLED MANIFEST SYNC ONLY |
| PROP-01 / REF-01 / MASK-01 | UNCHANGED / UNCHANGED / UNCHANGED |
| Report | `docs/task8b3_m1b1_external_delivery_sync.md` |
| STOP reason | none |
| Next action | Awaiting ChatGPT audit before M1B.2 external full regression. |

Watt was not needed for Task 8B.3-M1B.1 (no downloads, no transfers).

The external delivery was modified **only** by the controlled 135-entry manifest sync plus the separately copied
`source_manifest.json`; no model binary, component asset, user/runtime output, log or dataset was touched, and no
file was deleted. No pytest, predict, Demo or model inference was executed, and no PROP-01/REF-01/MASK-01 repair was
attempted.
