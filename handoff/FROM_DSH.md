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

# FROM_DSH — Task 8B.3-REF01-F1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1` |
| Status | **STOP** (preflight gates only; forensic procedure not executed) |
| Base branch @ HEAD | `fix/task8b3-prop01-a2-zero-proposals` @ `e00396c85e0fb0ac966f72fa7216b34c72b94cb7` |
| Task branch | `fix/task8b3-ref01-reference-forensics` (created at the same HEAD) |
| External RC1 integrity | 135/135 PASS · control manifest identical |
| v0.2 TEST identity gate | PASS — four immutable records resolve uniquely; `dataset_version v0.2`; `whu-native-vector-v1.0`; one native-vector reference each matching `reference_component_ids[0]` |
| GT access purpose | REFERENCE_FORENSICS_ONLY |
| Frozen detector passes used | 0 |
| Qwen / SAM2 / D-B1 / target segmentation | NOT EXECUTED |
| Candidate replacement / repair / visual judgement | NONE / NONE / NONE |
| External delivery / canonical RC1 modified | NO / NO |
| Forensic classification / outcome enum | NOT ASSERTED (inputs unavailable) |
| Locked candidate identities | UNCHANGED |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| STOP reason | execution budget exhausted after branch creation and the read-only preflight gates (§7–§9), before §10 GT-mask reconstruction, the single authorised detector pass and the §14–§18 IoU classification; nothing was partially implemented |
| Next action | Awaiting ChatGPT audit; restart the forensics from §10 on the same branch and HEAD |

Watt was not needed for Task 8B.3-REF01-F1 (no downloads, no transfers).

No model, detector, Qwen, SAM2 or D-B1 execution occurred; no delivery file was modified; no GT-derived metric was
computed or claimed.
