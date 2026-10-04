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

# FROM_DSH — Task 8B.3-REF01-F1-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-REF01-F1-R1` |
| Status | **STOP** (canonical GT reference masks reconstructed; detector pass still unused) |
| Branch | `fix/task8b3-ref01-reference-forensics` |
| Starting HEAD | `95f4403837bd7057fc331462503875fa847f43c8` |
| GT-reference masks reconstructed | 4/4 non-empty · area == `clipped_area_px` for all four = True |
| Per-candidate GT mask pixels | right=2478, left=3512, above=5013, below=2606 |
| Geometry cache / label key | right:label_map(uint8), left:label_map(uint8), above:label_map(uint8), below:label_map(uint8) |
| Frozen detector passes used | 0 (the one-per-image allowance is untouched) |
| Qwen / SAM2 / D-B1 / target segmentation | NOT EXECUTED |
| Candidate replacement / repair / visual judgement | NONE / NONE / NONE |
| External delivery / canonical RC1 modified | NO / NO |
| Forensic classification / outcome enum | NOT ASSERTED |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Report | `docs/task8b3_ref01_locked_reference_forensics.md` |
| STOP reason | the canonical GT side is complete, but the IoU comparison needs two frozen facts that are not yet pinned down in my evidence (how a proposal mask is reconstructed from `proposals.json`, and the fixed Task 7F coverage threshold); running the authorised detector pass without them would yield uninterpretable diagnostics |
| Next action | Awaiting ChatGPT audit — please confirm the proposal-mask reconstruction route and the Task 7F threshold so the §12–§18 chain can run in one turn |

Watt was not needed for Task 8B.3-REF01-F1-R1 (no downloads, no transfers).

No model, detector, Qwen, SAM2 or D-B1 execution occurred; no delivery file was modified; only this report and the
handoff changed.
