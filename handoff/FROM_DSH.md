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

# FROM_DSH — Task 8B.3-P1D12-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D12-R1` |
| Status | **COMPLETE** (read-only evidence closure) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `75e01defde3ad6e9f8a247481028536e68ef4ee4` |
| Candidate / predict / detector / model executions | NONE |
| External sync check | 135/135 PASS (Git canonical byte comparison) |
| External setup | READY |
| Locked raster SHA identities | 4/4 PASS (1010 `1688306c…`, 1003 `eea4edd0…`, 1008 `0efe8bc2…`, 1009 `c22134e6…`) |
| Locked raster dimensions | 4/4 512×512 |
| Locked raster RGB readability | 4/4 PASS (RGB, TIFF) |
| Detector constants | MATCH (512 / 128 / STRIDE=TILE_SIZE-TILE_OVERLAP / 640 / 0.05 / 300 / 0.50 / 0.20) |
| Existing diagnostics file set | 4/4 PASS (exactly `global_proposals.png`, `parsed_program.json`, `prompt.txt`, `proposals.json`, `result.json`) |
| Inspect-only proof | PASS — no sam2/relation/d-b1/decoder/language/reference/qwen field and no mask/overlay artifact in any candidate diagnostics |
| eligible_largest recomputation | 4/4 match P1D12 values (right 4 · left 42 · above 4 · below 3) |
| Locked candidate identities | UNCHANGED |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Outcome | **PROP01_LOCKED_DEMO_PROPOSAL_GATE_EVIDENCE_CLOSED** |
| Next gate (recommended, not executed) | `NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS` |
| Report | `docs/task8b3_p1d12_r1_evidence_closure.md` |
| Next action | Awaiting ChatGPT audit; the next gate needs its own task book |

Watt was not needed for Task 8B.3-P1D12-R1 (no downloads, no transfers).

No model, detector or candidate was executed and no delivery file was modified; only this report and the handoff
changed.
