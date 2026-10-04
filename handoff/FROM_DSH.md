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

# FROM_DSH — Task 8B.3-P1D12 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D12` |
| Status | **COMPLETE** (locked-candidate proposal-only gate) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `6651c4bf89663e5fff3e7854c9e2115d084ae003` |
| Locked candidate status | FINAL_METADATA_LOCK (unchanged) |
| Locked candidate runtime status | RUN (4 of 4, once each, order right → left → above → below) |
| Per-candidate raw/merged/eligible | right=6/6/4, left=66/53/42, above=9/9/4, below=7/6/3 |
| Exit codes | 0, 0, 0, 0 |
| Diagnostics produced | `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\{1010,1003,1008,1009}` (authorised inspect-mode output) |
| Internal consistency | status SUCCESS · merged == proposals.count · tile_count 1 · tile_size 512 · overlap 128 (all four) |
| Pre-existing candidate diagnostics | NONE |
| Core chain (Qwen / SAM2 / D-B1 / reference selection / GT / visual) | NOT EXECUTED |
| Candidate replacement | NONE |
| PROP-01 status | PROP01_OPEN_ENGINEERING_DEFECT |
| Outcome | **PROP01_LOCKED_DEMO_PROPOSAL_GATE_PASS** |
| Next gate (recommended, not executed) | `NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS` |
| Report | `docs/task8b3_p1d12_locked_demo_proposal_gate.md` |
| Next action | Awaiting ChatGPT audit; the next gate needs its own task book |

Watt was not needed for Task 8B.3-P1D12 (no downloads, no transfers).

Only the authorised `--inspect-proposals` diagnostics were created under the external delivery; the canonical tree, the
external source/config files, the locked candidates and all model assets were untouched.
