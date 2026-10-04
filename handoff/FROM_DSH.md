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

# FROM_DSH — Task 8B.3-P1D3 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D3` |
| Status | **COMPLETE** (single full-frame A2 detector probe) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `54fd6f4c8bf94f893a8c48cf4c0ce5f2132cf386` |
| Probe processes / predict calls / tiling | 1 / 1 / none (full 1024×1024 RGB) |
| Settings | original checkpoint (54 480 241 B) · imgsz 640 · conf 0.05 · max_det 300 · retina_masks False · device cpu · NMS unchanged |
| Full-frame result | boxes_count = 0 · masks_is_none = True · masks_count = 0 · conf_max = None · wrapper_equivalent_count = 0 |
| P1D2 tiled fact | 9/9 tiles boxes_count = 0, wrapper_output_count = 0 |
| NMS warning | NOT reproduced |
| Outcome | **PROP01_GLOBAL_ZERO_AT_FROZEN_CONF_CONFIRMED** |
| Recommendation (not executed) | `NEXT = CONTROLLED_SUBTHRESHOLD_A2_PROBE` |
| Post-run integrity | manifest 135/135 PASS · byte-identical YES |
| detector.py / tests / manifest / external functional files | NOT modified |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d3_a2_fullframe_context_probe.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommendation without a new task book. |

Watt was not needed for Task 8B.3-P1D3 (no downloads, no transfers).

No tiled A2 rerun, full predict, inspect-proposals or other-case run occurred; `RC1-DEMO-MEM-01` remains CLOSED.
