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

# FROM_DSH — Task 8B.3-P1D4 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D4` |
| Status | **COMPLETE** (single sub-threshold tiled A2 probe) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `9f260ab035b5072d44b1979042fbcc63f7c458b8` |
| Probe processes / predict calls / conf values | 1 / 9 (one per tile) / 0.001 only |
| Settings unchanged | original checkpoint (54 480 241 B) · 512/128/384 · 9 tiles · IMGSZ 640 · MAX_DET 300 · retina_masks False · device cpu · NMS |
| Per-tile result | all 9 tiles: boxes_count = 0, masks_is_none = True, masks_count = 0, conf_max = None |
| Threshold histogram | 0.001→0 · 0.005→0 · 0.010→0 · 0.020→0 · 0.030→0 · 0.040→0 · 0.050→0 |
| global_max_confidence | None |
| NMS warning | NOT reproduced |
| Outcome | **PROP01_NO_MEANINGFUL_SUBTHRESHOLD_SIGNAL** |
| Recommendation (not executed) | `NEXT = DETECTOR_DOMAIN_GAP_DECISION` |
| Post-run integrity | manifest 135/135 PASS · byte-identical YES |
| detector.py / tests / manifest / external functional files | NOT modified |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d4_a2_subthreshold_tile_probe.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommendation without a new task book. |

Watt was not needed for Task 8B.3-P1D4 (no downloads, no transfers).

No product configuration change, no second confidence value, no full predict and no other case run occurred;
`RC1-DEMO-MEM-01` remains CLOSED.
