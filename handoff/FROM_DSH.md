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

# FROM_DSH — Task 8B.3-P1D2 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D2` |
| Status | **COMPLETE** (single detector-only A2 probe) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `dfc66e5a97639074149ce66e6cd4a6f0180e3e71` |
| Probe processes / predict calls | 1 / 9 (one per tile, no rerun, `detect_tile()` not called) |
| Frozen settings | TILE_SIZE 512 · overlap 128 · stride 384 · IMGSZ 640 · CONF 0.05 · MAX_DET 300 · retina_masks False · original detector checkpoint (54 480 241 B) · device cpu |
| Per-tile result | all 9 tiles: `boxes_count = 0`, `masks_is_none = True`, `masks_count = 0`, `conf_max = None`, `wrapper_output_count = 0` |
| Aggregates | `sum_boxes_count = 0` · `sum_wrapper_output_count = 0` · `tiles_with_boxes = []` |
| NMS warning | NOT reproduced; attribution map empty |
| Outcome | **PROP01_MODEL_ZERO_AT_FROZEN_CONF_CONFIRMED** |
| P1D1 classification | normalized to `PROP01_INSUFFICIENT_EVIDENCE` (ChatGPT audit override recorded in that report) |
| Post-run integrity | external manifest 135/135 PASS · byte-identical YES |
| detector.py / tests / manifest / external functional files | NOT modified |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d2_a2_detector_result_probe.md` |
| Next action | Awaiting ChatGPT audit; no further A2 run without a new task book. |

Watt was not needed for Task 8B.3-P1D2 (no downloads, no transfers).

No full predict, Demo, threshold change, pytest or `check_setup.py` run occurred; `RC1-DEMO-MEM-01` remains CLOSED.
