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

# FROM_DSH — Task 8B.3-P1D7 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D7` |
| Status | **COMPLETE** (isolated YOLOv8m baseline A2 probe) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `48ea366e95fdfa022fd6e922f0d5656ad673941b` |
| Baseline checkpoint | `WHU_Building_Segment/runs/segment/logs/whu_building_v1/weights/best.pt` · 54 835 548 B · sha256 `d9a6a65b7e0819ce4ecbbd9d44a5c8f9dcd2e60ea78203ba8fdf90ba6aaa1f91` (matches frozen value) |
| Runtime used | RUNTIME python=3.11.16 torch=2.13.0+cu132 ultralytics=8.4.164 (RC1 pinned repository environment) |
| Frozen detector settings | 512/128/384 → 9 tiles · imgsz 640 · conf 0.05 · max_det 300 · retina_masks False · device cpu |
| Probe processes / predict calls | 1 / 9 (one per tile) |
| Per-tile result | all 9 tiles: boxes_count = 0 · masks_is_none = True · masks_count = 0 · conf_max = None · wrapper_equivalent_count = 0 |
| Aggregates | sum_boxes_count = 0 · sum_wrapper_equivalent_count = 0 · global_max_confidence = None |
| Outcome | **PROP01_YOLOV8M_ALSO_ZERO** |
| Recommendation (not executed) | `NEXT = A2_INPUT_DOMAIN_DECISION` |
| Post-run integrity | manifest before/after 135/135 PASS · byte-identical YES |
| External RC1 detector | NOT modified / NOT replaced; YOLOv8m NOT adopted |
| Other detector runs | NONE (no active, no epoch-18, no other checkpoint) |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d7_yolov8m_whu_a2_probe.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommended gate without a new task book. |

Watt was not needed for Task 8B.3-P1D7 (no downloads, no transfers).

No full predict, Demo, pytest or legacy-environment run occurred; the legacy environment was not switched to; no
external RC1 product/runtime/test/manifest file or checkpoint was modified. `RC1-DEMO-MEM-01` remains CLOSED.
