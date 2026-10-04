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

# FROM_DSH — Task 8B.3-P1D6 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D6` |
| Status | **COMPLETE** (isolated alternate-detector A2 probe) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `ede134cf2fea622c1e3ea2229ea6417a8633c8c1` |
| Alternate checkpoint | `artifacts/checkpoints/task6m/runs/m1_yolo26m_seg/weights/best.pt` · 162 481 487 B · sha256 `fd407db634a8a7ef83f09f8096686e73407095105f1b45c70c623d18dbf4ea44` |
| Changed variable | checkpoint only (frozen 512/128/384 · imgsz 640 · conf 0.05 · max_det 300 · retina_masks False · device cpu) |
| Probe processes / predict calls | 1 / 9 (one per tile) |
| Per-tile result | all 9 tiles: boxes_count = 0 · masks_is_none = True · masks_count = 0 · conf_max = None · wrapper_equivalent_count = 0 |
| Aggregates | sum_boxes_count = 0 · sum_wrapper_equivalent_count = 0 · tiles_with_boxes = [] · global_max_confidence = None |
| Outcome | **PROP01_EPOCH18_ALSO_ZERO** |
| Recommendation (not executed) | `NEXT = CONTROLLED_YOLOV8M_A2_PROBE` |
| Post-run integrity | manifest before/after 135/135 PASS · byte-identical YES |
| External RC1 / alternate adoption | NOT modified / NOT adopted |
| Model inference beyond the 9 alternate calls | NONE (no active rerun, no full-frame, no YOLOv8m, no full predict) |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d6_epoch18_alternate_a2_probe.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommended gate without a new task book. |

Watt was not needed for Task 8B.3-P1D6 (no downloads, no transfers).

No external RC1 product/runtime/test/manifest file and no checkpoint was modified; the alternate checkpoint was used
read-only for diagnostic evidence only. `RC1-DEMO-MEM-01` remains CLOSED.
