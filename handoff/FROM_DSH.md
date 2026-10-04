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

# FROM_DSH — Task 8B.3-P1D9 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D9` |
| Status | **COMPLETE** (isolated validation-moment rescue probe) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `a220faf0dd9872829ee2bd7b7be859bbd90db7bb` |
| Pre-declared transform | `y = clip(4.259966488571338 * x + -188.2294346811672, 0, 255)` (single transform, not adopted) |
| Transform effect | source Y_mean 65.6886/Y_std 6.3262 → transformed Y_mean 91.5928/Y_std 26.8645 |
| Clipping | below 0: 0.001974 · above 255: 0.000095 · either: 0.002070 |
| Temporary file | `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_p1d9\A2_validation_moment.png` (outside `inference/input`) |
| Detector / settings | active RC1 detector · 512/128/384 → 9 tiles · imgsz 640 · conf 0.05 · max_det 300 · retina_masks False · device cpu |
| Probe processes / predict calls | 1 / 9 (one per tile) |
| Per-tile result | all 9 tiles: boxes_count = 0 · masks_is_none = True · masks_count = 0 · wrapper_equivalent_count = 0 |
| Aggregates | sum_boxes_count = 0 · sum_wrapper_equivalent_count = 0 · global_max_confidence = None |
| Outcome | **PROP01_VALIDATION_MOMENT_RESCUE_REMAINS_ZERO** |
| Recommendation (not executed) | `NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION` |
| Integrity | original A2 unchanged = YES · active detector unchanged = YES · manifest 135/135 PASS · byte-identical YES |
| Transform adoption / other detectors / other enhancements | NONE / NONE / NONE |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d9_validation_moment_rescue_probe.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommended gate without a new task book. |

Watt was not needed for Task 8B.3-P1D9 (no downloads, no transfers).

No original-A2 run, no epoch-18/YOLOv8m run, no second enhancement, no CLAHE/gamma/equalisation, no full predict,
Qwen, SAM2, D-B1 or pytest; the RC1 input tree and all manifest-listed files are unchanged.
`RC1-DEMO-MEM-01` remains CLOSED.
