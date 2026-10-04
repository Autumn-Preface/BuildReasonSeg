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

# FROM_DSH — Task 8B.3-P1D5 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D5` |
| Status | **COMPLETE** (read-only provenance/domain audit) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `15a90ec11ad66c8f365284bf53eb320661a1e73f` |
| Active detector family/task | YOLO26m-seg instance segmentation (frozen U-C1 proposal model) |
| Active checkpoint | `model/buildreasonseg_advisor/detector.pt` · 54 480 241 B · sha256 `ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474` |
| Provenance match | equals documented `artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt` and `model.yaml` sha256 |
| Training/fine-tuning dataset | `artifacts/task6m_yolo_native/data.yaml` (WHU building lineage); epochs/hyperparameters NOT ESTABLISHED |
| U-C1 selection evidence | ESTABLISHED (CLI defaults + all task6m1 demo results + model.yaml) |
| A2 source/domain | NOT ESTABLISHED |
| Alternative detectors | VALIDATED_ALTERNATE_AVAILABLE (epoch-18 YOLO26m-seg, YOLOv8m-seg-WHU baseline, pretrained bases) — none validated on A2/RC1 |
| Scientific-freeze compatibility | **DETECTOR_CHANGE_TOUCHES_FROZEN_RESEARCH_CLAIMS** |
| Primary diagnosis | **PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE** |
| Next gate (recommended, not executed) | `NEXT = DOMAIN_EVIDENCE_RECOVERY` |
| Model inference / pytest / predict / training / download | NONE |
| Checkpoint binary loaded | NO (hash/metadata only) |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d5_detector_provenance_domain_gap.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommended gate without a new task book. |

Watt was not needed for Task 8B.3-P1D5 (no downloads, no transfers).

No checkpoint was loaded, replaced or fine-tuned; no functional file changed; `RC1-DEMO-MEM-01` remains CLOSED.
