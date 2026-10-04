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

# FROM_DSH — Task 8B.3-P1D5-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D5-R1` |
| Status | **COMPLETE** (evidence completion, docs only) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `d17517b53b1cb9b82cbecc0432a5023eaa756278` |
| Active detector class count / names | `class_count = 1` · `class_names = {0: building}` (source `artifacts/task6m_yolo_native/data.yaml`) |
| Stage-1 / stage-2 settings | parsed: epochs 80 · imgsz 640 · batch 16 · device `0` · seed 20260812 · optimizer auto · close_mosaic 10 · patience 15 · pretrained true · resume false (stage 1) / resume `last_resume.pt` (stage 2); final active-lineage metrics mAP50(B) 0.73888 · mAP50-95(B) 0.44319 · mAP50(M) 0.72415 · mAP50-95(M) 0.39172 |
| `baseline/yolo_whu` relation | **SEPARATE_HISTORICAL_BASELINE** (YOLOv8m-seg `d9a6a65b…`, frozen reference-only record, separate legacy project; no lineage to the YOLO26m-seg active model) |
| A2 non-model pixel statistics | R mean 60.0654/std 6.3738 · G 68.4044/6.4936 · B 66.4532/6.9203 · uint8 · 15 211 unique colours · no saturated pixels · luminance mean 64.9744/std 6.3520 |
| Training-domain vs A2 table | recorded; dataset radiometry/geography NOT ESTABLISHED ⇒ mismatch vs in-domain membership undecidable |
| Alternate validation status | **VALIDATED_ALTERNATE_AVAILABLE** (epoch-18 YOLO26m-seg and frozen YOLOv8m baseline have real project validation evidence; pretrained bases and smoke runs do not) |
| Alternate diagnostic use under freeze | **ALLOWED_AS_SEPARATE_DIAGNOSTIC** |
| Scientific-freeze impact | **DETECTOR_CHANGE_TOUCHES_FROZEN_RESEARCH_CLAIMS** (formal adoption would need re-validation and re-freeze) |
| Primary diagnosis | **PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE** (unchanged) |
| Next gate (recommended, not executed) | `NEXT = DOMAIN_EVIDENCE_RECOVERY` |
| Model inference / pytest / predict / training / alternate run / checkpoint load | NONE |
| Files changed | `docs/task8b3_p1d5_detector_provenance_domain_gap.md`, `handoff/FROM_DSH.md`, `handoff/TO_DSH.md` only |
| Report | `docs/task8b3_p1d5_detector_provenance_domain_gap.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommended gate without a new task book. |

Watt was not needed for Task 8B.3-P1D5-R1 (no downloads, no transfers).

No inference, test run, training, fine-tuning or checkpoint load occurred; external delivery, runtime, tests,
manifest and checkpoints were not modified; `RC1-DEMO-PROP-01` remains open and `RC1-DEMO-MEM-01` remains CLOSED.
