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

# FROM_DSH — Task 8B.3-P1D8-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D8-R1` |
| Status | **COMPLETE** (evidence completion, docs only) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `b4dc1967cdc6734d35315d29eed11eb284b50244` |
| Model runs / enhancement / detector rerun | NONE (read-only arithmetic only) |
| Full distributions | six metrics × train/val/test/combined (N, min, p01, p05, p25, p50, p75, p95, p99, max) written into the report §13.1 |
| Split file counts | train 10044 · val 3618 · test 3726 · total 17388 |
| Zero tail | train: Y_mean==0 1302, Y_std==0 1302, all-black 1302 (12.9630%), labels existing/empty/non-empty 1302/1302/0; val and test: 0 |
| Combined percentiles include all-black images | YES |
| A2 split-specific ranks | train Y_mean 38.88 / Y_std 13.45 · val Y_mean 11.22 / Y_std 0.00 · test Y_mean 1.83 / Y_std 0.27 · combined Y_mean 25.18 / Y_std 7.83 |
| A1/A2/A3/A4 tile summaries | recorded (report §13.4); A2 median tile Y_std 6.9624 vs A1 35.9739 / A3 34.1799 / A4 28.8475 |
| A2 p01/p05 tile counts | all twelve required counts = 0 (recorded) |
| Control ratios | Y_mean whole 0.8030 / tile 0.7834 · Y_std 0.1973 / 0.2037 · Y_dynamic_98 0.1746 / 0.2258 · gradient_mean 0.8457 / 0.9097 (A2 not lower than all three for gradient_mean) |
| Encoding enum | **A2_ENCODING_ANOMALY_NOT_FOUND** |
| Primary outcome | **PROP01_A2_NOT_PHOTOMETRIC_OUTLIER** (original rule retained; 4/4 core metrics ≥ p05 in the combined distribution) |
| Split robustness | **SENSITIVE_TO_SPLIT_COMPOSITION** |
| Recommendation (not executed) | `NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION` |
| A2 provenance/membership | NOT ESTABLISHED |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d8_a2_input_domain_audit.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommended gate without a new task book. |

Watt was not needed for Task 8B.3-P1D8-R1 (no downloads, no transfers).

No image was modified, enhanced, resized or normalised; no detector, Qwen, SAM2, D-B1, `predict.py` or pytest ran; the
machine-readable completion dump lives outside the tracked tree (`logs/task8b3_p1d8r1_completion.json`).
`RC1-DEMO-MEM-01` remains CLOSED.
