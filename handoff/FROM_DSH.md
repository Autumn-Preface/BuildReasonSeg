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

# FROM_DSH — Task 8B.3-P1D8 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D8` |
| Status | **COMPLETE** (non-model A2 input-domain audit) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `5f52338b7507cc377190cacb33611c644f223a88` |
| Model runs | NONE (no detector/predict/pytest; no model at all) |
| Training-domain source | `artifacts/task6m_yolo_native/data.yaml` → train 10044 · val 3618 · test 3726 files; processed **17388** readable images (0 unreadable) |
| A2 identity | 1024×1024 RGB · 1 677 040 B · sha256 `10286b1e…` (matches frozen) |
| A2 Y_mean / Y_std / Y_dynamic_98 / gradient_mean | 65.6886 / 6.3262 / 31.2880 / 4.1943 |
| A2 percentile ranks (whole) | Y_mean 25.18 · Y_std 7.83 · Y_dynamic_98 7.91 · gradient_mean 58.63 |
| A2 percentile ranks (mean of 9 tiles) | Y_mean 25.09 · Y_std 7.83 · Y_dynamic_98 7.90 · gradient_mean 63.29 |
| Successful controls | A1 Y_std pct 86.31 · A3 67.02 · A4 53.07 (all inside the training distribution) |
| Exact-match search | `A2_NO_EXACT_MATCH` (no hash match for A1/A2/A3/A4) |
| PNG/encoding audit | 8-bit truecolour RGB, no interlace, IHDR+IDAT+IEND only, no sRGB/gAMA/iCCP/tEXt/pHYs chunks → no anomaly |
| Outcome | **PROP01_A2_NOT_PHOTOMETRIC_OUTLIER** (4/4 core metrics ≥ p05; no encoding anomaly) |
| Recommendation (not executed) | `NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION` |
| A2 provenance/membership | NOT ESTABLISHED |
| Image modification / rescue | NONE (no CLAHE/gamma/equalisation, no rescue design) |
| PROP-01 | open, not fixed; REF-01 / MASK-01 / Task 8B.4 / 8C not entered |
| Report | `docs/task8b3_p1d8_a2_input_domain_audit.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommended gate without a new task book. |

Watt was not needed for Task 8B.3-P1D8 (no downloads, no transfers).

Statistics were computed read-only from decoded pixels; the machine-readable dump lives outside the tracked tree
(`logs/task8b3_p1d8_domain_audit.json`) and no dataset-sized artefact was committed. `RC1-DEMO-MEM-01` remains CLOSED.
