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

# FROM_DSH — Task 8B.3-P1D1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D1` |
| Status | **COMPLETE** (read-only A2 zero-proposal forensics) |
| Base | `main` @ `57b368d5647e842d8f31d6d1a9997bf1df3cc0fb` |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| A2 historical evidence | recovered: `raw=0`, `merged=0`, `tile_count=9`, `E401`, transcript contains exactly **1** `WARNING NMS time limit 2.050s exceeded` |
| A2 input identity | 1024×1024 RGB, sha256 `10286b1e…` = frozen hash |
| Frozen detector config | TILE_SIZE 512 / overlap 128 / stride 384 / IMGSZ 640 / CONF 0.05 / MAX_DET 300 / DUPLICATE_IOU 0.50 |
| Wrapper drop path | none — every returned mask is appended; empty only when ultralytics reports zero boxes |
| Ultralytics NMS semantics | `output[xi] = x[i]` before the time check; `break` leaves only the batch loop → with batch size 1 the tile's detections are returned (warning is a ~50 ms-budget timing symptom) |
| Cross-case | A1 133/52 · **A2 0/0** · A3 7/6 · A4 216/77 (identical frozen settings) |
| Primary conclusion | **PROP01_MODEL_ZERO_DETECTION_SUSPECT** |
| Recommended next gate (not executed) | **CONTROLLED_A2_INSPECT_PROPOSALS_RUN** |
| pytest / check_setup / predict / model | NOT RUN |
| detector.py / tests / manifest / external delivery | NOT modified |
| PROP-01 / REF-01 / MASK-01 | UNCHANGED / UNCHANGED / UNCHANGED |
| Report | `docs/task8b3_p1d1_a2_zero_proposal_forensics.md` |
| Next action | Awaiting ChatGPT audit; do not execute the recommended gate without a new task book. |

Watt was not needed for Task 8B.3-P1D1 (no downloads, no transfers).

No inference, no test run and no functional change occurred; `RC1-DEMO-MEM-01` remains CLOSED on `main`, and Task 8B.4
/ Task 8C were not entered.
