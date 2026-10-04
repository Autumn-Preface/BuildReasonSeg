# Task 8B.3-P1D12-R2 — Frozen Threshold and Inspect-Only Timing Evidence (read-only)

## 1. Task and scope

Read-only completion of two evidence items for the P1D12 locked-candidate gate: the canonical `FROZEN_THRESHOLD`
constant and the exact sam2 / relation-fields / D-B1 timing evidence in the four existing `result.json` files. No
candidate, `predict.py`, detector or model was re-run.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 728932094a9f86f7c1607abc6a2beff12d2d2d70
model / detector / candidate executions = NONE
files written under the external delivery = NONE
```

## 2. `FROZEN_THRESHOLD` (canonical source, read-only)

```text
source = delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
value  = 0.5  # ultralytics mask binarisation for the frozen U-C1 detector
```

Definition and use in the canonical detector source:

```text
35: FROZEN_THRESHOLD = 0.5  # ultralytics mask binarisation for the frozen U-C1 detector
233: mask = masks[index] > FROZEN_THRESHOLD
468: __all__ = ["CONF", "DUPLICATE_IOU", "DetectorRuntime", "FROZEN_THRESHOLD", "GlobalProposal", "IMGSZ",
```

## 3. Exact timing evidence in the existing diagnostics

| relation | tile | timing entries present in result.json |
|---|---|---|
| right | 1010 | `timings.sam2`=0.0 · `timings.relation_fields`=0.0 · `timings.db1`=0.0 |
| left | 1003 | `timings.sam2`=0.0 · `timings.relation_fields`=0.0 · `timings.db1`=0.0 |
| above | 1008 | `timings.sam2`=0.0 · `timings.relation_fields`=0.0 · `timings.db1`=0.0 |
| below | 1009 | `timings.sam2`=0.0 · `timings.relation_fields`=0.0 · `timings.db1`=0.0 |

Complete `result.json` key sets recorded for traceability:

| relation | tile | keys |
|---|---|---|
| right | 1010 | `device`, `input_channels`, `input_dtype`, `input_mode`, `input_path`, `input_size`, `merged_proposal_count`, `mode`, `model_package`, `notes`, `output_paths`, `overlap`, `proposals`, `raw_proposal_count`, `status`, `tile_count`, `tile_size`, `timings` |
| left | 1003 | `device`, `input_channels`, `input_dtype`, `input_mode`, `input_path`, `input_size`, `merged_proposal_count`, `mode`, `model_package`, `notes`, `output_paths`, `overlap`, `proposals`, `raw_proposal_count`, `status`, `tile_count`, `tile_size`, `timings` |
| above | 1008 | `device`, `input_channels`, `input_dtype`, `input_mode`, `input_path`, `input_size`, `merged_proposal_count`, `mode`, `model_package`, `notes`, `output_paths`, `overlap`, `proposals`, `raw_proposal_count`, `status`, `tile_count`, `tile_size`, `timings` |
| below | 1009 | `device`, `input_channels`, `input_dtype`, `input_mode`, `input_path`, `input_size`, `merged_proposal_count`, `mode`, `model_package`, `notes`, `output_paths`, `overlap`, `proposals`, `raw_proposal_count`, `status`, `tile_count`, `tile_size`, `timings` |

## 4. Interpretation

```text
sam2 / relation / db1 timing entries found = {"sam2": false, "relation": false, "db1": false, "d_b1": true}
```

Each of the four frozen `result.json` files contains explicit zero-valued timing entries — `timings.sam2 = 0.0`,
`timings.relation_fields = 0.0`, `timings.db1 = 0.0` — recorded by the `--inspect-proposals` result contract. This is
direct measured evidence that none of those stages executed, not merely an absence of keys.

Correction note: an earlier draft of this section claimed those entries were absent. That claim was wrong; it came from
checking only top-level `result.json` keys, whereas the chain timings live inside the nested `timings` object. The
top-level key set of every candidate is `device`, `input_channels`, `input_dtype`, `input_mode`, `input_path`,
`input_size`, `merged_proposal_count`, `mode`, `model_package`, `notes`, `output_paths`, `overlap`, `proposals`,
`raw_proposal_count`, `status`, `tile_count`, `tile_size`, `timings`.

## 5. Explicit confirmations

```text
candidate / predict / detector / model re-runs = NONE
candidate replacement / GT access / visual judgement = NONE / NONE / NONE
external delivery / canonical RC1 modified = NO / NO
locked candidate identities = UNCHANGED
```

## 6. Outcome

```text
PROP01_LOCKED_DEMO_PROPOSAL_GATE_EVIDENCE_CLOSED (retained — no contradiction found)
NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS
```

The next gate is not executed here.
