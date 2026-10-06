# CURRENT_TASK — PROP01_A2_CHANNEL_CONTRACT_CAUSAL_TEST_V1

## Metadata

Task ID:
PROP01_A2_CHANNEL_CONTRACT_CAUSAL_TEST_V1

Status:
AUTHORIZED

Decision owner:
ChatGPT Supervisor

Authorized executor:
CODEX

Required starting branch:
audit/task8b3-prop01-a2-zero-proposals-forensics-v1

Required starting HEAD:
59296d25195a656e6a75e303e3cbb43fa9df8b8b

Task branch:
audit/task8b3-prop01-a2-channel-contract-causal-test-v1


## 1. Accepted starting facts

Supervisor accepts the previous forensic milestone.

Established:

- exact A2 provenance is known;
- frozen baseline A2 raw proposals = 0;
- all nine Ultralytics results are already zero;
- no downstream BuildReasonSeg stage drops A2 proposals;
- A1 positive control produces raw 133 / merged 52;
- BuildReasonSeg supplies RGB NumPy arrays to model.predict;
- installed Ultralytics interprets NumPy color input as BGR;
- resulting A2 detector tensors have R/B channels reversed.

Not established:

- whether this channel-contract defect causes A2 zero proposals.

PROP-01 remains OPEN.


## 2. Goal

Perform one controlled causal experiment.

Compare:

BASELINE:
BuildReasonSeg existing RGB NumPy input
→ current Ultralytics preprocessing

COUNTERFACTUAL:
same exact RGB tile
→ convert only API-facing NumPy representation to BGR
→ unchanged Ultralytics preprocessing
→ resulting network tensor must equal intended RGB tensor

Everything except channel representation must remain frozen.

Question:

Does correcting the RGB/BGR API contract change A2 from zero raw detector proposals to non-zero proposals?


## 3. Allowed repository writes

Only:

handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml

docs/task8b3_prop01_a2_channel_contract_causal_test_v1.md
evaluation/task8b3_prop01_a2_channel_contract_causal_test_v1.json

scripts/diagnose_prop01_a2_channel_contract_causal_test.py


## 4. Forbidden

Do NOT modify product runtime source.

Do NOT modify:

delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/**
tests/**
configs/**
model/**
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
external RC1

Do NOT perform:

confidence sweep
threshold sweep
imgsz change
max_det change
NMS change
TTA
model replacement
weight change
ranking change
architecture change
product repair
external sync


## 5. Exact experiment

Use exact locked A2:

inference/input/A2.png

SHA256:

10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f

Use frozen detector:

YOLO26m-seg

Frozen parameters:

tile = 512
overlap = 128
stride = 384
imgsz = 640
conf = 0.05
max_det = 300
duplicate_iou = 0.50

For each of the 9 exact A2 tiles run a paired comparison.

BASELINE:

model.predict(source=RGB_tile, ...frozen args...)

COUNTERFACTUAL:

BGR_api_tile =
np.ascontiguousarray(RGB_tile[..., ::-1])

model.predict(source=BGR_api_tile, ...same frozen args...)

No other difference is allowed.


## 6. Tensor assertion

Observe Ultralytics preprocessing.

For baseline confirm:

network tensor
= source RGB with R/B reversed

For counterfactual confirm:

network tensor
= intended source RGB

If this assertion fails:

STOP

classification =
INPUT_CONTRACT_COUNTERFACTUAL_INVALID


## 7. Baseline guard

The paired baseline must reproduce:

total raw boxes = 0

If baseline becomes non-zero:

STOP

classification =
BASELINE_NOT_REPRODUCIBLE

Do not interpret the counterfactual.


## 8. Required per-tile evidence

For every tile record:

tile_id
top
left
tile RGB SHA256

baseline:
  network tensor SHA256
  raw box count
  masks present
  confidences

corrected:
  network tensor SHA256
  raw box count
  masks present
  confidences

All parameters and model identity must be identical between the pair.


## 9. Required classification

Use one:

C1_CHANNEL_CORRECTION_RECOVERS_PROPOSALS

Meaning:
baseline total raw = 0
corrected total raw > 0

Allowed conclusion:
channel-contract defect has experimentally established causal contribution to A2 zero-proposal failure under the frozen setup.

Do NOT claim it is the sole scientific root cause or that full pipeline correctness is established.


C2_CHANNEL_CORRECTION_STILL_ZERO

Meaning:
baseline total raw = 0
corrected total raw = 0

Allowed conclusion:
channel-contract defect is real, but correcting it is not sufficient to recover A2 proposals.

The valid corrected A2 detector path still returns zero proposals.

Do NOT tune threshold.


C3_BASELINE_NOT_REPRODUCIBLE

STOP.


C4_COUNTERFACTUAL_INVALID

STOP.


## 10. No repair

Even if C1 is obtained:

DO NOT edit detector.py.
DO NOT sync external RC1.
DO NOT run full pipeline as a repaired product.

Record the repair candidate only as:

API-compatible RGB→BGR conversion before passing NumPy tile to Ultralytics.

Actual product repair requires a new Supervisor task.


## 11. External integrity

External complete RC1 is read-only.

Before and after experiment verify relevant protected artifacts remain unchanged.

No image, log, cache, config or output may be written into external RC1.


## 12. Evidence

Create:

docs/task8b3_prop01_a2_channel_contract_causal_test_v1.md

evaluation/task8b3_prop01_a2_channel_contract_causal_test_v1.json

JSON must include:

task_id
status

starting_branch
starting_head
task_branch

A2 identity
detector identity
frozen parameters

baseline_total_raw
corrected_total_raw

per_tile paired results

baseline_tensor_contract
corrected_tensor_contract

primary_classification
causal_conclusion

threshold_sweep_run = false
parameter_tuning_run = false
product_source_changed = false
external_write = false

next_gate =
CHATGPT_PROP01_CHANNEL_CAUSAL_TEST_REMOTE_AUDIT


## 13. Git/checkpoint

Git safety rules from AGENTS.md remain active.

No reset --hard
No rebase
No amend
No stash
No clean
No force push

At completion:

CURRENT_TASK Status =
READY_FOR_SUPERVISOR_AUDIT

EXECUTOR_STATE Status =
READY_FOR_SUPERVISOR_AUDIT

Commit and push task branch.

Then STOP.


## Success

PROP01_A2_CHANNEL_CONTRACT_CAUSAL_TEST_V1
= READY_FOR_SUPERVISOR_AUDIT

No product repair is authorized.