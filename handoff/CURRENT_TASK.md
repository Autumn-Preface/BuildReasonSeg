# CURRENT_TASK — PROP01_A2_ZERO_PROPOSALS_ROOT_CAUSE_FORENSICS_V1

## 0. Metadata

```text
Task ID: PROP01_A2_ZERO_PROPOSALS_ROOT_CAUSE_FORENSICS_V1
Status: AUTHORIZED

Decision owner: ChatGPT Supervisor
Authorized executor: CODEX

Required starting branch:
docs/governance-v1-1-idle-state-semantics

Required starting HEAD:
504268128b7a9b058580b7702c9429b27705c053

Task branch:
audit/task8b3-prop01-a2-zero-proposals-forensics-v1
```

## 1. Supervisor Disposition

PROP-01 remains:

```text
OPEN_ENGINEERING_DEFECT
```

Observed locked symptom:

```text
case = A2
program = left_of
runtime = FAILED E401
raw proposals = 0
merged proposals = 0
```

The cause is NOT established.

Prior investigation did not justify blind threshold tuning.

Frozen detector policy:

```text
TILE_SIZE = 512
OVERLAP = 128
STRIDE = 384
IMGSZ = 640
CONF = 0.05
MAX_DET = 300
FROZEN_THRESHOLD = 0.5
DUPLICATE_IOU = 0.50
```

This milestone is forensic only.

Do not repair PROP-01.

## 2. Goal

Determine, with reproducible evidence, the earliest stage at which A2 becomes zero-proposal.

Required causal boundary:

```text
input/provenance
→ tiling
→ exact detector input
→ raw Ultralytics result
→ runtime detect_tile extraction
→ global accumulation
→ compact-mask filtering
→ duplicate merge
→ final merged proposals
```

The milestone must answer:

> **Does A2 already produce zero raw YOLO detections under the frozen detector policy, or are detections lost inside BuildReasonSeg preprocessing / adapter / accumulation / merge logic?**

Do not go beyond the evidence.

## 3. Allowed Scope

Read-only inspection is allowed across the repository.

Repository writes are limited to:

```text
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml

docs/task8b3_prop01_a2_zero_proposals_forensics_v1.md
evaluation/task8b3_prop01_a2_zero_proposals_forensics_v1.json

scripts/diagnose_prop01_a2_zero_proposals.py
```

The diagnostic script may be created only if useful for reproducible evidence.

No product runtime source may be modified.

## 4. External / Model Access

Read-only use of the complete external RC1 is allowed:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Allowed:

```text
read existing model weights
read existing locked-case inputs / diagnostics / logs
run detector inference
run read-only diagnostics
```

Forbidden:

```text
modify external RC1
sync external RC1
download assets
replace weights
edit logs
edit existing locked-case artifacts
```

Diagnostic outputs must be written only to the authorized repository evidence/report paths or temporary disposable locations.

## 5. Frozen Scientific / Engineering Constraints

Do NOT change or sweep:

```text
confidence threshold
mask threshold
imgsz
tile size
tile overlap / stride
max_det
duplicate IoU
model weights
model selection
NMS policy
TTA
reference ranking
architecture
```

Do NOT run:

```text
conf sweeps
threshold sweeps
alternative imgsz experiments
alternative detector models
TTA
post-hoc rescue heuristics
```

This milestone is not a tuning task.

## 6. Required Startup

Read in order:

```text
AGENTS.md
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
```

Then verify actual Git:

```text
branch
HEAD
working tree
```

Required start:

```text
branch =
docs/governance-v1-1-idle-state-semantics

HEAD =
504268128b7a9b058580b7702c9429b27705c053

working tree = clean
```

Mismatch:

```text
STOP
record actual state
do not reset/rebase/stash/clean
```

## 7. Phase A — Recover Exact A2 Provenance

Locate the exact locked A2 artifact used for the recorded E401 result.

Record at minimum:

```text
image path / provenance
image filename
image dimensions
file size
SHA-256
program
expected case identifier
historical raw proposal count
historical merged proposal count
```

Do not silently substitute another image.

If exact A2 cannot be established:

```text
STOP
classification = A2_PROVENANCE_NOT_ESTABLISHED
```

Do not continue with a guessed sample.

## 8. Phase B — Reproduce Frozen A2 Result

Using the existing frozen detector and exact A2:

run the current detector path without changing parameters.

Record:

```text
tile count
raw_proposal_count
merged_proposal_count
detector parameters
model path
device
runtime/library versions where relevant
```

Required comparison with historical symptom:

```text
raw = 0
merged = 0
```

If current result differs, do not “fix” it.

Classify:

```text
NON_REPRODUCIBLE_BASELINE
```

and investigate environment/model/input identity only.

## 9. Phase C — Per-Tile Forensics

For every A2 tile, record:

```text
tile id
top / left
shape
padding
tile SHA-256 or deterministic content hash

raw Ultralytics boxes count
raw Ultralytics masks present?
raw confidences
runtime detect_tile proposal count
```

Use exactly the frozen detector invocation.

Important:

`raw Ultralytics boxes count` means the detector library's result before BuildReasonSeg proposal extraction / global merge.

Do not lower `conf=0.05` to “see what is underneath”.

## 10. Phase D — Stage-Loss Localization

Establish counts through the pipeline:

```text
Ultralytics result
↓
detect_tile output
↓
detect_global accumulated entries
↓
compact masks
↓
merge input
↓
merged proposals
```

If count changes at any transition, identify:

```text
exact function
exact condition
number dropped
reason
```

No repair.

## 11. Phase E — Positive-Control Sanity Check

If the exact historical positive-control artifact can be established without guessing, run **one** known proposal-positive locked case under the exact same detector/model/environment.

Prefer an existing case whose prior evidence recorded non-zero raw proposals.

Purpose only:

```text
verify model/runtime is capable of producing proposals in the same environment
```

Do not use the control for threshold fitting or comparative tuning.

If exact positive-control provenance cannot be established, record:

```text
POSITIVE_CONTROL_NOT_AVAILABLE
```

and continue; this alone is not a failure.

## 12. Required Root-Cause Classification

Use exactly one primary classification if evidence permits:

```text
P0_PROVENANCE_FAILURE
exact A2 input cannot be established

P1_INPUT_OR_PREPROCESSING_DEFECT
wrong/corrupted/transformed pixels reach detector

P2_RAW_DETECTOR_ZERO
valid exact A2 reaches frozen YOLO detector,
but Ultralytics returns zero detections at frozen policy

P3_RUNTIME_ADAPTER_LOSS
Ultralytics returns detections,
but detect_tile extraction loses them

P4_GLOBAL_ACCUMULATION_LOSS
detect_tile returns proposals,
but detect_global accumulation/compact handling removes them

P5_MERGE_LOSS
valid accumulated proposals exist,
but merge results in zero

P6_MODEL_OR_ENVIRONMENT_MISMATCH
weights/runtime/input identity differ from locked baseline

P7_CAUSE_NOT_ESTABLISHED
evidence insufficient or contradictory
```

Do not invent another classification unless necessary; if needed, STOP for Supervisor review.

## 13. Scientific Claim Boundary

Even if `P2_RAW_DETECTOR_ZERO` is established, do NOT automatically conclude:

```text
YOLO26m is bad
domain gap is scientifically proven
threshold should be lowered
model should be replaced
```

The allowed conclusion is only the narrow observed causal fact.

Likewise, finding a code-stage loss does not authorize repair.

## 14. Checkpoints

This is a longer Codex task.

Use recoverable checkpoints after meaningful stages, especially after:

```text
A2 provenance established
baseline reproduced
per-tile raw detector evidence captured
root-cause classification established
```

At checkpoints:

```text
update EXECUTOR_STATE
commit + push when safe and useful
```

Do not leave the only copy of important evidence in the Codex conversation.

Multiple commits on the task branch are allowed if they represent meaningful recoverable checkpoints.

Do not amend/rewrite them.

## 15. EXECUTOR_STATE During Work

Maintain:

```text
task_id = PROP01_A2_ZERO_PROPOSALS_ROOT_CAUSE_FORENSICS_V1
status = IN_PROGRESS / STOPPED / READY_FOR_SUPERVISOR_AUDIT
executor = CODEX

completed
current_step
next_action
hypotheses
tests_run
files_modified
blocked_on_supervisor
block_reason
decision_level_required
```

Actual Git remains authoritative.

## 16. Validation

This task does not require the full pytest suite.

Allowed:

```text
read-only detector inference
diagnostic script
targeted existing detector/runtime tests if directly informative
Python syntax/import check for the diagnostic script
JSON parse validation
```

Do not run unrelated full-suite regression merely for ceremony.

## 17. Required Evidence

Create:

```text
docs/task8b3_prop01_a2_zero_proposals_forensics_v1.md

evaluation/task8b3_prop01_a2_zero_proposals_forensics_v1.json
```

JSON must record at least:

```text
task_id
status

starting_branch
starting_head
task_branch

A2 provenance
A2 SHA-256
image dimensions

detector model identity/path
frozen parameters

tile count

per_tile:
  tile id
  tile position
  raw ultralytics count
  runtime detect_tile count

pipeline_counts:
  raw_ultralytics
  detect_tile
  accumulated
  compact_valid
  merge_input
  merged

positive_control status/result

primary_root_cause_classification

root_cause_summary

threshold_sweep_run = false
parameter_tuning_run = false
product_source_changed = false
external_write = false

next_gate = CHATGPT_PROP01_FORENSICS_V1_REMOTE_AUDIT
```

## 18. Forbidden Product Changes

Absolutely no modifications under:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/**
buildreasonseg/**
tests/**
configs/**
model/**
```

No product fix is authorized.

If you identify a likely fix:

```text
record it as a hypothesis only
do not implement it
```

## 19. Git Safety

Forbidden:

```text
reset --hard
rebase
commit --amend
stash
clean
force push
history rewrite
```

Unknown changes:

```text
preserve
do not stage
STOP if they prevent safe work
```

## 20. Final State

If investigation completes:

```text
CURRENT_TASK Status = READY_FOR_SUPERVISOR_AUDIT
EXECUTOR_STATE Status = READY_FOR_SUPERVISOR_AUDIT
```

Do not change project defect status from OPEN.

Commit/push all authorized evidence.

Final response must report:

```text
task branch
actual remote HEAD
number of commits created
primary classification
one-sentence causal conclusion
evidence paths
whether any product code changed
whether any tuning/sweep occurred
```

Then STOP.

Do not implement a repair.

## 21. Success Definition

```text
PROP01_A2_ZERO_PROPOSALS_ROOT_CAUSE_FORENSICS_V1
= READY_FOR_SUPERVISOR_AUDIT

A2 exact provenance established
frozen baseline reproduced or discrepancy explained
zero-proposal stage localized
primary classification evidence-backed
no threshold tuning
no product repair
reproducible evidence persisted

NEXT =
CHATGPT_PROP01_FORENSICS_V1_REMOTE_AUDIT
```