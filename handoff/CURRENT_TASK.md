# CURRENT_TASK — DETECTOR_RGB_BGR_CONTRACT_REPAIR_V1

## 0. Metadata

Task ID:
DETECTOR_RGB_BGR_CONTRACT_REPAIR_V1

Status:
READY_FOR_SUPERVISOR_AUDIT

Decision owner:
ChatGPT Supervisor

Authorized executor:
CODEX

Required starting branch:
audit/task8b3-prop01-a2-channel-contract-causal-test-v1

Required starting HEAD:
4e3019c77984dcb27c7a10651ccc7df631abf0da

Task branch:
fix/task8b3-detector-rgb-bgr-contract-v1


## 1. Accepted Starting Facts

Supervisor accepts:

- PROP01_A2_ZERO_PROPOSALS_ROOT_CAUSE_FORENSICS_V1
- PROP01_A2_CHANNEL_CONTRACT_CAUSAL_TEST_V1

Established engineering facts:

1. BuildReasonSeg runtime image contract is RGB.
2. `tile_rgb` is an RGB NumPy array.
3. Current detector passes this NumPy array directly to Ultralytics `model.predict`.
4. Installed Ultralytics interprets NumPy color input as BGR.
5. Ultralytics preprocessing therefore flips the current RGB input and produces an R/B-reversed network tensor.
6. Converting API-facing RGB NumPy to contiguous BGR restores the intended RGB network tensor.
7. Correcting this channel contract did not recover A2 proposals:
   - baseline raw = 0
   - corrected raw = 0

Therefore:

RGB/BGR contract defect = VERIFIED

A2 zero-proposal causal root = NOT ESTABLISHED

PROP-01 = OPEN

A2 evaluation-case validity is subject to a later, separately authorized provenance / ground-truth audit.

This task MUST NOT use “A2 becomes non-zero” as a success criterion.


## 2. Goal

Repair the verified detector input contract defect.

Required product invariant after repair:

BuildReasonSeg internal image
= RGB

Ultralytics NumPy API-facing image
= contiguous BGR

Ultralytics preprocessing output
= intended RGB network tensor

This is an engineering input-contract repair only.


## 3. Frozen Parameters and Scientific Constraints

Do NOT change:

- confidence threshold
- mask threshold
- imgsz
- max_det
- NMS
- TTA
- tile size
- tile overlap
- tile stride
- duplicate IoU
- detector weights
- detector model
- reference ranking
- spatial reasoning
- architecture
- success semantics
- semantic acceptance rules

No threshold sweep.

No parameter tuning.

No locked-case rescue heuristic.

No A2-specific branch.


## 4. Intended Minimal Implementation

Preferred repair location:

Ultralytics API boundary inside detector runtime.

Conceptually:

```python
api_tile = np.ascontiguousarray(tile_rgb[..., ::-1])

model.predict(
    source=api_tile,
    imgsz=self.imgsz,
    conf=self.conf,
    max_det=self.max_det,
    verbose=False,
    retina_masks=False,
    device=...
)
```

Requirements:

- internal `tile_rgb` remains RGB;
- do not mutate `tile_rgb` in place;
- API-facing array must be contiguous;
- shape preserved;
- dtype preserved;
- all frozen detector arguments preserved.

Exact local implementation is L1 and may be chosen by the Executor if these contracts remain unchanged.


## 5. Allowed Product Changes

Allowed canonical product file:

`delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py`

Allowed tests:

only the minimum directly relevant tests under:

`delivery_src/BuildReasonSeg_Advisor_RC1/tests/`

If required by the existing canonical source-manifest contract:

the minimum necessary `source_manifest` update is allowed.

Also allowed:

- `handoff/CURRENT_TASK.md`
- `handoff/EXECUTOR_STATE.yaml`
- `docs/task8b3_detector_rgb_bgr_contract_repair_v1.md`
- `evaluation/task8b3_detector_rgb_bgr_contract_repair_v1.json`


## 6. Forbidden Changes

Do NOT modify:

- `governance/PROJECT_STATE.yaml`
- `governance/DECISIONS.md`
- ProgramHead
- SAM/SAM2
- GRF
- reference ranking
- reasoning context
- pipeline success semantics
- mask validity semantics

Do NOT modify external RC1 manually.

Do NOT opportunistically repair unrelated manifest drift or unrelated defects.


## 7. Required Contract Tests

Add or strengthen focused tests proving:

### A. API-facing channel contract

Given a known RGB tile:

BuildReasonSeg internal tile
= RGB

array passed to Ultralytics
= BGR equivalent


### B. Original input immutability

The original RGB array must remain byte-identical after detector invocation preparation.


### C. Shape / dtype preservation

API-facing conversion preserves:

- H
- W
- 3 channels
- dtype


### D. Contiguity

The array passed to Ultralytics must be C-contiguous.


### E. Frozen invocation

Verify unchanged:

- imgsz
- conf
- max_det
- verbose
- retina_masks
- device

No new inference parameter may be introduced.


### F. Existing proposal extraction contract

The channel-boundary fix must not alter downstream proposal extraction semantics beyond changed detector outputs resulting naturally from corrected pixels.


## 8. Canonical Validation

First run the minimum relevant canonical tests.

At minimum:

- new RGB/BGR contract tests;
- directly related detector/runtime tests.

Do not weaken tests to accommodate implementation.

If unrelated failures occur:

- classify them;
- persist evidence;
- do not modify unrelated product behavior.


## 9. Controlled External RC1 Sync

Only after canonical implementation and targeted tests pass:

use the repository's existing controlled RC1 synchronization mechanism.

Do NOT manually copy individual product files.

Use the existing source-manifest / sync contract.

Before sync:

record relevant protected external identities.

After sync:

verify canonical vs external source identity.

Must not alter:

- detector weights
- ProgramHead/Qwen assets
- SAM/SAM2 assets
- locked inputs
- historical logs
- unrelated runtime assets


## 10. Real Detector Regression

After controlled sync, run the frozen detector path on the existing six locked cases:

- A1
- A2
- A3
- A4
- B1
- B2

Record for each:

- raw proposal count
- merged proposal count
- reference selection where available
- runtime status where directly relevant

Compare against the previously accepted baseline.

Observed count changes are evidence, not automatically regressions.


## 11. A2 Interpretation Guard

A2 MUST NOT be used as the success criterion for this repair.

Accepted prior causal experiment already established:

correct RGB tensor
→ A2 raw proposals still = 0

Therefore if repaired product A2 remains:

raw = 0

record:

PROP01_REMAINS_OPEN

and continue normal validation.

Do NOT:

- lower conf;
- alter detector parameters;
- add A2 rescue logic.

If A2 unexpectedly becomes non-zero:

record the discrepancy exactly and continue only if no task STOP condition is triggered.

Do not claim PROP-01 closure.

Do not infer scientific correctness.

A2 ground-truth / evaluation-case validity is explicitly outside this task and will require separate Supervisor authorization.


## 12. Broader Regression

After targeted tests and external detector checks pass:

run the existing external RC1 regression suite according to repository practice.

Record:

- exact pass count
- exact fail count
- exact failing nodes if any

Do not modify unrelated tests merely to obtain green status.


## 13. STOP Conditions

STOP and persist a recoverable checkpoint if the repair causes any unexplained:

- runtime crash
- new E5xx failure
- detector model-loading issue
- gross proposal disappearance on previously proposal-positive controls
- input-contract regression
- unrelated product regression requiring L2-L4 decisions
- unsafe Git state
- external RC1 integrity violation

Do not perform a second repair automatically.


## 14. Scientific Claim Boundary

This task may establish only:

DETECTOR_RGB_BGR_INPUT_CONTRACT = CORRECTED

It does NOT establish:

- A2 is a valid evaluation case
- PROP-01 is closed
- detector scientific quality improved
- semantic target correctness
- segmentation correctness
- REF-01 closure
- architecture improvement
- final Challenge Cup performance improvement


## 15. Required Evidence

Create:

`docs/task8b3_detector_rgb_bgr_contract_repair_v1.md`

`evaluation/task8b3_detector_rgb_bgr_contract_repair_v1.json`

Evidence must include at least:

- task_id
- status
- starting_branch
- starting_head
- task_branch
- changed_product_paths
- old_channel_contract
- new_channel_contract
- targeted_test_results
- canonical_source_identity
- external_sync_result
- external_source_identity

Locked-case results must include:

- A1
- A2
- A3
- A4
- B1
- B2

Also record:

- external_regression_result
- threshold_sweep_run = false
- parameter_tuning_run = false
- model_change = false
- weight_change = false
- A2_special_handling = false
- PROP01_status = OPEN
- A2_case_validity = NOT_EVALUATED_IN_THIS_TASK

next_gate:

CHATGPT_DETECTOR_CHANNEL_REPAIR_REMOTE_AUDIT


## 16. Git Safety

Forbidden:

- git reset --hard
- git rebase
- git commit --amend
- git stash
- git clean
- force push
- history rewrite

Unknown pre-existing files or modifications:

- preserve them;
- do not stage them;
- STOP if they prevent safe execution.


## 17. Checkpoint Protocol

Use meaningful recoverable checkpoints.

Recommended:

1. canonical implementation + contract tests
2. controlled external sync
3. six-case detector regression
4. broader regression
5. final Supervisor handoff

When a stage is L0-L1 and its gate passes:

continue automatically.

Do not stop merely because a checkpoint was completed.

Only STOP for the task-defined escalation conditions.


## 18. Final State

On successful completion:

`handoff/CURRENT_TASK.md`

Status:

READY_FOR_SUPERVISOR_AUDIT

`handoff/EXECUTOR_STATE.yaml`

status:

READY_FOR_SUPERVISOR_AUDIT

Commit and push all authorized work.

Report actual remote HEAD.

Then STOP.

Do NOT automatically begin:

- deeper PROP-01 investigation
- A2 ground-truth audit
- threshold experiments
- detector replacement


## 19. Success Definition

DETECTOR_RGB_BGR_CONTRACT_REPAIR_V1
= READY_FOR_SUPERVISOR_AUDIT

Required:

- verified RGB/BGR product contract repaired;
- intended RGB reaches detector network;
- original internal RGB contract preserved;
- frozen detector parameters unchanged;
- targeted contract tests pass;
- external RC1 synchronized through controlled mechanism;
- locked-case impact recorded;
- broader regression recorded;
- no threshold tuning;
- no A2-specific rescue logic;
- PROP-01 remains OPEN;
- A2 case validity remains outside this task.

NEXT:

CHATGPT_DETECTOR_CHANNEL_REPAIR_REMOTE_AUDIT