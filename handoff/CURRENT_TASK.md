# CURRENT_TASK — TASK8B4_RUN_ISOLATED_OUTPUT_LAYOUT_V1

## 0. Metadata

Task ID:

TASK8B4_RUN_ISOLATED_OUTPUT_LAYOUT_V1

Status:

IN_PROGRESS

Decision owner:

ChatGPT Supervisor

Authorized executor:

CODEX

Accepted predecessor task:

A2_GROUND_TRUTH_AND_LOCKED_CASE_VALIDITY_AUDIT_V1

Accepted predecessor branch:

audit/task8b3-a2-ground-truth-case-validity-v1

Accepted predecessor remote HEAD:

c6d09e77d3c403771a92d096fe5977ebaa61db67

Required task branch:

fix/task8b4-run-isolated-output-layout-v1

Project phase:

RC1_ENGINEERING_RELIABILITY

Next gate:

CHATGPT_TASK8B4_OUTPUT_LAYOUT_REMOTE_AUDIT


## 1. Supervisor Disposition

The previous A2 provenance / GT case-validity milestone is ACCEPTED.

Accepted state:

- RGB/BGR detector input contract is corrected.
- A2 still returns zero proposals.
- A2 exact source provenance remains NOT_ESTABLISHED.
- A2 GT validity remains NOT_EVALUABLE_WITHOUT_PROVENANCE.
- PROP-01 remains OPEN.
- No further A2 forensic work is authorized by this task.
- MEM-01 remains CLOSED.
- MASK-01 engineering hardening remains CLOSED.
- REF-01 residual limitation remains unchanged.
- Task 8B.4 implementation is PARTIAL_ACCEPT / CONTINUE_AUTHORIZED under the Supervisor continuation disposition in section 29.

This milestone is purely a delivery-output isolation repair.

It MUST NOT modify scientific architecture, detector behaviour, relation semantics, target selection, segmentation behaviour, thresholds or acceptance semantics.


## 2. Historical Task 8B.4 Requirement

Task 8B.3 repeatedly recorded the following accepted/deferred requirement:

```text
Each predict run should own one independent directory under inference/output,
with subdirectories diagnostics/, masks/, overlays/.
```

Status was repeatedly:

```text
ACCEPTED / DEFERRED TO TASK 8B.4
```

This task implements that exact deferred requirement.


## 3. Current Defect

Current canonical RC1 implementation uses shared directories:

```text
inference/output/masks/
inference/output/overlays/
inference/output/diagnostics/
```

Current allocation in:

```text
buildreasonseg/runtime/outputs.py
```

uses the existence of:

```text
<stem>_mask.png
```

inside the shared mask directory to choose the run suffix.

Therefore a failed run that creates:

```text
diagnostics/<stem>/
```

but no mask does NOT reserve the suffix.

A later run of the same image may reuse the same diagnostics location and overwrite previous forensic evidence.

This is a delivery engineering defect.

It is NOT a model-quality defect.


## 4. Frozen New Output Contract

### 4.1 Run directory

Every output-owning image run receives exactly one independent directory under:

```text
inference/output/
```

Run slug:

```text
<stem>
<stem>_001
<stem>_002
...
```

The first available run-directory name is allocated.

Allocation MUST be based on existence of the run root itself.

It MUST NOT depend on mask existence, overlay existence or diagnostic-file existence.


### 4.2 Atomic reservation

Preferred implementation:

attempt to create the candidate run directory with filesystem exclusive/non-overwriting semantics.

If it already exists:

try the next suffix.

This avoids two runs choosing the same directory merely because no mask file exists yet.

Do not introduce locking infrastructure beyond what is necessary for directory reservation.


### 4.3 Required structure

Every allocated run root contains:

```text
<run_slug>/
├─ diagnostics/
├─ masks/
└─ overlays/
```

Example first run:

```text
inference/output/A1/
├─ diagnostics/
├─ masks/
│  └─ A1_mask.png
└─ overlays/
   └─ A1_overlay.png
```

Example second run:

```text
inference/output/A1_001/
├─ diagnostics/
├─ masks/
│  └─ A1_mask_001.png
└─ overlays/
   └─ A1_overlay_001.png
```


### 4.4 Filename compatibility

Preserve the existing filename suffix convention.

For run index 0:

```text
mask     = <stem>_mask.png
overlay  = <stem>_overlay.png
```

For run index N:

```text
mask     = <stem>_mask_<NNN>.png
overlay  = <stem>_overlay_<NNN>.png
```

The suffix now appears both in the run slug and in mask/overlay filenames.

This preserves existing artifact filename semantics while adding run-level isolation.


### 4.5 Diagnostics filenames

Existing diagnostics names remain unchanged:

```text
prompt.txt
parsed_program.json
global_proposals.png
proposals.json
selected_reference.png
reasoning_context.png
reference_context_mask.png
direction_field.png
nearest_field.png
relation_weight.png
prototype_similarity.png
maps.npz
result.json
```

They now live directly inside:

```text
<run_slug>/diagnostics/
```

There must NOT be a second nested `<stem>` directory.


## 5. Output Path API Compatibility

Existing payload keys MUST remain:

```text
output_paths.mask
output_paths.overlay
output_paths.diagnostics
```

Existing inspect-mode fields remain semantically equivalent.

CLI labels remain:

```text
Mask         :
Overlay      :
Diagnostics  :
```

Only filesystem path values change.

Do NOT introduce a new required CLI argument.

Do NOT change exit codes.

Do NOT change result status semantics.

Do NOT change SUCCESS semantics.


## 6. Allocation Timing

Preserve current error-order behaviour as much as possible.

For full prediction:

```text
load_image
→ allocate run output directory
→ remaining pipeline
```

is already the effective order and should remain so.

For inspect-proposals mode, move output allocation immediately after successful image loading if necessary so that the image run owns its run directory before detector execution.

Pre-image failures such as:

- nonexistent input;
- unsupported input path;
- decode failure before successful image load;

are NOT required to create an output run directory.

Do not reorder model/package validation or other unrelated error contracts merely to create output folders.


## 7. Failure Contract

A failed run after output allocation still permanently owns its run directory.

Example:

```text
output/A2/
├─ diagnostics/
│  └─ result.json
├─ masks/
└─ overlays/
```

If A2 is run again:

```text
output/A2_001/
```

MUST be allocated even though the first run produced no mask.

No previous diagnostics may be silently overwritten.


## 8. --no-save-diagnostics Contract

`--no-save-diagnostics` continues to suppress diagnostic files.

The run directory contract still applies.

It is acceptable for:

```text
diagnostics/
```

to exist but remain empty.

Do NOT reinterpret an empty directory as a saved diagnostic artifact.


## 9. Inspect-Proposals Contract

`--inspect-proposals` uses the same run-isolated directory allocator.

Expected shape:

```text
output/<run_slug>/
├─ diagnostics/
│  ├─ global_proposals.png
│  ├─ proposals.json
│  ├─ result.json
│  └─ ...
├─ masks/
└─ overlays/
```

Inspect mode MUST NOT create a fake mask or overlay.


## 10. Legacy Output Preservation

Existing historical layout may already contain:

```text
inference/output/masks/
inference/output/overlays/
inference/output/diagnostics/
review_task8b3/
...
```

Do NOT:

- delete them;
- migrate them;
- rename them;
- clean them;
- overwrite them for compatibility;
- reinterpret them as new run directories.

They are historical evidence.

New allocator uses only the new run-root contract.

If an image stem happens to collide with an existing root entry such as:

```text
masks
overlays
diagnostics
```

that candidate is already occupied and the allocator must choose the next suffix.


## 11. Allowed Functional Scope

Preferred changed delivery files:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/outputs.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Conditionally allowed only if mechanically required for the frozen path contract:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
```

Avoid modifying `predict.py` unless a concrete failing contract test proves it is necessary.

Repository evidence/handoff files allowed:

```text
docs/task8b4_run_isolated_output_layout_v1.md
evaluation/task8b4_run_isolated_output_layout_v1.json
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
```

No other path is authorized without Supervisor escalation.


## 12. Forbidden Scope

Do NOT change:

- detector implementation except already accepted RGB/BGR state;
- detector threshold;
- NMS;
- tile size;
- overlap;
- imgsz;
- max_det;
- ProgramHead;
- Qwen;
- language fallback;
- reference selection;
- relation semantics;
- SAM/SAM2;
- D-B1;
- GRF;
- mask validity rules;
- SUCCESS semantics;
- model weights;
- configs;
- scientific metrics;
- locked Demo candidate identities;
- PROP-01 classification;
- REF-01 classification;
- MASK-01 classification;
- governance/PROJECT_STATE.yaml;
- governance/DECISIONS.md.

Do NOT run real model inference in this milestone.

Do NOT run the historical six-image suite.

Do NOT run A2.

Do NOT run the four locked Demo candidates.


## 13. Required Preflight

Read in order:

1. AGENTS.md
2. governance/PROJECT_STATE.yaml
3. governance/DECISIONS.md
4. handoff/CURRENT_TASK.md
5. handoff/EXECUTOR_STATE.yaml

Then:

```text
git fetch
```

Verify actual remote predecessor branch:

```text
audit/task8b3-a2-ground-truth-case-validity-v1
```

Required accepted predecessor HEAD:

```text
c6d09e77d3c403771a92d096fe5977ebaa61db67
```

If remote advanced:

do NOT reset/rebase/amend.

Inspect the additional commits.

If they are not clearly Supervisor authorization-only changes for this milestone:

persist evidence and STOP.

If correct:

create:

```text
fix/task8b4-run-isolated-output-layout-v1
```

from the verified authorization point.


## 14. Required Pre-Implementation Inspection

Before editing, record:

- current `outputs.py` SHA256 / Git blob;
- all references to:
  - `allocate_outputs`
  - `allocate_run_suffix`
  - `suffix_of`
  - `SampleOutputs`
  - `output_dirs`
- current README output contract;
- current `inference/README.md` output tree;
- current relevant tests;
- current source-manifest entries for every manifest-listed file expected to change.

Also inspect existing external:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

read-only before implementation.

Record inventories for:

```text
inference/input/
inference/output/
```

and protected model assets.

Do not modify external delivery during this phase.


## 15. Required Unit Contract Tests

At minimum add/modify tests proving all of the following.

### T1 — first run layout

For `example.png`, first allocation creates:

```text
output/example/
output/example/diagnostics/
output/example/masks/
output/example/overlays/
```

and paths resolve to:

```text
output/example/masks/example_mask.png
output/example/overlays/example_overlay.png
output/example/diagnostics/
```


### T2 — second run isolation

A second allocation for the same stem creates:

```text
output/example_001/
```

with filenames:

```text
example_mask_001.png
example_overlay_001.png
```


### T3 — failed diagnostics-only run reserves suffix

First allocation:

```text
output/example/
```

Write diagnostics only.

Write NO mask.

Second allocation MUST be:

```text
output/example_001/
```

and MUST NOT reuse:

```text
output/example/diagnostics/
```


### T4 — existing run directory is never reused

Pre-create:

```text
output/example/
output/example_001/
```

Next allocation must use:

```text
output/example_002/
```


### T5 — legacy shared outputs preserved

Pre-create historical:

```text
output/masks/example_mask.png
output/overlays/example_overlay.png
output/diagnostics/example/result.json
```

Allocate a new run.

Assertions:

- historical files remain byte-identical;
- new run uses the new run-root layout;
- no legacy file is deleted or overwritten.


### T6 — mask/overlay functional contract preserved

Existing assertions remain true:

- original image dimensions;
- mask uint8;
- values subset `{0,255}`;
- overlay original size;
- configured alpha behaviour unchanged.


### T7 — diagnostics required fields preserved

Existing diagnostics-file test remains valid under the new run-local diagnostics directory.


### T8 — same run root

Assert:

```text
mask
overlay
diagnostics
```

belong to the same run root.


### T9 — collision with reserved/legacy root name

If:

```text
output/masks/
```

already exists and input stem is `masks`, allocator must not treat that legacy directory as a new run.

It must allocate a suffixed run slug.


### T10 — no-save-diagnostics semantics

No diagnostic files are produced when disabled.

Do not weaken existing CLI contract.


## 16. Canonical Test Gates

Use the established project Python environment.

From canonical RC1 root, first run targeted tests:

```text
python -m pytest tests/test_task8b_runtime.py -q
python -m pytest tests/test_cli_contract.py -q
```

If `test_cli_contract.py` is unchanged, it must still be run.

Supervisor L2 continuation disposition under frozen GOV-D007:

```text
canonical_full_suite_disposition = WAIVED_AS_INVALID_BY_SUPERVISOR_UNDER_GOV_D007
```

The previously executed canonical full suite remains recorded as 127 passed / 17 failed / 6 errors, exit 1. It is an invalid complete-delivery gate in the lightweight canonical source/config snapshot, not a Task 8B.4 product regression. Do not rerun it to manufacture green acceptance, add excluded VERSION/assets/weights/logs/fixtures/inference/input, or weaken its tests.

The complete full-suite acceptance gate belongs to external complete RC1 in section 20. Record exact external passed/failed count and exit code; do not assume a fixed test count.

Any unexplained regression:

STOP.


## 17. Source Manifest Contract

Manifest remains:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
entries = 135
```

Do NOT add or remove manifest paths merely because this task changes contents.

For every changed manifest-listed delivery file:

update:

```text
bytes
sha256
```

from committed Git canonical blob bytes.

Recommended sequence:

1. implement product/tests/docs;
2. run targeted tests;
3. commit checkpoint;
4. calculate Git canonical blob identity from that commit;
5. update only affected manifest entries;
6. validate all 135 entries against Git canonical blob bytes;
7. commit manifest checkpoint.

Do NOT calculate canonical manifest identities from CRLF-expanded Windows working-tree bytes.


## 18. External Sync Gate

External root:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Before any write sync:

run the existing Git-canonical helper in check mode.

Expected principle:

- mismatches must be limited to the exact manifest-listed files intentionally changed by Task 8B.4;
- no unexplained unrelated mismatch may exist.

If unrelated mismatches appear:

STOP before sync.


## 19. Controlled External Sync

Only after the pre-sync gate is clean and understood:

use the existing:

```text
scripts/sync_advisor_rc1_delivery.py
```

for one controlled source/config sync.

Then separately write the exact Git-canonical:

```text
source_manifest.json
```

to external RC1, matching the established P1D11B procedure.

Do NOT delete any external file.

Do NOT touch:

```text
model weights
Qwen assets
SAM2 assets
inference/input
existing inference/output history
logs except unavoidable test-generated logs
```

Protected asset hashes must remain unchanged.


## 20. External Post-Sync Gates

Required:

### Gate A

Git-canonical helper:

```text
checked = 135
match = 135
missing = 0
mismatch = 0
```

### Gate B

External `source_manifest.json`:

byte/hash identical to Git canonical manifest.

### Gate C

```text
python check_setup.py
```

must report READY.


### Gate D

Run external targeted tests:

```text
python -m pytest tests/test_task8b_runtime.py -q
python -m pytest tests/test_cli_contract.py -q
```

### Gate E

Run external full suite:

```text
python -m pytest tests/ -q
```

All must exit 0.

No model inference is authorized.


## 21. External Historical Output Preservation Gate

Compare pre/post inventory of:

```text
inference/output/
```

Because no real inference is authorized in this milestone, existing historical output contents must remain unchanged.

No migration or cleanup is permitted.

Also verify:

```text
inference/input/
```

is unchanged.


## 22. Documentation Update

Update canonical RC1 README to document the new run-isolated layout.

Replace the old shared example with:

```text
inference/output/<run_slug>/
├─ diagnostics/
├─ masks/
└─ overlays/
```

Document:

```text
first run  -> <stem>
next runs -> <stem>_001, <stem>_002, ...
```

State explicitly:

- failed runs reserve their run directory;
- existing run outputs are never silently overwritten;
- legacy pre-8B.4 shared output directories are preserved and are not migrated.


Update:

```text
inference/README.md
```

to the same contract.

Remove the stale claim that Task 8A has no real inference, since RC1 Task 8B runtime now exists.

Do not change model-quality or scientific claims.


## 23. Required Evidence

Create:

```text
docs/task8b4_run_isolated_output_layout_v1.md
evaluation/task8b4_run_isolated_output_layout_v1.json
```

JSON must include at least:

```text
task_id
starting_branch
starting_head
task_branch

historical_requirement
old_layout
new_layout

changed_paths
canonical_commits

unit_test_results
canonical_full_suite

manifest_identity_basis
manifest_entries
manifest_changed_entries
canonical_manifest_gate

external_precheck
external_sync
external_manifest_identity
external_postcheck
external_setup
external_targeted_tests
external_full_suite

legacy_output_inventory_before
legacy_output_inventory_after
legacy_output_unchanged

input_inventory_before
input_inventory_after
input_unchanged

protected_assets_before
protected_assets_after
protected_assets_unchanged

no_model_inference
no_threshold_change
no_model_change
no_scientific_contract_change
no_historical_output_migration
```


## 24. Checkpoints

### Checkpoint 1

Implementation + targeted canonical tests green.

Commit + push.


### Checkpoint 2

Canonical targeted runtime 62 passed and CLI 24 passed + source manifest 135/135 Git-canonical validation: ACCEPTED by Supervisor L2 continuation disposition. The prior canonical full-suite execution is WAIVED_AS_INVALID_BY_SUPERVISOR_UNDER_GOV_D007 and remains preserved as 127/17/6, exit 1.

Commit + push.


### Checkpoint 3

External pre-sync check understood and limited exactly to authorized changed files.

Persist evidence before external write.


### Checkpoint 4

Controlled external sync + manifest write complete.

Post-sync:

```text
135/135
check_setup READY
targeted tests PASS
full external tests PASS
```

Commit + push evidence.


## 25. STOP Conditions

STOP with evidence if:

- required starting Git state conflicts;
- unknown workspace changes exist;
- unrelated external manifest mismatches exist;
- implementation would require changing CLI schema;
- implementation would require changing status/exit semantics;
- any model or scientific component would need modification;
- manifest canonical identity cannot be established;
- external protected assets change unexpectedly;
- legacy inference/output content changes unexpectedly;
- tests expose an unrelated regression;
- destructive Git operation appears necessary.


## 26. Git Safety

Forbidden:

```text
git reset --hard
git rebase
git commit --amend
git stash
git clean
force push
history rewrite
autonomous revert
```

Preserve every real checkpoint and failure.


## 27. Scientific Claim Boundary

Task 8B.4 may establish only:

```text
RUN_ISOLATED_OUTPUT_LAYOUT_IMPLEMENTED
```

and associated delivery integrity.

It does NOT establish:

- better detector quality;
- improved segmentation quality;
- PROP-01 closure;
- REF-01 closure;
- MASK semantic correctness;
- final Demo success;
- architecture improvement.


## 28. Completion State

On successful completion:

```text
handoff/CURRENT_TASK.md
Status:
READY_FOR_SUPERVISOR_AUDIT
```

and:

```text
handoff/EXECUTOR_STATE.yaml
status:
READY_FOR_SUPERVISOR_AUDIT
```

Next gate:

```text
CHATGPT_TASK8B4_OUTPUT_LAYOUT_REMOTE_AUDIT
```

Then STOP.

Do NOT automatically start Final Demo.

## 29. Supervisor L2 Continuation Disposition

```text
TASK8B4_RUN_ISOLATED_OUTPUT_LAYOUT_V1 = PARTIAL_ACCEPT / CONTINUE_AUTHORIZED
accepted task branch = fix/task8b4-run-isolated-output-layout-v1
accepted task remote HEAD = 7e0e8b3115c2b3660d57ffa5170c4fb5393be39c
canonical_full_suite_disposition = WAIVED_AS_INVALID_BY_SUPERVISOR_UNDER_GOV_D007
```

Authority: direct user-relayed ChatGPT Supervisor remote audit and L2 disposition. This continuation supersedes the original canonical full-suite green requirement in sections 16 and 24. GOV-D007 remains frozen and governance files are unchanged.

Accepted: implementation, canonical runtime 62 passed, CLI 24 passed, manifest 135/135, authorized diff scope and scientific/inference/asset-preservation boundaries. Preserve the earlier canonical 127 passed / 17 failed / 6 errors execution record without relabeling it PASS.

Continuation startup: reread governance/handoff in canonical order, fetch and verify the exact accepted task HEAD above, inspect local changes and continue this same branch. Original predecessor creation steps in section 13 describe initial startup only. No new algorithm branch, reset/stash/clean/amend/discard is authorized. Unknown changes require STOP.

External closure remains governed by sections 18–21: existing helper read-only precheck, one controlled manifest-listed sync, then exact Git-canonical source_manifest write under P1D11B. Expected source/config mismatches are exactly README.md, buildreasonseg/runtime/outputs.py, buildreasonseg/runtime/pipeline.py, inference/README.md and tests/test_task8b_runtime.py. The accepted unsynchronized external manifest may be replaced only by the exact current committed canonical manifest. Any unexplained additional source/config mismatch requires STOP before writes.

Verify in order: helper 135/135, byte/hash-identical external manifest, external check_setup READY, external runtime and CLI targeted tests exit 0, then external complete full suite exit 0. No real model inference, A2, historical six-case suite, locked Demo candidates, or threshold/model/algorithm/scientific-contract change.

Preserve input, every pre-existing historical output and protected model/Qwen/SAM2 asset. Unit tests must use temporary roots and must not create real output runs. Do not delete unexpected evidence; any historical output change requires STOP. Historical logs/fixtures are preserved; naturally generated test logs are permitted.

Update the four authorized evidence/handoff files. On all gates passing, set CURRENT_TASK and EXECUTOR_STATE to READY_FOR_SUPERVISOR_AUDIT, next gate CHATGPT_TASK8B4_OUTPUT_LAYOUT_REMOTE_AUDIT, commit/push and STOP. Do not start Final Demo.
