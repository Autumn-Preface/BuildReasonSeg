# CURRENT_TASK — TASK8C_FINAL_DEMO_V1

## 0. Metadata

```text
Task ID:
TASK8C_FINAL_DEMO_V1

Status:
IN_PROGRESS

Decision owner:
ChatGPT Supervisor

Authorized executor:
CODEX

Project phase:
RC1_FINAL_DEMO

Accepted predecessor:
TASK8B4_RUN_ISOLATED_OUTPUT_LAYOUT_V1

Accepted predecessor branch:
fix/task8b4-run-isolated-output-layout-v1

Accepted predecessor remote HEAD:
92f28133931231d16b6053aca253b9e5073955ae

Required task branch:
eval/task8c-final-demo-v1

Next gate:
CHATGPT_TASK8C_FINAL_DEMO_REMOTE_AND_VISUAL_AUDIT
```


# 1. Supervisor Disposition

Task 8B.4 is formally ACCEPTED by the ChatGPT Supervisor.

Accepted engineering state includes:

```text
RUN_ISOLATED_OUTPUT_LAYOUT_IMPLEMENTED
external source/config identity = 135/135
external check_setup = READY
external complete suite = 150 passed
```

Current scientific limitations remain:

```text
MEM-01  = CLOSED
MASK-01 = CLOSED_ENGINEERING_HARDENING
REF-01  = ACTIVE_RESIDUAL_SELECTION_LIMITATION
PROP-01 = OPEN_ENGINEERING_DEFECT

REF-01:
  right = stable
  above = repaired
  left  = residual selection limitation
  below = residual selection limitation

PROP-01:
  A2 zero proposals remains OPEN
```

No repair is authorized by this task.

This task is an **evaluation/demo milestone only**.


# 2. Governance Staleness Note

At the accepted predecessor HEAD, `governance/PROJECT_STATE.yaml` still records:

```text
8B.4 = NOT_STARTED
final_demo = NOT_STARTED
```

The `8B.4 = NOT_STARTED` field is stale relative to the later explicit Supervisor acceptance at:

```text
92f28133931231d16b6053aca253b9e5073955ae
```

Do NOT edit governance files in Task 8C.

The task book and actual Git evidence have higher execution relevance for this milestone.

Final governance closure will occur only after Supervisor acceptance of Task 8C.


# 3. Scientific Purpose

The goal is to obtain one frozen, auditable qualitative end-to-end demonstration of the current RC1 system on the four immutable supported-domain candidates.

The demonstrated chain is:

```text
Natural-language instruction
→ Qwen 2B ProgramHead
→ deterministic supported program
→ YOLO26m proposals
→ automatic largest reference selection
→ spatial reasoning
→ SAM/SAM2 + GRF + target-aware segmentation
→ final mask / overlay
```

This task must answer:

1. Does the current frozen RC1 execute the four supported spatial-reasoning types end to end?
2. What reference does the automatic system actually select?
3. What target mask does it actually output?
4. How does that output compare with frozen GT after inference?
5. Which successes and limitations are honestly visible?

It does NOT attempt to improve the system.


# 4. Mandatory Scientific Reuse Disclosure

The following disclosure must appear verbatim in the final report:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Chinese translation must also be included:

> 这些定性 Demo 候选是在 Task 7J 最终冻结架构测试指标已经被消耗之后，从冻结的 BuildSpatialReason v0.2 测试划分中确定性选取的。其定性复用不会改变、替换或重新选择任何已报告的 Task 7J 指标、模型、阈值、随机种子或架构。


# 5. Frozen Supported Domain

The Demo is restricted to:

```text
BuildSpatialReason v0.2
WHU-EA-NativeVector v1.0
WHU East Asia
scene_disjoint_v1
RGB optical aerial/overhead imagery
tile-relative spatial reasoning
```

No claim may be made about:

```text
cross-city generalization
broad geographic generalization
cross-sensor generalization
arbitrary aerial-image robustness
SAR
infrared
raw multispectral imagery
unrestricted natural language
```


# 6. Four Immutable Final-Demo Candidates

Execution order is frozen:

```text
right → left → above → below
```

## 6.1 RIGHT

```text
relation:
right

sample_id:
buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91

image:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif

SHA256:
1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2

expected program:
largest_to_right_of_to_nearest

frozen prompt:
最大建筑右侧最近的建筑

GT reference tile_instance_id:
4

GT target tile_instance_id:
3
```


## 6.2 LEFT

```text
relation:
left

sample_id:
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3

image:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif

SHA256:
eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38

expected program:
largest_to_left_of_to_nearest

frozen prompt:
最大建筑左侧最近的建筑

GT reference tile_instance_id:
26

GT target tile_instance_id:
24
```


## 6.3 ABOVE

```text
relation:
above

sample_id:
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314

image:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif

SHA256:
0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd

expected program:
largest_to_above_to_nearest

frozen prompt:
最大建筑上方最近的建筑

GT reference tile_instance_id:
4

GT target tile_instance_id:
6
```


## 6.4 BELOW

```text
relation:
below

sample_id:
buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450

image:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif

SHA256:
c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7

expected program:
largest_to_below_to_nearest

frozen prompt:
最大建筑下方最近的建筑

GT reference tile_instance_id:
3

GT target tile_instance_id:
2
```


# 7. Important Historical-Evidence Boundary

Historical P1D12 proposal counts and historical REF-01 proposal IDs were produced before the current final detector/input-contract state.

They are historical forensic evidence only.

Do NOT use historical values such as:

```text
right selected id 1
left selected id 14 / GT-covered id 30
above selected id 5
below selected id 1 / GT-covered id 2
```

as current Demo oracles.

Do NOT require current proposal counts to equal historical P1D12 counts.

Do NOT pass any historical proposal ID through `--reference-id`.

Current Final Demo must observe the current production pipeline from scratch.


# 8. Final Demo Mode

All four cases MUST use:

```text
AUTOMATIC REFERENCE MODE
```

Forbidden during the formal Final Demo:

```text
--reference-id
--inspect-proposals
manual reference selection
manual proposal selection
manual target selection
```

There is no Assisted-Mode rescue in Task 8C.

If automatic reference selection is wrong, preserve that result.


# 9. No-Retry Rule

Each candidate may be executed exactly once.

Formal execution:

```text
RIGHT  = one attempt
LEFT   = one attempt
ABOVE  = one attempt
BELOW  = one attempt
```

No individual retry.

No prompt retry.

No altered paraphrase.

No second detector pass.

No Assisted Mode after failure.

No re-running successful cases for a prettier image.

A runtime failure remains part of the Final Demo evidence.


# 10. Language Driver Rules

Use the proven Task 8B.3 interactive-driver semantics or a newly isolated equivalent with dedicated tests.

The formal prompt strings are frozen in §6.

For each case:

### Direct confirmation

Answer:

```text
Y
```

ONLY when:

```text
initial_program == expected_program
```

Otherwise answer:

```text
N
```

Classification:

```text
DIRECT_CORRECT
LANGUAGE_ERROR_SUPPORTED_WRONG
```


### Suggestion confirmation

Answer:

```text
Y
```

ONLY when:

```text
suggested_program == expected_program
```

Otherwise:

```text
N
```

Classification:

```text
SUGGESTION_CORRECT
SUGGESTION_WRONG
```


### Runtime fallback request

If the product asks:

```text
是否进入有限兼容模式？ [Y/N]
```

answer:

```text
N
```

Task 8C is a Qwen-first Final Demo.

Do not convert Qwen runtime failure into fallback success.


# 11. Harness Architecture

Create:

```text
scripts/task8c_final_demo_runner.py
scripts/task8c_final_demo_evaluate.py
tests/test_task8c_final_demo.py
```

The two operational roles MUST remain separate.

## Runner

`task8c_final_demo_runner.py`:

- knows candidates, image hashes, prompts and expected programs;
- drives the external `predict.py`;
- contains NO GT mask reconstruction logic;
- contains NO GT reference/target decision logic;
- does NOT call `--reference-id`;
- does NOT call `--inspect-proposals`;
- invokes each formal candidate at most once;
- records transcripts and objective runtime evidence.

## Evaluator

`task8c_final_demo_evaluate.py`:

- performs ZERO model or detector calls;
- runs only after all four formal runs are complete;
- reads frozen output artifacts;
- reconstructs native-vector GT reference/target masks;
- computes post-hoc metrics and visual review panels;
- must not modify any RC1 output.


# 12. Harness Unit Gates Before Real Inference

Dedicated tests must prove at least:

1. candidate order is exactly right → left → above → below;
2. all four sample IDs are exact;
3. all four raster SHA256 values are exact;
4. all four prompts are exact;
5. all expected programs are exact;
6. direct prompt answers Y only for the expected program;
7. wrong supported program receives N;
8. correct suggestion receives Y;
9. wrong suggestion receives N;
10. runtime fallback receives N;
11. no case can be retried by the runner;
12. `--reference-id` cannot appear in formal argv;
13. `--inspect-proposals` cannot appear in formal argv;
14. GT/native-vector paths are not accessed by the runner;
15. evaluator contains no detector/model execution;
16. evaluator cannot call `predict.py`;
17. runner preserves per-case timeout;
18. a failed case does not stop later cases;
19. no output directory is cleared or deleted;
20. generated evidence is deterministic from frozen runtime results.

Required gate:

```text
python -m pytest tests/test_task8c_final_demo.py -q
```

must be 100% PASS before formal execution.


# 13. Preflight

Read in order:

1. `AGENTS.md`
2. `governance/PROJECT_STATE.yaml`
3. `governance/DECISIONS.md`
4. `handoff/CURRENT_TASK.md`
5. `handoff/EXECUTOR_STATE.yaml`

Then:

```text
git fetch
```

Verify:

```text
fix/task8b4-run-isolated-output-layout-v1
remote HEAD =
92f28133931231d16b6053aca253b9e5073955ae
```

If the remote predecessor advanced unexpectedly:

inspect only and STOP unless the additional commits are clearly Supervisor-authorized state-only changes.

Do NOT reset/rebase/amend/stash/clean.

Create:

```text
eval/task8c-final-demo-v1
```

from the accepted predecessor.


# 14. External RC1 Integrity Preflight

External root:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Before formal inference:

### Source identity

Existing sync helper `--check`:

```text
checked = 135
match = 135
missing = 0
mismatch = 0
```


### Setup

```text
python check_setup.py
```

must report:

```text
BuildReasonSeg environment: READY
```


### Asset snapshot

Record byte length + SHA256 for all protected:

```text
detector weights
decoder weights
SAM2 weights/config
ProgramHead weights
Qwen asset files
model metadata/config
```

No value may change during the milestone.


# 15. Candidate Identity Preflight

For all four images verify:

```text
exists = true
size = 512 × 512
mode = RGB
format = TIFF
SHA256 = exact §6 lock
```

Any mismatch:

```text
STOP
```

No substitute image is permitted.


# 16. Output Preflight

Snapshot current external:

```text
inference/output/
```

including path, byte length, SHA256 and mtime for all pre-existing files.

Do NOT delete or clear old outputs.

Task 8B.4 allocator decides the new run roots.

Record the actual allocated run root after each case.

Historical output must remain intact.


# 17. Formal Final-Demo Freeze

Before the first real case, persist:

```text
FORMAL_FINAL_DEMO_FREEZE_V1

task branch HEAD
runner SHA256
evaluator SHA256
dedicated-test SHA256

external source manifest SHA256
external 135/135 status

check_setup READY

protected asset hashes

candidate sample IDs
candidate raster paths
candidate raster SHA256 values

frozen prompts
expected programs

execution order

output inventory hash
```

Commit + push the freeze checkpoint BEFORE model inference.


# 18. Formal Execution

Exactly one invocation of the final runner:

```text
<ENV_PYTHON> scripts/task8c_final_demo_runner.py
```

Runner executes the four cases once each in frozen order.

For each case use actual external RC1:

```text
predict.py
--image "<locked raster>"
--prompt "<frozen prompt>"
```

No other inference option changing semantics.

Default automatic reference mode only.


# 19. Per-Case Runtime Evidence

Record:

```text
relation
sample_id
image path
image SHA256
prompt
expected_program

initial_program
initial_confidence
suggested_program
driver_decision
language_status

exit_code
runtime_status
error_code
error_reason

raw_proposal_count
merged_proposal_count
tile_count

reference_mode
reference_id
reference_area
reference_confidence
reference_bbox

direction
reasoning_context

mask_area
target_centroid
timings

run_root
mask_path
overlay_path
diagnostics_path
```

For every written artifact record:

```text
bytes
SHA256
```


# 20. SUCCESS Semantics

Retain GOV-D005.

A runtime result:

```text
status = SUCCESS
```

means only:

```text
validity_scope = RUNTIME_STRUCTURAL_ONLY
semantic_status = NOT_EVALUATED
```

The runner must NOT label a SUCCESS result:

```text
SEMANTIC_SUCCESS
CORRECT_TARGET
DEMO_PASS
```

before GT audit.


# 21. FAILED Semantics

If a case fails:

- preserve error code and reason;
- preserve diagnostics;
- require no fake mask;
- require no fake overlay;
- continue to the next locked case;
- do not retry.

A case-level runtime failure does NOT invalidate the formal procedure.


# 22. GT Embargo

Native-vector GT must not influence formal inference.

The evaluator may begin only after:

```text
all four child processes have exited
AND
formal runtime-results JSON has been written
AND
all output artifact hashes have been frozen
```

After this point:

```text
NO model rerun
NO prompt rerun
NO reference override
NO source change
```

GT is evaluation-only.


# 23. Frozen GT Reconstruction

Reuse the existing WHU-native-vector / BuildSpatialReason v0.2 infrastructure.

For each candidate reconstruct:

```text
canonical GT reference mask
canonical GT target mask
all GT building instance masks for that tile
```

Required frozen IDs are those in §6.

Verify reconstructed masks are nonempty and consistent with existing native-vector metadata.

Do NOT regenerate or alter annotations.


# 24. Reference Audit Without Detector Rerun

For a runtime SUCCESS case:

use:

```text
diagnostics/reference_context_mask.png
result.json -> reasoning_context.origin
result.json -> reasoning_context.size
```

to map the frozen selected-reference context mask back to the original 512×512 tile coordinate system.

Use the same deterministic crop-to-global mapping semantics as the product `context_to_global()`.

Do NOT rerun detector to recover the reference mask.

Then compute against all GT building masks:

```text
selected_reference_best_gt_instance_id
selected_reference_best_gt_iou
selected_reference_gt_reference_iou
reference_identity_best_overlap_match
```

Definition:

```text
reference_identity_best_overlap_match =
    best_gt_instance_id == canonical_GT_reference_id
    AND best_gt_iou > 0
```

This is an identity-best-overlap fact, NOT a segmentation-quality threshold pass.


# 25. Target Audit

For every runtime SUCCESS final mask compute:

```text
target_gt_iou
target_gt_dice
predicted_mask_area
gt_target_area

best_gt_instance_id
best_gt_instance_iou

target_identity_best_overlap_match
```

Definition:

```text
target_identity_best_overlap_match =
    best_gt_instance_id == canonical_GT_target_id
    AND best_gt_instance_iou > 0
```

Also report:

```text
reference_identity_best_overlap_match
target_identity_best_overlap_match
```

Do NOT introduce a new IoU pass threshold.

Do NOT tune any threshold after seeing these values.


# 26. Semantic-Chain Classification

For evidence organization only, use these exact enums:

```text
CHAIN_IDENTITY_MATCH
REFERENCE_IDENTITY_MISMATCH
TARGET_IDENTITY_MISMATCH
REFERENCE_AND_TARGET_IDENTITY_MISMATCH
RUNTIME_FAILED
LANGUAGE_FAILED
NOT_EVALUABLE_MISSING_ARTIFACT
```

Rules:

### CHAIN_IDENTITY_MATCH

```text
reference_identity_best_overlap_match = true
AND
target_identity_best_overlap_match = true
```

This does NOT mean perfect mask quality.

Continuous IoU/Dice values must still be reported.

### REFERENCE_IDENTITY_MISMATCH

reference mismatch but target identity matches.

### TARGET_IDENTITY_MISMATCH

reference identity matches but target identity does not.

### REFERENCE_AND_TARGET_IDENTITY_MISMATCH

both do not match.

### RUNTIME_FAILED

formal runtime failed before a final target mask exists.

### LANGUAGE_FAILED

the expected canonical program was not safely accepted/executed.

No case may be relabeled manually.


# 27. Review Images

After GT evaluation, create one review PNG per relation:

```text
evaluation/task8c_final_demo_v1/right_review.png
evaluation/task8c_final_demo_v1/left_review.png
evaluation/task8c_final_demo_v1/above_review.png
evaluation/task8c_final_demo_v1/below_review.png
```

Also create:

```text
evaluation/task8c_final_demo_v1/contact_sheet.png
```

Each per-case review should clearly show:

```text
A. original RGB input
B. automatic selected reference + canonical GT reference boundary
C. predicted target mask/overlay + canonical GT target boundary
D. compact text summary:
   program
   runtime status
   automatic reference ID
   reference GT IoU
   target GT IoU
   target Dice
   semantic-chain enum
```

Visualizations are evaluation-only.

They must not influence inference.


# 28. Review-Pack Accuracy

Review PNGs must use the exact saved formal-run artifacts.

Do NOT:

- rerun prediction to obtain a better overlay;
- substitute a different case;
- manually repaint a mask;
- crop away an obvious failure;
- omit a failed candidate.

If a case has no output mask, its review image must show:

```text
RUNTIME FAILED
<error code>
```

alongside the original image and available diagnostics.


# 29. Final-Demo Outcome

The executor does NOT decide whether the project Demo is scientifically successful.

It reports objective counts:

```text
cases_attempted
language_expected_program_executed
runtime_success
runtime_failed

chain_identity_match
reference_identity_mismatch
target_identity_mismatch
both_identity_mismatch
not_evaluable

mean_target_iou_over_runtime_success
mean_target_dice_over_runtime_success
```

Do not hide failures from aggregate statistics.


# 30. No-Tuning / No-Rescue Statement

Final evidence must explicitly state:

```text
candidate replacement = NONE
prompt replacement = NONE
prompt retry = NONE
inference retry = NONE
reference override = NONE
inspect-before-run = NONE
threshold tuning = NONE
ranking tuning = NONE
mask-quality tuning = NONE
architecture change = NONE
model change = NONE
checkpoint change = NONE
training = NONE
dependency installation = NONE
```

This section is mandatory.


# 31. Protected Historical Evidence

Task 8C MUST NOT modify or overwrite:

```text
historical A1/A2/A3/A4/B1/B2 suite
P1D12 inspect diagnostics
REF-01 forensic evidence
A2 forensic evidence
Task 8B.4 historical output snapshot
```

The new run-isolated layout must create separate Final-Demo run roots.


# 32. Product Immutability

Forbidden modifications:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/**
scripts/sync_advisor_rc1_delivery.py
source_manifest.json
external RC1 source/config
model/**
governance/**
```

Exception:

none.

If Final Demo exposes a product defect requiring code change:

record it and STOP.

Do not patch it inside Task 8C.


# 33. Allowed Repository Changes

Allowed:

```text
scripts/task8c_final_demo_runner.py
scripts/task8c_final_demo_evaluate.py
tests/test_task8c_final_demo.py

docs/task8c_final_demo_v1.md
evaluation/task8c_final_demo_v1.json
evaluation/task8c_final_demo_v1/*.png

handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
```

No other repository path is authorized.


# 34. External Writes Allowed

External writes are restricted to:

```text
inference/output/<new Final-Demo run roots>/**
logs/task8c_final_demo_v1/**
```

plus unavoidable runtime logs naturally produced by the existing RC1.

External source/config/model files are read-only.


# 35. Post-Run Integrity

After all inference/evaluation work:

verify external:

```text
source/config 135/135 unchanged
source_manifest unchanged
protected assets unchanged
inference/input unchanged
all pre-existing inference/output files unchanged
```

Only newly created Task 8C run roots and authorized new Task 8C logs may be added.


# 36. Required Evidence JSON

Create:

```text
evaluation/task8c_final_demo_v1.json
```

Required top-level fields:

```text
task_id
status

starting_branch
starting_head
task_branch
formal_freeze_commit

scientific_reuse_disclosure

source_manifest_identity
external_preflight
protected_assets_before
protected_assets_after

candidate_lock
runner_identity
evaluator_identity
dedicated_test_result

formal_runner_invocation_count
case_attempt_order

cases

gt_audit_started_after_runtime_freeze
model_calls_during_gt_audit

aggregate

legacy_output_unchanged
input_unchanged
protected_assets_unchanged
source_identity_unchanged

no_candidate_replacement
no_prompt_retry
no_inference_retry
no_reference_override
no_inspect_pre_run
no_threshold_tuning
no_ranking_tuning
no_model_change
no_architecture_change
no_training
no_dependency_install

review_artifacts
```

Each case must contain the §19 and §24–26 evidence.


# 37. Required Report

Create:

```text
docs/task8c_final_demo_v1.md
```

It must include:

1. scope;
2. frozen candidate table;
3. reuse disclosure;
4. formal freeze;
5. language results;
6. runtime results;
7. reference post-hoc audit;
8. target GT metrics;
9. semantic-chain classifications;
10. review-pack paths;
11. no-retry/no-tuning statement;
12. known limitations;
13. explicit statement that final scientific verdict belongs to ChatGPT Supervisor.


# 38. Git Checkpoints

## Checkpoint 1 — Harness Ready

Completed:

```text
runner
evaluator
dedicated tests 100% PASS
preflight identities PASS
```

No real candidate has run.

Commit + push.


## Checkpoint 2 — Formal Freeze

Persist:

```text
FORMAL_FINAL_DEMO_FREEZE_V1
```

Commit + push.

No product change after this point.


## Checkpoint 3 — Formal Run Complete

Exactly one runner invocation complete.

All four candidates attempted.

Raw runtime evidence and artifact hashes frozen.

Commit + push evidence BEFORE GT evaluation if repository state permits.

Do not rerun.


## Checkpoint 4 — GT Audit + Review Pack

Zero model calls.

Post-hoc metrics and review images complete.

Integrity checks complete.

Commit + push.


# 39. STOP Conditions

STOP with evidence if:

- accepted predecessor Git state conflicts;
- unknown repository modifications exist;
- external 135/135 preflight fails;
- `check_setup.py` is not READY;
- any locked raster hash mismatches;
- dedicated harness tests are not 100% green;
- formal runner would require product modification;
- formal runner crashes after real execution begins;
- a retry would be required;
- GT evaluation would require detector/model rerun;
- pre-existing output is modified;
- protected asset hash changes;
- source/config drift appears;
- destructive Git operation appears necessary.

Do not self-repair across these boundaries.


# 40. Git Safety

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

Unknown changes must be preserved.


# 41. Success Criterion of the Executor Task

Task 8C executor completion means:

```text
formal procedure executed correctly
+
all four locked candidates attempted once
+
all evidence preserved
+
post-hoc GT audit completed without model rerun
+
review pack generated
```

It does NOT require:

```text
4/4 runtime success
4/4 semantic-chain match
high IoU
visually perfect masks
```

A genuine failure is a valid scientific Final Demo result.


# 42. Final State

On procedural completion:

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
CHATGPT_TASK8C_FINAL_DEMO_REMOTE_AND_VISUAL_AUDIT
```

Commit and push all authorized evidence.

Then STOP.

Do NOT:

- modify governance;
- repair a failed Final-Demo case;
- run Assisted Mode;
- replace any candidate;
- start a new model/algorithm task.

# Executor STOP checkpoint — 2026-10-07

The dedicated harness revalidation exited 1: 27 passed / 15 errors. All 15 errors occurred in pytest temporary-directory fixture setup because the supplied nested --basetemp parent did not exist (WinError 3). The previous 40-pass invocation covered an earlier harness/test revision. Per section 39, execution stops with evidence; no rerun or automatic repair was performed. No external preflight, formal freeze, real candidate, or GT audit has run.

See docs/task8c_final_demo_v1.md and evaluation/task8c_final_demo_v1.json. Supervisor disposition is required before continuing; no formal attempt has been consumed.


# Supervisor continuation — 2026-10-07

Disposition: TASK8C_FINAL_DEMO_V1 = STOP_VALID / CONTINUE_AUTHORIZED. Accepted task-branch remote HEAD: 44a7dd1c3c3c8afe7ef1a0a569619017a7d4e4ea. Continue the same branch. Preserve prior STOP evidence. Runner/evaluator/tests bytes must not change. Only the first test invocation basetemp was corrected to a fresh path under an existing TEMP parent. Same HEAD revision: 42 passed / exit 0; all three harness SHA256 values unchanged. The prior STOP gate is superseded by this explicit continuation. Proceed with original external/static gates and pushed formal freeze before any real inference. All no-retry, automatic-reference, GT embargo, source/governance immutability and final audit rules remain in force.


# Executor external-preflight STOP checkpoint — 2026-10-07

Authorized same-revision dedicated test rerun: 42/42 PASS / exit 0; all runner/evaluator/test bytes unchanged. External 135/135, check_setup READY and all four TIFF identity gates PASS. However check_setup imported Ultralytics, which fell back from the uncreated nested task TEMP cache to the external working directory and created Ultralytics/settings.json (606 bytes; SHA256 1dae32b8abfc0f084cdee6c446dcf5420031bfcf005fb358f473a9a34d368cdf). This persistent external config file is outside section 34 allowed write roots. It is preserved without deletion, relocation or cleanup. Per sections 32/34/35/39, STOP before formal freeze or model execution.

Historical output (1543 files, including mtime), input (6 files), protected model assets (20 file byte/SHA identities), existing logs (327 files), external source/config 135/135 and source_manifest remain equal to accepted predecessor evidence. No formal runner invocation, real candidate attempt, model inference or GT audit has occurred. The settings file is not normalized into an accepted historical baseline. Supervisor disposition is required before continuation.


# Supervisor second-STOP continuation — 2026-10-07

Disposition: TASK8C_FINAL_DEMO_V1 = SECOND_STOP_VALID / CONTINUE_AUTHORIZED. Accepted remote HEAD: ce408ac73baa5c3e5fcd0b128c545b1717b456ee; same evaluation branch. Existing 42/42 PASS, 135/135, manifest MATCH, setup READY and four TIFF gates accepted; no tests/setup rerun. Three harness files remain unchanged.

External Ultralytics/settings.json (606 bytes; SHA256 1dae32b8abfc0f084cdee6c446dcf5420031bfcf005fb358f473a9a34d368cdf) is explicitly GRANDFATHERED_TASK8C_PREFLIGHT_RUNTIME_SIDE_EFFECT. Preserve its exact bytes; do not delete, move, edit or overwrite. It is not manifest-listed source/config or a model asset and changes no scientific/product contract. Include its immutable identity in formal freeze/final integrity; no additional external-root runtime config additions are authorized.

L0 correction completed before use: existing writable TEMP/task8c-yolo-config-v1 (including Ultralytics subdirectory) and TEMP/task8c-mpl-cache-v1 outside repo/external. Future Ultralytics processes must use the exact frozen YOLO_CONFIG_DIR. Complete formal-freeze commit/push before the single authorized runner invocation; no retries, automatic reference only, GT embargo and scientific immutability remain in force.
