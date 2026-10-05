# TO_DSH — MASK01_F1_R1_CORRECTIVE_FORENSICS

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `audit/task8b3-mask01-validity-forensics`
> Required starting HEAD: `590d080648ef5ae1fc6891dd18b97972c643c3cc`
> New task branch: `audit/task8b3-mask01-f1-r1-corrective-forensics`

## 0. CHATGPT AUDIT DISPOSITION OF F1

Previous task:

```text
MASK01_F1_VALIDITY_FORENSICS
```

is NOT accepted as a completed forensic gate.

Freeze:

```text
F1_DISPOSITION = REJECTED_INCOMPLETE_FORENSICS
```

Reasons established by ChatGPT's independent GitHub audit:

1. Git/diff safety was acceptable:
   - one commit on the expected parent;
   - only the four allowed report/handoff paths changed;
   - no product source/test/model modification.
2. The forensic content was incomplete/non-conforming:
   - no required three-way padding-gate conclusion was produced;
   - current `predict_one` post-inference SUCCESS contract was not correctly reconstructed;
   - failure taxonomy was not mapped to the predeclared categories;
   - observable-signal inventory was not actually built;
   - historical search expanded across large amounts of unrelated Task 6/7 evidence;
   - known Task 8B.3 explicit diagnostic paths were discovered but not correctly followed as the primary historical evidence;
   - the report conflated proposal/reference validity evidence with final-mask validity evidence.
3. The F1 JSON recorded a `final_head` that differs from the actual pushed commit.
   This is partly caused by a self-referential reporting requirement: a file committed in a single commit cannot
   truthfully contain the SHA of that same not-yet-created commit.

This R1 task corrects the forensic evidence only.

NO MASK-01 product repair is authorized.

---

## 1. GIT PRE-FLIGHT

Before any write, verify exactly:

```text
branch = audit/task8b3-mask01-validity-forensics
HEAD   = 590d080648ef5ae1fc6891dd18b97972c643c3cc
```

Allowed initial worktree state:

```text
clean
```

or only the user replacement of:

```text
handoff/TO_DSH.md
```

No other change is allowed.

If incompatible:

```text
STOP
```

Do not reset, rebase, amend, stash, clean, force-push, or discard unknown work.

After preflight create:

```text
audit/task8b3-mask01-f1-r1-corrective-forensics
```

directly from the required starting HEAD.

---

## 2. TASK SCOPE

This is a narrow corrective read-only forensic task.

Required objectives:

```text
A. Recover the exact production `predict_one` SUCCESS/failure validity contract.
B. Resolve the padding-gate semantics correctly.
C. Inspect the actual known Task 8B.3 diagnostic artifacts.
D. Perform a strictly Task-8B.3-bounded search for any additional saved final-mask artifacts.
E. Build the required failure taxonomy and observable-signal inventory.
F. Return evidence to ChatGPT.
```

Do NOT:

```text
modify product source
modify tests
modify configs
run model inference
run detector
run training
regenerate proposals
change thresholds
design a new mask validity rule
implement a repair
sync/write external RC1
```

No pytest is required or authorized.

---

## 3. TARGETED SOURCE AUDIT ONLY

Do not re-scan the full package.

Read only these canonical RC1 files unless a directly imported symbol requires one additional file:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/context.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/imageio.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/outputs.py
delivery_src/BuildReasonSeg_Advisor_RC1/configs/inference.yaml
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
```

If one additional directly referenced file is necessary, record why.

Do not read unrelated `_frozen/mvp` files merely because they contain words such as `padding`, `mask`, or `SUCCESS`.

---

## 4. EXACT CURRENT `predict_one` CONTRACT

Trace the actual `predict_one` control flow, not regex counts.

The report must identify, in execution order:

### Pre-core / upstream failures relevant to final segmentation

At minimum determine the actual current paths for:

```text
no proposals
invalid/missing assisted reference
automatic reference selection failure
reference too large for RC1 context
all directional candidates outside context
reference mask empty in context
core-chain engineering failure propagation
```

Record exact error code/reason where the source provides one.

### Post-inference validity gates

ChatGPT's source audit at starting HEAD found this sequence:

```text
mask_full, map_padding = context_to_global(...)

1. if not mask_full.any():
      E404 / empty_target_mask

2. if not (mask_full & _non_padding_mask(...)).any():
      E404 / mask_only_in_padding

3. target_centroid = _centroid(mask_full)

4. if not direction_satisfied(...):
      E404 / direction_constraint_violated

5. save outputs
6. payload status = SUCCESS
7. PipelineResult(status="SUCCESS", ...)
```

Verify this exact sequence against the local source.

If the local source contradicts the above at required HEAD, STOP and report the contradiction.

Do not call detector eligibility checks the final mask-validity contract.

The final report must explicitly state:

```text
POST_INFERENCE_GATES = 3
```

with their exact semantics.

---

## 5. `min_mask_pixels` / `min_mask_frac`

Perform only an exact, scoped identifier search under:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

for:

```text
min_mask_pixels
min_mask_frac
```

Also inspect `configs/inference.yaml`.

If absent, report exactly:

```text
min_mask_pixels = NOT_IMPLEMENTED
min_mask_frac   = NOT_IMPLEMENTED
```

Do not invent defaults.

Do not propose adding them in this task.

---

## 6. PADDING FORENSICS — REQUIRED RESOLUTION

Inspect together:

```text
pipeline.py::_context_rgb
pipeline.py::_non_padding_mask
context.py::context_to_global
imageio.py::crop_with_reflection
test_task8b_runtime.py::test_context_out_of_image_uses_reflection_and_maps_back
```

ChatGPT's current source interpretation is:

1. the reasoning context may contain reflected padding;
2. `context_to_global()` maps only the valid original-image subregion back to `mask_full`;
3. context pixels outside the original image are cropped away before `mask_full` exists;
4. `_non_padding_mask()` currently returns an all-True mask of original image size and does not use `padding`;
5. therefore `_non_padding_mask()` is not an effective independent padding classifier;
6. after the preceding `if not mask_full.any()` gate, the `mask_only_in_padding` condition cannot independently become true under the current `context_to_global` semantics;
7. an all-padding `mask_context` should be converted to an empty `mask_full` and fail as `empty_target_mask` before reaching the second branch;
8. this does NOT by itself establish padding leakage into successful output, because `context_to_global()` already crops padding away.

Verify each point from source/tests.

If all are confirmed, freeze:

```text
PADDING_GATE_CONCLUSION = PADDING_GATE_INEFFECTIVE
PADDING_LEAKAGE_ESTABLISHED = false
MASK_ONLY_IN_PADDING_BRANCH = LOGICALLY_REDUNDANT_OR_UNREACHABLE_AFTER_EMPTY_GATE
```

If any point is contradicted by the required source, do not silently choose another interpretation:

```text
STOP
```

and report the exact contradiction for ChatGPT.

Important:

```text
PADDING_GATE_INEFFECTIVE
```

means the `_non_padding_mask` gate is ineffective as an independent gate.

It does NOT mean successful masks are proven to contain padded pixels.

No repair is authorized.

---

## 7. TEST-COVERAGE FACTS

Read existing tests only.

At minimum report whether current tests cover:

```text
context outside image
reflection padding metadata
context_to_global valid-region mapping
all-padding mask_context
_non_padding_mask behavior
mask_only_in_padding reason
empty_target_mask reason
direction_constraint_violated reason
full predict_one SUCCESS validity chain
```

For each item use exactly:

```text
COVERED
NOT_COVERED
PARTIAL
```

Include the relevant test name(s).

Do not run pytest.

---

## 8. KNOWN TASK 8B.3 HISTORICAL ARTIFACTS — INSPECT FIRST

The previous F1 itself discovered versioned evidence that points to these exact external diagnostic files:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\result.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\proposals.json

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\result.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\proposals.json

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\result.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\proposals.json

C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\result.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\proposals.json
```

The authoritative versioned references include:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
evaluation/task8b3_ref01_locked_reference_forensics.json
evaluation/task8b3_ref01_e3c1_locked_record_replay.json
```

Inspect the exact external files READ ONLY if they still exist.

For each `result.json`, record:

```text
exists
sha256
mode
status
reason/error fields if any
reference_mode if present
reference_id/effective_reference_id if present
direction if present
mask_area if present
target_centroid if present
context_padding if present
output_paths
whether it represents inspect-proposals mode or full predict_one mode
whether it contains evidence about a final target mask
```

For each `proposals.json`, record only what is relevant to provenance/reference context.

Do not treat proposal records as final-mask evidence.

Do not write to the external RC1.

---

## 9. STRICTLY BOUNDED ADDITIONAL TASK-8B.3 SEARCH

The previous F1's broad scan is rejected.

Use a two-stage bounded search only.

### Stage A — select versioned Task-8B.3 files

Enumerate tracked paths only under:

```text
docs/
evaluation/
handoff/
```

Filter filenames/paths to those containing:

```text
task8b3
8b3
```

Do not inspect unrelated Task 6/7 files.

### Stage B — inspect only those Task-8B.3 versioned files

Search inside that filtered set only for explicit references containing terms such as:

```text
inference/output
diagnostics
masks
overlays
result.json
A1
A2
A3
A4
B1
B2
1010
1003
1008
1009
```

Follow only exact referenced local file paths.

If an exact referenced directory is given, list immediate children only.

Do not recursively walk:

```text
workspace
artifacts
logs
inference
delivery
```

unless the exact child path is explicitly referenced by a Task-8B.3 versioned file.

Record every followed path and its authorizing versioned source.

---

## 10. FINAL-MASK ARTIFACT CLASSIFICATION

For every historical artifact found, classify it as exactly one:

```text
FULL_FINAL_MASK_EVIDENCE
FULL_RESULT_METADATA_ONLY
PROPOSAL_REFERENCE_EVIDENCE_ONLY
NON_MASK_DIAGNOSTIC
MISSING
```

Only `FULL_FINAL_MASK_EVIDENCE` may be used to compute descriptive final-mask properties.

Do not regenerate missing masks.

Do not infer a saved final mask from the existence of proposals or a `SUCCESS` string.

If no `FULL_FINAL_MASK_EVIDENCE` is found, explicitly freeze:

```text
HISTORICAL_FINAL_MASK_EVIDENCE = NOT_FOUND
```

This is an acceptable R1 result.

---

## 11. FAILURE TAXONOMY — MUST USE PREDECLARED CATEGORIES

Use only:

```text
TAX_RUNTIME_STRUCTURAL
TAX_CONTEXT_PADDING_COORDINATE
TAX_UPSTREAM_REFERENCE_RELATION
TAX_SEMANTIC_TARGET_MISMATCH
TAX_RUNTIME_VALID_QUALITY_POOR
TAX_UNKNOWN_EVIDENCE_GAP
```

Do not fill this section with a list of generic exceptions.

For every actual observation/case, write:

```text
case_or_observation
taxonomy
evidence
what_is_established
what_is_not_established
```

Important evidence already frozen from prior REF work may be cited:

```text
right  -> automatic reference correct/stable
left   -> residual wrong automatic reference selection
above  -> eligibility blocker repaired in current selector
below  -> residual wrong automatic reference selection
```

But distinguish historical artifact state from current selector state.

A wrong reference may support:

```text
TAX_UPSTREAM_REFERENCE_RELATION
```

It does NOT by itself prove a specific final target mask was semantically wrong unless final-mask/manual/GT evidence links them.

---

## 12. OBSERVABLE-SIGNAL INVENTORY

Build a real inventory.

For each signal, record:

```text
signal
source
availability
persisted_where
can_detect_runtime_structural_failure
can_establish_semantic_target_correctness
notes
```

Availability must be exactly one of:

```text
RUNTIME_ALWAYS_AVAILABLE
RUNTIME_CONDITIONALLY_AVAILABLE
PERSISTED_ARTIFACT_AVAILABLE
REQUIRES_GT_OR_MANUAL_REVIEW
NOT_CURRENTLY_AVAILABLE
```

At minimum inspect and classify, where actually present:

```text
mask_full nonempty
mask_area
target_centroid
direction_satisfied result/reason
reference_mode
reference_id
reference_area
reference_confidence
reference_bbox
direction
reasoning_context
context_padding
directional_guard
proposal count
proposal metadata
field_mass
diagnostic probability/logit maps
saved mask path
GT IoU
manual semantic correctness
```

Do not add new runtime fields.

Do not invent unavailable fields.

---

## 13. REQUIRED INTERPRETATION BOUNDARY

The report must explicitly separate:

```text
RUNTIME_VALIDITY
SEMANTIC_TARGET_CORRECTNESS
```

Freeze the following principle:

```text
Passing current runtime validity gates is NOT evidence that the predicted mask
corresponds to the intended semantic target.
```

Also distinguish:

```text
padding-gate redundancy
```

from:

```text
False SUCCESS root cause
```

Do not claim the padding helper is the root cause of false SUCCESS unless evidence establishes that.

---

## 14. OUTPUTS

Create exactly:

```text
docs/task8b3_mask01_f1_r1_corrective_forensics.md
evaluation/task8b3_mask01_f1_r1_corrective_forensics.json
```

Modify exactly:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other repository path may change.

---

## 15. REQUIRED REPORT CONTENT

The Markdown report must contain:

```text
1. Git starting identity
2. Why F1 was rejected
3. Exact predict_one SUCCESS contract
4. Upstream and post-inference failure paths
5. min_mask_* exact search result
6. Padding semantics proof
7. Test coverage matrix
8. Known 8B.3 external artifact inspection
9. Additional bounded Task-8B.3 search
10. Final-mask artifact classification
11. Failure taxonomy
12. Observable-signal inventory
13. Runtime validity vs semantic correctness
14. Evidence gaps
15. R1 disposition
```

The JSON must represent the same facts structurally.

Required terminal fields:

```text
task_id = MASK01_F1_R1_CORRECTIVE_FORENSICS

previous_f1_disposition =
REJECTED_INCOMPLETE_FORENSICS

padding_gate_conclusion =
PADDING_GATE_INEFFECTIVE
```

only after the source verification in Section 6 succeeds.

Also record:

```text
padding_leakage_established = false
repair_implemented = false
product_source_modified = false
inference_executed = false
training_executed = false
repair_decision = DEFER_TO_CHATGPT
next_gate = CHATGPT_MASK01_F1_R1_REVIEW
```

Task status must be one of:

```text
COMPLETE
COMPLETE_WITH_EVIDENCE_GAPS
STOP
FAILED
```

---

## 16. COMMIT SHA REPORTING RULE — CORRECTED

Do NOT attempt to write the final commit SHA into files that are themselves part of that same commit.

Inside the committed report record only:

```text
starting_head = 590d080648ef5ae1fc6891dd18b97972c643c3cc
report_base_head = 590d080648ef5ae1fc6891dd18b97972c643c3cc
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```

After committing and pushing, print to terminal:

```text
LOCAL_FINAL_HEAD=<actual sha>
REMOTE_FINAL_HEAD=<actual sha>
FINAL_PARENT=<actual parent sha>
```

ChatGPT will independently retrieve these values from GitHub.

Do not create a second commit merely to insert the first commit's SHA.

Do not amend.

---

## 17. DIFF / COMMIT / PUSH

Before staging verify only the four authorized paths changed.

For `COMPLETE` or `COMPLETE_WITH_EVIDENCE_GAPS`, stage exactly:

```text
docs/task8b3_mask01_f1_r1_corrective_forensics.md
evaluation/task8b3_mask01_f1_r1_corrective_forensics.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Commit exactly once:

```text
git commit -m "docs(rc1): correct mask validity forensics"
```

Push:

```text
audit/task8b3-mask01-f1-r1-corrective-forensics
```

No force push.

Verify remote HEAD equals local HEAD and working tree is clean.

Then STOP.

---

## 18. STOP / SAFETY

If any source fact at the required starting HEAD contradicts ChatGPT's frozen source interpretation in Section 4 or 6:

```text
STOP
```

Record the exact source contradiction only.

Do not repair it.

If a historical artifact is missing:

```text
record MISSING / evidence gap
continue within bounded scope
```

Do not broaden the search.

---

## 19. SUCCESS DEFINITION

R1 is complete only if ChatGPT can answer, from its evidence:

```text
What exactly makes predict_one return SUCCESS?
Which runtime validity gates are real?
What is the exact role and reachability of mask_only_in_padding?
Does current code permit padding leakage, or merely contain a redundant gate?
Which Task-8B.3 artifacts actually contain final target masks?
Which failures are runtime-detectable?
Which failures require upstream correctness, GT, or manual semantic review?
What evidence is still missing before any MASK-01 repair can be designed?
```

No new validity rule is authorized.

After report + commit + push:

```text
STOP
```
