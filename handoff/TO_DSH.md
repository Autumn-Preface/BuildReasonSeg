# TO_DSH — RC1-DEMO-MASK-01-F1: SUCCESS Validity Source Audit & Historical Artifact Forensics

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `43c24de59625dfb97dcc605435fa83b88be46035`
> New task branch: `audit/task8b3-mask01-validity-forensics`

---

# 0. TASK POSITION

Previous accepted gate:

```text
RC1-DEMO-REF-01-E3C3
=
ACCEPTED
```

Frozen decisions retained:

```text
REJECT_FURTHER_SCALAR_RANK_REPAIR

ASSISTED_REFERENCE_OVERRIDE

REF-01 =
ACTIVE_RESIDUAL_SELECTION_LIMITATION
```

The current technical stage is:

```text
MASK01_VALIDITY_FORENSICS_DESIGN
```

This task is the first forensic phase only:

```text
RC1-DEMO-MASK-01-F1
=
SUCCESS Validity Source Audit
+
Historical Artifact Forensics
```

This task does NOT authorize a MASK-01 repair.

---

# 1. PURPOSE

Investigate factually why the current RC1 runtime may classify a poor, misleading, semantically incorrect, or otherwise unreliable segmentation result as:

```text
SUCCESS
```

The required order is:

```text
current source audit
→ current SUCCESS/failure contract
→ historical saved-artifact forensics
→ failure taxonomy
→ observable-signal inventory
→ evidence gaps
→ return to ChatGPT
```

Do NOT design or implement a new validity rule in this task.

---

# 2. CRITICAL RESEARCH BOUNDARY

The scientific architecture is frozen.

This task MUST NOT reopen or modify:

```text
MLLM / Qwen ProgramHead
detector configuration
proposal generation strategy
reference scalar rank
reference weighted score
centroid/location heuristic
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
GRF / spatial executor
SAM/SAM2 architecture
target segmentation architecture
training
Task 7J model/seed/threshold selection
```

Specifically forbidden:

```text
new mask-area threshold
new IoU threshold
new confidence threshold
new connected-component threshold
new morphology rule
new heuristic score
new learned validity model
new SAM/SAM2 refinement branch
new reference-selection repair
qualitative-case-specific tuning
```

No product behavior may change in F1.

---

# 3. GIT PRE-FLIGHT — MANDATORY FIRST ACTION

Before any branch switch, branch creation, file write, script execution, or analysis that writes output:

Verify exactly:

```text
current branch =
audit/task8b3-ref01-locked-replay-artifacts

HEAD =
43c24de59625dfb97dcc605435fa83b88be46035
```

Allowed initial `git status --short`:

```text
<empty>
```

or only:

```text
 M handoff/TO_DSH.md
```

or the equivalent staged state for the same single path.

No other modified, deleted, staged, renamed, or untracked path is allowed.

If branch, HEAD, or worktree state is incompatible:

```text
STOP
```

Do NOT:

```text
reset
rebase
amend
stash
clean
checkout away unknown work
delete files
force push
```

Do not attempt to make the repository fit this task.

Print factual STOP information and make no repository mutation beyond the user-supplied `handoff/TO_DSH.md`.

---

# 4. CREATE TASK BRANCH

Only after Section 3 passes, create and switch to:

```text
audit/task8b3-mask01-validity-forensics
```

The branch MUST descend directly from:

```text
43c24de59625dfb97dcc605435fa83b88be46035
```

The pre-existing user modification of:

```text
handoff/TO_DSH.md
```

may follow onto the new branch.

No other starting mutation is authorized.

---

# 5. SOURCE AUDIT SCOPE

Canonical RC1 source:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

Read only what is necessary to recover the exact runtime validity contract.

At minimum inspect:

```text
buildreasonseg/runtime/pipeline.py
```

and the directly referenced source definitions needed to establish:

```text
result/status representation
segmenter configuration
context crop construction
padding representation
context/global coordinate conversion
mask conversion
mask validity handling
failure-return handling
```

Also inspect the relevant existing runtime tests, especially:

```text
tests/test_task8b_runtime.py
```

and only other tests directly referenced or discovered from the relevant symbols.

No source or test file may be edited.

---

# 6. REQUIRED SUCCESS/FAILURE CONTRACT AUDIT

For the production `predict_one` path, reconstruct the actual current control flow.

Record, with source path + function/symbol + line numbers where practical:

1. every pre-segmentation failure return relevant to final mask production;
2. every segmentation-time failure return;
3. every post-segmentation validity gate;
4. the exact condition under which the function finally emits `SUCCESS`;
5. the exact status/reason values used by those paths;
6. which checks occur on context-space masks;
7. which checks occur after conversion to global image space;
8. whether any check currently evaluates semantic target correctness;
9. whether automatic and assisted `reference_id` paths eventually share the same final mask-validity path;
10. what evidence is available at runtime when `SUCCESS` is assigned.

Do not infer behavior merely from names or comments.

Trace the actual current code.

---

# 7. REQUIRED MASK VALIDITY AUDIT

Explicitly recover the current behavior and source of:

```text
mask shape validation
mask pixel count
mask fraction
min_mask_pixels
min_mask_frac
non-padding overlap
context_to_global or equivalent
final global mask
```

For configuration values such as:

```text
min_mask_pixels
min_mask_frac
```

report:

```text
field name
definition location
default/current value if determinable
how value reaches runtime
```

Do NOT modify those values.

Do NOT judge whether they should be increased or decreased.

---

# 8. SPECIFIC FORENSIC LEAD — PADDING GATE

ChatGPT has observed a source-level lead requiring factual resolution:

```text
pipeline.py::_non_padding_mask(...)
```

appears, at the current canonical Git anchor, to construct an all-True mask rather than visibly applying its `padding` argument.

This is NOT yet declared a bug.

DSH must determine, using current source and tests only:

```text
A. What does `padding` mean in the current context representation?

B. Are padded pixels physically present in the array passed to
   `_non_padding_mask`, or were they already removed/represented
   another way?

C. Is `_non_padding_mask` actually capable of distinguishing
   padded from non-padded pixels in the current production path?

D. Can the current `mask_only_in_padding` failure path be reached
   under real current context semantics?

E. What existing tests cover this behavior?

F. Is the evidence sufficient to classify the current padding gate?
```

Use exactly one of these predeclared conclusions:

```text
PADDING_GATE_EFFECTIVE

PADDING_GATE_INEFFECTIVE

PADDING_GATE_NOT_PROVABLE
```

The conclusion MUST be supported by source/test evidence.

Even if:

```text
PADDING_GATE_INEFFECTIVE
```

is established, DO NOT repair it in F1.

---

# 9. TEST-COVERAGE AUDIT

Do NOT run pytest in this task.

Read the existing tests and report which current validity behaviors are covered.

For each relevant test, record:

```text
test name
source file
behavior asserted
whether it exercises SUCCESS or failure
whether it exercises real padding semantics
whether it exercises context→global conversion
```

Do not treat the existence of a test as proof of semantic correctness.

---

# 10. BOUNDED HISTORICAL EVIDENCE DISCOVERY

Historical Task 8B.3 exact output paths are not fully restored.

Therefore discovery MUST be bounded.

First enumerate only versioned Task-8B.3-related material from these roots:

```text
docs/
evaluation/
handoff/
```

Use bounded Git/file-name or text search for terms such as:

```text
task8b3
8B.3
A1
A2
A3
A4
B1
B2
mask
SUCCESS
artifact
output
```

Do NOT recursively scan the entire repository, workspace, model cache, inference tree, logs tree, or artifact tree.

Specifically forbidden:

```text
repo-root recursive filesystem crawl
workspace-root recursive crawl
artifacts/ recursive crawl without an explicit referenced path
logs/ recursive crawl
inference/ recursive crawl
model cache crawl
```

---

# 11. FOLLOWING HISTORICAL ARTIFACT REFERENCES

If a versioned `docs/`, `evaluation/`, or `handoff/` file explicitly names a historical local artifact/file path:

- inspect that exact path if it exists;
- if it names a directory, list immediate children only;
- inspect only files clearly tied to the referenced Task 8B.3 case;
- do not recursively descend unrelated subtrees.

If another explicit subpath is named inside a permitted artifact, that exact subpath may be followed.

Every followed path must be recorded in the final evidence report together with the versioned source that authorized following it.

If historical masks/results cannot be found under this bounded procedure:

```text
HISTORICAL_ARTIFACT_AVAILABILITY =
NOT_FOUND
```

This is a valid forensic result.

Do NOT broaden the search merely to obtain a positive result.

---

# 12. SAVED ARTIFACT ANALYSIS

If saved historical masks/results are found, analyze them READ ONLY.

No model loading and no inference.

Permitted:

```text
read JSON / CSV / TXT / Markdown
read saved PNG masks
read saved NumPy arrays if directly referenced
inspect saved metadata
compute descriptive statistics from an existing saved mask
```

A short one-off read-only Python command/snippet may be used if necessary.

It must not create persistent helper code in the repository.

Do not regenerate missing evidence.

For each historical case for which evidence exists, record only evidence-backed facts such as:

```text
case id
saved status
saved reason
mask existence
mask shape
mask pixel count if derivable
mask fraction if derivable
mask/global bbox if derivable
reference id if saved
target/proposal id if saved
relation if saved
context/padding metadata if saved
manual or GT assessment if already present
source artifact path
```

Do not invent absent fields.

---

# 13. NO POST-HOC METRIC FITTING

Historical artifacts may be used to characterize failures.

They may NOT be used in F1 to fit or search:

```text
thresholds
weights
scalar scores
connected-component cutoffs
area ratios
IoU cutoffs
confidence cutoffs
location priors
case-specific rules
```

Do not evaluate candidate repair rules.

Do not report a “best threshold”.

---

# 14. PREDECLARED FAILURE TAXONOMY

Map observed evidence only into the following categories.

Do not invent new algorithmic categories.

### `TAX_RUNTIME_STRUCTURAL`

Mask/result is invalid under an already-existing structural/runtime contract, for example an explicit current failure condition.

### `TAX_CONTEXT_PADDING_COORDINATE`

Failure or suspicious behavior concerns crop/context/padding/global-coordinate handling.

### `TAX_UPSTREAM_REFERENCE_RELATION`

Final mask unreliability is attributable or plausibly attributable to an upstream reference/relation/target-selection error rather than mask morphology itself.

Do not redesign REF-01.

### `TAX_SEMANTIC_TARGET_MISMATCH`

A technically valid mask exists but saved GT/manual evidence establishes that it corresponds to the wrong semantic target.

### `TAX_RUNTIME_VALID_QUALITY_POOR`

Current runtime checks accept the mask, while available saved evidence shows visibly/quantitatively poor output, without sufficient evidence to label a different semantic target.

### `TAX_UNKNOWN_EVIDENCE_GAP`

Evidence is insufficient to establish one of the above.

A single case may have multiple applicable categories if evidence supports them.

Record the supporting evidence separately.

---

# 15. OBSERVABLE SIGNAL INVENTORY

Create an inventory of signals relevant to future MASK-01 design.

For every signal found, classify availability using exactly one of:

```text
RUNTIME_ALWAYS_AVAILABLE

RUNTIME_CONDITIONALLY_AVAILABLE

PERSISTED_ARTIFACT_AVAILABLE

REQUIRES_GT_OR_MANUAL_REVIEW

NOT_CURRENTLY_AVAILABLE
```

Examples to inspect, without assuming they exist:

```text
mask pixels
mask fraction
mask shape
context dimensions
global dimensions
padding metadata
context/global coordinates
reference id
proposal id
target seed
relation
detector confidence
segmenter metadata
connected components
border contact
GT IoU
manual semantic correctness
```

Do NOT compute or add a signal to product code merely because it appears useful.

---

# 16. REQUIRED DISTINCTION

The report MUST explicitly distinguish:

```text
engineering/runtime validity
```

from:

```text
semantic target correctness
```

and answer:

> Which observed failure types can potentially be detected using information already available at inference time, and which require GT/manual semantic knowledge or upstream reasoning correctness?

This answer is descriptive only.

DSH must NOT decide the final repair.

---

# 17. REQUIRED OUTPUTS

Exactly these repository paths may be created/modified:

```text
docs/task8b3_mask01_f1_validity_forensics.md

evaluation/task8b3_mask01_f1_validity_forensics.json

handoff/FROM_DSH.md

handoff/TO_DSH.md
```

No other repository path may change.

---

# 18. REQUIRED JSON SCHEMA

`evaluation/task8b3_mask01_f1_validity_forensics.json`

must contain, at minimum:

```text
task_id
task_status

git:
  repository
  starting_branch
  starting_head
  task_branch
  final_head
  final_parent
  initial_status
  final_status

source_audit:
  files_read
  source_blob_ids_if_available
  success_return
  failure_paths
  mask_validity_gates
  assisted_and_auto_validity_path_relationship

segmenter_config:
  min_mask_pixels
  min_mask_frac
  provenance

padding_forensics:
  helper
  context_padding_semantics
  current_behavior
  test_evidence
  conclusion

test_coverage:
  relevant_tests

historical_evidence:
  bounded_search_inputs
  followed_explicit_paths
  availability
  cases

failure_taxonomy:
  observations

observable_signals:
  signals

evidence_gaps

forbidden_changes_verified

commands_executed

changed_files

repair_decision

next_gate
```

Use:

```text
repair_decision =
DEFER_TO_CHATGPT
```

Use:

```text
next_gate =
CHATGPT_MASK01_DESIGN_REVIEW
```

Historical artifact availability must be exactly one of:

```text
FOUND
PARTIAL
NOT_FOUND
```

Task status must be exactly one of:

```text
COMPLETE
COMPLETE_WITH_EVIDENCE_GAPS
STOP
FAILED
```

---

# 19. REQUIRED MARKDOWN REPORT

`docs/task8b3_mask01_f1_validity_forensics.md`

must be human-readable and contain:

```text
1. Git identity
2. Scope and prohibitions
3. Exact current SUCCESS contract
4. Exact current failure-return paths
5. Existing mask-validity gates
6. Padding/context forensic result
7. Existing test coverage
8. Bounded historical-artifact search
9. Historical case evidence
10. Failure taxonomy
11. Observable-signal inventory
12. Runtime validity vs semantic correctness
13. Evidence gaps
14. What F1 does NOT establish
15. Return-to-ChatGPT disposition
```

Do not include a proposed threshold or implementation plan.

---

# 20. FROM_DSH.md

Replace/update `handoff/FROM_DSH.md` with the F1 factual report summary.

It must include:

```text
Task
Status

Starting branch
Starting HEAD
Task branch

Final commit
Final parent

Product source changed: YES/NO
Tests changed: YES/NO
Inference executed: YES/NO
Training executed: YES/NO
External RC1 written: YES/NO

Current SUCCESS contract summary

Padding gate conclusion

Historical artifact availability

Failure taxonomy summary

Evidence gaps

Repair decision:
DEFER_TO_CHATGPT

Next gate:
CHATGPT_MASK01_DESIGN_REVIEW
```

Do not declare MASK-01 CLOSED.

Do not declare a repair accepted.

---

# 21. NO TEST / INFERENCE / EXTERNAL SYNC

For F1:

```text
pytest = FORBIDDEN
model inference = FORBIDDEN
training = FORBIDDEN
proposal regeneration = FORBIDDEN
external RC1 sync = FORBIDDEN
external RC1 write = FORBIDDEN
source_manifest regeneration = FORBIDDEN
```

This is an audit/evidence task.

Existing source/tests may be read only.

---

# 22. DIFF GATE

Before staging, run:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
```

Only these paths may appear:

```text
docs/task8b3_mask01_f1_validity_forensics.md
evaluation/task8b3_mask01_f1_validity_forensics.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

If any other path changed:

```text
STOP
```

Do not clean or repair the unexpected path.

---

# 23. COMPLETE / COMPLETE_WITH_EVIDENCE_GAPS

Use `COMPLETE` when:

- all required source facts were recovered;
- padding semantics were resolved sufficiently;
- bounded historical evidence was sufficiently available for the requested forensic characterization;
- all required reports were produced.

Use `COMPLETE_WITH_EVIDENCE_GAPS` when:

- the source audit is complete;
- the bounded search was correctly performed;
- one or more historical artifacts / fields / semantic labels were unavailable;
- the missing evidence is explicitly recorded;
- no search boundary was violated.

Missing historical masks alone is NOT a reason to broaden scope or fail the task.

---

# 24. FAILED

Use `FAILED` only when a tooling/execution problem prevents completion of the authorized audit after a valid preflight.

Do not self-repair product code, environment, dependencies, or repository state.

Record factual failure within the allowed reporting paths if repository safety still permits.

---

# 25. STOP

Use `STOP` for safety/contract violations, including:

```text
wrong starting branch
wrong starting HEAD
unexpected initial worktree mutation
unexpected repository change
task contract conflict
need for an unauthorized write/action
```

If STOP occurs before safe task-branch creation:

- perform no repository mutation;
- do not create a commit merely to report STOP;
- print the factual STOP state;
- wait for ChatGPT.

If STOP occurs after safe task-branch creation and the only changes are authorized report paths, write the factual STOP report only if it can be done without hiding or altering the triggering evidence.

Never reset/rebase/stash/clean to manufacture a successful state.

---

# 26. STAGING / COMMIT / PUSH

For:

```text
COMPLETE
```

or:

```text
COMPLETE_WITH_EVIDENCE_GAPS
```

stage exactly:

```text
docs/task8b3_mask01_f1_validity_forensics.md
evaluation/task8b3_mask01_f1_validity_forensics.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Require:

```text
git diff --cached --name-only
```

to contain exactly those four paths.

Commit exactly once:

```text
git commit -m "docs(rc1): audit mask success validity forensics"
```

Push exactly:

```text
audit/task8b3-mask01-validity-forensics
```

No force push.

After push verify:

```text
local HEAD
remote branch HEAD
parent
git status
```

The working tree must be clean.

Then STOP.

---

# 27. ABSOLUTE PROHIBITIONS

Do NOT:

```text
modify product source
modify tests
modify manifest
modify detector
modify selector
modify segmenter
modify context logic
fix _non_padding_mask
change padding behavior
add threshold
change existing threshold
add validity rule
add failure enum
change result enum
run inference
run training
regenerate proposals
rerun final Demo
sync external RC1
modify external RC1
change dependencies
install packages
refactor
format unrelated files
reset
rebase
amend
stash
clean
force push
execute the next gate
```

If evidence suggests a repair:

```text
record evidence
→ DEFER_TO_CHATGPT
→ do not implement
```

---

# 28. COMPLETE DEFINITION

F1 is successful only when it gives ChatGPT a trustworthy factual basis to decide the next MASK-01 step.

The required terminal disposition is:

```text
SOURCE CONTRACT = RECOVERED

PADDING GATE =
PADDING_GATE_EFFECTIVE
or
PADDING_GATE_INEFFECTIVE
or
PADDING_GATE_NOT_PROVABLE

HISTORICAL ARTIFACT AVAILABILITY =
FOUND
or
PARTIAL
or
NOT_FOUND

NEW MASK VALIDITY RULE =
NONE

PRODUCT CHANGE =
NONE

REPAIR DECISION =
DEFER_TO_CHATGPT

NEXT =
CHATGPT_MASK01_DESIGN_REVIEW
```

Then STOP.
