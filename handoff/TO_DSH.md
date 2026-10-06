# TO_DSH — GOVERNANCE_V1_EXECUTOR_NEUTRAL_HANDOFF

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH for this migration task only
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `delivery/task8b3-mask01-d2-external-sync-regression`
> Required starting HEAD: `1e9ac792d901b44db8ed33a09ee1552e11a865d9`
> New task branch: `docs/governance-v1-executor-neutral-handoff`

## 0. CHATGPT FINAL D2 DECISION

D2 is ACCEPTED / CLOSED.

Accepted remote state:

```text
branch = delivery/task8b3-mask01-d2-external-sync-regression
HEAD   = 1e9ac792d901b44db8ed33a09ee1552e11a865d9
parent = 8ab53f3df2664a9ad8f9f6bc8b5659a2251b0e03
commit = test(rc1): verify external mask01 delivery closure
```

Accepted evidence:

```text
pre-sync source identity = 127/135 match
controlled sync          = 135 copied / 135 verified / 0 failures
post-sync identity       = 135/135 match
preserved assets         = unchanged
check_setup.py           = READY
MASK targeted contracts  = 7/7 PASS
external full suite      = 130 passed / 0 failed / 0 errors
real inference           = not run
training                 = not run
parameter tuning         = not run
```

Formal closure:

```text
SUCCESS_SEMANTICS_HARDENING_V1 = IMPLEMENTED_EXTERNAL_RC1
MASK01_ENGINEERING_HARDENING_CHAIN = CLOSED

runtime SUCCESS scope = RUNTIME_STRUCTURAL_ONLY
semantic status = NOT_EVALUATED
semantic target correctness = NOT ESTABLISHED by runtime SUCCESS
```

This task does NOT resume PROP-01 or 8B.4.

Its only purpose is to modernize project governance so Codex and DSH can hand work off through repository state instead of conversation-specific context.

---

## 1. MIGRATION PRINCIPLE

Create an executor-neutral repository protocol.

After this task, the project must conceptually follow:

```text
ChatGPT Chat
= supervisor / scientific authority / milestone auditor

Codex
= preferred high-capability executor for long engineering milestones

DSH
= fallback executor / alternate executor

GitHub + governance files
= source of truth for handoff state
```

No executor owns unique project state.

A new executor must be able to continue by reading repository state without needing the previous executor's chat history.

---

## 2. NON-DESTRUCTIVE MIGRATION

This migration is additive.

DO NOT delete, rename, or rewrite history for:

```text
handoff/TO_DSH.md
handoff/FROM_DSH.md
```

They remain historical compatibility artifacts.

New canonical executor-neutral handoff files are:

```text
AGENTS.md
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
```

Future tasks may stop using TO_DSH/FROM_DSH after ChatGPT explicitly freezes the new protocol.

---

## 3. ALLOWED PATHS

Only create/modify:

```text
AGENTS.md
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml

docs/governance_v1_executor_neutral_handoff.md
evaluation/governance_v1_executor_neutral_handoff.json

handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No product source, tests, canonical RC1 files, model assets, scientific reports, sync script, or external RC1 may be modified.

---

## 4. AGENTS.md — REQUIRED CONTRACT

Create repository-root:

```text
AGENTS.md
```

It must be executor-neutral and apply to Codex, DSH, or any future coding agent.

Required sections:

### 4.1 Authority model

State:

```text
Supervisor = ChatGPT project conversation
Executor = Codex / DSH / future agent
Repository = source of truth
```

Scientific/architecture decisions belong to Supervisor.

Executor may make implementation-level engineering choices only within frozen constraints.

### 4.2 Decision levels

Define exactly:

```text
L0 = execution mechanics
     executor may decide

L1 = local engineering implementation
     executor may decide if contract/behavior remains frozen

L2 = interface / contract / acceptance-rule change
     supervisor approval required

L3 = algorithm / threshold / ranking / model-selection change
     supervisor approval required

L4 = architecture / frozen scientific conclusion change
     supervisor approval required
```

For L2-L4:
- STOP at a recoverable checkpoint;
- persist evidence;
- update EXECUTOR_STATE;
- do not self-authorize.

### 4.3 Git safety

Explicitly forbid by default:

```text
git reset --hard
git rebase
git commit --amend
git stash
git clean
force push
history rewrite
```

Unknown files:
- do not delete;
- do not discard;
- do not stage unless authorized.

All task outcomes COMPLETE / STOP / FAILED must be persisted to GitHub.

### 4.4 Scientific governance

Freeze:

```text
Qwen 2B ProgramHead
YOLO26m proposal path
deterministic spatial executor
SAM/SAM2 visual path
GRF
target-aware segmentation
Reference -> Relation -> Target -> Segmentation
```

State project-specific prohibition without Supervisor approval:

```text
threshold tuning
scalar fitting
locked-case rescue heuristics
new ranking heuristic
new model branch
architecture rescue
post-hoc semantic-quality threshold repair
```

### 4.5 Evidence semantics

Freeze:

```text
test PASS != scientific correctness
runtime SUCCESS != semantic target correctness
FROM_EXECUTOR-style report != independent proof
```

Review source/diff/tests/evidence independently.

### 4.6 Author-review separation

State:

```text
Implementing agent cannot be the sole final approver of its own change.
```

For long Codex milestones:
- implementer may run self-checks;
- an independent reviewer agent is preferred;
- final milestone acceptance remains Supervisor.

### 4.7 Checkpoint protocol

Require recoverable checkpoints for long tasks.

At meaningful stages:
- commit + push when safe;
- update EXECUTOR_STATE;
- do not leave sole state only inside agent conversation.

### 4.8 Required startup read order

Every executor must first read:

```text
AGENTS.md
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
```

then inspect actual Git branch/HEAD/diff before work.

Actual source/Git state overrides stale prose.

---

## 5. PROJECT_STATE.yaml — REQUIRED CONTENT

Create:

```text
governance/PROJECT_STATE.yaml
```

Use valid YAML.

It must be concise and machine-readable.

Required top-level structure:

```yaml
schema_version: 1
project: BuildReasonSeg
project_title_cn: ...
phase: RC1_ENGINEERING_RELIABILITY
supervisor: CHATGPT_PROJECT_CONVERSATION
preferred_executor: CODEX
fallback_executor: DSH

canonical:
  repo: Autumn-Preface/BuildReasonSeg
  canonical_root: delivery_src/BuildReasonSeg_Advisor_RC1
  external_root: C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
  source_manifest_entries: 135
  source_identity_basis: GIT_CANONICAL_BLOB_BYTES

scientific_architecture:
  frozen: true
  components: [...]

defects:
  MEM-01: ...
  PROP-01: ...
  REF-01: ...
  MASK-01: ...

milestones:
  mask01_d2: ...
  8B.4: ...
  final_demo: ...

next_priority: ...

executor_protocol:
  current_task_file: handoff/CURRENT_TASK.md
  executor_state_file: handoff/EXECUTOR_STATE.yaml
  legacy_handoff_files: [...]
```

Freeze current defect states accurately:

```text
MEM-01 = CLOSED

PROP-01 =
OPEN_ENGINEERING_DEFECT
A2 zero proposals
blind threshold tuning prohibited

REF-01 =
ACTIVE_RESIDUAL_SELECTION_LIMITATION
right = stable
above = repaired
left = residual
below = residual
further scalar rank repair rejected

MASK-01 =
CLOSED_ENGINEERING_HARDENING
SUCCESS scope = RUNTIME_STRUCTURAL_ONLY
semantic status = NOT_EVALUATED
semantic correctness not established
external full suite = 130/130 PASS
```

Current next project priority:

```text
GOVERNANCE_V1 migration first
then ChatGPT chooses between PROP-01 continuation / 8B.4 sequencing
```

Do NOT invent that PROP-01 is solved.

---

## 6. DECISIONS.md — REQUIRED CONTENT

Create:

```text
governance/DECISIONS.md
```

This is NOT a conversation transcript.

It is a compact ledger of frozen decisions.

Each entry must contain:

```text
Decision ID
Status
Decision
Rationale / evidence summary
What is prohibited
Reopen condition
```

At minimum include:

### GOV-D001
```text
SCIENTIFIC_ARCHITECTURE_FROZEN
```

### GOV-D002
```text
REJECT_FURTHER_SCALAR_RANK_REPAIR
```

### GOV-D003
```text
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
```

### GOV-D004
```text
REJECT_THRESHOLD_BASED_MASK_VALIDITY_REPAIR
```

### GOV-D005
```text
SUCCESS_SEMANTICS_HARDENING_V1
status = IMPLEMENTED_EXTERNAL_RC1
```

### GOV-D006
```text
INSPECT_IMAGE_PREFLIGHT_BEFORE_MODEL_V1
status = IMPLEMENTED
```

### GOV-D007
```text
LIGHTWEIGHT_CANONICAL_VS_EXTERNAL_COMPLETE_DELIVERY
```

Decision:
- canonical source tree is lightweight source/config snapshot;
- complete-delivery full suite belongs on external RC1;
- sync is one-way manifest-listed source/config;
- preserved external weights/assets/log fixtures must not be overwritten by sync.

### GOV-D008
```text
EXECUTOR_NEUTRAL_HANDOFF_V1
```

Decision:
- Chat remains Supervisor;
- Codex preferred executor;
- DSH fallback executor;
- repo state is the handoff source of truth.

Do not fabricate exact historical metrics not needed by a decision.

---

## 7. CURRENT_TASK.md — INITIAL NEUTRAL STATE

Create:

```text
handoff/CURRENT_TASK.md
```

This file defines the Supervisor-authorized current task, not executor-specific instructions.

For this migration commit, after migration is complete, set it to a PAUSED / AWAITING_SUPERVISOR state.

Required content:

```text
Task ID: NONE
Status: AWAITING_SUPERVISOR
Authorized executor: ANY
Starting branch/head: <governance migration final state is filled as POST_COMMIT_EXTERNAL_FACT or omitted>
Goal:
  No engineering task is currently authorized after Governance V1 migration.

Next:
  ChatGPT Supervisor selects the next milestone.

Forbidden while no task is active:
  product edits
  tests edits
  algorithm changes
  external RC1 writes
```

Do NOT pre-authorize PROP-01 automatically.

---

## 8. EXECUTOR_STATE.yaml — INITIAL NEUTRAL STATE

Create valid YAML:

```text
handoff/EXECUTOR_STATE.yaml
```

Required fields:

```yaml
schema_version: 1
task_id: NONE
status: AWAITING_SUPERVISOR
executor: NONE
branch: docs/governance-v1-executor-neutral-handoff
head: POST_COMMIT_EXTERNAL_FACT

completed: []
current_step: null
next_action: null

hypotheses: []
tests_run: []
files_modified: []

uncommitted_changes: false

blocked_on_supervisor: false
block_reason: null

decision_level_required: null

last_checkpoint:
  commit: POST_COMMIT_EXTERNAL_FACT
  pushed: true
```

The final commit SHA cannot be known before commit; use `POST_COMMIT_EXTERNAL_FACT`.

Future executors update this file at recoverable checkpoints.

---

## 9. GOVERNANCE DOCUMENTATION

Create:

```text
docs/governance_v1_executor_neutral_handoff.md
```

Document:
- why migration was needed;
- old human-relay bottleneck;
- Chat / Codex / DSH roles;
- five canonical governance/handoff files;
- executor switch procedure;
- quota interruption procedure;
- checkpoint procedure;
- legacy TO_DSH/FROM_DSH compatibility status.

Keep it concise and operational.

---

## 10. VALIDATION

No pytest/model inference/training.

Run only governance validation.

Required:

### A. YAML parse

Parse:

```text
governance/PROJECT_STATE.yaml
handoff/EXECUTOR_STATE.yaml
```

with Python/PyYAML.

Required:
```text
PASS
```

### B. Required-file presence

Confirm all five canonical files exist:

```text
AGENTS.md
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
```

### C. No product changes

Verify Git diff contains no paths under:

```text
delivery_src/
scripts/
```

and no model/config/test source changes.

### D. State consistency

Programmatically or manually verify:

```text
MASK-01 = closed
PROP-01 = open
REF-01 = residual limitation
current task = NONE / awaiting supervisor
preferred executor = CODEX
fallback executor = DSH
```

Do not change scientific status.

---

## 11. REQUIRED EVIDENCE

Create:

```text
evaluation/governance_v1_executor_neutral_handoff.json
```

Record:

```text
task_id
status
starting_branch
starting_head
task_branch

created_files
yaml_parse_pass
required_files_present
product_paths_changed = false
scientific_state_changed = false
legacy_handoff_deleted = false

preferred_executor = CODEX
fallback_executor = DSH
supervisor = CHATGPT_PROJECT_CONVERSATION

current_task_status = AWAITING_SUPERVISOR

next_gate = CHATGPT_GOVERNANCE_V1_REMOTE_AUDIT
```

Also update legacy:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

only to record this migration result / task text.

---

## 12. DIFF GATE

Only stage:

```text
AGENTS.md
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
docs/governance_v1_executor_neutral_handoff.md
evaluation/governance_v1_executor_neutral_handoff.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Anything else:
- do not stage;
- do not delete;
- report.

---

## 13. COMMIT / PUSH

If validation passes, commit exactly:

```text
git commit -m "docs(governance): add executor-neutral handoff protocol"
```

Push:

```text
docs/governance-v1-executor-neutral-handoff
```

No force push.

All outcomes COMPLETE / STOP / FAILED must be pushed.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

Do not begin PROP-01, 8B.4, demo work, or any Codex migration execution.

---

## 14. SUCCESS DEFINITION

```text
GOVERNANCE_V1_EXECUTOR_NEUTRAL_HANDOFF = COMPLETE

five canonical executor-neutral governance/handoff files created
YAML parses
scientific state preserved
product tree unchanged
legacy TO_DSH/FROM_DSH retained
current task = AWAITING_SUPERVISOR

Codex and DSH can both resume future work from repository state

NEXT = CHATGPT_GOVERNANCE_V1_REMOTE_AUDIT
```

Then STOP.
