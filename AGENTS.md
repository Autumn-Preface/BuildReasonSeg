# AGENTS.md — Executor-Neutral Repository Contract

This contract applies to any coding agent working on this repository: **Codex**, **DSH**, or a future executor.

## 1. Authority model

```text
Supervisor = ChatGPT project conversation
Executor   = Codex / DSH / future agent
Repository = source of truth
```

Scientific and architecture decisions belong to the Supervisor. The Executor may make implementation-level engineering
choices only inside frozen constraints. Repository state (Git branch, HEAD, diff, tests, evidence) overrides any prose.

## 2. Decision levels

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

For L2-L4 the executor must: STOP at a recoverable checkpoint; persist evidence; update `handoff/EXECUTOR_STATE.yaml`;
and must not self-authorize.

## 3. Git safety

Forbidden by default:

```text
git reset --hard
git rebase
git commit --amend
git stash
git clean
force push
history rewrite
```

Unknown files must not be deleted, discarded, or staged unless authorized.

Every task outcome (COMPLETE / STOP / FAILED) must be persisted to GitHub.

## 4. Scientific governance

Frozen architecture:

```text
Qwen 2B ProgramHead
YOLO26m proposal path
deterministic spatial executor
SAM/SAM2 visual path
GRF
target-aware segmentation
Reference -> Relation -> Target -> Segmentation
```

Prohibited without Supervisor approval:

```text
threshold tuning
scalar fitting
locked-case rescue heuristics
new ranking heuristic
new model branch
architecture rescue
post-hoc semantic-quality threshold repair
```

## 5. Evidence semantics

```text
test PASS != scientific correctness
runtime SUCCESS != semantic target correctness
FROM_EXECUTOR-style report != independent proof
```

Review source, diff, tests and evidence independently.

## 6. Author-review separation

```text
Implementing agent cannot be the sole final approver of its own change.
```

For long milestones the implementer may run self-checks, an independent reviewer agent is preferred, and final milestone
acceptance remains with the Supervisor.

## 7. Checkpoint protocol

At meaningful stages of a long task: commit and push when safe, update `handoff/EXECUTOR_STATE.yaml`, and never leave the
only copy of state inside an agent conversation.

## 8. Required startup read order

Every executor must first read:

```text
AGENTS.md
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
```

and then inspect the actual Git branch / HEAD / diff before working. Actual source and Git state override stale prose.

## 9. Git Authority and Idle Handoff Semantics (Governance V1.1)

Actual Git branch / HEAD / diff are authoritative:

```text
actual git branch / HEAD / diff
> stale branch/head metadata in handoff files
```

`EXECUTOR_STATE.branch`, `EXECUTOR_STATE.head`, and `last_checkpoint.commit` are
observational/checkpoint metadata, not an alternate Git authority. They may be `null`
when there is no meaningful active checkpoint. CURRENT_TASK Git fields likewise describe
task/checkpoint metadata only.

When `task_id = NONE` and `status = AWAITING_SUPERVISOR`, use the neutral idle state:

```yaml
branch: null
head: null
last_checkpoint:
  commit: null
  pushed: null
```

Repository anchor is not persisted while idle. Every Executor MUST resolve the actual
branch and HEAD from Git at startup. Do not persist fabricated self-referential SHA
placeholders as if they were real Git facts. Report actual local and remote final HEAD
from Git after push; do not create an extra commit to write its own final SHA into itself.
