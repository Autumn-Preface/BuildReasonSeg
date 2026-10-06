# TO_DSH — GOVERNANCE_V1_R1_PROJECT_STATE_CORRECTION

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `docs/governance-v1-executor-neutral-handoff`
> Required starting HEAD: `a61daabdede77062f1156d316fdae5b9db654dbd`
> New task branch: `docs/governance-v1-r1-project-state-correction`

## 0. CHATGPT AUDIT

Governance V1 structure is substantially correct, but final acceptance is blocked by two stale/incorrect fields in:

```text
governance/PROJECT_STATE.yaml
```

### Error 1 — project identity

Current incorrect value:

```yaml
project_title_cn: "基于遥感影像的建筑物提取与变化检测"
```

Required exact value:

```yaml
project_title_cn: "空间推理引导的建筑物结构MLLM相关推理分割方法"
```

### Error 2 — next priority is stale

Current value still says Governance V1 migration is next, but migration has already completed.

Replace it with:

```yaml
next_priority: AWAITING_SUPERVISOR_MILESTONE_SELECTION
next_priority_note: "Governance V1 migration complete; PROP-01 remains open, REF-01 remains residual, 8B.4 and final demo remain not started. Supervisor must authorize the next milestone."
```

Do NOT pre-authorize PROP-01, 8B.4, or Demo.

---

## 1. PURPOSE

Correct only machine-readable project identity / post-migration state before Governance V1 is formally activated.

No scientific, algorithm, product, test, delivery, or executor-policy change is authorized.

---

## 2. GIT PREFLIGHT

Verify:

```text
current branch = docs/governance-v1-executor-neutral-handoff
HEAD = a61daabdede77062f1156d316fdae5b9db654dbd
```

Create:

```text
docs/governance-v1-r1-project-state-correction
```

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED PATHS

Only:

```text
governance/PROJECT_STATE.yaml
docs/governance_v1_r1_project_state_correction.md
evaluation/governance_v1_r1_project_state_correction.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do not modify:

```text
AGENTS.md
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
delivery_src/**
scripts/**
tests/**
external RC1
```

---

## 4. REQUIRED PROJECT_STATE CHANGES

Change exactly these logical fields:

```yaml
project_title_cn: "空间推理引导的建筑物结构MLLM相关推理分割方法"

next_priority: AWAITING_SUPERVISOR_MILESTONE_SELECTION

next_priority_note: "Governance V1 migration complete; PROP-01 remains open, REF-01 remains residual, 8B.4 and final demo remain not started. Supervisor must authorize the next milestone."
```

Preserve all other project-state facts.

In particular preserve:

```text
MASK-01 = CLOSED_ENGINEERING_HARDENING
PROP-01 = OPEN_ENGINEERING_DEFECT
REF-01 = ACTIVE_RESIDUAL_SELECTION_LIMITATION
preferred_executor = CODEX
fallback_executor = DSH
source_manifest_entries = 135
external_full_suite = 130/130 PASS
```

---

## 5. VALIDATION

Run only governance validation.

### A. YAML parse

`governance/PROJECT_STATE.yaml` must parse successfully.

### B. Exact field validation

Required:

```text
project_title_cn ==
空间推理引导的建筑物结构MLLM相关推理分割方法

next_priority ==
AWAITING_SUPERVISOR_MILESTONE_SELECTION
```

And the note must explicitly state that the Supervisor authorizes the next milestone.

### C. Frozen-state consistency

Confirm:

```text
MASK-01 closed
PROP-01 open
REF-01 residual
8B.4 not started
final demo not started
```

### D. Diff gate

No paths outside the five authorized paths may change.

No pytest, inference, training, sync, or external write.

---

## 6. REPORT / EVIDENCE

Create:

```text
docs/governance_v1_r1_project_state_correction.md
evaluation/governance_v1_r1_project_state_correction.json
```

Update legacy:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Record:

```text
task_id = GOVERNANCE_V1_R1_PROJECT_STATE_CORRECTION
status
starting branch/head
task branch

project_title_corrected = true/false
next_priority_corrected = true/false
yaml_parse_pass = true/false

scientific_state_changed = false
product_paths_changed = false
current_task_changed = false
executor_state_changed = false
external_write = false

next_gate = CHATGPT_GOVERNANCE_V1_R1_REMOTE_AUDIT
```

---

## 7. COMMIT / PUSH

If validation passes:

```text
git commit -m "docs(governance): correct project state identity"
```

Push:

```text
docs/governance-v1-r1-project-state-correction
```

All COMPLETE / STOP / FAILED outcomes must be pushed.

Then STOP.

---

## 8. SUCCESS DEFINITION

```text
GOVERNANCE_V1_R1_PROJECT_STATE_CORRECTION = COMPLETE

project identity corrected
post-migration next_priority corrected
YAML valid
scientific state unchanged
product tree unchanged
CURRENT_TASK / EXECUTOR_STATE unchanged

NEXT = CHATGPT_GOVERNANCE_V1_R1_REMOTE_AUDIT
```

Then STOP.
