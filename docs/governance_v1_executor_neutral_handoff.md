# Governance V1 — Executor-Neutral Handoff

## 1. Task record

```text
task_id = GOVERNANCE_V1_EXECUTOR_NEUTRAL_HANDOFF
status  = COMPLETE
starting branch/head = delivery/task8b3-mask01-d2-external-sync-regression / 1e9ac792d901b44db8ed33a09ee1552e11a865d9
task branch = docs/governance-v1-executor-neutral-handoff
migration type = governance / handoff protocol (no algorithm or product change)
```

## 2. Files created

```text
AGENTS.md
governance/PROJECT_STATE.yaml
governance/DECISIONS.md
handoff/CURRENT_TASK.md
handoff/EXECUTOR_STATE.yaml
```

```text
handoff/TO_DSH.md and handoff/FROM_DSH.md = kept (not deleted, not renamed) = {'handoff/TO_DSH.md': True, 'handoff/FROM_DSH.md': True}
```

## 3. Validation

```text
A. YAML parse pass            = True
B. required files present     = True
C. no product changes         = True (touched product paths: NONE)
D. state consistency          = True
E. diff gate                  = staged paths all inside the allowed set (6 staged)
```

```text
state consistency detail:
{
 "current_task_status_AWAITING_SUPERVISOR": true,
 "executor_state_status_AWAITING_SUPERVISOR": true,
 "project_state_points_to_current_task": true,
 "project_state_points_to_executor_state": true,
 "legacy_handoff_listed": true,
 "MASK01_closed": true,
 "PROP01_open": true,
 "REF01_residual": true,
 "no_auto_authorized_task": true,
 "no_prop01_preauthorization": true,
 "executor_neutral": true
}
```

## 4. Scientific state (unchanged)

```text
MEM-01 = CLOSED
PROP-01 = OPEN_ENGINEERING_DEFECT (A2 zero proposals; blind threshold tuning prohibited)
REF-01  = ACTIVE_RESIDUAL_SELECTION_LIMITATION (right stable, above repaired, left/below residual;
          further scalar rank repair rejected)
MASK-01 = CLOSED_ENGINEERING_HARDENING (SUCCESS = RUNTIME_STRUCTURAL_ONLY; semantic = NOT_EVALUATED;
          external full suite = 130/130 PASS)
no task auto-started = true (PROP-01 / 8B.4 / Demo not started)
```

## 5. Handoff neutrality

```text
Codex and DSH can resume from repository state alone: AGENTS.md + governance/PROJECT_STATE.yaml +
governance/DECISIONS.md + handoff/CURRENT_TASK.md + handoff/EXECUTOR_STATE.yaml
chat-history dependency = none
CURRENT_TASK.md status = AWAITING_SUPERVISOR (authorized executor = ANY)
```

## 6. Persistence

```text
github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_GOVERNANCE_V1_REMOTE_AUDIT
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```
