# CURRENT_TASK.md — Supervisor-Authorized Current Task

This file defines the Supervisor-authorized current task. It is executor-neutral and must not contain
executor-specific instructions.

```text
Task ID: NONE
Status: AWAITING_SUPERVISOR
Authorized executor: ANY
```

Repository anchor is not persisted while idle.
Every Executor MUST resolve the actual branch and HEAD from Git at startup.

## Goal

```text
No engineering task is currently authorized.
```

## Next

```text
ChatGPT Supervisor performs CHATGPT_GOVERNANCE_V1_1_REMOTE_AUDIT and selects the next milestone.
```

## Forbidden while no task is active

```text
product edits
test edits
algorithm changes
external RC1 writes
automatic start of PROP-01 / 8B.4 / final demo
```
