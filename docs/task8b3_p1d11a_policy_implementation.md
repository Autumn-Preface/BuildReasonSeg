# Task 8B.3-P1D11A — Supported-Domain Policy Implementation (STOP, docs only)

## 1. Task and scope

The task authorises canonical policy documentation plus a narrow two-entry manifest update, **gated** on a pre-edit
canonical manifest self-check of exactly 135/135 PASS. That gate failed, so per the task book **no README, model-card
or manifest edit was made** and the task stops after documentation.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = d414c33762968ac4e6ea441f334082fc43adc0d0
pre-edit canonical manifest check = FAIL (4 of 135 entries stale)
```

## 2. Pre-edit manifest self-check details

| entry | manifest bytes | actual bytes | manifest sha256 | actual sha256 |
|---|---:|---:|---|---|
| `buildreasonseg/runtime/core.py` | 11064 | 11310 | `37c38ba4acd1bb0f…` | `8913071d578dbfe6…` |
| `buildreasonseg/runtime/detector.py` | 20300 | 20772 | `82531dc3b758cd8a…` | `a6fa4bdd76db6f50…` |
| `buildreasonseg/runtime/outputs.py` | 9166 | 9367 | `f12f86cfcc04339b…` | `9c0e3a59c317d801…` |
| `tests/test_task8b_runtime.py` | 24983 | 25552 | `6cac7e9320f20888…` | `02fef130459f2bce…` |

Entry count is still 135 and no manifest path is missing; the four mismatching entries are the compact-proposal
runtime/test files whose recorded identity no longer matches the files in the canonical tree.

## 3. Actions taken

```text
README.md / docs/model_card.md edits        = NOT PERFORMED (blocked by the gate)
source_manifest.json edits                  = NOT PERFORMED
docs/task8b3_p1d11_supported_domain_policy.md = NOT CREATED
model runs / candidate runs / external sync = NONE / NONE / NONE
```

The gate exists to prevent policy wording from being committed on top of an inconsistent canonical manifest; per the
task book no stale entry was "repaired" here.

## 4. Status fields

```text
supported-domain policy   = NOT IMPLEMENTED (blocked)
A2 policy status          = unchanged (P1D10-R1 text stands: DOCUMENTED_PERSISTENT_NON_DETECTION in that report)
canonical README policy   = NOT UPDATED
canonical model card policy = NOT UPDATED
next gate                 = canonical manifest reconciliation (new task book required)
```

## 5. STOP reason

Pre-edit canonical manifest self-check is not 135/135: four manifest-listed canonical files
(`buildreasonseg/runtime/detector.py`, `buildreasonseg/runtime/core.py`, `buildreasonseg/runtime/outputs.py`,
`tests/test_task8b_runtime.py`) fail their recorded byte/SHA256 identity. The task book forbids editing the policy
documents or the manifest in that state and forbids repairing unrelated entries in this task.
