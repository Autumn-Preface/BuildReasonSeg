# Task 8B.3-P1D11A-R1 — Supported-Domain Policy (STOP: mixed-convention canonical manifest)

## 1. Task and scope

The task asks to re-verify the canonical manifest against Git canonical/index bytes, confirm the previous STOP was a
CRLF false positive, and then implement the supported-domain policy. The re-verification succeeded **only for the four
previously flagged entries**; the full 135-entry manifest cannot be verified under a single byte convention, so the
policy implementation remains blocked and the task stops.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 9f91d953b9ef5bb3993be4b7434b174de9856648
policy documents edited = NONE
external write sync    = NONE
model / test runs      = NONE
```

## 2. Confirmed part — the four previously flagged entries are a CRLF false positive

| entry | manifest bytes / sha256 | Git index bytes / sha256 | match | disk bytes / sha256 | match |
|---|---|---|---|---|---|
| `buildreasonseg/runtime/core.py` | 11064 / `37c38ba4acd1…` | 11064 / `37c38ba4acd1…` | YES | 11310 / `8913071d578d…` | NO |
| `buildreasonseg/runtime/detector.py` | 20300 / `82531dc3b758…` | 20300 / `82531dc3b758…` | YES | 20772 / `a6fa4bdd76db…` | NO |
| `buildreasonseg/runtime/outputs.py` | 9166 / `f12f86cfcc04…` | 9166 / `f12f86cfcc04…` | YES | 9367 / `9c0e3a59c317…` | NO |
| `tests/test_task8b_runtime.py` | 24983 / `6cac7e9320f2…` | 24983 / `6cac7e9320f2…` | YES | 25552 / `02fef130459f…` | NO |

So the previous STOP was indeed a line-ending/checkout artifact **for those four entries**.

## 3. New finding — the manifest mixes two byte conventions

Verifying all 135 entries under both conventions (Git index blob vs working-tree bytes):

```text
match index AND disk      = 93
match Git index only      = 4
match disk only           = 38
match neither             = 0
```

Examples of index-only entries: `buildreasonseg/runtime/core.py`, `buildreasonseg/runtime/detector.py`, `buildreasonseg/runtime/outputs.py`, `tests/test_task8b_runtime.py`
Examples of disk-only entries: `README.md`, `buildreasonseg/runtime/_frozen/PORT_PROVENANCE.md`, `buildreasonseg/runtime/_frozen/__init__.py`, `buildreasonseg/runtime/_frozen/mvp/geometric_relation_field_v02.py`

Consequences: the four runtime/test entries were recorded from **Git index (LF) bytes**, while most other entries were
recorded from **working-tree (CRLF) bytes**. No single byte convention can therefore satisfy the mandated
"exactly 135/135 PASS" gate, and the manifest is internally inconsistent about line endings rather than demonstrating
any functional file change.

## 4. Actions not taken

```text
docs/task8b3_p1d11_supported_domain_policy.md   = NOT CREATED
delivery_src/.../README.md                      = NOT MODIFIED
delivery_src/.../docs/model_card.md             = NOT MODIFIED
delivery_src/.../source_manifest.json           = NOT MODIFIED
```

The task book forbids policy/manifest edits while the pre-edit self-check is not 135/135, and forbids repairing
unrelated manifest entries inside this task.

## 5. STOP reason and required follow-up

The canonical manifest cannot be verified 135/135 under either the Git-index convention or the working-tree
convention because it mixes both. A dedicated manifest-normalisation task (choose one convention — recommended: Git
index/LF blob bytes, i.e. `.gitattributes`-independent canonical content — and refresh all 135 entries, then re-run
the self-check) is required before the supported-domain policy can be implemented.
