# TO_DSH — Task 8B.3-P1D11B: Controlled Supported-Domain Policy Sync to External RC1

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `f9bf6be13035445f45f956c2b6a5a64ccf7e6032`
> Canonical RC1: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Frozen entering state

ChatGPT approves P1D11A-R3.

Freeze:

```text
supported-domain policy =
IMPLEMENTED_IN_CANONICAL_RC1_DOCS_CORRECTED

source-manifest identity basis =
GIT_CANONICAL_BLOB_BYTES

canonical manifest =
135 entries

canonical manifest integrity =
135/135 PASS

locked candidates =
FINAL_METADATA_LOCK

locked candidate runtime status =
NOT YET RUN

A2 provenance/domain =
NOT ESTABLISHED

A2 policy =
DOCUMENTED_PERSISTENT_NON_DETECTION

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT
```

This task performs only the controlled canonical → external policy/document synchronization and verifies external
delivery integrity.

No locked candidate may run in this task.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = f9bf6be13035445f45f956c2b6a5a64ccf7e6032
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify canonical README/model_card/policy/runtime/tests/config/checkpoints;
- modify `source_manifest.json`;
- modify model binaries/checkpoints in external delivery;
- delete any external file or directory;
- run pytest;
- run `predict.py`;
- run detector/Qwen/SAM2/D-B1 inference;
- copy/run/inspect the four locked Demo candidates;
- modify or replace A1/A2/A3/A4/B1/B2;
- change thresholds/models/tiling/merge/reference policies;
- modify user/runtime outputs under `inference/input`, `inference/output`, `logs`, `runs`, `datasets`;
- fix REF-01 or MASK-01;
- enter Task 8B.4 / 8C;
- update main;
- force push.

`check_setup.py` is authorized before and after sync.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d11b_supported_domain_policy_external_sync.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

External delivery is intentionally modified by the controlled sync defined below.

# 4. Canonical manifest preflight — Git identity

Read:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
```

For all 135 entries, verify manifest `bytes/sha256` against exact binary Git canonical bytes:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<path>
```

Require:

```text
canonical Git identity = 135/135 PASS
missing = 0
duplicates = 0
```

Also record SHA256 of the Git canonical `source_manifest.json` blob itself:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

If any gate fails:
- do not write external;
- record STOP;
- commit/push report/handoff;
- STOP.

# 5. Pre-sync external controlled check

Run exactly:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

This helper compares actual canonical working-tree files to actual external files for the 135 manifest-listed paths.

Expected pre-sync state:

```text
checked = 135
match = 133
missing = 0
mismatch = 2
```

Expected mismatches exactly:

```text
README.md
docs/model_card.md
```

Rationale:
- P1D11A changed only those two manifest-listed delivery files;
- `source_manifest.json` is not itself one of the 135 entries.

If pre-sync result differs:
- do not sync;
- record exact unexpected paths/counts;
- STOP.

# 6. Pre-sync external setup readiness

Use the existing RC1 environment Python:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe
```

Run from external RC1 root:

```text
python check_setup.py
```

Require:

```text
exit code = 0
BuildReasonSeg environment: READY
```

If not READY:
- do not sync;
- STOP.

No inference is authorized.

# 7. Protected external asset snapshot

Before sync record existence, bytes and SHA256 for:

```text
model/buildreasonseg_advisor/decoder.pt
model/buildreasonseg_advisor/detector.pt
model/components/sam2/sam2.1_hiera_base_plus.pt
model/components/sam2/sam2.1_hiera_b+.yaml
model/components/program_head/program_parser_l3_rehearsal_v1.pt
```

Also record for:

```text
model/components/program_head/Qwen3-VL-2B-Instruct
```

at minimum:

```text
exists
regular file count
sorted file names
total bytes
```

Record existence/state only for protected runtime/user directories:

```text
inference/input
inference/output
logs
runs
datasets
```

Do not modify them.

# 8. Controlled 135-file sync

Run exactly once:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Require:

```text
copied = 135
verified = 135
failures = 0
```

The helper may overwrite all 135 listed source/config targets even when 133 were already identical.

It must not touch:
- model binaries;
- downloaded Qwen assets;
- input/output/log/run/dataset content;
- unlisted external files.

Any sync failure:
- STOP;
- do not rerun automatically.

# 9. Synchronize source_manifest.json separately

`source_manifest.json` is intentionally not self-listed among the 135 entries.

After the 135-file helper sync succeeds, update exactly:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json
```

using the exact **Git canonical blob bytes** from:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Important:
- binary capture/write only;
- do not pass through text newline conversion;
- do not regenerate JSON;
- do not use the CRLF-expanded working-tree file as the authoritative bytes.

Require:

```text
external source_manifest SHA256
==
Git canonical source_manifest SHA256
```

Interpretation:

```text
external source_manifest records Git canonical source identities.
It is provenance metadata and is NOT a claim that CRLF-expanded external working files byte-match those per-entry Git identities.
Actual external delivery equality is checked by the sync helper directly against canonical working-tree files.
```

# 10. Post-sync external 135-file gate

Run exactly once:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

Require:

```text
checked = 135
match = 135
missing = 0
mismatch = 0
exit code = 0
```

Also require:

```text
external source_manifest Git-canonical SHA = MATCH
```

# 11. Policy-content spot check in external delivery

Read only the synchronized external:

```text
README.md
docs/model_card.md
source_manifest.json
```

Require:

README:
```text
contains "软件接受的输入模态"
does NOT contain "正式输入域"
contains "已验证数据域与 Demo 边界"
contains A2 provenance/domain NOT ESTABLISHED meaning
contains locked candidates NOT YET RUN meaning
```

Model card:
```text
contains "软件输入模态"
contains separate "已验证域"
contains A2 NOT ESTABLISHED meaning
contains Demo policy
```

External source manifest:
```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
```

Do not inspect the locked candidate images.

# 12. Protected asset post-sync comparison

Recompute the §7 protected snapshot.

Require:

```text
all protected file SHA256/bytes unchanged = YES
Qwen directory file names/count/total bytes unchanged = YES
protected runtime/user directory content was not intentionally modified = YES
```

If a protected asset changed:
- STOP immediately;
- record exact fact;
- do not attempt repair without ChatGPT.

# 13. Post-sync external setup readiness

Run external:

```text
python check_setup.py
```

with the same environment Python as §6.

Require:

```text
exit code = 0
BuildReasonSeg environment: READY
```

Do not run pytest or inference.

# 14. Final task outcome

Choose exactly ONE.

## A — `PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_COMPLETE`

Require all gates pass.

Freeze:

```text
external policy docs = SYNCHRONIZED
external manifest-listed source/config = 135/135 MATCH
external source_manifest Git canonical identity = MATCH
protected assets = UNCHANGED
external setup = READY
locked candidates = NOT YET RUN
PROP-01 = OPEN
```

Next:

```text
NEXT = PROP01_LOCKED_DEMO_PROPOSAL_GATE
```

Do not execute.

## B — `PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_STOP`

Use for any failed gate.

Next:

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_EXTERNAL_SYNC_RECOVERY
```

Do not execute.

# 15. Report

Create:

```text
docs/task8b3_p1d11b_supported_domain_policy_external_sync.md
```

Required sections:

1. task / starting HEAD
2. canonical Git-manifest 135/135 gate
3. canonical source_manifest SHA256
4. pre-sync external helper check
5. pre-sync setup READY
6. protected asset pre-snapshot
7. controlled 135-file sync result
8. separate Git-canonical source_manifest copy
9. post-sync helper check 135/135
10. external policy spot-check
11. protected asset post comparison
12. post-sync setup READY
13. model/inference/pytest = NOT RUN
14. locked candidates = NOT YET RUN
15. PROP-01 = OPEN
16. exact outcome
17. exact next gate.

# 16. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11B
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: f9bf6be13035445f45f956c2b6a5a64ccf7e6032
Model/inference execution: NONE
Pytest: NOT RUN
Functional canonical files modified: NO
External delivery modified: YES — CONTROLLED POLICY/SOURCE SYNC ONLY / NO
Canonical Git manifest: 135/135 PASS / FAIL
Canonical source_manifest Git SHA256: <sha>
Pre-sync external check: 133/135 with only README.md + docs/model_card.md mismatched / other
Pre-sync external setup: READY / NOT READY
Manifest-listed files copied/verified: 135/135 / other
External source_manifest Git SHA match: YES / NO / NOT RUN
Post-sync external check: 135/135 PASS / FAIL / NOT RUN
External README policy: VERIFIED / NOT VERIFIED
External model-card policy: VERIFIED / NOT VERIFIED
Protected model/component assets changed: NO / YES / NOT CHECKED
Qwen asset directory changed: NO / YES / NOT CHECKED
Post-sync external setup: READY / NOT READY / NOT RUN
A2 domain classification: NOT ESTABLISHED
A2 policy status: DOCUMENTED_PERSISTENT_NON_DETECTION
Locked candidate status: FINAL_METADATA_LOCK
Locked candidate runtime status: NOT YET RUN
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: OPEN
MASK-01 status: OPEN
Outcome: <exact enum>
Next gate: <exact enum>
Report: docs/task8b3_p1d11b_supported_domain_policy_external_sync.md
Next action: Awaiting ChatGPT audit; do not run locked candidates.
```

# 17. Commit / push

If COMPLETE:

```text
docs(rc1): record prop01 policy external sync
```

If STOP/FAILED:

```text
docs(rc1): record prop01 policy sync stop
```

Push current branch normally.
No force push.

# 18. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- canonical Git manifest = 135/135;
- pre-sync external helper check is exactly 133/135 with only README/model_card mismatch;
- pre-sync setup READY;
- protected asset snapshot recorded;
- exactly one controlled 135-file helper sync succeeds;
- source_manifest separately written from Git canonical bytes;
- post-sync helper check = 135/135;
- external source_manifest Git SHA matches;
- external README/model-card policy spot-check passes;
- protected model/Qwen assets unchanged;
- post-sync setup READY;
- no pytest/inference/candidate run;
- no canonical functional modification;
- PROP-01 remains OPEN;
- next gate not executed;
- report/handoff committed and pushed;
- STOP.
