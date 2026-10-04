# TO_DSH — Task 8B.3-M1B.1: Controlled Canonical → External Delivery Sync

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `a215db142caec155e2f6787804378e442fbb55d5`
> Canonical RC1: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

M1A canonical compact-proposal implementation/forensics are frozen.

This task performs only the controlled source/config synchronization into the existing runnable external RC1 delivery,
then verifies delivery source integrity and setup readiness.

Do NOT run pytest or real inference in this task.

# 1. Frozen entering facts

```text
compact proposal runtime dedicated tests: 32 passed
5000×5000 fake-shape guard: PASS
canonical source_manifest: 135/135 VERIFIED
canonical full delivery-oriented suite: invalid as a canonical gate
23 canonical failures classified A=10/B=13/C=0/D=0/E=0
external delivery: not yet synchronized with compact proposal runtime
```

Forensic conclusion:

```text
CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE
```

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = a215db142caec155e2f6787804378e442fbb55d5
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Do not reset/rebase/stash/clean/merge.

# 3. Strict prohibitions

Do NOT:
- modify canonical product/test/manifest files;
- modify model binaries/checkpoints in external delivery;
- delete any external delivery file or directory;
- overwrite external user/runtime outputs except the manifest-listed controlled source/config targets;
- run pytest;
- run predict.py;
- run six-image Demo;
- run YOLO/Qwen/SAM2/D-B1 inference;
- modify thresholds/models/tiling/merge/reference policies;
- fix PROP-01 / REF-01 / MASK-01;
- enter Task 8B.4 / 8C;
- update `main`;
- force-push.

# 4. Allowed repository changes

Only:

```text
docs/task8b3_m1b1_external_delivery_sync.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

The external delivery itself is intentionally modified by controlled sync.

# 5. Pre-sync canonical manifest gate

Read:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Verify before any external write:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
file entry count = 135
all 135 canonical paths exist
every canonical path size equals manifest bytes
every canonical path sha256 equals manifest sha256
```

If not 135/135 PASS:
- do not sync;
- report STOP;
- commit/push documentation only;
- stop.

Record SHA256 of the canonical `source_manifest.json` file itself separately.

# 6. Pre-sync external delivery readiness gate

Require external root exists:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Before copying anything, run exactly once:

```bat
cd /d C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
ENV_PYTHON check_setup.py
```

Record exit code and READY / NOT READY.

If pre-sync check is NOT READY:
- do not sync;
- report the exact setup failure;
- STOP.

This command is allowed; it is a setup/integrity checker, not model inference.

# 7. Protected external asset snapshot

Before sync, record size + SHA256 for these files if present:

```text
model/buildreasonseg_advisor/decoder.pt
model/buildreasonseg_advisor/detector.pt
model/components/sam2/sam2.1_hiera_base_plus.pt
model/components/sam2/sam2.1_hiera_b+.yaml
model/components/program_head/program_parser_l3_rehearsal_v1.pt
model/components/program_head/qwen_asset_manifest.json
```

For the Qwen base directory:

```text
model/components/program_head/Qwen3-VL-2B-Instruct
```

record:
- existence;
- recursive regular-file count;
- sorted relative file-name list;
- total bytes.

Do not hash the multi-GB Qwen model solely for this task unless the existing checker already does so.

Also record that the following external runtime/user trees are not sync targets:

```text
inference/input
inference/output
logs
runs
datasets
model binary/component asset files not present in the 135-entry source manifest
```

# 8. Controlled 135-entry sync

Implement a one-off local sync script or equivalent mechanical copy operation.

For every entry in canonical `source_manifest.json`:

```text
source = canonical_root / entry.path
destination = external_root / entry.path
```

Before copying each file:
- verify source bytes and SHA256 equal the manifest entry.

Then:
- create destination parent directories as needed;
- copy the file bytes from canonical to the same relative path in external;
- overwrite only that exact manifest-listed destination path.

Do NOT:
- recursively copy the whole canonical directory;
- delete external files;
- synchronize by directory mirroring;
- touch an unlisted external file.

After the 135 files are copied, separately copy:

```text
canonical/source_manifest.json
→ external/source_manifest.json
```

This control file is not required to self-hash inside its own file list.

No sync helper/script may remain committed in the repository.

# 9. Post-sync external integrity gate

Verify:

```text
external manifest-listed files = 135/135 matching canonical manifest bytes + sha256
external/source_manifest.json byte-identical to canonical/source_manifest.json
```

Then compare protected asset snapshot:

- each protected file from §7 must still exist;
- size + SHA256 must be identical pre/post;
- Qwen directory file count, names and total bytes must be identical pre/post.

If any protected asset changes:
- STOP immediately;
- do not attempt repair;
- report exact changed path(s).

# 10. Post-sync setup checker

Only after §9 PASS, run exactly once:

```bat
cd /d C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
ENV_PYTHON check_setup.py
```

Require:
- exit code 0;
- `BuildReasonSeg environment: READY`.

If FAIL:
- do not edit product;
- do not rollback autonomously;
- report exact output;
- STOP.

# 11. No pytest / no inference

Explicitly do NOT run:

```text
pytest
predict.py
A1/A2/A3/A4/B1/B2
```

The external full regression suite is Task M1B.2 after ChatGPT audits this sync.

# 12. Report

Create:

```text
docs/task8b3_m1b1_external_delivery_sync.md
```

Required content:

1. Task/scope
2. Starting HEAD
3. canonical manifest gate result
4. canonical source_manifest SHA256
5. pre-sync external `check_setup.py` result
6. protected asset pre-sync snapshot
7. sync policy:
   - 135 exact manifest-listed relative paths
   - no deletion
   - source_manifest copied separately
8. post-sync external 135/135 result
9. protected asset post-sync comparison
10. post-sync `check_setup.py` result
11. pytest = NOT RUN
12. real inference = NOT RUN
13. PROP-01 / REF-01 / MASK-01 unchanged
14. next action = awaiting ChatGPT audit before M1B.2 external full suite

# 13. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-M1B.1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: a215db142caec155e2f6787804378e442fbb55d5
Canonical manifest: 135/135 PASS / FAIL
Pre-sync external setup: READY / NOT READY
Manifest-listed files copied: 135 / other
External manifest match: 135/135 PASS / FAIL
source_manifest byte-identical: YES / NO
Protected external assets changed: NO / YES
Post-sync external setup: READY / NOT READY
Pytest: NOT RUN BY DESIGN
Real inference: NOT RUN
External delivery modified: YES — CONTROLLED MANIFEST SYNC ONLY
PROP-01 / REF-01 / MASK-01: UNCHANGED / UNCHANGED / UNCHANGED
Report: docs/task8b3_m1b1_external_delivery_sync.md
Next action: Awaiting ChatGPT audit before M1B.2 external full regression.
```

# 14. Git / commit / push

After external sync and verification, repository tracked changes may only be:

```text
docs/task8b3_m1b1_external_delivery_sync.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Run:

```bat
git status --short
git diff --check
```

If COMPLETE, exact commit:

```text
docs(rc1): record controlled external sync
```

Otherwise:

```text
docs(rc1): record external sync stop
```

Push current fix branch, no force.

# 15. COMPLETE definition

COMPLETE only if:
- exact starting branch/HEAD;
- canonical manifest 135/135 PASS;
- pre-sync external setup READY;
- exactly 135 manifest-listed files copied by relative path;
- external source_manifest copied separately and byte-identical;
- no unlisted external file deleted/overwritten;
- protected assets unchanged;
- external manifest verification 135/135 PASS;
- post-sync setup READY;
- no pytest;
- no real inference;
- no PROP/REF/MASK repair;
- report/handoff committed and pushed;
- repository tree clean;
- stop.

# 16. Final response

```text
TASK 8B.3-M1B.1 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Canonical manifest:
135/135 PASS / other

Pre-sync external setup:
READY / NOT READY

Manifest-listed files copied:
135 / other

External manifest match:
135/135 PASS / other

source_manifest byte-identical:
YES / NO

Protected external assets changed:
NO / YES

Post-sync external setup:
READY / NOT READY

Pytest:
NOT RUN BY DESIGN

Real inference:
NOT RUN

External delivery:
CONTROLLED MANIFEST SYNC ONLY / other

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得运行 external pytest、不得运行真实 Demo、不得进入 Task 8B.4 或 Task 8C。
```
