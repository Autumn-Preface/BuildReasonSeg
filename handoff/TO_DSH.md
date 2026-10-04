# TO_DSH — Task 8B.3-M1B.2: External Full Regression Gate

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `25d879845ef1508228067d4faa45e98266ed6aa5`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Canonical RC1: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

Validate the synchronized runnable external RC1 delivery with the complete delivery-oriented pytest suite.

M1B.1 has already established:

```text
canonical manifest = 135/135 PASS
pre-sync external check_setup = READY
135 manifest-listed files copied
external manifest = 135/135 PASS
source_manifest byte-identical = YES
protected model/component assets changed = NO
post-sync external check_setup = READY
```

This task does exactly:

```text
external integrity preflight
→ run external `pytest tests -q` exactly once
→ record result
→ commit/push documentation only
→ STOP
```

No code/test/manifest edit is authorized.

# 1. Strict prohibitions

Do NOT modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/**
```

Do NOT modify any file under external delivery except normal transient pytest cache files that pytest itself may create.

In particular, do NOT modify:
- runtime source;
- tests;
- `source_manifest.json`;
- model/config metadata;
- decoder/detector/SAM2/ProgramHead/Qwen assets;
- thresholds;
- tiling;
- merge policy;
- Reference policy;
- SUCCESS validity.

Do NOT:
- run `predict.py`;
- run A1/A2/A3/A4/B1/B2;
- run YOLO/Qwen/SAM2/D-B1 inference;
- fix PROP-01 / REF-01 / MASK-01;
- enter Task 8B.4 / Task 8C;
- sync canonical → external again;
- install/download/train;
- update `main`;
- force push.

If the external full suite fails:
- do NOT patch code/tests;
- do NOT rerun pytest;
- record exact failures;
- commit/push PARTIAL/STOP;
- STOP.

# 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = 25d879845ef1508228067d4faa45e98266ed6aa5
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Do not reset/rebase/stash/clean/merge.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_m1b2_external_full_regression.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No canonical functional file may change.

# 4. External preflight — no pytest yet

Before running pytest, verify external root exists:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Read external `source_manifest.json` and verify:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
entry count = 135
all 135 listed external files exist
all 135 bytes match
all 135 SHA256 match
```

Require:

```text
external manifest = 135/135 PASS
```

Also verify external `source_manifest.json` is byte-identical to canonical `source_manifest.json`.

If either check fails:
- do NOT run pytest;
- record STOP;
- commit/push documentation;
- STOP.

# 5. External full regression — exactly once

Run exactly once:

```bat
cd /d C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
ENV_PYTHON -m pytest tests -q
```

Record:
- exact command;
- invocation count = 1;
- exact pass/fail/error/skip summary;
- runtime duration if printed;
- exit code.

No rerun.

# 6. Result policy

## 6.1 If PASS

If exit code = 0 and the full suite is green:

- mark `M1B external full regression gate = PASS`;
- do not run any other tests;
- do not run inference;
- continue only to report/handoff/commit/push.

## 6.2 If FAIL

If any failed/error node exists:

- mark `PARTIAL / STOP`;
- preserve exact pytest tail / failed node list;
- do not edit external delivery;
- do not edit canonical;
- do not rerun;
- continue only to report/handoff/commit/push;
- STOP.

# 7. Post-pytest integrity gate

After pytest, verify again:

```text
external manifest-listed files = 135/135 PASS
external/source_manifest.json remains byte-identical to canonical/source_manifest.json
```

Pytest cache files are not part of the 135-entry manifest and are not a failure by themselves.

Do not delete caches.

If a manifest-listed file changed:
- mark STOP;
- report exact path;
- do not repair autonomously.

# 8. Repository integrity

Back in repository:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git status --short
git diff --check
```

Only §3 paths may be modified.

# 9. Report

Create:

```text
docs/task8b3_m1b2_external_full_regression.md
```

Required sections:

1. Task / scope
2. Starting HEAD
3. External preflight manifest result
4. External source_manifest byte-identity result
5. Full pytest command
6. Full pytest invocation count = 1
7. Exact pytest result and exit code
8. Failed/error node list if any
9. Post-pytest external manifest result
10. Canonical product/tests/manifest modified = NO
11. External product/tests/manifest manually modified = NO
12. Real inference = NOT RUN
13. PROP-01 / REF-01 / MASK-01 = UNCHANGED
14. Gate result:
    - `M1B_EXTERNAL_FULL_REGRESSION_PASS`
    - or `M1B_EXTERNAL_FULL_REGRESSION_FAIL`
15. Next action:
    - awaiting ChatGPT audit;
    - no B1/B2 until audit.

# 10. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-M1B.2
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: 25d879845ef1508228067d4faa45e98266ed6aa5
External preflight manifest: 135/135 PASS / FAIL
source_manifest byte-identical before pytest: YES / NO
External full pytest invocation count: 1 / 0
External full suite: <exact result>
Pytest exit code: <code / NOT RUN>
Post-pytest external manifest: 135/135 PASS / FAIL / NOT RUN
Canonical product/tests/manifest modified: NO
External product/tests/manifest manually modified: NO
Real inference: NOT RUN
PROP-01 / REF-01 / MASK-01: UNCHANGED / UNCHANGED / UNCHANGED
Gate result: M1B_EXTERNAL_FULL_REGRESSION_PASS / M1B_EXTERNAL_FULL_REGRESSION_FAIL / NOT RUN
Report: docs/task8b3_m1b2_external_full_regression.md
Next action: Awaiting ChatGPT audit; do not run B1/B2.
```

# 11. Commit / push

If COMPLETE:

```text
test(rc1): pass external full regression
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record external full regression stop
```

Push:

```bat
git push origin fix/task8b3-mem01-compact-proposals
```

No force push.

# 12. COMPLETE definition

COMPLETE only if:
- exact branch/starting HEAD;
- external preflight manifest 135/135 PASS;
- external source_manifest byte-identical to canonical;
- external `pytest tests -q` invoked exactly once;
- suite PASS with exit code 0;
- post-pytest external manifest 135/135 PASS;
- no canonical functional change;
- no manual external functional change;
- no real inference;
- PROP/REF/MASK untouched;
- report/FROM_DSH complete;
- commit/push succeed;
- repository tracked tree clean;
- stop.

# 13. Final response

```text
TASK 8B.3-M1B.2 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

External preflight manifest:
135/135 PASS / other

source_manifest byte-identical:
YES / NO

External full pytest invocation count:
1 / 0

External full suite:
<exact result>

Pytest exit code:
<code>

Post-pytest external manifest:
135/135 PASS / other

Canonical functional files modified:
NO

External functional files manually modified:
NO

Real inference:
NOT RUN

Gate result:
M1B_EXTERNAL_FULL_REGRESSION_PASS / M1B_EXTERNAL_FULL_REGRESSION_FAIL / NOT RUN

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得运行 B1/B2、不得进入 Task 8B.4 或 Task 8C。
```
