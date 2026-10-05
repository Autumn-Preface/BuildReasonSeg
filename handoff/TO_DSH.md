请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3B2-R2 — External Validation Handoff Normalization**

# TO_DSH — Task 8B.3-REF01-E3B2-R2: External Validation Handoff Normalization

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required starting HEAD: `14d6949f3ae4fda4b514701f1442b11b612deac0`

# 0. CHATGPT AUDIT DISPOSITION

E3B2 technical execution is ACCEPTED.

E3B2-R1 artifact paths and authoritative evidence are ACCEPTED.

Frozen authoritative facts:

```text
Technical source task:
8B.3-REF01-E3B2

Authoritative closure source:
8B.3-REF01-E3B2-R1

Design:
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

External pre-sync:
133 match / 0 missing / 2 mismatch

Controlled sync:
1 run
135 copied
135 verified
0 failures

External post-sync:
135 match / 0 missing / 0 mismatch

External source_manifest:
identical to canonical Git object
bytes = 24375
sha256 = 5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c

Setup checker:
BuildReasonSeg environment: READY

External targeted regression:
40 passed in 1.13s

External full delivery suite:
124 passed in 112.81s (0:01:52)

Model inference:
NONE

Canonical source / manifest / sync helper changes:
NONE

Authoritative evidence:
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json

Authoritative dedicated report:
docs/task8b3_ref01_e3b2_external_sync_full_suite.md

Outcome:
REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED

Next gate:
REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
```

R1 is not formally signed only because two literal placeholder/status fields remain:

1. `docs/task8b3_ref01_e3b2_external_sync_full_suite.md` contains:
   `bytes = <observed §4 integer>`
   instead of:
   `bytes = 24375`

2. `handoff/FROM_DSH.md` contains:
   `Status: COMPLETE / STOP / FAILED`
   instead of:
   `Status: COMPLETE`

   and:
   `External source_manifest bytes: <observed §4 integer>`
   instead of:
   `External source_manifest bytes: 24375`

R2 fixes ONLY these final presentation defects.

# 1. EXECUTOR CONTRACT

DSH has ZERO technical discretion.

Allowed operations ONLY:

1. verify exact branch/head;
2. verify authoritative evidence exists and contains external source_manifest bytes `24375`;
3. verify dedicated report and FROM_DSH contain exactly the known placeholders described above;
4. replace the dedicated-report placeholder with `24375`;
5. replace FROM_DSH active Status with `COMPLETE`;
6. replace FROM_DSH manifest-byte placeholder with `24375`;
7. preserve ARTIFACT-FACTS exactly;
8. leave this taskbook as `handoff/TO_DSH.md`;
9. make exactly one commit;
10. push once;
11. STOP.

Forbidden:
- NO sync helper;
- NO external `--check`;
- NO external sync/write;
- NO source_manifest identity recheck;
- NO check_setup;
- NO pytest;
- NO py_compile;
- NO detector/model inference;
- NO evidence edit;
- NO implementation-report edit;
- NO canonical source/test/manifest/helper edit;
- NO dependency/environment change;
- NO rebase/reset/amend/stash/clean;
- NO force push;
- NO intermediate commit/push;
- NO NEXT/E3C.

Any uncovered condition => STOP.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-sync

git rev-parse HEAD
= 14d6949f3ae4fda4b514701f1442b11b612deac0
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No commit/push until §8.

# 3. AUTHORITATIVE EVIDENCE READ-ONLY GATE

Read ONLY:

```text
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
```

Require these exact values:

```text
task =
8B.3-REF01-E3B2-R1

external_source_manifest.bytes =
24375

external_source_manifest.sha256 =
5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c

setup_checker.final_status =
BuildReasonSeg environment: READY

external_targeted_test.summary =
40 passed in 1.13s

external_full_suite.summary =
124 passed in 112.81s (0:01:52)

overall_outcome =
REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED

next_gate =
REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
```

Do NOT modify the evidence file.

Any mismatch => STOP.

# 4. DEDICATED REPORT EXACT NORMALIZATION

Target:

```text
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
```

Require exactly one occurrence of:

```text
bytes = <observed §4 integer>
```

Replace it with exactly:

```text
bytes = 24375
```

No other byte of this report may be changed.

After replacement require:
- `<observed §4 integer>` occurs zero times;
- `bytes = 24375` occurs exactly once.

# 5. FROM_DSH EXACT NORMALIZATION

Target:

```text
handoff/FROM_DSH.md
```

Preserve the complete block:

```text
<!-- ARTIFACT-FACTS:BEGIN -->
...
<!-- ARTIFACT-FACTS:END -->
```

byte-for-byte.

Below that block, require exactly one occurrence of:

```text
Status: COMPLETE / STOP / FAILED
```

Replace with:

```text
Status: COMPLETE
```

Require exactly one occurrence of:

```text
External source_manifest bytes: <observed §4 integer>
```

Replace with:

```text
External source_manifest bytes: 24375
```

No other active handoff field may be changed.

After replacement require:
- `Status: COMPLETE / STOP / FAILED` occurs zero times;
- `Status: COMPLETE` occurs exactly once in the active handoff;
- `<observed §4 integer>` occurs zero times in FROM_DSH;
- `External source_manifest bytes: 24375` occurs exactly once.

# 6. PROTECTED-FILE IMMUTABILITY GATE

Require NO working-tree/index change for:

```text
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
docs/task8b3_ref01_eligibility_repair_impl.md
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

Any change => STOP.

# 7. NO-RERUN ASSERTION

Confirm:

```text
sync helper run in R2 = NO
external --check run in R2 = NO
external sync/write in R2 = NO
source_manifest identity recheck in R2 = NO
check_setup run in R2 = NO
pytest run in R2 = NO
py_compile run in R2 = NO
detector/model inference in R2 = NO
```

Any non-NO => STOP.

# 8. FINAL DIFF GATE

Before commit:

```text
git diff --name-only 14d6949f3ae4fda4b514701f1442b11b612deac0
```

Allowed ONLY:

```text
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path is allowed.

Also require:

```text
git rev-list --count 14d6949f3ae4fda4b514701f1442b11b612deac0..HEAD
= 0
```

# 9. STATUS → COMMIT MESSAGE

If §§2–8 all PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): normalize external validation handoff
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record external handoff normalization stop
```

Commit exactly once.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count 14d6949f3ae4fda4b514701f1442b11b612deac0..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute NEXT/E3C.

Then STOP.

# 10. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- authoritative evidence unchanged;
- dedicated report changes only placeholder -> `24375`;
- FROM_DSH changes only final Status and manifest bytes placeholder;
- ARTIFACT-FACTS preserved exactly;
- protected files untouched;
- no technical command rerun;
- only report/FROM_DSH/TO_DSH differ;
- exactly one commit with exact message;
- NEXT/E3C not executed;
- STOP.
