请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3B2-R3 — Final Status Normalization**

# TO_DSH — Task 8B.3-REF01-E3B2-R3: Final Status Normalization

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required starting HEAD: `2502cff88fcdf231c6a07b868f8b089d0522eeb1`

# 0. CHATGPT AUDIT DISPOSITION

E3B2 technical execution is accepted.
E3B2-R1 authoritative evidence/report are accepted.
E3B2-R2 remote publication is accepted.

R2 fixed:
- dedicated report manifest bytes placeholder -> `24375`;
- FROM_DSH manifest bytes placeholder -> `24375`.

One final active-handoff defect remains:

```text
Current:
Status: COMPLETE / STOP / FAILED

Required:
Status: COMPLETE
```

R3 fixes ONLY this one active-handoff line.

# 1. EXECUTOR CONTRACT

Allowed operations ONLY:
1. verify exact branch/head;
2. verify `handoff/FROM_DSH.md` contains exactly one current status placeholder;
3. replace only that line with `Status: COMPLETE`;
4. preserve ARTIFACT-FACTS exactly;
5. leave this taskbook as `handoff/TO_DSH.md`;
6. commit exactly once;
7. push once;
8. STOP.

Forbidden:
- NO edit to dedicated report;
- NO edit to evidence;
- NO edit to implementation report;
- NO canonical source/test/manifest/helper edit;
- NO sync helper;
- NO external `--check`;
- NO external write/sync;
- NO check_setup;
- NO pytest;
- NO py_compile;
- NO inference;
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
= 2502cff88fcdf231c6a07b868f8b089d0522eeb1
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

# 3. FROM_DSH PRECONDITION

Target:

```text
handoff/FROM_DSH.md
```

Require:
- exactly one occurrence of `Status: COMPLETE / STOP / FAILED`;
- zero occurrence of `Status: COMPLETE` as a standalone active status line;
- exactly one occurrence of `External source_manifest bytes: 24375`;
- zero occurrence of `<observed §4 integer>`;
- ARTIFACT-FACTS block present.

Any mismatch => STOP.

# 4. EXACT EDIT

Replace exactly:

```text
Status: COMPLETE / STOP / FAILED
```

with:

```text
Status: COMPLETE
```

No other byte of `handoff/FROM_DSH.md` may change.

After edit require:
- old status placeholder occurs zero times;
- `Status: COMPLETE` occurs exactly once;
- `External source_manifest bytes: 24375` still occurs exactly once;
- ARTIFACT-FACTS block unchanged byte-for-byte.

# 5. PROTECTED FILE GATE

Require NO working-tree/index change for:

```text
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
docs/task8b3_ref01_eligibility_repair_impl.md
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

Any change => STOP.

# 6. NO-RERUN ASSERTION

Confirm:

```text
sync helper run in R3 = NO
external --check run in R3 = NO
external write/sync in R3 = NO
check_setup run in R3 = NO
pytest run in R3 = NO
py_compile run in R3 = NO
detector/model inference in R3 = NO
```

Any non-NO => STOP.

# 7. FINAL DIFF GATE

Before commit:

```text
git diff --name-only 2502cff88fcdf231c6a07b868f8b089d0522eeb1
```

Allowed ONLY:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path is allowed.

Also require:

```text
git rev-list --count 2502cff88fcdf231c6a07b868f8b089d0522eeb1..HEAD
= 0
```

# 8. STATUS → COMMIT MESSAGE

If §§2–7 all PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): finalize external validation status
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record final status normalization stop
```

Commit exactly once.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count 2502cff88fcdf231c6a07b868f8b089d0522eeb1..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute NEXT/E3C.

Then STOP.

# 9. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- only active Status line changes to `COMPLETE`;
- FROM_DSH manifest bytes remains `24375`;
- ARTIFACT-FACTS unchanged;
- only FROM_DSH and TO_DSH differ;
- no technical rerun or external write;
- exactly one commit with exact message;
- push succeeds;
- NEXT/E3C not executed;
- STOP.
