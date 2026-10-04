# TO_DSH — Task 8B.3-M1C: Integrate Closed MEM-01 Fix into main

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Source branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `509ed5ea6c712da0d65d31c499fa40cec16e94cc`
> Required origin/main before integration: `c45ecbec7fd293c454ccced22310db32c1542be4`

# 0. Purpose

Formally integrate the fully validated `RC1-DEMO-MEM-01` repair into `main` by fast-forward only.

Frozen ChatGPT audit:

```text
M1A dedicated compact tests = 32 passed
synthetic 5000×5000 detect_global guard = PASS
canonical manifest = 135/135 VERIFIED
controlled canonical → external sync = PASS
external full regression = 116 passed
B1 real 5000×5000 = SUCCESS, proposal stage complete, old memory signature absent
B2 real 5000×5000 = SUCCESS, proposal stage complete, old memory signature absent
post-run external manifest = 135/135 PASS
post-run source_manifest byte-identical = YES
MEM01_REAL_GATE_PASS
RC1-DEMO-MEM-01 = CLOSED
M1B.3 formal gate = PASS
```

Remote ancestry audited by ChatGPT:

```text
origin/main = c45ecbec7fd293c454ccced22310db32c1542be4
fix branch = 509ed5ea6c712da0d65d31c499fa40cec16e94cc
merge-base = origin/main
ahead = 25
behind = 0
```

This task performs documentation closure plus a fast-forward integration only.

# 1. Strict prohibitions

Do NOT:
- modify runtime code;
- modify tests;
- modify canonical or external `source_manifest.json`;
- modify external delivery;
- run pytest;
- run check_setup.py;
- run predict.py;
- run any model or Demo;
- change thresholds/config/tiling/merge/reference/validity policy;
- fix PROP-01 / REF-01 / MASK-01;
- enter Task 8B.4 / Task 8C;
- rebase;
- squash;
- cherry-pick;
- create a merge commit;
- force push;
- delete the MEM fix branch.

If a clean fast-forward cannot be proven, STOP.

# 2. Starting Git gate

Require exactly:

```text
current branch = fix/task8b3-mem01-compact-proposals
HEAD = 509ed5ea6c712da0d65d31c499fa40cec16e94cc
```

Allowed tracked working tree initially:
- clean; or
- only `M handoff/TO_DSH.md`.

Run:

```bat
git fetch origin
```

Then verify:

```text
origin/main = c45ecbec7fd293c454ccced22310db32c1542be4
```

Verify ancestry mechanically:

```bat
git merge-base origin/main HEAD
git rev-list --left-right --count origin/main...HEAD
```

Require:

```text
merge-base = c45ecbec7fd293c454ccced22310db32c1542be4
left/right count = 0 25
```

If not exact:
- no integration;
- record STOP;
- commit/push documentation on fix branch only if Git remains safe;
- STOP.

# 3. Allowed repository changes before integration

Only:

```text
docs/task8b3_m1c_mem01_main_integration.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No functional file may change.

# 4. Closure report

Create:

```text
docs/task8b3_m1c_mem01_main_integration.md
```

Record:

1. Task / scope
2. Starting fix HEAD
3. Pre-integration origin/main
4. Ancestry proof:
   - merge-base
   - ahead/behind
5. Frozen MEM evidence:
   - 32 dedicated tests
   - synthetic 5000 guard
   - external 116-test regression
   - B1/B2 real 5000×5000 SUCCESS
   - post-run manifest 135/135
   - source_manifest byte-identical
6. Final defect state:
   `RC1-DEMO-MEM-01 = CLOSED`
7. No functional changes in M1C
8. Integration policy = fast-forward only
9. Final main SHA after push
10. Next priority after audit = `RC1-DEMO-PROP-01`
11. `REF-01 / MASK-01` remain OPEN and untouched
12. Task 8B.4 / 8C not entered.

# 5. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required fields:

```text
Task: 8B.3-M1C
Status: COMPLETE / PARTIAL / STOP / FAILED
Source branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: 509ed5ea6c712da0d65d31c499fa40cec16e94cc
Pre-integration origin/main: c45ecbec7fd293c454ccced22310db32c1542be4
Merge-base: c45ecbec7fd293c454ccced22310db32c1542be4 / other
Ahead/behind before doc commit: 25/0 / other
Functional files modified: NO
Tests/inference executed: NO
RC1-DEMO-MEM-01: CLOSED
PROP-01 / REF-01 / MASK-01: OPEN / OPEN / OPEN
Integration method: FAST-FORWARD ONLY / NOT RUN
Final origin/main: <sha / unchanged>
Report: docs/task8b3_m1c_mem01_main_integration.md
Next action: Awaiting ChatGPT audit before PROP-01 work.
```

# 6. Documentation commit on fix branch

Before touching `main`, verify:

```bat
git status --short
git diff --check
```

Only §3 paths may differ.

Commit exactly:

```text
docs(rc1): record mem01 closure integration
```

Let the resulting new fix-branch commit SHA be:

```text
INTEGRATION_TARGET
```

Push source branch:

```bat
git push origin fix/task8b3-mem01-compact-proposals
```

No force.

If commit/push fails:
- do not touch main;
- STOP.

# 7. Re-prove fast-forward after documentation commit

Run:

```bat
git fetch origin
git merge-base origin/main origin/fix/task8b3-mem01-compact-proposals
git rev-list --left-right --count origin/main...origin/fix/task8b3-mem01-compact-proposals
```

Require:
- merge-base still equals `c45ecbec7fd293c454ccced22310db32c1542be4`;
- left count = 0;
- right count = 26.

If not:
- do not touch main;
- STOP.

# 8. Fast-forward main

Checkout local main:

```bat
git checkout main
```

Require clean tracked tree.

Synchronize local main only by:

```bat
git fetch origin
git reset --hard origin/main
```

This reset is authorized ONLY here, only on local `main`, to match verified `origin/main`.

Then fast-forward only:

```bat
git merge --ff-only origin/fix/task8b3-mem01-compact-proposals
```

Require local main HEAD == `INTEGRATION_TARGET`.

Push normally:

```bat
git push origin main
```

No force.

# 9. Post-integration verification

Run:

```bat
git fetch origin
git rev-parse origin/main
git rev-parse origin/fix/task8b3-mem01-compact-proposals
```

Require both equal `INTEGRATION_TARGET`.

Verify:

```bat
git status --short
```

must be clean.

Do not delete either branch.

# 10. COMPLETE definition

COMPLETE only if:
- exact starting source HEAD;
- exact pre-integration origin/main;
- ancestry is linear;
- no functional changes;
- no tests/inference;
- closure report/handoff committed on fix branch;
- source branch push PASS;
- `main` advanced by `--ff-only`;
- origin/main == origin/fix branch == integration target;
- no merge commit/rebase/force push;
- working tree clean;
- MEM-01 remains CLOSED;
- stop.

# 11. Final response

```text
TASK 8B.3-M1C COMPLETE / PARTIAL / STOP / FAILED

Documentation commit / integration target:
<sha or NONE>

Source branch push:
PASS / FAIL

Pre-integration origin/main:
c45ecbec7fd293c454ccced22310db32c1542be4 / other

Pre-doc ancestry:
merge-base = <sha>
ahead/behind = 25/0 / other

Post-doc ancestry:
ahead/behind = 26/0 / other

Functional files modified:
NO

Tests / inference:
NOT RUN

Integration method:
FAST-FORWARD ONLY / NOT RUN

Final origin/main:
<sha>

Final origin/fix branch:
<sha>

RC1-DEMO-MEM-01:
CLOSED

PROP-01 / REF-01 / MASK-01:
OPEN / OPEN / OPEN

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得开始 PROP-01、REF-01、MASK-01、Task 8B.4 或 Task 8C。
```
