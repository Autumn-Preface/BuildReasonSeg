# TO_DSH — Task 8B.3-M1A.2C: Full Canonical Regression Gate

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `c432ec41f2d8bad9ddc67d65d0dbc033724e482a`
> Canonical RC1: `delivery_src/BuildReasonSeg_Advisor_RC1`

# 0. Purpose

Close the canonical validation phase for the MEM-01 compact-proposal refactor.

Frozen status entering this task:

```text
compact proposal runtime implemented
reference crop correction implemented
dedicated runtime tests = 32 passed
5000×5000 fake-shape allocation guard = PASS
pairwise proposal_iou exercised exactly once in guard
source_manifest = 135/135 verified
external delivery = unchanged
real inference = not run
```

This task does exactly:

```text
correct one stale report sentence
→ run the full canonical test suite exactly once
→ record result
→ commit/push documentation only
```

No runtime/test/manifest modification is authorized.

# 1. Strict prohibitions

Do NOT modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/**
delivery_src/BuildReasonSeg_Advisor_RC1/tests/**
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Do NOT modify external delivery.

Do NOT:
- run real `predict.py`;
- run A1–B2 Demo;
- run YOLO/Qwen/SAM2/D-B1 inference;
- change any model/checkpoint/config/threshold/tiling;
- change compact representation;
- change merge/Reference/SUCCESS semantics;
- fix PROP-01 / REF-01 / MASK-01;
- enter Task 8B.4 or Task 8C;
- install packages;
- train/download;
- access final test.

If the full canonical suite fails:
- do NOT patch code/tests;
- do NOT rerun;
- record exact failure;
- commit/push PARTIAL;
- STOP.

# 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = c432ec41f2d8bad9ddc67d65d0dbc033724e482a
```

Allowed initial tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Do not reset/rebase/stash/clean/merge.

# 3. Allowed changed paths

Only:

```text
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No canonical product/test/manifest file may change in this task.

# 4. Correct one stale R3 report sentence

In:

```text
docs/task8b3_m1a_compact_proposal_masks.md
```

the R3 section currently ends with stale text equivalent to:

```text
Per the task book, a failing single run is recorded here and the task stops without runtime changes,
without a rerun and without a manifest update.
```

That sentence contradicts the same R3 section, which correctly records:

```text
32 passed in 0.30s
manifest 135/135 verified
```

Replace only that stale R3 closing sentence with:

```text
The single dedicated run passed. The manifest was then refreshed and verified at 135/135.
Runtime files and external delivery remained unchanged, and no real inference or full canonical suite was run in R3.
```

Do not rewrite historical R2 failure text.

# 5. Full canonical suite — exactly once

Run exactly once:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1
ENV_PYTHON -m pytest tests -q
```

Record:
- exact command;
- invocation count = 1;
- exact passed/failed/skipped count;
- runtime duration if printed;
- exit code.

No rerun.

# 6. Post-test file-integrity gate

After the test run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git status --short
git diff --check
```

Only these paths may be modified:

```text
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

If pytest modified/created tracked canonical files -> STOP and report exact paths.

Untracked normal pytest caches are not to be committed; do not use `git clean`.

# 7. Report

Append:

```text
## Task 8B.3-M1A.2C — Full Canonical Regression Gate
```

Required content:

```text
starting HEAD = c432ec41f2d8bad9ddc67d65d0dbc033724e482a
full canonical invocation count = 1
command = ENV_PYTHON -m pytest tests -q
exact result = <pytest result>
exit code = <code>
runtime/test/manifest files modified in this task = NO
source_manifest entering gate = 135/135 verified
external delivery = UNCHANGED
real inference = NOT RUN
PROP-01 / REF-01 / MASK-01 = UNCHANGED
```

If PASS, state:

```text
M1A canonical validation gate = PASS
Next gate = ChatGPT audit before canonical → external delivery sync
```

Do not claim MEM-01 is closed in external delivery yet.

# 8. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
Save UTF-8 without BOM.

Required fields:

```text
Task: 8B.3-M1A.2C
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: c432ec41f2d8bad9ddc67d65d0dbc033724e482a
Runtime files modified: NO
Test files modified: NO
Manifest modified: NO
Full canonical invocation count: 1
Full canonical suite: <exact result>
Manifest entering gate: 135/135 VERIFIED
External delivery modified: NO
Real inference executed: NO
PROP-01 / REF-01 / MASK-01: UNCHANGED / UNCHANGED / UNCHANGED
M1A canonical validation gate: PASS / FAIL
Report: docs/task8b3_m1a_compact_proposal_masks.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit; do not sync delivery.
```

# 9. Commit / push

Allowed changed paths remain only report/FROM_DSH/TO_DSH.

If COMPLETE:

```text
test(rc1): pass compact proposal canonical regression
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record compact proposal canonical regression stop
```

Push:

```bat
git push origin fix/task8b3-mem01-compact-proposals
```

No force push.

# 10. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- stale R3 sentence corrected;
- no product/test/manifest modification;
- full canonical suite invoked exactly once;
- full canonical suite PASS;
- external delivery untouched;
- real inference not run;
- report/FROM_DSH complete;
- ARTIFACT-FACTS preserved;
- commit/push succeed;
- working tree clean;
- DSH stops.

# 11. Final response

```text
TASK 8B.3-M1A.2C COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Runtime files modified:
NO

Test files modified:
NO

Manifest modified:
NO

Full canonical invocation count:
1

Full canonical suite:
<exact result>

Manifest entering gate:
135/135 VERIFIED

External delivery:
UNCHANGED

Real inference:
NOT RUN

M1A canonical validation gate:
PASS / FAIL

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得同步 delivery、不得运行真实 Demo、不得进入 Task 8B.4 或 Task 8C。
```
