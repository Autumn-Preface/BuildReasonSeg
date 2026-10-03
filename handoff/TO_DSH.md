# TO_DSH — Task 8B.3-R4A.1: Confirm Full Repository Gate

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `d4b7d36c68be7cee6970b9f3d58cc1cd4f47caf1`

# 0. Purpose

This is a **single verification task only**.

Task 8B.3-R4A harness corrections are accepted by ChatGPT.
The only unresolved gate is that the full repository suite was not rerun after the allowed `handoff/FROM_DSH.md` Watt wording repair.

Do not modify harness/product code.

# 1. Permanent reporting rule

For COMPLETE / PARTIAL / STOP / FAILED, if Git is safe:
- update `docs/task8b3_six_image_demo_suite.md`;
- update `handoff/FROM_DSH.md`;
- commit;
- push current branch;
- stop.

# 2. Strict prohibitions

Do not:
- modify `scripts/task8b3_interactive_suite.py`;
- modify `tests/test_task8b3_interactive_suite.py`;
- modify any other test;
- modify RC1 canonical/product/delivery source;
- run `scripts/task8b3_interactive_suite.py`;
- run real A1–B2 prediction;
- create/modify review pack;
- use Assisted Mode;
- install packages;
- enter Task 8B.4 or Task 8C.

# 3. Git safety

Run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git branch --show-current
git rev-parse HEAD
git status --short
```

Continue only if:

```text
branch = eval/task8b3-six-image-demo-suite
HEAD = d4b7d36c68be7cee6970b9f3d58cc1cd4f47caf1
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else -> STOP and report/push if Git-safe.

# 4. Full repository gate

Using the designated project runtime Python, run exactly once:

```bat
ENV_PYTHON -m pytest tests/ -q
```

## PASS

If the complete suite passes:
- record exact result;
- do not run any other test or real Demo;
- continue to §5.

## FAIL

If any test fails:
- do not modify code/docs to make the test pass, except the mandatory report/handoff recording;
- record exact failing test(s);
- mark PARTIAL/STOP;
- commit/push evidence;
- stop.

# 5. Report update

Append to:

```text
docs/task8b3_six_image_demo_suite.md
```

section:

```text
## Task 8B.3-R4A.1 — Full Repository Gate Confirmation
```

Record:
- starting HEAD;
- exact `pytest tests/ -q` result;
- harness code unchanged;
- tests unchanged;
- formal six-image suite = NOT RUN BY DESIGN;
- next action = Awaiting ChatGPT audit.

# 6. FROM_DSH update

Update `handoff/FROM_DSH.md`, preserving `ARTIFACT-FACTS` verbatim.

Required:

```text
Task: 8B.3-R4A.1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Starting HEAD: d4b7d36c...
Harness code: UNCHANGED
Harness tests: UNCHANGED
Repository tests: <exact result>
Formal suite: NOT RUN BY DESIGN
A1–B2: NOT RUN
Review pack: NOT CREATED / unchanged
Output-layout proposal: ACCEPTED / DEFERRED TO TASK 8B.4
Report: docs/task8b3_six_image_demo_suite.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit.
```

# 7. Git gate / commit / push

Allowed repository changes only:

```text
handoff/TO_DSH.md
handoff/FROM_DSH.md
docs/task8b3_six_image_demo_suite.md
```

Run:

```bat
git status --short
git diff --check
```

If COMPLETE, commit exactly:

```text
test(demo): confirm repository gate
```

If PARTIAL/STOP/FAILED:

```text
docs(demo): record repository gate stop
```

Push:

```bat
git push origin eval/task8b3-six-image-demo-suite
```

No force.

# 8. COMPLETE definition

COMPLETE only if:
1. exact start state;
2. no harness/product/test code changed;
3. full `pytest tests/ -q` passes;
4. formal suite not run;
5. report/handoff updated;
6. commit/push succeed;
7. clean tree;
8. stop.

# 9. Final response

```text
TASK 8B.3-R4A.1 COMPLETE / PARTIAL / STOP / FAILED

Branch:
eval/task8b3-six-image-demo-suite

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

Repository tests:
<exact result>

Harness/product code:
UNCHANGED

Formal suite:
NOT RUN BY DESIGN

Report:
docs/task8b3_six_image_demo_suite.md

Handoff:
handoff/FROM_DSH.md

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得运行正式六图、Task 8B.4、Task 8C 或任何新研发。
```
