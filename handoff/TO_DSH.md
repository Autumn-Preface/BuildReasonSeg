# TO_DSH — Task 8B.3-R4A: Harness Corrections Only

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `2562c3bc53b2600098824e93572cca291d334e33`
> Main/origin-main: `c45ecbec7fd293c454ccced22310db32c1542be4`

# 0. Permanent reporting rule

For `COMPLETE / PARTIAL / STOP / FAILED`, if Git is safe:
1. update `handoff/FROM_DSH.md`;
2. update `docs/task8b3_six_image_demo_suite.md`;
3. commit;
4. push `eval/task8b3-six-image-demo-suite`;
5. stop.

# 1. Executor-only boundary

This task is **harness correction + tests only**.

Do not:
- run the real six-image suite;
- run A1/A2/A3/A4/B1/B2 through real `predict.py`;
- modify `delivery_src/BuildReasonSeg_Advisor_RC1/**`;
- modify external delivery source/config;
- modify product `predict.py`;
- modify ProgramHead/Qwen/suggestion/Validator;
- modify detector/Reference/SAM2/D-B1;
- modify thresholds/config/checkpoints;
- install packages;
- use Assisted Mode;
- implement Task 8B.4;
- enter Task 8C.

Only these repository paths may change:

```text
scripts/task8b3_interactive_suite.py
tests/test_task8b3_interactive_suite.py
docs/task8b3_six_image_demo_suite.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 2. Git safety

Run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git branch --show-current
git rev-parse HEAD
git rev-parse main
git rev-parse origin/main
git status --short
```

Continue only if:

```text
branch = eval/task8b3-six-image-demo-suite
HEAD = 2562c3bc53b2600098824e93572cca291d334e33
main = c45ecbec7fd293c454ccced22310db32c1542be4
origin/main = c45ecbec7fd293c454ccced22310db32c1542be4
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else -> STOP and persist/push report if Git-safe.

# 3. Fix exactly four harness issues

## 3.1 A3 frozen oracle

In `scripts/task8b3_interactive_suite.py`, replace the current computed A3 expected program with the exact literal:

```python
("A3", "inference/input/A3.png",
 "以最大建筑为准，分割位于其上方且距离最近的建筑",
 "largest_to_above_to_nearest")
```

Do not change any other case tuple.

## 3.2 Changed-output detection

Current snapshots already store per-file:

```text
[size_bytes, mtime_ns]
```

Add a pure helper with equivalent semantics:

```python
def changed_outputs(before: dict, after: dict) -> dict:
    result = {}
    for sub in ("masks", "overlays", "diagnostics"):
        result[sub] = sorted(
            path
            for path, after_meta in after.get(sub, {}).items()
            if path not in before.get(sub, {})
            or before[sub][path] != after_meta
        )
    return result
```

`run_case()` must use this changed set instead of filename set-difference.

This changed set must be used to locate this run's `result.json`.

Do not delete/clear old output files.

## 3.3 Repeated-prompt test

Replace the current blocking repeated-prompt test.

Fake child must:
1. print a valid `[解析] ... (largest_to_right_of_to_nearest)` line;
2. write the exact direct confirmation substring twice consecutively;
3. flush;
4. read stdin exactly once;
5. print `ANSWER=<answer>`;
6. exit.

Assertions:
- outcome is not timeout;
- child receives `Y`;
- transcript contains exactly one driver decision marker:
  `[driver] direct -> Y (DIRECT_CORRECT)`.

The driver must not be changed to answer the same prompt twice.

## 3.4 AST test

Replace the current AST test.

Parse the module and locate only function definitions named:

```text
_reader_thread
_run_interactive_process
```

Walk only those functions.

For executable `ast.Call` nodes, assert no called attribute/name is:

```text
readline
communicate
```

Do not call `ast.get_docstring()` on arbitrary AST nodes.

# 4. Add changed-output unit coverage

Add pure tests for:

1. new path -> changed;
2. same path + same `[size, mtime_ns]` -> unchanged;
3. same path + different size -> changed;
4. same path + same size + different mtime -> changed;
5. pre-existing `A1/result.json` whose metadata changes -> included in `changed["diagnostics"]`.

No real external delivery file may be used.

# 5. Preserve existing valid harness behavior

Do not alter:
- reader-thread + queue design;
- `time.monotonic()` timeout;
- `CASE_TIMEOUT_SECONDS = 900`;
- UTF-8 incremental decoding;
- exact product prompt strings;
- `PROGRAM_RE`;
- `DISPLAY_TO_PROGRAM`;
- exact three child-environment overrides;
- Y/N decision rules;
- remaining five frozen cases.

# 6. Dedicated harness test gate

Run once:

```bat
ENV_PYTHON -m pytest tests/test_task8b3_interactive_suite.py -q
```

Required: 100% PASS.

If it fails:
- do not make a second patch attempt in this task;
- update report/FROM_DSH;
- commit/push STOP;
- stop.

# 7. Full repository test gate

Only if dedicated harness tests PASS:

```bat
ENV_PYTHON -m pytest tests/ -q
```

Required: PASS.

If it fails:
- do not repair unrelated code;
- update report/FROM_DSH;
- commit/push STOP;
- stop.

# 8. Explicit prohibition on formal suite

Even if all tests PASS:

```text
DO NOT RUN scripts/task8b3_interactive_suite.py
```

against the real six images in this task.

Do not create or modify the review pack.

The one formal six-image run will be a separate ChatGPT-issued task after GitHub audit.

# 9. Report

Append to:

```text
docs/task8b3_six_image_demo_suite.md
```

section:

```text
## Task 8B.3-R4A — Harness Corrections Only
```

Record:
- starting HEAD;
- exact four corrections;
- dedicated test result;
- repository test result;
- formal suite = NOT RUN BY DESIGN;
- A1–B2 = NOT RUN;
- no product/delivery change;
- output-layout proposal = ACCEPTED / DEFERRED TO TASK 8B.4;
- next action = Awaiting ChatGPT audit.

# 10. FROM_DSH

Update `handoff/FROM_DSH.md`, preserving `ARTIFACT-FACTS` verbatim.

Required fields:

```text
Task: 8B.3-R4A
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Starting HEAD: 2562c3bc...
Harness corrections: PASS / FAIL
Driver tests: <result>
Repository tests: <result>
Formal suite: NOT RUN BY DESIGN
A1–B2: NOT RUN
Review pack: NOT CREATED / unchanged
Output-layout proposal: ACCEPTED / DEFERRED TO TASK 8B.4
Report: docs/task8b3_six_image_demo_suite.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit.
```

# 11. Git gate / commit / push

Run:

```bat
git status --short
git diff --check
git diff
```

Only allowed paths from §1 may change.

Stage individually.

If COMPLETE, commit exactly:

```text
test(demo): finalize six-image harness
```

If PARTIAL/STOP/FAILED:

```text
docs(demo): record task8b3-r4a stop
```

Push:

```bat
git push origin eval/task8b3-six-image-demo-suite
```

No force.

Working tree must be clean after commit.

# 12. COMPLETE definition

COMPLETE only if:
1. exact starting state;
2. A3 literal corrected;
3. overwritten-output detection corrected;
4. repeated-prompt test corrected;
5. AST test corrected;
6. changed-output tests added;
7. dedicated harness tests PASS;
8. full repository tests PASS;
9. no real six-image run occurred;
10. no product/delivery code changed;
11. report/FROM_DSH updated;
12. commit/push succeed;
13. clean tree;
14. DSH stops.

# 13. Final response

```text
TASK 8B.3-R4A COMPLETE / PARTIAL / STOP / FAILED

Branch:
eval/task8b3-six-image-demo-suite

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

Harness corrections:
PASS / FAIL

Driver tests:
<result>

Repository tests:
<result>

Formal suite:
NOT RUN BY DESIGN

A1–B2:
NOT RUN

Report:
docs/task8b3_six_image_demo_suite.md

Handoff:
handoff/FROM_DSH.md

Output layout proposal:
ACCEPTED / DEFERRED TO TASK 8B.4

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得运行正式六图、Task 8B.4、Task 8C 或任何新研发。
```
