# TO_DSH — Task 8B.3-R3: Final Harness Corrections and One Formal Six-Image Run

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `c08c65998f045d0e712c7e212f37a63042ec0f27`
> Main/origin-main: `c45ecbec7fd293c454ccced22310db32c1542be4`
> External runnable RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Runtime Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe`

---

# 0. Permanent reporting rule

On every terminal state (`COMPLETE / PARTIAL / STOP / FAILED`), if Git remains safe:

1. update `handoff/FROM_DSH.md`;
2. update `docs/task8b3_six_image_demo_suite.md`;
3. commit current evidence;
4. push `eval/task8b3-six-image-demo-suite`;
5. stop and wait for ChatGPT.

Do not leave decisive evidence only in local logs.

---

# 1. Executor-only rule

DSH is executor only.

Do not:
- modify `delivery_src/BuildReasonSeg_Advisor_RC1/**`;
- modify external delivery source/config;
- modify `predict.py`;
- modify ProgramHead/Qwen/suggestion/Validator;
- modify detector/Reference/SAM2/D-B1;
- modify threshold/config/checkpoint/model assets;
- install/uninstall packages;
- change/retry any frozen prompt;
- use Assisted Mode, `--reference-id`, or `--inspect-proposals`;
- access final test;
- train/download;
- implement Task 8B.4 output layout;
- enter Task 8B.4 or Task 8C.

Only validation harness, harness tests, report, and handoff may change.

---

# 2. ChatGPT audit of R2

R2 correctly implemented:
- reader-thread + queue;
- bounded main-thread polling;
- monotonic timeout;
- exact three child-environment overrides;
- no RC1 product changes.

Formal suite remains unrun.

Four final harness issues must be corrected before the one formal run.

## 2.1 Repeated-prompt test is invalid

Current fake child calls the same Y/N prompt twice and blocks waiting for a second stdin answer.
The production contract is “same prompt instance/type is answered at most once”, so deliberately refusing the second answer is correct.

Frozen test replacement:
- fake child must print the same direct confirmation substring twice **before a single stdin read**;
- then perform exactly one stdin read;
- then print `ANSWER=<value>`;
- assert driver transcript contains exactly one `[driver] direct -> ...` decision line;
- assert child receives exactly one Y/N answer;
- do not require a second answer.

## 2.2 AST test implementation is invalid

Do not call `ast.get_docstring()` indiscriminately on every AST node.

Frozen replacement:
- parse the module with `ast.parse`;
- locate only function definitions named:
  - `_reader_thread`
  - `_run_interactive_process`
- walk only those function bodies;
- collect executable `ast.Call` nodes;
- assert no call target attribute/name is `readline` or `communicate`;
- no assertion about docstring contents is required.

## 2.3 A3 frozen oracle is wrong in the driver

Current driver contains:

```python
"largest_to_above_of_to_nearest".replace("_above_of_", "_above_to_")
```

This is not an acceptable representation of the frozen oracle.

Replace that A3 expected program with the exact literal:

```text
largest_to_above_to_nearest
```

No other CASES tuple may change.

## 2.4 Output diff misses overwritten artifacts

Current `run_case()` derives artifacts by set difference of file names only.
A1 already has prior R2 outputs, so a formal A1 run may overwrite an existing file name; that must still count as this run's artifact.

Frozen fix:
- compare both path presence and recorded metadata;
- a path is “changed in this run” if:
  1. absent before and present after, OR
  2. present before and after but its stored metadata differs.
- use the existing `(size, mtime_ns)` snapshot information; do not hash large outputs.
- result.json discovery must search the changed diagnostics set, not only newly named files.
- preserve old files; do not delete outputs before the suite.

Add a pure helper such as:

```python
def changed_outputs(before: dict, after: dict) -> dict:
    ...
```

or equivalent.

---

# 3. Phase A — Git safety

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
HEAD = c08c65998f045d0e712c7e212f37a63042ec0f27
main = c45ecbec7fd293c454ccced22310db32c1542be4
origin/main = c45ecbec7fd293c454ccced22310db32c1542be4
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Any other state -> STOP and persist/push report if Git-safe.

No reset/stash/clean/rebase/merge.

---

# 4. Phase B — Allowed repository files

Only:

```text
scripts/task8b3_interactive_suite.py
tests/test_task8b3_interactive_suite.py
docs/task8b3_six_image_demo_suite.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other repository path may change.

---

# 5. Phase C — Driver changes

Modify `scripts/task8b3_interactive_suite.py` only as follows.

## C1. A3 oracle

Set exactly:

```python
("A3", "inference/input/A3.png",
 "以最大建筑为准，分割位于其上方且距离最近的建筑",
 "largest_to_above_to_nearest")
```

Do not use `.replace()` or computed expressions in any frozen expected program.

## C2. Changed-output helper

Introduce a deterministic helper with semantic behavior:

```python
changed[sub] = sorted(
    path
    for path, after_meta in after[sub].items()
    if path not in before[sub] or before[sub][path] != after_meta
)
```

It must support the existing `masks`, `overlays`, `diagnostics` dictionaries.

`run_case()` must use this changed set for:
- per-run artifact evidence;
- locating this run's `result.json`.

Do not clear output directories.

## C3. No other driver change

Do not change:
- reader thread;
- queue timing;
- timeout;
- prompt constants;
- regex;
- DISPLAY_TO_PROGRAM;
- remaining five CASES;
- child environment;
- process invocation;
- Y/N rules.

---

# 6. Phase D — Harness test corrections

Modify only `tests/test_task8b3_interactive_suite.py`.

## D1. Repeated prompt test

Replace the current blocking two-read fake child.

Use a fake child that:
1. prints a valid `[解析]` line;
2. writes `DIRECT_PROMPT` twice consecutively without newline requirement;
3. flushes;
4. performs one `sys.stdin.readline()`;
5. prints `ANSWER=<answer>`;
6. exits.

Assertions:
- child receives `Y` when expected program matches;
- transcript contains exactly one driver direct-decision marker;
- process exits normally;
- no timeout.

## D2. AST test

Locate only `_reader_thread` and `_run_interactive_process`.

For each:
- walk function AST;
- collect executable calls;
- assert no called attribute/name equals `readline` or `communicate`.

Do not scan docstrings with `ast.get_docstring()`.

## D3. Frozen A3 test

Assert `driver.CASES` exactly contains literal:

```text
largest_to_above_to_nearest
```

for A3.

## D4. Changed-output regression

Add pure unit tests for the helper:
1. new path is reported changed;
2. same path + same metadata is not changed;
3. same path + different size is changed;
4. same path + same size but different `mtime_ns` is changed;
5. changed diagnostics `A1/result.json` is included even when the path existed before.

No real delivery files used.

## D5. Preserve prior required tests

All prior valid coverage remains:
- direct Y/N;
- UTF-8 no-newline;
- suggestion Y/N;
- fallback N;
- silent-child timeout;
- exact child environment;
- six frozen cases.

---

# 7. Phase E — Dedicated test gate

Run once:

```bat
ENV_PYTHON -m pytest tests/test_task8b3_interactive_suite.py -q
```

Must be 100% PASS.

If any failure:
- do not patch further in this task;
- do not run real suite;
- update report/FROM_DSH;
- commit/push STOP;
- stop.

---

# 8. Phase F — Full repository gate

Only if dedicated tests PASS:

```bat
ENV_PYTHON -m pytest tests/ -q
```

Must PASS.

Any failure -> no formal suite, no unrelated repair; report/push STOP.

---

# 9. Phase G — External runtime/input gate

Run external:

```bat
ENV_PYTHON check_setup.py
```

Must end:

```text
BuildReasonSeg environment: READY
```

Verify 6/6 inputs and exact previously recorded SHA256 identities.

Any input hash mismatch -> STOP.

---

# 10. Phase H — Formal suite freeze

Before formal suite, append to report:

```text
FORMAL_SUITE_FREEZE
driver_sha256 = ...
tests_sha256 = ...
A1_sha256 = ...
A2_sha256 = ...
A3_sha256 = ...
A4_sha256 = ...
B1_sha256 = ...
B2_sha256 = ...
dedicated_tests = PASS
repository_tests = PASS
check_setup = READY
```

After this point:
- no driver/test/prompt/product edit;
- no individual sample retry.

---

# 11. Phase I — One formal suite run

Run exactly once:

```bat
ENV_PYTHON scripts/task8b3_interactive_suite.py
```

Required order:

```text
A1 -> A2 -> A3 -> A4 -> B1 -> B2
```

Sample-level language/model failures are recorded and later cases continue if environment remains READY.

If driver orchestration itself crashes:
- no patch;
- no rerun;
- report/push partial;
- STOP.

---

# 12. Phase J — Objective result collection

For each A1–B2 record:
- initial program/confidence;
- Y/N;
- language status;
- executed_visual;
- exit/result/error;
- Reference ID;
- mask/overlay/diagnostics;
- tile/raw/merged proposals if available;
- context-limit flag.

Runtime success label:

```text
AUTOMATIC_RUNTIME_SUCCESS_PENDING_VISUAL_REVIEW
```

Do not issue a semantic mask-quality verdict.

For A1 specifically, verify the per-run evidence can identify changed/overwritten result artifacts even though prior R2 A1 outputs already existed.

---

# 13. Phase K — Review pack

Create local-only if outputs exist:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\review_task8b3
```

Create `INDEX.md`.

Copy successful overlays to `<ID>_overlay_review.png`.

For failures, only copy proposal evidence already generated by normal execution; do not launch inspect-proposals.

Do not Git-track review pack.

---

# 14. Phase L — Report

Update:

```text
docs/task8b3_six_image_demo_suite.md
```

Preserve R1/R2 history and add:

```text
## Task 8B.3-R3 — Final Harness Corrections and Formal Suite
```

Include:
- four ChatGPT audit findings above;
- exact corrections;
- dedicated tests;
- repo tests;
- setup/input gate;
- freeze hashes;
- formal-run statement: RUN ONCE / NOT RUN / PARTIAL;
- six-case table;
- runtime/language summary;
- review pack;
- Visual Verdict = `PENDING CHATGPT/USER REVIEW`;
- output layout = `ACCEPTED / DEFERRED TO TASK 8B.4`;
- no-tuning statement;
- stop reason if any.

---

# 15. Phase M — FROM_DSH

Always update:

```text
handoff/FROM_DSH.md
```

Preserve `ARTIFACT-FACTS` verbatim.

Required:

```text
Task: 8B.3-R3
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Starting HEAD: c08c6599...
Main/origin-main: c45ecbec...
Driver tests: ...
Repository tests: ...
check_setup: ...
Inputs: ...
Formal suite: RUN ONCE / NOT RUN / PARTIAL
A1: ...
A2: ...
A3: ...
A4: ...
B1: ...
B2: ...
Review pack: ...
Visual verdict: PENDING CHATGPT/USER REVIEW
Output-layout proposal: ACCEPTED / DEFERRED TO TASK 8B.4
Report: docs/task8b3_six_image_demo_suite.md
STOP reason: ...
Next action: Awaiting ChatGPT audit.
```

---

# 16. Phase N — Git gate / commit / push

Allowed paths only:

```text
scripts/task8b3_interactive_suite.py
tests/test_task8b3_interactive_suite.py
docs/task8b3_six_image_demo_suite.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Run:

```bat
git status --short
git diff --check
git diff
```

Stage individually.

COMPLETE commit:

```text
test(demo): complete frozen six-image suite
```

PARTIAL/STOP/FAILED commit:

```text
docs(demo): record task8b3-r3 execution stop
```

Push:

```bat
git push origin eval/task8b3-six-image-demo-suite
```

No force.

Working tree must be clean after commit.

Do not merge eval branch to main.

---

# 17. COMPLETE definition

COMPLETE only if:
1. exact start state;
2. only allowed files changed;
3. A3 oracle exact literal fixed;
4. repeated-prompt test valid;
5. AST test valid;
6. changed/overwritten output detection implemented and tested;
7. dedicated tests 100% PASS;
8. repo tests PASS;
9. check_setup READY;
10. 6 input hashes unchanged;
11. formal freeze recorded;
12. formal suite run exactly once;
13. all six attempted in order;
14. no retry/tuning;
15. objective evidence captured;
16. review pack created as applicable;
17. no product source change;
18. report/FROM_DSH updated;
19. output-layout deferred;
20. commit/push succeed;
21. clean tree;
22. stop.

COMPLETE does not mean 6/6 visually correct.

---

# 18. Final response

```text
TASK 8B.3-R3 COMPLETE / PARTIAL / STOP / FAILED

Branch:
eval/task8b3-six-image-demo-suite

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

Driver tests:
<result>

Repository tests:
<result>

check_setup:
READY / NOT READY / NOT RUN

Inputs:
6/6 / other

Formal suite:
RUN ONCE / NOT RUN / PARTIAL

A1:
language=<status>
runtime=<status>

A2:
language=<status>
runtime=<status>

A3:
language=<status>
runtime=<status>

A4:
language=<status>
runtime=<status>

B1:
language=<status>
runtime=<status>

B2:
language=<status>
runtime=<status>

Review pack:
<path or NOT CREATED>

Visual verdict:
PENDING CHATGPT/USER REVIEW

Output layout proposal:
ACCEPTED / DEFERRED TO TASK 8B.4

Report:
docs/task8b3_six_image_demo_suite.md

Handoff:
handoff/FROM_DSH.md

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得进入 Assisted Mode、Task 8B.4、Task 8C 或任何新研发。
```
