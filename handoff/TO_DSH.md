# TO_DSH — Task 8B.3-R2: Finalize Deterministic Driver and Run the Frozen Six-Image Suite

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `1287d0a2bab50a278452d4cd0ec8d492347afdfe`
> Audited RC1/main HEAD: `c45ecbec7fd293c454ccced22310db32c1542be4`
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

Only Git safety/auth/network failure may prevent commit/push. Never destroy user changes to force a push.

# 1. Executor-only rule

DSH is executor only.

Do not:
- modify any RC1 product/runtime source;
- modify `delivery_src/BuildReasonSeg_Advisor_RC1/**`;
- edit external delivery source/config directly;
- modify `predict.py`;
- modify ProgramHead/Qwen/suggestion/Validator;
- modify detector/Reference/SAM2/D-B1;
- modify thresholds/config/checkpoints;
- install/uninstall packages;
- change or retry any frozen natural-language prompt;
- use Assisted Mode, `--reference-id`, or `--inspect-proposals`;
- access final test;
- train/download anything;
- implement the future per-run output-directory product layout;
- enter Task 8B.4 or Task 8C.

If an uncovered decision is required: STOP and report.

# 2. ChatGPT audit verdict on Task 8B.3-R1

The R1 PARTIAL/STOP is accepted.

Git reporting protocol worked correctly:
- branch was pushed;
- report exists;
- `FROM_DSH` exists;
- remote `main` is now correctly `c45ecbec7fd293c454ccced22310db32c1542be4`.

The real six-image suite has not started.

ChatGPT found four harness-level items that must be fixed before the one formal suite run.

## 2.1 Test defect A

`tests/test_task8b3_interactive_suite.py::_run_fake(... expected=...)` receives an expected program but ignores it and hard-codes `largest_to_right_of_to_nearest`.

Frozen fix:
- make the fake-driver decision compare the parsed program to the `expected` argument;
- do not hard-code A1's program inside `_run_fake`.

## 2.2 Test defect B

`test_decision_path_has_no_readline` scans the whole source text and falsely matches the driver's explanatory docstring.

Frozen fix:
- parse the driver with Python `ast`;
- inspect executable call nodes / executable source semantics rather than raw docstring text;
- assert there is no executable `.readline(...)` call and no `.communicate(...)` call in the interaction path;
- do not remove useful documentation merely to satisfy string search.

## 2.3 Driver timeout defect

The current driver calls blocking `os.read(...)` directly in the main control loop. If the child stays alive but emits no bytes, the main thread can block beyond the intended 15-minute deadline.

Frozen fix:
- move blocking stdout reads into a dedicated daemon reader thread;
- reader thread pushes raw byte chunks and one EOF sentinel into `queue.Queue`;
- main thread uses `queue.get(timeout=0.10)` or equivalent bounded wait;
- main thread alone owns wall-clock deadline checks and Y/N decisions;
- timeout must trigger even if child emits no output at all.

## 2.4 Environment-scope deviation

The current driver adds:
`HF_HUB_OFFLINE=1`
`TRANSFORMERS_OFFLINE=1`

Frozen fix:
- remove those two harness-added assignments;
- `child_environment()` must copy `os.environ` and then explicitly set only:
  - `PYTHONIOENCODING=utf-8`
  - `PYTHONUNBUFFERED=1`
  - `PYTHONUTF8=1`
- do not remove those variables if they already existed naturally in the inherited parent environment; simply do not create/override them in the harness.

No other driver behavior is authorized to change.

# 3. Phase A — Git safety

Run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git branch --show-current
git rev-parse HEAD
git rev-parse main
git rev-parse origin/main
git status --short
git remote -v
```

Continue only if:

```text
branch = eval/task8b3-six-image-demo-suite
HEAD = 1287d0a2bab50a278452d4cd0ec8d492347afdfe
main = c45ecbec7fd293c454ccced22310db32c1542be4
origin/main = c45ecbec7fd293c454ccced22310db32c1542be4
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Any other state -> STOP, report/push if Git-safe.

No reset/stash/clean/rebase/merge.

# 4. Phase B — Allowed files

Only these repository files may change:

```text
scripts/task8b3_interactive_suite.py
tests/test_task8b3_interactive_suite.py
docs/task8b3_six_image_demo_suite.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other repository path may change.
No external delivery source/config file may be edited.
Runtime outputs/transcripts/review pack remain local-only outside Git.

# 5. Phase C — Fix child environment exactly

In `scripts/task8b3_interactive_suite.py`, `child_environment()` must have this semantic behavior:

```python
def child_environment() -> dict:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONUTF8"] = "1"
    return environment
```

No harness-created/overridden environment variable besides those three.

# 6. Phase D — Implement reader-thread/queue architecture

Use Python standard library only:
`queue`, `threading`.

## 6.1 Reader thread

After `Popen`, create one daemon reader thread whose only responsibilities are:
1. obtain `process.stdout`;
2. repeatedly perform blocking binary reads;
3. put each non-empty `bytes` chunk into `queue.Queue`;
4. when EOF is reached, put one fixed EOF sentinel into the queue;
5. never decode text;
6. never decide Y/N;
7. never mutate language/program state.

A fixed read size such as 4096 is allowed.

## 6.2 Main control loop

The main thread must:
- create incremental UTF-8 decoder;
- consume queue with bounded wait of at most 0.10 s;
- append decoded text to buffer/transcript;
- parse initial program;
- detect exact prompt substrings;
- make Y/N decisions;
- check `time.monotonic()` deadline on every control-loop cycle;
- observe process state;
- exit only after process ended and EOF consumed, or timeout handling completes.

Do not use:
- `readline()`;
- `for line in process.stdout`;
- `communicate(input=...)`.

## 6.3 Timeout

Use:
```python
deadline = time.monotonic() + CASE_TIMEOUT_SECONDS
```

When deadline expires while child is alive:
1. mark timeout;
2. `process.terminate()`;
3. wait up to 10 s;
4. if still alive, `process.kill()`;
5. drain already queued output for a bounded short period;
6. save transcript;
7. classify `DRIVER_TIMEOUT`.

This timeout must work even if the child printed zero bytes.

Do not retry timed-out sample.

# 7. Phase E — Keep frozen interaction semantics

Do not change:
```text
DIRECT_PROMPT
SUGGESTION_PROMPT
FALLBACK_PROMPT
PROGRAM_RE
DISPLAY_TO_PROGRAM
CASES
CASE_TIMEOUT_SECONDS = 900
```

Direct:
- parsed exact expected -> Y;
- supported but wrong -> N.

Suggestion:
- suggestion exact expected -> Y;
- other suggestion -> N.

Fallback:
- always N.

Do not add keyword parsing of user prompt.

# 8. Phase F — Repair and strengthen harness unit tests

Modify only `tests/test_task8b3_interactive_suite.py`.

Required coverage:

1. right expected + right parse -> Y;
2. left expected + left parse -> Y;
3. left expected + right parse -> N;
4. direct Y/N prompt without newline;
5. Chinese prompt survives binary pipe and UTF-8 incremental decoding;
6. suggestion above + expected above -> Y;
7. suggestion above + expected below -> N;
8. fallback prompt -> N;
9. repeated prompt -> only one stdin answer;
10. no-newline output before/after prompt preserved in transcript;
11. AST-based check: no executable `readline()` or `communicate()` call in driver interaction implementation;
12. silent-child timeout regression:
   - monkeypatch `CASE_TIMEOUT_SECONDS = 0.5`;
   - fake child emits zero output and sleeps longer;
   - driver helper returns/terminates within <5 seconds;
   - timeout classified;
   - child not left running;
13. child environment explicitly overrides only the three frozen encoding/buffering variables;
14. driver does not explicitly assign `HF_HUB_OFFLINE` or `TRANSFORMERS_OFFLINE`;
15. suite contains exactly A1/A2/A3/A4/B1/B2 with exact frozen prompts/programs.

Use fake child scripts only; no real models/images.

If needed, refactor the validation harness into a reusable internal helper such as `_run_interactive_process(...)`, but do not change product runtime.

# 9. Phase G — Dedicated test gate

Run with designated runtime Python:

```bat
ENV_PYTHON -m pytest tests/test_task8b3_interactive_suite.py -q
```

Must be 100% PASS.

If any dedicated test fails:
- do not run real suite;
- do not patch again in this task after this gate;
- update report/FROM_DSH;
- commit/push PARTIAL/STOP;
- stop for ChatGPT.

# 10. Phase H — Full repository test gate

Only after dedicated tests pass:

```bat
ENV_PYTHON -m pytest tests/ -q
```

Must PASS.

If any failure:
- do not run real suite;
- do not modify unrelated code;
- report/push STOP.

# 11. Phase I — External runtime/input gate

Run:

```bat
cd /d C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
ENV_PYTHON check_setup.py
```

Must end:
`BuildReasonSeg environment: READY`

Verify all six frozen inputs and preserve the SHA256 identities already recorded in R1 report.

If any SHA differs -> STOP before formal suite.

Do not substitute images.

# 12. Phase J — Formal suite freeze point

Immediately before real suite, record in the report:

```text
FORMAL_SUITE_FREEZE:
driver working-copy sha256 = <sha256>
test file sha256 = <sha256>
six input hashes = <A1..B2>
dedicated tests = PASS
repository tests = PASS
check_setup = READY
```

After this point:
- do not edit driver;
- do not edit tests;
- do not edit prompt suite;
- do not edit RC1;
- do not rerun an individual formal sample.

# 13. Phase K — Run the formal suite exactly once

From repository root run:

```bat
ENV_PYTHON scripts/task8b3_interactive_suite.py
```

Required order:
`A1 -> A2 -> A3 -> A4 -> B1 -> B2`

A sample-level model/language failure is recorded and later samples continue if environment remains healthy.

If the driver orchestration itself crashes after formal suite begins:
- do not patch;
- do not rerun;
- report/push partial evidence;
- STOP.

No semantic-result-driven changes allowed after formal run begins.

# 14. Phase L — Objective result interpretation

For every case record:
- initial program;
- confidence;
- language status;
- sent Y/N;
- whether visual chain executed;
- exit code;
- runtime result/error;
- Reference ID;
- mask/overlay/diagnostics;
- tile count;
- raw/merged proposals if available;
- context-limit flag if present.

A runtime `SUCCESS` must be labeled:
`AUTOMATIC_RUNTIME_SUCCESS_PENDING_VISUAL_REVIEW`

Never label visual correctness from successful exit alone.

# 15. Phase M — Review pack

If formal suite produces outputs, create local-only:

`C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\review_task8b3`

Create `INDEX.md`.

Copy successful overlays as:
- `A1_overlay_review.png`
- `A2_overlay_review.png`
- `A3_overlay_review.png`
- `A4_overlay_review.png`
- `B1_overlay_review.png`
- `B2_overlay_review.png`

Only if actually produced.

Failed cases may include copied `global_proposals` if already produced by normal diagnostics. Do not invoke new `--inspect-proposals`.

Review pack is not Git-tracked.

# 16. Phase N — Report update

Update:
`docs/task8b3_six_image_demo_suite.md`

Preserve previous R1 PARTIAL history and add:
`## Task 8B.3-R2 — Driver Audit Closure and Formal Suite`

Include:
1. ChatGPT R1 audit findings;
2. exact driver fixes;
3. dedicated test result;
4. repository suite result;
5. check_setup;
6. input hash confirmation;
7. FORMAL_SUITE_FREEZE hashes;
8. six-case objective table;
9. language summary;
10. runtime summary;
11. B1/B2 mechanical information;
12. review pack contents;
13. visual verdict = `PENDING CHATGPT/USER REVIEW`;
14. no-tuning statement;
15. output-layout requirement remains `ACCEPTED / DEFERRED TO TASK 8B.4`;
16. exact STOP reason if not COMPLETE.

If formal suite does not run, mark all unrun sections `NOT RUN`.

# 17. Phase O — FROM_DSH mandatory

Update:
`handoff/FROM_DSH.md`

Preserve `ARTIFACT-FACTS` verbatim.

Required fields:
```text
Task: 8B.3-R2
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Starting HEAD: 1287d0a2...
Main/origin-main: c45ecbec...
Driver: PASS/FAIL
Driver tests: <result>
Repository tests: <result>
check_setup: <result>
Inputs: 6/6 or other
Formal suite: RUN / NOT RUN / PARTIAL
A1: ...
A2: ...
A3: ...
A4: ...
B1: ...
B2: ...
Review pack: <path/not created>
Visual verdict: PENDING CHATGPT/USER REVIEW
Output-layout proposal: ACCEPTED / DEFERRED TO TASK 8B.4
Report: docs/task8b3_six_image_demo_suite.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit.
```

No autonomous next-task decision.

# 18. Phase P — Git safety and commit

Allowed repo paths only:
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

Any other repo change -> STOP/report.

Stage individually. Never `git add .` or `git add -A`.

If COMPLETE:
`test(demo): close deterministic six-image suite`

If PARTIAL/STOP/FAILED:
`docs(demo): record task8b3-r2 execution stop`

After commit:
```bat
git status --short
git show --stat --oneline HEAD
```

Working tree must be clean.

# 19. Phase Q — Push

Push:
```bat
git push origin eval/task8b3-six-image-demo-suite
```

No force.

Do not merge eval branch to main.

# 20. COMPLETE definition

`TASK 8B.3-R2 COMPLETE` only if:
1. exact start state matched;
2. only harness/tests/docs/handoff changed;
3. test defects A/B fixed;
4. reader-thread/queue timeout architecture implemented;
5. harness no longer explicitly sets HF/Transformers offline flags;
6. silent-child timeout regression passes;
7. all dedicated driver tests pass;
8. full repository suite passes;
9. check_setup READY;
10. six input hashes unchanged;
11. formal freeze recorded;
12. formal suite run exactly once;
13. A1–B2 all attempted in order;
14. no formal sample retry;
15. no prompt/model/config tuning;
16. objective evidence captured;
17. review pack created as applicable;
18. DSH makes no visual semantic verdict;
19. report updated;
20. FROM_DSH updated;
21. output-layout request remains deferred to Task 8B.4;
22. commit/push succeed;
23. tree clean;
24. DSH stops.

`COMPLETE` does not mean 6/6 segmentation quality is correct.

# 21. Final DSH response

```text
TASK 8B.3-R2 COMPLETE / PARTIAL / STOP / FAILED

Branch:
eval/task8b3-six-image-demo-suite

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

Main/origin-main:
c45ecbec... / other

Driver:
PASS / FAIL

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

Then stop.
