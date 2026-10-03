# TO_DSH — Task 8B.3-R1: Deterministic Interactive Driver and Mandatory Handoff

> Status: ACTIVE  
> Role boundary: ChatGPT decides; DSH executes only.  
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`  
> Existing local validation branch: `eval/task8b3-six-image-demo-suite`  
> Audited RC1 commit: `c45ecbec7fd293c454ccced22310db32c1542be4`  
> External runnable RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`  
> Runtime Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe`  
> Predecessor: Task 8B.3 PARTIAL — suite not executed because the ad-hoc interactive driver blocked on no-newline Y/N prompts and mixed GBK/UTF-8 decoding.

---

# 0. Permanent DSH reporting rule

This rule applies to this and later DSH tasks unless ChatGPT explicitly overrides it.

For every outcome (`COMPLETE / PARTIAL / STOP / FAILED`), whenever the Git repository is still safe to write, DSH must:

1. update `handoff/FROM_DSH.md`;
2. create/update the task report or stop report in the repository;
3. commit the current task evidence;
4. push the current task branch;
5. stop and wait for ChatGPT.

Do not leave the only useful execution evidence inside local delivery logs.

Only if Git write/push is genuinely unsafe or impossible because of unauthorized pre-existing changes, repository/ref ambiguity, corruption, authentication, or network push failure may this be skipped. In that case print the exact blocking state and preserve a local stop report if possible.

---

# 1. Executor-only rule

You are executor only. ChatGPT has frozen all decisions below.

Do not:
- modify RC1 product/runtime source or `predict.py`;
- modify language frontend, ProgramHead, Qwen suggestion logic or Validator;
- modify detector/Reference/SAM2/D-B1;
- modify thresholds/config/checkpoints;
- edit external delivery source/config directly;
- install packages or PTY/ConPTY libraries;
- change the six fixed prompts;
- retry with alternate wording;
- use Assisted Mode, `--reference-id`, or `--inspect-proposals`;
- access final test, train, download anything, or start Task 8C;
- implement the proposed new per-run output-directory product layout in this task.

The user's proposed per-run output layout is **accepted in principle but deferred to a separate complete delivery iteration after Task 8B.3**, currently reserved as Task 8B.4.

---

# 2. Frozen diagnosis and solution

The failed Task 8B.3 driver had two orchestration bugs:

1. confirmation prompts are emitted without trailing newline (`input(prompt)`);
2. child-process output encoding was not deterministic, causing GBK/UTF-8 mismatch.

The RC1 product must not be changed.

Frozen solution:

```text
subprocess.Popen
stdin=PIPE
stdout=PIPE
stderr=STDOUT
text=False
bufsize=0
PYTHONIOENCODING=utf-8
PYTHONUNBUFFERED=1
PYTHONUTF8=1
binary incremental read (not readline)
incremental UTF-8 decoder
respond only after exact known Y/N prompt is detected
```

No PTY and no package installation.

---

# 3. Exact product prompt strings

Recognize these exact substrings:

```text
DIRECT_PROMPT = 是否按此理解执行？ [Y/N]: 
SUGGESTION_PROMPT = 是否使用建议指令继续？ [Y/N]: 
FALLBACK_PROMPT = 是否进入有限兼容模式？ [Y/N]: 
```

Do not require a newline.

If `FALLBACK_PROMPT` appears:
- send exactly `N\n`;
- classify `LANGUAGE_RUNTIME_ERROR_OR_FALLBACK_REQUEST`;
- continue next sample only if environment remains healthy.

---

# 4. Frozen six cases

Run in this exact order.

## A1
```text
file = inference/input/A1.png
prompt = 找出最大的建筑，然后把它右边离它最近的那栋分割出来
expected_program = largest_to_right_of_to_nearest
```

## A2
```text
file = inference/input/A2.png
prompt = 以面积最大的建筑为参考，分割它左侧最近的建筑
expected_program = largest_to_left_of_to_nearest
```

## A3
```text
file = inference/input/A3.png
prompt = 以最大建筑为准，分割位于其上方且距离最近的建筑
expected_program = largest_to_above_to_nearest
```

## A4
```text
file = inference/input/A4.png
prompt = 请分割最大建筑下方距离最近的一栋建筑
expected_program = largest_to_below_to_nearest
```

## B1
```text
file = inference/input/B1.tif
prompt = 请找出面积最大的建筑，并分割它右边离它最近的那栋楼。
expected_program = largest_to_right_of_to_nearest
```

## B2
```text
file = inference/input/B2.tif
prompt = 最大建筑物的上面，离它最近的那一栋是什么，分割出来
expected_program = largest_to_above_to_nearest
```

Never alter or retry these strings.

---

# 5. Phase A — Recover exact local Git state

Run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse main
git rev-parse eval/task8b3-six-image-demo-suite
git fetch origin main
git rev-parse origin/main
git remote -v
```

Continue only if:

```text
local main = c45ecbec7fd293c454ccced22310db32c1542be4
local eval/task8b3-six-image-demo-suite = c45ecbec7fd293c454ccced22310db32c1542be4
```

Current branch may be `main` or `eval/task8b3-six-image-demo-suite`. If on main, switch exactly to the eval branch.

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Remote `origin/main` may be either:
- `6c2b915dbc64acdeb099d194005d74c7180c95fa`, or
- `c45ecbec7fd293c454ccced22310db32c1542be4`.

If remote main is old `6c2b915...`, push the already audited fast-forward:

```bat
git push origin main
git fetch origin main
```

Then require remote main = `c45ecbec...`.

Any third SHA -> STOP. Do not merge/rebase again.

---

# 6. Phase B — Create deterministic validation driver in workspace

Create exactly:

```text
scripts/task8b3_interactive_suite.py
```

This is a validation harness, not RC1 product code. It may hard-code the six evaluation cases and expected programs as test-oracle data. It must not be copied to delivery.

For each case invoke the real external RC1 CLI using:

```python
subprocess.Popen(
    ...,
    cwd=EXTERNAL_RC1_ROOT,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=False,
    bufsize=0,
    env=child_env,
)
```

`child_env` = `os.environ.copy()` plus exactly:

```python
child_env["PYTHONIOENCODING"] = "utf-8"
child_env["PYTHONUNBUFFERED"] = "1"
child_env["PYTHONUTF8"] = "1"
```

Do not set CUDA/model/device variables.

---

# 7. Phase C — Binary reader and transcript

The Y/N decision path must not use:
- `readline()`;
- `for line in pipe`;
- `communicate(input=...)`.

Use a binary reader thread/loop that reads bytes without waiting for newline, feeds an incremental UTF-8 decoder, and appends decoded characters to a full transcript buffer.

Use equivalent to:

```python
codecs.getincrementaldecoder("utf-8")(errors="replace")
```

Save every case transcript locally:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_transcripts\A1.txt
...
B2.txt
```

Do not Git-commit transcripts.

Each detected prompt instance may be answered only once.

---

# 8. Phase D — Deterministic Y/N decision

## D1 Initial program

Extract initial program only from product output:

```text
[解析] ...   (<program_id>)
```

using equivalent regex:

```regex
\[解析\].*?\(([^()\r\n]+)\)
```

Do not derive it from Chinese keywords.

## D2 Direct confirmation

When `DIRECT_PROMPT` appears:

- if `initial_program == expected_program`: send `Y\n`, status `DIRECT_CORRECT`;
- otherwise: send `N\n`, status `LANGUAGE_ERROR_SUPPORTED_WRONG`.

Never send Y for a different direct program.

## D3 Suggestion confirmation

The product prints one of these exact display forms:

```text
largest -> left_of -> nearest
largest -> right_of -> nearest
largest -> above -> nearest
largest -> below -> nearest
```

The validation harness may map those four display strings to their corresponding four supported program IDs for test-output decoding only.

When `SUGGESTION_PROMPT` appears:
- mapped suggestion == expected -> `Y\n`, `FALLBACK_CORRECT`;
- otherwise -> `N\n`, `FALLBACK_WRONG`.

This mapping must never enter product runtime.

## D4 Fallback prompt

When `FALLBACK_PROMPT` appears:
- send `N\n`;
- status `LANGUAGE_RUNTIME_ERROR_OR_FALLBACK_REQUEST`.

---

# 9. Phase E — Fixed timeout

Per case wall-clock timeout = **15 minutes**.

On timeout:
1. terminate child;
2. wait 10 seconds;
3. kill only if still alive;
4. classify `DRIVER_TIMEOUT`;
5. preserve transcript;
6. rerun `check_setup.py`;
7. continue next sample only if environment remains READY.

Do not retry the timed-out case.

---

# 10. Phase F — Driver unit tests

Create exactly:

```text
tests/test_task8b3_interactive_suite.py
```

Use temporary fake child scripts only. Do not load real models/images.

Required coverage:

1. direct Y/N prompt without newline;
2. UTF-8 Chinese prompt recognized through pipe;
3. correct direct program -> Y;
4. wrong direct program -> N;
5. suggestion prompt without newline -> Y when expected;
6. wrong suggestion -> N;
7. fallback prompt -> N;
8. same prompt answered only once;
9. transcript complete without newline;
10. interaction decision path does not depend on `readline()`;
11. suite contains exactly A1/A2/A3/A4/B1/B2;
12. all frozen prompts/expected programs exactly match this task book.

Fake child must flush its prompt before reading stdin.

Run:

```bat
ENV_PYTHON -m pytest tests/test_task8b3_interactive_suite.py -q
ENV_PYTHON -m pytest tests/ -q
```

Both must PASS before real suite.

If unrelated repo tests fail, do not fix unrelated code; STOP and persist/push report/handoff.

---

# 11. Phase G — Environment/input gate

External RC1:

```bat
ENV_PYTHON check_setup.py
```

Must end `BuildReasonSeg environment: READY`.

Verify all six inputs exist/read and record:
- bytes;
- width;
- height;
- mode/channels if available;
- SHA256.

Any missing/unreadable input -> STOP before suite and persist/push evidence.

---

# 12. Phase H — One formal real-suite run

Run exactly once:

```bat
ENV_PYTHON scripts/task8b3_interactive_suite.py
```

It must execute A1→A2→A3→A4→B1→B2.

Do not manually rerun a case.
Do not edit the driver after observing any semantic result.

If the driver itself crashes due to orchestration after the real suite starts:
- do not patch and rerun;
- STOP;
- persist/push partial evidence;
- wait for ChatGPT.

---

# 13. Phase I — Sample continuation rules

Sample-level failure does not stop later samples if environment remains healthy.

Continue after:
- language error;
- NO_SAFE_SUGGESTION;
- E1xx/E2xx/E3xx/E4xx/E5xx;
- proposal/reference/context/segmentation failure.

Do not retry or tune.

If a runtime anomaly may affect later samples, run `check_setup.py`; if NOT READY, stop suite.

---

# 14. Phase J — Objective evidence

For each case record:
- initial program/confidence;
- language status;
- Y/N sent;
- visual executed?;
- exit code/result/error;
- Reference ID;
- mask/overlay/diagnostics paths;
- tile count;
- raw proposal count;
- merged proposal count;
- `directional_candidates_outside_rc1_context` if present.

Do not make semantic visual verdicts.

A runtime SUCCESS is:

```text
AUTOMATIC_RUNTIME_SUCCESS_PENDING_VISUAL_REVIEW
```

---

# 15. Phase K — Review pack

Create local-only:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\review_task8b3
```

Create `INDEX.md`.

For successful samples copy overlays as:

```text
A1_overlay_review.png
A2_overlay_review.png
A3_overlay_review.png
A4_overlay_review.png
B1_overlay_review.png
B2_overlay_review.png
```

Only if produced.

For failed samples, if a `global_proposals.png` exists, copy as:

```text
<ID>_global_proposals_review.png
```

Do not commit review pack.

---

# 16. Phase L — User output-layout proposal

Record this accepted future requirement in the report:

```text
Each predict run should own one independent directory under inference/output,
with subdirectories diagnostics/, masks/, overlays/.
```

Status:

```text
ACCEPTED / DEFERRED TO SEPARATE DELIVERY ITERATION
```

Do not implement it in 8B.3-R1. It requires a later complete canonical-source→tests→sync delivery iteration (reserved as Task 8B.4).

---

# 17. Phase M — Report always required

Create/update:

```text
docs/task8b3_six_image_demo_suite.md
```

This file must be created even if the task STOPs before real suite.

If STOP occurs, mark unrun sections `NOT RUN` and include exact STOP phase/evidence.

If suite runs, include:
- deterministic driver design;
- driver tests;
- fixed suite/input identities;
- per-case table;
- language/runtime summaries;
- B1/B2 mechanical evidence;
- review pack;
- Visual Verdict = `PENDING CHATGPT/USER REVIEW`;
- no-tuning statement;
- output-layout requirement;
- next gate.

---

# 18. Phase N — FROM_DSH always required

Update `handoff/FROM_DSH.md` on every terminal state and preserve `ARTIFACT-FACTS` verbatim.

Required fields:

```text
Task: 8B.3-R1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Base audited RC1: c45ecbec...
Remote main: <actual>
Driver tests: <actual/not run>
Repository tests: <actual/not run>
check_setup: <actual/not run>
Inputs: <actual/not run>
A1: ...
A2: ...
A3: ...
A4: ...
B1: ...
B2: ...
Review pack: <path/not created>
Report: docs/task8b3_six_image_demo_suite.md
Output-layout proposal: ACCEPTED / DEFERRED TO SEPARATE DELIVERY ITERATION
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit.
```

No future technical decision beyond awaiting ChatGPT audit.

---

# 19. Phase O — Git change gate

Allowed repository changes only:

```text
handoff/TO_DSH.md
handoff/FROM_DSH.md
docs/task8b3_six_image_demo_suite.md
scripts/task8b3_interactive_suite.py
tests/test_task8b3_interactive_suite.py
```

No canonical RC1 product source/config may change.

Run:

```bat
git status --short
git diff --check
git diff
```

Any other repo path -> STOP.

---

# 20. Phase P — Commit and push even on safe STOP

Stage allowed files individually. Never `git add .` or `git add -A`.

For COMPLETE commit exactly:

```text
test(demo): add deterministic six-image interactive suite
```

For PARTIAL/STOP after report/evidence exists commit exactly:

```text
docs(demo): record task8b3 execution stop
```

Push current branch:

```bat
git push -u origin eval/task8b3-six-image-demo-suite
```

No force push.

A safe PARTIAL/STOP must still be pushed so ChatGPT can audit it.

---

# 21. Completion rule

COMPLETE only if:
- remote main reconciled to audited RC1;
- deterministic driver + tests pass;
- repo suite passes;
- check_setup READY;
- 6/6 inputs valid;
- six cases attempted once in order;
- no prompt retry/tuning;
- Y/N obeyed frozen program comparison;
- evidence/review pack/report/handoff produced;
- no product code changed;
- output-layout request recorded but not implemented;
- commit/push succeed;
- working tree clean;
- DSH stops.

Otherwise PARTIAL/STOP, but still report/FROM_DSH/commit/push when Git-safe.

---

# 22. Final DSH reply

```text
TASK 8B.3-R1 COMPLETE / PARTIAL / STOP / FAILED

Branch:
eval/task8b3-six-image-demo-suite

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

Remote main:
<sha>

Driver:
PASS / FAIL / NOT RUN

Driver tests:
<result>

Repository tests:
<result>

check_setup:
READY / NOT READY / NOT RUN

Inputs:
6/6 / other / NOT RUN

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

Report:
docs/task8b3_six_image_demo_suite.md

Handoff:
handoff/FROM_DSH.md

Output layout proposal:
ACCEPTED / DEFERRED TO SEPARATE DELIVERY ITERATION

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得进入 Assisted Mode、Task 8B.4、Task 8C 或任何新研发。
```

Then stop.
