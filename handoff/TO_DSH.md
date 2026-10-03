# TO_DSH — Task 8B.3-R4B: One Formal Six-Image Demo Run

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `b579148df7aac9c979265b2e4da5b91c9d41d34b`
> Main/origin-main: `c45ecbec7fd293c454ccced22310db32c1542be4`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Runtime Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe`

# 0. Permanent reporting rule

For COMPLETE / PARTIAL / STOP / FAILED, if Git is safe:
1. update `docs/task8b3_six_image_demo_suite.md`;
2. update `handoff/FROM_DSH.md`;
3. commit;
4. push current branch;
5. stop.

# 1. Purpose

This task performs the **single formal first-pass six-image Automatic Demo suite**.

All harness and repository gates are already accepted:
- dedicated harness tests: 20 passed;
- full repository suite: 1555 passed;
- harness code frozen;
- formal suite has never been run before this task.

This task is execution/reporting only.

# 2. Strict prohibitions

Do not:
- modify `scripts/task8b3_interactive_suite.py`;
- modify `tests/test_task8b3_interactive_suite.py`;
- modify any product/canonical/delivery source or config;
- modify prompts/programs;
- rerun an individual sample;
- rerun the formal suite;
- use Assisted Mode;
- use `--reference-id`;
- use `--inspect-proposals`;
- tune thresholds/config;
- install packages;
- train/download;
- access final test;
- enter Task 8B.4 or Task 8C.

The formal suite may be invoked **exactly once**.

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
HEAD = b579148df7aac9c979265b2e4da5b91c9d41d34b
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else -> STOP and report/push if Git-safe.

# 4. Runtime gate

Run external:

```bat
cd /d C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
ENV_PYTHON check_setup.py
```

Required:

```text
BuildReasonSeg environment: READY
```

and live Ultralytics runtime must still be `8.4.164`.

If not READY -> STOP. Do not repair.

# 5. Six-input identity gate

Verify the six files exist, are readable, and SHA256 exactly match:

```text
A1.png
8a4b459d65773a7dfb0ffcc509c26b5d3a7cea23cd759cdadf94cd46be84c227

A2.png
10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f

A3.png
f3cd05870385bd7f978b345b703d407cca757d909fc7be64011449ca711b19fd

A4.png
a3962ed18467997366de7b070d7ae39b3ac9916ca01497014ad5c03b434775a3

B1.tif
8b68c9e2fe3438b511e482998f34f772a870b99020eb5e7fc1b83b08b4828e31

B2.tif
c91663edd7abfaa3d1f198a9903ce28b5d8c99f80028fc24bd883e285b62319a
```

Any mismatch -> STOP. Do not substitute images.

# 6. Formal freeze record

Before running the suite, append to the task report:

```text
FORMAL_SUITE_FREEZE_R4B
branch_head = b579148df7aac9c979265b2e4da5b91c9d41d34b
driver_sha256 = <sha256 of scripts/task8b3_interactive_suite.py>
tests_sha256 = <sha256 of tests/test_task8b3_interactive_suite.py>
check_setup = READY
A1_sha256 = 8a4b459...
A2_sha256 = 10286b1...
A3_sha256 = f3cd058...
A4_sha256 = a3962ed...
B1_sha256 = 8b68c9e...
B2_sha256 = c91663e...
repository_gate = 1555 passed
```

After this line is recorded:
- no code/test/prompt edit;
- no formal sample retry;
- no suite rerun.

# 7. Run the formal suite exactly once

Return to repo root and run exactly once:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
ENV_PYTHON scripts/task8b3_interactive_suite.py
```

Do not invoke that script a second time.

Expected order:

```text
A1 -> A2 -> A3 -> A4 -> B1 -> B2
```

Sample-level language/model failure is valid evidence and does not authorize a retry.

If the driver itself crashes or stops before all six:
- do not patch;
- do not rerun;
- mark PARTIAL;
- preserve existing transcripts/results;
- continue to report/commit/push.

# 8. Objective evidence source

After the single run, use:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_suite_results.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_transcripts\
```

and each normal diagnostics `result.json`.

For each case record:
- expected program;
- initial program;
- initial confidence;
- Y/N sent;
- language status;
- visual executed yes/no;
- exit code;
- runtime result/error;
- Reference ID;
- mask area;
- tile count;
- raw proposal count;
- merged proposal count;
- mask path;
- overlay path;
- diagnostics path.

Do not infer values that are absent.

Runtime `SUCCESS` must be described as:

```text
AUTOMATIC_RUNTIME_SUCCESS_PENDING_VISUAL_REVIEW
```

Do not call it visually correct.

# 9. Review pack

Create local-only:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\review_task8b3
```

This is a runtime result/review bundle, not a source modification.

Create:

```text
review_task8b3\INDEX.md
```

For every A1–B2 include:
- frozen prompt;
- expected program;
- language status;
- runtime status;
- mask path if any;
- overlay path if any;
- diagnostics path;
- error if any.

For every sample that actually produced an overlay, copy it as:

```text
A1_overlay_review.png
A2_overlay_review.png
A3_overlay_review.png
A4_overlay_review.png
B1_overlay_review.png
B2_overlay_review.png
```

Only copy files that actually exist.

For failed samples, if normal execution already produced `global_proposals.png`, copy it as:

```text
<ID>_global_proposals_review.png
```

Do not launch `--inspect-proposals`.

Do not Git-track review pack.

# 10. Report

Update:

```text
docs/task8b3_six_image_demo_suite.md
```

Add:

```text
## Task 8B.3-R4B — One Formal Six-Image Demo Run
```

Include:
1. formal freeze record;
2. statement `FORMAL SUITE INVOCATIONS = 1`;
3. exact per-case table;
4. language summary;
5. runtime summary;
6. A-group vs B-group runtime counts;
7. B1/B2 tile/proposal evidence if present;
8. review pack file list;
9. Visual Verdict = `PENDING CHATGPT/USER REVIEW`;
10. no-retry/no-tuning statement;
11. output layout = `ACCEPTED / DEFERRED TO TASK 8B.4`;
12. exact STOP reason if PARTIAL.

# 11. FROM_DSH

Update `handoff/FROM_DSH.md`, preserving `ARTIFACT-FACTS` verbatim.

Required:

```text
Task: 8B.3-R4B
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Starting HEAD: b579148d...
check_setup: READY / other
Inputs: 6/6 / other
Formal suite invocations: 1 / 0
A1: language=<...> runtime=<...>
A2: language=<...> runtime=<...>
A3: language=<...> runtime=<...>
A4: language=<...> runtime=<...>
B1: language=<...> runtime=<...>
B2: language=<...> runtime=<...>
Review pack: <path or NOT CREATED>
Visual verdict: PENDING CHATGPT/USER REVIEW
Output-layout proposal: ACCEPTED / DEFERRED TO TASK 8B.4
Report: docs/task8b3_six_image_demo_suite.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT/user visual review.
```

# 12. Git change gate

Allowed repository changes only:

```text
handoff/TO_DSH.md
handoff/FROM_DSH.md
docs/task8b3_six_image_demo_suite.md
```

No script/test/product file may change.

Run:

```bat
git status --short
git diff --check
git diff
```

Any other repo path changed -> STOP. Do not discard it; report.

# 13. Commit/push

Stage only the three allowed paths individually.

If the suite was invoked once and all six cases were attempted in order, commit exactly:

```text
docs(demo): record formal six-image suite
```

This commit message is used even if some samples fail at model/runtime level.

If the driver/suite was not invoked or stopped before all six due orchestration/environment failure:

```text
docs(demo): record formal suite partial
```

Push:

```bat
git push origin eval/task8b3-six-image-demo-suite
```

No force.

Working tree must be clean.

# 14. COMPLETE definition

COMPLETE means the formal evaluation procedure completed, not that 6/6 outputs are correct.

COMPLETE only if:
1. exact starting state;
2. check_setup READY;
3. six hashes match;
4. freeze recorded;
5. suite invoked exactly once;
6. all six attempted in order;
7. no individual retry;
8. no code/prompt/config tuning;
9. results recorded objectively;
10. review pack created;
11. no semantic visual verdict made;
12. only report/handoff/taskbook changed in Git;
13. commit/push succeed;
14. clean tree;
15. DSH stops.

# 15. Final response

```text
TASK 8B.3-R4B COMPLETE / PARTIAL / STOP / FAILED

Branch:
eval/task8b3-six-image-demo-suite

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

check_setup:
READY / NOT READY / NOT RUN

Inputs:
6/6 / other

Formal suite invocations:
1 / 0

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

Runtime successes pending visual review:
<n>/6

Runtime failures:
<n>/6

Review pack:
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\review_task8b3
or NOT CREATED

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

等待 ChatGPT/用户审核；不得进入 Assisted Mode、Task 8B.4、Task 8C 或任何新研发。
```
