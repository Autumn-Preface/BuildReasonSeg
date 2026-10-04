# TO_DSH — Task 8B.3-M1B.3: Real B1/B2 5000×5000 Memory Gate

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `f73b4294667219f093aa6333c3fd5d4a36832814`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

Determine whether `RC1-DEMO-MEM-01` is actually fixed in the runnable external RC1 delivery.

Frozen entering evidence:

```text
canonical compact runtime dedicated tests = 32 passed
5000×5000 synthetic detect_global guard = PASS
controlled canonical → external sync = PASS
external full regression = 116 passed
external manifest = 135/135 PASS
```

The original real failure was:

```text
B1/B2 input size = 5000×5000
Unable to allocate 23.8 MiB for an array with shape (5000, 5000) and data type bool
RC1 error = E502
```

This task runs the two original 5000×5000 Demo inputs through the real external prediction path, once each, with the
same frozen prompts. It is a memory/scalability closure gate only.

Do NOT fix PROP-01, REF-01 or MASK-01 in this task.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = f73b4294667219f093aa6333c3fd5d4a36832814
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT modify:
- canonical runtime;
- canonical tests;
- canonical `source_manifest.json`;
- external runtime source;
- external tests;
- external `source_manifest.json`;
- model weights/components;
- detector settings;
- tiling/overlap/imgsz/conf/max_det;
- duplicate IoU threshold;
- merge winner/stable-ID rules;
- Reference selector;
- reasoning context;
- ProgramHead/SAM2/D-B1;
- SUCCESS validity.

Do NOT:
- run A1/A2/A3/A4;
- run the six-image Demo as a batch;
- manually repair output;
- tune any threshold;
- add retry logic;
- rerun B1 or B2;
- implement Task 8B.4 output restructuring;
- fix PROP-01 / REF-01 / MASK-01;
- update `main`;
- force push.

Normal inference outputs under external `inference/output` / diagnostics are allowed.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_m1b3_large_image_memory_gate.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No functional repository file may change.

# 4. Preflight

Before inference, verify external manifest again:

```text
135/135 PASS
```

and verify external `source_manifest.json` remains byte-identical to canonical.

Verify input files exist:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\B1.tif
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\B2.tif
```

Read image metadata only and record dimensions.

Required:

```text
B1 = 5000×5000
B2 = 5000×5000
```

If either input is missing or dimensions differ:
- do not run inference;
- STOP and report.

# 5. Frozen prompts / expected programs

B1:

```text
prompt:
请找出面积最大的建筑，并分割它右边离它最近的那栋楼。

expected program:
largest_to_right_of_to_nearest
```

B2:

```text
prompt:
最大建筑物的上面，离它最近的那一栋是什么，分割出来

expected program:
largest_to_above_to_nearest
```

Do not alter punctuation or wording.

# 6. Interactive-language policy

The goal is to reach the real visual pipeline, not to retune language behavior.

If the CLI asks:

```text
是否进入有限兼容模式？ [Y/N]
```

because Qwen is unavailable at runtime, answer:

```text
Y
```

If the CLI offers a Qwen suggestion / replacement command:
- continue with `Y` only if the displayed/suggested executable program is exactly the expected program in §5;
- otherwise terminate that run and record `LANGUAGE_GATE_MISMATCH`.

Do not invent or edit a command manually.

If a run terminates before detector/proposal execution because of a language failure/mismatch, MEM-01 is `NOT EVALUATED`
for that image.

# 7. Run B1 exactly once

From:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

execute normal prediction for B1 with the exact frozen prompt.

Equivalent command:

```bat
ENV_PYTHON predict.py --image inference/input/B1.tif --prompt "请找出面积最大的建筑，并分割它右边离它最近的那栋楼。"
```

Invocation count for B1 must be exactly 1.

Capture stdout/stderr/exit code without modifying program behavior.

After completion, record all available evidence:

```text
exit code
reported status
reported error code/reason, if any
executed/parsed program
language mode if printed/diagnosed
raw proposal count
merged proposal count
reference_id if available
mask_area if available
result/diagnostic/output paths
whether detector/proposal stage completed
whether SAM2/D-B1 stage was reached
```

Search the captured run output and produced diagnostics for:

```text
Unable to allocate 23.8 MiB
shape (5000, 5000)
MemoryError
E502
```

Important:
- an unrelated later `E502` must not automatically be called MEM-01; record the exact reason/signature;
- the old full-frame allocation signature is the key regression signature.

## B1 stop rule

If B1 shows the old 5000×5000 bool allocation failure, a `MemoryError`, or fails inside proposal compaction/merge before
the detector/proposal stage can complete:
- do NOT run B2;
- mark MEM gate FAIL;
- report and STOP.

If B1 reaches and completes the detector/proposal stage without the old memory signature, proceed to B2 even if a
later, clearly unrelated PROP/REF/MASK or downstream issue occurs.

If B1 never reaches detector/proposal because of language gating, STOP with `MEM-01 NOT EVALUATED`.

# 8. Run B2 exactly once

Only if B1 satisfies the proceed condition above.

Run:

```bat
ENV_PYTHON predict.py --image inference/input/B2.tif --prompt "最大建筑物的上面，离它最近的那一栋是什么，分割出来"
```

Invocation count for B2 must be exactly 1.

Capture the same evidence as B1.

Apply the same memory-signature checks and classification.

No rerun.

# 9. MEM-01 closure rule

Use exactly these verdicts.

## `MEM01_REAL_GATE_PASS`

Only if BOTH B1 and B2:

1. are confirmed 5000×5000 inputs;
2. enter the real detector/proposal path;
3. complete global proposal generation/merge far enough to establish that the old per-proposal full-frame allocation
   failure did not occur;
4. contain no old signature:
   `Unable to allocate 23.8 MiB for an array with shape (5000, 5000) and data type bool`;
5. contain no `MemoryError` or equivalent allocation failure attributable to proposal storage/merge.

End-to-end SUCCESS is desirable but is NOT required for the MEM-only verdict if a later error is unambiguously outside
the proposal-memory path. Any later failure must be recorded without reclassifying PROP/REF/MASK.

If all five conditions hold:

```text
RC1-DEMO-MEM-01 = CLOSED
```

## `MEM01_REAL_GATE_FAIL`

If either evaluated image reproduces the old allocation failure or another proposal-memory allocation failure.

Then:

```text
RC1-DEMO-MEM-01 = OPEN
```

## `MEM01_NOT_EVALUATED`

If an image cannot reach the detector/proposal path because of language/input/setup issues.

Then MEM-01 remains open pending valid evidence.

# 10. Do not evaluate semantic quality here

Do NOT manually judge:
- whether the selected largest building is visually correct;
- target-mask completeness;
- qualitative overlay correctness.

Those belong to REF-01 / MASK-01 and the later fixed Demo.

You may record runtime status/counts only.

# 11. Post-run integrity

After the allowed run(s), verify external manifest again:

```text
135/135 PASS
```

and `source_manifest.json` remains byte-identical to canonical.

Runtime output/diagnostic files are allowed and excluded from this manifest gate.

Back in repository:

```bat
git status --short
git diff --check
```

Only §3 paths may be tracked changes.

# 12. Report

Create:

```text
docs/task8b3_m1b3_large_image_memory_gate.md
```

Required sections:

1. Task / scope
2. Starting HEAD
3. External preflight manifest
4. B1/B2 input existence and dimensions
5. Exact prompts and expected programs
6. B1 invocation count and complete runtime evidence
7. B2 invocation count and complete runtime evidence, or explicit NOT RUN reason
8. Old memory-signature search results
9. Detector/proposal-stage completion evidence
10. Any later unrelated failure, clearly separated from MEM-01
11. Post-run manifest
12. MEM verdict:
   - `MEM01_REAL_GATE_PASS`
   - `MEM01_REAL_GATE_FAIL`
   - `MEM01_NOT_EVALUATED`
13. `RC1-DEMO-MEM-01 = CLOSED / OPEN`
14. `PROP-01 / REF-01 / MASK-01 = UNCHANGED`
15. No semantic visual-quality verdict
16. Next action = awaiting ChatGPT audit.

# 13. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-M1B.3
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: f73b4294667219f093aa6333c3fd5d4a36832814
External preflight manifest: 135/135 PASS / FAIL
B1 dimensions: 5000x5000 / other
B1 invocation count: 1 / 0
B1 detector/proposal stage completed: YES / NO
B1 old memory signature: ABSENT / PRESENT / NOT EVALUATED
B1 final runtime status: <status/error>
B2 dimensions: 5000x5000 / other / NOT CHECKED
B2 invocation count: 1 / 0
B2 detector/proposal stage completed: YES / NO / NOT RUN
B2 old memory signature: ABSENT / PRESENT / NOT EVALUATED
B2 final runtime status: <status/error/not run>
Post-run external manifest: 135/135 PASS / FAIL / NOT RUN
MEM gate verdict: MEM01_REAL_GATE_PASS / MEM01_REAL_GATE_FAIL / MEM01_NOT_EVALUATED
RC1-DEMO-MEM-01: CLOSED / OPEN
PROP-01 / REF-01 / MASK-01: UNCHANGED / UNCHANGED / UNCHANGED
Semantic visual-quality verdict: NOT PERFORMED
Report: docs/task8b3_m1b3_large_image_memory_gate.md
Next action: Awaiting ChatGPT audit.
```

# 14. Git / commit / push

If `MEM01_REAL_GATE_PASS`:

```text
test(rc1): close large-image proposal memory defect
```

If FAIL / NOT EVALUATED:

```text
docs(rc1): record large-image memory gate stop
```

Push current fix branch, no force.

# 15. COMPLETE definition

COMPLETE only if:
- exact branch/starting HEAD;
- external preflight 135/135 PASS;
- B1/B2 both confirmed 5000×5000;
- B1 invoked exactly once;
- B2 invoked exactly once only when B1 proceed condition is met;
- no reruns;
- both real detector/proposal paths complete without proposal-memory failure;
- old exact allocation signature absent from both;
- post-run external manifest 135/135 PASS;
- no functional code/test/manifest changes;
- no semantic visual-quality judgment;
- report/handoff committed and pushed;
- repository tracked tree clean;
- stop.

# 16. Final response

```text
TASK 8B.3-M1B.3 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

External preflight manifest:
135/135 PASS / other

B1:
dimensions = 5000x5000
invocations = 1
proposal stage = YES / NO
old memory signature = ABSENT / PRESENT
final status = <...>

B2:
dimensions = 5000x5000
invocations = 1 / 0
proposal stage = YES / NO / NOT RUN
old memory signature = ABSENT / PRESENT / NOT EVALUATED
final status = <...>

Post-run external manifest:
135/135 PASS / other

MEM gate verdict:
MEM01_REAL_GATE_PASS / MEM01_REAL_GATE_FAIL / MEM01_NOT_EVALUATED

RC1-DEMO-MEM-01:
CLOSED / OPEN

PROP-01 / REF-01 / MASK-01:
UNCHANGED / UNCHANGED / UNCHANGED

Semantic visual-quality verdict:
NOT PERFORMED

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得进入 PROP-01、REF-01、MASK-01、Task 8B.4 或 Task 8C。
```
