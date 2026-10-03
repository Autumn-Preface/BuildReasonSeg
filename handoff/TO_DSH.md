# TO_DSH — Task 8B.3-D1.1: Correct Large-Image Memory Forensics

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `4bcd26b5403bcdaf3ee73f9855c1ffca8db5ac63`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Why this correction task exists

Task 8B.3-D1 is **not yet approved**.

ChatGPT independently audited commit:

```text
4bcd26b5403bcdaf3ee73f9855c1ffca8db5ac63
docs(demo): record six-image failure forensics
```

The commit chain and allowed-file scope are correct, and the Reference / SUCCESS-validity / A2 findings are accepted.

However, two factual defects remain in the D1 memory forensics:

1. D1 enumerated only the explicit per-detection allocation:

```python
global_mask = np.zeros((height_px, width_px), dtype=bool)
```

but did **not** enumerate other full-frame boolean allocations in the same pre-return execution path, especially
`iou_of()` inside proposal merge:

```python
np.logical_and(first, second)
np.logical_or(first, second)
```

Both operands are full-image `global_mask` arrays, so these operations also create full-frame boolean temporaries.

2. D1 described `RC1-DEMO-MEM-01` as a blocker for `>512 px inputs`. This is unsupported and contradicted by
the frozen Demo evidence: 1024×1024 A1/A3/A4 ran through the pipeline (and A2 reached proposal processing).
The observed failure is on the 5000×5000 B1/B2 path. The exact failure-size threshold is **not established**.

This task corrects documentation/forensics only.

# 1. Absolute prohibitions

Do NOT:

- run `predict.py`;
- run `scripts/task8b3_interactive_suite.py`;
- run detector / Qwen / SAM2 / D-B1;
- rerun A1–B2;
- modify external delivery;
- modify product/canonical source;
- modify harness/tests;
- modify configs/thresholds/checkpoints/model assets;
- install packages;
- train/download;
- access final test;
- implement a memory fix;
- implement proposal/reference or mask-validity fixes;
- enter Task 8B.4;
- enter Task 8C.

Only these repository files may change:

```text
docs/task8b3_d1_demo_failure_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 2. Git safety gate

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
HEAD = 4bcd26b5403bcdaf3ee73f9855c1ffca8db5ac63
```

Allowed initial working tree:

- clean; or
- only `M handoff/TO_DSH.md`.

Anything else -> STOP.

Do not reset/stash/clean/discard/switch branches.

# 3. Findings that are already accepted and must not be reworked

Preserve the D1 findings for:

```text
RC1-DEMO-REF-01
RC1-DEMO-MASK-01
RC1-DEMO-PROP-01
```

Specifically preserve:

- A1/A3/A4 selected Reference = largest **eligible** proposal;
- user-facing “largest building” != implementation “largest eligible detected proposal”;
- A1/A4 SUCCESS hard gates are only non-empty + non-padding + directional-centroid;
- A2 = detector/proposal-stage zero-proposal failure;
- A2 NMS warning causality remains NOT CONFIRMED;
- frozen six-image result remains manual end-to-end semantic success 0/6;
- no research metric claim.

Do not change these except for wording needed to keep the report internally consistent.

# 4. Static memory audit — required correction

Read only:

```text
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

Do not execute it.

Enumerate **all relevant full-frame boolean allocation/retention mechanisms in `detect_global()` → `merge_proposals()` → `iou_of()` before `detect_global()` returns**.

At minimum document separately:

## 4.1 Per-raw-detection retained full-frame mask

Inside `detect_global()`:

```python
global_mask = np.zeros((height_px, width_px), dtype=bool)
```

Facts to record:

- allocated once per raw detection that reaches this point;
- for 5000×5000 bool:
  - `25,000,000` bytes;
  - `23.84185791015625 MiB` (~23.84 MiB);
- the array is stored in the `accumulated` entry as `"mask": global_mask`;
- therefore full-frame proposal masks can remain simultaneously retained until merge;
- if `N` such raw proposals are retained, mask payload alone is approximately:

```text
N × 23.84 MiB
```

for 5000×5000, excluding Python/object/other-array overhead.

Do NOT invent `N` for B1/B2 if the frozen artifacts do not expose how many masks had already accumulated before failure.

## 4.2 Pairwise merge temporary arrays

Inside `iou_of()`:

```python
np.logical_and(first, second)
np.logical_or(first, second)
```

Facts to record:

- `first` and `second` are full-frame proposal masks;
- for 5000×5000, each result temporary is also a 23.84 MiB bool array;
- each pairwise IoU comparison performs both a full-frame logical-AND and logical-OR operation;
- the two expressions execute sequentially in the current Python code, so do NOT claim both 23.84 MiB temporaries must coexist simultaneously;
- the proposal merge loop can perform up to `N(N-1)/2` pairwise comparisons for `N` proposals;
- this creates substantial allocation churn in addition to the retained proposal masks.

## 4.3 Other retention/copy facts

State whether `GlobalProposal.global_mask` reuses the accumulated mask object or explicitly copies it in the constructor.

Do not infer a copy if the source does not make one.

If any additional full-frame bool allocation/copy is found in this exact execution path, document it.

# 5. Correct interpretation of the B1/B2 exception

Frozen exception:

```text
Unable to allocate 23.8 MiB for an array with shape (5000, 5000) and data type bool
```

Required conclusion:

```text
Signature match: EXACT_SIGNATURE_MATCH
Unique allocation-site attribution from existing artifacts: NOT CONFIRMED
```

Explanation must state:

- the shape/dtype/size exactly matches a 5000×5000 full-frame bool allocation;
- the explicit `np.zeros((height_px, width_px), dtype=bool)` is one direct matching site;
- `np.logical_and` / `np.logical_or` over two full-frame bool masks can also request a same-shaped bool result;
- unless the frozen transcript/result contains a traceback pinpointing a source line, the existing error text alone does not uniquely prove which matching allocation expression threw the exception.

Do NOT downgrade the broader defect:

```text
RC1-DEMO-MEM-01 = CONFIRMED
```

The confirmed defect is that the large-image proposal path uses O(H×W) full-frame boolean masks per proposal plus full-frame pairwise merge operations and failed on both 5000×5000 Demo inputs.

# 6. Correct the unsupported `>512` wording

Remove every claim equivalent to:

```text
blocker for >512 px inputs
```

Replace it with wording equivalent to:

```text
Observed blocker for the 5000×5000 B1/B2 Demo inputs and a confirmed large-image scalability defect.
The exact image-size / proposal-count failure threshold is not established by the frozen evidence.
Inputs merely larger than 512 are not universally failing; the 1024×1024 Demo samples provide counterexamples.
```

Do not invent a safe maximum image dimension.

# 7. Correct defect taxonomy entry

Keep:

```text
RC1-DEMO-MEM-01 = CONFIRMED
severity = blocker for the demonstrated 5000×5000 large-image Demo path
needs product code change = yes
needs scientific model/checkpoint change = no
```

The evidence cell must mention both:

1. retained per-proposal full-frame bool masks;
2. pairwise full-frame boolean IoU temporaries.

Do not state that `detector.py:265` is proven to be the unique throwing line.

# 8. Priority recommendation

The existing priority may remain:

```text
MEM-01
→ PROP-01
→ REF-01
→ MASK-01
→ Task 8B.4 packaging
→ free manual Demo
```

But justify MEM-01 as:

```text
it blocks the demonstrated 5000×5000 B1/B2 execution path
```

not as blocking every image above 512 px.

No fix is authorized.

# 9. Output-layout handoff wording

In `handoff/FROM_DSH.md`, make the output-layout status unambiguous:

```text
Output-layout proposal: STILL DEFERRED TO TASK 8B.4
```

Do not leave both choice strings joined by `/`.

# 10. Report edits

Update:

```text
docs/task8b3_d1_demo_failure_forensics.md
```

Required corrected sections:

- §7 B1/B2 E502 memory forensics;
- §8 defect taxonomy MEM-01 row;
- §9 dependency/prioritization wording;
- any other sentence containing the unsupported `>512` generalization or unique-line attribution.

Add a short subsection:

```text
### D1.1 audit correction
```

stating that ChatGPT audit found:

1. pairwise `logical_and` / `logical_or` full-frame temporaries had been omitted;
2. `>512 px blocker` was an overgeneralization;
3. the static signature is exact, but unique allocation-site attribution is not proven without a traceback.

Do not alter the frozen R4B sample outcomes.

# 11. Handoff update

Update `handoff/FROM_DSH.md`, preserving `ARTIFACT-FACTS` verbatim.

Required:

```text
Task: 8B.3-D1.1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Starting HEAD: 4bcd26b5403bcdaf3ee73f9855c1ffca8db5ac63
Inference executed: NO
Delivery modified: NO
Product/harness/tests modified: NO
MEM-01: CONFIRMED
5000x5000 bool payload: 25,000,000 B = 23.84 MiB per full-frame bool array
Retained proposal masks: one full-frame bool mask per retained raw proposal
Pairwise merge temporaries: logical_and + logical_or full-frame bool results per comparison
Exception signature: EXACT_SIGNATURE_MATCH
Unique throwing allocation site: NOT CONFIRMED FROM EXISTING ARTIFACTS
Failure threshold: NOT ESTABLISHED
Output-layout proposal: STILL DEFERRED TO TASK 8B.4
Report: docs/task8b3_d1_demo_failure_forensics.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit.
```

# 12. Git gate

Allowed changes exactly:

```text
docs/task8b3_d1_demo_failure_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Before commit:

```bat
git status --short
git diff --check
```

No other path may be staged.

Stage individually.

# 13. Commit and push

If COMPLETE, exact commit message:

```text
docs(demo): correct large-image memory forensics
```

If PARTIAL/STOP/FAILED:

```text
docs(demo): record memory forensics correction stop
```

Push only:

```bat
git push origin eval/task8b3-six-image-demo-suite
```

No force push.

# 14. COMPLETE definition

COMPLETE only if:

- no inference executed;
- no delivery/source/test/harness modification;
- retained per-proposal full-frame masks documented;
- pairwise `logical_and` / `logical_or` full-frame temporaries documented;
- 5000×5000 byte/MiB calculation correct;
- `EXACT_SIGNATURE_MATCH` retained only as a shape/dtype/size signature statement;
- unique throwing allocation site explicitly NOT CONFIRMED absent traceback;
- all `>512 inputs fail` implications removed;
- exact failure threshold marked NOT ESTABLISHED;
- output-layout handoff unambiguous;
- report/handoff corrected;
- only allowed files changed;
- commit/push succeed;
- working tree clean;
- DSH stops.

# 15. Final response

```text
TASK 8B.3-D1.1 COMPLETE / PARTIAL / STOP / FAILED

Branch:
eval/task8b3-six-image-demo-suite

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

Inference executed:
NO

Delivery modified:
NO

Product/harness/tests modified:
NO

MEM-01:
CONFIRMED

5000x5000 bool array:
25,000,000 B = 23.84 MiB

Retained masks:
<summary>

Pairwise IoU temporaries:
<summary>

Exception signature:
EXACT_SIGNATURE_MATCH

Unique throwing allocation site:
NOT CONFIRMED FROM EXISTING ARTIFACTS

Failure threshold:
NOT ESTABLISHED

Unsupported >512 generalization:
REMOVED

Output layout:
STILL DEFERRED TO TASK 8B.4

Report:
docs/task8b3_d1_demo_failure_forensics.md

Handoff:
handoff/FROM_DSH.md

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得修复内存实现、不得运行 Demo、不得进入 Task 8B.4 或 Task 8C。
```
