# TO_DSH — Task 8B.3-D1.2: Normalize Forensics Report and Handoff

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `e9bbfa4fc536335c8cbfd3f1afecdd8d6dedae6f`

# 0. Purpose

Task 8B.3-D1.1 corrected the memory-forensics substance, but the canonical report still contains stale pre-correction text
in §7/§8 and then appends the correction later. This leaves the report internally inconsistent.

This task performs documentation normalization only.

No inference, source change, test change, delivery change, or defect fix is authorized.

# 1. Strict prohibitions

Do NOT:

- run `predict.py`;
- run the six-image suite;
- run detector/Qwen/SAM2/D-B1;
- modify any product/canonical/runtime/harness/test source;
- modify external delivery;
- modify configs/thresholds/checkpoints/model assets;
- install packages;
- train/download;
- access final test;
- implement MEM-01 / PROP-01 / REF-01 / MASK-01 fixes;
- implement Task 8B.4;
- enter Task 8C;
- rewrite or squash Git history.

Only these repository paths may change:

```text
docs/task8b3_d1_demo_failure_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 2. Git safety

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
HEAD = e9bbfa4fc536335c8cbfd3f1afecdd8d6dedae6f
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else -> STOP.

Do not reset/stash/clean/discard.

# 3. Normalize report encoding

Rewrite:

```text
docs/task8b3_d1_demo_failure_forensics.md
```

as UTF-8 **without BOM**.

Do not alter frozen A1–B2 facts, REF-01/MASK-01/PROP-01 conclusions, or the qualitative `0/6` Demo audit.

# 4. Replace §7 directly — do not append another correction

Replace the existing §7 with one internally consistent section.

Required facts:

```text
B1/B2 image size = 5000 × 5000
exception = Unable to allocate 23.8 MiB for an array with shape (5000, 5000) and data type bool
one full-frame bool array = 25,000,000 B = 23.84 MiB
```

Document both memory mechanisms:

## A. Retained full-frame proposal masks

```python
global_mask = np.zeros((height_px, width_px), dtype=bool)
```

- one allocation per raw detection reaching this point;
- stored in `accumulated` as `"mask": global_mask`;
- retained masks may coexist until merge;
- approximate payload = `N × 23.84 MiB` at 5000×5000, excluding overhead;
- `N` must not be invented;
- `GlobalProposal.global_mask` reuses the same mask object; no explicit constructor copy.

## B. Pairwise merge temporaries

`iou_of()` uses:

```python
np.logical_and(first, second)
np.logical_or(first, second)
```

- operands are full-frame masks;
- each result is also a full-frame bool array;
- each result is 23.84 MiB at 5000×5000;
- expressions execute sequentially, so do not claim both temporaries necessarily coexist;
- for `N` proposals, pairwise merge can perform up to `N(N-1)/2` comparisons.

Required conclusion:

```text
Exception signature: EXACT_SIGNATURE_MATCH (shape/dtype/size)
Unique throwing allocation site: NOT CONFIRMED FROM EXISTING ARTIFACTS
RC1-DEMO-MEM-01: CONFIRMED
Exact failure threshold: NOT ESTABLISHED
```

Do not describe `detector.py:265` as the uniquely proven throwing line.

# 5. Replace MEM-01 row in §8

The MEM-01 taxonomy row must be internally consistent with §7.

Required meaning:

```text
status = CONFIRMED
affected = B1, B2
layer = large-image runtime/engineering
severity = blocker for demonstrated 5000×5000 B1/B2 Demo path
needs product code change = yes
needs scientific model/checkpoint change = no
```

Evidence must mention:

- retained per-proposal full-frame masks;
- pairwise full-frame `logical_and` / `logical_or` temporaries;
- exact exception signature;
- unique throwing expression not confirmed.

Remove all wording equivalent to:

```text
blocker for >512 px inputs
```

# 6. Normalize §9 priority wording

The order may remain:

```text
MEM-01 → PROP-01 → REF-01 → MASK-01 → Task 8B.4 → free manual Demo
```

But MEM-01 justification must say only:

```text
blocks the demonstrated 5000×5000 B1/B2 path
```

and:

```text
exact size/proposal-count failure threshold is not established
```

Do not claim all images larger than 512 fail.

# 7. Fold D1.1 correction into the main report

After replacing §7/§8/§9, remove the long appended:

```text
### D1.1 audit correction
```

section, or reduce it to a short historical note that contains **no duplicate technical conclusions**.

Preferred historical note:

```text
### Audit history
Task 8B.3-D1.1 corrected the original D1 memory-forensics report by adding pairwise full-frame IoU temporaries,
removing the unsupported >512-px generalization, and separating exact allocation-signature matching from unique
throwing-site attribution. The normalized §7–§9 above are authoritative.
```

There must be only one authoritative technical statement for MEM-01 in the document.

# 8. Normalize FROM_DSH encoding and ARTIFACT-FACTS

Rewrite:

```text
handoff/FROM_DSH.md
```

as UTF-8 **without BOM**.

The complete block from:

```text
<!-- ARTIFACT-FACTS:BEGIN -->
...
<!-- ARTIFACT-FACTS:END -->
```

must be byte-for-byte text-equivalent to the pre-D1.1 block:
- no BOM before the opening marker;
- no edits inside the block;
- same ordering and values.

Do not change any ARTIFACT-FACTS value.

# 9. FROM_DSH content

Set:

```text
Task: 8B.3-D1.2
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Starting HEAD: e9bbfa4fc536335c8cbfd3f1afecdd8d6dedae6f
Inference executed: NO
Delivery modified: NO
Product/harness/tests modified: NO
MEM-01: CONFIRMED
Exception signature: EXACT_SIGNATURE_MATCH
Unique throwing allocation site: NOT CONFIRMED FROM EXISTING ARTIFACTS
Failure threshold: NOT ESTABLISHED
5000×5000 path: BLOCKED IN B1/B2 BASELINE
>512 universal-failure claim: REMOVED
Report consistency: NORMALIZED
Encoding: UTF-8 WITHOUT BOM
Output-layout proposal: STILL DEFERRED TO TASK 8B.4
Report: docs/task8b3_d1_demo_failure_forensics.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit.
```

Also preserve accepted D1 facts for REF-01 / MASK-01 / PROP-01 in a concise paragraph or table.

# 10. Validation

Before commit, perform text-only checks.

Verify:

1. `docs/task8b3_d1_demo_failure_forensics.md` first bytes are NOT UTF-8 BOM (`EF BB BF`);
2. `handoff/FROM_DSH.md` first bytes are NOT UTF-8 BOM;
3. report contains exactly one authoritative MEM-01 forensic section;
4. report contains:
   - `EXACT_SIGNATURE_MATCH`;
   - `NOT CONFIRMED FROM EXISTING ARTIFACTS`;
   - `NOT ESTABLISHED`;
   - `logical_and`;
   - `logical_or`;
5. report does not contain any active claim equivalent to `blocker for >512 px inputs`;
6. `ARTIFACT-FACTS` content and values are unchanged.

Do not run pytest; no code changed.

# 11. Git gate

Allowed changed paths exactly:

```text
docs/task8b3_d1_demo_failure_forensics.md
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

Do not use `git add .` or `git add -A`.

# 12. Commit / push

Create **one new commit** for D1.2 with exact message:

```text
docs(demo): normalize memory forensics report
```

Do not amend, squash, reset, or rewrite the previous two D1.1 commits.

Push:

```bat
git push origin eval/task8b3-six-image-demo-suite
```

No force push.

# 13. COMPLETE definition

COMPLETE only if:

- no inference/source/test/delivery change;
- report §7/§8/§9 directly corrected;
- stale contradictory MEM-01 text removed;
- no duplicate authoritative correction section remains;
- both report and FROM_DSH are UTF-8 without BOM;
- ARTIFACT-FACTS opening marker and block restored without BOM/edit;
- unique throwing site remains NOT CONFIRMED;
- exact failure threshold remains NOT ESTABLISHED;
- >512 universal-failure claim absent;
- only allowed paths changed;
- exactly one new D1.2 commit created;
- push succeeds;
- working tree clean;
- DSH stops.

# 14. Final response

```text
TASK 8B.3-D1.2 COMPLETE / PARTIAL / STOP / FAILED

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

Report consistency:
NORMALIZED / FAIL

Report encoding:
UTF-8 WITHOUT BOM / FAIL

FROM_DSH encoding:
UTF-8 WITHOUT BOM / FAIL

ARTIFACT-FACTS:
PRESERVED / FAIL

MEM-01:
CONFIRMED

Exception signature:
EXACT_SIGNATURE_MATCH

Unique throwing allocation site:
NOT CONFIRMED FROM EXISTING ARTIFACTS

Failure threshold:
NOT ESTABLISHED

>512 universal-failure claim:
REMOVED

Output layout:
STILL DEFERRED TO TASK 8B.4

Report:
docs/task8b3_d1_demo_failure_forensics.md

Handoff:
handoff/FROM_DSH.md

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得修复缺陷、不得运行 Demo、不得进入 Task 8B.4 或 Task 8C。
```
