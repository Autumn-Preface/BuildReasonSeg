# TO_DSH — Task 8B.3-M1A.2B-R2: Fix Final Two Test Defects

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `66d00a95231fb516e4f137dd812077b985f21df4`

# 0. Purpose

Fix exactly the two remaining dedicated-test defects from M1A.2B-R1.

Current dedicated result:

```text
2 failed, 30 passed
```

Audit says both failures are test defects:
1. merge-equivalence test still uses an expanded copied mask while asserting area 1600;
2. large-image guard uses a 50×60 mask (area 3000) but asserts 1500, and currently does not force pairwise IoU.

Runtime is frozen and must not change.

# 1. Strict prohibitions

Do NOT modify:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
```

Do NOT modify external delivery, models, thresholds, tiling, merge/reference policies.

Do NOT run real inference or full canonical suite.

If the single dedicated rerun fails:
- no runtime patch;
- no second rerun;
- record exact failure and STOP.

# 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = 66d00a95231fb516e4f137dd812077b985f21df4
```

Allowed initial tree: clean or only `M handoff/TO_DSH.md`.

# 3. Allowed changed paths

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

# 4. Fix merge-equivalence test

In:

```python
test_merge_equivalence_with_compact_masks()
```

replace the bad construction:

```python
b = a.copy()
b[104:144, 104:144] = True
```

with an independently allocated equal-area block:

```python
b = np.zeros((512, 512), dtype=bool)
b[104:144, 104:144] = True
```

Now:
- area(a) = 1600;
- area(b) = 1600;
- IoU > 0.50;
- confidence tie-break remains meaningful;
- existing `[1600,1600]` and `<=1600` assertions are valid.

Do not weaken those assertions.

# 5. Fix and strengthen 5000×5000 detect_global guard

Current 50×60 block area is 3000, not 1500.

Use two overlapping synthetic detections so product pairwise merge is actually exercised.

Preferred:

```python
mask1 = np.zeros((512,512), dtype=bool)
mask1[100:150, 200:260] = True      # 3000 px

mask2 = np.zeros((512,512), dtype=bool)
mask2[104:154, 204:264] = True      # 3000 px, overlap > 0.50
```

Monkeypatched `detect_tile()` returns both, with different confidence values.

Expected:
- raw_count == 2;
- duplicate merge executes;
- merged count == 1;
- retained proposal mask_area == 3000;
- winner confidence follows frozen priority if all earlier priority keys tie appropriately.

Keep fake RGB:

```text
shape = (5000,5000,3)
```

No real 5000×5000 RGB or bool array.

# 6. Guard full-frame logical operations

In addition to `np.zeros` guard, wrap detector-module NumPy logical operations.

Save originals before monkeypatch:

```python
real_and = np.logical_and
real_or = np.logical_or
```

Wrappers must raise if any operand has shape `(5000,5000)`.

Then monkeypatch:

```python
detector.np.logical_and
detector.np.logical_or
```

with wrappers.

Small overlap-crop operands are allowed.

Expected:
- `detect_global(fake_rgb)` completes;
- no 5000×5000 bool zeros allocation;
- no 5000×5000 operand reaches logical_and/or;
- pairwise duplicate merge occurs.

# 7. Dedicated test gate — exactly once

After edits, run exactly once:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1
ENV_PYTHON -m pytest tests/test_task8b_runtime.py -q
```

Record exact result and exit code.

No rerun.

If FAIL:
- runtime unchanged;
- manifest unchanged;
- PARTIAL/STOP;
- commit/push and stop.

# 8. Manifest only if PASS

If dedicated PASS:
- update manifest hashes/sizes for all changed manifest-listed files since last valid manifest;
- expected changed entries:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

Verify:
- 135 entries;
- identical path set;
- all 135 files exist;
- all hash/size values match current canonical files.

Record:

```text
135/135 PASS
```

# 9. Report / FROM_DSH

Append:

```text
## Task 8B.3-M1A.2B-R2 — Final Dedicated-Test Corrections
```

Record:
- starting HEAD;
- merge test corrected to two independent 40×40 blocks;
- guard now uses two overlapping 50×60 detections;
- expected mask area = 3000;
- logical_and/or full-frame operand guards;
- dedicated invocation count = 1;
- exact test result;
- manifest result;
- runtime unchanged;
- delivery unchanged;
- real inference/full suite not run.

FROM_DSH must be UTF-8 without BOM and preserve ARTIFACT-FACTS exactly.

# 10. Commit / push

If COMPLETE:

```text
test(rc1): validate compact proposal runtime
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record final compact proposal test stop
```

Push current branch, no force.

# 11. COMPLETE

COMPLETE only if:
- runtime files untouched;
- merge-equivalence test fixed;
- 5000 guard uses 2 overlapping detections;
- pairwise proposal_iou path exercised;
- full-frame zeros and logical operands guarded;
- dedicated test invoked exactly once and PASS;
- manifest 135/135 PASS;
- delivery untouched;
- real inference not run;
- full canonical suite not run;
- commit/push succeed;
- clean tree;
- stop.

# 12. Final response

```text
TASK 8B.3-M1A.2B-R2 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Runtime files modified:
NO

Merge-equivalence correction:
PASS / FAIL

5000x5000 detect_global guard:
PASS / FAIL

Pairwise proposal_iou exercised:
YES / NO

Dedicated invocation count:
1

Dedicated tests:
<exact result>

Manifest:
135/135 PASS / other

External delivery:
UNCHANGED

Real inference:
NOT RUN

Full canonical suite:
NOT RUN BY DESIGN

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得同步 delivery、不得运行真实 Demo、不得进入 Task 8B.4 或 Task 8C。
```
