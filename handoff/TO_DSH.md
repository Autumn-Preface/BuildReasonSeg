# TO_DSH — Task 8B.3-M1A.2B-R3: Fix Final Large-Image Guard Test

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `872d45b3ad6adee3b3d9f0fa5bdfa95fd058a37c`

# 0. Purpose

Fix the last remaining dedicated-test defect only.

Current dedicated result:

```text
1 failed, 31 passed
```

The only failing test is:

```text
test_large_image_path_never_allocates_full_frame_bool
```

Frozen audit conclusion:
- runtime is not authorized to change;
- the failure is still test-side;
- merge-equivalence test is now accepted;
- only the 5000×5000 guard must be corrected.

# 1. Strict prohibitions

Do NOT modify:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
```

Do NOT modify external delivery, model/config/threshold/tiling/merge/reference policy.

Do NOT run real inference or full canonical suite.

If the single dedicated rerun fails:
- no runtime patch;
- no second rerun;
- no manifest update;
- record exact failure, commit/push PARTIAL, STOP.

# 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = 872d45b3ad6adee3b3d9f0fa5bdfa95fd058a37c
```

Allowed initial tree:
- clean; or
- only `M handoff/TO_DSH.md`.

# 3. Allowed changed paths

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

# 4. Keep the accepted merge-equivalence test unchanged

Do not edit `test_merge_equivalence_with_compact_masks()` except formatting-only changes if unavoidable.

It is now accepted:
- two independent equal-area 40×40 masks;
- one disjoint equal-area mask;
- duplicate winner confidence assertion valid.

# 5. Replace the 5000×5000 guard with one clean single-pass scenario

In:

```python
test_large_image_path_never_allocates_full_frame_bool
```

use exactly one fake tile window and exactly two synthetic detections returned by `detect_tile()`.

Preferred masks:

```python
mask1 = np.zeros((512,512), dtype=bool)
mask1[100:150, 200:260] = True      # 50×60 = 3000

mask2 = np.zeros((512,512), dtype=bool)
mask2[104:154, 204:264] = True      # 50×60 = 3000, IoU > 0.50
```

Use:

```text
plan_tiles() -> [one FakeWindow]
extract_tile() -> one 512×512 tile
detect_tile() -> [detection1, detection2]
```

Expected:
- raw_count == 2;
- proposal_iou is called exactly once;
- merged count == 1;
- retained proposal mask_area == 3000;
- retained proposal mask_crop.shape == (50,60).

Do NOT use two windows in this guard.

# 6. Guard only what can be safely monkeypatched

Keep the full-frame allocation guard:

```python
real_zeros = detector.np.zeros

def zero_guard(shape, *args, **kwargs):
    dtype = kwargs.get("dtype")
    if tuple(shape) == (5000,5000) and dtype is bool:
        raise AssertionError(...)
    return real_zeros(shape, *args, **kwargs)

monkeypatch.setattr(detector.np, "zeros", zero_guard)
```

Do NOT monkeypatch:

```text
detector.np.logical_and
detector.np.logical_or
```

Reason:
`detector.np` and the test module's `np` refer to the same imported NumPy module object. Replacing the ufunc with a Python wrapper removes ufunc methods such as `.reduce` and can break NumPy internals, which caused the observed `fromnumeric.py:99 AttributeError`.

# 7. Verify pairwise compact operands safely

Instead of monkeypatching NumPy logical ufuncs, wrap only the product function:

```python
real_iou = detector.proposal_iou
pairwise_shapes = []

def guarded_iou(first, second):
    pairwise_shapes.append((first.mask_crop.shape, second.mask_crop.shape))
    assert first.mask_crop.shape != (5000,5000)
    assert second.mask_crop.shape != (5000,5000)
    return real_iou(first, second)

monkeypatch.setattr(detector, "proposal_iou", guarded_iou)
```

After `detect_global(fake_rgb)` assert:

```text
len(pairwise_shapes) == 1
both recorded shapes are small compact crops
```

This proves the actual duplicate-merge call receives compact operands without corrupting NumPy itself.

# 8. No real 5000×5000 arrays

The fake RGB object may expose only:

```python
shape = (5000,5000,3)
```

Do not allocate:
- real 5000×5000 RGB;
- real 5000×5000 bool mask;
- legacy full-frame 5000×5000 oracle.

# 9. Dedicated test gate — exactly once

After editing the test, run exactly once:

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

# 10. Manifest only if PASS

If dedicated PASS:
- update manifest hashes/sizes for all changed manifest-listed files since the last valid manifest.

Expected changed entries:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

Verify:

```text
entry count = 135
path set unchanged
all 135 files exist
all 135 hash/size values match current canonical files
```

Record:

```text
135/135 PASS
```

# 11. Report / handoff

Append:

```text
## Task 8B.3-M1A.2B-R3 — Final Large-Image Guard Correction
```

Record:
- starting HEAD;
- old NumPy monkeypatch problem;
- one-window/two-detection guard;
- both masks 50×60, area 3000;
- proposal_iou wrapper call count;
- dedicated invocation count = 1;
- exact test result;
- manifest result;
- runtime unchanged;
- external delivery unchanged;
- real inference/full suite not run.

`handoff/FROM_DSH.md`:
- UTF-8 without BOM;
- ARTIFACT-FACTS preserved exactly.

Required fields:

```text
Task: 8B.3-M1A.2B-R3
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: 872d45b3ad6adee3b3d9f0fa5bdfa95fd058a37c
Runtime files modified: NO
5000x5000 zero-allocation guard: PASS / FAIL
Pairwise proposal_iou call count: 1 / other
Pairwise operands compact: PASS / FAIL
Dedicated invocation count: 1
Dedicated tests: <exact result>
Manifest: 135/135 PASS / UNCHANGED DUE FAIL / other
External delivery modified: NO
Real inference: NO
Full canonical suite: NOT RUN BY DESIGN
Next action: Awaiting ChatGPT audit.
```

# 12. Commit / push

If COMPLETE:

```text
test(rc1): validate compact proposal runtime
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record final large-image guard stop
```

Push current branch, no force.

# 13. COMPLETE

COMPLETE only if:
- runtime files untouched;
- guard uses one window + two overlapping 50×60 detections;
- no real 5000×5000 arrays allocated;
- no NumPy logical ufunc monkeypatch remains;
- proposal_iou called exactly once;
- operands confirmed compact;
- dedicated test file invoked exactly once and PASS;
- manifest 135/135 PASS;
- delivery untouched;
- real inference not run;
- full canonical suite not run;
- commit/push succeed;
- clean tree;
- stop.

# 14. Final response

```text
TASK 8B.3-M1A.2B-R3 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Runtime files modified:
NO

5000x5000 zero-allocation guard:
PASS / FAIL

Pairwise proposal_iou call count:
1 / other

Pairwise operands compact:
PASS / FAIL

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
