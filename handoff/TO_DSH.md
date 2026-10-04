# TO_DSH — Task 8B.3-M1A.2B-R1: Correct Dedicated Tests and Re-run Once

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `e5229d60a869c1fc02536d41a039229f2936090b`

# 0. Purpose

Correct test-adaptation defects found in Task 8B.3-M1A.2B.

Frozen audit conclusion:
- runtime files are NOT authorized to change;
- prior dedicated run failed mainly because old tests still supplied legacy `{"mask": ...}` entries to the new compact `merge_proposals()`;
- one new merge-equivalence test had an incorrect area assumption;
- the 5000×5000 guard did not actually exercise `DetectorRuntime.detect_global()` and allocated a real 5000×5000 bool oracle.

This task edits tests only, re-runs the dedicated file exactly once, then updates manifest only if PASS.

# 1. Strict prohibitions

Do NOT modify:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
```

Do NOT modify external delivery or any model/config/threshold/policy.

Do NOT run real inference, full canonical suite, Task 8B.4, or Task 8C.

If the corrected dedicated run still fails:
- do NOT patch runtime;
- do NOT rerun;
- record exact failures;
- commit/push PARTIAL;
- STOP.

# 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = e5229d60a869c1fc02536d41a039229f2936090b
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

# 4. Add one compact-entry test helper

In `test_task8b_runtime.py`, add a test-only helper equivalent to:

```python
def _entry(mask, confidence, tile, raw_index, tile_index):
    compact = detector._compact_mask(mask, top=0, left=0)
    assert compact is not None
    crop, bbox = compact
    return {
        "mask_crop": crop,
        "global_bbox": bbox,
        "image_size": mask.shape,
        "confidence": confidence,
        "source_tile_id": tile,
        "tile_index": tile_index,
        "raw_index": raw_index,
    }
```

Use this helper wherever `merge_proposals()` is called in existing tests.

# 5. Repair all remaining legacy merge calls

At minimum update these existing tests so they no longer pass `{"mask": ...}`:

```text
test_merge_duplicate_iou_threshold_and_no_union
test_merge_priority_prefers_non_border_then_area
test_stable_ids_by_centroid_then_area
test_select_reference_tie_break
```

Search the entire test file for every call to:

```python
merge_proposals(...)
```

Every entry passed to product `merge_proposals()` must use the compact entry contract.

Legacy full-frame masks may remain only as test oracles.

# 6. Correct duplicate/confidence test geometry

Where the test claims equal-area duplicate masks and expects confidence tie-break, construct genuinely equal-area masks.

Preferred:

```python
first = zeros
first[100:140,100:140] = True
second = zeros
second[104:144,104:144] = True
```

Both area = 1600; IoU > 0.50.

Do not use `second = first.copy(); second[104:144,...] = True` while asserting area 1600.

If a test specifically isolates confidence tie-break, use geometry with equal border-clearance/area or exact-identical masks so confidence is the first differing priority key.

# 7. Strengthen required test coverage

Keep/repair the required groups:

1. compact geometry equivalence;
2. proposal_iou equivalence;
3. merge equivalence;
4. reference-context equivalence;
5. preview smoke;
6. serialization contract;
7. 5000×5000 detect_global allocation guard.

Do not weaken assertions merely to pass.

# 8. Replace the 5000×5000 guard with the frozen contract

The guard must mechanically call:

```python
DetectorRuntime.detect_global(fake_rgb)
```

Do NOT allocate:
- a real 5000×5000 RGB image;
- a real 5000×5000 bool proposal oracle.

Use a minimal fake object whose only required property is:

```python
shape = (5000, 5000, 3)
```

Monkeypatch:

```text
plan_tiles() -> tiny fixed window list
extract_tile() -> fixed 512×512 RGB tile + no-padding dict
DetectorRuntime.detect_tile() -> several small 512×512 bool detections
```

Include at least two overlapping detections so pairwise duplicate merge executes.

Monkeypatch detector-module `np.zeros`:
- if shape `(5000,5000)` and dtype bool -> raise sentinel;
- otherwise delegate to original.

Wrap detector-module `np.logical_and` and `np.logical_or`:
- raise if either operand has shape `(5000,5000)`.

Expected:
- `detect_global(fake_rgb)` completes;
- sentinel never fires;
- merged proposals exist;
- every merged proposal has `mask_crop`;
- none has `global_mask`;
- all proposal crops are far smaller than full-frame dimensions.

This test must exercise the product `detect_global → merge_proposals → proposal_iou` path.

# 9. Dedicated test gate — exactly once in R1

After all test edits, run exactly once:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1
ENV_PYTHON -m pytest tests/test_task8b_runtime.py -q
```

Record exact result and exit code.

No rerun.

If FAIL -> PARTIAL/STOP, no manifest update, no runtime edit.

# 10. Manifest — only if dedicated PASS

If PASS:
- update hashes/sizes for all manifest-listed files changed since last valid manifest;
- this includes prior runtime changes plus current test change.

Expected affected manifest-listed paths:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

Manifest requirements:

```text
entry count = 135
path set unchanged
all 135 files exist
all 135 hashes/sizes match current canonical content
```

If any unexpected manifest-listed changed path appears -> STOP.

# 11. Report

Append:

```text
## Task 8B.3-M1A.2B-R1 — Dedicated Test Correction
```

Record:
- starting HEAD;
- why prior 5 failures were test-adaptation defects;
- all repaired legacy merge calls;
- corrected equal-area geometry;
- real detect_global guard design;
- dedicated invocation count = 1;
- exact result;
- manifest status;
- runtime files unchanged;
- external delivery unchanged;
- real inference not run;
- full canonical suite not run.

# 12. FROM_DSH

Preserve ARTIFACT-FACTS exactly; UTF-8 without BOM.

Required fields:

```text
Task: 8B.3-M1A.2B-R1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: e5229d60a869c1fc02536d41a039229f2936090b
Runtime files modified: NO
Legacy merge-call adaptation: PASS / FAIL
Equal-area duplicate test: PASS / FAIL
5000x5000 detect_global guard: PASS / FAIL
Dedicated invocation count: 1
Dedicated tests: <exact result>
Manifest: 135/135 PASS / UNCHANGED DUE FAIL / other
External delivery modified: NO
Real inference: NO
Full canonical suite: NOT RUN BY DESIGN
Next action: Awaiting ChatGPT audit.
```

# 13. Git / commit / push

Allowed paths only §3.

If COMPLETE:

```text
test(rc1): validate compact proposal runtime
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record compact proposal retest stop
```

Push current branch, no force.

# 14. COMPLETE

COMPLETE only if:
- runtime files untouched;
- all legacy merge calls adapted;
- duplicate-area logic corrected;
- guard truly calls detect_global on fake 5000×5000 shape;
- dedicated test run exactly once;
- dedicated tests PASS;
- manifest 135/135 PASS;
- external delivery untouched;
- real inference not run;
- full canonical suite not run;
- commit/push succeed;
- clean tree;
- stop.

# 15. Final response

```text
TASK 8B.3-M1A.2B-R1 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Runtime files modified:
NO

Legacy merge-call adaptation:
PASS / FAIL

5000x5000 detect_global guard:
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
