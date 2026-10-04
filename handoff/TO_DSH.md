# TO_DSH — Task 8B.3-M1A.2B: Dedicated Compact-Mask Tests + Manifest

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `39b36031294f1ec1ddab3f5e9681170af687af5a`
> Canonical RC1: `delivery_src/BuildReasonSeg_Advisor_RC1`

## 0. Purpose

Validate the already-implemented compact-proposal runtime without changing runtime code.

This task does only:

```text
adapt/add dedicated tests
→ run tests/test_task8b_runtime.py exactly once
→ update source_manifest.json
→ verify manifest 135/135
→ commit/push
```

No runtime implementation edit, no external-delivery sync, no real inference.

## 1. Strict prohibitions

Do NOT modify:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
```

Do NOT modify detector/model parameters, tiling/overlap/imgsz/conf/max_det, `DUPLICATE_IOU`, merge winner policy, stable-ID policy, Reference eligibility/selection, reasoning context, ProgramHead/SAM2/D-B1/SUCCESS validity, PROP-01, REF-01, MASK-01, or external delivery.

Do NOT run `predict.py`, the six-image Demo, real YOLO/Qwen/SAM2/D-B1, or the full canonical suite `pytest tests -q`.

If dedicated tests expose a runtime bug, do NOT patch runtime in this task: record exact failure, commit/push PARTIAL, STOP.

## 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = 39b36031294f1ec1ddab3f5e9681170af687af5a
```

Allowed initial tree: clean, or only `M handoff/TO_DSH.md`.

## 3. Allowed changed paths

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

## 4. Adapt test helpers

Update synthetic proposal helpers so tests construct `GlobalProposal(mask_crop=..., global_bbox=..., image_size=...)` instead of `global_mask`.

A test-only conversion helper may reconstruct compact representation from a synthetic full mask. Product code must not change.

## 5. Required test groups

### 5.1 Geometry equivalence

Use interior, top-border, right-border, and irregular masks. Compare compact proposal against test-only legacy full-mask calculations for bbox, tight crop shape, area, centroid, border touch, border clearance, bbox extent ratio.

### 5.2 `proposal_iou()` equivalence

Compare new `proposal_iou()` to legacy `iou_of()` for:
1. identical masks;
2. partial overlap;
3. disjoint bboxes;
4. overlapping bboxes but disjoint foreground;
5. one bbox contained inside the other.

Require `pytest.approx` equality.

### 5.3 Merge equivalence

Build deterministic synthetic proposals containing duplicate IoU >= 0.50, nonduplicate IoU < 0.50, border vs non-border duplicate, confidence tie-break, and distinct centroids.

Implement a test-only legacy full-frame merge oracle. Assert identical duplicate grouping, winner source/raw identity, proposal count, area, bbox, centroid, confidence, and stable proposal IDs.

### 5.4 Reference-context equivalence

Compare compact `reference_mask_from_proposal()` against test-only legacy full-image behavior for:
- fully inside context;
- context begins above/left of proposal;
- clipped top/left;
- clipped bottom/right;
- no intersection.

Require bit-identical 512×512 results.

### 5.5 Preview smoke

Construct compact proposals and call `proposals_preview_image()`. Assert output size matches input, bbox/centroid/outline execution succeeds, and no `global_mask` attribute is required.

### 5.6 Serialization contract

`GlobalProposal.to_dict()` must expose exactly the pre-existing public keys:

```text
proposal_id
source_tile_id
confidence
mask_area
global_bbox
centroid
touches_image_border
border_clearance
bbox_extent_ratio
raw_index
```

Assert no `mask_crop`, `image_size`, or `global_mask` key.

### 5.7 5000×5000 detector allocation guard

No real image/model.

Mechanically exercise `DetectorRuntime.detect_global()` with:
- fake RGB object exposing `shape=(5000,5000,3)`;
- monkeypatched `plan_tiles()` returning a tiny fixed list;
- monkeypatched `extract_tile()` returning a fixed 512×512 RGB tile with no padding;
- monkeypatched `DetectorRuntime.detect_tile()` returning several small synthetic bool masks, including overlapping masks that force duplicate merge.

Monkeypatch detector-module `np.zeros` so a request for `(5000,5000)` bool raises a sentinel; otherwise delegate to real zeros.

Wrap `np.logical_and`/`np.logical_or` so the test fails if an operand has shape `(5000,5000)`.

Expected:
- `detect_global()` completes;
- sentinel never fires;
- merged proposals exist;
- every proposal has `mask_crop`;
- no proposal has `global_mask`;
- all crop extents are small relative to 5000×5000.

Do not allocate a real 5000×5000 RGB array.

## 6. Legacy helper policy

Product `iou_of(first, second)` may remain for test/reference compatibility. Normal `detect_global → merge_proposals` must use `proposal_iou()`. Tests may call `iou_of()` only as an oracle.

## 7. Dedicated test gate — exactly once

Run exactly once:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1
ENV_PYTHON -m pytest tests/test_task8b_runtime.py -q
```

Record exact pass/fail count and exit code.

If FAIL:
- no runtime patch;
- no rerun;
- mark PARTIAL;
- record failure;
- commit/push and STOP.

If PASS, continue.

## 8. Manifest update

Only after dedicated tests PASS.

Starting manifest must have exactly 135 entries and unchanged path set.

Update hash/size for all manifest-listed canonical files changed since the last valid manifest. At minimum:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

If verification finds another changed manifest-listed path, STOP rather than broadening scope.

Verify:

```text
entry count = 135
path set unchanged
all 135 files exist
all 135 size/hash values match current canonical files
```

Record `135/135 PASS`.

## 9. No full canonical suite

Do NOT run `pytest tests -q`. That is reserved for M1A.2C after ChatGPT audit.

## 10. Report

Append `## Task 8B.3-M1A.2B — Dedicated Tests + Manifest` to `docs/task8b3_m1a_compact_proposal_masks.md`.

Record starting HEAD, test groups, dedicated test exact result, invocation count=1, 5000×5000 guard result, manifest result, runtime files unchanged, external delivery unchanged, real inference not run, full canonical suite NOT RUN BY DESIGN, PROP/REF/MASK unchanged.

## 11. FROM_DSH

Preserve `ARTIFACT-FACTS` exactly and save UTF-8 without BOM.

Required fields:

```text
Task: 8B.3-M1A.2B
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: 39b36031294f1ec1ddab3f5e9681170af687af5a
Runtime files modified in this task: NO
Dedicated test invocation count: 1
Dedicated tests: <exact result>
Geometry equivalence: PASS / FAIL
IoU equivalence: PASS / FAIL
Merge equivalence: PASS / FAIL
Reference-context equivalence: PASS / FAIL
Preview smoke: PASS / FAIL
Serialization contract: PASS / FAIL
5000x5000 guard: PASS / FAIL
Manifest: 135/135 PASS / ...
External delivery modified: NO
Real inference executed: NO
Full canonical suite: NOT RUN BY DESIGN
PROP-01 / REF-01 / MASK-01: UNCHANGED / UNCHANGED / UNCHANGED
Next action: Awaiting ChatGPT audit.
```

## 12. Git / commit / push

Allowed changed paths only those in §3.

Run `git status --short`, `git diff --check`, `git diff`. Stage individually.

If COMPLETE:

```text
test(rc1): validate compact proposal runtime
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record compact proposal test stop
```

Push current branch, no force.

## 13. COMPLETE definition

COMPLETE only if runtime files were not modified in this task; all required test groups exist; dedicated test file was invoked exactly once and passes; 5000×5000 guard passes; manifest 135/135 passes; external delivery remains untouched; no real inference/full canonical suite; report/FROM_DSH complete; commit/push succeed; clean tree; stop.

## 14. Final response

```text
TASK 8B.3-M1A.2B COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Runtime files modified:
NO

Dedicated test invocation count:
1

Dedicated tests:
<exact result>

5000x5000 guard:
PASS / FAIL

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
