# TO_DSH — Task 8B.3-M1A.2A: Compact Proposal Runtime Implementation Only

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `2c51c48252fcc705da5d9ee914299775d985a725`
> Canonical RC1: `delivery_src/BuildReasonSeg_Advisor_RC1`

# 0. Scope

Do exactly one thing:

```text
replace full-frame proposal masks with tight bbox-local mask_crop
in detector.py + adapt core.py + adapt outputs.py
```

No test-file edits, no manifest edit, no delivery sync, no real inference.

# 1. Prohibitions

Do NOT modify:
- external delivery;
- tests;
- source_manifest.json;
- model/checkpoint/config/threshold/tiling;
- duplicate IoU threshold;
- merge winner/stable-ID policy;
- Reference eligibility/selection;
- reasoning context;
- ProgramHead/SAM2/D-B1/SUCCESS validity;
- PROP-01 / REF-01 / MASK-01.

Do NOT run:
- predict.py;
- six-image Demo;
- real YOLO/Qwen/SAM2/D-B1;
- pytest full suite.

Allowed functional paths only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/outputs.py
```

Docs/handoff:

```text
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = 2c51c48252fcc705da5d9ee914299775d985a725
```

Allowed initial tree: clean or only `M handoff/TO_DSH.md`.

# 3. detector.py — representation

Replace:

```python
global_mask: np.ndarray
```

with:

```python
mask_crop: np.ndarray
```

Keep `global_bbox` inclusive `(top,left,bottom,right)` and all other metadata.

Invariant:

```text
mask_crop.dtype == bool
mask_crop.shape == (bottom-top+1, right-left+1)
mask_crop is tight around foreground
```

`to_dict()` external keys stay unchanged and must not expose `mask_crop`.

# 4. detector.py — compaction

Add helper equivalent to:

```python
def _compact_mask(mask, *, top, left):
    box = bbox_of(mask)
    if box is None:
        return None
    t,l,b,r = box
    crop = np.asarray(mask[t:b+1, l:r+1], dtype=bool).copy()
    return crop, (top+t, left+l, top+b, left+r)
```

In `detect_global()`:
- preserve existing padding crop;
- preserve valid-image clipping;
- compact the valid local mask;
- accumulated entry stores `mask_crop`, `global_bbox`, `image_size`, existing IDs/confidence fields;
- remove per-proposal `np.zeros((height_px,width_px), dtype=bool)`.

# 5. detector.py — compact proposal geometry

Construct `GlobalProposal` from crop+bbox+image_size:

```text
area = crop.sum()
centroid = foreground local centroid + bbox offset
touch border = top==0 or left==0 or bottom==H-1 or right==W-1
clearance = min(top,left,H-1-bottom,W-1)
```

Keep `bbox_extent_ratio` semantics unchanged.

Eligibility non-empty condition becomes compact-equivalent (`mask_area > 0`); all other eligibility rules unchanged.

# 6. detector.py — compact IoU

Add:

```python
proposal_iou(first, second)
```

Semantics:
1. intersect inclusive bboxes;
2. no bbox overlap -> 0.0;
3. slice each `mask_crop` to overlap;
4. intersection = AND count;
5. union = first.area + second.area - intersection;
6. union<=0 -> 0.0;
7. return intersection/union.

Do not use full-frame `logical_or`.

Merge must call `proposal_iou()`.

Do not change:
- `DUPLICATE_IOU=0.50`;
- union-find;
- winner priority;
- stable-ID ordering;
- no-union policy.

# 7. core.py adaptation

`reference_mask_from_proposal()` must:
- allocate only existing 512×512 context mask;
- intersect proposal bbox with reasoning context;
- slice corresponding region from `proposal.mask_crop`;
- paste into context mask;
- never construct full-image mask.

No other core behavior changes.

# 8. outputs.py adaptation

`proposals_preview_image()`:
- bbox and centroid drawing unchanged;
- run `_erode()` on `proposal.mask_crop`;
- compute local outline;
- paste magenta outline into preview bbox region;
- never construct full-image mask.

# 9. Minimal implementation smoke only

Do NOT edit tests.

Run only:

```bat
ENV_PYTHON -m py_compile ^
  buildreasonseg/runtime/detector.py ^
  buildreasonseg/runtime/core.py ^
  buildreasonseg/runtime/outputs.py
```

Then run a short one-off Python smoke script that:
- creates two synthetic compact `GlobalProposal`s directly;
- calls `proposal_iou`;
- calls `reference_mask_from_proposal` with a synthetic context;
- calls `proposals_preview_image`;
- confirms no AttributeError / shape error;
- does NOT load any model.

Do not persist the smoke script in Git.

No pytest in this task.

# 10. Report / handoff

Append to:

```text
docs/task8b3_m1a_compact_proposal_masks.md
```

section:

```text
## Task 8B.3-M1A.2A — Runtime Implementation Only
```

Record:
- starting HEAD;
- changed runtime files;
- representation;
- compact IoU;
- core/preview adaptations;
- py_compile result;
- synthetic smoke result;
- tests = NOT RUN BY DESIGN;
- manifest = NOT UPDATED BY DESIGN;
- external delivery = UNCHANGED;
- real inference = NOT RUN.

Update `handoff/FROM_DSH.md`, preserving ARTIFACT-FACTS exactly.

Required status fields:

```text
Task: 8B.3-M1A.2A
Representation: tight bbox + mask_crop
Full-frame proposal mask allocation in detector path: REMOVED / ...
Full-frame pairwise IoU: REMOVED / ...
py_compile: ...
Synthetic smoke: ...
Tests: NOT RUN BY DESIGN
Manifest: NOT UPDATED BY DESIGN
External delivery modified: NO
Real inference: NO
Next action: Awaiting ChatGPT audit.
```

# 11. Commit/push

Allowed changed paths only the 3 runtime files + report/FROM_DSH/TO_DSH.

If COMPLETE:

```text
fix(rc1): implement compact proposal runtime
```

If STOP/PARTIAL:

```text
docs(rc1): record compact runtime implementation stop
```

Push current branch, no force.

# 12. COMPLETE

COMPLETE only if:
- all 3 runtime adaptations implemented together;
- no full-frame proposal mask allocation remains in normal detector/merge path;
- no full-frame pairwise IoU remains;
- py_compile passes;
- synthetic smoke passes;
- tests not edited/run;
- manifest not edited;
- delivery untouched;
- no real inference;
- report/handoff committed and pushed;
- clean tree;
- stop.

# 13. Final response

```text
TASK 8B.3-M1A.2A COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Runtime implementation:
PASS / FAIL

Full-frame proposal allocation:
REMOVED / STILL PRESENT / NOT VERIFIED

Full-frame pairwise IoU:
REMOVED / STILL PRESENT / NOT VERIFIED

py_compile:
PASS / FAIL

Synthetic smoke:
PASS / FAIL / NOT RUN

Tests:
NOT RUN BY DESIGN

Manifest:
NOT UPDATED BY DESIGN

External delivery:
UNCHANGED

Real inference:
NOT RUN

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得编辑 tests/manifest、不得同步 delivery、不得运行真实 Demo。
```
