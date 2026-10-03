# TO_DSH — Task 8B.3-M1A.1: Implement Compact Proposal Masks + Dedicated Tests

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `8fc71ca49797112cd0a551faadf27ad7e1de29f1`
> Canonical RC1: `delivery_src/BuildReasonSeg_Advisor_RC1`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Runtime Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe`

# 0. Purpose

Implement the already-frozen `RC1-DEMO-MEM-01` compact-proposal refactor in the workspace canonical source and run only the dedicated runtime test file.

This task intentionally does **not** run the full canonical test suite and does **not** update `source_manifest.json`.
Those are deferred to Task 8B.3-M1A.2 after ChatGPT audits this implementation.

# 1. Permanent reporting rule

For COMPLETE / PARTIAL / STOP / FAILED, if Git is safe:

1. update `docs/task8b3_m1a_compact_proposal_masks.md`;
2. update `handoff/FROM_DSH.md`;
3. commit;
4. push `fix/task8b3-mem01-compact-proposals`;
5. stop.

# 2. Strict prohibitions

Do NOT:

- edit external delivery;
- sync canonical → delivery;
- run `predict.py`;
- run the six-image suite;
- run real YOLO/Qwen/SAM2/D-B1 inference;
- modify detector checkpoint/model parameters;
- modify detector tiling/overlap/imgsz/conf/max_det;
- modify `DUPLICATE_IOU`;
- modify duplicate winner priority;
- modify stable proposal-ID ordering;
- modify Reference eligibility/selection;
- modify ProgramHead/language;
- modify reasoning-context policy;
- modify SAM2/D-B1/relation fields;
- modify SUCCESS validity;
- modify PROP-01 / REF-01 / MASK-01;
- update `source_manifest.json`;
- run full `pytest tests -q`;
- enter Task 8B.4 or Task 8C;
- install packages;
- train/download;
- access final test.

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
branch = fix/task8b3-mem01-compact-proposals
HEAD = 8fc71ca49797112cd0a551faadf27ad7e1de29f1
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else -> STOP.

Do not branch/switch/rebase/reset/stash/clean.

# 4. Dependency inventory is already accepted

Do NOT repeat the broad dependency audit.

The previous turn already established executable `global_mask` use is confined to:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

This task may modify only those four functional/test files.

# 5. Allowed repository paths

Functional/test:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/outputs.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
```

Documentation/handoff:

```text
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

# 6. `GlobalProposal` frozen representation

In `detector.py`:

Replace:

```python
global_mask: np.ndarray
```

with:

```python
mask_crop: np.ndarray
```

Keep:

```python
global_bbox: tuple[int, int, int, int]
```

Invariant:

```text
bbox order = (top, left, bottom, right), inclusive
mask_crop.dtype = bool
mask_crop.shape = (bottom-top+1, right-left+1)
mask_crop is tight around foreground
```

Keep all existing metadata fields and external `to_dict()` keys/semantics unchanged.

`mask_crop` must NOT appear in `to_dict()`.

No lazy or compatibility `global_mask` property may materialize a full H×W array.

# 7. Compact raw detections

Add exactly one helper:

```python
def _compact_mask(mask: np.ndarray, *, top: int, left: int
                  ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
```

Required semantics:

1. `bbox_of(mask)`;
2. empty -> `None`;
3. tight local crop copied as bool;
4. convert tight local inclusive bbox to global inclusive bbox by adding `top`/`left`;
5. return `(mask_crop, global_bbox)`.

In `DetectorRuntime.detect_global()`:

After existing padding crop and valid source-range calculation, compact only the valid mask region.

The retained `accumulated` entry must contain:

```text
mask_crop
global_bbox
image_size = (height_px, width_px)
source_tile_id
tile_index
confidence
raw_index
tile_top
tile_left
```

Normal product path must not execute:

```python
np.zeros((height_px, width_px), dtype=bool)
```

for proposals.

Do not retain a 512×512 detection mask after the corresponding compact entry is formed.

# 8. Compact proposal geometry

In `merge_proposals()` create `GlobalProposal` directly from:

```text
mask_crop
global_bbox
image_size
```

Compute:

```text
mask_area = mask_crop.sum()
centroid = foreground centroid within crop + global_bbox top/left
touches_image_border = bbox touches image boundary
border_clearance = min(top, left, height-1-bottom, width-1-right)
```

For a tight crop these must be equivalent to the legacy full-frame mask calculations.

Do not change `bbox_extent_ratio`; it continues using the global bbox and frozen `TILE_SIZE=512`.

# 9. Compact IoU

Add:

```python
def proposal_iou(first: GlobalProposal, second: GlobalProposal) -> float:
```

Exact semantics:

1. find inclusive global bbox intersection;
2. no bbox intersection -> `0.0`;
3. convert global intersection to local slices for each `mask_crop`;
4. intersection count = logical AND of those two local slices;
5. union = `first.mask_area + second.mask_area - intersection`;
6. union <= 0 -> 0.0;
7. return intersection / union.

No full-frame `logical_or`.

Any temporary bool array may be no larger than the bbox-intersection slice.

`merge_proposals()` must call `proposal_iou()`.

Keep exactly:

```text
DUPLICATE_IOU = 0.50
winner priority = touches_border → -clearance → -area → -confidence → source_tile_id → raw_index
stable IDs = centroid_y → centroid_x → mask_area
no mask union
```

# 10. Adapt core Reference extraction

In `core.py`, modify only `reference_mask_from_proposal()`.

It must:

1. allocate only the existing 512×512 context bool target;
2. intersect proposal `global_bbox` and reasoning context in global coordinates;
3. slice `proposal.mask_crop` by that intersection;
4. paste into the correct context coordinates;
5. return the 512×512 bool mask.

Do not materialize a full-image proposal mask.

# 11. Adapt diagnostics preview

In `outputs.py`, modify only proposal outline handling in `proposals_preview_image()`.

Keep bbox drawing, colors, centroid marker and resize semantics unchanged.

For outline:

1. run `_erode()` on `proposal.mask_crop`;
2. compute crop-local outline;
3. paste magenta outline into the proposal bbox region of `preview`.

Do not allocate/materialize full H×W proposal masks.

# 12. Dedicated tests only

Modify `tests/test_task8b_runtime.py`.

Update existing test helpers that construct `GlobalProposal`.

Add focused tests for all of these:

## A. compact geometry equivalence

Using small synthetic legacy full masks, convert to compact proposal and assert:

```text
mask_area
global_bbox
centroid
touches_image_border
border_clearance
```

match test-only legacy full-mask calculations.

## B. `proposal_iou()` equivalence

Cases:

```text
identical
partial overlap
disjoint bboxes
overlapping bboxes but zero foreground intersection
```

Compare to a test-only legacy full-frame IoU.

## C. merge equivalence

Synthetic set must include:
- duplicate IoU >0.50;
- non-duplicate IoU <0.50;
- border/non-border duplicate;
- confidence tie-break;
- distinct centroids.

Use a small test-only legacy merge reference.

Assert identical:
- duplicate groups;
- winner source/raw identity;
- selected count;
- area;
- bbox;
- centroid;
- confidence;
- stable IDs.

## D. reference-context equivalence

Compare compact implementation to test-only legacy full-mask slicing for:
- fully inside;
- clipped left/top;
- clipped right/bottom.

## E. preview smoke

Compact proposals only:
- function succeeds;
- output shape equals RGB shape;
- no `global_mask` field needed.

## F. 5000×5000 synthetic guard

No real large RGB allocation and no model inference.

Mechanically exercise `DetectorRuntime.detect_global()` by monkeypatching:
- `plan_tiles()` -> tiny fixed window list;
- `extract_tile()` -> fixed small 512×512 RGB tile;
- `detect_tile()` -> fixed small bool proposal masks.

Use a lightweight fake RGB object exposing:

```text
shape = (5000, 5000, 3)
```

Monkeypatch detector-module `np.zeros` to raise if requested:

```text
shape == (5000, 5000)
dtype == bool
```

The compact detector/merge path must finish without triggering that guard.

Also guard `np.logical_and`/equivalent so the test fails if either operand/result uses a `(5000,5000)` proposal-mask shape.

Do NOT allocate a real 5000×5000 RGB array.

## G. serialization

`GlobalProposal.to_dict()` must preserve the same public keys/meanings as before and must not include `mask_crop`.

# 13. One dedicated test gate

Run only:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1
ENV_PYTHON -m pytest tests/test_task8b_runtime.py -q
```

If PASS:
- do not run additional tests;
- proceed to report/commit.

If FAIL:
- do not perform a second code-fix attempt in this task;
- record failure;
- commit/push PARTIAL;
- stop.

# 14. Manifest status

Do NOT edit `source_manifest.json`.

Because functional canonical files change, report:

```text
source_manifest status = INTENTIONALLY STALE ON FIX BRANCH
```

This is allowed only because:
- external delivery is not synced;
- M1A.2 will update/verify the manifest after ChatGPT audits implementation.

Do not claim 135/135 after code changes.

# 15. External delivery

Must remain byte-untouched.

Do not call sync helper in write mode.

No package/environment change.

# 16. Report

Update/create:

```text
docs/task8b3_m1a_compact_proposal_masks.md
```

Preserve the previous PARTIAL history, then add:

```text
## Task 8B.3-M1A.1 — Implementation + Dedicated Tests
```

Record:

- starting HEAD;
- four functional/test files changed;
- exact compact representation;
- IoU semantics;
- merge-equivalence tests;
- core/preview adaptations;
- dedicated test result;
- 5000×5000 synthetic guard result;
- full canonical tests = NOT RUN BY DESIGN;
- source manifest = INTENTIONALLY STALE ON FIX BRANCH;
- external delivery = UNCHANGED;
- real inference = NOT RUN;
- PROP-01/REF-01/MASK-01 = UNCHANGED;
- next gate = Awaiting ChatGPT audit before M1A.2.

# 17. FROM_DSH

Update `handoff/FROM_DSH.md`, preserving `ARTIFACT-FACTS` exactly.

Required:

```text
Task: 8B.3-M1A.1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: 8fc71ca49797112cd0a551faadf27ad7e1de29f1
Compact proposal representation: PASS / FAIL
Full-frame proposal masks retained: NO / YES / NOT VERIFIED
Full-frame pairwise IoU temporaries: NO / YES / NOT VERIFIED
Dedicated tests: <exact result>
5000x5000 synthetic guard: PASS / FAIL / NOT RUN
Canonical full tests: NOT RUN BY DESIGN
Source manifest: INTENTIONALLY STALE ON FIX BRANCH
External delivery modified: NO
Real inference executed: NO
Scientific model/checkpoint changed: NO
PROP-01: UNCHANGED
REF-01: UNCHANGED
MASK-01: UNCHANGED
Report: docs/task8b3_m1a_compact_proposal_masks.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit before M1A.2.
```

# 18. Git gate

Allowed changed paths exactly:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/outputs.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

`source_manifest.json` must NOT change.

Run:

```bat
git status --short
git diff --check
git diff
```

Stage individually only.

# 19. Commit / push

If COMPLETE, exact commit:

```text
fix(rc1): implement compact proposal masks
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record compact proposal implementation stop
```

Push:

```bat
git push origin fix/task8b3-mem01-compact-proposals
```

No force push.

# 20. COMPLETE definition

COMPLETE only if:

- exact starting state;
- only allowed files changed;
- no external delivery modification;
- no real inference;
- `GlobalProposal` uses `mask_crop`, not `global_mask`;
- detect path retains only tight crops;
- merge uses bbox-intersection IoU;
- no full-frame proposal mask allocation in normal detector/merge path;
- Reference context adapted without full mask;
- preview adapted without full mask;
- dedicated runtime tests all pass;
- geometry/IoU/merge/context equivalence covered;
- 5000×5000 synthetic guard passes;
- source manifest intentionally left stale and explicitly reported;
- no full canonical suite run;
- report/FROM_DSH complete;
- commit/push succeed;
- working tree clean;
- DSH stops.

# 21. Final response

```text
TASK 8B.3-M1A.1 COMPLETE / PARTIAL / STOP / FAILED

Branch:
fix/task8b3-mem01-compact-proposals

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

Compact proposal representation:
PASS / FAIL

Full-frame proposal masks retained:
NO / YES / NOT VERIFIED

Full-frame pairwise IoU temporaries:
NO / YES / NOT VERIFIED

Dedicated tests:
<exact result>

5000x5000 synthetic guard:
PASS / FAIL / NOT RUN

Canonical full tests:
NOT RUN BY DESIGN

Source manifest:
INTENTIONALLY STALE ON FIX BRANCH

External delivery modified:
NO

Real inference executed:
NO

Scientific model/checkpoint changed:
NO

PROP-01:
UNCHANGED

REF-01:
UNCHANGED

MASK-01:
UNCHANGED

Report:
docs/task8b3_m1a_compact_proposal_masks.md

Handoff:
handoff/FROM_DSH.md

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得运行全 canonical suite、不得更新 manifest、不得同步 delivery、不得运行真实 Demo、不得进入 Task 8B.4 或 Task 8C。
```
