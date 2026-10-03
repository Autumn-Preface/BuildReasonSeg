# TO_DSH — Task 8B.3-M1A: Compact Proposal Masks (Canonical Only)

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `7a9c576692abf82510d61fb1c814df47d4b053a5`
> New branch: `fix/task8b3-mem01-compact-proposals`
> Canonical RC1: `delivery_src/BuildReasonSeg_Advisor_RC1`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Runtime Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\python.exe`

# 0. Purpose

Fix the canonical implementation of `RC1-DEMO-MEM-01` without changing scientific/runtime semantics.

Frozen architectural decision:

```text
OLD:
one full H×W bool mask retained per raw proposal
+ pairwise H×W logical_and/logical_or during duplicate merge

NEW:
one tight bbox-local bool crop retained per raw proposal
+ duplicate IoU computed only in the two proposal bboxes' intersection
```

This task modifies and tests the **workspace canonical source only**.

It does NOT sync or edit external delivery and does NOT run real inference.

# 1. Permanent reporting rule

For COMPLETE / PARTIAL / STOP / FAILED, when Git is safe:

1. update the task report;
2. update `handoff/FROM_DSH.md`;
3. commit;
4. push the current fix branch;
5. stop and wait for ChatGPT.

# 2. Strict prohibitions

Do NOT:

- edit any file under external delivery;
- run `predict.py`;
- run the six-image suite;
- run real YOLO/Qwen/SAM2/D-B1 inference;
- modify detector checkpoint or model parameters;
- modify `TILE_SIZE`, overlap, stride, `IMGSZ`, `CONF`, `MAX_DET`;
- modify `DUPLICATE_IOU`;
- change duplicate winner priority;
- change stable-ID ordering;
- change Reference eligibility or `MERGE_BBOX_EXTENT_RATIO_MAX`;
- change Reference selection semantics;
- change reasoning context;
- change ProgramHead/language;
- change SAM2/D-B1/relation fields;
- change post-inference SUCCESS validity;
- fix PROP-01 / REF-01 / MASK-01;
- implement Task 8B.4;
- access final test;
- install packages;
- train/download.

Unexpected dependency/use of `global_mask` outside the explicitly allowed files below -> STOP and report.

# 3. Git safety and branch

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
HEAD = 7a9c576692abf82510d61fb1c814df47d4b053a5
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

If `fix/task8b3-mem01-compact-proposals` already exists -> STOP.

Create:

```bat
git switch -c fix/task8b3-mem01-compact-proposals
```

Do not merge/rebase/reset/stash/clean.

# 4. Pre-change dependency gate

Run read-only:

```bat
git grep -n "global_mask" -- delivery_src/BuildReasonSeg_Advisor_RC1
```

Expected product/test usages must be confined to:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

Comments/docs inside the canonical package are allowed only if they describe the same API.

If an executable use appears in any other canonical file -> STOP before code changes and report the path/line.

# 5. Allowed repository paths

Functional canonical edits are limited to:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/outputs.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Engineering documentation:

```text
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

# 6. Frozen `GlobalProposal` representation

In canonical `detector.py`, change `GlobalProposal` so proposal mask storage is:

```python
mask_crop: np.ndarray
global_bbox: tuple[int, int, int, int]
```

where:

- bbox order remains `(top, left, bottom, right)`;
- bounds remain inclusive;
- `mask_crop.shape == (bottom-top+1, right-left+1)`;
- crop is tight: first/last bbox coordinates come from true foreground extent;
- `mask_crop` is bool;
- no `global_mask` full-image field remains on `GlobalProposal`.

Keep existing metadata fields and meanings:

```text
proposal_id
source_tile_id
tile_index
confidence
mask_area
touches_image_border
border_clearance
centroid
raw_index
pad_mask_empty
```

`to_dict()` must preserve the existing external JSON contract:
- same keys;
- same coordinate/order semantics;
- do NOT serialize `mask_crop`.

# 7. Frozen raw-detection compaction

Inside `detect_global()`:

After any existing padding crop, convert each non-empty tile-local detection mask into a **tight crop** before retaining it.

Add exactly one helper with this semantic role:

```python
def _compact_mask(mask: np.ndarray, *, top: int, left: int
                  ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
```

Required semantics:

1. compute tight local bbox using existing `bbox_of(mask)`;
2. empty -> `None`;
3. copy only the tight bool crop;
4. convert local bbox to inclusive global bbox using `top`, `left`;
5. return `(mask_crop, global_bbox)`.

The `accumulated` entry must retain:

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

Do NOT allocate any `(height_px, width_px)` bool array per proposal.

Do NOT retain the original 512×512 mask after compaction beyond the current tile loop.

# 8. Frozen compact geometry

When constructing `GlobalProposal` in `merge_proposals()`:

Compute from `mask_crop + global_bbox + image_size`:

- `mask_area` = `mask_crop.sum()`;
- centroid = local foreground centroid offset by bbox top/left;
- `touches_image_border` from tight global bbox against image dimensions;
- `border_clearance` from tight global bbox against image dimensions.

Because the crop is tight, border-touch/clearance derived from bbox must be exactly equivalent to the prior full-mask result.

Do not change `bbox_extent_ratio`; it must still divide maximum bbox extent by frozen `TILE_SIZE=512`.

# 9. Frozen compact IoU

Replace full-frame duplicate IoU with:

```python
def proposal_iou(first: GlobalProposal, second: GlobalProposal) -> float:
```

Required exact semantics:

1. intersect the two inclusive global bboxes;
2. no bbox overlap -> `0.0` without mask allocation;
3. slice each proposal's `mask_crop` only over the global intersection rectangle;
4. intersection = count of logical AND in that overlap;
5. union = `first.mask_area + second.mask_area - intersection`;
6. union <= 0 -> `0.0`;
7. return `intersection / union`.

Temporary bool allocation, if any, may only have the overlap-crop shape.

Do NOT use full-frame `np.logical_or`.

Duplicate grouping threshold remains exactly:

```text
IoU >= 0.50
```

Winner priority remains exactly:

```text
touches_image_border
→ -border_clearance
→ -mask_area
→ -confidence
→ source_tile_id
→ raw_index
```

Stable global proposal IDs remain sorted exactly as before:

```text
centroid_y
→ centroid_x
→ mask_area
```

No union of duplicate masks is introduced.

# 10. Adapt Reference context without full masks

In canonical `core.py`, update `reference_mask_from_proposal()`.

It must:

1. allocate only the existing 512×512 reasoning-context bool mask;
2. intersect `proposal.global_bbox` with the 512 reasoning context in global coordinates;
3. slice the corresponding portion of `proposal.mask_crop`;
4. paste that slice into the correct location in the 512×512 context mask;
5. preserve all existing padding/context behavior.

Do NOT materialize a full-image proposal mask.

# 11. Adapt diagnostics preview without full masks

In canonical `outputs.py`, update `proposals_preview_image()`.

For each proposal:

- bbox drawing remains unchanged;
- centroid marker remains unchanged;
- compute `_erode()` only on `proposal.mask_crop`;
- compute magenta outline in crop coordinates;
- paste that outline only into `preview[top:bottom+1, left:right+1]`.

Do NOT materialize a full-image proposal mask.

Preview output meaning/colors must remain unchanged.

# 12. Existing helper policy

Existing generic helpers such as:

```text
bbox_of
centroid_of
touches_original_border
border_clearance
```

may remain for compatibility/tests.

However, the normal `detect_global → merge_proposals → downstream` product path must not construct or retain full-image proposal masks.

Do not add a lazy `global_mask` property that materializes H×W arrays.

# 13. Unit/regression tests

Modify only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
```

Update existing proposal helpers/tests for `mask_crop`.

Add all of the following.

## 13.1 Compact representation invariant

For synthetic full masks converted by a test helper:

- crop shape equals global bbox extent;
- crop is tight;
- area, bbox, centroid, border-touch and clearance equal a test-only legacy full-mask calculation.

## 13.2 IoU equivalence

For at least:

- identical masks;
- partially overlapping masks;
- disjoint masks;
- masks whose bboxes overlap but foreground does not;

assert:

```text
proposal_iou(compact proposals)
==
legacy full-frame intersection/union IoU
```

within floating tolerance.

## 13.3 Merge-equivalence regression

Create a deterministic synthetic proposal set containing:
- duplicate overlap > 0.50;
- non-duplicate overlap < 0.50;
- border vs non-border duplicate;
- confidence tie-break case;
- distinct centroids.

Implement a **test-only** small legacy full-frame merge reference.

Assert new compact merge gives identical:
- duplicate grouping;
- winner source/raw identity;
- proposal count;
- mask area;
- bbox;
- centroid;
- confidence;
- stable proposal IDs.

Do not put legacy full-frame implementation into product code.

## 13.4 Reference context equivalence

For synthetic proposals and contexts:
- fully inside;
- clipped by context left/top;
- clipped by context right/bottom;

assert new `reference_mask_from_proposal()` is bit-identical to a test-only legacy full global-mask slice.

## 13.5 Preview smoke

Construct compact proposals and call `proposals_preview_image()`.

Assert:
- output size equals input RGB;
- bbox/centroid/outline execution succeeds;
- no full proposal mask is required.

## 13.6 5000×5000 no-full-bool regression

No model inference.

Mechanically test `DetectorRuntime.detect_global()` with:

- a fake RGB-like object exposing `shape=(5000,5000,3)`;
- monkeypatched `plan_tiles()` returning a tiny fixed set of windows;
- monkeypatched `extract_tile()` returning a fixed 512×512 RGB tile;
- monkeypatched `detect_tile()` returning fixed small bool detections.

Monkeypatch detector-module `np.zeros` so it raises if code requests:

```text
shape == (5000, 5000)
and dtype == bool
```

The compact detector path must complete without triggering that guard.

Also ensure pairwise IoU receives/slices compact masks only; no operand with shape `(5000,5000)` may reach a logical operation.

This test must not allocate a real 5000×5000 RGB image.

## 13.7 Serialization contract

Assert `GlobalProposal.to_dict()` retains the same existing externally visible keys/semantics and does not serialize `mask_crop`.

# 14. Dedicated canonical test gate

From canonical delivery source:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1
ENV_PYTHON -m pytest tests/test_task8b_runtime.py -q
```

Must PASS.

If it fails:
- no second implementation attempt in this task after the gate;
- report/commit/push PARTIAL;
- STOP.

# 15. Canonical full test gate

Only if dedicated tests pass:

```bat
ENV_PYTHON -m pytest tests -q
```

Must PASS.

Do not run external delivery tests.

Do not run real inference.

# 16. Manifest update and verification

Keep `source_manifest.json` path set exactly unchanged from starting HEAD.

Expected manifest entry count remains:

```text
135
```

Update hash/size only for canonical files that actually changed and are manifest-listed.

Expected functional manifest changes are limited to:

```text
buildreasonseg/runtime/detector.py
buildreasonseg/runtime/core.py
buildreasonseg/runtime/outputs.py
tests/test_task8b_runtime.py
```

Verify locally, without syncing external delivery:

- manifest has exactly 135 entries;
- same path set as at starting HEAD;
- all 135 canonical files exist;
- every manifest size/hash matches current canonical content.

If any other manifest-listed path changed -> STOP.

# 17. External delivery must remain untouched

Do NOT call the sync helper in write mode.

Do NOT copy any changed canonical file into:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Do NOT install/change runtime packages.

This task intentionally leaves canonical and external delivery different until ChatGPT audits M1A.

# 18. Report

Create:

```text
docs/task8b3_m1a_compact_proposal_masks.md
```

Required sections:

1. Task / scope
2. Starting branch/HEAD
3. Frozen architecture decision
4. Changed canonical paths
5. Compact mask representation
6. IoU/merge semantic-equivalence proof
7. Downstream core/preview adaptations
8. Dedicated test result
9. Canonical full-test result
10. Manifest 135/135 verification
11. External delivery = UNCHANGED
12. Real inference = NOT RUN
13. Scientific scope statement:
   - no detector model/threshold/tiling change;
   - no merge threshold/winner/ID change;
   - no Reference eligibility/selection change;
   - no language/SAM2/D-B1/relation/context/validity change;
14. Remaining defects:
   - PROP-01 unchanged;
   - REF-01 unchanged;
   - MASK-01 unchanged;
15. Next gate = Awaiting ChatGPT audit before any canonical→delivery sync.

# 19. FROM_DSH

Update `handoff/FROM_DSH.md`, preserving `ARTIFACT-FACTS` exactly.

Required:

```text
Task: 8B.3-M1A
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: 7a9c576692abf82510d61fb1c814df47d4b053a5
MEM-01 implementation: <status>
Representation: tight bbox + mask_crop
Full-frame proposal masks retained: NO
Full-frame pairwise IoU temporaries: NO
Dedicated tests: <result>
Canonical tests: <result>
Manifest: 135/135 / other
External delivery modified: NO
Real inference executed: NO
Scientific model/checkpoint changed: NO
PROP-01: UNCHANGED
REF-01: UNCHANGED
MASK-01: UNCHANGED
Report: docs/task8b3_m1a_compact_proposal_masks.md
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit; do not sync delivery.
```

# 20. Git gate

Allowed repository changes exactly:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/outputs.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Run:

```bat
git status --short
git diff --check
git diff
```

Anything else -> STOP.

Stage individually. Never `git add .` or `git add -A`.

# 21. Commit / push

If COMPLETE, exact commit:

```text
fix(rc1): use compact proposal masks
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record compact proposal refactor stop
```

Push:

```bat
git push -u origin fix/task8b3-mem01-compact-proposals
```

No force push.

Do NOT merge to main.

# 22. COMPLETE definition

COMPLETE only if:

- exact starting state and new branch;
- pre-change `global_mask` dependency gate matches expectations;
- no external delivery modification;
- no real inference;
- no model/config/semantic policy change;
- raw proposals retained as tight crops only;
- no full H×W proposal mask allocation in normal detector/merge path;
- duplicate IoU uses bbox-overlap crops only;
- merge/winner/ID semantics preserved;
- core Reference crop adapted;
- diagnostics preview adapted;
- dedicated tests pass;
- all canonical tests pass;
- 5000×5000 synthetic guard proves no full bool proposal allocation;
- manifest path set unchanged and 135/135 valid;
- report and FROM_DSH complete;
- only allowed paths changed;
- commit/push succeed;
- working tree clean;
- DSH stops.

# 23. Final response

```text
TASK 8B.3-M1A COMPLETE / PARTIAL / STOP / FAILED

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
<result>

Canonical tests:
<result>

5000x5000 synthetic guard:
PASS / FAIL / NOT RUN

Manifest:
135/135 / other

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

等待 ChatGPT 审核；不得同步 delivery、不得运行真实 Demo、不得进入 Task 8B.4 或 Task 8C。
```
