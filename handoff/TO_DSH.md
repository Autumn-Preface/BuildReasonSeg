# TO_DSH — Task 8B.3-REF01-F1-R2: Execute the Frozen Reference IoU Classification

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `141af9cfd943dd13366947ea3a31637a3f8c7bc9`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> RC1 Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit decision

Task 8B.3-REF01-F1-R1 STOP is accepted as a safe STOP.

The two items DSH said were "not pinned down" were already explicitly frozen in the prior task book. This R2 removes all ambiguity.

## Frozen fact A — proposal mask source

DO NOT reconstruct proposal masks from `proposals.json`.

`proposals.json` is metadata-only and is used only for deterministic reproduction checks.

The proposal mask used for IoU is the **in-memory `GlobalProposal.mask_crop` returned by the one authorised detector rerun**.

For each rerun proposal:

```python
full = np.zeros((512, 512), dtype=bool)

top, left, bottom, right = proposal.global_bbox

assert proposal.mask_crop.shape == (
    bottom - top + 1,
    right - left + 1,
)

full[top:bottom + 1, left:right + 1] = proposal.mask_crop
```

No resize, morphology, dilation, erosion, union, SAM2 refinement or threshold change is permitted.

## Frozen fact B — diagnostic coverage threshold

The reference coverage threshold is exactly:

```text
IoU >= 0.50
```

This is the already frozen Task 7F threshold.

It is NOT tuned, selected or changed in this task.

No further decision is needed before detector execution.

# 1. Entering evidence — already accepted, do not redo unless required for script input

The previous two STOP commits already establish:

```text
external RC1 manifest-listed source/config = 135/135 PASS
four immutable v0.2 TEST records = UNIQUE and provenance-valid
canonical GT reference masks = 4/4 reconstructed
GT mask non-empty = 4/4
GT mask area == clipped_area_px = 4/4
detector passes used so far = 0
Qwen / SAM2 / D-B1 / target segmentation = NONE
candidate replacement / repair / visual judgement = NONE
external / canonical RC1 modified = NO / NO
```

Frozen GT reference facts:

```text
right:
tile = 1010
reference tile_instance_id = 4
GT area = 2478

left:
tile = 1003
reference tile_instance_id = 26
GT area = 3512

above:
tile = 1008
reference tile_instance_id = 4
GT area = 5013

below:
tile = 1009
reference tile_instance_id = 3
GT area = 2606
```

Do not spend this task repeating large preflight prose. The forensic script may load the same records/caches as inputs, but do not stop for a new technical decision unless an actual contradiction occurs.

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = 141af9cfd943dd13366947ea3a31637a3f8c7bc9
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No new branch.
No merge/rebase/reset/stash/clean/cherry-pick.

If branch/HEAD differs:
- detector passes remain 0;
- STOP.

# 3. Strict prohibitions

Do NOT:
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- modify canonical RC1;
- modify external RC1;
- run sync write;
- change any detector threshold/config/weight;
- change selector/eligibility;
- change `IoU >= 0.50`;
- inspect source images/proposal PNGs visually;
- use subjective visual judgement;
- replace any locked candidate;
- regenerate dataset records or native-vector caches;
- compute Task 7J metrics;
- design or implement a repair;
- update `main`;
- force push.

# 4. Allowed tracked changes

Only:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 5. Scientific-use disclosure

The final report MUST contain exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Also include a faithful Chinese disclosure.

# 6. Immutable candidates and source rasters

Use exactly:

```text
right:
sample_id = buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91
tile = 1010
image = C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif
image_sha256 = 1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2

left:
sample_id = buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
tile = 1003
image = C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif
image_sha256 = eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38

above:
sample_id = buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314
tile = 1008
image = C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif
image_sha256 = 0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd

below:
sample_id = buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
tile = 1009
image = C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif
image_sha256 = c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7
```

Before detector execution:
- require each SHA match exactly;
- require each image = RGB-readable 512×512.

Any mismatch:
- STOP;
- no candidate substitution.

# 7. Existing P1D12 proposal metadata

Read only existing:

```text
external\inference\output\diagnostics\1010\proposals.json
external\inference\output\diagnostics\1003\proposals.json
external\inference\output\diagnostics\1008\proposals.json
external\inference\output\diagnostics\1009\proposals.json
```

Frozen expected counts:

```text
right  = raw 6  / merged 6  / eligible 4
left   = raw 66 / merged 53 / eligible 42
above  = raw 9  / merged 9  / eligible 4
below  = raw 7  / merged 6  / eligible 3
```

`proposals.json` is ONLY the metadata reference for §10.
It is NOT the proposal-mask source.

# 8. External detector implementation gate

The forensic script must force/import the implementation from external RC1.

Use:

```python
from buildreasonseg.runtime.detector import (
    DetectorRuntime,
    eligible_proposals,
    select_reference,
)
from buildreasonseg.runtime.imageio import load_image
```

Require the imported `detector.py` and `imageio.py` resolve under:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Record and require:

```text
TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = 384
IMGSZ = 640
CONF = 0.05
MAX_DET = 300
DUPLICATE_IOU = 0.50
MERGE_BBOX_EXTENT_RATIO_MAX = 0.20
FROZEN_THRESHOLD = 0.5
```

If any mismatch:
- STOP before detector execution.

# 9. GT reference masks

Use the already validated native-vector source:

```python
dataset = NativeVectorDataset()
label_map = dataset.label_map(tile_id)
gt_mask = label_map == gt_reference_id
```

Expected reference IDs:

```text
1010 -> 4
1003 -> 26
1008 -> 4
1009 -> 3
```

Expected GT areas:

```text
1010 -> 2478
1003 -> 3512
1008 -> 5013
1009 -> 2606
```

Require exact.

Do not regenerate missing caches.

# 10. Detector execution — exactly four calls total

Run in this exact order:

```text
1010
1003
1008
1009
```

Create one default:

```python
detector = DetectorRuntime()
```

For each candidate exactly once:

```python
loaded = load_image(source_path)
detection = detector.detect_global(loaded.rgb)
```

Maximum authorised detector calls in this task:

```text
4
```

No retry.
No second call for any candidate.

If a detector call errors:
- STOP;
- do not retry.

# 11. Mandatory reproduction gate before any GT classification

For each rerun require exact:

```text
raw_count
merged count
eligible count
```

Expected:

```text
1010 = 6 / 6 / 4
1003 = 66 / 53 / 42
1008 = 9 / 9 / 4
1009 = 7 / 6 / 3
```

Compare every rerun merged proposal to its existing `proposals.json` item by `proposal_id`.

Require exact:
- proposal_id
- source_tile_id
- mask_area
- global_bbox
- touches_image_border
- raw_index

Tolerance ≤ `1e-6`:
- confidence
- centroid row/col
- border_clearance
- bbox_extent_ratio

If any mismatch:
- STOP;
- do not classify;
- do not rerun.

# 12. Proposal-mask source — explicit and final

For IoU, use each rerun `GlobalProposal.mask_crop`.

Do NOT use a mask from `proposals.json`.

For each proposal:

```python
top, left, bottom, right = proposal.global_bbox

assert proposal.mask_crop.dtype == bool
assert proposal.mask_crop.shape == (
    bottom - top + 1,
    right - left + 1,
)

proposal_mask = np.zeros((512, 512), dtype=bool)
proposal_mask[top:bottom + 1, left:right + 1] = proposal.mask_crop
```

Compute:

```python
intersection = np.logical_and(proposal_mask, gt_mask).sum()
union = np.logical_or(proposal_mask, gt_mask).sum()
assert union > 0
iou = float(intersection / union)
```

No other mask transformation is authorized.

# 13. Production selected reference

Use exactly:

```python
selected = select_reference(detection["merged"], family="largest")
```

Require not None.

Record:

```text
selected_id
selected_mask_area
selected_confidence
selected_bbox
selected_iou_to_gt
selected_centroid_error_px
selected_area_to_gt_ratio
```

# 14. Best eligible / best any diagnostic

Use exactly:

```python
eligible = eligible_proposals(detection["merged"], family="largest")
```

Coverage threshold:

```text
0.50
```

Rank both eligible proposals and all merged proposals by:

```python
(
    -iou_to_gt,
    -confidence,
    proposal_id,
)
```

Record:

```text
best_eligible_id
best_eligible_iou
best_eligible_confidence
best_eligible_area
best_any_id
best_any_iou
best_any_is_eligible
selected_to_best_eligible_iou_gap
```

# 15. Exclusive classification — mechanical only

Exactly one per candidate.

```text
REFERENCE_SELECTED_CORRECT
if selected_iou_to_gt >= 0.50
```

```text
REFERENCE_SELECTION_WRONG_COVERED
if selected_iou_to_gt < 0.50
and best_eligible_iou >= 0.50
```

```text
REFERENCE_ELIGIBILITY_BLOCKED
if selected_iou_to_gt < 0.50
and best_eligible_iou < 0.50
and best_any_iou >= 0.50
```

```text
REFERENCE_COVERAGE_MISSING
if best_any_iou < 0.50
```

No visual override.
No fifth class.

# 16. Overall outcome — exact priority

If ANY candidate is:

```text
REFERENCE_COVERAGE_MISSING
```

then:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = COVERAGE
NEXT = REF01_COVERAGE_FRAGMENTATION_FORENSICS
```

Else if ANY candidate is:

```text
REFERENCE_ELIGIBILITY_BLOCKED
```

then:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Else if ANY candidate is:

```text
REFERENCE_SELECTION_WRONG_COVERED
```

then:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = SELECTION
NEXT = REF01_SELECTION_REPAIR_DESIGN
```

Else all four are:

```text
REFERENCE_SELECTED_CORRECT
```

then:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_GATE_PASS
Dominant next blocker = NONE_IN_LOCKED_REFERENCE_SET
NEXT = MASK01_LOCKED_DEMO_END_TO_END_FORENSICS
```

Do not execute NEXT.

# 17. Script output

Create/update:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

It may write only:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

No PNGs.
No masks.
No external outputs.
No checkpoints.

Evidence JSON must contain:
- starting HEAD;
- exact scientific reuse disclosure;
- detector module paths/config;
- detector call count;
- coverage threshold = 0.50;
- candidate records;
- GT facts;
- reproduction facts;
- selected/best eligible/best any facts;
- exact exclusive classifications;
- overall outcome;
- dominant next blocker;
- NEXT.

# 18. Report

Update:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Preserve both prior STOP histories.

Add final R2 section containing:
- explicit clarification that `proposals.json` was metadata-only;
- in-memory rerun `GlobalProposal.mask_crop` was the mask source;
- threshold = frozen Task 7F IoU 0.50;
- detector calls exactly 4;
- reproduction table;
- GT comparison table;
- class counts;
- outcome/NEXT.

Required table:

```text
relation | GT_ref | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_IoU | class
```

# 19. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: 141af9cfd943dd13366947ea3a31637a3f8c7bc9
Prior detector passes: 0
Proposal mask source: RERUN_GLOBALPROPOSAL_MASK_CROP
proposals.json role: METADATA_REPRODUCTION_ONLY
Coverage threshold: 0.50_TASK7F_FROZEN
Locked raster SHA identities: 4/4 PASS / other
GT reference-mask integrity: 4/4 PASS / other
External detector module identity: PASS / FAIL
Detector calls: 4 / other
Detector reproduction right: 6/6 eligible4 MATCH / other
Detector reproduction left: 66/53 eligible42 MATCH / other
Detector reproduction above: 9/9 eligible4 MATCH / other
Detector reproduction below: 7/6 eligible3 MATCH / other
Qwen execution: NONE
SAM2 execution: NONE
Relation-field execution: NONE
D-B1 execution: NONE
Target segmentation: NONE
Manual visual inspection: NO
Candidate replacement: NO
right selected_IoU / bestEligible_IoU / bestAny_IoU: <values>
left selected_IoU / bestEligible_IoU / bestAny_IoU: <values>
above selected_IoU / bestEligible_IoU / bestAny_IoU: <values>
below selected_IoU / bestEligible_IoU / bestAny_IoU: <values>
right class: <exact class>
left class: <exact class>
above class: <exact class>
below class: <exact class>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / GATE_PASS / other
Outcome: <exact enum>
Dominant next blocker: <exact value>
Next gate: <exact enum>
Evidence: evaluation/task8b3_ref01_locked_reference_forensics.json
Report: docs/task8b3_ref01_locked_reference_forensics.md
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 20. Commit / push

If COMPLETE:

```text
docs(rc1): complete locked reference IoU forensics
```

If STOP/FAILED:

```text
docs(rc1): record locked reference IoU stop
```

Push only:

```text
fix/task8b3-ref01-reference-forensics
```

No force push.
Do not update main.
After push: STOP.

# 21. COMPLETE definition

COMPLETE only if:

```text
branch/head exact
only allowed tracked files changed
four raster SHA identities exact
four GT reference masks exact
external detector module identity exact
detector calls exactly 4
P1D12 reproduction exact for all four
proposal IoU uses rerun in-memory GlobalProposal.mask_crop
coverage threshold exactly 0.50
Qwen/SAM2/relation/D-B1/target segmentation not run
manual visual judgement none
candidate replacement none
exactly one class per candidate
mechanical overall Outcome/NEXT
evidence JSON/report/FROM_DSH committed and pushed
STOP
```
