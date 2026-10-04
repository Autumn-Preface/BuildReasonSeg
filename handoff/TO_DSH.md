# TO_DSH — Task 8B.3-REF01-F1-R1: Continue Locked Demo Reference Forensics

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `95f4403837bd7057fc331462503875fa847f43c8`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> RC1 Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

The previous Task 8B.3-REF01-F1 STOP is accepted as a compliant STOP.

Verified entering facts:

```text
branch = fix/task8b3-ref01-reference-forensics
STOP commit = 95f4403837bd7057fc331462503875fa847f43c8
parent = e00396c85e0fb0ac966f72fa7216b34c72b94cb7
commit message = docs(rc1): record reference forensics stop
detector passes used = 0
Qwen / SAM2 / D-B1 / target segmentation = NONE
candidate replacement = NONE
manual visual judgement = NONE
external / canonical RC1 modification = NONE
main unchanged = 57b368d5647e842d8f31d6d1a9997bf1df3cc0fb
```

The prior turn established only:
- 135/135 external manifest-listed-file integrity;
- the four immutable v0.2 TEST records resolve uniquely;
- their native-vector reference IDs match the stored reference provenance.

It did **not** complete the GT-mask / detector / IoU classification procedure.

Therefore:

```text
REF01-F1 = NOT COMPLETE
forensic classification = NOT YET ESTABLISHED
```

This continuation completes the same frozen task. It is not a new experiment and does not change the four candidates.

# 1. Scientific-use disclosure — mandatory

The final report MUST contain this exact English sentence:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Also include a faithful Chinese disclosure.

This is engineering forensics using TEST-ground-truth metadata after Task 7J final-test consumption.

Do not call the TEST split untouched.

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = 95f4403837bd7057fc331462503875fa847f43c8
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Do not create another branch.
Do not merge/rebase/reset/stash/clean/cherry-pick.

If branch or HEAD differs:
- STOP before any detector execution.

# 3. Strict prohibitions

Do NOT:
- modify canonical RC1 source/config/model;
- modify external RC1 source/config/model;
- run write sync;
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- run training/fine-tuning/export/download;
- change detector weights/config/thresholds;
- change `FROZEN_THRESHOLD=0.5`;
- change detector `imgsz/conf/max_det/tiling/NMS`;
- change merge IoU;
- change eligibility rules;
- change reference selector;
- change the 0.50 reference-coverage diagnostic threshold;
- inspect source images or proposal previews visually;
- make any subjective judgement;
- replace any locked candidate;
- regenerate BuildSpatialReason v0.2;
- regenerate WHU native-vector annotations/caches;
- compute any Task 7J performance metric;
- update `main`;
- force push.

GT may be accessed only for the canonical **reference** of the four locked records.

# 4. Allowed tracked changes

Only:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No runtime/test/config/model file may change.

# 5. Re-establish read-only preflight

Do not rely only on the previous STOP report.

Run:

```text
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

Require:

```text
checked = 135
match = 135
missing = 0
mismatch = 0
exit = 0
```

Then from the external RC1 root run:

```text
<RC1_PYTHON> check_setup.py
```

Require:

```text
exit = 0
BuildReasonSeg environment: READY
```

If either fails:
- detector passes used remain 0;
- STOP.

# 6. Frozen v0.2 TEST identity gate

Require local file:

```text
datasets/build_spatial_reason/v0.2/test.jsonl
```

Exact frozen identity:

```text
bytes = 14415287
sha256 = 72525bff76ef5bdc34dedcb257a5f9c72c420fce0d78dbaa34b02d5f230301dc
records = 6219
```

Require manifest:

```text
dataset_version = v0.2
split_view = scene_disjoint_v1
source_component_representation_version = whu-native-vector-v1.0
```

If missing/mismatched:
- do NOT regenerate;
- STOP.

# 7. Immutable locked records

Exactly:

```text
right:
buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91
tile = 1010
query = largest_to_right_of_to_nearest

left:
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
tile = 1003
query = largest_to_left_of_to_nearest

above:
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314
tile = 1008
query = largest_to_above_to_nearest

below:
buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
tile = 1009
query = largest_to_below_to_nearest
```

Read `test.jsonl` once.

Each must resolve exactly once and require:

```text
split = test
level = 3
image_id = exact tile
query_type = exact query
len(reference_component_ids) = 1
len(native_vector.references) = 1
reasoning_steps[0].operation = argmax_area
reasoning_steps[0].output_component_id = reference_component_ids[0]
native_vector.references[0].tile_instance_id = reference_component_ids[0]
```

Instantiate existing `NativeVectorDataset()` and require:

```python
validate_reasoning_record(record, dataset) == []
```

Require:

```text
dataset.get_source_feature_id(tile, reference_id)
=
native_vector.references[0].source_feature_id
```

Any mismatch:
- STOP.

# 8. Exact source rasters

Use exactly these local source rasters:

```text
right:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif
SHA256 = 1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2

left:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif
SHA256 = eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38

above:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif
SHA256 = 0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd

below:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif
SHA256 = c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7
```

Require every SHA exactly matches before detector execution.

Require every image decodes as RGB 512×512.

If not:
- STOP.

# 9. Canonical GT reference masks

For each record:

```python
gt_reference_id = record["reference_component_ids"][0]
label_map = dataset.label_map(tile_id)
gt_reference_mask = (label_map == gt_reference_id)
```

Require:
- label_map exists;
- shape = `(512, 512)`;
- GT reference mask non-empty;
- the per-tile native-vector cache already exists;
- no cache regeneration is performed.

From `dataset.list_instances(tile_id)`, find the exact reference row.

Require:

```text
int(gt_reference_mask.sum()) = clipped_area_px
```

Record:

```text
gt_reference_id
gt_source_feature_id
gt_area_px
gt_bbox_xyxy_px
gt_centroid_px
gt_touches_tile_border
```

No target mask/target GT metric in this task.

If cache is missing:
- STOP.

# 10. Existing P1D12 diagnostics gate

Read only these existing external files:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\proposals.json
```

Frozen counts:

```text
1010 raw=6  merged=6  eligible=4
1003 raw=66 merged=53 eligible=42
1008 raw=9  merged=9  eligible=4
1009 raw=7  merged=6  eligible=3
```

Do not modify those files.

If their recorded counts disagree:
- STOP before detector execution.

# 11. External implementation identity gate

The forensic script must import these from the **external RC1**:

```python
from buildreasonseg.runtime.detector import (
    DetectorRuntime,
    eligible_proposals,
    select_reference,
)
from buildreasonseg.runtime.imageio import load_image
```

Before inference, inspect module `__file__`.

Require both module paths resolve underneath exactly:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

If repository/canonical modules are imported:
- STOP before detector execution.

Also record detector constants from the imported external module:

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

Any mismatch:
- STOP.

# 12. Detector execution — exact allowance

Previous STOP used zero detector passes.

Remaining allowance:

```text
right: 1 pass
left: 1 pass
above: 1 pass
below: 1 pass
```

Run exactly once per candidate, in this order:

```text
1010
1003
1008
1009
```

Use one default `DetectorRuntime()` object.

For each image exactly:

```python
loaded = load_image(exact_source_path)
detection = detector.detect_global(loaded.rgb)
```

Do not call `detect_global` twice for any candidate.
Do not call `predict.py`.
Do not retry a mismatch.
Do not write any external diagnostic.

# 13. Deterministic reproduction gate

Before using proposal masks for GT analysis, require detector rerun metadata reproduces P1D12.

For each candidate require exact:

```text
raw_count
len(merged)
eligible_count
```

Expected:

```text
right  = 6 / 6 / 4
left   = 66 / 53 / 42
above  = 9 / 9 / 4
below  = 7 / 6 / 3
```

Match rerun `merged` against frozen P1D12 `proposals.json` by `proposal_id`.

Require exact:
- `proposal_id`
- `source_tile_id`
- `mask_area`
- `global_bbox`
- `touches_image_border`
- `raw_index`

Require absolute difference ≤ `1e-6`:
- `confidence`
- each `centroid` coordinate
- `border_clearance`
- `bbox_extent_ratio`

If any mismatch:
- STOP immediately;
- no IoU classification;
- no second detector call.

# 14. Reconstruct proposal masks

For each merged proposal:

```python
full = np.zeros((512, 512), dtype=bool)

top, left, bottom, right = proposal.global_bbox

assert proposal.mask_crop.shape == (
    bottom - top + 1,
    right - left + 1,
)

full[top:bottom + 1, left:right + 1] = proposal.mask_crop
```

No resize/morphology/dilation/erosion/union/refinement.

IoU:

```python
intersection = np.logical_and(full, gt_reference_mask).sum()
union = np.logical_or(full, gt_reference_mask).sum()
iou = float(intersection / union)
```

Require `union > 0`.

Centroid error:

```text
Euclidean distance in pixels:
proposal.centroid ↔ GT reference centroid
```

# 15. Production selected reference

Use exact frozen function:

```python
selected = select_reference(merged, family="largest")
```

Do not reimplement selection.

Require selected is not `None`.

Record:

```text
selected_proposal_id
selected_mask_area
selected_confidence
selected_bbox
selected_iou_to_gt
selected_centroid_error_px
selected_area_to_gt_ratio
```

# 16. Diagnostic oracle proposals

Coverage threshold remains frozen from Task 7F:

```text
REFERENCE_COVERAGE_IOU_THRESHOLD = 0.50
```

This is diagnostic only, not tuned here.

Eligible set:

```python
eligible = eligible_proposals(merged, family="largest")
```

Define `best_eligible` by ascending tuple:

```python
(
    -iou_to_gt,
    -confidence,
    proposal_id,
)
```

over eligible proposals.

Define `best_any_merged` with the same tuple over all merged proposals.

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

No production decision is changed.

# 17. Exclusive per-candidate classification

Exactly one class.

## `REFERENCE_SELECTED_CORRECT`

```text
selected_iou_to_gt >= 0.50
```

## `REFERENCE_SELECTION_WRONG_COVERED`

```text
selected_iou_to_gt < 0.50
AND
best_eligible_iou >= 0.50
```

## `REFERENCE_ELIGIBILITY_BLOCKED`

```text
selected_iou_to_gt < 0.50
AND
best_eligible_iou < 0.50
AND
best_any_iou >= 0.50
```

## `REFERENCE_COVERAGE_MISSING`

```text
best_any_iou < 0.50
```

No fifth class.
No subjective override.

# 18. Overall outcome / next gate

Priority is fixed:

## If any candidate = `REFERENCE_COVERAGE_MISSING`

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = COVERAGE
NEXT = REF01_COVERAGE_FRAGMENTATION_FORENSICS
```

## Else if any = `REFERENCE_ELIGIBILITY_BLOCKED`

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

## Else if any = `REFERENCE_SELECTION_WRONG_COVERED`

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = SELECTION
NEXT = REF01_SELECTION_REPAIR_DESIGN
```

## Else all four = `REFERENCE_SELECTED_CORRECT`

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_GATE_PASS
Dominant next blocker = NONE_IN_LOCKED_REFERENCE_SET
NEXT = MASK01_LOCKED_DEMO_END_TO_END_FORENSICS
```

Do not execute NEXT.

# 19. Forensic script

Create:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

Requirements:
- read-only toward external RC1 and dataset;
- implements §§5–18 exactly;
- one detector call per candidate maximum;
- no image/PNG/mask/checkpoint output;
- writes only:
  `evaluation/task8b3_ref01_locked_reference_forensics.json`;
- non-zero exit on any gate failure.

Evidence JSON required top-level keys:

```text
task
continuation_of
starting_head
external_root
external_integrity
external_setup
test_jsonl_sha256
test_jsonl_bytes
test_record_count
scientific_reuse_disclosure
coverage_threshold
detector_config
candidates
overall_outcome
dominant_next_blocker
next_gate
```

Each candidate must contain:
- immutable sample ID/tile/query;
- exact raster SHA;
- canonical GT reference facts;
- detector reproduction facts;
- selected proposal facts;
- best eligible facts;
- best any facts;
- exact exclusive class.

# 20. Canonical report

Update the existing:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Do not erase the fact that the prior turn STOPPED at preflight.

Make the report clearly state:

```text
prior STOP commit = 95f4403837bd7057fc331462503875fa847f43c8
continuation = Task 8B.3-REF01-F1-R1
detector passes before continuation = 0
```

Required final sections:
1. prior STOP and continuation;
2. exact English + Chinese TEST-reuse disclosure;
3. external 135/135 + setup READY;
4. v0.2 TEST identity;
5. four immutable record/reference identities;
6. exact four raster identities;
7. GT reference-mask integrity;
8. detector reproduction gate;
9. proposal-to-GT measurements;
10. exclusive classifications;
11. aggregate class counts;
12. dominant next blocker;
13. nonclaims;
14. exact Outcome;
15. exact NEXT.

Required table:

```text
relation | GT_ref_id | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_IoU | class
```

# 21. No claim inflation

Do not claim:
- REF-01 globally fixed;
- detector generally solved;
- historical A2 fixed;
- cross-city generalization;
- arbitrary aerial-image robustness;
- final Demo success;
- target segmentation success;
- Task 7J metrics improved;
- untouched TEST.

# 22. FROM_DSH

Preserve `ARTIFACT-FACTS` exactly.

Required fields:

```text
Task: 8B.3-REF01-F1-R1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: 95f4403837bd7057fc331462503875fa847f43c8
Prior STOP detector passes: 0
External sync check: 135/135 PASS / FAIL
External setup: READY / NOT READY
BuildSpatialReason v0.2 test identity: PASS / FAIL
v0.2 test records: 6219 / other
Locked record resolution: 4/4 UNIQUE / other
Native-vector provenance validation: 4/4 PASS / other
GT reference-mask integrity: 4/4 PASS / other
Locked raster SHA identity: 4/4 PASS / other
Detector inference: 4 locked rasters × 1 pass / other
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
Task7J metric/model/seed/threshold/architecture changed: NO
right class: <exact class>
left class: <exact class>
above class: <exact class>
below class: <exact class>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / GATE_PASS / other
Outcome: <exact enum>
Dominant next blocker: <exact enum>
Next gate: <exact enum>
Evidence: evaluation/task8b3_ref01_locked_reference_forensics.json
Report: docs/task8b3_ref01_locked_reference_forensics.md
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 23. Commit / push / STOP

If COMPLETE:

```text
docs(rc1): complete locked demo reference forensics
```

If STOP/FAILED:

```text
docs(rc1): record reference forensics continuation stop
```

Push only:

```text
fix/task8b3-ref01-reference-forensics
```

No force push.
Do not update old PROP-01 branch.
Do not update main.

After push:
- STOP;
- wait for ChatGPT audit.

# 24. COMPLETE definition

COMPLETE only if all are true:

```text
required branch/head exact
only five allowed tracked paths changed
external integrity 135/135 PASS
external setup READY
v0.2 TEST exact bytes/SHA/count
four immutable records 4/4 unique
native-vector provenance 4/4 valid
four GT reference masks 4/4 valid
four raster SHA identities exact
exactly one detector pass per candidate
detector rerun reproduces P1D12 exactly
no Qwen/SAM2/relation/D-B1/target segmentation
no external write
no manual visual judgement
no candidate replacement
0.50 diagnostic threshold unchanged
one exclusive class per candidate
Outcome/NEXT chosen mechanically
evidence JSON + canonical report + FROM_DSH committed
branch pushed
STOP
```
