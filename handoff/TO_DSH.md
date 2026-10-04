# TO_DSH — Task 8B.3-REF01-F1: Locked Demo Reference Forensics

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Starting branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `e00396c85e0fb0ac966f72fa7216b34c72b94cb7`
> Task branch: `fix/task8b3-ref01-reference-forensics`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> RC1 Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT frozen entering state

Task 8B.3-P1D12 is now APPROVED.

Freeze:

```text
P1D12 proposal path =
PROP01_LOCKED_DEMO_PROPOSAL_GATE_PASS

P1D12 evidence =
CLOSED

locked raster identity =
4/4 exact immutable SHA match

inspect-only execution =
CONFIRMED

right raw / merged / eligible =
6 / 6 / 4

left raw / merged / eligible =
66 / 53 / 42

above raw / merged / eligible =
9 / 9 / 4

below raw / merged / eligible =
7 / 6 / 3

historical PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

NEXT =
REF01_LOCKED_DEMO_REFERENCE_FORENSICS
```

P1D12-R2 also established:

```text
FROZEN_THRESHOLD = 0.5
timings.sam2 = 0.0
timings.relation_fields = 0.0
timings.db1 = 0.0
```

for all four existing inspect-mode results.

The old handoff line saying timing entries were `NONE` is superseded by the later correction commit:
the authoritative fact is that the three entries exist inside the nested `timings` object and are explicitly `0.0`.

# 1. Scientific-use disclosure — mandatory

The four locked qualitative Demo candidates come from the frozen BuildSpatialReason v0.2 TEST split, and Task 7J
final frozen-architecture test metrics had already been consumed before these qualitative candidates were selected.

This task is an engineering forensic use of TEST-ground-truth metadata after final-test consumption.

It MUST NOT:
- change/recompute any reported Task 7J scientific metric;
- select/re-select any model;
- select/re-select any seed;
- tune any threshold;
- change architecture;
- replace a Demo candidate;
- claim the TEST split was untouched.

The final report must include this exact English disclosure:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Also include a faithful Chinese disclosure.

# 2. Purpose

For each of the four immutable locked candidates, determine mechanically whether the production `largest` reference
failure mode is:

```text
A. selected proposal already covers the canonical GT reference;
B. wrong eligible proposal was selected although another eligible proposal covers the GT reference;
C. a covering proposal exists but the frozen eligibility rule excludes it;
D. no merged proposal covers the canonical GT reference.
```

This is diagnostics only.

No repair is authorized.

# 3. Immutable locked candidates

Exactly:

```text
right
sample_id = buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91
tile_id = 1010
query_type = largest_to_right_of_to_nearest

left
sample_id = buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
tile_id = 1003
query_type = largest_to_left_of_to_nearest

above
sample_id = buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314
tile_id = 1008
query_type = largest_to_above_to_nearest

below
sample_id = buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
tile_id = 1009
query_type = largest_to_below_to_nearest
```

No replacement under any result.

# 4. Git gate and task branch

Before any edit/inference require exactly:

```text
current branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = e00396c85e0fb0ac966f72fa7216b34c72b94cb7
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Then handle the task branch mechanically:

```text
if refs/heads/fix/task8b3-ref01-reference-forensics does not exist:
    git switch -c fix/task8b3-ref01-reference-forensics
elif that branch exists AND its HEAD == e00396c85e0fb0ac966f72fa7216b34c72b94cb7:
    git switch fix/task8b3-ref01-reference-forensics
else:
    STOP
```

After switching require:

```text
branch = fix/task8b3-ref01-reference-forensics
starting HEAD = e00396c85e0fb0ac966f72fa7216b34c72b94cb7
```

No rebase/reset/stash/merge/cherry-pick.

# 5. Strict prohibitions

Do NOT:
- modify any canonical RC1 file;
- modify any external RC1 source/config/model file;
- run sync helper in write mode;
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- run training/fine-tuning/export/download;
- modify detector weights;
- change detector `imgsz/conf/max_det/NMS/tiling`;
- change merge IoU;
- change reference eligibility;
- change the selector;
- choose a new threshold;
- visually inspect source images or proposal previews;
- use visual judgement to label reference correctness;
- replace a locked candidate;
- edit/regenerate BuildSpatialReason v0.2;
- edit/regenerate WHU native-vector annotations;
- regenerate missing local dataset caches;
- compute or report Task 7J performance metrics;
- update `main`;
- force push.

Ground-truth access is authorized ONLY as specified in §§9–13 for reference-stage diagnosis.

# 6. Allowed tracked changes

Only:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No product/runtime/test/config source file may change.

# 7. External RC1 integrity preflight

Run read-only:

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

Run external:

```text
<RC1_PYTHON> check_setup.py
```

from the external RC1 root.

Require:

```text
exit = 0
BuildReasonSeg environment: READY
```

If either fails:
- do not run detector;
- STOP.

# 8. BuildSpatialReason v0.2 TEST identity gate

Require local file:

```text
datasets/build_spatial_reason/v0.2/test.jsonl
```

exists.

Require exact frozen identity from `evaluation/task6l_artifact_index.json`:

```text
bytes  = 14415287
sha256 = 72525bff76ef5bdc34dedcb257a5f9c72c420fce0d78dbaa34b02d5f230301dc
records = 6219
tracked = false
```

Do not regenerate it if missing/mismatched.

Also require:

```text
datasets/build_spatial_reason/v0.2/manifest.json
dataset_version = v0.2
split_view = scene_disjoint_v1
source_component_representation_version = whu-native-vector-v1.0
```

If any gate fails:
- STOP.

# 9. Resolve the four canonical v0.2 records

Read `test.jsonl` once.

For each immutable sample ID require exactly ONE matching record.

For each record require:

```text
sample_id = exact immutable ID
split = test
image_id = exact tile_id
level = 3
query_type = exact locked query_type
split_view = scene_disjoint_v1
source_component_representation_version = whu-native-vector-v1.0
len(reference_component_ids) = 1
len(native_vector.references) = 1
```

Require first reasoning step:

```text
reasoning_steps[0].operation = argmax_area
reasoning_steps[0].output_component_id = reference_component_ids[0]
```

Instantiate the existing read-only:

```python
NativeVectorDataset()
```

and require for all four:

```python
validate_reasoning_record(record, dataset) == []
```

Require the native-vector reference block agrees with the record:

```text
native_vector.references[0].tile_instance_id
==
reference_component_ids[0]

dataset.get_source_feature_id(tile_id, reference_component_ids[0])
==
native_vector.references[0].source_feature_id
```

Any duplicate/missing/schema/provenance discrepancy:
- STOP.

# 10. Canonical GT reference mask

For each candidate:

```text
gt_reference_id = reference_component_ids[0]
label_map = NativeVectorDataset.label_map(tile_id)
gt_reference_mask = (label_map == gt_reference_id)
```

Require:
- label map exists;
- shape = 512×512;
- GT reference mask non-empty;
- GT pixel area equals `clipped_area_px` from `dataset.list_instances(tile_id)` for that id.

Record:

```text
gt_reference_id
gt_source_feature_id
gt_area_px
gt_bbox
gt_centroid
gt_touches_tile_border
```

No target GT metric is needed in this task.

If a required local native-vector per-tile cache is missing:
- do NOT regenerate;
- STOP.

# 11. Existing P1D12 proposal evidence gate

Read only existing external:

```text
inference/output/diagnostics/1010/proposals.json
inference/output/diagnostics/1003/proposals.json
inference/output/diagnostics/1008/proposals.json
inference/output/diagnostics/1009/proposals.json
```

Expected frozen counts:

```text
1010 raw=6  merged=6  eligible=4
1003 raw=66 merged=53 eligible=42
1008 raw=9  merged=9  eligible=4
1009 raw=7  merged=6  eligible=3
```

Do not rewrite these diagnostics.

# 12. Exact detector execution authorization

Detector inference is authorized now ONLY because the stored proposal JSON does not contain proposal-mask pixels needed
for IoU-to-GT diagnosis.

Run exactly ONE frozen detector pass per candidate, in order:

```text
right 1010
left  1003
above 1008
below 1009
```

Do NOT use `predict.py`.

The forensic script must import and use the **external RC1** implementations:

```text
buildreasonseg.runtime.detector.DetectorRuntime
buildreasonseg.runtime.detector.eligible_proposals
buildreasonseg.runtime.detector.select_reference
buildreasonseg.runtime.imageio.load_image
```

Before inference assert the imported module files resolve underneath:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

If they resolve to repository/canonical source instead:
- STOP before inference.

Use default frozen `DetectorRuntime` configuration only.

For each source raster:

```python
loaded = load_image(path)
detection = detector.detect_global(loaded.rgb)
```

One `DetectorRuntime` object may be reused across the four rasters.

No second detector call per candidate is allowed.

No output image/diagnostic may be written to external RC1.

# 13. Deterministic reproduction gate before using masks

For each detector rerun require:

```text
raw_count = exact P1D12 raw count
len(merged) = exact P1D12 merged count
```

Compare each rerun merged proposal against the corresponding frozen P1D12 `proposals.json` item by `proposal_id`.

Require exact:
- proposal_id;
- source_tile_id;
- mask_area;
- global_bbox;
- touches_image_border;
- raw_index.

Require absolute difference ≤ 1e-6 for:
- confidence;
- centroid values;
- border_clearance;
- bbox_extent_ratio.

Also require the mechanically computed eligible count equals P1D12:

```text
1010 = 4
1003 = 42
1008 = 4
1009 = 3
```

If any mismatch:
- STOP;
- do not use the rerun masks for GT diagnosis;
- no retry.

# 14. Proposal-mask reconstruction and IoU

For each merged `GlobalProposal`, reconstruct its 512×512 bool mask mechanically:

```python
full = np.zeros((512, 512), dtype=bool)
top, left, bottom, right = proposal.global_bbox
assert proposal.mask_crop.shape == (bottom - top + 1, right - left + 1)
full[top:bottom + 1, left:right + 1] = proposal.mask_crop
```

IoU:

```text
intersection = (proposal_mask AND gt_reference_mask).sum()
union        = (proposal_mask OR  gt_reference_mask).sum()
iou          = intersection / union
```

No morphology, dilation, resize, threshold change or postprocessing.

For centroid error use Euclidean distance in pixels between:
- proposal centroid;
- GT reference mask centroid.

# 15. Production selected reference

Use the frozen external function:

```python
selected = select_reference(merged, family="largest")
```

Do not reimplement/tune its selection rule.

Require selected is not None.

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

# 16. Oracle diagnostic proposals — fixed Task 7F semantics

Use the frozen Task 7F reference threshold:

```text
IoU coverage threshold = 0.50
```

This is NOT a new tuned threshold.

Define eligible set only through external:

```python
eligible = eligible_proposals(merged, family="largest")
```

Define `best_eligible` by:

```text
max:
1. IoU to canonical GT reference
2. higher detector confidence
3. lower proposal_id
```

Define `best_any_merged` using the same ranking over all merged proposals.

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

These are diagnostic oracle measurements only.
They MUST NOT alter production inference in this task.

# 17. Per-candidate exclusive classification

Use exactly one class per candidate.

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

Meaning:
a sufficiently covering eligible proposal exists, but production selected another eligible proposal.

## `REFERENCE_ELIGIBILITY_BLOCKED`

```text
selected_iou_to_gt < 0.50
AND
best_eligible_iou < 0.50
AND
best_any_iou >= 0.50
```

Meaning:
a sufficiently covering merged proposal exists, but no sufficiently covering proposal survives frozen `largest`
eligibility.

## `REFERENCE_COVERAGE_MISSING`

```text
best_any_iou < 0.50
```

Meaning:
the merged proposal set has no proposal reaching the frozen Task 7F coverage threshold for the canonical GT reference.

Do NOT create any additional class.

Do NOT visually reinterpret a class.

# 18. Overall forensic outcome and NEXT

After all four classifications, choose mechanically.

## Case A — any `REFERENCE_COVERAGE_MISSING`

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = COVERAGE
NEXT = REF01_COVERAGE_FRAGMENTATION_FORENSICS
```

## Case B — no coverage-missing, but any `REFERENCE_ELIGIBILITY_BLOCKED`

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

## Case C — neither above, but any `REFERENCE_SELECTION_WRONG_COVERED`

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = SELECTION
NEXT = REF01_SELECTION_REPAIR_DESIGN
```

## Case D — all four `REFERENCE_SELECTED_CORRECT`

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_GATE_PASS
Dominant next blocker = NONE_IN_LOCKED_REFERENCE_SET
NEXT = MASK01_LOCKED_DEMO_END_TO_END_FORENSICS
```

Do not execute NEXT.

# 19. No claim inflation

Regardless of outcome, do NOT claim:
- detector generally solved;
- REF-01 globally fixed;
- cross-city generalization;
- arbitrary aerial-image robustness;
- final Demo success;
- final target mask correctness;
- historical A2 fixed;
- scientific Task 7J metrics improved.

This is four-case post-final-test engineering forensics only.

# 20. Forensic script contract

Create:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

It must:
- be read-only toward dataset/external RC1;
- implement §§8–17 exactly;
- write exactly one repository evidence JSON:
  `evaluation/task8b3_ref01_locked_reference_forensics.json`;
- not write PNGs, masks, model outputs, checkpoints or external diagnostics;
- return non-zero on any gate failure.

The evidence JSON must include:

```text
task
starting_head
external_root
test_jsonl_sha256
test_jsonl_bytes
test_record_count
coverage_threshold
scientific_reuse_disclosure
detector_config
candidates[]
overall_outcome
dominant_next_blocker
next_gate
```

Each candidate object must include all IDs, GT reference facts, reproduction facts, selected/best proposal facts and
exclusive class.

# 21. Report

Create:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Required sections:

1. task / branch / starting HEAD;
2. scientific-use disclosure in English + Chinese;
3. external integrity gate;
4. v0.2 TEST identity gate;
5. immutable four records and native-vector provenance;
6. detector rerun/reproduction gate;
7. per-candidate reference comparison table;
8. per-candidate exclusive class;
9. aggregate class counts;
10. dominant next blocker;
11. what is NOT concluded;
12. exact Outcome;
13. exact NEXT.

Required compact table:

```text
relation | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_IoU | class
```

No preview images or subjective labels.

# 22. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: e00396c85e0fb0ac966f72fa7216b34c72b94cb7
External sync check: 135/135 PASS / FAIL
External setup: READY / NOT READY
BuildSpatialReason v0.2 test identity: PASS / FAIL
v0.2 test records: 6219 / other
Locked record resolution: 4/4 UNIQUE / other
Native-vector provenance validation: 4/4 PASS / other
GT access purpose: REFERENCE_FORENSICS_ONLY
Candidate replacement: NO
Manual visual inspection: NO
Task7J metric/model/seed/threshold/architecture changed: NO
Detector inference: 4 locked rasters × 1 pass / other
Qwen execution: NONE
SAM2 execution: NONE
Relation-field execution: NONE
D-B1 execution: NONE
Target segmentation: NONE
Detector reproduction right: 6/6 eligible4 MATCH / other
Detector reproduction left: 66/53 eligible42 MATCH / other
Detector reproduction above: 9/9 eligible4 MATCH / other
Detector reproduction below: 7/6 eligible3 MATCH / other
right class: <exact class>
left class: <exact class>
above class: <exact class>
below class: <exact class>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / other
Outcome: <exact enum>
Dominant next blocker: <exact value>
Next gate: <exact enum>
Evidence: evaluation/task8b3_ref01_locked_reference_forensics.json
Report: docs/task8b3_ref01_locked_reference_forensics.md
Next action: Awaiting ChatGPT audit; do not repair selector/eligibility/coverage and do not run full inference.
```

# 23. Commit / push

If COMPLETE:

```text
docs(rc1): record locked demo reference forensics
```

If STOP/FAILED:

```text
docs(rc1): record reference forensics stop
```

Push only:

```text
fix/task8b3-ref01-reference-forensics
```

No force push.
Do not update the old PROP-01 branch.
Do not update main.

# 24. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- task branch rules followed;
- only allowed tracked files changed;
- external 135/135 + setup READY;
- frozen v0.2 test JSONL exact SHA/bytes/count;
- four immutable records resolve uniquely and validate against native-vector provenance;
- canonical GT reference masks resolve without regeneration;
- exactly one frozen detector pass per candidate;
- detector rerun metadata exactly reproduces P1D12;
- no Qwen/SAM2/relation/D-B1/target segmentation;
- no external write;
- no manual visual judgement;
- no candidate replacement;
- all proposal-to-GT measurements use unchanged 0.50 Task 7F threshold;
- exactly one class assigned per candidate;
- outcome/NEXT selected mechanically;
- report/evidence/handoff committed and pushed;
- STOP.
