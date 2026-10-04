# TO_DSH — Task 8B.3-P1D10-R3: Recover Canonical Raster Paths and Finalize Locked Demo Pool

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `cbde8755f1cae0593274746f128760e13ebe86a1`

# 0. Purpose

P1D10-R2 correctly STOPPED after discovering that the four metadata-locked BuildSpatialReason v0.2 TEST candidates
had not yet been proven to resolve to their original RGB raster tiles.

The four provisional metadata locks are:

```text
right:
buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91

left:
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3

above:
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314

below:
buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
```

Their `image_id`s are:

```text
1010
1003
1008
1009
```

R2 proved their component maps exist, but the v0.2 record's `image_path` field is not a directly usable filesystem
path.

The repository already defines the canonical WHU source-root contract:

```text
buildreasonseg_mvp/whu_vector_audit.py:
ORIGINAL_ROOT = C:\D\resources\Satellite dataset Ⅱ (East Asia)

buildreasonseg_mvp/whu_native_vector.py:
TileRecord.source_image_ref is a logical path relative to ORIGINAL_ROOT/source_root
datasets/whu_native_vector/v1.0/tiles/index.jsonl is the canonical tile index
```

This task closes ONLY that raster-resolution gap.

No model may run.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = cbde8755f1cae0593274746f128760e13ebe86a1
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any detector/model/inference;
- run `predict.py`, `--inspect-proposals`, Qwen, SAM2, D-B1;
- run pytest/check_setup;
- train/fine-tune/export/download;
- inspect detector outputs or historical Demo outputs;
- inspect/manual-rank image visual quality;
- change any locked candidate ID;
- substitute another candidate if a raster is missing;
- copy any candidate into RC1;
- modify runtime/tests/config/manifests/checkpoints;
- enter REF-01/MASK-01/Task 8B.4/8C;
- modify main;
- force push.

Read-only file lookup, hashing and image decoding are allowed.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d10_prop01_resolution_decision.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 4. Correct the handoff typo first

The final R2 `FROM_DSH` contains a typo in the LEFT locked candidate:

incorrect:
```text
buildsr_test_1003_3_largest_to_left_of_nearest_f3fcb14e14c3
```

canonical locked ID from the R2 report:
```text
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
```

Use the canonical ID with `_left_of_to_nearest_`.

Do not alter any of the four locked records otherwise.

# 5. Canonical source-root evidence

Read, do not modify:

```text
buildreasonseg_mvp/whu_vector_audit.py
buildreasonseg_mvp/whu_native_vector.py
```

Record exact evidence for:

```text
ORIGINAL_ROOT
CROPPED_ROOT
source_root_default()
TileRecord.source_image_ref semantics
SCENE_DISJOINT_SPLITS
```

Required expected root:

```text
C:\D\resources\Satellite dataset Ⅱ (East Asia)
```

If the code on this exact branch disagrees with this root/contract:
- STOP.

# 6. Canonical tile-index gate

Require local file:

```text
datasets/whu_native_vector/v1.0/tiles/index.jsonl
```

Record:

```text
exists
bytes
SHA256
line count
JSON parse failures
```

Do NOT regenerate it.

For each locked `image_id`:

```text
1010
1003
1008
1009
```

find exactly one canonical tile-index row where:

```text
tile_id == image_id
```

Record:

```text
tile_id
grid_id
legacy_category
source_raster
grid_row
grid_column
source_image_ref
source_label_ref
width
height
scene_disjoint_split
legacy_compat_split
```

Require:

```text
scene_disjoint_split == test
legacy_category == test
width == 512
height == 512
```

Any zero/multiple row or mismatch:
- do not change candidate;
- STOP.

# 7. Resolve the original raster paths

For each row define ONLY:

```text
resolved_source_image =
Path(r"C:\D\resources\Satellite dataset Ⅱ (East Asia)") / source_image_ref
```

No repository-relative interpretation is allowed.

Expected logical shape:

```text
1. The cropped image data and raster labels/test/image/<image_id>.tif
```

Record exact resolved absolute paths.

For every locked raster require:

```text
exists = YES
is_file = YES
```

If one fails:
- no fallback path search;
- no candidate substitution;
- STOP.

# 8. Decode and identity-check the original raster

Read-only decode each resolved TIFF.

Record:

```text
bytes
SHA256
format
mode
width
height
dtype if directly available
bands/channel count if directly available
```

Require:

```text
width = 512
height = 512
```

Do not reject based on brightness, contrast, content or appearance.

Do not visually rank or manually inspect semantic quality.

# 9. Cross-check against canonical reasoning metadata

For each of the four provisional sample records in:

```text
datasets/build_spatial_reason/v0.2/test.jsonl
```

re-confirm:

```text
sample_id
image_id
query_type
target_component_id
reference_component_ids
native_vector.target.tile_instance_id
native_vector.references[*].tile_instance_id
split == test
```

Require exact agreement with the R2 lock.

Also resolve its `image_metadata_ref` row and confirm that it refers to the same `image_id` / tile identity.

Do NOT use model output fields.

# 10. Optional non-authoritative mirror check

Only if this exact path already exists:

```text
artifacts/task6m_yolo_native/images/test/<image_id>.tif
```

or an equivalent documented frozen mirror path,

record:

```text
exists
SHA256
byte-identical to canonical ORIGINAL_ROOT raster? YES / NO
```

This check is diagnostic only.

The canonical source identity remains:

```text
ORIGINAL_ROOT / source_image_ref
```

Do not STOP merely because an optional mirror is absent.

# 11. Raster-resolution outcome

Choose exactly ONE.

## A — `PROP01_LOCKED_DEMO_RASTERS_RESOLVED`

Require all four:
- unique tile-index row;
- `scene_disjoint_split=test`;
- canonical source raster exists;
- 512×512 decode succeeds;
- v0.2 sample identity matches the R2 lock.

Then:

```text
replacement selection policy = READY
locked candidate status = FINAL_METADATA_LOCK
Demo policy feasibility = DEMO_POLICY_PATH_READY
primary resolution = PROP01_RESOLUTION_DEMO_POLICY
PROP-01 status = PROP01_OPEN_ENGINEERING_DEFECT
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION
```

No candidate is run yet.

## B — `PROP01_LOCKED_DEMO_RASTER_RESOLUTION_FAILED`

If any required raster/index/identity check fails.

Then:

```text
replacement selection policy = NOT READY
Demo policy feasibility = DEMO_POLICY_PATH_NOT_READY
primary resolution = PROP01_RESOLUTION_BLOCKED
PROP-01 status = PROP01_OPEN_ENGINEERING_DEFECT
NEXT = PROP01_DECISION_EVIDENCE_RECOVERY
```

Do NOT substitute candidates.

# 12. Candidate immutability

If outcome A, freeze exactly these IDs:

```text
right = buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91
left  = buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
above = buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314
below = buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
```

Future runtime failure does NOT authorize replacing one.

Any replacement requires a new ChatGPT decision that explicitly records the failed locked candidate.

# 13. Support-scope wording

Retain evidence-bounded scope:

```text
WHU Building Dataset — Satellite dataset II (East Asia)
native-vector building instances
scene_disjoint_v1
tile-relative spatial reasoning
```

Retain limitation:

```text
scene separation does not establish cross-city or broad geographic generalization
```

Retain:

```text
A2 provenance/domain = NOT ESTABLISHED
A2 = documented persistent non-detection/stress case
```

Do not call A2 out-of-domain.

# 14. Report update

Append `P1D10-R3` to:

```text
docs/task8b3_p1d10_prop01_resolution_decision.md
```

Required:

1. R2 STOP reason
2. canonical source-root evidence
3. tile-index identity
4. four tile-index rows
5. four canonical resolved raster absolute paths
6. four raster bytes/SHA256/format/mode/dimensions
7. v0.2 record cross-check
8. optional mirror check if available
9. exact outcome
10. final/provisional candidate-lock status
11. replacement policy readiness
12. Demo-policy feasibility
13. primary resolution
14. PROP-01 status
15. exact next gate
16. support scope / limitations
17. no model/test execution
18. no functional modification.

Mark R2 §20.8 raster gap as:
- RESOLVED if outcome A;
- STILL OPEN if outcome B.

# 15. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D10-R3
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: cbde8755f1cae0593274746f128760e13ebe86a1
Model/test execution: NONE
Functional files modified: NO
Canonical source root: C:\D\resources\Satellite dataset Ⅱ (East Asia)
Tile index: <path>
Tile index identity: MATCH / MISMATCH
Right raster: <absolute path / NONE> | <sha / NONE>
Left raster: <absolute path / NONE> | <sha / NONE>
Above raster: <absolute path / NONE> | <sha / NONE>
Below raster: <absolute path / NONE> | <sha / NONE>
All four raster dimensions: 512x512 / other
Locked candidate count: 4
Locked right candidate: buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91
Locked left candidate: buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
Locked above candidate: buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314
Locked below candidate: buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
Locked candidate status: FINAL_METADATA_LOCK / PROVISIONAL_ONLY
Outcome: <exact enum>
Replacement selection policy: READY / NOT READY
Detector adaptation feasibility: DETECTOR_ADAPTATION_NOT_READY
Demo policy feasibility: <exact enum>
Primary resolution: <exact enum>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
A2 domain classification: NOT ESTABLISHED
Scientific freeze preserved: YES
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d10_prop01_resolution_decision.md
Next action: Awaiting ChatGPT audit; do not run or copy locked candidates.
```

# 16. Commit / push

If outcome A / COMPLETE:

```text
docs(rc1): resolve locked prop01 demo rasters
```

If STOP/FAILED:

```text
docs(rc1): record locked raster resolution stop
```

Push normally.
No force push.

# 17. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model/test execution;
- no functional modification;
- handoff left-candidate typo corrected;
- canonical source-root contract proven from repository code;
- tile-index row uniquely resolves each of the four image IDs;
- all four source rasters exist and decode as 512×512;
- all four SHA256 values recorded;
- v0.2 sample identities still match the R2 locks;
- no candidate is substituted or selected using runtime/manual output;
- exact outcome and next gate recorded;
- report/FROM_DSH committed/pushed;
- STOP.
