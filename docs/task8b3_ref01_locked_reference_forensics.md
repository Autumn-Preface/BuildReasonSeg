# Task 8B.3-REF01-F1 — Locked Demo Reference Forensics (STOP: preflight only)

## 1. Task, branch and starting HEAD

```text
base branch    = fix/task8b3-prop01-a2-zero-proposals @ e00396c85e0fb0ac966f72fa7216b34c72b94cb7
task branch    = fix/task8b3-ref01-reference-forensics (created from the base at the same HEAD)
model executions in this task = NONE
```

## 2. Scientific-use disclosure

GT access in this task is restricted to **REFERENCE_FORENSICS_ONLY**: the native-vector annotations are used solely to
measure whether the frozen proposal set covers the canonical GT reference of the four locked candidates. No target GT
metric is computed, no repair is designed, no candidate is replaced, and no visual judgement is made.

## 3. §7 External RC1 integrity preflight

```text
external manifest-listed files equal Git canonical bytes = 135/135 PASS (mismatches: none)
external source_manifest.json identical to canonical control = True
```

## 4. §8/§9 BuildSpatialReason v0.2 TEST identity gate and record resolution

| relation | sample_id | image_id | query_type | level | target | reference_component_ids | native_vector.references[0].tile_instance_id | .source_feature_id |
|---|---|---|---|---|---|---|---|---|
| right | `buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91` | 1010 | largest_to_right_of_to_nearest | 3 | 3 | [4] | 4 | 25833 |
| left | `buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3` | 1003 | largest_to_left_of_to_nearest | 3 | 24 | [26] | 26 | 25456 |
| above | `buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314` | 1008 | largest_to_above_to_nearest | 3 | 6 | [4] | 4 | 25394 |
| below | `buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450` | 1009 | largest_to_below_to_nearest | 3 | 2 | [3] | 3 | 25431 |

All four immutable records resolve uniquely in the frozen v0.2 TEST split and validate against the native-vector
provenance: `dataset_version = v0.2`, `source_component_representation_version = whu-native-vector-v1.0`,
`len(native_vector.references) = 1`, and the native-vector reference `tile_instance_id` equals the record's
`reference_component_ids[0]`.

## 5. What was NOT performed (STOP reason)

The remaining forensic procedure of the task book was **not executed** in this turn:

```text
§10 canonical GT reference mask reconstruction (native-vector per-tile cache + clipped_area_px check) = NOT RUN
§11 existing P1D12 proposal evidence gate                                                              = NOT RUN
§12 the authorised frozen detector pass (for IoU-to-GT diagnosis)                                      = NOT RUN
§13 deterministic reproduction gate before using masks                                                 = NOT RUN
§14 proposal-mask reconstruction and IoU to the GT reference                                           = NOT RUN
§15 production selected reference comparison                                                           = NOT RUN
§16 oracle diagnostic proposals (fixed Task 7F semantics)                                              = NOT RUN
§17 per-candidate exclusive classification and §18 overall outcome                                      = NOT RUN
```

Consequently **no forensic conclusion, classification or outcome enum is asserted** in this report: none of the
classification inputs (GT reference mask, reproduction evidence, IoU measurements) exists yet. Per the task book's
"no claim inflation" rule, no provisional or inferred classification is recorded.

## 6. STOP reason

The execution turn reached its budget after the mandatory branch creation and the read-only preflight gates
(§7 external integrity, §8/§9 record resolution), before the GT-mask reconstruction, the single authorised detector
pass and the IoU-based classification could be performed. Nothing was partially implemented: no detector was run, no
mask was reconstructed, no diagnostics were written, and the repository contains only this report plus the handoff
update. The next turn must restart from §10 with the same branch and HEAD.

## 7. Explicit confirmations

```text
frozen detector passes = 0 (of the 1-per-image allowance)
Qwen / SAM2 / D-B1 / target segmentation = NOT EXECUTED
candidate replacement / repair / visual judgement = NONE / NONE / NONE
external delivery / canonical RC1 modified = NO / NO
locked candidate identities = UNCHANGED
PROP-01 status = PROP01_OPEN_ENGINEERING_DEFECT
```


---

## 8. REF01-F1-R1 — canonical GT reference masks reconstructed (detector still not run)

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = 95f4403837bd7057fc331462503875fa847f43c8
frozen detector passes used in this turn = 0
```

### 8.1 Instance index rows (frozen native-vector truth)

Resolved 4 of 4 required `(tile_id, tile_instance_id)` rows from
`datasets/whu_native_vector/v1.0/instances/index.jsonl`.

| relation | tile | tile_instance_id | GT mask pixels | index clipped_area_px | area match | GT mask bbox [x0,y0,x1,y1] | GT mask centroid [row,col] |
|---|---|---:|---:|---:|---|---|---|
| right | 1010 | 4 | 2478 | 2478 | True | [206, 18, 280, 86] | [59.91525423728814, 231.81033091202582] |
| left | 1003 | 26 | 3512 | 3512 | True | [374, 210, 470, 292] | [248.99373576309796, 425.59367881548974] |
| above | 1008 | 4 | 5013 | 5013 | True | [27, 243, 93, 398] | [317.2363853979653, 58.7245162577299] |
| below | 1009 | 3 | 2606 | 2606 | True | [187, 222, 232, 320] | [269.7674597083653, 210.03875671527246] |

### 8.2 Geometry cache structure (recorded verbatim)

| relation | cache keys | label key | dtype | shape |
|---|---|---|---|---|
| right | `['bboxes', 'centroids', 'clipped_area', 'full_area', 'instance_ids', 'label_map', 'multipart', 'n_holes', 'points', 'ring_instance_ids', 'ring_kinds', 'ring_lengths', 'ring_offsets', 'source_feature_ids', 'tiny', 'touches_border', 'visible_fraction']` | `label_map` | uint8 | [512, 512] |
| left | `['bboxes', 'centroids', 'clipped_area', 'full_area', 'instance_ids', 'label_map', 'multipart', 'n_holes', 'points', 'ring_instance_ids', 'ring_kinds', 'ring_lengths', 'ring_offsets', 'source_feature_ids', 'tiny', 'touches_border', 'visible_fraction']` | `label_map` | uint8 | [512, 512] |
| above | `['bboxes', 'centroids', 'clipped_area', 'full_area', 'instance_ids', 'label_map', 'multipart', 'n_holes', 'points', 'ring_instance_ids', 'ring_kinds', 'ring_lengths', 'ring_offsets', 'source_feature_ids', 'tiny', 'touches_border', 'visible_fraction']` | `label_map` | uint8 | [512, 512] |
| below | `['bboxes', 'centroids', 'clipped_area', 'full_area', 'instance_ids', 'label_map', 'multipart', 'n_holes', 'points', 'ring_instance_ids', 'ring_kinds', 'ring_lengths', 'ring_offsets', 'source_feature_ids', 'tiny', 'touches_border', 'visible_fraction']` | `label_map` | uint8 | [512, 512] |

```text
GT reference masks non-empty AND area == clipped_area_px for all four candidates = True
```

The canonical GT reference mask of every locked candidate is therefore reconstructed deterministically from the frozen
native-vector geometry cache, with no regeneration and no detector involvement.

### 8.3 Still not executed (unchanged STOP boundary)

```text
§11 existing P1D12 proposal evidence gate                                  = NOT RUN
§12 authorised frozen detector pass (IoU-to-GT diagnosis)                  = NOT RUN
§13 deterministic reproduction gate                                        = NOT RUN
§14 proposal-mask reconstruction and IoU to the GT reference               = NOT RUN
§15 production selected reference comparison                               = NOT RUN
§16 oracle diagnostic proposals (fixed Task 7F semantics)                   = NOT RUN
§17/§18 per-candidate classification and overall outcome                    = NOT RUN
```

No classification or outcome enum is asserted. The block requiring an explicit decision before the detector pass is
that the frozen proposal-mask representation used for IoU (i.e. how a proposal's mask is reconstructed from
`proposals.json`) and the fixed Task 7F coverage threshold are not yet pinned down in my evidence; running the
authorised detector pass without them would produce diagnostics that cannot be interpreted.


---

## 10. REF01-F1-R2 — execution attempt STOP

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = 141af9cfd943dd13366947ea3a31637a3f8c7bc9
proposal mask source (as instructed) = RERUN_GLOBALPROPOSAL_MASK_CROP
coverage threshold                   = 0.50 (frozen Task 7F)
forensic script                      = scripts/task8b3_ref01_locked_reference_forensics.py (written)
evidence json produced                = False
```

The forensic script was written exactly as instructed and executed. It failed, and my runner then crashed while
printing the child's stderr (GBK console encoding), so the failure text was not captured on the first attempt. The
child was therefore re-executed once with a UTF-8 process environment to capture the real diagnostic:

```text
Traceback (most recent call last):
  File "C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\scripts\task8b3_ref01_locked_reference_forensics.py", line 117, in <module>
    sys.exit(main())
             ^^^^^^
  File "C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\scripts\task8b3_ref01_locked_reference_forensics.py", line 53, in main
    raster = np.asarray(Image.open(ROOT / f"{tile}.tif").convert("RGB"))
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-mvp\Lib\site-packages\PIL\Image.py", line 3639, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\D\\resources\\Satellite dataset \\Ⅱ (East Asia)\\1. The cropped image data and raster labels\\test\\image\\1010.tif'
```

State recorded honestly:

* no report table, class assignment or outcome enum is asserted in this report;
* `evaluation\task8b3_ref01_locked_reference_forensics.json` was NOT produced;
* detector passes actually consumed: NOT DETERMINISTICALLY ESTABLISHED — the child failed after constructing the runtime and before writing evidence, so between 0 and the full 4-pass allowance may have executed; the next task must treat the allowance as exhausted and re-run all four passes in a single invocation;
* Qwen / SAM2 / relation-field / D-B1 / target segmentation: NOT EXECUTED;
* manual visual inspection / candidate replacement / repair: NO / NO / NO;
* no delivery file was modified; no PNG or mask artifact was written by this task.
