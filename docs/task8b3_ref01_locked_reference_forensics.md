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


---

## 12. REF01-F1-R3 — harness repair attempt STOP

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = ac1e4a9f8f452581c509ab0d73024c44c5de1be0
STATIC_HARNESS_GATE = PASS (data path root verified to exist, external RC1 detector module located,
                      DetectorRuntime construction present, reproduction gate present)
harness runs consumed = 1
detector calls executed = 0
evidence json produced   = False
```

The repaired harness passed the static gate but failed at **module import time** on a Python operator-precedence
mistake in the raster-root construction:

```text
RASTER_ROOT = (Path(r"C:\D\resources") / "Satellite dataset " + chr(0x2161) + " (East Asia)" ...
TypeError: unsupported operand type(s) for +: 'WindowsPath' and 'str'
```

The correct form parenthesises the concatenation before the `/` operator:

```text
RASTER_ROOT = (Path(r"C:\D\resources") / ("Satellite dataset " + chr(0x2161) + " (East Asia)") ...
```

Because the failure occurred before `main()` ran, **no detector call was made** and no measurement, class or outcome was
produced. The task book allows the harness exactly one execution, and that execution has now been consumed by this
import-time failure, so this task stops without re-running and without asserting any classification.

The single blocking defect for the next attempt is the one-line parenthesisation above; everything else in the harness
(static gate, external RC1 import, default `DetectorRuntime`, reproduction gate, GT mask reconstruction, IoU
classification and outcome priority) was already verified by the static gate and remains in place.

```text
Qwen / SAM2 / relation fields / D-B1 / target segmentation = NOT EXECUTED
manual visual inspection / candidate replacement / repair  = NO / NO / NO
external delivery / canonical RC1 modified                 = NO / NO
classification / outcome enum                              = NOT ASSERTED
```


---

## 13. REF01-F1-R4 — triple preflight, single harness run, GT IoU classification

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = 35105fb8b4255375923ff3174d3efd2356717333
PREFLIGHT 1/3 compile      = PASS (py_compile)
PREFLIGHT 2/3 import-only  = PASS (module imported without executing main(); detector calls 0)
PREFLIGHT 3/3 contract     = PASS (external RC1 detector module imported; DetectorRuntime present;
                             eligible_proposals/select_reference present; detector calls 0)
prior detector calls       = 0
runtime construction       = external RC1 default DetectorRuntime()
harness runs               = 1
detector calls             = 4 (exactly one per locked candidate)
proposal mask source       = RERUN_GLOBALPROPOSAL_MASK_CROP
proposals.json role        = METADATA_REPRODUCTION_ONLY
coverage threshold         = 0.50 (frozen Task 7F)
```

### 13.1 Reproduction table (harness rerun vs stored P1D12 metadata)

| relation | expected raw/merged | stored raw/merged | stored eligible | rerun raw/merged | rerun eligible | gate |
|---|---|---|---|---|---|---|
| right | 6/6 | 6/6 | 4 | 6/6 | 4 | MATCH |
| left | 66/53 | 66/53 | 42 | 66/53 | 42 | MATCH |
| above | 9/9 | 9/9 | 4 | 9/9 | 4 | MATCH |
| below | 7/6 | 7/6 | 3 | 7/6 | 3 | MATCH |

All four candidates reproduce the P1D12 proposal metadata exactly (raw, merged and eligible counts).

### 13.2 GT comparison and mechanical classification

| relation | GT_ref | selected_id | selected_IoU | best_eligible_IoU | best_any_IoU | class |
|---|---|---|---|---|---|---|
| right | 4 | 1 | 0.5589 | 0.5589 | 0.5589 | REFERENCE_SELECTED_CORRECT |
| left | 26 | 14 | 0.0000 | 0.6501 | 0.6501 | REFERENCE_SELECTION_WRONG_COVERED |
| above | 4 | 4 | 0.0000 | 0.0000 | 0.9033 | REFERENCE_ELIGIBILITY_BLOCKED |
| below | 3 | 1 | 0.0000 | 0.6165 | 0.6165 | REFERENCE_SELECTION_WRONG_COVERED |

Class counts: `{"REFERENCE_SELECTED_CORRECT": 1, "REFERENCE_SELECTION_WRONG_COVERED": 2, "REFERENCE_ELIGIBILITY_BLOCKED": 1}`

### 13.3 Outcome

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

### 13.4 Explicit non-execution

```text
Qwen / SAM2 / relation fields / D-B1 / target segmentation = NOT EXECUTED
manual visual inspection / candidate replacement / product repair = NO / NO / NO
external delivery / canonical RC1 modified = NO / NO
```


---

## 14. REF01-F1-R5 — deterministic verification run

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = d1c7f6374f325fec71bce6a350609b2d23865c0a
detector calls = 4 (one additional pass per locked candidate)
runtime        = external RC1 default DetectorRuntime()
proposal mask  = RERUN_GLOBALPROPOSAL_MASK_CROP
threshold      = 0.50 (frozen Task 7F) · tolerance = 1e-06
DETERMINISTIC_VERIFICATION = PASS
```

### 14.1 Raster SHA identity and P1D12 metadata reproduction

| relation | tile | raster SHA256 (prefix) | lock | rerun raw/merged | eligible | per-proposal metadata |
|---|---|---|---|---|---|---|
| right | 1010 | 1688306c5edbffe4... | MATCH | 6/6 | 4 | True |
| left | 1003 | eea4edd0db9e079e... | MATCH | 66/53 | 42 | True |
| above | 1008 | 0efe8bc2e1d1f3f5... | MATCH | 9/9 | 4 | True |
| below | 1009 | c22134e671f2d0b7... | MATCH | 7/6 | 3 | True |

Per-proposal reproduction compared `proposal_id`, `mask_area`, `global_bbox`, `touches_image_border` and
`bbox_extent_ratio` for every merged proposal against the stored P1D12 `proposals.json` items.

### 14.2 Frozen constants and external identity

| constant | value |
|---|---|
| `TILE_SIZE` | `512` |
| `TILE_OVERLAP` | `128` |
| `STRIDE_EXPECTED` | `384` |
| `IMGSZ` | `640` |
| `CONF` | `0.05` |
| `MAX_DET` | `300` |
| `DUPLICATE_IOU` | `0.5` |
| `MERGE_BBOX_EXTENT_RATIO_MAX` | `0.2` |
| `FROZEN_THRESHOLD` | `0.5` |
| `device` | `cuda` |
| `checkpoint` | `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\model\buildreasonseg_advisor\detector.pt` |

```text
external detector module   = C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
external/canonical detector identical = False
imageio = IMPORT_FAILED: No module named 'imageio' (NOT AVAILABLE)
pillow  = 12.3.0 · numpy = 2.4.6
```

### 14.3 IoU determinism versus R4 (tolerance 1e-6)

| relation | selected_id | R5 selected_IoU | R4 selected_IoU | R5 best_eligible | R4 best_eligible | R5 best_any | R4 best_any | within 1e-6 |
|---|---|---|---|---|---|---|---|---|
| right | 1 | 0.558870 | 0.558870 | 0.558870 | 0.558870 | 0.558870 | 0.558870 | PASS |
| left | 14 | 0.000000 | 0.000000 | 0.650067 | 0.650067 | 0.650067 | 0.650067 | PASS |
| above | 4 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.903250 | 0.903250 | PASS |
| below | 1 | 0.000000 | 0.000000 | 0.616550 | 0.616550 | 0.616550 | 0.616550 | PASS |

### 14.4 Outcome

```text
Outcome (unchanged) = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Evidence: `evaluation\task8b3_ref01_locked_reference_forensics_r5.json` (R4 evidence preserved at `evaluation\task8b3_ref01_locked_reference_forensics.json`)

### 14.5 Explicit non-execution

```text
Qwen / SAM2 / relation fields / D-B1 / target segmentation = NOT EXECUTED
manual visual inspection / candidate replacement / product repair = NO / NO / NO
external delivery / canonical RC1 modified = NO / NO
```


---

## 15. REF01-F1-R6 — authoritative closure

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = 53de75ae4f5246b4e18685b78bd5a9f913447984
detector calls = 4 (final pass, one per locked candidate)
runtime        = external RC1 default DetectorRuntime()
manifest basis = GIT_CANONICAL_BLOB_BYTES · entries = 135
external detector == Git canonical blob == manifest identity = True
external control manifest == Git canonical control   = True
imageio        = IMPORT_FAILED: No module named 'imageio' (NOT AVAILABLE)
pillow/numpy   = 12.3.0 / 2.4.6
AUTHORITATIVE_CLOSURE = PASS
```

### 15.1 Raster identity and proposal reproduction

| relation | tile | raster SHA256 | lock | raw/merged | eligible | 10-field reproduction |
|---|---|---|---|---|---|---|
| right | 1010 | 1688306c5edbffe4... | MATCH | 6/6 | 4 | True |
| left | 1003 | eea4edd0db9e079e... | MATCH | 66/53 | 42 | True |
| above | 1008 | 0efe8bc2e1d1f3f5... | MATCH | 9/9 | 4 | True |
| below | 1009 | c22134e671f2d0b7... | MATCH | 7/6 | 3 | True |

Per-proposal field agreement (matched/total) over the ten frozen fields
`proposal_id`, `source_tile_id`, `confidence`, `mask_area`, `global_bbox`, `centroid`, `touches_image_border`, `border_clearance`, `bbox_extent_ratio`, `raw_index`:

| relation | proposal_id | source_tile_id | confidence | mask_area | global_bbox | centroid | touches_image_border | border_clearance | bbox_extent_ratio | raw_index |
|---|---|---|---|---|---|---|---|---|---|---|
| right | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| left | 53/53 | 53/53 | 53/53 | 53/53 | 53/53 | 53/53 | 53/53 | 53/53 | 53/53 | 53/53 |
| above | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 | 9/9 |
| below | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |

### 15.2 IoU determinism versus R4 and R5 (tolerance 1e-6)

| relation | R6 selected | R4 selected | R5 selected | R6 best_eligible | R4 best_eligible | R6 best_any | R4 best_any | within 1e-6 |
|---|---|---|---|---|---|---|---|---|
| right | 0.558870 | 0.558870 | 0.558870 | 0.558870 | 0.558870 | 0.558870 | 0.558870 | PASS |
| left | 0.000000 | 0.000000 | 0.000000 | 0.650067 | 0.650067 | 0.650067 | 0.650067 | PASS |
| above | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.903250 | 0.903250 | PASS |
| below | 0.000000 | 0.000000 | 0.000000 | 0.616550 | 0.616550 | 0.616550 | 0.616550 | PASS |

### 15.3 Outcome

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Evidence: `evaluation\task8b3_ref01_locked_reference_forensics_r6.json` (R4/R5 evidence preserved)

### 15.4 Explicit non-execution

```text
Qwen / SAM2 / relation fields / D-B1 / target segmentation = NOT EXECUTED
manual visual inspection / candidate replacement / product repair = NO / NO / NO
external delivery / canonical RC1 modified = NO / NO
```


---

## 16. REF01-F1-R7 — zero-call authoritative replay and canonical evidence consolidation

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = 12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c
detector / model calls in R7 = 0 (no DetectorRuntime instantiation, no predict)
sources replayed = R4 canonical records, R5 verification, R6 closure, P1D12 diagnostics metadata
```

### 16.1 Git-canonical identity of `buildreasonseg.runtime.detector`

```text
manifest basis / entries      = GIT_CANONICAL_BLOB_BYTES / 135
manifest bytes / sha256       = 20300 / 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
Git blob bytes / sha256       = 20300 / 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
external bytes / sha256       = 20300 / 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
all three identities agree    = True
```

### 16.2 Git-canonical identity of imageio

```text
manifest declares an imageio entry        = False
requirements.txt imageio lines            = NONE
requirements.txt Git-canonical sha256     = adfdd3b481a0fd213fea4dde32df0d9601552d044c3139abd7e614c6b5fabce4
requirements.txt Git-blob sha256          = adfdd3b481a0fd213fea4dde32df0d9601552d044c3139abd7e614c6b5fabce4
requirements.txt external sha256          = adfdd3b481a0fd213fea4dde32df0d9601552d044c3139abd7e614c6b5fabce4
installed modules                         = {"imageio": false, "imageio_ffmpeg": false, "PIL": true, "cv2": true}
conclusion                                = imageio is NOT declared as a runtime requirement and NOT importable in the delivery environment; image I/O is performed by Pillow/ultralytics
```

The delivery's dependency declaration is verified by its Git-canonical identity, and the runtime fact is recorded:
`imageio` is neither declared nor importable, so image I/O runs through Pillow/ultralytics. No imageio identity is
invented and no substitution is performed.

### 16.3 Read-only replay of the classification

| relation | stored class (R4) | recomputed class | reproduced | R4 IoU (selected/best_eligible/best_any) | R4=R5=R6 within 1e-6 |
|---|---|---|---|---|---|
| right | REFERENCE_SELECTED_CORRECT | REFERENCE_SELECTED_CORRECT | True | 0.558870 / 0.558870 / 0.558870 | True |
| left | REFERENCE_SELECTION_WRONG_COVERED | REFERENCE_SELECTION_WRONG_COVERED | True | 0.000000 / 0.650067 / 0.650067 | True |
| above | REFERENCE_ELIGIBILITY_BLOCKED | REFERENCE_ELIGIBILITY_BLOCKED | True | 0.000000 / 0.000000 / 0.903250 | True |
| below | REFERENCE_SELECTION_WRONG_COVERED | REFERENCE_SELECTION_WRONG_COVERED | True | 0.000000 / 0.616550 / 0.616550 | True |

Class counts: `{"REFERENCE_SELECTED_CORRECT": 1, "REFERENCE_SELECTION_WRONG_COVERED": 2, "REFERENCE_ELIGIBILITY_BLOCKED": 1}` · Outcome `REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE` · blocker `ELIGIBILITY` · `NEXT = REF01_ELIGIBILITY_FORENSICS`

### 16.4 Evidence consolidation

```text
canonical evidence overwritten = evaluation\task8b3_ref01_locked_reference_forensics.json (now the consolidated authoritative record)
temporary evidence removed     = ['task8b3_ref01_locked_reference_forensics_r5.json', 'task8b3_ref01_locked_reference_forensics_r6.json']
```

### 16.5 Explicit non-execution

```text
detector / model / Qwen / SAM2 / relation fields / D-B1 / target segmentation = NONE
manual visual inspection / candidate replacement / product repair = NO / NO / NO
external delivery / canonical RC1 modified = NO / NO
```


---

## 17. REF01-F1-R8 — manifest identity for detector.py / imageio.py, read-only verifier, ID completion

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = 65643f802a0a187b93160155f976689c0b50b8c6
detector / model calls in R8 = 0
tracked script rewritten as = pure read-only replay verifier (no DetectorRuntime, no detect_global, no predict)
```

### 17.1 Manifest / Git / external identity of the two runtime modules

| module | in manifest | manifest sha256 | Git blob sha256 | external sha256 | all agree |
|---|---|---|---|---|---|
| detector | True | 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738 | 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738 | 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738 | True |
| imageio | True | b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878 | b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878 | b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878 | True |

`buildreasonseg/runtime/detector.py` is a manifest entry whose manifest, Git-canonical and external identities are
identical. `buildreasonseg/runtime/imageio.py` is **a manifest entry**;
its Git blob presence = True and external file presence =
True. Whatever the state is, it is recorded verbatim rather than assumed, and no
imageio identity is claimed that the manifest does not support.

### 17.2 Completed proposal IDs (read-only replay)

| relation | selected_id | best_eligible_id | best_any_id | classification |
|---|---|---|---|---|
| right | 1 | 1 | 1 | REFERENCE_SELECTED_CORRECT |
| left | 14 | 30 | 30 | REFERENCE_SELECTION_WRONG_COVERED |
| above | 4 | 2 | 5 | REFERENCE_ELIGIBILITY_BLOCKED |
| below | 1 | 2 | 2 | REFERENCE_SELECTION_WRONG_COVERED |

The IDs were derived by replaying the stored per-proposal `iou_to_gt` values under the frozen eligibility rule
(`mask_area > 0` and not touching the image border and `bbox_extent_ratio <= 0.20`) and the frozen Task 7F coverage
threshold 0.50, with **no detector call**.

### 17.3 Verifier result

```text
REPLAY_VERIFIER: PASS
```

### 17.4 Explicit non-execution

```text
detector / model / Qwen / SAM2 / relation fields / D-B1 / target segmentation = NONE
manual visual inspection / candidate replacement / product repair = NO / NO / NO
external delivery / canonical RC1 modified = NO / NO
```


---

## 18. REF01-F1-R9 — live-metadata replay and final schema closure

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = f268d03a9a70b494b7134c6b2f2647ed3468caa3
detector / model calls in R9 = 0
verification_mode = READ_ONLY_HISTORICAL_EVIDENCE_REPLAY
tie-break = (-IoU, -confidence, proposal_id) over live P1D12 proposal metadata
canonical evidence schema = REPLACED (not appended)
obsolete imageio false-claim structures = REMOVED
```

### 18.1 Exact scientific reuse disclosure (mandatory)

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

中文对照：

> 这些定性 Demo 候选是在 Task 7J 最终冻结架构测试指标**已被消耗之后**，从冻结的 BuildSpatialReason v0.2 测试划分中**确定性**选取的。其定性复用**不会**改变、替换或重新选择任何已报告的 Task 7J 指标、模型、阈值、随机种子或架构。

### 18.2 Module identity (manifest / Git-canonical / external)

| module | manifest bytes | manifest sha256 | external bytes | external sha256 | manifest_match |
|---|---:|---|---:|---|---|
| `buildreasonseg/runtime/detector.py` | 20300 | `82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738` | 20300 | `82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738` | True |
| `buildreasonseg/runtime/imageio.py` | 7978 | `b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878` | 7978 | `b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878` | True |

`buildreasonseg.runtime.imageio` is the delivery's own module (imported for identity only, never executed); there is no
third-party `imageio` package check and no contradictory imageio structure remains in the evidence.

### 18.3 Replayed selection with the full frozen tie-break

| relation | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_id | best_any_IoU | gap | classification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| right | 1 | 0.558870 | 1 | 0.558870 | 1 | 0.558870 | 0.000000 | REFERENCE_SELECTED_CORRECT |
| left | 14 | 0.000000 | 30 | 0.650067 | 30 | 0.650067 | 0.650067 | REFERENCE_SELECTION_WRONG_COVERED |
| above | 4 | 0.000000 | 4 | 0.000000 | 5 | 0.903250 | 0.000000 | REFERENCE_ELIGIBILITY_BLOCKED |
| below | 1 | 0.000000 | 2 | 0.616550 | 2 | 0.616550 | 0.616550 | REFERENCE_SELECTION_WRONG_COVERED |

Class counts: `{"REFERENCE_SELECTED_CORRECT": 1, "REFERENCE_SELECTION_WRONG_COVERED": 2, "REFERENCE_ELIGIBILITY_BLOCKED": 1}` · Outcome `REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE` · blocker `ELIGIBILITY` · `NEXT = REF01_ELIGIBILITY_FORENSICS`

### 18.4 Verifier result

```text
REPLAY_VERIFIER: PASS
```

### 18.5 Explicit non-execution

```text
detector / model / Qwen / SAM2 / relation fields / D-B1 / target segmentation = NONE
manual visual inspection / candidate replacement / product repair = NO / NO / NO
external delivery / canonical RC1 modified = NO / NO
```


---

## 19. REF01-F1-R10 — standalone tracked verifier

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = 367144990e510aaacbae2b545a08d99aff91776a
detector / model calls = 0
canonical evidence modified = NO (sha256 unchanged: f7495796577cb26a...)
tracked verifier = scripts/task8b3_ref01_locked_reference_forensics.py (standalone, read-only)
```

The verifier now independently checks four things and exits non-zero on any mismatch:

1. **Git-canonical module identity** — `buildreasonseg/runtime/detector.py` and `buildreasonseg/runtime/imageio.py`
   must satisfy manifest entry == Git blob == external file, and the evidence record must agree; a standalone
   third-party `import imageio` line must not exist in `predict.py`;
2. **historical R6 evidence** — the ten-field reproduction flags and the R4=R5=R6 IoU consistency (tolerance 1e-6) must
   be asserted in the evidence, together with the exact Task 7J disclosure;
3. **live P1D12 metadata** — proposal metadata is re-read from the external diagnostics directories;
4. **mechanical replay** — `selected` / `bestEligible` / `bestAny` are recomputed with the frozen tie-break
   `(-IoU, -confidence, proposal_id)` and the classification is re-derived and compared with the evidence.

Verifier output:

```text
== 1. Git-canonical module identity ==
  buildreasonseg/runtime/detector.py: manifest==git==external=True evidence_record_consistent=True
  buildreasonseg/runtime/imageio.py: manifest==git==external=True evidence_record_consistent=True
  standalone third-party 'import imageio' lines = 0 �� delivery module referenced = False
== 2. schema, disclosure and historical R6 evidence ==
  required top-level keys present = True (missing [])
  disclosure verbatim present = True �� chinese translation = True
  R6 ten-field reproduction = True �� R4=R5=R6 within 1e-06 = True
== 3. live P1D12 metadata + 4. mechanical replay ==
  right: live_proposals=6 eligible=4 ids_match=True iou_match=True class_match=True (REFERENCE_SELECTED_CORRECT)
  left: live_proposals=53 eligible=42 ids_match=True iou_match=True class_match=True (REFERENCE_SELECTION_WRONG_COVERED)
  above: live_proposals=9 eligible=4 ids_match=True iou_match=True class_match=True (REFERENCE_ELIGIBILITY_BLOCKED)
  below: live_proposals=6 eligible=3 ids_match=True iou_match=True class_match=True (REFERENCE_SELECTION_WRONG_COVERED)
  class counts recomputed = {"REFERENCE_SELECTED_CORRECT": 1, "REFERENCE_SELECTION_WRONG_COVERED": 2, "REFERENCE_ELIGIBILITY_BLOCKED": 1} �� matches evidence = True
STANDALONE_VERIFIER: PASS
detector_or_model_calls = 0 �� canonical evidence untouched = true
```

### 19.1 Explicit non-execution

```text
detector / model / Qwen / SAM2 / relation fields / D-B1 / target segmentation = NONE
canonical evidence / external delivery / canonical RC1 modified = NO / NO / NO
manual visual inspection / candidate replacement / product repair = NO / NO / NO
```


---

## 20. REF01-F1-R11 — independent verifier: module files, R6 history, live counts, role facts

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD   = 7d9492fa7628c4b4cc379a3c4ba976c3bd1fe228
detector / model calls = 0
canonical evidence modified = NO (sha256 unchanged: f7495796577cb26a...)
```

The standalone verifier now additionally checks:

1. **module `__file__` identity** — `buildreasonseg.runtime.detector` and `buildreasonseg.runtime.imageio` are imported
   from the external delivery; each module's `__file__` must resolve to the external file and that file's SHA256 must
   equal the Git-canonical manifest value;
2. **historical R6 ten-field evidence recovered with `git show`** — the R6 evidence file no longer exists in the working
   tree, so it is recovered from Git history and its ten-field flags must all be true;
3. **live P1D12 counts** — raw from `result.json`, merged/eligible from `proposals.json`, compared against the frozen
   6/6/4, 66/53/42, 9/9/4 and 7/6/3 expectations;
4. **production `selected` replay** — the production selection must be identical across the historically recovered
   R4/R5/R6 evidence and equal to the recorded value, anchored to the Git-canonical detector source;
5. **`bestEligible` / `bestAny` replay** with the frozen tie-break `(-IoU, -confidence, proposal_id)`;
6. **12/12 role facts** — `(proposal_id, IoU)` for the three roles across the four candidates compared within 1e-6.

Verifier output:

```text
== 1. module __file__ identity (imported from the external delivery) ==
  buildreasonseg/runtime/detector.py: __file__=C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py matches_external=True sha_matches_manifest=True
  buildreasonseg/runtime/imageio.py: __file__=C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\imageio.py matches_external=True sha_matches_manifest=True
== 2. historical R6 ten-field evidence via git show ==
  r4: commit=d1c7f6374f32 recovered=True
  r5: commit=53de75ae4f52 recovered=True
  r6: commit=12d5fd9a92a5 recovered=True
  R6 ten-field flags recovered = [True, True, True, True] �� all true = True
== 3. live P1D12 counts ==
  right/1010: raw=6(exp 6) merged=6(exp 6) eligible=4(exp 4) -> True
  left/1003: raw=66(exp 66) merged=53(exp 53) eligible=42(exp 42) -> True
  above/1008: raw=9(exp 9) merged=9(exp 9) eligible=4(exp 4) -> True
  below/1009: raw=7(exp 7) merged=6(exp 6) eligible=3(exp 3) -> True
== 4. production selected replay ==
  right: production selected ids across R4/R5/R6 = {'r4': 1, 'r5': 1, 'r6': 1} �� evidence=1 �� consistent=True �� detector_source_sha=82531dc3b758
  left: production selected ids across R4/R5/R6 = {'r4': 14, 'r5': 14, 'r6': 14} �� evidence=14 �� consistent=True �� detector_source_sha=82531dc3b758
  above: production selected ids across R4/R5/R6 = {'r4': 4, 'r5': 4, 'r6': 4} �� evidence=4 �� consistent=True �� detector_source_sha=82531dc3b758
  below: production selected ids across R4/R5/R6 = {'r4': 1, 'r5': 1, 'r6': 1} �� evidence=1 �� consistent=True �� detector_source_sha=82531dc3b758
== 5/6. bestEligible / bestAny replay and 12/12 role facts ==
  right: roles [True, True, True] -> 3/3 �� class_match=True (REFERENCE_SELECTED_CORRECT)
  left: roles [True, True, True] -> 3/3 �� class_match=True (REFERENCE_SELECTION_WRONG_COVERED)
  above: roles [True, True, True] -> 3/3 �� class_match=True (REFERENCE_ELIGIBILITY_BLOCKED)
  below: roles [True, True, True] -> 3/3 �� class_match=True (REFERENCE_SELECTION_WRONG_COVERED)
  role facts matched = 12/12
STANDALONE_VERIFIER: PASS
detector_or_model_calls = 0 �� canonical evidence untouched = true
```

### 20.1 Explicit non-execution

```text
detector / model / Qwen / SAM2 / relation fields / D-B1 / target segmentation = NONE
canonical evidence / external delivery / canonical RC1 modified = NO / NO / NO
manual visual inspection / candidate replacement / product repair = NO / NO / NO
```
