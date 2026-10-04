# TO_DSH — Task 8B.3-P1D9: Isolated Validation-Moment Photometric Rescue Probe

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `a220faf0dd9872829ee2bd7b7be859bbd90db7bb`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

P1D8-R1 is accepted by ChatGPT.

Frozen evidence:

```text
active continued YOLO26m-seg:
  A2 tiled @ conf=0.05  -> 9/9 boxes=0
  A2 full-frame @ 0.05  -> boxes=0
  A2 tiled @ 0.001      -> 9/9 boxes=0

same-lineage epoch-18 YOLO26m-seg:
  A2 tiled @ 0.05 -> 9/9 boxes=0

independent WHU YOLOv8m-seg:
  A2 tiled @ 0.05 -> 9/9 boxes=0

A2 encoding anomaly:
  NOT FOUND
```

P1D8-R1 also established:

```text
combined-distribution primary outcome:
PROP01_A2_NOT_PHOTOMETRIC_OUTLIER

split robustness:
SENSITIVE_TO_SPLIT_COMPOSITION

A2 split-specific percentile:
                    train     val      test     combined
Y_mean              38.88    11.22     1.83      25.18
Y_std               13.45     0.00     0.27       7.83
Y_dynamic_98        13.60     0.00     0.27       7.91
gradient_mean       46.00    69.18    82.45      58.63

train all-black tiles:
1302 / 10044 = 12.9630%, all with empty labels
```

Thus A2 is not an extreme outlier under the original combined rule, but it is extremely low-contrast relative to
the active validation/test distributions and to successful controls.

This task performs ONE isolated diagnostic of ONE predeclared deterministic transform.

No product adoption is authorized.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = a220faf0dd9872829ee2bd7b7be859bbd90db7bb
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify the frozen A2 file;
- modify A1/A3/A4/B1/B2;
- modify external RC1 source/config/model/checkpoint;
- replace the active detector;
- train/fine-tune/resume/export/download anything;
- run epoch-18 YOLO26;
- run YOLOv8m;
- run any checkpoint except the active RC1 detector;
- run original/untransformed A2 again;
- run full-frame A2;
- run more than one photometric transform;
- tune transform parameters after seeing detector output;
- use CLAHE;
- use histogram equalization;
- use gamma correction;
- use per-tile normalization;
- change detector checkpoint/settings;
- change tile geometry;
- change `imgsz=640`;
- change `conf=0.05`;
- change `max_det=300`;
- change `retina_masks=False`;
- change device from CPU;
- enable TTA;
- run normal `predict.py`;
- run `--inspect-proposals`;
- run Qwen/SAM2/D-B1;
- run A1/A3/A4/B1/B2 detector inference;
- run pytest/check_setup;
- save transformed image into RC1 `inference/input`;
- save alternate outputs into RC1 normal output directories;
- implement fallback logic;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d9_validation_moment_rescue_probe.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Temporary transformed A2 and diagnostic transcript must remain outside the repository and outside external RC1 normal
input/output paths.

# 4. Preflight integrity

Before creating any transformed image verify:

```text
external manifest = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

Verify frozen A2:

```text
path = external inference/input/A2.png
1024x1024
RGB
sha256 = 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
```

Verify active detector identity:

```text
external model/buildreasonseg_advisor/detector.pt
sha256 = ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474
```

Mismatch -> STOP before model call.

# 5. Frozen transform

Use exactly these PREDECLARED constants from the accepted P1D8-R1 evidence:

```text
A2 source luminance mean = 65.6886
A2 source luminance std  = 6.3262

active validation median Y_mean = 91.6018
active validation median Y_std  = 26.9494
```

Define exactly:

```text
alpha = 26.9494 / 6.3262
      = 4.259966488571338

beta  = 91.6018 - alpha * 65.6886
      = -188.2294346811672
```

For each original RGB uint8 pixel, independently on R/G/B:

```python
x = original_rgb.astype(float64)
y = clip(alpha * x + beta, 0.0, 255.0)
rescued_rgb = round(y).astype(uint8)
```

Rounding rule MUST be:

```text
numpy.rint / round-to-nearest-even semantics before uint8 cast
```

No per-channel statistics.
No per-tile transform.
No adaptive second pass.
No other transform.

The transform is applied once to the WHOLE 1024x1024 A2 image, then the normal 9 detector tiles are extracted from
that transformed whole image.

# 6. Temporary transformed-image guard

Write the transformed image only to a temporary non-repository, non-delivery-input path, for example:

```text
C:\D\DeepSeekHarness\scratch\task8b3_p1d9\A2_validation_moment_affine.png
```

Record:

```text
temporary path
SHA256
bytes
dimensions
mode
```

Do NOT overwrite A2.

Before detector inference calculate transformed whole-image:

```text
Y_mean
Y_std
Y_dynamic_98
gradient_mean
fraction any RGB channel == 0
fraction any RGB channel == 255
fraction all RGB channels == 0
fraction all RGB channels == 255
```

Also calculate per-channel min/max/mean/std.

These are diagnostic facts only.

# 7. Transform sanity gate

Require:

```text
dimensions = 1024x1024
mode = RGB
dtype = uint8
original A2 sha256 still unchanged
```

No hard clipping threshold is used to tune/abort based on appearance.

Only STOP if:
- transform implementation does not exactly match §5;
- output cannot be decoded as 1024x1024 RGB uint8;
- original A2 changed.

# 8. Detector runtime

Use the SAME active RC1 detector and pinned runtime as P1D2.

Record:

```text
Python executable
Python version
PyTorch version
Ultralytics version
active detector SHA256
```

Expected Ultralytics:

```text
8.4.164
```

# 9. Frozen detector geometry/settings

Use exactly:

```text
TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = 384

tops  = [0, 384, 512]
lefts = [0, 384, 512]
tile_count = 9

imgsz = 640
conf = 0.05
max_det = 300
verbose = False
retina_masks = False
device = cpu
TTA = disabled
```

Extract tiles from the transformed WHOLE image.

# 10. One diagnostic process / nine calls

Create one temporary diagnostic script outside the repository.

Run it exactly ONCE.

Inside:
1. load original A2 once;
2. verify hash;
3. create transformed whole image once using §5;
4. load active detector once;
5. extract normal 9 tiles from transformed image;
6. call active `model.predict()` once per tile.

Exactly:

```text
diagnostic process count = 1
active model.predict count = 9
```

Do NOT call detector on original A2.

# 11. Required per-tile evidence

Bracket each call:

```text
BEGIN_TILE <tile_id>
...
END_TILE <tile_id>
```

Record:

```text
boxes_is_none
boxes_count
masks_is_none
masks_count
wrapper_equivalent_count
boxes_conf_min
boxes_conf_max
boxes_conf_top10
NMS warning = YES / NO
```

For up to top 10 detections:

```text
rank
confidence
xyxy
mask_area if mask exists
```

Do not perform downstream merge/reference/reasoning.

# 12. Aggregate

Record:

```text
sum_boxes_count
sum_masks_count
sum_wrapper_equivalent_count
tiles_with_boxes
tiles_with_masks
tiles_wrapper_nonzero
global_max_confidence
global_top20_confidences
NMS warning count
NMS warning tile ids
```

# 13. Outcome classification

Choose exactly ONE.

## A — `PROP01_VALIDATION_MOMENT_RESCUE_RECOVERS_SIGNAL`

Require:

```text
sum_wrapper_equivalent_count > 0
```

and at least one tile has usable segmentation masks.

Interpretation:

```text
a single predeclared validation-derived global affine photometric transform restores usable active-detector A2
proposal signal under otherwise frozen detector settings.
```

This makes photometric sensitivity actionable as an engineering rescue hypothesis.

It does NOT authorize adoption.

## B — `PROP01_VALIDATION_MOMENT_RESCUE_REMAINS_ZERO`

Require:

```text
sum_boxes_count = 0
sum_wrapper_equivalent_count = 0
```

Interpretation:

```text
even a strong predeclared validation-moment normalization fails to recover active-detector signal.
```

This materially weakens a simple photometric-rescue strategy.

## C — `PROP01_VALIDATION_MOMENT_BOXES_WITHOUT_MASKS`

Require:

```text
sum_boxes_count > 0
sum_wrapper_equivalent_count = 0
```

because masks are unavailable.

## D — `PROP01_VALIDATION_MOMENT_DIAGNOSTIC_INCONCLUSIVE`

For implementation/runtime/result-structure errors or any other state.

# 14. Next-gate recommendation — DO NOT EXECUTE

If A:

```text
NEXT = PHOTOMETRIC_RESCUE_QUALITY_AND_POLICY_GATE
```

Future task will inspect whether recovered proposals are semantically plausible and define the re-validation burden
before any product adoption.

If B:

```text
NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION
```

If C:

```text
NEXT = PHOTOMETRIC_RESCUE_MASK_OUTPUT_FORENSICS
```

If D:

```text
NEXT = PHOTOMETRIC_PROBE_RECOVERY
```

Do NOT execute the next gate.

# 15. Post-run integrity

After the one process verify again:

```text
original A2 SHA256 unchanged = YES
active detector SHA256 unchanged = YES
external manifest = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

Delete is optional for the temporary transformed image.
Do not copy it into RC1.

# 16. Report

Create:

```text
docs/task8b3_p1d9_validation_moment_rescue_probe.md
```

Required sections:

1. task/scope/starting HEAD
2. P1D2–P1D8-R1 frozen facts
3. preflight integrity
4. exact transform formula/constants
5. temporary transformed-image identity
6. transformed-image diagnostic statistics
7. clipping fractions
8. runtime identity
9. frozen detector settings
10. process count = 1
11. model.predict count = 9
12. 9-row per-tile table
13. aggregate facts
14. warning/error evidence
15. exact outcome enum
16. causal interpretation and limits
17. exact next gate
18. post-run integrity
19. original A2 unchanged
20. active RC1 detector unchanged
21. transformed image NOT adopted into RC1
22. functional files modified = NO
23. MEM-01 CLOSED
24. PROP-01 OPEN
25. REF-01/MASK-01 OPEN untouched.

# 17. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D9
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: a220faf0dd9872829ee2bd7b7be859bbd90db7bb
Functional files modified: NO
Original A2 modified: NO
Active RC1 detector modified/replaced: NO
Transform adopted into RC1: NO
Diagnostic process invocation count: 1 / 0
Active detector model.predict call count: 9 / other
Transform alpha: 4.259966488571338
Transform beta: -188.2294346811672
Temporary transformed image SHA256: <sha>
Transformed Y_mean: <value>
Transformed Y_std: <value>
Transformed Y_dynamic_98: <value>
Transformed gradient_mean: <value>
Any-channel zero fraction: <value>
Any-channel 255 fraction: <value>
Geometry: 9 x 512 tiles, overlap 128, stride 384
Device: cpu
imgsz/conf/max_det: 640 / 0.05 / 300
sum_boxes_count: <n>
sum_masks_count: <n>
sum_wrapper_equivalent_count: <n>
tiles_with_boxes: <...>
tiles_with_masks: <...>
global_max_confidence: <value / NONE>
NMS warning count: <n>
Outcome: <exact enum>
Next gate: <exact enum>
Pre-run external manifest: 135/135 PASS / FAIL
Post-run external manifest: 135/135 PASS / FAIL / NOT RUN
source_manifest byte-identical: YES / NO
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d9_validation_moment_rescue_probe.md
Next action: Awaiting ChatGPT audit; no rescue adoption authorized.
```

# 18. Commit / push

If COMPLETE:

```text
test(rc1): probe validation-moment rescue on a2
```

If STOP/FAILED:

```text
docs(rc1): record validation-moment rescue probe stop
```

Push current branch normally.
No force push.

# 19. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- original A2 and detector identity verified;
- exact single transform used;
- no other transform attempted;
- original A2 never modified;
- one diagnostic process;
- exactly 9 active-detector calls on transformed tiles only;
- no original-A2 detector rerun;
- no alternate detector;
- no downstream stages;
- one exact outcome selected;
- one next gate recommended but not executed;
- RC1 integrity remains intact;
- only allowed docs/handoff files changed;
- commit/push succeeds;
- tracked tree clean;
- STOP.

# 20. Final response

```text
TASK 8B.3-P1D9 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Transform:
alpha = 4.259966488571338
beta = -188.2294346811672

Diagnostic process:
1

model.predict calls:
9

A2 transformed:
Y_mean = <...>
Y_std = <...>
Y_dynamic_98 = <...>
gradient_mean = <...>
any-channel zero fraction = <...>
any-channel 255 fraction = <...>

Detector aggregate:
boxes = <n>
masks = <n>
wrapper usable proposals = <n>
tiles_with_boxes = <...>
tiles_with_masks = <...>
global_max_conf = <...>

Outcome:
<enum>

Next gate:
<enum>

Original A2 modified:
NO

Active detector modified:
NO

Transform adopted:
NO

Pre/post external manifest:
135/135 PASS / 135/135 PASS

Functional files modified:
NO

MEM-01:
CLOSED

PROP-01:
OPEN

REF-01 / MASK-01:
OPEN / OPEN

等待 ChatGPT 审核；不得采用该变换、不得执行 next gate。
```
