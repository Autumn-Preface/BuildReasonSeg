# TO_DSH — Task 8B.3-P1D7: Isolated YOLOv8m-WHU Baseline A2 Probe

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `48ea366e95fdfa022fd6e922f0d5656ad673941b`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

P1D6 is accepted by ChatGPT.

Frozen evidence now includes:

```text
active continued YOLO26m-seg @ A2:
9 normal tiles, conf=0.05 → 9/9 boxes=0

same-lineage epoch-18 YOLO26m-seg @ A2:
9 normal tiles, conf=0.05 → 9/9 boxes=0

epoch-18 outcome:
PROP01_EPOCH18_ALSO_ZERO
```

Therefore a continuation-specific regression is no longer the leading explanation.

P1D5-R1 established an independently trained, quantitatively validated historical WHU building baseline:

```text
family = YOLOv8m-seg
training domain = WHU Building Dataset
class = building
historical validation evidence = PRESENT
relation to active detector = SEPARATE_HISTORICAL_BASELINE
historical checkpoint sha256 =
d9a6a65b7e0819ce4ecbbd9d44a5c8f9dcd2e60ea78203ba8fdf90ba6aaa1f91
```

This task performs ONE isolated A2 diagnostic with that baseline checkpoint.

No RC1 adoption/replacement is authorized.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 48ea366e95fdfa022fd6e922f0d5656ad673941b
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify external delivery;
- copy/replace external `detector.pt`;
- modify runtime/tests/manifests/config/model package;
- modify any checkpoint;
- train/fine-tune/resume/export/download anything;
- run active YOLO26 continued checkpoint again;
- run epoch-18 YOLO26 checkpoint again;
- run any other alternate checkpoint;
- run full-frame A2;
- change tile geometry;
- change `imgsz=640`;
- change `conf=0.05`;
- change `max_det=300`;
- enable TTA;
- change `retina_masks=False`;
- use the historical mutable external Ultralytics source tree as runtime;
- run normal `predict.py`;
- run `--inspect-proposals`;
- run Qwen/SAM2/D-B1;
- run A1/A3/A4/B1/B2;
- run pytest/check_setup;
- merge/save alternate predictions into RC1 output directories;
- implement fallback/adoption logic;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d7_yolov8m_whu_a2_probe.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Temporary diagnostic script/transcript may remain outside the repo or in non-tracked diagnostic logs.

# 4. Preflight RC1 integrity

Before alternate model load verify:

```text
external manifest = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

Verify A2:

```text
1024×1024
RGB
sha256 =
10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
```

If mismatch:
- do not load alternate model;
- STOP.

# 5. Baseline checkpoint identity gate

Use exactly the frozen legacy checkpoint identified in the P1D5-R1 provenance:

```text
C:\D\resources\project\WHU_Building_Segment\runs\segment\logs\whu_building_v1\weights\best.pt
```

Before model load record:

```text
exists
bytes
full SHA256
```

Require full SHA256 exactly:

```text
d9a6a65b7e0819ce4ecbbd9d44a5c8f9dcd2e60ea78203ba8fdf90ba6aaa1f91
```

Verify through read-only records:

```text
family = YOLOv8m-seg
task = instance segmentation
class count = 1
class = building
training dataset = WHU Building Dataset
historical training epochs = 100
```

If path/hash/family/task mismatch:
- no model inference;
- STOP.

# 6. Runtime isolation rule

Historical baseline records mention a legacy/mutable Ultralytics environment.

Do NOT use that environment/fork.

Load the YOLOv8m checkpoint using the SAME pinned/current diagnostic Python + Ultralytics runtime that was used for P1D6 / current RC1 detector probes, so the comparison is:

```text
same A2
same runtime
same tile geometry
same imgsz/conf/max_det/device
different trained model family/checkpoint only
```

Record:

```text
Python executable
Ultralytics version
PyTorch version
```

Expected current Ultralytics version is the RC1 pinned version:

```text
8.4.164
```

If the checkpoint cannot load safely under the current pinned runtime:
- do not switch to legacy runtime;
- classify diagnostic INCONCLUSIVE;
- document and STOP.

# 7. Frozen geometry and inference settings

Use exactly the real A2 tile geometry:

```text
TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = 384

tops  = [0, 384, 512]
lefts = [0, 384, 512]
tile_count = 9
```

Use exactly:

```text
checkpoint = frozen YOLOv8m-seg WHU baseline
imgsz = 640
conf = 0.05
max_det = 300
verbose = False
retina_masks = False
device = cpu
TTA = disabled
```

No other detector variable may be changed.

# 8. One diagnostic process only

Create one temporary script outside the repository.

Run it exactly ONCE.

Inside the process:

1. load A2 RGB once;
2. create the normal 9 windows;
3. load the exact YOLOv8m checkpoint once using current pinned Ultralytics;
4. for each tile call:

```python
model.predict(
    source=tile_rgb,
    imgsz=640,
    conf=0.05,
    max_det=300,
    verbose=False,
    retina_masks=False,
    device="cpu",
)
```

exactly once.

Require:

```text
diagnostic process count = 1
model.predict count = 9
```

No second checkpoint/model.

# 9. Required per-tile evidence

Bracket every call:

```text
BEGIN_TILE <tile_id>
...
END_TILE <tile_id>
```

Record per tile:

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

Do not perform manual semantic judgments in this task.

Do not save masks/overlays into RC1 outputs.

# 10. Aggregate

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

Do not run merge/reference/downstream stages.

# 11. Outcome classification

Choose exactly ONE.

## A — `PROP01_YOLOV8M_RECOVERS_A2_SIGNAL`

Require:

```text
sum_wrapper_equivalent_count > 0
```

and at least one tile has usable segmentation masks.

Interpretation:

```text
an independently trained WHU building instance-segmentation detector produces usable
A2 proposal signal under the same runtime/geometry/conf/imgsz/device where both YOLO26
checkpoints produced zero.
```

This supports a YOLO26-lineage/model-specific A2 blind-spot hypothesis.

It does NOT authorize adopting YOLOv8m into RC1.

## B — `PROP01_YOLOV8M_ALSO_ZERO`

Require:

```text
sum_boxes_count = 0
sum_wrapper_equivalent_count = 0
```

Interpretation:

```text
two independent WHU-trained detector lineages both produce zero A2 boxes under the
same product-like diagnostic settings.
```

This materially strengthens the hypothesis that A2 is outside the effective detector domain
or otherwise has severe input/domain incompatibility, while still not proving geographic provenance.

## C — `PROP01_YOLOV8M_BOXES_WITHOUT_MASKS`

Require:

```text
sum_boxes_count > 0
sum_wrapper_equivalent_count = 0
```

because usable masks are absent.

## D — `PROP01_YOLOV8M_DIAGNOSTIC_INCONCLUSIVE`

Use for:
- checkpoint load incompatibility under pinned runtime;
- runtime error;
- identity mismatch;
- unexpected result structure;
- any state not fitting A/B/C.

# 12. Next-gate recommendation — do NOT execute

Choose exactly one.

If A:

```text
NEXT = YOLOV8M_RESCUE_POLICY_DESIGN
```

Purpose:
design the minimum scientifically honest zero-proposal rescue/adaptation strategy and the required re-validation burden.
No adoption yet.

If B:

```text
NEXT = A2_INPUT_DOMAIN_DECISION
```

Purpose:
decide whether A2 should remain a formal RC1 success case, become an explicit unsupported-domain failure case,
or require a detector adaptation data phase.

If C:

```text
NEXT = YOLOV8M_SEGMENTATION_OUTPUT_FORENSICS
```

If D:

```text
NEXT = YOLOV8M_PROBE_RECOVERY
```

Do not execute the recommendation.

# 13. Post-run integrity

After the one process verify:

```text
external manifest = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

Also confirm:

```text
active RC1 detector unchanged = YES
YOLOv8m checkpoint unchanged = YES
alternate adopted into RC1 = NO
```

# 14. Report

Create:

```text
docs/task8b3_p1d7_yolov8m_whu_a2_probe.md
```

Required sections:

1. scope / starting HEAD
2. P1D2–P1D6 frozen evidence
3. RC1 preflight integrity
4. A2 identity
5. YOLOv8m checkpoint path/bytes/full SHA256
6. historical provenance/validation facts
7. current pinned runtime identity
8. exact detector settings
9. diagnostic process count = 1
10. model.predict calls = 9
11. 9-row tile table
12. aggregate facts including sum_masks_count
13. warning/error evidence
14. exact outcome enum
15. causal interpretation and limits
16. exact next gate
17. post-run integrity
18. no RC1 detector modification/adoption
19. no functional repository modification
20. MEM-01 CLOSED
21. PROP-01 OPEN
22. REF-01 / MASK-01 OPEN untouched.

# 15. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D7
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 48ea366e95fdfa022fd6e922f0d5656ad673941b
Functional files modified: NO
Active RC1 detector modified/replaced: NO
Alternate adopted into RC1: NO
Diagnostic process invocation count: 1 / 0
Detector model.predict call count: 9 / other
A2 dimensions/hash: 1024x1024 / MATCH
Alternate checkpoint path: <path>
Alternate checkpoint bytes: <n>
Alternate checkpoint SHA256: <full sha>
Alternate family/task: YOLOv8m-seg / instance segmentation
Runtime Ultralytics: <version>
Geometry: 9 x 512 tiles, overlap 128, stride 384
Device: cpu
imgsz/conf/max_det: 640 / 0.05 / 300
sum_boxes_count: <n>
sum_masks_count: <n>
sum_wrapper_equivalent_count: <n>
tiles_with_boxes: <n or ids>
tiles_with_masks: <n or ids>
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
Report: docs/task8b3_p1d7_yolov8m_whu_a2_probe.md
Next action: Awaiting ChatGPT audit; no detector adoption authorized.
```

# 16. Commit / push

If COMPLETE:

```text
test(rc1): probe yolov8m whu detector on a2
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record yolov8m a2 probe stop
```

Push current branch normally.
No force push.

# 17. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- RC1 preflight PASS;
- exact baseline checkpoint identity/hash proven;
- current pinned runtime used;
- no legacy mutable Ultralytics runtime used;
- one diagnostic process only;
- exactly 9 YOLOv8m calls, one per A2 tile;
- no active/epoch18/other detector calls;
- no product/runtime/checkpoint modification;
- exact outcome enum selected;
- exact next gate recommended but NOT executed;
- post-run RC1 integrity PASS;
- only allowed docs/handoff files changed;
- commit/push succeed;
- tracked tree clean;
- STOP.

# 18. Final response

```text
TASK 8B.3-P1D7 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Diagnostic process:
1

model.predict calls:
9

YOLOv8m checkpoint:
<full SHA256>

Runtime Ultralytics:
<version>

A2 aggregate:
boxes = <n>
masks = <n>
wrapper usable proposals = <n>
tiles_with_boxes = <...>
tiles_with_masks = <...>
global_max_conf = <...>
NMS warnings = <n>

Outcome:
<enum>

Next gate:
<enum>

Active RC1 detector modified/replaced:
NO

Alternate adopted:
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

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得采用 YOLOv8m，不得执行 next gate。
```
