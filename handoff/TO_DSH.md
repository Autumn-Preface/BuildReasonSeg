# TO_DSH — Task 8B.3-P1D6: Isolated Epoch-18 Alternate Detector A2 Probe

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `ede134cf2fea622c1e3ea2229ea6417a8633c8c1`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

P1D5-R1 is accepted by ChatGPT as completing the detector-provenance evidence contract.

Frozen facts:

```text
active detector =
YOLO26m-seg continued checkpoint
sha256 ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474

active training lineage =
WHU building dataset
1 class: building

P1D2 = active checkpoint, 9 normal A2 tiles, conf=0.05 → 9/9 boxes=0
P1D3 = active checkpoint, full A2, conf=0.05 → boxes=0
P1D4 = active checkpoint, 9 normal A2 tiles, conf=0.001 → 9/9 boxes=0

A2 source/domain = NOT ESTABLISHED

validated alternate available =
YOLO26m-seg epoch-18 checkpoint
artifacts/checkpoints/task6m/runs/m1_yolo26m_seg/weights/best.pt
sha256 prefix fd407db634a8a7ef…
```

R1 also established:

```text
formal replacement/adoption of the RC1 detector:
DETECTOR_CHANGE_TOUCHES_FROZEN_RESEARCH_CLAIMS

isolated alternate-detector A2 diagnostic only:
ALLOWED_AS_SEPARATE_DIAGNOSTIC
```

Therefore this task performs ONE isolated diagnostic with the closest available alternate: the earlier epoch-18
YOLO26m-seg checkpoint from the same training lineage.

The only intended causal variable relative to P1D2 is:

```text
checkpoint:
active continued best.pt
→ epoch-18 best.pt
```

No product change or detector adoption is authorized.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = ede134cf2fea622c1e3ea2229ea6417a8633c8c1
```

Allowed initial tracked tree: clean; or only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify external delivery;
- replace/copy `detector.pt`;
- modify runtime/tests/manifests/config/model package;
- modify the active detector checkpoint;
- train/fine-tune/resume/export/download anything;
- run the active continued detector again;
- run full-frame A2;
- run YOLOv8m baseline or any third checkpoint;
- change tile geometry, `imgsz=640`, `conf=0.05`, `max_det=300`, CPU device, TTA, or `retina_masks=False`;
- run normal `predict.py`, `--inspect-proposals`, Qwen/SAM2/D-B1, A1/A3/A4/B1/B2, pytest/check_setup;
- merge alternate outputs into RC1;
- implement a fallback;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main or force push.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d6_epoch18_alternate_a2_probe.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Temporary diagnostic scripts/transcripts must remain outside the repository or under non-tracked runtime logs.

# 4. Preflight — active RC1 integrity

Before loading the alternate checkpoint verify:

```text
external source manifest = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

Verify frozen A2:

```text
path = external inference/input/A2.png
dimensions = 1024×1024
RGB
sha256 = 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
```

If mismatch: no model call; STOP.

# 5. Alternate checkpoint identity gate

Use exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\artifacts\checkpoints\task6m\runs\m1_yolo26m_seg\weights\best.pt
```

Before loading it record `exists`, bytes, and full SHA256.

Require that it matches the P1D5-R1 epoch-18 artifact identity:

```text
bytes = 162481487
sha256 starts with fd407db634a8a7ef
family = YOLO26m-seg
training dataset = artifacts/task6m_yolo_native/data.yaml
class = building
```

If identity/provenance does not match: no inference; STOP.

# 6. Frozen geometry/settings

Use the same real tiled geometry as P1D2:

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
checkpoint = epoch-18 alternate
imgsz = 640
conf = 0.05
max_det = 300
verbose = False
retina_masks = False
device = cpu
TTA = disabled
```

The checkpoint is the only intended changed detector variable relative to P1D2.

# 7. One diagnostic process only

Create one temporary script outside the repo and run it exactly ONCE.

Inside the single process:
1. load A2 RGB once;
2. create the normal 9 windows;
3. load the epoch-18 YOLO26m-seg checkpoint once;
4. for each tile call underlying `model.predict()` exactly once with the frozen settings.

Total `model.predict` calls must be exactly 9. Do NOT invoke any second model/checkpoint in the process.

# 8. Required per-tile evidence

Bracket each call with `BEGIN_TILE <tile_id>` / `END_TILE <tile_id>`.

For each tile record:

```text
boxes_is_none
boxes_count
masks_is_none
masks_count
wrapper_equivalent_count
boxes_conf_min
boxes_conf_max
boxes_conf_top10
NMS warning inside tile bracket = YES/NO
```

For up to top 10 returned detections record rank, confidence, xyxy, and mask_area if available.

Do NOT make semantic/manual visual judgments and do NOT save predictions into RC1 output directories.

# 9. Aggregate facts

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

No merge/reference/downstream reasoning is run.

# 10. Outcome classification

Choose exactly ONE.

## A — `PROP01_EPOCH18_RECOVERS_A2_SIGNAL`

Require `sum_wrapper_equivalent_count > 0` and at least one tile has usable masks.

Interpretation: the earlier same-lineage YOLO26m-seg checkpoint produces usable A2 proposal signal under the same
geometry/conf/imgsz/device where the active continued checkpoint produced zero. This supports a checkpoint-specific
regression/blind-spot hypothesis. It does NOT authorize replacing the RC1 detector.

## B — `PROP01_EPOCH18_ALSO_ZERO`

Require:

```text
sum_boxes_count = 0
sum_wrapper_equivalent_count = 0
```

Interpretation: the closest same-lineage earlier checkpoint also fails on A2, weakening a continuation-specific
regression hypothesis.

## C — `PROP01_EPOCH18_BOXES_WITHOUT_MASKS`

Require `sum_boxes_count > 0` but `sum_wrapper_equivalent_count = 0` because boxes lack usable segmentation masks.

## D — `PROP01_EPOCH18_DIAGNOSTIC_INCONCLUSIVE`

For runtime/identity/result-structure errors or any other state.

# 11. Next-gate recommendation — do NOT execute

If A:
```text
NEXT = EPOCH18_ZERO_PROPOSAL_RESCUE_DESIGN
```

If B:
```text
NEXT = CONTROLLED_YOLOV8M_A2_PROBE
```

If C:
```text
NEXT = EPOCH18_SEGMENTATION_OUTPUT_FORENSICS
```

If D:
```text
NEXT = ALTERNATE_PROBE_RECOVERY
```

Do NOT execute it.

# 12. Post-run integrity

After the one diagnostic process verify again:

```text
external source manifest = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

No external source/model file may have changed.

# 13. Report

Create:

```text
docs/task8b3_p1d6_epoch18_alternate_a2_probe.md
```

Required sections:
1. scope / starting HEAD
2. P1D2–P1D5-R1 frozen facts
3. active RC1 integrity preflight
4. A2 identity
5. alternate checkpoint path/bytes/full SHA256/provenance
6. exact settings and statement that checkpoint is the only detector variable changed
7. diagnostic process count = 1
8. model.predict count = 9
9. 9-row per-tile results
10. aggregate facts
11. NMS warning evidence
12. exact outcome enum
13. causal interpretation and limits
14. exact next gate
15. post-run RC1 integrity
16. no active RC1 detector modification
17. no functional repository modification
18. `RC1-DEMO-PROP-01 = OPEN`
19. `RC1-DEMO-MEM-01 = CLOSED`
20. REF-01/MASK-01 OPEN untouched
21. no adoption/replacement authorized.

# 14. FROM_DSH

Preserve ARTIFACT-FACTS exactly. UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D6
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: ede134cf2fea622c1e3ea2229ea6417a8633c8c1
Functional files modified: NO
Active RC1 detector modified/replaced: NO
Diagnostic process invocation count: 1 / 0
Detector model.predict call count: 9 / other
A2 dimensions/hash: 1024x1024 / MATCH
Alternate checkpoint: <path>
Alternate checkpoint bytes: <n>
Alternate checkpoint SHA256: <full sha>
Alternate family/task: YOLO26m-seg / instance segmentation
Geometry: 9 x 512 tiles, overlap 128, stride 384
Device: cpu
imgsz/conf/max_det: 640 / 0.05 / 300
sum_boxes_count: <n>
sum_masks_count: <n>
sum_wrapper_equivalent_count: <n>
tiles_with_boxes: <n>
tiles_with_masks: <n>
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
Report: docs/task8b3_p1d6_epoch18_alternate_a2_probe.md
Next action: Awaiting ChatGPT audit; no detector adoption authorized.
```

# 15. Commit / push

If COMPLETE:

```text
test(rc1): probe epoch18 detector on a2
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record epoch18 a2 probe stop
```

Push current branch normally. No force push.

# 16. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- active RC1 integrity preflight PASS;
- alternate checkpoint identity/provenance verified;
- one diagnostic process only;
- exactly 9 epoch-18 detector calls, one per normal A2 tile;
- no active detector call and no second alternate;
- no product/runtime modification;
- one outcome enum selected;
- one next gate recommended but not executed;
- post-run RC1 integrity PASS;
- only allowed docs/handoff files changed;
- commit and push succeed;
- tracked tree clean;
- STOP.

# 17. Final response

```text
TASK 8B.3-P1D6 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Diagnostic process:
1

model.predict calls:
9

Alternate checkpoint:
<full SHA256>

A2 aggregate:
boxes = <n>
masks = <n>
wrapper usable proposals = <n>
tiles_with_boxes = <n>
tiles_with_masks = <n>
global_max_conf = <...>
NMS warnings = <n>

Outcome:
<enum>

Next gate:
<enum>

Active RC1 detector modified/replaced:
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

等待 ChatGPT 审核；不得采用 alternate detector，不得执行 next gate。
```
