# TO_DSH — Task 8B.3-P1D4: Controlled A2 Subthreshold Tile Probe

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `9f260ab035b5072d44b1979042fbcc63f7c458b8`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

P1D2 established on the real tiled geometry:

```text
A2 = 1024×1024
9 × 512 tiles
conf = 0.05
all 9 tiles boxes_count = 0
outcome = PROP01_MODEL_ZERO_AT_FROZEN_CONF_CONFIRMED
```

P1D3 established that full-image context does not rescue the detector:

```text
full 1024×1024 A2
conf = 0.05
boxes_count = 0
outcome = PROP01_GLOBAL_ZERO_AT_FROZEN_CONF_CONFIRMED
```

Therefore the next unresolved causal question is:

```text
Does A2 contain detector responses below the frozen 0.05 threshold?
```

This is a diagnostic task only.

Change exactly one detector variable relative to P1D2:

```text
conf: 0.05 → 0.001
```

Return to the real 9-tile geometry.

Do NOT change product/runtime configuration.
Do NOT fix PROP-01.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 9f260ab035b5072d44b1979042fbcc63f7c458b8
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- edit detector/runtime source;
- edit tests/manifests;
- edit external delivery;
- edit model assets;
- change checkpoint;
- change tile size/overlap/stride;
- change `imgsz=640`;
- change `max_det=300`;
- change NMS IoU/config;
- change `retina_masks=False`;
- change device from `cpu`;
- enable TTA;
- run full-frame A2;
- run normal `predict.py`;
- run `--inspect-proposals`;
- call `DetectorRuntime.detect_tile()` after the probe;
- call `DetectorRuntime.detect_global()`;
- run Qwen/SAM2/D-B1;
- run A1/A3/A4/B1/B2;
- run pytest/check_setup;
- perform more than one diagnostic process;
- perform more than one `model.predict()` per tile;
- run another confidence value;
- save this threshold into product config;
- fix PROP-01;
- enter REF-01 / MASK-01 / Task 8B.4 / 8C.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d4_a2_subthreshold_tile_probe.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

A temporary probe script/transcript may exist outside the repository or under external runtime logs.
Do not commit it.

# 4. Preflight integrity

Before model execution verify:

```text
external manifest = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

Verify A2:

```text
path = inference/input/A2.png
dimensions = 1024×1024
mode = RGB
sha256 = 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
```

If mismatch:
- do not run model;
- STOP.

# 5. Frozen geometry/settings

Use the synchronized external runtime's existing:

```text
TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = 384
```

Require the A2 tile plan:

```text
9 tiles
tops  = [0, 384, 512]
lefts = [0, 384, 512]
```

Use the same external checkpoint as P1D2/P1D3.

Use exactly:

```text
imgsz = 640
conf = 0.001        ← the only diagnostic change
max_det = 300
verbose = False
retina_masks = False
device = cpu
TTA = disabled
```

# 6. One diagnostic process

Create one temporary script outside the repo.

Run the script exactly once.

Inside the one process:

1. load A2 RGB once;
2. use the existing external `plan_tiles` and `extract_tile`;
3. load the existing detector checkpoint once;
4. obtain the normal 9 tile windows;
5. for each tile call the underlying model exactly once:

```python
model.predict(
    source=tile_rgb,
    imgsz=640,
    conf=0.001,
    max_det=300,
    verbose=False,
    retina_masks=False,
    device="cpu",
)
```

Total detector calls must be exactly:

```text
9
```

No rerun.

# 7. Required per-tile capture

For each tile print markers:

```text
BEGIN_TILE <source_tile_id>
...
END_TILE <source_tile_id>
```

Record:

```text
results_len
boxes_is_none
boxes_count
masks_is_none
masks_count
wrapper_equivalent_count
boxes_conf_min
boxes_conf_max
boxes_conf_top10
```

Also for up to the top 10 boxes by confidence record:

```text
rank
confidence
xyxy
mask_area (if corresponding mask exists)
```

Do not judge semantic quality.

Do not apply a second model threshold.

# 8. Derive threshold counts WITHOUT rerunning inference

From each tile's returned box confidences from the `conf=0.001` call, compute counts at:

```text
>= 0.001
>= 0.005
>= 0.010
>= 0.020
>= 0.030
>= 0.040
>= 0.050
```

This is arithmetic on the single returned result only.

Do NOT call the model again.

Aggregate across all 9 tiles:

```text
total_boxes_at_0.001
total_boxes_at_0.005
total_boxes_at_0.010
total_boxes_at_0.020
total_boxes_at_0.030
total_boxes_at_0.040
total_boxes_at_0.050
global_max_confidence
global_top20_confidences
tiles_with_boxes
tiles_with_masks
sum_masks_count
NMS warning count
NMS warning tile ids
```

Important:
`total_boxes_at_*` are post-NMS returned boxes from the one low-conf call, filtered arithmetically afterward.

# 9. Outcome classification

Choose exactly ONE.

## A — `PROP01_SUBTHRESHOLD_SIGNAL_CONFIRMED`

Require:

```text
total_boxes_at_0.001 > 0
global_max_confidence < 0.05
```

Interpretation:

```text
A2 has post-NMS detector responses below the product threshold;
0.05 is causally involved in the zero-proposal outcome.
```

Do NOT call those responses valid buildings yet.

## B — `PROP01_NO_MEANINGFUL_SUBTHRESHOLD_SIGNAL`

Require:

```text
total_boxes_at_0.001 = 0
```

Interpretation:
even at diagnostic conf=0.001, the frozen detector returns no post-NMS boxes on all 9 tiles.

## C — `PROP01_FROZEN_THRESHOLD_RESULT_INCONSISTENT`

If any returned confidence is:

```text
>= 0.05
```

despite P1D2 having all-zero boxes at `conf=0.05` under the same checkpoint/geometry/device.

Do not explain away the inconsistency.
Record it and STOP after documentation.

## D — `PROP01_SUBTHRESHOLD_DIAGNOSTIC_INCONCLUSIVE`

For runtime errors, missing result structure, or any state that cannot satisfy A/B/C.

# 10. Next-gate recommendation — DO NOT EXECUTE

Choose exactly one based on outcome.

If A:

```text
NEXT = SUBTHRESHOLD_CANDIDATE_QUALITY_AUDIT
```

Purpose of future task:
determine whether the low-confidence masks correspond to real buildings and whether a generic zero-proposal rescue can
be designed without globally lowering the product threshold.

If B:

```text
NEXT = DETECTOR_DOMAIN_GAP_DECISION
```

Purpose:
decide whether PROP-01 requires alternate detector/model-level intervention rather than threshold logic.

If C:

```text
NEXT = DETECTOR_REPRODUCIBILITY_FORENSICS
```

If D:

```text
NEXT = DIAGNOSTIC_RECOVERY_REQUIRED
```

Do not execute the recommendation.

# 11. Post-run integrity

After the one 9-tile process verify:

```text
external manifest = 135/135 PASS
external source_manifest byte-identical = YES
```

No second model run.

# 12. Report

Create:

```text
docs/task8b3_p1d4_a2_subthreshold_tile_probe.md
```

Required sections:

1. task/scope/starting HEAD
2. P1D2 and P1D3 frozen facts
3. preflight integrity
4. A2 identity
5. exact settings and single changed variable
6. process count = 1
7. model.predict calls = 9
8. 9-row per-tile table
9. top box confidence/bbox/mask-area evidence
10. derived threshold-count table
11. aggregate facts
12. NMS warning evidence
13. exact outcome enum
14. causal interpretation with explicit limits
15. next gate recommendation
16. post-run integrity
17. functional modifications = NO
18. `RC1-DEMO-PROP-01 = OPEN`
19. MEM-01 CLOSED
20. REF-01/MASK-01 OPEN untouched
21. no product threshold change.

# 13. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D4
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 9f260ab035b5072d44b1979042fbcc63f7c458b8
Functional files modified: NO
Diagnostic process invocation count: 1 / 0
Detector model.predict call count: 9 / other
A2 dimensions/hash: 1024x1024 / MATCH
Geometry: 9 x 512 tiles, overlap 128, stride 384
Device: cpu
imgsz/max_det/retina_masks: 640 / 300 / False
Diagnostic conf: 0.001
Product conf modified: NO
total_boxes_at_0.001: <n>
total_boxes_at_0.005: <n>
total_boxes_at_0.010: <n>
total_boxes_at_0.020: <n>
total_boxes_at_0.030: <n>
total_boxes_at_0.040: <n>
total_boxes_at_0.050: <n>
global_max_confidence: <value / NONE>
tiles_with_boxes: <n>
tiles_with_masks: <n>
sum_masks_count: <n>
NMS warning count: <n>
Outcome: <one enum>
Next gate: <one enum>
Pre-run external manifest: 135/135 PASS / FAIL
Post-run external manifest: 135/135 PASS / FAIL / NOT RUN
source_manifest byte-identical: YES / NO
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d4_a2_subthreshold_tile_probe.md
Next action: Awaiting ChatGPT audit; no fix authorized.
```

# 14. Commit / push

If COMPLETE:

```text
test(rc1): probe a2 subthreshold detector signal
```

If STOP/FAILED:

```text
docs(rc1): record a2 subthreshold probe stop
```

Push current branch normally, no force.

# 15. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- one diagnostic process;
- exactly 9 detector calls, one per normal tile;
- only changed detector variable is diagnostic conf=0.001;
- no product threshold/config change;
- no second threshold inference;
- confidence distribution and derived counts recorded;
- exact outcome enum chosen;
- next gate recommended but NOT executed;
- post-run manifest intact;
- no functional code/test/manifest change;
- report/handoff committed and pushed;
- clean tracked tree;
- STOP.

# 16. Final response

```text
TASK 8B.3-P1D4 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Diagnostic process:
1

model.predict calls:
9

Diagnostic conf:
0.001

Product conf modified:
NO

A2 aggregate:
boxes@0.001 = <n>
boxes@0.005 = <n>
boxes@0.010 = <n>
boxes@0.020 = <n>
boxes@0.030 = <n>
boxes@0.040 = <n>
boxes@0.050 = <n>
global_max_conf = <...>
tiles_with_boxes = <n>
tiles_with_masks = <n>

Outcome:
<enum>

Next gate:
<enum>

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

等待 ChatGPT 审核；不得修改产品阈值、不得重跑 A2、不得执行下一 gate、不得进入 REF-01/MASK-01/Task 8B.4/8C。
```
