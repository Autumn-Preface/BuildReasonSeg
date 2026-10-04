# TO_DSH — Task 8B.3-P1D3: A2 Full-Frame Context Probe

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `54fd6f4c8bf94f893a8c48cf4c0ce5f2132cf386`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

P1D2 formally established:

```text
A2 = 1024×1024
frozen tiled path = 9 tiles
all 9 tiles:
    boxes_count = 0
    masks_count = 0
    wrapper_output_count = 0
historical NMS warning did not reproduce
outcome = PROP01_MODEL_ZERO_AT_FROZEN_CONF_CONFIRMED
```

This does NOT yet distinguish:

```text
A. detector cannot detect A2 at conf=0.05 even with global image context
B. 512-px tiling removes enough context that every tile becomes zero-detection
```

This task changes exactly one diagnostic variable:

```text
512-px tiled input
→ one full A2 frame
```

Everything else remains frozen.

No fix is authorized.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 54fd6f4c8bf94f893a8c48cf4c0ce5f2132cf386
```

Allowed tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify detector/runtime source;
- modify tests/manifests;
- modify external delivery/model assets;
- change detector checkpoint;
- change `imgsz=640`;
- change `conf=0.05`;
- change `max_det=300`;
- enable TTA;
- change NMS IoU/config;
- change device away from the P1D2 device (`cpu`);
- run tiled A2 again;
- call `DetectorRuntime.detect_tile()`;
- call `DetectorRuntime.detect_global()`;
- run `predict.py`;
- run `--inspect-proposals`;
- run Qwen/SAM2/D-B1;
- run A1/A3/A4/B1/B2;
- run pytest/check_setup;
- lower threshold;
- run any second A2 inference;
- fix PROP-01;
- enter REF-01/MASK-01/Task 8B.4/8C.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d3_a2_fullframe_context_probe.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Temporary diagnostic script/transcript must remain outside the repository or in external runtime logs and must not be committed.

# 4. Preflight integrity

Before model execution verify:

```text
external manifest = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

Verify frozen A2:

```text
path = inference/input/A2.png
dimensions = 1024×1024
mode = RGB
sha256 = 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
```

If any mismatch:
- no model run;
- STOP.

# 5. Frozen detector setup

Use the exact external detector checkpoint:

```text
model/buildreasonseg_advisor/detector.pt
```

Use exactly:

```text
source = full 1024×1024 A2 RGB image
imgsz = 640
conf = 0.05
max_det = 300
verbose = False
retina_masks = False
device = cpu
TTA = disabled
```

No preprocessing crop/tiling.

Do not manually resize the image; pass the original full RGB array to the existing YOLO model exactly once.

# 6. One diagnostic process / one model call

Create a one-off script outside the repo.

Run the script exactly once.

Inside it:
1. load A2 RGB once;
2. load the same detector checkpoint once;
3. execute exactly ONE:

```python
model.predict(
    source=full_a2_rgb,
    imgsz=640,
    conf=0.05,
    max_det=300,
    verbose=False,
    retina_masks=False,
    device="cpu",
)
```

Do not run any second detector call.

# 7. Required result capture

Record:

```text
results_len
boxes_is_none
boxes_count
masks_is_none
masks_count
boxes_conf_min
boxes_conf_max
boxes_conf_top10
NMS warning count
runtime exception, if any
```

Rules:
- boxes_count = len(result.boxes) when available;
- masks_count = len(result.masks.data) when available;
- confidence values = NONE if no boxes;
- no new thresholding;
- no merge/proposal eligibility/reference logic.

Also calculate the frozen wrapper-equivalent count:

```text
wrapper_equivalent_count =
0 if:
    no results
    OR result.masks is None
    OR result.boxes is None
    OR len(result.boxes) == 0
else:
    len(result.masks.data)
```

# 8. Outcome classification

Choose exactly ONE.

## A — `PROP01_TILING_CONTEXT_LOSS_CONFIRMED`

Only if:

```text
full-frame boxes_count > 0
full-frame masks_count > 0
wrapper_equivalent_count > 0
```

combined with frozen P1D2 fact:

```text
all 9 tiled calls boxes_count = 0
```

Interpretation:
at the same checkpoint/conf/imgsz/device, the detector can detect A2 with full-image context but fails when A2 is decomposed into 512-px tiles.

This identifies tiling/context as the key causal layer.

## B — `PROP01_GLOBAL_ZERO_AT_FROZEN_CONF_CONFIRMED`

Only if:

```text
full-frame boxes_count = 0
wrapper_equivalent_count = 0
```

Interpretation:
the detector remains zero-detection at conf=0.05 even with full-image context.

Do NOT infer whether sub-0.05 detections exist.

## C — `PROP01_FULLFRAME_MASK_OUTPUT_MISSING`

Only if:

```text
full-frame boxes_count > 0
but masks_is_none = True or masks_count = 0
```

Interpretation:
full-frame detection has boxes but segmentation-mask output is unavailable.

## D — `PROP01_FULLFRAME_DIAGNOSTIC_INCONCLUSIVE`

For runtime errors or any other state.

# 9. Next-gate recommendation — do not execute

Record exactly one recommendation based on outcome:

If A:
```text
NEXT = DESIGN_ZERO_PROPOSAL_GLOBAL_CONTEXT_RESCUE
```

If B:
```text
NEXT = CONTROLLED_SUBTHRESHOLD_A2_PROBE
```

If C:
```text
NEXT = SEGMENTATION_OUTPUT_PATH_FORENSICS
```

If D:
```text
NEXT = DIAGNOSTIC_RECOVERY_REQUIRED
```

Do NOT execute the recommendation.

# 10. Post-run integrity

After the single detector call verify again:

```text
external manifest = 135/135 PASS
external source_manifest byte-identical = YES
```

No second model call.

# 11. Report

Create:

```text
docs/task8b3_p1d3_a2_fullframe_context_probe.md
```

Required sections:
1. Scope / starting HEAD
2. P1D2 frozen facts
3. Preflight integrity
4. A2 identity
5. Exact full-frame detector configuration
6. Diagnostic process invocation count = 1
7. Detector model.predict call count = 1
8. Full-frame result structure
9. NMS warning evidence
10. Outcome classification
11. Causal interpretation
12. Next-gate recommendation
13. Post-run integrity
14. Functional files modified = NO
15. `RC1-DEMO-PROP-01 = OPEN`
16. MEM-01 CLOSED; REF/MASK OPEN untouched
17. no fix performed.

# 12. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D3
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 54fd6f4c8bf94f893a8c48cf4c0ce5f2132cf386
Functional files modified: NO
Diagnostic process invocation count: 1 / 0
Detector model.predict call count: 1 / 0
A2 dimensions/hash: 1024x1024 / MATCH
Device: cpu
imgsz/conf/max_det: 640 / 0.05 / 300
Full-frame boxes_count: <n>
Full-frame masks_count: <n>
Full-frame masks_is_none: YES / NO
Full-frame wrapper_equivalent_count: <n>
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
Report: docs/task8b3_p1d3_a2_fullframe_context_probe.md
Next action: Awaiting ChatGPT audit; no fix authorized.
```

# 13. Commit / push

If COMPLETE:

```text
test(rc1): probe a2 full-frame detector context
```

If STOP/FAILED:

```text
docs(rc1): record a2 full-frame probe stop
```

Push current branch normally, no force.

# 14. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- preflight intact;
- exactly one diagnostic process;
- exactly one full-frame model.predict call;
- same checkpoint/imgsz/conf/max_det/device as P1D2;
- no tiled call;
- no threshold change;
- no functional code/test/manifest modification;
- one outcome enum chosen;
- one next gate recommended but not run;
- post-run integrity PASS;
- report/handoff committed and pushed;
- clean tracked tree;
- STOP.

# 15. Final response

```text
TASK 8B.3-P1D3 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Diagnostic process:
1

model.predict calls:
1

A2 full-frame:
boxes_count = <n>
masks_count = <n>
masks_is_none = YES / NO
wrapper_equivalent_count = <n>
NMS warnings = <n>

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

等待 ChatGPT 审核；不得执行下一诊断、不得修改 detector、不得降低阈值、不得进入 REF-01/MASK-01/Task 8B.4/8C。
```
