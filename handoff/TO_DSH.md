# TO_DSH — Task 8B.3-P1D2: Controlled A2 Detector Result Probe

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `dfc66e5a97639074149ce66e6cd4a6f0180e3e71`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

Resolve the one remaining ambiguity in `RC1-DEMO-PROP-01`:

```text
historical A2:
1024×1024
9 tiles
raw=0
merged=0
E401
one "WARNING NMS time limit 2.050s exceeded"
```

P1D1 established:
- NMS time-limit warning is emitted after current-image NMS output is stored;
- the warning itself does not erase the current tile output;
- wrapper has no confidence/class/size post-filter;
- BUT `DetectorRuntime.detect_tile()` returns [] when `result.masks is None` even if boxes may exist.

Therefore ChatGPT supersedes the P1D1 primary classification:

```text
PROP01_MODEL_ZERO_DETECTION_SUSPECT  →  PROP01_INSUFFICIENT_EVIDENCE
```

This task performs exactly ONE controlled detector-only A2 probe to distinguish:

```text
A. post-NMS boxes really zero
B. boxes nonzero but segmentation masks missing
C. current frozen detector no longer reproduces historical raw=0
```

No fix is allowed.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = dfc66e5a97639074149ce66e6cd4a6f0180e3e71
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Do not rebase/reset/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- modify detector.py or any runtime source;
- modify tests;
- modify manifests;
- modify external delivery source/model assets;
- change TILE_SIZE / overlap / stride;
- change imgsz / conf / max_det / NMS IoU;
- lower confidence threshold;
- enable TTA;
- change `retina_masks`;
- run full `predict.py`;
- run `--inspect-proposals`;
- run Qwen / SAM2 / D-B1;
- run A1/A3/A4/B1/B2;
- run pytest/check_setup;
- rerun A2 after the one probe;
- fix PROP-01;
- touch REF-01 / MASK-01;
- enter Task 8B.4 / 8C.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d2_a2_detector_result_probe.md
docs/task8b3_p1d1_a2_zero_proposal_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

A temporary diagnostic script/transcript may be created OUTSIDE the repository or under external runtime logs.
It must not be committed.

# 4. Preflight integrity

Before any model load, verify external manifest:

```text
135/135 PASS
```

Verify external `source_manifest.json` byte-identical to canonical.

Verify:

```text
inference/input/A2.png
size = 1024×1024
sha256 = 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
```

If mismatch:
- do not run model;
- STOP.

# 5. Frozen detector parameters

The diagnostic MUST use exactly the runtime constants from the synchronized external code:

```text
TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = 384
IMGSZ = 640
CONF = 0.05
MAX_DET = 300
FROZEN_THRESHOLD = 0.5
retina_masks = False
TTA = disabled
```

Use the same detector checkpoint and resolved device as the external RC1.

Do not override any of these.

# 6. One diagnostic process only

Create a one-off diagnostic script OUTSIDE the repository.

Run that script exactly ONCE.

Inside this single process:

1. load A2 RGB once;
2. import the existing external:
   - `plan_tiles`
   - `extract_tile`
   - `DetectorRuntime`
   - frozen constants;
3. instantiate/load the existing detector once;
4. generate the normal 9 A2 tile windows;
5. for each tile, call the underlying loaded YOLO model exactly once using the same arguments as `detect_tile()`:

```python
model.predict(
    source=tile_rgb,
    imgsz=detector.imgsz,
    conf=detector.conf,
    max_det=detector.max_det,
    verbose=False,
    retina_masks=False,
    device=<same resolved device>
)
```

Do NOT call `DetectorRuntime.detect_tile()` afterward because that would perform a second inference.

# 7. Required per-tile instrumentation

Immediately before each `model.predict()` print:

```text
BEGIN_TILE <source_tile_id>
```

Immediately after it returns, record:

```text
results_len
boxes_is_none
boxes_count
masks_is_none
masks_count
boxes_conf_min
boxes_conf_max
boxes_conf_top5
```

Rules:
- `boxes_count = len(result.boxes)` when boxes exists;
- `masks_count = len(result.masks.data)` when masks exists;
- confidence fields = `NONE` if no boxes;
- do not save/change predictions;
- do not apply another threshold.

Then compute, WITHOUT another model call, the count that the frozen wrapper would return:

```text
wrapper_output_count =
0 if:
    not results
    OR result.masks is None
    OR result.boxes is None
    OR len(result.boxes) == 0
else:
    number of masks returned by result.masks.data
```

Record this per tile.

Also print:

```text
END_TILE <source_tile_id>
```

This brackets any Ultralytics NMS warning so the warning can be attributed to a specific tile.

# 8. Aggregate required facts

After all 9 tiles, record:

```text
tile_count = 9
sum_boxes_count
sum_masks_count
sum_wrapper_output_count
tiles_boxes_nonzero
tiles_masks_none
tiles_boxes_zero
tiles_wrapper_nonzero
NMS warning count
NMS warning tile ids
```

Do not call merge_proposals; this gate ends at detector-result structure.

# 9. Outcome classification

Choose exactly ONE:

## A — `PROP01_MODEL_ZERO_AT_FROZEN_CONF_CONFIRMED`

Only if:

```text
all 9 tiles:
boxes_count = 0
wrapper_output_count = 0
```

Interpretation:
the frozen detector currently returns no post-NMS boxes at conf=0.05.

Do NOT infer anything about boxes below 0.05 because no lower-threshold run is allowed.

## B — `PROP01_MASK_OUTPUT_MISSING_CONFIRMED`

Only if:
- at least one tile has `boxes_count > 0`;
- that tile has `masks_is_none = True` or zero usable mask output;
- aggregate `sum_wrapper_output_count = 0`.

Interpretation:
historical `raw=0` is structurally explainable by wrapper requiring segmentation masks despite nonzero boxes.

## C — `PROP01_HISTORICAL_ZERO_NOT_REPRODUCED`

If:

```text
sum_wrapper_output_count > 0
```

Interpretation:
with frozen current assets/settings, A2 now produces raw wrapper proposals and the historical zero-proposal symptom does not reproduce.

Do NOT call the defect fixed.

## D — `PROP01_DIAGNOSTIC_INCONCLUSIVE`

For any other state or runtime error.

# 10. NMS warning interpretation

Record whether the warning reproduced.

Do NOT use warning presence alone to select A/B/C/D.

If a warning occurs, the BEGIN/END tile markers must identify which tile emitted it.

# 11. Post-run integrity

After the single probe:
- verify external manifest 135/135 PASS;
- verify external `source_manifest.json` byte-identical to canonical.

No second model run.

# 12. Normalize P1D1 report

In:

```text
docs/task8b3_p1d1_a2_zero_proposal_forensics.md
```

append a short ChatGPT-audit correction:

```text
P1D1 primary classification `PROP01_MODEL_ZERO_DETECTION_SUSPECT`
was superseded before P1D2 because `result.masks is None` is an independent
detect_tile empty-return condition. The authoritative pre-P1D2 classification
is `PROP01_INSUFFICIENT_EVIDENCE`.
```

Do not rewrite historical evidence.

# 13. P1D2 report

Create:

```text
docs/task8b3_p1d2_a2_detector_result_probe.md
```

Required:
1. scope / starting HEAD
2. preflight manifest/hash
3. exact detector settings
4. exact diagnostic script path
5. process invocation count = 1
6. per-tile 9-row table with all §7 fields
7. aggregate §8 facts
8. NMS warnings and tile attribution
9. outcome classification
10. post-run manifest/byte identity
11. no functional modifications
12. `RC1-DEMO-PROP-01 = OPEN`
13. MEM-01 CLOSED; REF/MASK OPEN untouched
14. next action = awaiting ChatGPT audit; no fix yet.

# 14. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 no BOM.

Required:

```text
Task: 8B.3-P1D2
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: dfc66e5a97639074149ce66e6cd4a6f0180e3e71
Functional files modified: NO
Diagnostic process invocation count: 1 / 0
A2 dimensions/hash: 1024x1024 / MATCH
Tile count: 9
sum_boxes_count: <n>
sum_masks_count: <n>
sum_wrapper_output_count: <n>
tiles_boxes_nonzero: <n>
tiles_masks_none: <n>
NMS warning count: <n>
Outcome: <one enum>
Pre-run external manifest: 135/135 PASS / FAIL
Post-run external manifest: 135/135 PASS / FAIL / NOT RUN
source_manifest byte-identical: YES / NO
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d2_a2_detector_result_probe.md
Next action: Awaiting ChatGPT audit; no fix authorized.
```

# 15. Commit / push

If COMPLETE:

```text
test(rc1): probe a2 detector result structure
```

If STOP/FAILED:

```text
docs(rc1): record a2 detector probe stop
```

Push current branch normally, no force.

# 16. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- one diagnostic process only;
- exactly one detector inference per each of 9 tiles;
- no second inference through detect_tile/predict;
- exact frozen detector settings;
- per-tile boxes/masks/wrapper counts captured;
- one outcome enum selected exactly;
- post-run manifest intact;
- no functional code/test/manifest change;
- reports/handoff committed and pushed;
- tracked tree clean;
- STOP.

# 17. Final response

```text
TASK 8B.3-P1D2 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Diagnostic process invocations:
1 / 0

A2:
1024x1024
tiles = 9

Aggregate:
sum_boxes_count = <n>
sum_masks_count = <n>
sum_wrapper_output_count = <n>
tiles_boxes_nonzero = <n>
tiles_masks_none = <n>
NMS warnings = <n>

Outcome:
<enum>

Pre/post external manifest:
135/135 PASS / 135/135 PASS

Functional files modified:
NO

PROP-01:
OPEN

MEM-01:
CLOSED

REF-01 / MASK-01:
OPEN / OPEN

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得修改 detector、不得降低阈值、不得重跑 A2、不得进入 REF-01/MASK-01/Task 8B.4/8C。
```
