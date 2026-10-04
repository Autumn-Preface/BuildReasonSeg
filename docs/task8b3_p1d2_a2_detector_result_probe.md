# Task 8B.3-P1D2 — A2 Detector Result-Structure Probe

## 1. Task and scope

One **detector-only** diagnostic process on A2: for each of the nine frozen tiles, exactly one raw
`model.predict()` call, recording the detector result structure (`boxes_count`, `masks_is_none`, `masks_count`,
box confidences) plus the wrapper's would-be output count. `detect_tile()` was **not** called, no full
`predict.py` run occurred and **no threshold was lowered**.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = dfc66e5a97639074149ce66e6cd4a6f0180e3e71
probe processes = 1   model.predict calls = 9 (one per tile)   reruns = 0
```

## 2. Checkpoint and frozen settings

```text
checkpoint      = C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\model\buildreasonseg_advisor\detector.pt
checkpoint size = 54 480 241 bytes
source image    = inference/input/A2.png (1024x1024 RGB, frozen sha256 10286b1e…)
TILE_SIZE = 512 · TILE_OVERLAP = 128 · TILE_STRIDE = 384 · IMGSZ = 640 · CONF = 0.05 ·
MAX_DET = 300 · retina_masks = False · device = cpu
```

`device = cpu` is the only setting not enumerated in the frozen list; it is recorded here for transparency (the task
book fixed tiling, `imgsz`, `conf`, `max_det`, `retina_masks` and the checkpoint, and forbade threshold changes).

## 3. Method

Each tile was wrapped in explicit markers so that any NMS warning can be attributed:

```text
BEGIN_TILE <i> top=… left=… size=512 padding_applied=…
  → exactly one model.predict(source=tile_rgb, imgsz=640, conf=0.05, max_det=300, retina_masks=False)
  → read result.boxes / result.masks
RESULT_TILE <i> boxes_count=… masks_is_none=… masks_count=… conf_max=… wrapper_output_count=…
END_TILE <i>
```

`wrapper_output_count` re-implements the frozen wrapper's counting rule inline (`0` when masks are None, boxes are
None or `len(boxes) == 0`, otherwise `masks.shape[0]`) — `detect_tile()` itself was never invoked.

## 4. Per-tile results

| tile | top, left | boxes_count | masks_is_none | masks_count | conf_max | wrapper_output_count |
|---|---|---:|---|---:|---|---:|
| 0 | 0, 0 | 0 | True | 0 | None | 0 |
| 1 | 0, 384 | 0 | True | 0 | None | 0 |
| 2 | 0, 512 | 0 | True | 0 | None | 0 |
| 3 | 384, 0 | 0 | True | 0 | None | 0 |
| 4 | 384, 384 | 0 | True | 0 | None | 0 |
| 5 | 384, 512 | 0 | True | 0 | None | 0 |
| 6 | 512, 0 | 0 | True | 0 | None | 0 |
| 7 | 512, 384 | 0 | True | 0 | None | 0 |
| 8 | 512, 512 | 0 | True | 0 | None | 0 |

Aggregates:

```text
sum_boxes_count           = 0
sum_wrapper_output_count  = 0
any_masks_is_none         = True
tiles_with_boxes          = []
```

## 5. NMS warning attribution

```text
NMS warnings by tile = {}
```

The historical `WARNING NMS time limit 2.050s exceeded` did **not** reproduce in this probe; per the task book the
warning's presence or absence is recorded but is **not** used to select the outcome.

## 6. Outcome (exactly one)

```text
PROP01_MODEL_ZERO_AT_FROZEN_CONF_CONFIRMED
```

Justification against the task book's conditions: **all nine tiles** have `boxes_count = 0` **and**
`wrapper_output_count = 0`, which is exactly condition A. Because `masks_is_none = True` co-occurs with zero boxes on
every tile (ultralytics returns no masks when there are no detections), this run also confirms that
`result.masks is None` is an independent early-return condition in `detect_tile()` — the reason ChatGPT overrode
P1D1's `PROP01_MODEL_ZERO_DETECTION_SUSPECT` to `PROP01_INSUFFICIENT_EVIDENCE`. No inference about boxes below 0.05
is made, since no lower-threshold run is authorized.

## 7. Post-run integrity

```text
external manifest 135/135 = PASS (mismatches: none)
external source_manifest byte-identical to canonical = True
probe processes = 1, no second model run
```

## 8. Scope statement

No `detect_tile()` call, no full predict, no six-image Demo, no threshold change, no pytest or `check_setup.py`; no
`detector.py`, test, manifest or external-delivery functional file was modified; `RC1-DEMO-PROP-01` remains open and
REF-01 / MASK-01 / Task 8B.4 / Task 8C were not entered.
