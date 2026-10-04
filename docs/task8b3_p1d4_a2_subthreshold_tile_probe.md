# Task 8B.3-P1D4 — A2 Sub-Threshold (conf=0.001) Tiled Detector Probe

## 1. Task and scope

One diagnostic process running the **real 9-tile geometry** with a **single changed variable**: `conf = 0.001`
instead of the frozen `0.05`. One `model.predict()` per tile, nine calls total. No second confidence value, no
product-config change, no `detector.py`/test/manifest edit, no full `predict.py` and no other sample.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 9f260ab035b5072d44b1979042fbcc63f7c458b8
probe processes = 1   model.predict calls = 9 (one per tile)   conf values used = 0.001 only
```

## 2. Settings

```text
checkpoint = C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\model\buildreasonseg_advisor\detector.pt (54 480 241 bytes)
input      = inference/input/A2.png (1024x1024 RGB, frozen sha256 10286b1e…)
TILE_SIZE 512 · TILE_OVERLAP 128 · TILE_STRIDE 384 · tiles = 9 · IMGSZ 640 · MAX_DET 300 ·
retina_masks False · device cpu · DIAGNOSTIC_CONF 0.001 (only changed variable)
```

## 3. Per-tile results (post-NMS, from the single low-conf call)

| tile | top, left | boxes_count | masks_is_none | masks_count | conf_max |
|---|---|---:|---|---:|---|
| 0 | 0, 0 | 0 | True | 0 | None |
| 1 | 0, 384 | 0 | True | 0 | None |
| 2 | 0, 512 | 0 | True | 0 | None |
| 3 | 384, 0 | 0 | True | 0 | None |
| 4 | 384, 384 | 0 | True | 0 | None |
| 5 | 384, 512 | 0 | True | 0 | None |
| 6 | 512, 0 | 0 | True | 0 | None |
| 7 | 512, 384 | 0 | True | 0 | None |
| 8 | 512, 512 | 0 | True | 0 | None |

## 4. Threshold histogram (arithmetic filtering of the same single call's confidences)

| confidence ≥ | boxes |
|---|---:|
| 0.001 | 0 |
| 0.005 | 0 |
| 0.010 | 0 |
| 0.020 | 0 |
| 0.030 | 0 |
| 0.040 | 0 |
| 0.050 | 0 |

```text
global_max_confidence = None
sum_boxes_count       = 0
any_masks_is_none     = True
NMS warning reproduced = NO
```

## 5. Outcome (exactly one)

```text
PROP01_NO_MEANINGFUL_SUBTHRESHOLD_SIGNAL
```

Justification: `total_boxes_at_0.001 = 0`, which is exactly condition B — even at diagnostic `conf = 0.001` the frozen
detector returns no post-NMS boxes on all nine tiles, so A2 carries no measurable post-NMS detector response at or
above 0.001 (and therefore none at 0.005–0.050 either). Conditions A and C cannot hold because no boxes were returned
at all. No inference is made about pre-NMS candidate distribution, which this probe cannot observe.

## 6. Recommendation (exactly one — NOT executed)

```text
NEXT = DETECTOR_DOMAIN_GAP_DECISION
```

Purpose: decide whether `RC1-DEMO-PROP-01` requires alternate detector/model-level intervention rather than threshold
logic. Recorded only; not executed here.

## 7. Post-run integrity

```text
external manifest 135/135 = PASS (mismatches: none)
external source_manifest byte-identical to canonical = True
probe processes = 1 · predict calls = 9 · second conf value = none
```

## 8. Scope statement

No product configuration, `detector.py`, test, manifest or external functional file was modified; no full predict,
inspect-proposals, Qwen/SAM2/D-B1, pytest, `check_setup.py` or other sample was run; `RC1-DEMO-PROP-01` remains open
and unfixed; REF-01 / MASK-01 / Task 8B.4 / Task 8C were not entered.
