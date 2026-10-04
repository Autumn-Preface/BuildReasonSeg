# Task 8B.3-P1D3 — A2 Full-Frame (Untiled) Detector Context Probe

## 1. Task and scope

A single diagnostic process feeding the **complete 1024×1024 A2 RGB image** (no tiling) to the same detector
checkpoint with the same settings as P1D2, to test whether A2's zero detection is a tiling/context effect. Exactly
one `model.predict()` call was made. No tiled A2 run, no full `predict.py`, no `--inspect-proposals`, no Qwen/SAM2/
D-B1, no pytest and no other case was executed; no threshold, `device`, `imgsz`, `max_det` or NMS setting was changed.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 54fd6f4c8bf94f893a8c48cf4c0ce5f2132cf386
probe processes = 1   model.predict calls = 1   tiled = False   device = cpu
```

## 2. Settings and inputs

```text
checkpoint = C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\model\buildreasonseg_advisor\detector.pt (54 480 241 bytes, same as P1D2)
input      = inference/input/A2.png (1024x1024 RGB, frozen sha256 10286b1e…)
imgsz = 640 · conf = 0.05 · max_det = 300 · retina_masks = False · device = cpu
```

## 3. Full-frame result (single call, BEGIN_FULLFRAME / END_FULLFRAME markers)

```text
results_len                 = 1
boxes_count                 = 0
masks_is_none               = True
masks_count                 = 0
confidence_max              = None
confidence_mean             = None
wrapper_equivalent_count    = 0
boxes_xyxy (top 5)          = []
NMS warning reproduced      = NO
```

## 4. Comparison with the frozen P1D2 fact

| probe | tiling | calls | boxes_count | wrapper_equivalent_count |
|---|---|---:|---:|---:|
| P1D2 | 9 × 512 px tiles | 9 | 0 (every tile) | 0 |
| P1D3 | none (full 1024×1024) | 1 | **0** | **0** |

## 5. Outcome (exactly one)

```text
PROP01_GLOBAL_ZERO_AT_FROZEN_CONF_CONFIRMED
```

Justification: the full-frame call returned `boxes_count = 0` and `wrapper_equivalent_count = 0`, which is exactly
condition B — the detector remains zero-detection at the frozen `conf = 0.05` even with full-image context. No
inference is made about whether sub-0.05 detections exist, because no lower-threshold run is authorized.

## 6. Recommendation (exactly one — NOT executed)

```text
NEXT = CONTROLLED_SUBTHRESHOLD_A2_PROBE
```

Recorded only; it is not executed in this task and requires a new task book.

## 7. Post-run integrity

```text
external manifest 135/135 = PASS (mismatches: none)
external source_manifest byte-identical to canonical = True
probe processes = 1 · model.predict calls = 1 · no second run
```

## 8. Scope statement

No tiled A2 rerun, full predict, inspect-proposals, Qwen/SAM2/D-B1, pytest, `check_setup.py` or other-case run
occurred; `detector.py`, tests, `source_manifest.json` and external delivery functional files were not modified;
`RC1-DEMO-PROP-01` remains open and unfixed; REF-01 / MASK-01 / Task 8B.4 / Task 8C were not entered.
