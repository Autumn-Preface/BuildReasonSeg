# Task 8B.3-P1D6 — Isolated Epoch-18 Alternate Detector A2 Probe

## 1. Task and scope

One isolated diagnostic run of the **epoch-18 YOLO26m-seg** checkpoint
(`artifacts/checkpoints/task6m/runs/m1_yolo26m_seg/weights/best.pt`, 162 481 487 bytes,
sha256 `fd407db634a8a7ef…`) on A2's frozen nine-tile geometry: exactly one `model.predict()` per tile, nine calls
total, with the **checkpoint as the only changed detector variable**. Frozen `imgsz = 640`, `conf = 0.05`,
`max_det = 300`, `retina_masks = False`, `device = cpu` and the original tile geometry 512/128/384 were preserved.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = ede134cf2fea622c1e3ea2229ea6417a8633c8c1
probe processes = 1   model.predict calls = 9 (one per tile)   full-frame A2 = NO
active detector re-run = NO   YOLOv8m baseline = NO   adopted into RC1 = NO
```

## 2. Checkpoint guard

```text
alternate checkpoint = artifacts/checkpoints/task6m/runs/m1_yolo26m_seg/weights/best.pt
bytes                = 162 481 487
sha256               = fd407db634a8a7ef83f09f8096686e73407095105f1b45c70c623d18dbf4ea44
equals active checkpoint (ef852b58…) = NO
is the YOLOv8m WHU baseline path      = NO
```

## 3. Per-tile results

| tile | top, left | boxes_count | masks_is_none | masks_count | conf_max | wrapper_equivalent_count |
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

Aggregate:

```text
sum_boxes_count              = 0
sum_wrapper_equivalent_count = 0
tiles_with_boxes             = []
any_masks_is_none            = True
global_max_confidence        = None
NMS warnings                 = []
```

## 4. Outcome (exactly one)

```text
PROP01_EPOCH18_ALSO_ZERO
```

Justification: `sum_boxes_count = 0` and `sum_wrapper_equivalent_count = 0`, which is exactly condition B — the
closest same-lineage earlier checkpoint also fails on A2 under identical geometry/conf/imgsz/device. This **weakens**
a continuation-specific regression hypothesis: the zero-proposal symptom is not explained by the difference between
the epoch-18 snapshot and the continued checkpoint.

## 5. Recommendation (exactly one — NOT executed)

```text
NEXT = CONTROLLED_YOLOV8M_A2_PROBE
```

## 6. Post-run integrity

```text
external manifest before = 135/135 PASS
external manifest after  = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

The alternate checkpoint was loaded read-only from the research artifact tree; no external RC1 product file, runtime
file, test, manifest entry or checkpoint was modified, and the alternate was **not** adopted.

## 7. Scope statement

No active-detector rerun, no full-frame A2, no YOLOv8m run, no full `predict.py`, no Qwen/SAM2/D-B1, no pytest and no
next-gate execution; `RC1-DEMO-PROP-01` remains open; REF-01 / MASK-01 / Task 8B.4 / Task 8C were not entered.
