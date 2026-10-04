# Task 8B.3-P1D7 — Isolated YOLOv8m-seg WHU Baseline A2 Probe

## 1. Task and scope

One isolated diagnostic run of the **independent historical WHU YOLOv8m-seg baseline** on A2's frozen nine-tile
geometry: exactly one `model.predict()` per tile, nine calls total, with the checkpoint as the only changed detector
variable. The **current RC1 pinned repository runtime** was used (not the legacy mutable environment).

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 48ea366e95fdfa022fd6e922f0d5656ad673941b
probe processes = 1   model.predict calls = 9 (one per tile)
active / epoch-18 checkpoint runs = NONE   full predict / Qwen / SAM2 / D-B1 / pytest = NONE
external RC1 detector modified/replaced = NO   YOLOv8m adopted = NO
```

## 2. Checkpoint identity and runtime

```text
checkpoint = C:\D\DeepSeekHarness\workspace\project\WHU_Building_Segment\runs\segment\logs\whu_building_v1\weights\best.pt
bytes      = 54 835 548
sha256     = d9a6a65b7e0819ce4ecbbd9d44a5c8f9dcd2e60ea78203ba8fdf90ba6aaa1f91
matches frozen P1D5-R1 value d9a6a65b… = YES
equals active YOLO26m-seg ef852b58…    = NO
equals epoch-18 YOLO26m-seg fd407db6…  = NO
RUNTIME python=3.11.16 torch=2.13.0+cu132 ultralytics=8.4.164 (RC1 pinned repository environment)
```

The legacy record's original invocation used ultralytics 8.4.67 inside a separate mutable project environment; here
the checkpoint was loaded with the RC1 pinned runtime in the repository environment, with the legacy environment
untouched.

## 3. Frozen detector settings

```text
tile geometry 512/128/384 → 9 tiles · imgsz 640 · conf 0.05 · max_det 300 · retina_masks False · device cpu
```

## 4. Per-tile results

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
tiles_with_masks             = []
any_masks_is_none            = True
global_max_confidence        = None
NMS warnings                 = []
```

## 5. Outcome (exactly one)

```text
PROP01_YOLOV8M_ALSO_ZERO
```

## 6. Recommendation (exactly one — NOT executed)

```text
NEXT = A2_INPUT_DOMAIN_DECISION
```

## 7. Post-run integrity

```text
external manifest before = 135/135 PASS
external manifest after  = 135/135 PASS
external source_manifest byte-identical to canonical = YES
```

## 8. Scope statement

No active or epoch-18 checkpoint run, no additional detector, no full `predict.py`, no Qwen/SAM2/D-B1, no pytest and
no next-gate execution; the external RC1 detector was neither modified nor replaced and YOLOv8m was not adopted;
`RC1-DEMO-PROP-01` remains open; REF-01 / MASK-01 / Task 8B.4 / Task 8C were not entered.
