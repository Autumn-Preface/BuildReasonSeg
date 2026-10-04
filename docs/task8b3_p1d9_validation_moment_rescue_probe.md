# Task 8B.3-P1D9 — Validation-Moment Photometric Rescue Probe (isolated)

## 1. Task and scope

One isolated diagnostic run of the **active RC1 detector** on A2 after a single **pre-declared** global affine
photometric transform. No other enhancement, no second transform, no CLAHE/gamma/equalisation, no original-A2 run, no
epoch-18/YOLOv8m run, no full `predict.py`, no Qwen/SAM2/D-B1 and no pytest.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = a220faf0dd9872829ee2bd7b7be859bbd90db7bb
probe processes = 1   model.predict calls = 9 (one per tile)
adoption of the transform = NO
```

## 2. Pre-declared transform

```text
alpha = 4.259966488571338
beta  = -188.2294346811672
y     = clip(alpha * x + beta, 0.0, 255.0)   → rounded to uint8
```

Derived from the frozen validation moment: `alpha = 26.9494 / 6.3262`, `beta = 91.6018 - alpha * 65.6886`
(A2's frozen whole-image `Y_std` and `Y_mean` mapped onto the frozen validation/combined moment).

## 3. Transform statistics and clipping

```text
source A2                      : Y_mean 65.6886 · Y_std 6.3262 · Y_dynamic_98 31.2880
transformed image              : Y_mean 91.5928 · Y_std 26.8645 · Y_dynamic_98 133.1520
transformed dark_fraction      : 0.1355
transformed bright_fraction    : 0.0022
clipped below 0 fraction       : 0.001974
clipped above 255 fraction     : 0.000095
clipped (either side) fraction : 0.002070
temporary file                 : C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_p1d9\A2_validation_moment.png (1979525 bytes)
```

The temporary transform was written **outside** `inference/input` (the RC1 input tree is untouched).

## 4. Per-tile results (active detector, frozen settings)

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

```text
sum_boxes_count              = 0
sum_wrapper_equivalent_count = 0
tiles_with_boxes             = []
tiles_with_masks             = []
global_max_confidence        = None
NMS warnings                 = []
```

## 5. Outcome (exactly one)

```text
PROP01_VALIDATION_MOMENT_RESCUE_REMAINS_ZERO
```

## 6. Recommendation (exactly one — NOT executed)

```text
NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION
```

## 7. Integrity verification

```text
original A2 unchanged (sha256 + mtime)        = YES
active RC1 detector unchanged (sha256)        = YES
external manifest before / after              = 135/135 PASS / 135/135 PASS
external source_manifest byte-identical       = YES
```

## 8. Scope statement

The transform was **not** adopted, no RC1 input/config/runtime file was modified, and no other detector,
enhancement or pipeline stage was exercised. `RC1-DEMO-PROP-01` remains open; REF-01 / MASK-01 / Task 8B.4 / Task 8C
were not entered.
