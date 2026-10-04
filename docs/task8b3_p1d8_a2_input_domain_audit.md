# Task 8B.3-P1D8 — A2 Input-Domain (Photometric) Audit

## 1. Task and scope

Non-model input-domain forensics for A2: control identity, active training-domain resolution, frozen photometric
metrics for **every readable training-domain image**, percentile placement of A2 (whole image and nine tiles) and of
the historical successful controls, exact-hash provenance search and PNG encoding audit.

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD   = 5f52338b7507cc377190cacb33611c644f223a88
model runs = NONE (no detector, no predict, no pytest)
image modifications = NONE (no CLAHE, gamma, histogram equalisation or any write to inputs)
```

## 2. Control identity (§4)

| control | path | size | mode | bytes | sha256 | matches frozen |
|---|---|---|---|---|---|---|
| A1 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\A1.png | 1024×1024 | RGB | 1607301 | `8a4b459d65773a7d…` | YES |
| A2 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\A2.png | 1024×1024 | RGB | 1677040 | `10286b1e76db9e38…` | YES |
| A3 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\A3.png | 1024×1024 | RGB | 1775940 | `f3cd05870385bd7f…` | YES |
| A4 | C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\A4.png | 1024×1024 | RGB | 1529883 | `a3962ed184679973…` | YES |

A2 is byte-identical to the frozen value `10286b1e…`; A1/A3/A4 are uniquely identified by the same paths used by the
frozen R4B suite.

## 3. Training-domain source gate (§5)

`artifacts/task6m_yolo_native/data.yaml` resolves to:

```text
path  = artifacts/task6m_yolo_native
train = images/train · val = images/val · test = images/test
names = {0: building}
```

```text
train files = 10044
val files   = 3618
test files  = 3726
formats     = {".tif": 17388}
dimensions  = {"512x512": 17388}
processed   = 17388 readable images (0 unreadable)
```

All readable training-domain images were processed — no random sampling was used.

## 4. Frozen metric definitions (§7) and A2 results

`Y = 0.299*R + 0.587*G + 0.114*B` on decoded uint8 RGB converted to float, no normalisation.

| metric | A2 whole image | A2 whole pct | A2 mean-of-tiles pct | A2 tile-min pct | A2 tile-max pct |
|---|---:|---:|---:|---:|---:|
| Y_mean | 65.6886 | 25.18 | 25.09 | 24.76 | 25.59 |
| Y_std | 6.3262 | 7.83 | 7.83 | 7.68 | 7.98 |
| Y_dynamic_98 | 31.2880 | 7.91 | 7.90 | 7.77 | 8.02 |
| gradient_mean | 4.1943 | 58.63 | 63.29 | 31.35 | 72.59 |
| dark_fraction | 0.4039 | 71.54 | 72.37 | 67.43 | 74.20 |
| very_dark_fraction | 0.0000 | 10.85 | 11.51 | 0.00 | 12.53 |
| bright_fraction | 0.0000 | 0.00 | 0.00 | 0.00 | 0.00 |

A2 nine-tile detail:

| tile | top, left | Y_mean | Y_std | Y_dynamic_98 | gradient_mean | dark_fraction |
|---|---|---|---|---|---|---|
| 0 | 0, 0 | 66.5203 | 5.8834 | 28.7720 | 3.9387 | 0.3240 |
| 1 | 0, 384 | 65.2469 | 5.0018 | 25.6470 | 3.3520 | 0.4050 |
| 2 | 0, 512 | 65.0366 | 4.7920 | 24.3420 | 3.1844 | 0.4366 |
| 3 | 384, 0 | 65.6077 | 7.1210 | 34.1960 | 4.7770 | 0.4255 |
| 4 | 384, 384 | 65.1710 | 6.9624 | 33.4620 | 4.7977 | 0.4481 |
| 5 | 384, 512 | 65.7056 | 6.8457 | 32.6690 | 4.7613 | 0.4176 |
| 6 | 512, 0 | 65.8718 | 7.2442 | 34.8640 | 4.8427 | 0.4109 |
| 7 | 512, 384 | 64.9438 | 7.0895 | 34.0034 | 4.7968 | 0.4637 |
| 8 | 512, 512 | 65.3259 | 6.9817 | 33.2340 | 4.8100 | 0.4440 |

## 5. Training-domain distributions (§8)

`Y_mean` distribution, per split and combined:

| split | N | min | p01 | p05 | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10044 | 0.0000 | 0.0000 | 0.0000 | 82.9159 | 164.6649 | 193.9186 | 226.4681 |
| val | 3618 | 28.6729 | 37.3317 | 51.6796 | 91.6018 | 115.4262 | 126.4882 | 147.7226 |
| test | 3726 | 40.1055 | 59.8574 | 73.6721 | 99.7265 | 135.3221 | 146.5781 | 161.1132 |
| combined | 17388 | 0.0000 | 0.0000 | 0.0000 | 91.4470 | 156.7404 | 186.7173 | 226.4681 |

Full per-split and combined distributions for `Y_mean`, `Y_std`, `Y_dynamic_98`, `gradient_mean`, `dark_fraction`
and `very_dark_fraction` (N, min, p01, p05, p25, p50, p75, p95, p99, max) are recorded in the machine-readable
evidence file `logs/task8b3_p1d8_domain_audit.json`.

## 6. Controls in the training distribution (§9)

| control | Y_mean | Y_std | Y_dynamic_98 | gradient_mean |
|---|---:|---:|---:|---:|
| A1 | 95.0214 | 40.2295 | 179.1770 | 5.2537 |
| A2 | 65.6886 | 6.3262 | 31.2880 | 4.1943 |
| A3 | 81.8057 | 32.0700 | 181.2812 | 4.9596 |
| A4 | 72.3022 | 27.1124 | 135.7002 | 3.5647 |

| control | Y_mean pct | Y_std pct | Y_dynamic_98 pct | gradient_mean pct |
|---|---:|---:|---:|---:|
| A1 | 55.73 | 86.31 | 80.80 | 77.74 |
| A2 | 25.18 | 7.83 | 7.91 | 58.63 |
| A3 | 37.04 | 67.02 | 81.61 | 74.30 |
| A4 | 28.72 | 53.07 | 58.32 | 40.56 |

The three historically successful controls sit inside the training distribution; A2 is a low-contrast sample
(Y_std ≈ 8th percentile, Y_dynamic_98 ≈ 8th percentile) but **above** the 5th percentile on all four core metrics,
and its gradient/texture statistic is above the median.

## 7. Exact-match provenance search (§6)

```text
A2 = A2_NO_EXACT_MATCH
A1 matches = []
A3 matches = []
A4 matches = []
```

No exact byte/hash identity exists between A2 (or A1/A3/A4) and any image in the active train/val/test roots, so
A2's membership in the training domain is **NOT ESTABLISHED** (file identity absent; geographic provenance still
NOT ESTABLISHED).

## 8. PNG / encoding audit (§10)

```text
A1/A2/A3/A4 = PNG signature OK · IHDR bit_depth 8 · colour_type 2 (truecolour RGB) · compression 0 · filter 0 ·
              interlace 0 · chunks = IHDR + IDAT… + IEND only
encoding anomaly found = NONE
```

Byte sizes differ (1 529 883–1 775 940) as expected for different content; there is no ancillary-chunk, bit-depth,
interlace or colour-type difference between A2 and the successful controls.

## 9. Outcome (exactly one)

```text
PROP01_A2_NOT_PHOTOMETRIC_OUTLIER
```

Criteria check: training-domain statistics exist (§3, §5) ✓; A2 whole-image is at or above **p05** on
**4/4** core metrics (Y_mean, Y_std, Y_dynamic_98, gradient_mean) ✓; the encoding audit found no anomaly (§8)
✓ → outcome C applies. Outcomes A and B are not met: A2 is **not** below the training p01 on any of `Y_std`,
`Y_dynamic_98`, `gradient_mean`, and its brightness/dark-fraction position is not extreme.

## 10. Recommendation (exactly one — NOT executed)

```text
NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION
```

## 11. Explicitly NOT ESTABLISHED

```text
A2 geographic/sensor/site provenance                                  : NOT ESTABLISHED
A2 membership in the active training/validation/test splits           : NOT ESTABLISHED (no exact hash match)
whether A2's content is inside the detector's effective domain         : NOT ESTABLISHED
cause of the zero-proposal behaviour (any photometric rescue)          : NOT ESTABLISHED (no rescue attempted)
```

## 12. Scope statement

No detector, Qwen, SAM2, D-B1, `predict.py`, six-image suite or pytest run occurred; no image was modified, resized
or normalised (no CLAHE/gamma/equalisation); no rescue design was implemented; the external RC1 delivery,
repository functional files, tests and manifest were not modified; `RC1-DEMO-PROP-01` remains open and unfixed;
REF-01 / MASK-01 / Task 8B.4 / Task 8C were not entered.
