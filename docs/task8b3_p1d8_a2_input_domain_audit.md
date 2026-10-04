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


---

## 13. P1D8-R1 — evidence completion (docs only)

No model inference, image enhancement, detector rerun or functional change occurred. The completion pass re-read
every readable training-domain image (read-only arithmetic) to reproduce the required distributions and counts.
**The original P1D8 outcome rule is unchanged**; §13.8 re-applies it to the combined distribution.

### 13.1 Full photometric/texture distributions (train / val / test / combined)

**Y_mean**

| split | N | min | p01 | p05 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10044 | 0.0000 | 0.0000 | 0.0000 | 42.1462 | 82.9159 | 118.1027 | 164.6649 | 193.9186 | 226.4681 |
| val | 3618 | 28.6729 | 37.3317 | 51.6796 | 79.4697 | 91.6018 | 100.9860 | 115.4262 | 126.4882 | 147.7226 |
| test | 3726 | 40.1055 | 59.8574 | 73.6721 | 88.6776 | 99.7265 | 111.7470 | 135.3221 | 146.5781 | 161.1132 |
| combined | 17388 | 0.0000 | 0.0000 | 0.0000 | 65.3772 | 91.4470 | 109.1341 | 156.7404 | 186.7173 | 226.4681 |

**Y_std**

| split | N | min | p01 | p05 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10044 | 0.0000 | 0.0000 | 0.0000 | 18.6128 | 28.7728 | 36.6120 | 49.1570 | 63.9761 | 94.3804 |
| val | 3618 | 6.7987 | 10.6096 | 13.6067 | 20.6706 | 26.9494 | 35.1930 | 51.7849 | 59.0997 | 69.2128 |
| test | 3726 | 5.1775 | 7.7372 | 10.4010 | 16.0292 | 21.2292 | 27.3724 | 38.1184 | 48.3486 | 59.4205 |
| combined | 17388 | 0.0000 | 0.0000 | 0.0000 | 18.3570 | 26.0873 | 34.7892 | 48.4081 | 59.9328 | 94.3804 |

**Y_dynamic_98**

| split | N | min | p01 | p05 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10044 | 0.0000 | 0.0000 | 0.0000 | 90.3599 | 129.7218 | 171.4385 | 240.6460 | 255.0000 | 255.0000 |
| val | 3618 | 34.3920 | 51.7365 | 65.6173 | 96.2658 | 128.9393 | 175.8658 | 220.8581 | 230.0639 | 238.1110 |
| test | 3726 | 20.2770 | 39.1052 | 50.6463 | 79.4313 | 103.2506 | 138.8862 | 191.4315 | 216.4212 | 230.3037 |
| combined | 17388 | 0.0000 | 0.0000 | 0.0000 | 88.5322 | 123.5879 | 165.3901 | 226.4489 | 255.0000 | 255.0000 |

**gradient_mean**

| split | N | min | p01 | p05 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10044 | 0.0000 | 0.0000 | 0.0000 | 3.2599 | 4.3457 | 6.1162 | 8.0427 | 8.8612 | 10.7644 |
| val | 3618 | 1.6354 | 1.9109 | 2.2416 | 3.0451 | 3.7469 | 4.3291 | 5.6329 | 6.5546 | 7.9779 |
| test | 3726 | 0.7658 | 1.4292 | 1.8283 | 2.4357 | 3.1037 | 3.8861 | 5.0463 | 5.8375 | 7.5830 |
| combined | 17388 | 0.0000 | 0.0000 | 0.0000 | 2.8697 | 3.8877 | 5.0212 | 7.6754 | 8.6131 | 10.7644 |

**dark_fraction**

| split | N | min | p01 | p05 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10044 | 0.0000 | 0.0000 | 0.0001 | 0.0383 | 0.2555 | 0.7692 | 1.0000 | 1.0000 | 1.0000 |
| val | 3618 | 0.0000 | 0.0005 | 0.0058 | 0.0614 | 0.1750 | 0.3130 | 0.7073 | 0.8983 | 0.9526 |
| test | 3726 | 0.0000 | 0.0000 | 0.0002 | 0.0067 | 0.0425 | 0.1246 | 0.3322 | 0.6010 | 0.8363 |
| combined | 17388 | 0.0000 | 0.0000 | 0.0002 | 0.0260 | 0.1480 | 0.4852 | 1.0000 | 1.0000 | 1.0000 |

**very_dark_fraction**

| split | N | min | p01 | p05 | p25 | p50 | p75 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10044 | 0.0000 | 0.0000 | 0.0000 | 0.0044 | 0.0707 | 0.4388 | 1.0000 | 1.0000 | 1.0000 |
| val | 3618 | 0.0000 | 0.0000 | 0.0000 | 0.0017 | 0.0284 | 0.0815 | 0.2471 | 0.4476 | 0.5932 |
| test | 3726 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0008 | 0.0106 | 0.0733 | 0.1849 | 0.4120 |
| combined | 17388 | 0.0000 | 0.0000 | 0.0000 | 0.0008 | 0.0244 | 0.1776 | 1.0000 | 1.0000 | 1.0000 |

### 13.2 Zero-valued training tail (§6)

| split | files | Y_mean == 0 | Y_std == 0 | exactly all-black RGB | all-black fraction | label exists | label empty | label non-empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10044 | 1302 | 1302 | 1302 | 0.129630 | 1302 | 1302 | 0 |
| val | 3618 | 0 | 0 | 0 | 0.000000 | 0 | 0 | 0 |
| test | 3726 | 0 | 0 | 0 | 0.000000 | 0 | 0 | 0 |

The combined p01/p05 = 0 values are therefore explained: the **train** split contains 1 302 exactly-all-black RGB
tiles (12.9630 % of train), each with an existing but **empty** label file, while val and test contain none.

```text
combined percentile results include these all-black images: YES
(no image was excluded from any pre-registered distribution)
```

### 13.3 Split-specific A2 percentile ranks (§7)

| metric | train | val | test | combined |
|---|---:|---:|---:|---:|
| Y_mean | 38.88 | 11.22 | 1.83 | 25.18 |
| Y_std | 13.45 | 0.00 | 0.27 | 7.83 |
| Y_dynamic_98 | 13.60 | 0.00 | 0.27 | 7.91 |
| gradient_mean | 46.00 | 69.18 | 82.45 | 58.63 |
| dark_fraction | 57.94 | 83.31 | 96.81 | 71.54 |
| very_dark_fraction | 7.57 | 6.41 | 23.99 | 10.85 |

### 13.4 A1/A2/A3/A4 nine-tile summaries (§8)

| case | median Y_mean | median Y_std | median Y_dynamic_98 | median gradient_mean | min Y_mean | max Y_mean | min Y_std | max Y_std | min gradient_mean | max gradient_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 93.2671 | 35.9739 | 146.6843 | 5.2512 | 87.3690 | 99.8234 | 33.8467 | 49.3272 | 4.6490 | 6.4085 |
| A2 | 65.3259 | 6.9624 | 33.2340 | 4.7770 | 64.9438 | 66.5203 | 4.7920 | 7.2442 | 3.1844 | 4.8427 |
| A3 | 83.3839 | 34.1799 | 178.3190 | 5.3521 | 77.7171 | 86.3261 | 23.6443 | 41.2171 | 4.2942 | 5.8064 |
| A4 | 76.9564 | 28.8475 | 147.1644 | 3.9934 | 65.7461 | 79.8907 | 18.8375 | 33.3409 | 2.7338 | 4.5625 |

### 13.5 A2 tile counts against the combined distribution (§9)

| count | tiles |
|---|---:|
| tiles_below_p01_Y_mean | 0 |
| tiles_below_p05_Y_mean | 0 |
| tiles_below_p01_Y_std | 0 |
| tiles_below_p05_Y_std | 0 |
| tiles_below_p01_Y_dynamic_98 | 0 |
| tiles_below_p05_Y_dynamic_98 | 0 |
| tiles_below_p01_gradient_mean | 0 |
| tiles_below_p05_gradient_mean | 0 |
| tiles_above_p95_dark_fraction | 0 |
| tiles_above_p99_dark_fraction | 0 |
| tiles_above_p95_very_dark_fraction | 0 |
| tiles_above_p99_very_dark_fraction | 0 |

All twelve required counts are zero; they are reported even though they are zero.

### 13.6 Successful-control ratios (§10)

| metric | A2 whole | median(A1,A3,A4) whole | whole_ratio | A2 lower than all three (whole) | A2 median tile | median(A1,A3,A4) median tile | tile_ratio | A2 lower than all three (tile) |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Y_mean | 65.6886 | 81.8057 | 0.8030 | True | 65.3259 | 83.3839 | 0.7834 | True |
| Y_std | 6.3262 | 32.0700 | 0.1973 | True | 6.9624 | 34.1799 | 0.2037 | True |
| Y_dynamic_98 | 31.2880 | 179.1770 | 0.1746 | True | 33.2340 | 147.1644 | 0.2258 | True |
| gradient_mean | 4.1943 | 4.9596 | 0.8457 | False | 4.7770 | 5.2512 | 0.9097 | False |

A2 is lower than all three successful controls for `Y_mean`, `Y_std` and `Y_dynamic_98` at both whole-image and
median-tile level, but **not** for `gradient_mean` (its texture statistic is comparable to the controls). Ratios are
descriptive only and are not treated as causal proof.

### 13.7 Encoding metadata evidence and exact enum (§11)

```text
A1/A2/A3/A4: PNG signature valid · 8-bit · RGB (colour_type 2, no alpha) · no interlace · compression 0 · filter 0
             chunk structure IHDR + IDAT… + IEND only
             no sRGB / gAMA / iCCP / tEXt / pHYs or other ancillary profile, gamma, EXIF or text metadata

A2_ENCODING_ANOMALY_NOT_FOUND
```

### 13.8 Re-evaluated primary outcome (original rule retained) and split robustness

Apply the original P1D8 combined-distribution criteria:

```text
A2 whole-image at/above p05 on core metrics = 4/4 (Y_mean, Y_std, Y_dynamic_98, gradient_mean)
PROP01_A2_NOT_PHOTOMETRIC_OUTLIER   (retained, unchanged)
```

Separate robustness statement (does **not** replace the primary outcome):

```text
SENSITIVE_TO_SPLIT_COMPOSITION
```

Per-split positions differ: A2 is at/above p05 for all four core metrics within **train**, but falls **below** p05
for `Y_std` and `Y_dynamic_98` within **val** and **test** (and below p05 for `Y_mean` within test). The combined
placement that the pre-registered rule uses is therefore a composition of split-dependent behaviour rather than a
uniform property, which is why the robustness enum is `SENSITIVE_TO_SPLIT_COMPOSITION`.

### 13.9 Recommendation (unchanged, NOT executed)

```text
NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION
```

### 13.10 Still NOT ESTABLISHED

```text
A2 geographic/sensor/site provenance                                  : NOT ESTABLISHED
A2 membership in the active train/val/test splits                     : NOT ESTABLISHED (no exact hash match)
whether A2's content lies inside the detector's effective domain       : NOT ESTABLISHED
cause of the zero-proposal behaviour                                   : NOT ESTABLISHED (no rescue attempted)
```
