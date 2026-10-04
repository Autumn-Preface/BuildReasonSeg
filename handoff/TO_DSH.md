# TO_DSH — Task 8B.3-P1D8: A2 Input Photometric / Domain Audit

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `5f52338b7507cc377190cacb33611c644f223a88`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

P1D2–P1D7 have established:

```text
active continued YOLO26m-seg:
  A2 tiled @ conf=0.05  → 9/9 boxes=0
  A2 full-frame @ 0.05  → boxes=0
  A2 tiled @ conf=0.001 → 9/9 boxes=0

same-lineage epoch-18 YOLO26m-seg:
  A2 tiled @ 0.05 → 9/9 boxes=0

independent WHU-trained YOLOv8m-seg baseline:
  A2 tiled @ 0.05 → 9/9 boxes=0
```

P1D7 outcome:

```text
PROP01_YOLOV8M_ALSO_ZERO
```

This materially weakens checkpoint-specific explanations.

P1D5-R1 also established A2 descriptive facts:

```text
1024x1024 RGB uint8
R mean/std = 60.0654 / 6.3738
G mean/std = 68.4044 / 6.4936
B mean/std = 66.4532 / 6.9203
no pixels at 0 or 255
A2 source/domain = NOT ESTABLISHED
```

The remaining question is whether A2 is a strong *input photometric/texture outlier* relative to:
1. the detector's actual WHU training/validation image domain; and
2. the successful 1024×1024 Demo controls A1/A3/A4.

This task is NON-MODEL forensic analysis only.

No detector inference or image correction is authorized.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 5f52338b7507cc377190cacb33611c644f223a88
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any detector or model;
- call YOLO `.predict()` or forward;
- run Qwen/SAM2/D-B1;
- run normal `predict.py` or `--inspect-proposals`;
- run pytest/check_setup;
- train/fine-tune/download/export;
- modify any image;
- overwrite A2/A1/A3/A4;
- create a corrected A2;
- run histogram equalization/CLAHE/gamma/contrast stretch as a product action;
- test any normalization through a detector;
- change RC1 source/config/model/checkpoint;
- modify external delivery;
- modify manifests/tests/runtime;
- remove or replace A2 from the Demo;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

Read-only image decoding and arithmetic image statistics are allowed.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d8_a2_input_domain_audit.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Temporary CSV/JSON/statistics scripts must remain outside the repo or under ignored diagnostic storage.
Do not commit a dataset-sized statistics dump.

# 4. Input identity gate

Verify frozen Demo inputs before analysis.

Require A2:

```text
A2.png
1024x1024 RGB
sha256 =
10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
```

Locate A1/A3/A4 used by Task 8B.3 R4B and record:

```text
path
dimensions
mode
file bytes
SHA256
```

Do not use a replacement or similarly named copy.

If A2 identity mismatches:
- STOP.

If one of A1/A3/A4 cannot be uniquely identified:
- record that control as unavailable;
- continue only if at least TWO of A1/A3/A4 are available.
Otherwise STOP.

# 5. Training-domain image source gate

Inspect:

```text
artifacts/task6m_yolo_native/data.yaml
```

Resolve the exact train/val/test image roots used by the active YOLO26 training lineage.

Record:

```text
train image root
val image root
test image root
file counts
image formats
native dimensions distribution
```

Do not silently substitute `baseline/yolo_whu` image paths if the active task6m dataset is available.

If the active training images are unavailable locally:
- do NOT use a different dataset as if equivalent;
- classify training-domain statistics as unavailable;
- continue with Demo-control comparison only;
- final outcome must not claim a WHU training-domain outlier.

# 6. Exact-match provenance search

Without modifying files, compute/compare SHA256 for image files in the active task6m train/val/test image roots.

Search for exact byte/hash match to A2.

Record exactly one:

```text
A2_EXACT_MATCH_TRAIN
A2_EXACT_MATCH_VAL
A2_EXACT_MATCH_TEST
A2_NO_EXACT_MATCH
TRAINING_IMAGES_UNAVAILABLE
```

If exact match exists:
- record exact path and split;
- do not infer more than exact file identity.

Also search exact SHA256 for A1/A3/A4 and record matches if any.

# 7. Frozen metric definitions

Use these exact non-model metrics for every decoded RGB image/tile.

Let luminance:

```text
Y = 0.299*R + 0.587*G + 0.114*B
```

For each image/tile calculate:

```text
R_mean, G_mean, B_mean
R_std,  G_std,  B_std

Y_mean
Y_std
Y_p01
Y_p50
Y_p99
Y_dynamic_98 = Y_p99 - Y_p01

mean_abs_dx =
mean(abs(Y[:,1:] - Y[:,:-1]))

mean_abs_dy =
mean(abs(Y[1:,:] - Y[:-1,:]))

gradient_mean =
(mean_abs_dx + mean_abs_dy) / 2

dark_fraction =
fraction(Y < 64)

very_dark_fraction =
fraction(Y < 32)

bright_fraction =
fraction(Y > 192)
```

Use decoded uint8 RGB values converted to float only for arithmetic.
Do not normalize before measuring.

Report calculations to at least 4 decimal places for means/std/gradient/fractions.

# 8. Active training-domain distribution

If active task6m train/val/test images are available:

Process ALL readable images in the resolved active training dataset.

Do not random-sample unless full processing is impossible due a documented execution failure.
If full processing fails:
- STOP before drawing a training-domain percentile conclusion.

For each metric below construct distributions:

```text
Y_mean
Y_std
Y_dynamic_98
gradient_mean
dark_fraction
very_dark_fraction
```

Report for each split individually AND for the combined dataset:

```text
N
min
p01
p05
p25
p50
p75
p95
p99
max
mean
std
```

Record unreadable-image count.

No model execution.

# 9. Demo-case whole-image comparison

Compute the same frozen metrics for whole images:

```text
A1
A2
A3
A4
```

Build one comparison table.

For A2 and each available control, if training-domain stats exist, report percentile rank within the COMBINED active
training image distribution for:

```text
Y_mean
Y_std
Y_dynamic_98
gradient_mean
dark_fraction
very_dark_fraction
```

Percentile-rank method must be stated and deterministic.

# 10. Demo-case tile comparison

Using the existing 1024×1024 9-tile geometry:

```text
tile size = 512
tops  = [0, 384, 512]
lefts = [0, 384, 512]
```

compute the same metrics for all 9 tiles of:

```text
A1
A2
A3
A4
```

No detector call.

For each case report:

```text
median tile Y_mean
median tile Y_std
median tile Y_dynamic_98
median tile gradient_mean

min/max tile Y_mean
min/max tile Y_std
min/max tile gradient_mean
```

If training distribution exists, report for A2:

```text
number of its 9 tiles below training p01 / p05 for:
Y_mean
Y_std
Y_dynamic_98
gradient_mean

number above training p95 / p99 for:
dark_fraction
very_dark_fraction
```

If the active training source consists of 512×512 tiles, compare A2 tile metrics directly to that distribution and
state that geometry alignment.

# 11. Compare with successful controls

Historical detector evidence:

```text
A1 raw = 133
A3 raw = 7
A4 raw = 216
A2 raw = 0
```

These counts are historical evidence only; do not rerun models.

For each core metric:

```text
Y_mean
Y_std
Y_dynamic_98
gradient_mean
```

determine whether A2 is lower than ALL available successful controls.

Also record the ratio:

```text
A2 value / median(A1,A3,A4)
```

for the whole-image metric and median-tile metric.

Do not call this causal proof.

# 12. PNG / encoding metadata audit

Read non-model metadata for A2 and controls:

```text
format
mode
bit depth if available
ICC profile present?
gamma/chromaticity metadata if available?
transparency/alpha?
EXIF present?
PNG textual metadata?
```

Record only what is actually present.

Specifically check whether A2 has an obvious encoding anomaly relative to A1/A3/A4.

Choose one:

```text
A2_ENCODING_ANOMALY_FOUND
A2_ENCODING_ANOMALY_NOT_FOUND
A2_ENCODING_METADATA_INCONCLUSIVE
```

Do NOT edit metadata.

# 13. Outcome classification

Choose exactly ONE.

## A — `PROP01_A2_PHOTOMETRIC_OUTLIER_STRONGLY_SUPPORTED`

Use only if:
- active training-domain statistics are available;
- A2 is a strong outlier in at least TWO of:
  - Y_std
  - Y_dynamic_98
  - gradient_mean
- for each qualifying metric, A2 whole-image OR at least 7/9 A2 tiles lies below the active training p01;
- and A2 is lower than all available successful controls on the same metrics.

This supports photometric/texture incompatibility as a major suspect.
It does NOT prove geographic domain mismatch.

## B — `PROP01_A2_DARKNESS_OUTLIER_ONLY`

Use if:
- A2 Y_mean/dark_fraction is extreme versus training,
- but contrast/texture metrics do NOT satisfy outcome A.

Interpretation:
brightness is unusual, but low information/contrast is not strongly established.

## C — `PROP01_A2_NOT_PHOTOMETRIC_OUTLIER`

Use only if:
- training-domain statistics exist;
- A2 lies at or above p05 for at least 3 of the 4 core metrics:
  Y_mean, Y_std, Y_dynamic_98, gradient_mean;
- and no encoding anomaly is found.

## D — `PROP01_A2_INPUT_DOMAIN_INCONCLUSIVE`

Use if:
- training images are unavailable;
- control evidence is contradictory;
- or criteria A/B/C are not met.

# 14. Next-gate recommendation — DO NOT EXECUTE

If outcome A:

```text
NEXT = CONTROLLED_A2_PHOTOMETRIC_RESCUE_DESIGN
```

Future task designs one deterministic, training-domain-derived normalization transform.
No model run in this task.

If outcome B:

```text
NEXT = A2_BRIGHTNESS_RESCUE_DESIGN
```

If outcome C:

```text
NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION
```

If outcome D:

```text
NEXT = A2_INPUT_DOMAIN_POLICY_DECISION
```

Do not execute next gate.

# 15. Report

Create:

```text
docs/task8b3_p1d8_a2_input_domain_audit.md
```

Required sections:

1. scope / starting HEAD
2. P1D2–P1D7 frozen detector facts
3. Demo image identity table
4. active task6m dataset roots/counts/dimensions
5. exact-match SHA provenance search
6. metric definitions
7. full active training-domain statistics
8. whole-image A1/A2/A3/A4 comparison
9. percentile-rank table
10. 9-tile comparison
11. A2 vs successful controls ratios
12. PNG/encoding metadata audit
13. exact encoding enum
14. exact outcome enum
15. causal interpretation and explicit limits
16. exact next gate
17. model inference = NONE
18. image modifications = NONE
19. functional files modified = NO
20. MEM-01 CLOSED
21. PROP-01 OPEN
22. REF-01/MASK-01 OPEN untouched.

# 16. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D8
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 5f52338b7507cc377190cacb33611c644f223a88
Model inference: NONE
Image modifications: NONE
Functional files modified: NO
A2 identity: MATCH / MISMATCH
Available successful controls: <A1/A3/A4 subset>
Active training images: AVAILABLE / UNAVAILABLE
Training image count: <n / NOT ESTABLISHED>
A2 exact-match provenance: <enum>
A2 whole Y_mean: <value>
A2 whole Y_std: <value>
A2 whole Y_dynamic_98: <value>
A2 whole gradient_mean: <value>
A2 training percentile Y_mean: <value / NA>
A2 training percentile Y_std: <value / NA>
A2 training percentile Y_dynamic_98: <value / NA>
A2 training percentile gradient_mean: <value / NA>
A2 tiles below p01 Y_std: <n / NA>
A2 tiles below p01 Y_dynamic_98: <n / NA>
A2 tiles below p01 gradient_mean: <n / NA>
Encoding audit: <exact enum>
Outcome: <exact enum>
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d8_a2_input_domain_audit.md
Next action: Awaiting ChatGPT audit; no rescue or detector change authorized.
```

# 17. Commit / push

If COMPLETE:

```text
docs(rc1): audit a2 input photometric domain
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record a2 input-domain audit stop
```

Push current branch normally.
No force push.

# 18. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model execution;
- A2 identity matches;
- at least two successful controls are available;
- active task6m image roots are resolved or explicitly unavailable;
- exact-match provenance search performed;
- all frozen metrics computed for A2 and controls;
- if active images are available, ALL readable training-domain images are processed;
- split and combined distribution quantiles recorded;
- A2 whole/tile percentile comparison recorded;
- encoding metadata audited;
- one exact outcome chosen;
- one exact next gate recommended but not executed;
- no image/product/runtime modification;
- only allowed docs/handoff files changed;
- commit/push succeeds;
- tracked tree clean;
- STOP.

# 19. Final response

```text
TASK 8B.3-P1D8 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Model inference:
NONE

A2 exact-match provenance:
<enum>

Training images:
<AVAILABLE/UNAVAILABLE> · N=<...>

A2:
Y_mean = <...>
Y_std = <...>
Y_dynamic_98 = <...>
gradient_mean = <...>

Training percentiles:
Y_mean = <...>
Y_std = <...>
Y_dynamic_98 = <...>
gradient_mean = <...>

A2 tiles below p01:
Y_std = <n>
Y_dynamic_98 = <n>
gradient_mean = <n>

Encoding audit:
<enum>

Outcome:
<enum>

Next gate:
<enum>

Image modifications:
NONE

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

等待 ChatGPT 审核；不得做图像增强、不得重跑 detector、不得执行 next gate。
```
