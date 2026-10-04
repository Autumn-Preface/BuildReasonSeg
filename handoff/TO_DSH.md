# TO_DSH — Task 8B.3-P1D8-R1: Complete A2 Input-Domain Evidence Contract

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `b4dc1967cdc6734d35315d29eed11eb284b50244`

# 0. Audit disposition

P1D8 commit:

```text
b4dc1967cdc6734d35315d29eed11eb284b50244
docs(rc1): audit a2 input photometric domain
```

has the correct parent and only documentation/handoff changes.

The current P1D8 conclusion is provisionally supported:

```text
PROP01_A2_NOT_PHOTOMETRIC_OUTLIER
```

because the committed report records, against the combined active task6m image distribution:

```text
A2 whole:
Y_mean         percentile = 25.18
Y_std          percentile = 7.83
Y_dynamic_98   percentile = 7.91
gradient_mean  percentile = 58.63
```

and no obvious PNG encoding anomaly.

However P1D8 is NOT YET APPROVED because required evidence was left only in an untracked JSON or omitted from the report.

This R1 is evidence completion only.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = b4dc1967cdc6734d35315d29eed11eb284b50244
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any detector/model/forward/predict;
- run Qwen/SAM2/D-B1;
- run pytest/check_setup;
- modify any input image;
- perform photometric rescue;
- run CLAHE/gamma/equalization/contrast correction through a detector;
- modify RC1 source/runtime/tests/manifests/checkpoints;
- enter PROP-01 fix implementation;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

Reading the existing P1D8 JSON/statistics is preferred.

If the untracked statistics JSON no longer exists, recomputing the SAME non-model metrics from the SAME frozen images is allowed.
Do not change metric definitions.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d8_a2_input_domain_audit.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do not commit the dataset-sized raw JSON.

# 4. Preserve frozen metric definitions

Use exactly the P1D8 definitions:

```text
Y = 0.299*R + 0.587*G + 0.114*B

Y_mean
Y_std
Y_p01
Y_p50
Y_p99
Y_dynamic_98 = Y_p99 - Y_p01
mean_abs_dx
mean_abs_dy
gradient_mean = (mean_abs_dx + mean_abs_dy)/2
dark_fraction = fraction(Y < 64)
very_dark_fraction = fraction(Y < 32)
bright_fraction = fraction(Y > 192)
```

No normalization before measuring.

# 5. Commit the full training-domain summary tables

The report MUST contain, not merely reference an untracked file, complete tables for:

```text
Y_mean
Y_std
Y_dynamic_98
gradient_mean
dark_fraction
very_dark_fraction
```

For each metric include these four rows:

```text
train
val
test
combined
```

and all columns:

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

Use the already processed population:

```text
train = 10044
val   = 3618
test  = 3726
combined = 17388
unreadable = 0
```

If those counts cannot be reproduced from existing evidence, STOP.

# 6. Audit the zero-valued training tail

The current report shows:

```text
train Y_mean p01 = 0
train Y_mean p05 = 0
combined Y_mean p01 = 0
combined Y_mean p05 = 0
```

This must be characterized because it can materially affect percentile interpretation.

For each split record:

```text
count images with Y_mean == 0
count images with Y_std == 0
count images that are exactly all-black RGB
fraction of split exactly all-black RGB
```

If label files are trivially resolvable by matching stem, also record for exactly-all-black images:

```text
label file exists count
empty label count
non-empty label count
```

Do not perform a semantic visual review.

Do not exclude these images from any pre-registered distribution.

State explicitly whether the combined percentile results include these images: YES.

# 7. Split-specific A2 percentile robustness

For A2 whole-image metrics compute deterministic percentile rank separately within:

```text
train
val
test
combined
```

for:

```text
Y_mean
Y_std
Y_dynamic_98
gradient_mean
dark_fraction
very_dark_fraction
```

Add one table.

This is a robustness audit only.

Do NOT change the original pre-registered P1D8 outcome rule, which used the combined active distribution.

# 8. Required A1/A2/A3/A4 tile summaries

For each case A1/A2/A3/A4, using the same frozen 9-tile geometry, report:

```text
median tile Y_mean
median tile Y_std
median tile Y_dynamic_98
median tile gradient_mean

min tile Y_mean
max tile Y_mean
min tile Y_std
max tile Y_std
min tile gradient_mean
max tile gradient_mean
```

One table with four case rows is sufficient.

No model run.

# 9. Required A2 p01/p05 tile counts

Against the COMBINED active training distribution, record A2's 9-tile counts:

```text
tiles below p01 Y_mean
tiles below p05 Y_mean

tiles below p01 Y_std
tiles below p05 Y_std

tiles below p01 Y_dynamic_98
tiles below p05 Y_dynamic_98

tiles below p01 gradient_mean
tiles below p05 gradient_mean

tiles above p95 dark_fraction
tiles above p99 dark_fraction

tiles above p95 very_dark_fraction
tiles above p99 very_dark_fraction
```

These are required even if all counts are zero.

# 10. Required successful-control ratios

Let successful controls be:

```text
A1, A3, A4
```

For each core metric:

```text
Y_mean
Y_std
Y_dynamic_98
gradient_mean
```

record:

```text
whole_ratio =
A2 whole value / median(A1 whole, A3 whole, A4 whole)

tile_ratio =
A2 median-tile value /
median(A1 median-tile, A3 median-tile, A4 median-tile)
```

Also record whether A2 is lower than ALL THREE successful controls for each metric at:
- whole-image level;
- median-tile level.

Do not interpret ratio as causal proof.

# 11. Normalize the encoding enum

Replace/augment the informal:

```text
encoding anomaly found = NONE
```

with exactly one required enum:

```text
A2_ENCODING_ANOMALY_FOUND
A2_ENCODING_ANOMALY_NOT_FOUND
A2_ENCODING_METADATA_INCONCLUSIVE
```

If existing evidence remains:

```text
all A1/A2/A3/A4:
PNG
8-bit RGB
no alpha
same core PNG chunk structure
no anomalous ancillary profile/gamma/EXIF/text metadata
```

then use:

```text
A2_ENCODING_ANOMALY_NOT_FOUND
```

only if those facts are actually supported.

# 12. Re-evaluate the P1D8 outcome without changing its rule

Choose exactly one again:

```text
PROP01_A2_PHOTOMETRIC_OUTLIER_STRONGLY_SUPPORTED
PROP01_A2_DARKNESS_OUTLIER_ONLY
PROP01_A2_NOT_PHOTOMETRIC_OUTLIER
PROP01_A2_INPUT_DOMAIN_INCONCLUSIVE
```

Apply the ORIGINAL P1D8 criteria exactly.

Do NOT create a new threshold or use split-specific robustness tables to retroactively redefine the rule.

If the original combined-distribution criterion still gives:

```text
PROP01_A2_NOT_PHOTOMETRIC_OUTLIER
```

retain it.

Separately state whether split-specific analysis makes the interpretation:

```text
ROBUST_ACROSS_SPLITS
SENSITIVE_TO_SPLIT_COMPOSITION
INCONCLUSIVE_ACROSS_SPLITS
```

This robustness enum does not replace the primary outcome.

# 13. Re-evaluate next gate

If primary outcome remains:

```text
PROP01_A2_NOT_PHOTOMETRIC_OUTLIER
```

then keep:

```text
NEXT = DETECTOR_ADAPTATION_OR_DEMO_POLICY_DECISION
```

Otherwise follow the original P1D8 decision table.

Do NOT execute it.

# 14. Repair report

Update:

```text
docs/task8b3_p1d8_a2_input_domain_audit.md
```

The committed report must now contain:

1. original task scope and HEAD
2. P1D2–P1D7 frozen detector evidence
3. input identities
4. dataset roots/counts/dimensions
5. exact hash search
6. fixed metric definitions
7. full six-metric × four-split distribution tables
8. all-black-tail audit
9. A2 split-specific percentile table
10. whole A1/A2/A3/A4 metrics
11. 9-tile summary table for all four cases
12. A2 p01/p05 tile-count table
13. successful-control ratio table
14. encoding metadata evidence
15. exact encoding enum
16. exact primary outcome
17. split robustness enum
18. explicit limits
19. exact next gate
20. model inference = NONE
21. image modification = NONE
22. MEM-01 CLOSED
23. PROP-01 OPEN
24. REF-01/MASK-01 OPEN untouched.

# 15. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required fields:

```text
Task: 8B.3-P1D8-R1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: b4dc1967cdc6734d35315d29eed11eb284b50244
Original P1D8 starting HEAD: 5f52338b7507cc377190cacb33611c644f223a88
Model inference: NONE
Image modifications: NONE
Functional files modified: NO
Training image counts: 10044 / 3618 / 3726 / 17388
Full training distribution tables: RECORDED / NOT RECORDED
All-black-tail audit: RECORDED / NOT RECORDED
A2 split-specific percentiles: RECORDED / NOT RECORDED
All-control tile summary: RECORDED / NOT RECORDED
A2 p01/p05 tile counts: RECORDED / NOT RECORDED
Successful-control ratios: RECORDED / NOT RECORDED
Encoding audit: <exact enum>
Primary outcome: <exact enum>
Split robustness: <exact enum>
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d8_a2_input_domain_audit.md
Next action: Awaiting ChatGPT audit; no rescue/detector change authorized.
```

# 16. Commit / push

If COMPLETE:

```text
docs(rc1): complete a2 input-domain audit
```

If STOP/FAILED:

```text
docs(rc1): record a2 input-domain audit correction stop
```

Push normally. No force push.

# 17. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model execution;
- no image modification;
- all six training-domain distribution tables are committed in the report;
- zero-valued training tail is quantified;
- split-specific A2 percentile robustness is recorded;
- all four Demo cases have tile summaries;
- A2 p01/p05 tile counts are explicit;
- successful-control ratios are explicit;
- encoding enum is exact;
- original P1D8 outcome is re-evaluated without changing the rule;
- split robustness enum is reported separately;
- next gate is recommended but not run;
- only allowed files changed;
- commit/push succeeds;
- STOP.

# 18. Final response

```text
TASK 8B.3-P1D8-R1 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Model inference:
NONE

Full distribution tables:
RECORDED / NOT RECORDED

All-black tail:
RECORDED / NOT RECORDED

Split robustness:
<enum>

A2 p01/p05 tile counts:
RECORDED / NOT RECORDED

Control ratios:
RECORDED / NOT RECORDED

Encoding:
<enum>

Primary outcome:
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

等待 ChatGPT 审核；不得做图像增强、不得重跑 detector、不得执行 next gate。
```
