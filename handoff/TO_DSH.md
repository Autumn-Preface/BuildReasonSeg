# TO_DSH — Task 8B.3-P1D5-R1: Complete Detector Provenance Evidence Contract

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `d17517b53b1cb9b82cbecc0432a5023eaa756278`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Audit disposition

P1D5 commit:

```text
d17517b53b1cb9b82cbecc0432a5023eaa756278
docs(rc1): audit detector provenance for a2
```

has the correct parent and modifies only the three authorized documentation/handoff paths.

Its central evidence is plausible and currently retained:

```text
active detector:
  YOLO26m-seg
  54,480,241 B
  sha256 ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474
  documented source = task6m1 continued best.pt

training lineage:
  WHU building dataset

A2 source/domain:
  NOT ESTABLISHED

scientific-freeze classification:
  DETECTOR_CHANGE_TOUCHES_FROZEN_RESEARCH_CLAIMS

primary diagnosis:
  PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE

recommended next gate:
  DOMAIN_EVIDENCE_RECOVERY
```

However P1D5 is NOT YET APPROVED because the required evidence contract was incomplete.

The missing/insufficient items are:

```text
1. no exact baseline/yolo_whu relation enum was reported;
2. no active detector class names/count were established or explicitly marked NOT ESTABLISHED;
3. the task required direct A2 dtype/bit-depth/channel statistics, but they were omitted;
4. the required training-domain-vs-A2 comparison table was omitted;
5. stage-1/stage-2 args were identified but not actually parsed for available training settings;
6. alternate candidates were labeled VALIDATED_ALTERNATE_AVAILABLE, but validation evidence was not normalized
   candidate-by-candidate; pretrained/smoke-only artifacts must not be conflated with validated alternatives;
7. FROM_DSH omitted several exact required fields from the P1D5 task contract.
```

This task only repairs those evidence/reporting gaps.

No new scientific or engineering intervention is authorized.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = d17517b53b1cb9b82cbecc0432a5023eaa756278
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any model inference or forward call;
- call YOLO `.predict()` or detector runtime methods;
- run `predict.py`;
- run `--inspect-proposals`;
- run A1/A2/A3/A4/B1/B2 inference;
- run Qwen/SAM2/D-B1;
- run pytest/check_setup;
- train/fine-tune/resume/export;
- download anything;
- replace/copy/modify any checkpoint;
- modify external delivery;
- modify runtime/tests/manifests/config/model package;
- change thresholds;
- implement a detector fallback;
- run an alternate detector;
- enter DOMAIN_EVIDENCE_RECOVERY beyond this report repair;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

Reading text/YAML/JSON/CSV, hashing files, and reading image pixels for non-model descriptive statistics are allowed.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d5_detector_provenance_domain_gap.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No functional path may change.

# 4. Freeze already-accepted P1D5 facts unless contradicted by direct evidence

Retain these unless a directly inspected artifact proves them wrong:

```text
active checkpoint bytes =
54480241

active checkpoint sha256 =
ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474

active detector family =
YOLO26m-seg

active task =
instance segmentation

documented source =
artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/weights/best.pt

A2 source/domain =
NOT ESTABLISHED

MEM-01 =
CLOSED

PROP-01 =
OPEN

REF-01 =
OPEN

MASK-01 =
OPEN
```

Do not rerun P1D1–P1D4.

# 5. Establish active detector class names/count

Inspect authoritative local evidence in this order:

1. `artifacts/task6m_yolo_native/data.yaml`;
2. stage-1/stage-2 training args;
3. model/package metadata;
4. safe checkpoint metadata only if still necessary.

Record:

```text
class_count = <integer or NOT ESTABLISHED>
class_names = <exact list/mapping or NOT ESTABLISHED>
```

If the data YAML says one class named `building`, record that exact fact and cite the source path in the report.

Do not infer class count only from the general phrase "WHU building dataset".

# 6. Parse available stage-1 / stage-2 training settings

Inspect these if present:

```text
artifacts/checkpoints/task6m/runs/m1_yolo26m_seg/args.yaml
artifacts/checkpoints/task6m1/runs/m1_yolo26m_seg_continued/args.yaml
```

and any directly referenced frozen run record needed to interpret them.

Record separately for stage 1 and stage 2, when present:

```text
model/source checkpoint
data
epochs
imgsz
batch
device
seed
resume
pretrained
optimizer
close_mosaic
patience
save_dir/run identity
```

For each unavailable field write `NOT ESTABLISHED`.

Do NOT leave a present readable field unparsed merely because it is not central to the final diagnosis.

# 7. Classify baseline/yolo_whu relation exactly

Inspect:

```text
baseline/yolo_whu/README.md
baseline/yolo_whu/manifest.json
baseline/yolo_whu/run_record/args.yaml
```

and the active detector provenance.

Choose exactly one:

```text
SAME_CHECKPOINT
DIRECT_ANCESTOR_OR_FINETUNE_SOURCE
SEPARATE_HISTORICAL_BASELINE
RELATION_NOT_ESTABLISHED
```

Rules:

- Different family + different checkpoint hash + no direct training ancestry evidence => cannot be SAME_CHECKPOINT.
- `DIRECT_ANCESTOR_OR_FINETUNE_SOURCE` requires explicit lineage evidence, not both being trained on WHU.
- If the record is the independent YOLOv8m-seg WHU baseline and active is YOLO26m-seg with separate lineage, use
  `SEPARATE_HISTORICAL_BASELINE`.

Record the exact evidence.

# 8. Compute the required non-model A2 descriptive statistics

Using the frozen:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\A2.png
```

First re-confirm:

```text
1024×1024
RGB
sha256 = 10286b1e76db9e38c474635a465c9e677dbcf58375c1d39f7b742eeb991f434f
```

Then read pixels only; no model call.

Record:

```text
file bytes
array dtype
bit depth / channel representation if directly established
R min / max / mean / std
G min / max / mean / std
B min / max / mean / std
R fraction == 0
G fraction == 0
B fraction == 0
R fraction == 255
G fraction == 255
B fraction == 255
```

Use a stated numeric precision and keep it consistent.

These are descriptive facts only.

Do NOT interpret ordinary pixel-statistic differences as proof of domain gap.

# 9. Add the required training-domain vs A2 table

The report MUST contain this table with no omitted rows:

| property | active detector training/validation domain | A2 |
|---|---|---|
| dataset/source | | |
| task/class definition | | |
| native image size | | |
| training crop/tile size | | |
| RGB/bit depth | | |
| known spatial resolution/GSD | | |
| known augmentation/resizing | | |
| building morphology/context | | |

Rules:
- use only documented facts;
- use `NOT ESTABLISHED` for unknowns;
- distinguish detector training `imgsz` from source/native tile dimensions;
- do not fabricate GSD;
- do not visually invent morphology/context.

# 10. Normalize alternate-candidate validation status

Re-audit each previously listed candidate separately:

```text
A. YOLO26m-seg epoch-18 / task6m best.pt
B. YOLOv8m-seg WHU baseline best.pt
C. YOLO26m-seg pretrained base
D. YOLO26s-seg pretrained/smoke artifact
```

For each record:

```text
candidate path
family/task
hash if available
training/fine-tuning status
quantitative validation evidence = exact metric/report OR NONE
prior pipeline/demo-use evidence = exact artifact/report OR NONE
A2 validation = YES/NO
RC1-path validation = YES/NO
candidate status =
    VALIDATED_PROJECT_ALTERNATE
    USED_BUT_NOT_QUANTITATIVELY_VALIDATED
    PRETRAINED_BASE_ONLY
    SMOKE_ONLY
    INCOMPATIBLE_OR_OTHER
```

Important:
- being a pretrained base is NOT project validation;
- being used in a smoke run is NOT project validation;
- being referenced by a script is NOT quantitative validation;
- the frozen YOLOv8m baseline's validation-split metrics may count as project validation if the frozen record actually
  contains those metrics;
- A2/RC1 validation must be reported separately from generic WHU validation.

Then choose the overall enum exactly:

```text
VALIDATED_ALTERNATE_AVAILABLE
NO_VALIDATED_ALTERNATE_FOUND
ALTERNATE_STATUS_NOT_ESTABLISHED
```

`VALIDATED_ALTERNATE_AVAILABLE` is allowed if at least one non-active candidate has real prior project validation
evidence, even if it has never been run on A2/RC1. That does NOT authorize adopting it into RC1.

# 11. Clarify scientific-freeze impact vs diagnostic use

Retain or revise the exact enum:

```text
DETECTOR_CHANGE_ENGINEERING_ONLY_IF_REVALIDATED
DETECTOR_CHANGE_TOUCHES_FROZEN_RESEARCH_CLAIMS
SCIENTIFIC_FREEZE_IMPACT_NOT_ESTABLISHED
```

Then add two separate statements:

```text
A. adopting/replacing the RC1 proposal detector would:
   <effect on frozen claims>

B. merely running a future isolated alternate-detector A2 diagnostic, without changing RC1 or research claims, would:
   <ALLOWED_AS_SEPARATE_DIAGNOSTIC / NOT_ALLOWED_BY_FREEZE / NOT_ESTABLISHED>
```

This distinction is mandatory.

Do not run such a diagnostic in this task.

# 12. Re-evaluate primary diagnosis

Choose exactly one again:

```text
PROP01_DETECTOR_DOMAIN_GAP_STRONGLY_SUPPORTED
PROP01_SINGLE_IMAGE_BLINDSPOT_WITHIN_EXPECTED_DOMAIN
PROP01_CHECKPOINT_PROVENANCE_INCOMPLETE
PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE
```

Given the currently known evidence, do NOT strengthen beyond `PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE` unless new
direct provenance evidence establishes A2's domain.

# 13. Re-evaluate next gate

Apply the original decision logic only after sections 5–12 are complete.

Record exactly one:

```text
CONTROLLED_ALTERNATE_DETECTOR_A2_PROBE
DETECTOR_ADAPTATION_DESIGN
DETECTOR_ROBUSTNESS_DECISION
CHECKPOINT_PROVENANCE_RECOVERY
DOMAIN_EVIDENCE_RECOVERY
```

Important interpretation:

- a validated alternate existing does NOT automatically authorize a product swap;
- if a separate diagnostic probe is scientifically permissible but the original P1D5 rule still routes to
  `DOMAIN_EVIDENCE_RECOVERY`, preserve the original rule outcome and explicitly state that the diagnostic candidate
  remains available for a later ChatGPT decision;
- do not execute any gate.

# 14. Repair the report

Update:

```text
docs/task8b3_p1d5_detector_provenance_domain_gap.md
```

It must now include all of:

1. Task/scope/starting HEAD of original P1D5 and R1
2. P1D1–P1D4 frozen evidence
3. active detector path/bytes/SHA256
4. active family/task/classes
5. U-C1 selection history
6. stage-1/stage-2 training provenance and parsed args
7. validation/selection evidence
8. exact `baseline/yolo_whu` relation enum
9. checkpoint metadata method/evidence
10. A2 provenance
11. A2 required pixel statistics
12. exact training-domain-vs-A2 comparison table
13. alternate candidates with normalized status
14. overall alternate availability enum
15. scientific-freeze enum
16. adoption-vs-diagnostic distinction
17. primary diagnosis enum
18. evidence gaps
19. exact next gate
20. `RC1-DEMO-PROP-01 = OPEN`
21. `RC1-DEMO-MEM-01 = CLOSED`
22. REF-01/MASK-01 OPEN untouched
23. model inference/training = NONE
24. functional modifications = NONE.

# 15. Repair FROM_DSH

Preserve ARTIFACT-FACTS byte-for-byte.
UTF-8 without BOM.

Required exact fields:

```text
Task: 8B.3-P1D5-R1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: d17517b53b1cb9b82cbecc0432a5023eaa756278
Original P1D5 HEAD: 15a90ec11ad66c8f365284bf53eb320661a1e73f
Model inference/training: NONE
Functional files modified: NO
Active detector SHA256: ef852b5801e6bdf902ddc581ada6f04a5673deecba092f3b2c24c0efa861f474
Active detector family/task: <value>
Active detector classes: <value / NOT ESTABLISHED>
Training/fine-tuning dataset: <value / NOT ESTABLISHED>
Stage-1 args: <summary>
Stage-2 args: <summary>
U-C1 selection evidence: ESTABLISHED / PARTIAL / NOT ESTABLISHED
baseline/yolo_whu relation: <exact enum>
A2 source/domain: <value / NOT ESTABLISHED>
A2 descriptive stats: RECORDED / NOT RECORDED
Training-vs-A2 comparison table: RECORDED / NOT RECORDED
Validated alternate candidate: <overall enum>
Alternate diagnostic use under freeze: <ALLOWED_AS_SEPARATE_DIAGNOSTIC / NOT_ALLOWED_BY_FREEZE / NOT_ESTABLISHED>
Scientific-freeze impact: <exact enum>
Primary diagnosis: <exact enum>
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d5_detector_provenance_domain_gap.md
Next action: Awaiting ChatGPT audit; no detector change or next gate authorized.
```

# 16. Commit / push

If COMPLETE:

```text
docs(rc1): complete detector provenance audit
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record detector provenance audit correction stop
```

Push current branch normally.
No force push.

# 17. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model inference/training/tests;
- no external/functional modifications;
- all missing P1D5 evidence items are repaired;
- baseline relation has one exact enum;
- detector classes are established or explicitly NOT ESTABLISHED;
- readable active training args are parsed;
- A2 required pixel statistics are recorded;
- the full training-vs-A2 table is present;
- alternate validation status is normalized candidate-by-candidate;
- adoption vs separate diagnostic is distinguished;
- primary diagnosis and next gate are explicitly re-evaluated;
- report and FROM_DSH are updated;
- commit and push succeed;
- tracked tree is clean;
- STOP.

# 18. Final response

```text
TASK 8B.3-P1D5-R1 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Model inference/training:
NONE

Functional files modified:
NO

Active detector classes:
<...>

baseline/yolo_whu:
<exact relation enum>

A2 descriptive stats:
RECORDED / NOT RECORDED

Training-vs-A2 table:
RECORDED / NOT RECORDED

Validated alternate:
<overall enum>

Alternate diagnostic use:
<enum>

Scientific-freeze impact:
<enum>

Primary diagnosis:
<enum>

Next gate:
<enum>

MEM-01:
CLOSED

PROP-01:
OPEN

REF-01 / MASK-01:
OPEN / OPEN

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得运行任何 detector，不得替换 checkpoint，不得执行 next gate。
```
