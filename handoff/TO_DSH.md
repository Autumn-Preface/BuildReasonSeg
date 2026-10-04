# TO_DSH — Task 8B.3-P1D5: Detector Provenance and A2 Domain-Gap Forensics

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `15a90ec11ad66c8f365284bf53eb320661a1e73f`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

P1D1–P1D4 have excluded these explanations for A2 zero proposals:

```text
NMS warning erasing returned detections             = EXCLUDED
wrapper masks=None hiding returned boxes             = EXCLUDED by direct P1D2 measurement
512-px tiling/context loss at conf=0.05              = EXCLUDED by P1D3
product threshold merely being slightly too high     = EXCLUDED down to conf=0.001
```

Frozen evidence:

```text
A2 = 1024×1024 RGB
P1D2: 9 tiled calls @ conf=0.05 → 9/9 boxes_count=0
P1D3: one full-frame call @ conf=0.05 → boxes_count=0
P1D4: 9 tiled calls @ conf=0.001 → 9/9 boxes_count=0
P1D4 outcome = PROP01_NO_MEANINGFUL_SUBTHRESHOLD_SIGNAL
```

Before any detector replacement, retraining, fine-tuning, ensemble, preprocessing rescue, or fallback proposal source
is authorized, establish the provenance and validated scope of the actual RC1 `detector.pt`.

This task is READ-ONLY FORENSICS. No detector/model inference is allowed.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 15a90ec11ad66c8f365284bf53eb320661a1e73f
```

Allowed tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run `predict.py` or `--inspect-proposals`;
- call YOLO/model forward/predict;
- run Qwen/SAM2/D-B1;
- run A1/A2/A3/A4/B1/B2 inference;
- run pytest/check_setup;
- lower/change thresholds;
- alter detector runtime/config/checkpoint/model package;
- train/fine-tune/resume/export/download a model;
- replace `detector.pt`;
- copy an alternate checkpoint into the external package;
- modify tests/manifests/external delivery;
- implement fallback proposal logic;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

Checkpoint METADATA inspection is allowed only if it performs no forward/inference/training/export and writes nothing
back to the checkpoint.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d5_detector_provenance_domain_gap.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No functional file may change.

# 4. Establish exact active detector identity

For:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\model\buildreasonseg_advisor\detector.pt
```

record:

```text
exists
file size
SHA256
```

Find corresponding canonical/source model reference, manifest entry, model-package metadata, model card, or build record.

Record every authoritative source path that identifies:
- model family/version;
- task type;
- class names/count;
- checkpoint origin;
- expected hash/size if recorded.

Do not infer provenance from filename alone.

# 5. Recover U-C1 detector decision history

Search repository/history and local workspace artifacts for:

```text
U-C1
YOLO26m
YOLO26m-seg
detector.pt
proposal detector
proposal quality
proposal baseline
WHU
WHU-EA
WHU-EA-NativeVector
checkpoint selection
model selection
proposal recall
proposal coverage
```

At minimum inspect relevant `docs/`, `handoff/`, `evaluation/`, `baseline/`, model-package metadata/configs/manifests,
and available git history/prior task reports.

Recover, if supported:

```text
task/commit where U-C1 was selected
candidate models compared
selection metrics
validation split/domain
detector evaluation result
training/fine-tuning dataset
training image/tile size
class definition
pretrained initialization
epoch/checkpoint selection
whether active external detector.pt is byte-identical to selected checkpoint
```

Unsupported items = `NOT ESTABLISHED`. Do not reconstruct missing facts from memory.

# 6. Separate baseline/yolo_whu from active detector

Inspect `baseline/yolo_whu/` README, args/config/manifest and checkpoint identity/history evidence.

Classify relation to active `detector.pt` exactly as:

```text
SAME_CHECKPOINT
DIRECT_ANCESTOR_OR_FINETUNE_SOURCE
SEPARATE_HISTORICAL_BASELINE
RELATION_NOT_ESTABLISHED
```

Require hash/metadata/history evidence for the first two.

# 7. Read-only checkpoint metadata inspection

Only if docs/history do not establish needed fields, inspect active checkpoint metadata without inference.

Allowed fields:
- embedded model YAML/name;
- `names`;
- task;
- train args;
- epoch/date;
- source/pretrained metadata.

Do NOT call `.predict()`, model forward, train/resume/export, or save checkpoint.

Record method and confirm:

```text
forward/inference calls = 0
checkpoint writes = 0
```

If safe extraction is uncertain, SKIP and record `NOT ESTABLISHED`.

# 8. Recover A2 provenance and input-domain facts

Search Demo/R4B artifacts and file metadata for A2 provenance.

Record if supported:

```text
source/origin
whether from WHU / WHU-EA / another aerial dataset / external user-selected image
GSD/spatial resolution
sensor/source
bit depth
original dimensions before demo preparation
whether resized/cropped/compressed
```

Unsupported fields = `NOT ESTABLISHED`.

Record directly measurable frozen A2 facts:

```text
1024×1024
RGB
file size
SHA256
dtype / bit depth
per-channel min/max/mean/std
fraction of exactly 0 and 255 pixels per channel
```

These statistics are diagnostic facts only; do not call them domain-gap proof.

# 9. Training-domain vs A2 comparison

Build a table:

```text
property                         active detector training/validation domain    A2
dataset/source
task/class definition
native image size
training crop/tile size
RGB/bit depth
known spatial resolution/GSD
known augmentation/resizing
building morphology/context
```

Unavailable fields = `NOT ESTABLISHED`.
For morphology/context use only documented descriptions; do not invent visual interpretation.

# 10. Search for existing alternate detector candidates

Search workspace/repository/external-support directories for detector checkpoints/model packages already created or
evaluated earlier in this project. Do NOT download or run them.

For each plausible candidate record:

```text
path
file size
SHA256
model family/task if documented
origin task/report
validation evidence
why it is/is not a legitimate already-validated candidate
```

Exclude arbitrary unvalidated snapshots, optimizer checkpoints without evaluation, and incompatible tasks/classes.

Classify availability:

```text
VALIDATED_ALTERNATE_AVAILABLE
NO_VALIDATED_ALTERNATE_FOUND
ALTERNATE_STATUS_NOT_ESTABLISHED
```

# 11. Scientific-freeze compatibility audit

From Task7/RC1 documentation classify detector change impact exactly as:

```text
DETECTOR_CHANGE_ENGINEERING_ONLY_IF_REVALIDATED
DETECTOR_CHANGE_TOUCHES_FROZEN_RESEARCH_CLAIMS
SCIENTIFIC_FREEZE_IMPACT_NOT_ESTABLISHED
```

Do NOT actually change detector.

# 12. Primary diagnosis classification

Choose exactly ONE:

`PROP01_DETECTOR_DOMAIN_GAP_STRONGLY_SUPPORTED`
- only if detector training/validation domain and A2 source/domain are established and materially mismatched.

`PROP01_SINGLE_IMAGE_BLINDSPOT_WITHIN_EXPECTED_DOMAIN`
- only if A2 is established to belong to the expected detector domain and still fails through P1D4.

`PROP01_CHECKPOINT_PROVENANCE_INCOMPLETE`
- if active checkpoint training/selection provenance cannot be established sufficiently.

`PROP01_DOMAIN_GAP_INSUFFICIENT_EVIDENCE`
- if checkpoint provenance exists but A2 provenance/domain is too incomplete to distinguish mismatch from in-domain blind spot.

Do not choose a stronger classification than evidence supports.

# 13. Next-gate decision — recommend only

Use exactly:

If `VALIDATED_ALTERNATE_AVAILABLE` and scientifically compatible:
```text
NEXT = CONTROLLED_ALTERNATE_DETECTOR_A2_PROBE
```

Else if `PROP01_DETECTOR_DOMAIN_GAP_STRONGLY_SUPPORTED`:
```text
NEXT = DETECTOR_ADAPTATION_DESIGN
```

Else if `PROP01_SINGLE_IMAGE_BLINDSPOT_WITHIN_EXPECTED_DOMAIN`:
```text
NEXT = DETECTOR_ROBUSTNESS_DECISION
```

Else if `PROP01_CHECKPOINT_PROVENANCE_INCOMPLETE`:
```text
NEXT = CHECKPOINT_PROVENANCE_RECOVERY
```

Else:
```text
NEXT = DOMAIN_EVIDENCE_RECOVERY
```

Do NOT execute the next gate.

# 14. Report

Create:

```text
docs/task8b3_p1d5_detector_provenance_domain_gap.md
```

Required sections:
1. scope/starting HEAD
2. P1D1–P1D4 frozen evidence
3. active detector identity
4. active model family/task/classes
5. U-C1 selection history
6. training/fine-tuning provenance
7. validation/selection evidence
8. baseline/yolo_whu relationship
9. checkpoint metadata method/evidence
10. A2 provenance and measurable facts
11. training-domain vs A2 comparison
12. alternate detector candidates
13. alternate availability enum
14. scientific-freeze impact enum
15. primary diagnosis enum
16. evidence gaps
17. exact next gate
18. `RC1-DEMO-PROP-01 = OPEN`
19. `MEM-01 = CLOSED`
20. REF-01/MASK-01 OPEN untouched
21. model inference/training = NONE
22. functional modifications = NONE.

# 15. FROM_DSH

Preserve ARTIFACT-FACTS exactly. UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D5
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 15a90ec11ad66c8f365284bf53eb320661a1e73f
Model inference/training: NONE
Functional files modified: NO
Active detector SHA256: <sha>
Active detector family/task: <value / NOT ESTABLISHED>
Training/fine-tuning dataset: <value / NOT ESTABLISHED>
U-C1 selection evidence: ESTABLISHED / PARTIAL / NOT ESTABLISHED
baseline/yolo_whu relation: <enum>
A2 source/domain: <value / NOT ESTABLISHED>
Validated alternate candidate: <enum>
Scientific-freeze impact: <enum>
Primary diagnosis: <enum>
Next gate: <enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d5_detector_provenance_domain_gap.md
Next action: Awaiting ChatGPT audit; no detector change authorized.
```

# 16. Commit / push

If COMPLETE:

```text
docs(rc1): audit detector provenance for a2
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record detector provenance forensic stop
```

Push current branch normally, no force.

# 17. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no inference/training/test execution;
- no functional/external modification;
- active detector identity/hash recorded;
- provenance searched exhaustively in available local evidence;
- baseline/yolo_whu relation classified without assumption;
- A2 provenance searched and unknowns retained;
- validated alternate candidates audited;
- scientific-freeze impact classified;
- exactly one primary diagnosis selected;
- exactly one next gate recommended but NOT executed;
- report/FROM_DSH committed/pushed;
- clean tracked tree;
- STOP.

# 18. Final response

```text
TASK 8B.3-P1D5 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Model inference/training:
NONE

Functional files modified:
NO

Active detector:
SHA256 = <...>
family/task = <...>
training dataset = <...>

U-C1 provenance:
<ESTABLISHED / PARTIAL / NOT ESTABLISHED>

baseline/yolo_whu:
<relation enum>

A2 source/domain:
<...>

Validated alternate candidate:
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

等待 ChatGPT 审核；不得更换/训练 detector，不得运行新模型诊断，不得进入 REF-01/MASK-01/Task 8B.4/8C。
```
