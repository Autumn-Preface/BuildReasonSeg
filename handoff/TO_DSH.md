# TO_DSH — Task 8B.3-P1D1: A2 Zero-Proposal Forensics

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required base branch: `main`
> Required base HEAD: `57b368d5647e842d8f31d6d1a9997bf1df3cc0fb`
> New branch to create: `fix/task8b3-prop01-a2-zero-proposals`

# 0. Purpose

Investigate `RC1-DEMO-PROP-01`:

```text
A2.png
prompt: 以面积最大的建筑为参考，分割它左侧最近的建筑
expected program: largest_to_left_of_to_nearest

historical R4B detector result:
raw proposals = 0
merged proposals = 0
runtime fallback
NMS warning observed
```

This task is forensic only.

Do NOT modify detector behavior.
Do NOT run model inference.
Do NOT attempt a fix.

# 1. Frozen defect ordering

Current defect state after MEM-01 closure:

```text
RC1-DEMO-MEM-01 = CLOSED
RC1-DEMO-PROP-01 = OPEN   ← active
RC1-DEMO-REF-01 = OPEN
RC1-DEMO-MASK-01 = OPEN
```

Do not investigate/fix REF-01 or MASK-01 here.

# 2. Git gate and branch creation

Require:

```text
current branch = main
HEAD = 57b368d5647e842d8f31d6d1a9997bf1df3cc0fb
origin/main = 57b368d5647e842d8f31d6d1a9997bf1df3cc0fb
```

Allowed tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Run:

```bat
git fetch origin
```

If `origin/main` differs, STOP.

Create exactly:

```bat
git checkout -b fix/task8b3-prop01-a2-zero-proposals
```

No rebase/reset/stash/clean.

# 3. Strict prohibitions

Do NOT:
- run `predict.py`;
- run pytest;
- run check_setup.py;
- run detector/YOLO/Qwen/SAM2/D-B1;
- load model checkpoints;
- change `detector.py`;
- change tests;
- change manifests;
- change external delivery;
- tune conf/imgsz/max_det/NMS;
- change tiling/overlap;
- change mask threshold;
- change fallback/reference/SUCCESS policy;
- copy or regenerate outputs;
- fix PROP-01;
- touch REF-01 / MASK-01;
- enter Task 8B.4 / 8C.

Read-only file inspection and image metadata inspection are allowed.

# 4. Allowed repository changes

Only:

```text
docs/task8b3_p1d1_a2_zero_proposal_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 5. Recover historical A2 evidence

Locate and inspect all existing tracked/untracked local artifacts from the original Task 8B.3 R4B / Demo run that mention A2.

Search at minimum:

```text
docs/
handoff/
logs/
delivery_src/BuildReasonSeg_Advisor_RC1/
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\
```

Search terms:

```text
A2
A2.png
raw_count
raw proposals
merged
NMS
WARNING
time limit
0 proposals
E401
largest_to_left_of_to_nearest
```

Do NOT delete or mutate any artifact.

Record:
- exact artifact path;
- timestamp if present;
- exact A2 status;
- exact raw/merged counts;
- exact NMS warning text if recoverable;
- whether warning came from Ultralytics/torch/runtime wrapper;
- whether the run reached all expected tiles;
- whether there was an exception versus a successful detector return with zero detections.

If the exact warning text cannot be recovered, say `NOT RECOVERED`; do not reconstruct it from memory.

# 6. Inspect A2 image metadata only

Locate the frozen A2 image used in the Demo.

Expected likely path:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\input\A2.png
```

Read metadata only:
- file exists;
- width × height;
- channels/mode;
- bytes;
- file hash SHA256.

Do not run a model.

Use current `plan_tiles(width,height)` logic by static reasoning or a tiny model-free Python import only if importing
`buildreasonseg.runtime.detector` does not load YOLO/checkpoints.

Record:
- expected tile count;
- tile windows/top-lefts;
- whether padding is involved;
- whether all windows are full 512×512.

# 7. Static detector-path audit

Read current `detector.py` at main HEAD and record the exact frozen settings:

```text
TILE_SIZE
TILE_OVERLAP
TILE_STRIDE
IMGSZ
CONF
MAX_DET
FROZEN_THRESHOLD
DUPLICATE_IOU
```

Trace A2 from:

```text
plan_tiles
→ extract_tile
→ DetectorRuntime.detect_tile
→ ultralytics model.predict
→ result.masks / result.boxes
→ mask resize/binarize
→ detect_global raw_count
→ compact proposal append
→ merge_proposals
```

Identify every code condition that can produce:

```text
raw_count = 0
```

Separate them into:

A. `model.predict` returns no result / no boxes / no masks;
B. each tile legitimately returns zero detections at detector threshold;
C. detector/NMS abort or timeout causes empty result;
D. wrapper discards detections before raw_count increment;
E. other statically supported mechanism.

Important:
`raw_count` increments immediately after `detect_tile(tile_rgb)`.
Therefore distinguish pre-raw-count loss from merge/eligibility loss.

# 8. NMS-warning semantics audit

Inspect pinned Ultralytics version/config already present in the project and existing source/package code without internet.

Determine, from installed/local package source if available and WITHOUT invoking inference:
- what exact warning class/string corresponds to the historical A2 NMS warning;
- what condition emits it;
- whether that code path returns partial detections, empty detections, or raises;
- whether the timeout is global-batch, per-image, or per-NMS invocation;
- whether the configured 512-tile / imgsz=640 path can plausibly hit it.

If local package source cannot establish this, mark `NOT ESTABLISHED`.

Do not propose a fix yet.

# 9. Compare with neighboring A1/A3/A4 evidence

Using historical artifacts only, build a compact comparison:

```text
case | image dimensions | tile count | raw | merged | NMS warning | final detector status
A1
A2
A3
A4
```

Purpose:
determine whether A2 is uniquely zero-proposal and uniquely associated with the warning.

Do not rerun these cases.

# 10. Root-cause classification

Use only one primary conclusion:

```text
PROP01_NMS_TIMEOUT_SUSPECT
PROP01_MODEL_ZERO_DETECTION_SUSPECT
PROP01_WRAPPER_DROP_SUSPECT
PROP01_INPUT_OR_TILE_PATH_SUSPECT
PROP01_INSUFFICIENT_EVIDENCE
```

A conclusion ending in `_SUSPECT` is not a confirmed root cause.

To call `PROP01_NMS_TIMEOUT_SUSPECT`, require at minimum:
- historical A2 warning is recovered and maps to local Ultralytics NMS time-limit behavior;
- A2 raw_count is truly zero;
- static wrapper does not discard already-returned detections before raw_count;
- neighboring cases show nonzero detector output under same frozen settings.

If evidence is insufficient, use `PROP01_INSUFFICIENT_EVIDENCE`.

# 11. Decide next diagnostic gate — do not execute it

Recommend exactly one next gate:

A. `CONTROLLED_A2_INSPECT_PROPOSALS_RUN`
   - if one model run is necessary to distinguish timeout vs true zero detection.

B. `STATIC_FIX_DESIGN_READY`
   - only if existing evidence already proves the root mechanism sufficiently for ChatGPT to design a fix.

C. `MORE_ARTIFACT_RECOVERY_REQUIRED`
   - if key historical evidence is missing.

Do NOT execute the recommended gate.

# 12. Report

Create:

```text
docs/task8b3_p1d1_a2_zero_proposal_forensics.md
```

Required sections:

1. Scope / starting HEAD
2. Historical A2 evidence
3. Exact NMS warning or `NOT RECOVERED`
4. A2 image metadata/hash
5. Tile-plan facts
6. Frozen detector settings
7. Static raw_count=0 path analysis
8. Local Ultralytics NMS warning semantics
9. A1/A2/A3/A4 historical comparison
10. Root-cause classification
11. Evidence gaps
12. Recommended next diagnostic gate
13. Explicit:
   `RC1-DEMO-PROP-01 = OPEN`
14. `MEM-01 = CLOSED`
15. `REF-01 / MASK-01 = OPEN, untouched`
16. no model/test execution.

# 13. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Base main HEAD: 57b368d5647e842d8f31d6d1a9997bf1df3cc0fb
Model/test execution: NONE
Functional files modified: NO
A2 image: <path>
A2 dimensions: <WxH>
A2 historical raw/merged: 0/0
Historical NMS warning: RECOVERED / NOT RECOVERED
Warning semantics: <short>
A1/A3/A4 nonzero comparison: ESTABLISHED / INCOMPLETE
Root-cause classification: <one enum>
Next diagnostic gate: <one enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d1_a2_zero_proposal_forensics.md
Next action: Awaiting ChatGPT audit.
```

# 14. Commit / push

If COMPLETE:

```text
docs(rc1): investigate a2 zero-proposal defect
```

If PARTIAL/STOP/FAILED:

```text
docs(rc1): record a2 proposal forensic stop
```

Push:

```bat
git push -u origin fix/task8b3-prop01-a2-zero-proposals
```

No force.

# 15. COMPLETE definition

COMPLETE only if:
- exact main base;
- new branch created exactly;
- no model/test/check_setup/predict execution;
- no functional modification;
- historical A2 evidence exhaustively searched;
- A2 metadata/tile plan documented;
- detector raw_count=0 paths statically classified;
- local NMS warning semantics audited if source available;
- neighboring historical cases compared;
- one evidence-based root-cause classification chosen;
- one next diagnostic gate recommended but NOT run;
- report/handoff committed and pushed;
- clean tracked tree;
- STOP.

# 16. Final response

```text
TASK 8B.3-P1D1 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Branch:
fix/task8b3-prop01-a2-zero-proposals

Model/test execution:
NONE

Functional files modified:
NO

A2:
dimensions = <...>
historical raw/merged = 0/0
NMS warning = RECOVERED / NOT RECOVERED

Root-cause classification:
<enum>

Next diagnostic gate:
<enum>

MEM-01:
CLOSED

PROP-01:
OPEN

REF-01 / MASK-01:
OPEN / OPEN

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得执行诊断推理、不得修改 detector、不得进入 REF-01/MASK-01/Task 8B.4/8C。
```
