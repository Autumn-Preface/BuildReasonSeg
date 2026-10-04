# TO_DSH — Task 8B.3-REF01-F1-R5: Verify and Close Locked Reference Forensics

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `d1c7f6374f325fec71bce6a350609b2d23865c0a`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

R4 produced useful preliminary IoU results, but R4 is **NOT APPROVED** because the committed harness did not implement the full R4 contract.

Preliminary R4 observations are NOT YET the authoritative closure:

```text
right:
selected IoU = 0.5588697017
best eligible IoU = 0.5588697017
best any IoU = 0.5588697017
preliminary class = REFERENCE_SELECTED_CORRECT

left:
selected IoU = 0.0
best eligible IoU = 0.6500672948
best any IoU = 0.6500672948
preliminary class = REFERENCE_SELECTION_WRONG_COVERED

above:
selected IoU = 0.0
best eligible IoU = 0.0
best any IoU = 0.9032501890
preliminary class = REFERENCE_ELIGIBILITY_BLOCKED

below:
selected IoU = 0.0
best eligible IoU = 0.6165496860
best any IoU = 0.6165496860
preliminary class = REFERENCE_SELECTION_WRONG_COVERED
```

R4 used exactly 4 detector calls.

This R5 authorizes **one additional deterministic verification pass per locked candidate** solely to close the missing
reproduction/evidence contract.

This additional engineering verification:
- does not tune a model;
- does not tune a threshold;
- does not select a candidate;
- does not alter Task 7J metrics;
- does not alter architecture.

# 1. Why R4 is not approved

The committed R4 harness still lacks required proof:

```text
1. no raster SHA gate in the script;
2. no external imageio module identity gate;
3. no complete frozen detector constant gate;
4. no exact required scientific disclosure in evidence;
5. no full proposal-by-proposal comparison against P1D12 proposals.json;
6. only raw/merged/eligible counts were reproduced;
7. no best_eligible_id / best_any_id / confidence / area / eligibility evidence;
8. evidence says detector module path but does not prove actual module __file__;
9. image input still uses PIL directly instead of external RC1 load_image();
10. evidence does not record runtime checkpoint path/default device.
```

Do not treat the R4 COMPLETE handoff as final approval.

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = d1c7f6374f325fec71bce6a350609b2d23865c0a
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No new branch.
No merge/rebase/reset/stash/clean/cherry-pick.

# 3. Strict prohibitions

Do NOT:
- modify canonical or external RC1;
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- change detector weights/config/thresholds;
- change eligibility or selector;
- change IoU threshold 0.50;
- visually inspect images/proposal previews;
- replace candidates;
- regenerate dataset/cache;
- compute Task 7J metrics;
- implement product repair;
- update main;
- force push.

# 4. Allowed tracked changes

Only:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 5. Exact scientific reuse disclosure

Evidence JSON and report MUST include exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Report must also include a faithful Chinese version.

# 6. Exact roots and immutable rasters

Use:

```python
REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
RASTER_ROOT = Path(
    r"C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image"
)
```

Immutable SHA:

```text
1010:
1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2

1003:
eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38

1008:
0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd

1009:
c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7
```

Before detector construction:
- all four files exist;
- SHA exact;
- external RC1 `load_image()` returns 512×512, channels=3.

# 7. External runtime import identity

Before any `buildreasonseg` import:

```python
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))
```

Import:

```python
import buildreasonseg.runtime.detector as detector_module
import buildreasonseg.runtime.imageio as imageio_module

from buildreasonseg.runtime.detector import (
    DetectorRuntime,
    eligible_proposals,
    select_reference,
)
from buildreasonseg.runtime.imageio import load_image
```

Require actual:

```text
detector_module.__file__ under EXTERNAL
imageio_module.__file__ under EXTERNAL
```

Store actual resolved paths in evidence.

# 8. Frozen detector contract

Require:

```text
TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = 384
IMGSZ = 640
CONF = 0.05
MAX_DET = 300
DUPLICATE_IOU = 0.50
MERGE_BBOX_EXTENT_RATIO_MAX = 0.20
FROZEN_THRESHOLD = 0.5
```

Construct exactly:

```python
runtime = DetectorRuntime()
```

No arguments.

Require before first detector call:

```text
runtime.checkpoint.resolve()
=
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\model\buildreasonseg_advisor\detector.pt

runtime.device = cuda
runtime.imgsz = 640
runtime.conf = 0.05
runtime.max_det = 300
```

Record all in evidence.

# 9. Frozen GT reference facts

Use existing native-vector caches only.

Exact:

```text
1010 GT ref = 4, area = 2478
1003 GT ref = 26, area = 3512
1008 GT ref = 4, area = 5013
1009 GT ref = 3, area = 2606
```

Require each label map:
- shape `(512,512)`;
- mask non-empty;
- mask pixel area exact.

# 10. Stored P1D12 metadata contract

Load each existing external `proposals.json`.

Expected:

```text
1010 raw=6  merged=6  eligible=4
1003 raw=66 merged=53 eligible=42
1008 raw=9  merged=9  eligible=4
1009 raw=7  merged=6  eligible=3
```

Build stored proposal map by `proposal_id`.

Required stored fields for every item:

```text
proposal_id
source_tile_id
confidence
mask_area
global_bbox
centroid
touches_image_border
border_clearance
bbox_extent_ratio
raw_index
```

Missing field → STOP before detector execution.

# 11. Full metadata comparator — mandatory

For every rerun merged proposal compare with stored P1D12 proposal of the same `proposal_id`.

Require exact:

```text
proposal_id
source_tile_id
mask_area
global_bbox
touches_image_border
raw_index
```

Require abs difference ≤ `1e-6`:

```text
confidence
centroid[0]
centroid[1]
border_clearance
bbox_extent_ratio
```

Require:
- exact proposal-id set;
- exact merged count;
- exact raw count;
- exact eligible count.

Evidence must store:

```text
full_metadata_match = true
metadata_mismatch = []
```

for every successful candidate.

Any mismatch:
- script exits non-zero;
- no authoritative R5 classification;
- no retry.

# 12. Proposal mask / IoU contract

Only source:

```text
rerun in-memory GlobalProposal.mask_crop
```

Reconstruct exactly:

```python
top, left, bottom, right = proposal.global_bbox
assert proposal.mask_crop.dtype == bool
assert proposal.mask_crop.shape == (
    bottom - top + 1,
    right - left + 1,
)

full = np.zeros((512,512), dtype=bool)
full[top:bottom+1, left:right+1] = proposal.mask_crop
```

No transformation.

Coverage threshold:

```text
0.50
```

Use:

```python
selected = select_reference(merged, family="largest")
eligible = eligible_proposals(merged, family="largest")
```

For every merged proposal calculate IoU to canonical GT reference.

`best_eligible` and `best_any` ranking:

```python
(-iou_to_gt, -confidence, proposal_id)
```

# 13. Required per-candidate output

Evidence must store:

```text
sample_id
tile
query_type
raster_path
raster_sha256
gt_reference_id
gt_area_px
raw_count
merged_count
eligible_count
full_metadata_match

selected:
  proposal_id
  iou
  confidence
  mask_area
  global_bbox
  eligible

best_eligible:
  proposal_id
  iou
  confidence
  mask_area

best_any:
  proposal_id
  iou
  confidence
  mask_area
  eligible

selected_to_best_eligible_iou_gap
classification
```

# 14. Exclusive classification

Exactly:

```text
REFERENCE_SELECTED_CORRECT
if selected_iou >= 0.50
```

```text
REFERENCE_SELECTION_WRONG_COVERED
if selected_iou < 0.50
and best_eligible_iou >= 0.50
```

```text
REFERENCE_ELIGIBILITY_BLOCKED
if selected_iou < 0.50
and best_eligible_iou < 0.50
and best_any_iou >= 0.50
```

```text
REFERENCE_COVERAGE_MISSING
if best_any_iou < 0.50
```

# 15. Cross-run consistency with R4 preliminary evidence

R4 preliminary values are diagnostic only.

For each candidate require R5 IoUs differ from R4 by ≤ `1e-6`:

```text
right:
selected/bestEligible/bestAny =
0.5588697017268446 /
0.5588697017268446 /
0.5588697017268446

left:
0.0 /
0.6500672947510094 /
0.6500672947510094

above:
0.0 /
0.0 /
0.9032501889644747

below:
0.0 /
0.6165496859992612 /
0.6165496859992612
```

Require same four classes.

If cross-run consistency fails:
- STOP;
- no tuning;
- no retry.

# 16. Overall outcome priority

If any `REFERENCE_COVERAGE_MISSING`:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = COVERAGE
NEXT = REF01_COVERAGE_FRAGMENTATION_FORENSICS
```

Else if any `REFERENCE_ELIGIBILITY_BLOCKED`:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Else if any `REFERENCE_SELECTION_WRONG_COVERED`:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = SELECTION
NEXT = REF01_SELECTION_REPAIR_DESIGN
```

Else:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_GATE_PASS
Dominant next blocker = NONE_IN_LOCKED_REFERENCE_SET
NEXT = MASK01_LOCKED_DEMO_END_TO_END_FORENSICS
```

Do not execute NEXT.

# 17. Pre-execution gates

Before `main()`:

## 17.1 compile

```text
<REQUIRED_PYTHON> -m py_compile scripts/task8b3_ref01_locked_reference_forensics.py
```

Require PASS.

## 17.2 import-only

Import module without calling `main()`.

Require:
- RASTER_ROOT valid;
- detector module under external;
- imageio module under external;
- no detector call.

## 17.3 contract precheck

Require all of §§6–10 pass with detector calls still 0.

Only then execute harness.

# 18. Harness execution

Run the harness exactly once.

This R5 explicitly authorizes:

```text
4 additional verification detector calls
= exactly 1 per locked candidate
```

No retry.

If harness exits non-zero:
- STOP.

# 19. Evidence JSON

Overwrite/update:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Required top-level:

```text
task = 8B.3-REF01-F1-R5
starting_head
scientific_reuse_disclosure
python_executable
external_detector_module_path
external_imageio_module_path
runtime_checkpoint_path
runtime_default_device
detector_config
proposal_mask_source
proposals_json_role
coverage_threshold
r4_preliminary_crosscheck
detector_call_count = 4
candidates
class_counts
overall_outcome
dominant_next_blocker
next_gate
```

# 20. Report

Update:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Preserve all previous STOP / preliminary R4 history.

Append R5 authoritative verification section.

Required table:

```text
relation | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_id | best_any_IoU | metadata_match | class
```

State clearly:

```text
R4 = preliminary, not formally approved
R5 = authoritative verification if all gates pass
```

# 21. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R5
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: d1c7f6374f325fec71bce6a350609b2d23865c0a
R4 status entering R5: PRELIMINARY_NOT_APPROVED
Prior detector calls total: 4
R5 additional detector calls authorised: 4
PY_COMPILE_GATE: PASS / FAIL
IMPORT_ONLY_GATE: PASS / FAIL
CONTRACT_PRECHECK: PASS / FAIL
Detector calls before harness main: 0 / other
Locked raster SHA: 4/4 PASS / other
External detector module identity: PASS / FAIL
External imageio module identity: PASS / FAIL
Detector runtime construction: DEFAULT / other
Runtime checkpoint: <path>
Runtime device: <value>
Coverage threshold: 0.50_TASK7F_FROZEN
Proposal mask source: RERUN_GLOBALPROPOSAL_MASK_CROP
right full metadata reproduction: PASS / FAIL
left full metadata reproduction: PASS / FAIL
above full metadata reproduction: PASS / FAIL
below full metadata reproduction: PASS / FAIL
R4→R5 IoU consistency: 4/4 PASS / other
R5 detector calls: 4 / other
Qwen/SAM2/relation/D-B1/target: NONE/NONE/NONE/NONE/NONE
Manual visual inspection: NO
Candidate replacement: NO
right selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
left selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
above selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
below selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
right class: <class>
left class: <class>
above class: <class>
below class: <class>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / other
Outcome: <enum>
Dominant next blocker: <value>
Next gate: <enum>
Evidence: evaluation/task8b3_ref01_locked_reference_forensics.json
Report: docs/task8b3_ref01_locked_reference_forensics.md
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 22. Commit / push

If COMPLETE:

```text
docs(rc1): verify locked reference forensics
```

If STOP/FAILED:

```text
docs(rc1): record reference verification stop
```

Push only current task branch.
No force push.
Do not update main.

Then STOP.

# 23. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only five allowed tracked paths changed;
- compile/import/contract gates pass;
- 4 raster SHA exact;
- external detector + imageio module identities pass;
- default DetectorRuntime only;
- full frozen constants pass;
- exactly 4 R5 detector calls;
- full proposal-by-proposal P1D12 metadata reproduction = 4/4 PASS;
- R4→R5 IoU consistency = 4/4 PASS;
- exact mandatory scientific disclosure;
- one exclusive class per candidate;
- no model stage beyond detector;
- no visual judgement / candidate replacement / repair;
- evidence/report/handoff committed and pushed;
- NEXT not executed;
- STOP.
