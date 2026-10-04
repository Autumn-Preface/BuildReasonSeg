# TO_DSH — Task 8B.3-REF01-F1-R4: Finalize Forensic Harness Before Detector Execution

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `35105fb8b4255375923ff3174d3efd2356717333`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit decision

Task 8B.3-REF01-F1-R3 STOP is accepted as a safe STOP.

No forensic classification exists yet.

## Detector-pass accounting

R3 failed while evaluating the module-level `RASTER_ROOT` expression:

```text
TypeError: unsupported operand type(s) for +: 'WindowsPath' and 'str'
```

This happened before `main()`.

Therefore:

```text
detector passes consumed through F1/R1/R2/R3 = 0
remaining detector allowance:
1010 = 1
1003 = 1
1008 = 1
1009 = 1
```

## R3 static-gate claim is NOT accepted

The committed R3 script still violates the previous contract in several ways:

```text
1. RASTER_ROOT expression is invalid at import time.
2. DetectorRuntime is still constructed with explicit checkpoint + device="cpu".
3. imageio module identity is not verified.
4. raster SHA identity is not enforced before detector execution.
5. full frozen detector constants are not enforced.
6. rerun proposals are not compared proposal-by-proposal with frozen P1D12 metadata.
7. best eligible / best any proposal IDs and deterministic tie-break evidence are incomplete.
8. the exact mandatory scientific reuse disclosure is not written to evidence.
```

Do NOT rerun the current script as-is.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = 35105fb8b4255375923ff3174d3efd2356717333
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No branch creation.
No merge/rebase/reset/stash/clean/cherry-pick.

# 2. Strict prohibitions

Do NOT:
- execute the existing forensic script before completing §§5–12;
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- modify canonical RC1;
- modify external RC1;
- run sync write;
- change detector weights or any runtime threshold/config;
- pass explicit `checkpoint`, `device`, `imgsz`, `conf`, or `max_det` to `DetectorRuntime`;
- change eligibility or reference selector;
- change IoU threshold `0.50`;
- inspect source/proposal images visually;
- replace a candidate;
- regenerate dataset records/caches;
- compute Task 7J metrics;
- implement product repair;
- update `main`;
- force push.

# 3. Allowed tracked changes

Only:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 4. Mandatory scientific disclosure

The final evidence JSON and report MUST contain exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Also include a faithful Chinese version in the report.

# 5. Rewrite the forensic script to the frozen contract

Edit:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

Do not perform incremental ad-hoc execution while editing.

## 5.1 Exact roots

Use literal UTF-8 paths:

```python
REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")

RASTER_ROOT = Path(
    r"C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image"
)
```

Do NOT construct the dataset name with `+`, `chr(...)`, or mixed `/` precedence.

## 5.2 Python package precedence

Before importing any `buildreasonseg` module:

```python
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))
```

Then import and retain module objects:

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

Research native-vector utilities may come from the repo.

## 5.3 Detector construction

Construct exactly:

```python
runtime = DetectorRuntime()
```

No keyword/positional override.

Before the first detector call require:

```text
runtime.checkpoint.resolve()
=
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\model\buildreasonseg_advisor\detector.pt
```

Record the default runtime device value but do not change it.

# 6. Immutable candidates / raster SHA gate

Use exactly:

```text
right:
sample_id = buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91
tile = 1010
GT ref = 4
GT area = 2478
SHA256 = 1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2

left:
sample_id = buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
tile = 1003
GT ref = 26
GT area = 3512
SHA256 = eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38

above:
sample_id = buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314
tile = 1008
GT ref = 4
GT area = 5013
SHA256 = 0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd

below:
sample_id = buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
tile = 1009
GT ref = 3
GT area = 2606
SHA256 = c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7
```

Before creating `DetectorRuntime()` require for all four:
- path exists;
- SHA256 exact;
- `load_image(path)` returns width=512, height=512, channels=3.

If any fail:
- detector calls = 0;
- exit non-zero.

# 7. External module identity and frozen constants

Before creating `DetectorRuntime()` require:

```text
Path(detector_module.__file__).resolve()
under EXTERNAL

Path(imageio_module.__file__).resolve()
under EXTERNAL
```

Require exact constants from `detector_module`:

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

Any mismatch:
- detector calls = 0;
- exit non-zero.

# 8. Frozen v0.2 / GT reference pre-detector gate

Require:

```text
datasets/build_spatial_reason/v0.2/test.jsonl
bytes = 14415287
sha256 = 72525bff76ef5bdc34dedcb257a5f9c72c420fce0d78dbaa34b02d5f230301dc
records = 6219
```

Resolve each immutable sample ID exactly once.

Require:
- exact tile/query;
- level=3;
- first reasoning operation=`argmax_area`;
- canonical reference ID exact;
- native-vector reference provenance exact;
- `validate_reasoning_record(...) == []`.

For GT mask:
- existing cache only;
- `(512,512)`;
- non-empty;
- pixel area equals the frozen GT area above.

No cache regeneration.

# 9. P1D12 stored metadata pre-detector gate

Read existing external:

```text
inference/output/diagnostics/1010/proposals.json
inference/output/diagnostics/1003/proposals.json
inference/output/diagnostics/1008/proposals.json
inference/output/diagnostics/1009/proposals.json
```

Require counts:

```text
1010 raw=6  merged=6  eligible=4
1003 raw=66 merged=53 eligible=42
1008 raw=9  merged=9  eligible=4
1009 raw=7  merged=6  eligible=3
```

Store all proposal metadata for the later reproduction gate.

`proposals.json` is metadata only, never a mask source.

# 10. Full proposal-by-proposal reproduction function

The script MUST contain a function that compares rerun merged proposals to stored P1D12 items by `proposal_id`.

For every proposal require exact equality of:

```text
proposal_id
source_tile_id
mask_area
global_bbox
touches_image_border
raw_index
```

Require absolute difference ≤ `1e-6` for:

```text
confidence
centroid[0]
centroid[1]
border_clearance
bbox_extent_ratio
```

Also require:
- same proposal ID set;
- same proposal count;
- raw count exact;
- eligible count exact.

If any field fails:
- record exact mismatch in memory;
- return non-zero;
- do NOT write a successful classification;
- no detector retry.

# 11. Frozen proposal mask / IoU / ranking

Mask source ONLY:

```text
rerun in-memory GlobalProposal.mask_crop
```

Require each:

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

No transformations.

Coverage threshold exactly:

```text
0.50
```

Production selected proposal:

```python
selected = select_reference(merged, family="largest")
```

Eligible proposals:

```python
eligible = eligible_proposals(merged, family="largest")
```

`best_eligible` and `best_any` ranking:

```python
(-iou_to_gt, -confidence, proposal_id)
```

Record:
- IDs;
- IoUs;
- confidence;
- area;
- whether best_any is eligible;
- selected→bestEligible IoU gap.

# 12. Exact exclusive classes and NEXT

Per candidate exactly one:

```text
REFERENCE_SELECTED_CORRECT
if selected_iou >= 0.50
```

```text
REFERENCE_SELECTION_WRONG_COVERED
if selected_iou < 0.50 and best_eligible_iou >= 0.50
```

```text
REFERENCE_ELIGIBILITY_BLOCKED
if selected_iou < 0.50 and best_eligible_iou < 0.50 and best_any_iou >= 0.50
```

```text
REFERENCE_COVERAGE_MISSING
if best_any_iou < 0.50
```

Overall priority:

```text
any COVERAGE_MISSING
→ blocker=COVERAGE
→ NEXT=REF01_COVERAGE_FRAGMENTATION_FORENSICS

else any ELIGIBILITY_BLOCKED
→ blocker=ELIGIBILITY
→ NEXT=REF01_ELIGIBILITY_FORENSICS

else any SELECTION_WRONG_COVERED
→ blocker=SELECTION
→ NEXT=REF01_SELECTION_REPAIR_DESIGN

else all SELECTED_CORRECT
→ blocker=NONE_IN_LOCKED_REFERENCE_SET
→ NEXT=MASK01_LOCKED_DEMO_END_TO_END_FORENSICS
```

# 13. PRE-EXECUTION COMPILE GATE

After editing, run exactly:

```text
<REQUIRED_PYTHON> -m py_compile scripts/task8b3_ref01_locked_reference_forensics.py
```

Require exit 0.

Record:

```text
PY_COMPILE_GATE = PASS
```

If fail:
- do not import harness;
- do not execute detector;
- STOP.

# 14. IMPORT-ONLY HARNESS GATE

After §13 PASS, import the script as a module WITHOUT calling `main()`.

The import-only check must assert:

```text
module.RASTER_ROOT.is_dir() == True

"Satellite dataset Ⅱ (East Asia)"
in str(module.RASTER_ROOT)

Path(module.detector_module.__file__).resolve()
under EXTERNAL

Path(module.imageio_module.__file__).resolve()
under EXTERNAL
```

Also inspect source text and assert:

```text
"DetectorRuntime()" present
'device="cpu"' absent
"device='cpu'" absent
"checkpoint=" absent in DetectorRuntime construction
"RERUN_GLOBALPROPOSAL_MASK_CROP" present
"proposals.json" present
"1e-6" or equivalent 0.000001 tolerance logic present
```

Require:

```text
IMPORT_ONLY_GATE = PASS
```

If import or assertion fails:
- detector calls = 0;
- do NOT execute main;
- STOP.

# 15. CONTRACT PRECHECK — NO DETECTOR

Before the single harness execution, run a read-only/pre-detector check using imported helper functions or an explicit
`preflight()` function in the script.

It must establish without constructing/running the detector:

```text
Required Python executable exact
4 raster SHA identities = PASS
4 raster load_image identities = 512x512 RGB PASS
external detector module identity = PASS
external imageio module identity = PASS
all frozen constants = PASS
v0.2 TEST identity = PASS
4 locked records/provenance = PASS
4 GT masks/areas = PASS
4 P1D12 stored proposal metadata sets loaded = PASS
```

Require:

```text
CONTRACT_PRECHECK = PASS
detector calls so far = 0
```

If any fail:
- do not execute main;
- STOP.

# 16. Execute harness exactly once

Only after §§13–15 all PASS:

```text
<REQUIRED_PYTHON> scripts/task8b3_ref01_locked_reference_forensics.py
```

Exactly one process invocation.

No wrapper retry.
Set UTF-8 environment before invocation if necessary.

Expected:

```text
exit = 0
detector_call_count = 4
```

If non-zero:
- do not retry;
- STOP.

# 17. Evidence JSON contract

On successful execution write:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Required top-level fields:

```text
task = 8B.3-REF01-F1-R4
starting_head
scientific_reuse_disclosure
proposal_mask_source = RERUN_GLOBALPROPOSAL_MASK_CROP
proposals_json_role = METADATA_REPRODUCTION_ONLY
coverage_threshold = 0.50
python_executable
external_detector_module_path
external_imageio_module_path
runtime_checkpoint_path
runtime_default_device
detector_config
detector_call_count = 4
test_jsonl_identity
candidates
class_counts
overall_outcome
dominant_next_blocker
next_gate
```

Each candidate must include:
- sample_id/tile/query;
- raster path/SHA;
- GT reference ID/source feature/area;
- reproduction match;
- raw/merged/eligible counts;
- selected proposal full facts;
- best eligible full facts;
- best any full facts;
- exclusive class.

# 18. Report

Update:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Preserve all previous STOP history.

Append R4 section containing:
- R3 audit correction;
- PY_COMPILE_GATE;
- IMPORT_ONLY_GATE;
- CONTRACT_PRECHECK;
- exact default runtime checkpoint/device;
- detector calls;
- reproduction gate;
- GT-IoU table;
- class counts;
- exact Outcome/NEXT;
- nonclaims.

Required table:

```text
relation | GT_ref | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_id | best_any_IoU | class
```

# 19. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required fields:

```text
Task: 8B.3-REF01-F1-R4
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: 35105fb8b4255375923ff3174d3efd2356717333
Prior actual detector passes through R3: 0
PY_COMPILE_GATE: PASS / FAIL
IMPORT_ONLY_GATE: PASS / FAIL
CONTRACT_PRECHECK: PASS / FAIL
Detector calls before main: 0 / other
Locked raster SHA identities: 4/4 PASS / other
External detector module identity: PASS / FAIL
External imageio module identity: PASS / FAIL
Detector runtime construction: DEFAULT / other
Runtime checkpoint path: <path>
Runtime default device: <value>
Proposal mask source: RERUN_GLOBALPROPOSAL_MASK_CROP
proposals.json role: METADATA_REPRODUCTION_ONLY
Coverage threshold: 0.50_TASK7F_FROZEN
GT reference-mask integrity: 4/4 PASS / other
Harness process invocations: 1 / other
Detector calls in successful harness: 4 / other
Detector reproduction right: 6/6 eligible4 FULL_METADATA_MATCH / other
Detector reproduction left: 66/53 eligible42 FULL_METADATA_MATCH / other
Detector reproduction above: 9/9 eligible4 FULL_METADATA_MATCH / other
Detector reproduction below: 7/6 eligible3 FULL_METADATA_MATCH / other
Qwen execution: NONE
SAM2 execution: NONE
Relation-field execution: NONE
D-B1 execution: NONE
Target segmentation: NONE
Manual visual inspection: NO
Candidate replacement: NO
right selected_IoU / bestEligible_IoU / bestAny_IoU: <values>
left selected_IoU / bestEligible_IoU / bestAny_IoU: <values>
above selected_IoU / bestEligible_IoU / bestAny_IoU: <values>
below selected_IoU / bestEligible_IoU / bestAny_IoU: <values>
right class: <class>
left class: <class>
above class: <class>
below class: <class>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / GATE_PASS / other
Outcome: <exact enum>
Dominant next blocker: <exact>
Next gate: <exact>
Evidence: evaluation/task8b3_ref01_locked_reference_forensics.json
Report: docs/task8b3_ref01_locked_reference_forensics.md
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 20. Commit / push

If COMPLETE:

```text
docs(rc1): complete locked reference forensics
```

If STOP/FAILED:

```text
docs(rc1): record reference forensics preflight stop
```

Push only:

```text
fix/task8b3-ref01-reference-forensics
```

No force push.
Do not update main.

Then STOP.

# 21. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only allowed five tracked paths changed;
- prior detector pass count correctly frozen at 0;
- compile gate PASS;
- import-only gate PASS;
- contract precheck PASS with detector calls still 0;
- required Python executable exact;
- exact raster paths/SHA/readability;
- runtime imports from external RC1;
- all frozen constants exact;
- runtime = `DetectorRuntime()` with no override;
- one harness process invocation;
- exactly four detector calls;
- full proposal-by-proposal P1D12 reproduction for all four;
- in-memory mask_crop is the only IoU source;
- threshold exactly 0.50;
- one exclusive class per candidate;
- no Qwen/SAM2/relation/D-B1/target segmentation;
- no visual judgement/candidate replacement;
- evidence/report/handoff committed and pushed;
- NEXT not executed;
- STOP.
