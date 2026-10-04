# TO_DSH — Task 8B.3-REF01-F1-R6: Authoritative Reference-Forensics Closure

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `53de75ae4f5246b4e18685b78bd5a9f913447984`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

R5 produced a second perfectly stable set of IoUs, but R5 is **NOT formally approved**.

Accepted R5 facts:

```text
4 immutable raster SHA values = MATCH
R5 detector calls = 4
R4 -> R5 IoU deltas = exactly 0 for all four cases
preliminary classes remained:
  right = REFERENCE_SELECTED_CORRECT
  left  = REFERENCE_SELECTION_WRONG_COVERED
  above = REFERENCE_ELIGIBILITY_BLOCKED
  below = REFERENCE_SELECTION_WRONG_COVERED
preliminary dominant blocker = ELIGIBILITY
```

R5 formal defects:

```text
D1. The tracked forensic harness was not upgraded to the R5 contract.
D2. R5 checked the third-party package name `imageio` instead of
    `buildreasonseg.runtime.imageio`.
D3. R5 compared external Git-canonical bytes against the Windows working-tree
    copy and incorrectly reported detector_identical_to_canonical = false.
D4. R5 did NOT compare all required P1D12 per-proposal fields:
    source_tile_id, raw_index, confidence, centroid, border_clearance were missing.
D5. R5 did NOT record best_eligible_id / best_any_id and their required facts.
D6. The exact mandatory scientific-reuse disclosure was absent from evidence.
D7. R5 added the unauthorized file:
    evaluation/task8b3_ref01_locked_reference_forensics_r5.json
    instead of replacing the canonical evidence file.
```

This R6 is the final authoritative verification task.

It authorizes one final deterministic detector pass per locked candidate because the missing metadata cannot be
reconstructed from R5 evidence after the fact.

# 1. Scientific meaning of the R6 verification

R6 is engineering reproducibility verification after Task 7J test metrics were already consumed.

It MUST NOT:
- alter any reported scientific metric;
- select/re-select model/seed/threshold/architecture;
- choose a new Demo candidate;
- tune the detector or selector.

The final evidence and report MUST include exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

The report must also contain a faithful Chinese translation.

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = 53de75ae4f5246b4e18685b78bd5a9f913447984
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No new branch.
No reset/rebase/stash/clean/merge/cherry-pick.

# 3. Strict prohibitions

Do NOT:
- modify canonical RC1;
- modify external RC1;
- run sync write;
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- change detector weights/config/thresholds;
- change eligibility;
- change selector;
- change IoU threshold 0.50;
- visually inspect source images or proposal previews;
- replace candidates;
- regenerate dataset/cache;
- compute Task 7J metrics;
- implement product repair;
- update main;
- force push.

# 4. Allowed tracked changes

ONLY:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
evaluation/task8b3_ref01_locked_reference_forensics_r5.json   # DELETE ONLY
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

The `_r5.json` file must be deleted in this task.
Its history remains in Git; it must not remain as the final authoritative evidence artifact.

No other tracked path may change.

# 5. Exact identity basis — IMPORTANT

Do NOT compare external RC1 files against Windows working-tree bytes.

The authoritative identity basis is:

```text
GIT_CANONICAL_BLOB_BYTES
```

Use the external `source_manifest.json` as the identity control.

Require:

```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
```

Required manifest identities:

```text
buildreasonseg/runtime/detector.py
sha256 = 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
bytes = 20300

buildreasonseg/runtime/imageio.py
sha256 = b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
bytes = 7978
```

Require actual external bytes for both files exactly match those manifest entries.

Do NOT calculate or report a `detector_identical_to_worktree` result.

# 6. Exact package import contract

The forensic script MUST contain:

```python
REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")

sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))

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
Path(detector_module.__file__).resolve()
under EXTERNAL

Path(imageio_module.__file__).resolve()
under EXTERNAL
```

Do NOT import the third-party package `imageio`.
Do NOT record `No module named imageio` as relevant evidence.

# 7. Immutable raster gate

Exact raster root:

```python
RASTER_ROOT = Path(
    r"C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image"
)
```

Exact SHA identities:

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

Before runtime construction, for all four:
- exists;
- SHA exact;
- `load_image()` succeeds;
- width=512;
- height=512;
- channels=3.

# 8. Frozen detector contract

Read constants from `detector_module`.

Require exact:

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

No args.

Require:

```text
runtime.checkpoint.resolve()
=
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\model\buildreasonseg_advisor\detector.pt

runtime.device = cuda
runtime.imgsz = 640
runtime.conf = 0.05
runtime.max_det = 300
```

# 9. Frozen GT reference contract

Use existing frozen native-vector cache only.

Exact GT:

```text
right  / 1010 / ref 4  / area 2478
left   / 1003 / ref 26 / area 3512
above  / 1008 / ref 4  / area 5013
below  / 1009 / ref 3  / area 2606
```

Require for all four:
- label map shape `(512,512)`;
- reference mask non-empty;
- reference pixel count exact.

No regeneration.

# 10. P1D12 stored proposal metadata

Read existing external:

```text
inference/output/diagnostics/1010/proposals.json
inference/output/diagnostics/1003/proposals.json
inference/output/diagnostics/1008/proposals.json
inference/output/diagnostics/1009/proposals.json
```

Expected counts:

```text
1010 = raw 6  / merged 6  / eligible 4
1003 = raw 66 / merged 53 / eligible 42
1008 = raw 9  / merged 9  / eligible 4
1009 = raw 7  / merged 6  / eligible 3
```

For every stored proposal item, require these fields exist:

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

If any required field is missing:
- detector calls remain 0;
- STOP.

# 11. Full metadata comparator — exact contract

Map stored items by `proposal_id`.

Map rerun merged proposals by `proposal_id`.

Require exact proposal ID set.

For every proposal require exact:

```text
proposal_id
source_tile_id
mask_area
global_bbox
touches_image_border
raw_index
```

Require absolute difference <= 1e-6:

```text
confidence
centroid[0]
centroid[1]
border_clearance
bbox_extent_ratio
```

Also require exact:
- raw_count;
- merged_count;
- eligible_count.

Evidence must store for each candidate:

```json
"reproduction": {
  "full_metadata_match": true,
  "metadata_mismatches": [],
  "raw_match": true,
  "merged_match": true,
  "eligible_match": true
}
```

Evidence must NOT reduce the check to only four fields.

# 12. Proposal mask / IoU contract

Only mask source:

```text
RERUN_GLOBALPROPOSAL_MASK_CROP
```

For each rerun proposal:

```python
top, left, bottom, right = proposal.global_bbox

assert proposal.mask_crop.dtype == bool
assert proposal.mask_crop.shape == (
    bottom - top + 1,
    right - left + 1,
)

full = np.zeros((512, 512), dtype=bool)
full[top:bottom+1, left:right+1] = proposal.mask_crop
```

No transformation.

Coverage threshold remains:

```text
0.50
```

Production:

```python
selected = select_reference(merged, family="largest")
eligible = eligible_proposals(merged, family="largest")
```

For every proposal compute IoU to canonical GT reference.

Rank `best_eligible` and `best_any` exactly by:

```python
(-iou_to_gt, -confidence, proposal_id)
```

# 13. Required proposal facts

For each locked candidate write:

```text
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
```

Do not write only IoU values.

# 14. Classification — frozen

Exactly:

```text
REFERENCE_SELECTED_CORRECT
if selected IoU >= 0.50
```

```text
REFERENCE_SELECTION_WRONG_COVERED
if selected < 0.50 and best eligible >= 0.50
```

```text
REFERENCE_ELIGIBILITY_BLOCKED
if selected < 0.50 and best eligible < 0.50 and best any >= 0.50
```

```text
REFERENCE_COVERAGE_MISSING
if best any < 0.50
```

# 15. R4/R5 consistency gate

R6 must reproduce both previous runs within `1e-6`.

Frozen expected IoUs:

```text
right:
selected     = 0.5588697017268446
bestEligible = 0.5588697017268446
bestAny      = 0.5588697017268446

left:
selected     = 0.0
bestEligible = 0.6500672947510094
bestAny      = 0.6500672947510094

above:
selected     = 0.0
bestEligible = 0.0
bestAny      = 0.9032501889644747

below:
selected     = 0.0
bestEligible = 0.6165496859992612
bestAny      = 0.6165496859992612
```

Expected classes:

```text
right = REFERENCE_SELECTED_CORRECT
left  = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Any inconsistency:
- STOP;
- no retry.

# 16. Mechanical overall outcome

With no `REFERENCE_COVERAGE_MISSING` and at least one `REFERENCE_ELIGIBILITY_BLOCKED`:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Do not execute NEXT.

# 17. Compile / import / contract gates

Before detector calls:

## 17.1 compile

```text
<REQUIRED_PYTHON> -m py_compile scripts/task8b3_ref01_locked_reference_forensics.py
```

PASS required.

## 17.2 import-only

Import module without `main()`.

Require:
- raster root valid;
- actual detector module under EXTERNAL;
- actual imageio module under EXTERNAL;
- no detector call.

## 17.3 contract precheck

Without detector inference require:
- exact Python executable;
- source_manifest basis and 135 entries;
- external detector SHA/bytes match manifest;
- external imageio SHA/bytes match manifest;
- 4 raster SHA/readability PASS;
- all frozen constants PASS;
- 4 GT masks PASS;
- stored P1D12 proposal schemas/counts PASS.

Require detector calls still 0.

# 18. Harness execution

After all three gates pass, run the harness exactly once.

R6 authorizes:

```text
4 detector calls
= exactly 1 additional verification pass per locked candidate
```

No retry.

# 19. Canonical evidence artifact

Overwrite:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Delete:

```text
evaluation/task8b3_ref01_locked_reference_forensics_r5.json
```

Required top-level:

```text
task = 8B.3-REF01-F1-R6
starting_head
scientific_reuse_disclosure   # exact sentence from §1
python_executable
source_identity_basis = GIT_CANONICAL_BLOB_BYTES
external_detector_module_path
external_detector_sha256
external_imageio_module_path
external_imageio_sha256
runtime_checkpoint_path
runtime_default_device
detector_config
proposal_mask_source
proposals_json_role
coverage_threshold
detector_call_count = 4
cross_run_consistency
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

Preserve all historical STOP/R4/R5 sections.

Append R6 authoritative closure section.

It must explicitly state:

```text
R4 = preliminary
R5 = deterministic but contract-incomplete
R6 = authoritative only if all R6 gates pass
```

Required table:

```text
relation | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_id | best_any_IoU | full_metadata_match | class
```

Also state:
- external file identity was verified against Git-canonical `source_manifest.json`;
- Windows working-tree byte comparison is not an identity gate.

# 21. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R6
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: 53de75ae4f5246b4e18685b78bd5a9f913447984
R4 status: PRELIMINARY
R5 status: CONTRACT_INCOMPLETE
Prior detector calls total: 8
R6 detector calls authorised: 4
PY_COMPILE_GATE: PASS / FAIL
IMPORT_ONLY_GATE: PASS / FAIL
CONTRACT_PRECHECK: PASS / FAIL
Detector calls before main: 0 / other
Source identity basis: GIT_CANONICAL_BLOB_BYTES
External detector identity: PASS / FAIL
External imageio identity: PASS / FAIL
Locked raster SHA: 4/4 PASS / other
Detector runtime construction: DEFAULT / other
Runtime checkpoint: <path>
Runtime device: <value>
Coverage threshold: 0.50_TASK7F_FROZEN
Proposal mask source: RERUN_GLOBALPROPOSAL_MASK_CROP
right full metadata reproduction: PASS / FAIL
left full metadata reproduction: PASS / FAIL
above full metadata reproduction: PASS / FAIL
below full metadata reproduction: PASS / FAIL
R4/R5/R6 IoU consistency: 4/4 PASS / other
R6 detector calls: 4 / other
Qwen/SAM2/relation/D-B1/target: NONE/NONE/NONE/NONE/NONE
Manual visual inspection: NO
Candidate replacement: NO
right selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
left selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
above selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
below selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
right class: REFERENCE_SELECTED_CORRECT / other
left class: REFERENCE_SELECTION_WRONG_COVERED / other
above class: REFERENCE_ELIGIBILITY_BLOCKED / other
below class: REFERENCE_SELECTION_WRONG_COVERED / other
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / other
Outcome: REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE / other
Dominant next blocker: ELIGIBILITY / other
Next gate: REF01_ELIGIBILITY_FORENSICS / other
Canonical evidence: evaluation/task8b3_ref01_locked_reference_forensics.json
R5 temporary evidence file remaining: NO / YES
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 22. Commit / push

If COMPLETE:

```text
docs(rc1): close locked reference forensics
```

If STOP/FAILED:

```text
docs(rc1): record authoritative reference closure stop
```

Push only current task branch.
No force push.
Do not update main.

Then STOP.

# 23. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only authorized tracked paths change;
- `_r5.json` removed from final tree;
- compile/import/precheck all PASS;
- identity compared against Git-canonical source_manifest, not Windows worktree;
- actual external detector module identity PASS;
- actual external buildreasonseg.runtime.imageio identity PASS;
- 4 raster SHA exact;
- default DetectorRuntime only;
- all frozen constants exact;
- exactly 4 R6 detector calls;
- FULL 10-field proposal-by-proposal reproduction = 4/4 PASS;
- R4/R5/R6 IoUs consistent to 1e-6;
- exact scientific disclosure present;
- selected/bestEligible/bestAny IDs and facts present;
- exact classes/outcome/NEXT;
- no stage beyond detector;
- no visual judgement/candidate replacement/product repair;
- authoritative canonical evidence/report/handoff committed and pushed;
- NEXT not executed;
- STOP.
