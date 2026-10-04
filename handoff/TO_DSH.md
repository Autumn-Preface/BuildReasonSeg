# TO_DSH — Task 8B.3-REF01-F1-R8: Final Zero-Call Reference Evidence Closure

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `65643f802a0a187b93160155f976689c0b50b8c6`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

R7 is **NOT APPROVED**, but it is a safe zero-call run.

Accepted R7 facts:

```text
detector/model calls = 0
temporary _r5.json deleted
temporary _r6.json deleted
canonical evidence file was edited
R4/R5/R6 IoUs remained numerically identical
historical technical classes remained stable
```

R7 formal defects:

```text
D1. R7 incorrectly asserted that source_manifest.json had no
    buildreasonseg/runtime/imageio.py entry.

D2. R7 inspected/imported the unrelated third-party package name `imageio`
    instead of `buildreasonseg.runtime.imageio`.

D3. The tracked verifier script was NOT rewritten; it still contains:
    DetectorRuntime()
    detect_global()
    inference logic.

D4. The canonical evidence still contains the old non-mandatory disclosure:
    "GT access purpose: REFERENCE_FORENSICS_ONLY ..."
    instead of the exact frozen scientific-reuse sentence.

D5. Canonical evidence still does not contain the required replayed
    selected / best_eligible / best_any proposal IDs and facts in the final
    normalized candidate records.
```

No more detector/model inference is authorized.

# 1. Immutable technical result

The following result is already established by three historical detector runs and MUST NOT be changed:

```text
right:
selected IoU     = 0.5588697017268446
bestEligible IoU = 0.5588697017268446
bestAny IoU      = 0.5588697017268446
class            = REFERENCE_SELECTED_CORRECT

left:
selected IoU     = 0.0
bestEligible IoU = 0.6500672947510094
bestAny IoU      = 0.6500672947510094
class            = REFERENCE_SELECTION_WRONG_COVERED

above:
selected IoU     = 0.0
bestEligible IoU = 0.0
bestAny IoU      = 0.9032501889644747
class            = REFERENCE_ELIGIBILITY_BLOCKED

below:
selected IoU     = 0.0
bestEligible IoU = 0.6165496859992612
bestAny IoU      = 0.6165496859992612
class            = REFERENCE_SELECTION_WRONG_COVERED
```

Mechanical overall:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

R8 only fixes the evidence package and independently replays proposal IDs/facts from stored metadata.

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = 65643f802a0a187b93160155f976689c0b50b8c6
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No new branch.
No reset/rebase/stash/clean/merge/cherry-pick.

# 3. Absolute prohibitions

Do NOT:
- instantiate DetectorRuntime;
- call detect_global;
- call any detector/model inference;
- run predict.py;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- modify canonical RC1;
- modify external RC1;
- run sync write;
- visually inspect images;
- replace candidates;
- regenerate dataset/cache;
- change IoU threshold 0.50;
- implement product repair;
- update main;
- force push.

Required detector/model call count:

```text
0
```

# 4. Allowed tracked changes ONLY

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No new evaluation file is allowed.

`_r5.json` and `_r6.json` must remain absent.

# 5. Exact scientific-reuse disclosure

The canonical evidence and report MUST contain exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

The report must also contain a faithful Chinese translation.

# 6. Correct source_manifest contract

Read BOTH:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json
```

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
len(files) = 135
```

The canonical manifest MUST contain exactly one entry each for:

```text
buildreasonseg/runtime/detector.py
bytes = 20300
sha256 = 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738

buildreasonseg/runtime/imageio.py
bytes = 7978
sha256 = b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
```

The external manifest must contain the same two entries.

If either imageio entry is missing:
- Status = STOP
- do not substitute requirements.txt
- do not discuss third-party `imageio`
- STOP.

# 7. Correct external file identity

Hash actual external files:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\imageio.py
```

Require:

```text
detector:
bytes = 20300
sha256 = 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738

imageio:
bytes = 7978
sha256 = b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
```

Record:

```text
EXTERNAL_DETECTOR_GIT_IDENTITY = PASS
EXTERNAL_IMAGEIO_GIT_IDENTITY = PASS
```

Do NOT compare Windows working-tree bytes.

# 8. Correct package-module identity

The verifier script must set:

```python
REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")

sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))
```

Import exactly:

```python
import buildreasonseg.runtime.detector as detector_module
import buildreasonseg.runtime.imageio as imageio_module
```

Require:

```text
Path(detector_module.__file__).resolve()
under EXTERNAL

Path(imageio_module.__file__).resolve()
under EXTERNAL
```

Record the exact paths.

Do NOT import:

```python
import imageio
```

and do not inspect the third-party package namespace.

# 9. Historical R6 evidence must be read from Git history

R6 temporary evidence is deleted from the current tree by R7.

Read it without restoring it:

```text
git show 12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c:evaluation/task8b3_ref01_locked_reference_forensics_r6.json
```

Require:

```text
detector_calls = 4
identity.manifest_identity_basis = GIT_CANONICAL_BLOB_BYTES
identity.detector_matches_git_canonical = true
proposal_fields_compared contains exactly the 10 frozen fields
```

For all four candidates require:

```text
ten_field_all_match = true
within_tolerance = true
```

Use this only as historical evidence.
Do NOT restore `_r6.json`.

# 10. R4 per-proposal IoU source

Use the CURRENT canonical evidence's historical:

```text
r4_records[]
```

For each proposal obtain:

```text
relation
proposal_id
iou_to_gt
```

This is the historical proposal-to-GT IoU table from R4.

Do NOT recompute masks.
Do NOT run detector.

# 11. P1D12 proposal metadata source

Read external, read-only:

```text
inference/output/diagnostics/1010/proposals.json
inference/output/diagnostics/1003/proposals.json
inference/output/diagnostics/1008/proposals.json
inference/output/diagnostics/1009/proposals.json
```

Every item must provide:

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

Expected raw / merged / eligible counts:

```text
1010 = 6 / 6 / 4
1003 = 66 / 53 / 42
1008 = 9 / 9 / 4
1009 = 7 / 6 / 3
```

# 12. Read-only replay of production selected proposal

Frozen eligibility:

```text
mask_area > 0
AND touches_image_border == false
AND bbox_extent_ratio <= 0.20
```

Production largest ordering:

```python
(-mask_area, -confidence, proposal_id)
```

Recompute selected ID from P1D12 metadata only.

Require:

```text
right selected_id = 1
left selected_id = 14
above selected_id = 4
below selected_id = 1
```

# 13. Read-only replay of best eligible / best any

Join:

```text
R4 per-proposal IoU
+
P1D12 proposal metadata
```

by:

```text
relation + proposal_id
```

For each joined proposal retain:

```text
proposal_id
iou
confidence
mask_area
global_bbox
eligible
```

Rank `best_eligible` among eligible proposals by:

```python
(-iou, -confidence, proposal_id)
```

Rank `best_any` among all merged proposals by the same tuple.

Do NOT hard-code their IDs.

Record all required facts.

Require replayed IoUs match the immutable values in §1 within `1e-6`.

# 14. Final candidate facts

Canonical evidence must have exactly four final candidate objects.

Each must contain:

```text
sample_id
relation
tile
raster_sha256

historical_reproduction:
  r6_ten_field_all_match = true
  r4_r5_r6_iou_consistent = true

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

# 15. Final classification

Coverage threshold:

```text
0.50
```

Exactly:

```text
right = REFERENCE_SELECTED_CORRECT
left = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Then:

```text
class_counts:
REFERENCE_SELECTED_CORRECT = 1
REFERENCE_SELECTION_WRONG_COVERED = 2
REFERENCE_ELIGIBILITY_BLOCKED = 1
REFERENCE_COVERAGE_MISSING = 0

Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Do not execute NEXT.

# 16. Rewrite tracked script into true zero-call verifier

Replace:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

with the read-only verifier implementing §§6–15.

The final tracked script MUST NOT contain:

```text
DetectorRuntime(
detect_global(
model.predict(
YOLO(
```

It MUST contain:

```text
buildreasonseg.runtime.detector
buildreasonseg.runtime.imageio
GIT_CANONICAL_BLOB_BYTES
task8b3_ref01_locked_reference_forensics_r6.json
REF01_ELIGIBILITY_FORENSICS
```

The historical R6 evidence path may appear only inside the `git show` command/ref logic; the file must not be recreated.

# 17. Static no-inference gate

Run:

```text
<REQUIRED_PYTHON> -m py_compile scripts/task8b3_ref01_locked_reference_forensics.py
```

Require PASS.

Then source-text assertions:

```text
"DetectorRuntime(" absent
"detect_global(" absent
"model.predict(" absent
"YOLO(" absent
"import imageio" absent
"buildreasonseg.runtime.imageio" present
"buildreasonseg/runtime/imageio.py" present
"GIT_CANONICAL_BLOB_BYTES" present
```

Require:

```text
NO_INFERENCE_STATIC_GATE = PASS
```

# 18. Execute verifier once

Run exactly once:

```text
<REQUIRED_PYTHON> scripts/task8b3_ref01_locked_reference_forensics.py
```

Require:

```text
exit = 0
detector/model calls = 0
```

# 19. Canonical evidence schema

Overwrite only:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Top-level required:

```text
task = 8B.3-REF01-F1-R8
starting_head
scientific_reuse_disclosure
verification_mode = READ_ONLY_HISTORICAL_EVIDENCE_REPLAY
detector_model_calls_this_task = 0
source_identity_basis = GIT_CANONICAL_BLOB_BYTES

external_identity:
  detector:
    module_path
    bytes
    sha256
    manifest_match = true
  imageio:
    module_path
    bytes
    sha256
    manifest_match = true

coverage_threshold = 0.50
proposal_mask_source = HISTORICAL_R4_RERUN_GLOBALPROPOSAL_MASK_CROP
proposal_metadata_source = P1D12_PROPOSALS_JSON
historical_r6_ten_field_reproduction = 4/4 PASS
historical_r4_r5_r6_iou_consistency = 4/4 PASS
candidates
class_counts
overall_outcome
dominant_next_blocker
next_gate
```

The old `imageio_identity.manifest_declares_imageio_entry=false` structure must not remain.

# 20. Temporary evidence files

Require final tree:

```text
evaluation/task8b3_ref01_locked_reference_forensics_r5.json = ABSENT
evaluation/task8b3_ref01_locked_reference_forensics_r6.json = ABSENT
```

Do not recreate either.

# 21. Report

Update:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Preserve previous history.

Append R8 final closure section and explicitly correct R7:

```text
R7 statement "manifest declares imageio entry = False" = INCORRECT
Correct entry:
buildreasonseg/runtime/imageio.py
7978 bytes
b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
```

Required final table:

```text
relation | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_id | best_any_IoU | class
```

State:

```text
R8 detector/model calls = 0
R8 = authoritative evidence closure
```

# 22. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R8
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: 65643f802a0a187b93160155f976689c0b50b8c6
Verification mode: READ_ONLY_HISTORICAL_EVIDENCE_REPLAY
Detector/model calls this task: 0
PY_COMPILE_GATE: PASS / FAIL
NO_INFERENCE_STATIC_GATE: PASS / FAIL
Source identity basis: GIT_CANONICAL_BLOB_BYTES
source_manifest imageio entry: PRESENT / MISSING
External detector Git identity: PASS / FAIL
External imageio Git identity: PASS / FAIL
External detector module path: <path>
External imageio module path: <path>
Historical R6 10-field reproduction: 4/4 PASS / other
R4/R5/R6 IoU consistency: 4/4 PASS / other
Selected replay right/left/above/below: 1/14/4/1 / other
right selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
left selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
above selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
below selected/bestEligible/bestAny: <id,iou> / <id,iou> / <id,iou>
right class: REFERENCE_SELECTED_CORRECT / other
left class: REFERENCE_SELECTION_WRONG_COVERED / other
above class: REFERENCE_ELIGIBILITY_BLOCKED / other
below class: REFERENCE_SELECTION_WRONG_COVERED / other
Temporary _r5 evidence remaining: NO / YES
Temporary _r6 evidence remaining: NO / YES
Canonical evidence updated: YES / NO
Tracked verifier is zero-inference replay: YES / NO
Qwen/SAM2/relation/D-B1/target: NONE/NONE/NONE/NONE/NONE
Manual visual inspection: NO
Candidate replacement: NO
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / other
Outcome: REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE / other
Dominant next blocker: ELIGIBILITY / other
Next gate: REF01_ELIGIBILITY_FORENSICS / other
Report: docs/task8b3_ref01_locked_reference_forensics.md
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 23. Commit / push

If COMPLETE:

```text
docs(rc1): finalize reference evidence replay
```

If STOP/FAILED:

```text
docs(rc1): record reference evidence replay stop
```

Push only current task branch.
No force push.
Do not update main.

Then STOP.

# 24. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only five allowed tracked paths changed;
- detector/model calls = 0;
- tracked verifier contains no inference path;
- source_manifest correctly resolves imageio entry;
- actual external detector identity PASS;
- actual external buildreasonseg.runtime.imageio identity PASS;
- historical R6 ten-field evidence = 4/4 PASS;
- R4/R5/R6 IoUs = 4/4 consistent;
- production selected IDs replay correctly;
- best eligible / best any IDs and facts are actually present;
- exact scientific disclosure is present;
- canonical evidence overwritten;
- `_r5.json` and `_r6.json` remain absent;
- final classes/outcome/NEXT exact;
- report/handoff committed and pushed;
- NEXT not executed;
- STOP.
