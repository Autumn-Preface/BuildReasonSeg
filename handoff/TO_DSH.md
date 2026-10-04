# TO_DSH — Task 8B.3-REF01-F1-R7: Read-Only Authoritative Evidence Normalization

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

R6 is **NOT formally approved as a task artifact closure**, but its numerical/metadata observations are retained as historical evidence.

Accepted R6 observations:

```text
detector calls = 4
raster SHA locks = 4/4 MATCH
10-field reproduction summary = 4/4 FULL MATCH
R4/R5/R6 selected/bestEligible/bestAny IoU deltas = 0.0 for all four
external detector Git-canonical identity = PASS
```

R6 formal defects:

```text
1. `buildreasonseg.runtime.imageio` was not actually verified; the evidence again checked third-party `imageio`.
2. The canonical evidence file was NOT overwritten.
3. A second unauthorized temporary evidence file `_r6.json` was created.
4. `_r5.json` was not removed.
5. The tracked forensic script was not updated to the R6 contract.
6. Best-eligible / best-any proposal IDs and full proposal facts were still absent.
7. Exact mandatory scientific-reuse disclosure was absent from canonical evidence.
```

## Critical decision

NO MORE DETECTOR RERUNS are authorized for REF01-F1.

The technical reference-forensics result has already reproduced identically across R4, R5 and R6.
R7 is a **read-only evidence normalization and independent replay from stored metadata only**.

# 1. Frozen technical result entering R7

Historical detector-run evidence:

```text
R4 = preliminary detector run
R5 = deterministic verification detector run
R6 = final detector verification run

R4/R5/R6 IoUs = identical within 1e-6 (actual recorded deltas = 0.0)
```

Frozen values:

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

Frozen classes:

```text
right = REFERENCE_SELECTED_CORRECT
left  = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Frozen overall:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

R7 must independently replay/verify those facts using only existing evidence + P1D12 stored proposal metadata.

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = 12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No branch creation.
No reset/rebase/stash/clean/merge/cherry-pick.

# 3. Absolute prohibitions

Do NOT:
- instantiate `DetectorRuntime`;
- call `detect_global`;
- call any detector/model inference;
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- run training/export/download;
- modify canonical RC1;
- modify external RC1;
- run sync write;
- visually inspect images/proposal PNGs;
- replace candidates;
- regenerate datasets/caches;
- recompute GT masks from model outputs;
- change threshold 0.50;
- implement any product repair;
- update main;
- force push.

R7 model/detector call count must be exactly:

```text
0
```

# 4. Allowed tracked changes ONLY

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
evaluation/task8b3_ref01_locked_reference_forensics_r5.json   # DELETE ONLY
evaluation/task8b3_ref01_locked_reference_forensics_r6.json   # DELETE ONLY
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other tracked path may change.

# 5. Exact scientific-reuse disclosure

The final canonical evidence and report MUST contain exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Also include a faithful Chinese translation in the report.

# 6. Correct Git-canonical external identity gate

Read external:

```text
source_manifest.json
```

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
entry count = 135
```

Require manifest entries:

```text
buildreasonseg/runtime/detector.py
bytes = 20300
sha256 = 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738

buildreasonseg/runtime/imageio.py
bytes = 7978
sha256 = b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
```

Hash the actual external files.

Require actual external bytes/sha equal the manifest entries.

Do NOT compare either file against the Windows working-tree copy.

Record:

```text
EXTERNAL_DETECTOR_GIT_IDENTITY = PASS
EXTERNAL_IMAGEIO_GIT_IDENTITY = PASS
```

# 7. Correct external module import gate

The read-only verifier script must set:

```python
REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")

sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))
```

Then:

```python
import buildreasonseg.runtime.detector as detector_module
import buildreasonseg.runtime.imageio as imageio_module
```

Require:

```text
Path(detector_module.__file__).resolve() under EXTERNAL
Path(imageio_module.__file__).resolve() under EXTERNAL
```

This task MUST NOT import the third-party package named `imageio`.

Record exact resolved paths.

# 8. Historical evidence inputs

Before deleting temporary files, read:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json      # R4 canonical evidence
evaluation/task8b3_ref01_locked_reference_forensics_r5.json   # R5 temporary evidence
evaluation/task8b3_ref01_locked_reference_forensics_r6.json   # R6 temporary evidence
```

Require R6:

```text
manifest_identity_basis = GIT_CANONICAL_BLOB_BYTES
detector_matches_git_canonical = true
detector_calls = 4
proposal_fields_compared =
[
 proposal_id,
 source_tile_id,
 confidence,
 mask_area,
 global_bbox,
 centroid,
 touches_image_border,
 border_clearance,
 bbox_extent_ratio,
 raw_index
]
```

Require all four R6 records:

```text
raster_lock_match = true
ten_field_all_match = true
within_tolerance = true
```

Require all 10 field-match counts equal total proposal count for every candidate.

Require R4, R5 and R6 IoU triples agree to <= 1e-6 with the frozen values in §1.

Any contradiction:
- STOP;
- do not normalize evidence.

# 9. Existing P1D12 proposal metadata — read only

Read external:

```text
inference/output/diagnostics/1010/proposals.json
inference/output/diagnostics/1003/proposals.json
inference/output/diagnostics/1008/proposals.json
inference/output/diagnostics/1009/proposals.json
```

Expected:

```text
1010 raw=6  merged=6  eligible=4
1003 raw=66 merged=53 eligible=42
1008 raw=9  merged=9  eligible=4
1009 raw=7  merged=6  eligible=3
```

For every proposal item require:

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

No external file is modified.

# 10. Read-only independent selection replay

The frozen largest-reference policy is:

```text
eligible if:
mask_area > 0
AND touches_image_border == false
AND bbox_extent_ratio <= 0.20
```

Production selected ordering:

```python
(-mask_area, -confidence, proposal_id)
```

For each candidate, recompute the selected proposal ID from P1D12 stored metadata only.

Require exact historical selected IDs:

```text
right = 1
left = 14
above = 4
below = 1
```

If any differs:
- STOP.

# 11. Read-only IoU table replay

Use the per-proposal `iou_to_gt` table stored in the R4 canonical evidence.

Map by:

```text
relation + proposal_id
```

Join it with the corresponding P1D12 stored proposal metadata.

Do NOT recompute proposal masks.
Do NOT run detector.

For each proposal, the joined record contains:
- proposal_id;
- IoU from R4;
- confidence from P1D12;
- mask_area from P1D12;
- eligibility from P1D12.

Compute `best_eligible` over eligible proposals by:

```python
(-iou_to_gt, -confidence, proposal_id)
```

Compute `best_any` over all proposals by the same ranking.

Record IDs and facts.

Require the replayed IoUs exactly match §1 within 1e-6.

# 12. Mechanical final classification

Coverage threshold remains:

```text
0.50
```

Classification:

```text
REFERENCE_SELECTED_CORRECT
if selected_iou >= 0.50

REFERENCE_SELECTION_WRONG_COVERED
if selected_iou < 0.50 and best_eligible_iou >= 0.50

REFERENCE_ELIGIBILITY_BLOCKED
if selected_iou < 0.50
and best_eligible_iou < 0.50
and best_any_iou >= 0.50

REFERENCE_COVERAGE_MISSING
if best_any_iou < 0.50
```

Require exact:

```text
right = REFERENCE_SELECTED_CORRECT
left  = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Then mechanically:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Do not execute NEXT.

# 13. Rewrite the tracked forensic script as a READ-ONLY evidence verifier

Update:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

The final tracked script must implement §§6–12 using existing evidence/metadata only.

It MUST NOT contain or call:

```text
DetectorRuntime(
detect_global(
model.predict(
predict.py
```

It may import `buildreasonseg.runtime.detector` and `.imageio` only to establish module identity and to record frozen
constant values.

The script writes only:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

No external write.

# 14. Static no-inference gate

Before executing the verifier, run:

```text
<REQUIRED_PYTHON> -m py_compile scripts/task8b3_ref01_locked_reference_forensics.py
```

Require PASS.

Then inspect source text and require:

```text
"DetectorRuntime(" NOT present
"detect_global(" NOT present
"model.predict(" NOT present
"import imageio" NOT present
"buildreasonseg.runtime.imageio" present
"GIT_CANONICAL_BLOB_BYTES" present
"REFERENCE_ELIGIBILITY_BLOCKED" present
```

Record:

```text
NO_INFERENCE_STATIC_GATE = PASS
```

# 15. Execute read-only verifier exactly once

Run exactly once:

```text
<REQUIRED_PYTHON> scripts/task8b3_ref01_locked_reference_forensics.py
```

Require:

```text
exit = 0
detector/model calls = 0
```

# 16. Final canonical evidence

Overwrite only:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Required top-level:

```text
task = 8B.3-REF01-F1-R7
starting_head
scientific_reuse_disclosure
verification_mode = READ_ONLY_HISTORICAL_EVIDENCE_REPLAY
detector_calls_this_task = 0
source_identity_basis = GIT_CANONICAL_BLOB_BYTES
external_detector_module_path
external_detector_sha256
external_imageio_module_path
external_imageio_sha256
coverage_threshold = 0.50
historical_runs_consistency
proposal_reproduction_evidence
candidates
class_counts
overall_outcome
dominant_next_blocker
next_gate
```

For each candidate require:

```text
sample_id / relation / tile
raster SHA
historical R4/R5/R6 IoUs
historical 10-field reproduction PASS
selected:
  proposal_id
  iou
  confidence
  mask_area
  global_bbox

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

# 17. Delete temporary evidence artifacts

After the canonical verifier succeeds and canonical evidence has been written, delete:

```text
evaluation/task8b3_ref01_locked_reference_forensics_r5.json
evaluation/task8b3_ref01_locked_reference_forensics_r6.json
```

Require both absent from final tree.

Do NOT delete their Git history.

# 18. Report

Update:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Preserve R4/R5/R6 history.

Append R7 authoritative normalization section stating clearly:

```text
R4 = preliminary detector evidence
R5 = stable deterministic evidence but contract-incomplete
R6 = full 10-field historical reproduction evidence but artifact/import packaging incomplete
R7 = no-inference authoritative evidence replay and normalization
```

Required final table:

```text
relation | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_id | best_any_IoU | class
```

State:
- external detector/imageio identity verified against Git-canonical source_manifest;
- no Windows working-tree identity comparison;
- R7 ran zero detector/model calls.

# 19. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R7
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: 12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c
Verification mode: READ_ONLY_HISTORICAL_EVIDENCE_REPLAY
Detector/model calls this task: 0
PY_COMPILE_GATE: PASS / FAIL
NO_INFERENCE_STATIC_GATE: PASS / FAIL
Source identity basis: GIT_CANONICAL_BLOB_BYTES
External detector Git identity: PASS / FAIL
External imageio Git identity: PASS / FAIL
External detector module path: <path>
External imageio module path: <path>
R6 10-field reproduction evidence: 4/4 PASS / other
R4/R5/R6 IoU consistency: 4/4 PASS / other
Read-only selected replay right/left/above/below: 1/14/4/1 / other
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

# 20. Commit / push

If COMPLETE:

```text
docs(rc1): normalize authoritative reference evidence
```

If STOP/FAILED:

```text
docs(rc1): record reference evidence normalization stop
```

Push only current branch.
No force push.
Do not update main.

Then STOP.

# 21. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only allowed tracked paths changed;
- zero detector/model calls;
- tracked verifier contains no detector inference path;
- detector and `buildreasonseg.runtime.imageio` identities PASS against Git-canonical manifest;
- R6 10-field evidence structurally validates 4/4;
- R4/R5/R6 IoUs replay consistently 4/4;
- production selected IDs replay mechanically from P1D12 metadata;
- best eligible/best any IDs and facts replay mechanically;
- final four classes/outcome/NEXT exact;
- exact scientific disclosure present;
- canonical evidence overwritten;
- temporary `_r5.json` and `_r6.json` deleted;
- report/handoff committed and pushed;
- NEXT not executed;
- STOP.
