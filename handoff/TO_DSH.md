# TO_DSH — Task 8B.3-REF01-F1-R9: Canonical Evidence Schema Closure

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `f268d03a9a70b494b7134c6b2f2647ed3468caa3`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

R8 is a safe zero-call task but is NOT formally approved.

Accepted R8 facts:

```text
detector/model calls = 0
tracked script no longer instantiates DetectorRuntime or calls detect_global
detector.py Git-canonical identity = PASS
imageio.py Git-canonical identity values were finally discovered correctly
temporary _r5.json / _r6.json remain absent
historical technical classes remain stable
```

R8 formal defects:

```text
D1. canonical evidence still contains contradictory old imageio_identity:
    manifest_declares_imageio_entry = false
    while a later imageio_module_identity block says in_manifest = true.

D2. exact mandatory scientific reuse disclosure is still absent.

D3. canonical evidence was patched/appended rather than normalized to the required final schema.

D4. tracked verifier does not read P1D12 confidence/global metadata;
    it chooses best_eligible/best_any using max(iou) only.

D5. frozen ranking requires:
    (-iou_to_gt, -confidence, proposal_id)

D6. therefore best_eligible_id for tied-IoU cases is not yet contract-level proven.

D7. final candidate objects still do not contain complete selected /
    best_eligible / best_any facts required by the closure contract.
```

NO detector/model rerun is authorized.

# 1. Immutable technical measurements

Do not change these historical measurements:

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

Historical R4/R5/R6 detector runs already established identical IoU triples.

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = f268d03a9a70b494b7134c6b2f2647ed3468caa3
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No branch creation.
No reset/rebase/stash/clean/merge/cherry-pick.

# 3. Absolute prohibitions

Do NOT:
- instantiate DetectorRuntime;
- call detect_global;
- call model.predict / YOLO / any model inference;
- run predict.py;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- modify external RC1;
- modify canonical delivery source;
- run sync write;
- inspect images visually;
- change candidates;
- regenerate datasets/caches;
- change threshold 0.50;
- implement product repair;
- update main;
- force push.

Detector/model calls this task MUST be:

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

No new evaluation file.

The following must remain absent:

```text
evaluation/task8b3_ref01_locked_reference_forensics_r5.json
evaluation/task8b3_ref01_locked_reference_forensics_r6.json
```

# 5. Exact scientific reuse disclosure

Canonical evidence and report MUST contain this exact English sentence:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

The report must also include a faithful Chinese translation.

The old string beginning:

```text
GT access purpose: REFERENCE_FORENSICS_ONLY
```

must NOT remain as the value of `scientific_reuse_disclosure`.

It may appear only in historical prose if explicitly labelled superseded.

# 6. Correct Git-canonical source identity

Read canonical and external `source_manifest.json`.

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
files count = 135
```

Resolve exactly one entry each:

```text
buildreasonseg/runtime/detector.py
bytes = 20300
sha256 = 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738

buildreasonseg/runtime/imageio.py
bytes = 7978
sha256 = b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
```

Hash actual external files and require exact match.

No requirements.txt substitution.
No third-party `imageio` package checks.

# 7. Correct module identity

Verifier must use:

```python
REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")

sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))

import buildreasonseg.runtime.detector as detector_module
import buildreasonseg.runtime.imageio as imageio_module
```

Require actual `__file__` for both modules resolves under EXTERNAL.

Importing these modules for identity is allowed.

Instantiating runtime classes is forbidden.

# 8. Historical R6 10-field evidence

Read historical R6 JSON without restoring it:

```text
git show 12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c:evaluation/task8b3_ref01_locked_reference_forensics_r6.json
```

Require:

```text
detector_calls = 4
proposal_fields_compared contains exactly:
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

For all four candidates:

```text
ten_field_all_match = true
within_tolerance = true
```

# 9. Historical R4 per-proposal IoU table

Use CURRENT canonical evidence `r4_records`.

For each proposal take only:

```text
relation
proposal_id
iou_to_gt
```

Do not recompute proposal masks.

# 10. P1D12 proposal metadata — mandatory live read

Read external, read-only:

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

Every proposal must provide:

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

# 11. Recompute production selected ID from P1D12 metadata

Eligibility for family=`largest`:

```text
mask_area > 0
AND touches_image_border == false
AND bbox_extent_ratio <= 0.20
```

Production selector ordering:

```python
(-mask_area, -confidence, proposal_id)
```

Require selected IDs:

```text
right = 1
left  = 14
above = 4
below = 1
```

Do not trust/hard-code the old selected IDs without recomputing.

# 12. Recompute best_eligible and best_any with full tie-break

Join each P1D12 proposal with the matching R4 `iou_to_gt` by:

```text
relation + proposal_id
```

Require exact proposal ID sets.

For best eligible rank eligible proposals by:

```python
(-iou_to_gt, -confidence, proposal_id)
```

For best any rank all merged proposals by:

```python
(-iou_to_gt, -confidence, proposal_id)
```

This exact ranking is mandatory.

Do NOT use:

```python
max(..., key=lambda p: p["iou_to_gt"])
```

without confidence + proposal_id tie-break.

For every selected / best eligible / best any record:

```text
proposal_id
iou
confidence
mask_area
global_bbox
eligible
```

Require their IoUs equal §1 within `1e-6`.

# 13. Final classification

Threshold:

```text
0.50
```

Exactly:

```text
right = REFERENCE_SELECTED_CORRECT
left  = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Class counts:

```text
REFERENCE_SELECTED_CORRECT = 1
REFERENCE_SELECTION_WRONG_COVERED = 2
REFERENCE_ELIGIBILITY_BLOCKED = 1
REFERENCE_COVERAGE_MISSING = 0
```

Overall:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Do not execute NEXT.

# 14. Rewrite tracked verifier fully

Replace:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

with a verifier that actually implements §§6–13.

It must:
- read the two manifests;
- verify external detector/imageio SHA;
- import `buildreasonseg.runtime.detector` and `.imageio` only for module identity;
- read historical R6 via `git show`;
- read canonical R4 IoU table;
- read all four P1D12 proposals.json files;
- recompute selected/best eligible/best any mechanically;
- write only canonical evidence.

It MUST NOT contain:

```text
DetectorRuntime(
detect_global(
model.predict(
YOLO(
```

# 15. Static gate

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
"import imageio" absent as standalone third-party import
"buildreasonseg.runtime.imageio" present
"source_manifest.json" present
"proposals.json" present
"confidence" present
"border_clearance" present
"GIT_CANONICAL_BLOB_BYTES" present
```

Require:

```text
NO_INFERENCE_STATIC_GATE = PASS
```

# 16. Execute verifier exactly once

Run:

```text
<REQUIRED_PYTHON> scripts/task8b3_ref01_locked_reference_forensics.py
```

Require:

```text
exit = 0
detector/model calls = 0
```

No retry unless process was not actually started.
Any logic/data failure -> STOP.

# 17. Replace canonical evidence schema completely

Overwrite:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Do NOT append another correction block to old malformed schema.

Required top-level keys:

```text
task
starting_head
branch
scientific_reuse_disclosure
verification_mode
detector_model_calls_this_task
source_identity_basis
external_identity
coverage_threshold
historical_reproduction
historical_iou_consistency
candidates
class_counts
overall_outcome
dominant_next_blocker
next_gate
```

Required exact values:

```text
task = 8B.3-REF01-F1-R9
verification_mode = READ_ONLY_HISTORICAL_EVIDENCE_REPLAY
detector_model_calls_this_task = 0
source_identity_basis = GIT_CANONICAL_BLOB_BYTES
coverage_threshold = 0.50
```

`external_identity` must contain ONLY the correct two module identities:

```text
detector:
  module_path
  manifest_bytes
  manifest_sha256
  external_bytes
  external_sha256
  manifest_match = true

imageio:
  module_path
  manifest_bytes
  manifest_sha256
  external_bytes
  external_sha256
  manifest_match = true
```

The following obsolete structures MUST NOT remain:

```text
imageio_identity.manifest_declares_imageio_entry
requirements_imageio_lines
installed_modules.imageio
conclusion about third-party imageio
r8_id_completion
r7_zero_call_guarantee
```

# 18. Final candidate schema

Exactly four `candidates`:

```text
right
left
above
below
```

Each:

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
  global_bbox
  eligible = true

best_any:
  proposal_id
  iou
  confidence
  mask_area
  global_bbox
  eligible

selected_to_best_eligible_iou_gap
classification
```

# 19. Report

Update:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Preserve prior history.

Append R9 final schema closure section.

State explicitly:

```text
R8 zero-call execution was safe, but its evidence schema was not authoritative.
R9 replaces rather than patches the canonical evidence schema.
R9 uses the full frozen tie-break (-IoU, -confidence, proposal_id).
```

Include exact English disclosure from §5 and Chinese translation.

Required final table:

```text
relation | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_id | best_any_IoU | class
```

# 20. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R9
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: f268d03a9a70b494b7134c6b2f2647ed3468caa3
Verification mode: READ_ONLY_HISTORICAL_EVIDENCE_REPLAY
Detector/model calls this task: 0
PY_COMPILE_GATE: PASS / FAIL
NO_INFERENCE_STATIC_GATE: PASS / FAIL
Source identity basis: GIT_CANONICAL_BLOB_BYTES
External detector Git identity: PASS / FAIL
External imageio Git identity: PASS / FAIL
External detector module path: <path>
External imageio module path: <path>
Historical R6 10-field reproduction: 4/4 PASS / other
R4/R5/R6 IoU consistency: 4/4 PASS / other
P1D12 metadata live read: 4/4 PASS / other
Selected replay right/left/above/below: 1/14/4/1 / other
Best eligible replay right/left/above/below: <ids>
Best any replay right/left/above/below: <ids>
right class: REFERENCE_SELECTED_CORRECT / other
left class: REFERENCE_SELECTION_WRONG_COVERED / other
above class: REFERENCE_ELIGIBILITY_BLOCKED / other
below class: REFERENCE_SELECTION_WRONG_COVERED / other
Canonical evidence schema replaced: YES / NO
Obsolete imageio false-claim fields remaining: NO / YES
Exact scientific reuse disclosure present: YES / NO
Temporary _r5 evidence remaining: NO / YES
Temporary _r6 evidence remaining: NO / YES
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

# 21. Commit / push

If COMPLETE:

```text
docs(rc1): close canonical reference evidence
```

If STOP/FAILED:

```text
docs(rc1): record canonical reference evidence stop
```

Push only current task branch.
No force push.
Do not update main.

Then STOP.

# 22. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only five allowed tracked paths changed;
- detector/model calls = 0;
- tracked verifier is truly zero-inference;
- source_manifest correctly resolves detector.py and imageio.py;
- both external module identities PASS;
- historical R6 ten-field evidence = 4/4 PASS;
- P1D12 metadata live read = 4/4 PASS;
- selected IDs replay from production ordering;
- best eligible / best any use full (-IoU,-confidence,proposal_id) tie-break;
- exact scientific disclosure present;
- canonical evidence is replaced with clean final schema;
- contradictory old imageio structures are absent;
- `_r5.json` and `_r6.json` remain absent;
- final classes/outcome/NEXT exact;
- report/handoff committed and pushed;
- NEXT not executed;
- STOP.
