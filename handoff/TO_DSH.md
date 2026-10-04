# TO_DSH — Task 8B.3-REF01-F1-R10: Final Independent Zero-Call Verifier

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `367144990e510aaacbae2b545a08d99aff91776a`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

R9 materially improved the evidence package and its technical conclusions remain accepted.

R9 still is NOT the final formal closure because the tracked verifier does not independently verify all facts it reports.

The current R9 verifier:
- reads live P1D12 metadata for best_eligible / best_any;
- uses the correct `(-IoU, -confidence, proposal_id)` tie-break;
- contains no detector/model inference;

but it does NOT independently verify:
- source_manifest identity basis / detector.py / imageio.py hashes;
- actual `buildreasonseg.runtime.detector` / `.imageio` module paths;
- historical R6 ten-field reproduction via `git show`;
- production `selected_id` by replaying `(-mask_area, -confidence, proposal_id)`;
- all selected / best_eligible / best_any facts against canonical evidence.

R10 fixes only the verifier and records the verifier PASS.

NO detector/model rerun is authorized.

# 1. Frozen technical result

The following is immutable:

```text
right:
selected=1
bestEligible=1
bestAny=1
class=REFERENCE_SELECTED_CORRECT

left:
selected=14
bestEligible=30
bestAny=30
class=REFERENCE_SELECTION_WRONG_COVERED

above:
selected=4
bestEligible=4
bestAny=5
class=REFERENCE_ELIGIBILITY_BLOCKED

below:
selected=1
bestEligible=2
bestAny=2
class=REFERENCE_SELECTION_WRONG_COVERED
```

Frozen IoUs:

```text
right  = 0.5588697017268446 / 0.5588697017268446 / 0.5588697017268446
left   = 0.0 / 0.6500672947510094 / 0.6500672947510094
above  = 0.0 / 0.0 / 0.9032501889644747
below  = 0.0 / 0.6165496859992612 / 0.6165496859992612
```

Overall:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = 367144990e510aaacbae2b545a08d99aff91776a
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No branch creation.
No merge/rebase/reset/stash/clean/cherry-pick.

# 3. Absolute prohibitions

Do NOT:
- instantiate `DetectorRuntime`;
- call `detect_global`;
- run any model inference;
- run `predict.py`;
- run Qwen / SAM2 / relation fields / D-B1 / target segmentation;
- modify external RC1;
- modify canonical delivery source;
- run sync write;
- change candidate set;
- change threshold 0.50;
- change canonical technical measurements;
- implement product repair;
- update main;
- force push.

Detector/model calls this task:

```text
0
```

# 4. Allowed tracked changes ONLY

```text
scripts/task8b3_ref01_locked_reference_forensics.py
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do NOT modify:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

unless the verifier discovers a contradiction, in which case STOP instead of editing it.

Do not recreate `_r5.json` or `_r6.json`.

# 5. Final verifier responsibilities

Rewrite:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

into a pure read-only independent verifier.

It must independently verify all §§6–11 and return non-zero on any mismatch.

# 6. Source manifest and external module identity

Read:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json
```

Require both:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
len(files) = 135
```

Require exact single entries:

```text
buildreasonseg/runtime/detector.py
bytes = 20300
sha256 = 82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738

buildreasonseg/runtime/imageio.py
bytes = 7978
sha256 = b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
```

Hash actual external files and require exact match.

Then set:

```python
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))
```

Import only for identity:

```python
import buildreasonseg.runtime.detector as detector_module
import buildreasonseg.runtime.imageio as imageio_module
```

Require both `__file__` paths under EXTERNAL.

Do NOT instantiate any runtime object.

# 7. Historical R6 ten-field verification from Git history

Read without restoring:

```text
git show 12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c:evaluation/task8b3_ref01_locked_reference_forensics_r6.json
```

Require:

```text
detector_calls = 4
proposal_fields_compared exactly =
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

For all four candidates require:

```text
ten_field_all_match = true
within_tolerance = true
```

# 8. Canonical evidence contract

Read only:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Require exact:

```text
task = 8B.3-REF01-F1-R9
verification_mode = READ_ONLY_HISTORICAL_EVIDENCE_REPLAY
detector_model_calls_this_task = 0
source_identity_basis = GIT_CANONICAL_BLOB_BYTES
coverage_threshold = 0.50
```

Require exact English disclosure:

```text
The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.
```

Require obsolete keys absent from the entire parsed object:

```text
manifest_declares_imageio_entry
requirements_imageio_lines
installed_modules
r7_zero_call_guarantee
r8_id_completion
```

Require external detector/imageio identities match §6.

# 9. Live P1D12 metadata replay

Read, read-only:

```text
external\inference\output\diagnostics\1010\proposals.json
external\inference\output\diagnostics\1003\proposals.json
external\inference\output\diagnostics\1008\proposals.json
external\inference\output\diagnostics\1009\proposals.json
```

Require counts:

```text
1010 = raw6 / merged6 / eligible4
1003 = raw66 / merged53 / eligible42
1008 = raw9 / merged9 / eligible4
1009 = raw7 / merged6 / eligible3
```

Require every item contains:

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

# 10. Replay selected / best eligible / best any

Join canonical:

```text
historical_iou_by_proposal[relation]
```

to live P1D12 metadata by `proposal_id`.

Frozen eligibility:

```text
mask_area > 0
AND touches_image_border == false
AND bbox_extent_ratio <= 0.20
```

Production selected ordering:

```python
(-mask_area, -confidence, proposal_id)
```

Best eligible / best any ordering:

```python
(-iou_to_gt, -confidence, proposal_id)
```

Require exact IDs:

```text
right: selected 1 / bestEligible 1 / bestAny 1
left:  selected 14 / bestEligible 30 / bestAny 30
above: selected 4 / bestEligible 4 / bestAny 5
below: selected 1 / bestEligible 2 / bestAny 2
```

For each of all 12 proposal-role records, compare canonical evidence against live metadata for:

```text
proposal_id
confidence
mask_area
global_bbox
eligible
```

and compare IoU against historical IoU table within `1e-12`.

Any mismatch = FAIL.

# 11. Classification / aggregate replay

Using threshold 0.50 require:

```text
right = REFERENCE_SELECTED_CORRECT
left = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Require canonical class counts:

```text
REFERENCE_SELECTED_CORRECT = 1
REFERENCE_SELECTION_WRONG_COVERED = 2
REFERENCE_ELIGIBILITY_BLOCKED = 1
REFERENCE_COVERAGE_MISSING = 0
```

Require:

```text
overall_outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
dominant_next_blocker = ELIGIBILITY
next_gate = REF01_ELIGIBILITY_FORENSICS
```

# 12. Static zero-inference gate

Run:

```text
<REQUIRED_PYTHON> -m py_compile scripts/task8b3_ref01_locked_reference_forensics.py
```

Require PASS.

Source-text require absent:

```text
DetectorRuntime(
detect_global(
model.predict(
YOLO(
```

Require present:

```text
buildreasonseg.runtime.detector
buildreasonseg.runtime.imageio
source_manifest.json
git show
proposals.json
confidence
border_clearance
GIT_CANONICAL_BLOB_BYTES
```

Require:

```text
NO_INFERENCE_STATIC_GATE = PASS
```

# 13. Execute verifier exactly once

Run:

```text
<REQUIRED_PYTHON> scripts/task8b3_ref01_locked_reference_forensics.py
```

Require exit 0.

Required stdout summary:

```text
SOURCE_IDENTITY: PASS
R6_TEN_FIELD_HISTORY: PASS
P1D12_METADATA: PASS
ROLE_REPLAY: PASS
CLASSIFICATION_REPLAY: PASS
FINAL_VERIFIER: PASS
```

No second run unless process never started.

# 14. Report

Update:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Append R10 section only.

State:

```text
R9 canonical evidence is unchanged.
R10 independently verifies the R9 evidence from:
- Git-canonical manifests and external files;
- historical R6 evidence via git show;
- live P1D12 proposals.json metadata;
- historical R4 per-proposal IoU table.

R10 detector/model calls = 0.
FINAL_VERIFIER = PASS.
```

# 15. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R10
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: 367144990e510aaacbae2b545a08d99aff91776a
Detector/model calls this task: 0
PY_COMPILE_GATE: PASS / FAIL
NO_INFERENCE_STATIC_GATE: PASS / FAIL
Source identity verification: PASS / FAIL
External detector module identity: PASS / FAIL
External imageio module identity: PASS / FAIL
Historical R6 ten-field verification: PASS / FAIL
Live P1D12 metadata verification: 4/4 PASS / other
Selected replay right/left/above/below: 1/14/4/1 / other
Best eligible replay right/left/above/below: 1/30/4/2 / other
Best any replay right/left/above/below: 1/30/5/2 / other
12 role-record fact comparisons: PASS / FAIL
Classification replay: 4/4 PASS / other
Aggregate outcome replay: PASS / FAIL
Canonical evidence modified this task: NO / YES
Temporary _r5/_r6 files remaining: NO / YES
FINAL_VERIFIER: PASS / FAIL
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / other
Outcome: REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE / other
Dominant next blocker: ELIGIBILITY / other
Next gate: REF01_ELIGIBILITY_FORENSICS / other
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 16. Commit / push

If COMPLETE:

```text
docs(rc1): verify final reference evidence
```

If STOP/FAILED:

```text
docs(rc1): record final reference verifier stop
```

Push only current branch.
No force push.
Do not update main.

Then STOP.

# 17. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only four allowed tracked paths changed;
- detector/model calls = 0;
- verifier contains no inference path;
- source identities verified independently;
- historical R6 ten-field evidence verified independently;
- live P1D12 metadata read 4/4;
- selected/bestEligible/bestAny all independently replayed with exact frozen ranking;
- all 12 role-record facts match canonical evidence;
- classifications and aggregate outcome replay;
- canonical evidence unchanged;
- temporary evidence files absent;
- report/handoff committed and pushed;
- NEXT not executed;
- STOP.
