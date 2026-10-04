# TO_DSH — Task 8B.3-REF01-F1-R12: Minimal Assertion-Only Reference Closure Verifier

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `d5341c5e45915e5b0877ebbd1ddbb48f467e7eeb`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

R11 is a safe zero-call run but is NOT formally approved.

The technical REF-01 result is already frozen and MUST NOT change.

R11 verifier defects to fix:

```text
D1 selected was NOT replayed from live P1D12 metadata by
   (-mask_area, -confidence, proposal_id).

D2 "12/12 role facts" compared only (proposal_id, IoU), not:
   confidence, mask_area, global_bbox, eligible.

D3 historical R6 verification did NOT assert:
   detector_calls = 4
   exact 10-field proposal_fields_compared
   within_tolerance = true for all four.

D4 exact scientific reuse disclosure equality was not asserted.

D5 aggregate Outcome / blocker / NEXT was not independently replayed.

D6 source identity did not assert BOTH canonical/external manifest:
   schema, identity_basis, file count = 135.
```

R12 is assertion-only. No detector/model execution. No canonical evidence modification.

# 1. Frozen expected result

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

Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

# 2. Git gate

Require:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = d5341c5e45915e5b0877ebbd1ddbb48f467e7eeb
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No branch creation / merge / rebase / reset / stash / clean / cherry-pick.

# 3. Absolute prohibitions

Do NOT:
- instantiate `DetectorRuntime`;
- call `detect_global`;
- call model inference of any kind;
- run `predict.py`;
- run Qwen / ProgramHead / SAM2 / relation fields / D-B1 / target segmentation;
- modify canonical evidence;
- modify external RC1;
- modify canonical delivery source;
- run sync write;
- change candidates / threshold / frozen results;
- update main;
- force push.

Detector/model calls = exactly 0.

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

Do NOT recreate `_r5.json` / `_r6.json`.

# 5. Replace verifier with direct assertions

Rewrite:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
```

The verifier must perform the checks below directly. Do not substitute a weaker equivalent.

## 5.1 Exact disclosure

Require:

```python
EXACT_DISCLOSURE = (
    "The qualitative Demo candidates are deterministically selected from the frozen "
    "BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture "
    "test metrics were already consumed. Their qualitative reuse does not alter, "
    "replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture."
)

assert evidence["scientific_reuse_disclosure"]["en"] == EXACT_DISCLOSURE
```

No prefix/substring match.

## 5.2 Both manifests

Read:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json
```

For BOTH require:

```text
schema == BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis == GIT_CANONICAL_BLOB_BYTES
len(files) == 135
```

Require exactly one entry each for:

```text
buildreasonseg/runtime/detector.py
20300
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738

buildreasonseg/runtime/imageio.py
7978
b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
```

Canonical and external manifest entries must be identical.

Hash actual external files and require exact bytes/SHA.

Import:

```python
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))

import buildreasonseg.runtime.detector as detector_module
import buildreasonseg.runtime.imageio as imageio_module
```

Require exact resolved `__file__` paths equal:

```text
EXTERNAL/buildreasonseg/runtime/detector.py
EXTERNAL/buildreasonseg/runtime/imageio.py
```

No runtime object construction.

## 5.3 Exact historical R6 evidence

Do not search history heuristically.

Read exactly:

```text
git -C <REPO> show 12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c:evaluation/task8b3_ref01_locked_reference_forensics_r6.json
```

Require parsed JSON.

Require:

```text
detector_calls == 4
```

Require `proposal_fields_compared` as an exact ordered list:

```text
[
 "proposal_id",
 "source_tile_id",
 "confidence",
 "mask_area",
 "global_bbox",
 "centroid",
 "touches_image_border",
 "border_clearance",
 "bbox_extent_ratio",
 "raw_index"
]
```

For every one of four result records require:

```text
ten_field_all_match is True
within_tolerance is True
```

Require exactly 4 result records.

## 5.4 Live P1D12 counts and field schema

Read for tiles 1010 / 1003 / 1008 / 1009:

```text
result.json
proposals.json
```

Require exact:

```text
1010 = raw6 / merged6 / eligible4
1003 = raw66 / merged53 / eligible42
1008 = raw9 / merged9 / eligible4
1009 = raw7 / merged6 / eligible3
```

Every proposal must contain all:

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

Eligibility:

```python
eligible = (
    mask_area > 0
    and touches_image_border is False
    and bbox_extent_ratio <= 0.20
)
```

## 5.5 LIVE production selected replay

For every candidate compute selected from the LIVE eligible P1D12 proposals:

```python
selected = min(
    eligible_proposals,
    key=lambda p: (
        -int(p["mask_area"]),
        -float(p["confidence"]),
        int(p["proposal_id"]),
    ),
)
```

Require:

```text
right=1
left=14
above=4
below=1
```

Do NOT use historical R4/R5/R6 selected IDs to determine selected.
Historical selected IDs may only be an extra consistency check after live replay.

## 5.6 bestEligible / bestAny replay

Join live P1D12 proposals with:

```text
evidence["historical_iou_by_proposal"][relation]
```

Require exact proposal-ID-set equality.

Ranking:

```python
rank_key = lambda p: (
    -float(p["_iou"]),
    -float(p["confidence"]),
    int(p["proposal_id"]),
)
```

Require:

```text
right: bestEligible=1 bestAny=1
left:  bestEligible=30 bestAny=30
above: bestEligible=4 bestAny=5
below: bestEligible=2 bestAny=2
```

## 5.7 12/12 role facts — FULL comparison

For each candidate and each role:

```text
selected
best_eligible
best_any
```

Construct an independently replayed role record from LIVE P1D12 metadata + historical IoU.

Compare canonical evidence.

Exact:
```text
proposal_id
mask_area
global_bbox
eligible
```

Tolerance <= 1e-12:
```text
confidence
iou
```

Required:

```text
12/12 PASS
```

Do not count only `(proposal_id, IoU)` as a role-fact PASS.

## 5.8 Classification replay

Using IoU threshold 0.50 require:

```text
right = REFERENCE_SELECTED_CORRECT
left = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Require evidence matches.

Recompute class counts. Treat absent coverage key as zero only for comparison.

## 5.9 Aggregate replay

From recomputed classes, using priority:

```text
COVERAGE > ELIGIBILITY > SELECTION > ALL_CORRECT
```

derive independently.

Require exactly:

```text
overall_outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
dominant_next_blocker = ELIGIBILITY
next_gate = REF01_ELIGIBILITY_FORENSICS
```

Require canonical evidence matches.

# 6. Static gate

Run:

```text
<REQUIRED_PYTHON> -m py_compile scripts/task8b3_ref01_locked_reference_forensics.py
```

Require PASS.

Source text must not contain executable calls:

```text
DetectorRuntime(
detect_global(
model.predict(
YOLO(
```

Require source contains:

```text
12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c
proposal_fields_compared
buildreasonseg.runtime.detector
buildreasonseg.runtime.imageio
mask_area
confidence
global_bbox
eligible
overall_outcome
dominant_next_blocker
next_gate
```

# 7. Execute exactly once

Run:

```text
<REQUIRED_PYTHON> scripts/task8b3_ref01_locked_reference_forensics.py
```

Require exit 0.

Required stdout:

```text
DISCLOSURE_EXACT: PASS
SOURCE_MANIFESTS: PASS
MODULE_FILE_IDENTITY: PASS
R6_EXACT_HISTORY: PASS
P1D12_COUNTS_AND_SCHEMA: PASS
SELECTED_LIVE_REPLAY: PASS
BEST_ROLE_REPLAY: PASS
ROLE_FACTS_FULL_12_OF_12: PASS
CLASSIFICATION_REPLAY: PASS
AGGREGATE_REPLAY: PASS
FINAL_VERIFIER: PASS
detector_or_model_calls = 0
canonical_evidence_modified = false
```

# 8. Report

Append R12 section to:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

State:
- R11 safe but verifier-incomplete;
- R12 is assertion-only;
- live selected replay uses `(-mask_area,-confidence,proposal_id)`;
- R6 exact commit/path used;
- role facts = six fields per role, 12/12;
- aggregate replay PASS;
- canonical evidence unchanged;
- detector/model calls 0.

# 9. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R12
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: d5341c5e45915e5b0877ebbd1ddbb48f467e7eeb
Detector/model calls this task: 0
Canonical evidence modified: NO / YES
PY_COMPILE_GATE: PASS / FAIL
NO_INFERENCE_STATIC_GATE: PASS / FAIL
Exact disclosure equality: PASS / FAIL
Canonical manifest contract: PASS / FAIL
External manifest contract: PASS / FAIL
External detector module __file__: PASS / FAIL
External imageio module __file__: PASS / FAIL
R6 exact git-show detector_calls=4: PASS / FAIL
R6 exact 10-field list: PASS / FAIL
R6 within_tolerance 4/4: PASS / FAIL
P1D12 counts/schema: 4/4 PASS / other
Live selected replay right/left/above/below: 1/14/4/1 / other
Best eligible replay right/left/above/below: 1/30/4/2 / other
Best any replay right/left/above/below: 1/30/5/2 / other
Full role facts: 12/12 PASS / other
Classification replay: 4/4 PASS / other
Aggregate replay: PASS / FAIL
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

# 10. Commit / push

If COMPLETE:

```text
docs(rc1): assert final reference closure
```

If STOP/FAILED:

```text
docs(rc1): record assertion verifier stop
```

Push only current branch.
No force push.
Do not update main.

Then STOP.

# 11. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only four allowed tracked paths changed;
- detector/model calls 0;
- canonical evidence unchanged;
- exact disclosure equality PASS;
- both manifests fully verified;
- module __file__ identities verified;
- exact R6 commit/path verified;
- R6 detector_calls / 10-field list / 4 within_tolerance flags verified;
- live P1D12 counts/schema verified;
- selected independently replayed from LIVE metadata;
- bestEligible/bestAny independently replayed;
- full six-field role comparison = 12/12 PASS;
- classifications replayed;
- aggregate Outcome/blocker/NEXT replayed;
- temporary evidence files absent;
- report/handoff committed and pushed;
- NEXT not executed;
- STOP.
