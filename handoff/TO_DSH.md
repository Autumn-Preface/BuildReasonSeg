# TO_DSH — Task 8B.3-REF01-F1-R11: Strict Independent Reference Closure Verifier

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes mechanically.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `7d9492fa7628c4b4cc379a3c4ba976c3bd1fe228`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit disposition

R10 is a safe zero-call verification attempt, but it is NOT yet a complete independent verifier.

Accepted R10 facts:

```text
branch/head chain = correct
detector/model calls = 0
canonical evidence was not modified
main remained unchanged
tracked verifier contains no DetectorRuntime/detect_global inference path
bestEligible/bestAny live metadata replay still matches the canonical evidence
```

R10 formal defects:

```text
D1. It does not import buildreasonseg.runtime.detector and
    buildreasonseg.runtime.imageio and therefore does not verify their actual
    __file__ paths resolve under the external RC1 root.

D2. It does not read the historical R6 evidence with `git show`.
    It only trusts the already-normalized canonical evidence boolean:
    historical_reproduction.r6_ten_field_all_match.

D3. It does not replay production selected_id from live P1D12 metadata using:
    (-mask_area, -confidence, proposal_id).
    It takes selected_iou and selected role facts directly from canonical evidence.

D4. It does not compare the complete 12 role records
    (selected / best_eligible / best_any × 4 candidates)
    against live metadata field-by-field.

D5. It does not independently validate the required P1D12 raw/merged/eligible
    counts against the frozen values.

D6. It does not independently replay/verify the aggregate:
    overall_outcome / dominant_next_blocker / next_gate.

D7. It checks only a disclosure prefix (`startswith`) rather than the exact frozen
    English sentence required by the contract.
```

No detector/model run is authorized in R11.

# 1. Frozen technical result

Do not change:

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

Frozen aggregate:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

# 2. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = 7d9492fa7628c4b4cc379a3c4ba976c3bd1fe228
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No branch creation.
No merge/rebase/reset/stash/clean/cherry-pick.

# 3. Absolute prohibitions

Do NOT:
- instantiate DetectorRuntime;
- call detect_global;
- call any model inference;
- run predict.py;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- modify canonical evidence;
- modify external RC1;
- modify canonical delivery source;
- run sync write;
- change candidate set;
- change threshold 0.50;
- change any frozen technical result;
- update main;
- force push.

Detector/model calls:

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

Do NOT recreate:

```text
evaluation/task8b3_ref01_locked_reference_forensics_r5.json
evaluation/task8b3_ref01_locked_reference_forensics_r6.json
```

# 5. Exact disclosure contract

The verifier must require exact equality to:

```text
The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.
```

Do NOT use startswith/substring/prefix matching.

# 6. Independent source identity + module-path verification

The verifier must:

1. read canonical `delivery_src/.../source_manifest.json`;
2. read external `source_manifest.json`;
3. require:
   - schema exact;
   - identity_basis = `GIT_CANONICAL_BLOB_BYTES`;
   - 135 entries;
4. resolve exactly one entry each:

```text
buildreasonseg/runtime/detector.py
20300 bytes
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738

buildreasonseg/runtime/imageio.py
7978 bytes
b6223be7cb2ab0e0ecae1ae350d7c546d3788a02c35439d167684ad9f359c878
```

5. hash actual external files and require exact match;
6. before importing package modules:

```python
sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))
```

7. import only for identity:

```python
import buildreasonseg.runtime.detector as detector_module
import buildreasonseg.runtime.imageio as imageio_module
```

8. require:

```text
Path(detector_module.__file__).resolve() under EXTERNAL
Path(imageio_module.__file__).resolve() under EXTERNAL
```

No runtime class construction.

# 7. Independent historical R6 verification

The verifier must execute:

```text
git -C <REPO> show 12d5fd9a92a5c6bdfbec8e681efb6ea55cf7de2c:evaluation/task8b3_ref01_locked_reference_forensics_r6.json
```

Parse stdout as JSON.

Require:

```text
detector_calls = 4
```

Require `proposal_fields_compared` is exactly the 10-field set:

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

For each of the four candidates require:

```text
ten_field_all_match = true
within_tolerance = true
```

Do NOT infer R6 PASS only from canonical evidence booleans.

# 8. Canonical evidence read-only verification

Read:

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

Require exact English disclosure per §5.

Require external identity entries match §6.

Require exactly four candidates:

```text
right / left / above / below
```

# 9. Live P1D12 metadata verification

Read these read-only files:

```text
external\inference\output\diagnostics\1010\proposals.json
external\inference\output\diagnostics\1003\proposals.json
external\inference\output\diagnostics\1008\proposals.json
external\inference\output\diagnostics\1009\proposals.json
```

Require expected counts:

```text
1010 raw=6  merged=6  eligible=4
1003 raw=66 merged=53 eligible=42
1008 raw=9  merged=9  eligible=4
1009 raw=7  merged=6  eligible=3
```

Require exact proposal ID set lengths = merged count.

Every proposal must contain:

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

# 10. Independent production selected replay

Eligibility:

```text
mask_area > 0
AND touches_image_border == false
AND bbox_extent_ratio <= 0.20
```

For each candidate:

```python
eligible = ...
selected = min(
    eligible,
    key=lambda p: (-mask_area, -confidence, proposal_id)
)
```

Require selected IDs:

```text
right=1
left=14
above=4
below=1
```

This selected ID must be computed from LIVE P1D12 metadata, not copied from canonical evidence.

# 11. Independent best_eligible / best_any replay

Read canonical historical per-proposal IoU table:

```text
historical_iou_by_proposal[relation]
```

Require exact proposal ID set equality with live P1D12 metadata.

Join by proposal_id.

Rank:

```python
(-iou_to_gt, -confidence, proposal_id)
```

Require:

```text
right bestEligible=1 bestAny=1
left  bestEligible=30 bestAny=30
above bestEligible=4 bestAny=5
below bestEligible=2 bestAny=2
```

# 12. Full 12-role fact comparison

For each of 4 candidates and each role:

```text
selected
best_eligible
best_any
```

Compare canonical evidence against independently replayed/live facts.

Required exact:
- proposal_id
- mask_area
- global_bbox
- eligible

Required tolerance <= 1e-12:
- confidence
- iou

Total comparisons:

```text
4 candidates × 3 roles = 12 role records
```

Require:

```text
12/12 PASS
```

# 13. Classification replay

Threshold = 0.50.

Require exact:

```text
right = REFERENCE_SELECTED_CORRECT
left  = REFERENCE_SELECTION_WRONG_COVERED
above = REFERENCE_ELIGIBILITY_BLOCKED
below = REFERENCE_SELECTION_WRONG_COVERED
```

Require canonical evidence class values match.

Require class counts exactly:

```text
REFERENCE_SELECTED_CORRECT = 1
REFERENCE_SELECTION_WRONG_COVERED = 2
REFERENCE_ELIGIBILITY_BLOCKED = 1
REFERENCE_COVERAGE_MISSING = 0
```

If canonical evidence omits explicit zero-count coverage key, treat missing as zero only for comparison; do not modify evidence.

# 14. Aggregate replay

Mechanically derive priority:

```text
coverage missing > eligibility blocked > selection wrong > all correct
```

Require:

```text
overall_outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
dominant_next_blocker = ELIGIBILITY
next_gate = REF01_ELIGIBILITY_FORENSICS
```

# 15. Static zero-inference gate

Run:

```text
<REQUIRED_PYTHON> -m py_compile scripts/task8b3_ref01_locked_reference_forensics.py
```

Require PASS.

Source text must NOT contain executable inference use:

```text
DetectorRuntime(
detect_global(
model.predict(
YOLO(
```

Source MUST contain:

```text
buildreasonseg.runtime.detector
buildreasonseg.runtime.imageio
git
show
proposal_fields_compared
proposals.json
mask_area
confidence
border_clearance
overall_outcome
dominant_next_blocker
next_gate
```

# 16. Execute verifier exactly once

Run:

```text
<REQUIRED_PYTHON> scripts/task8b3_ref01_locked_reference_forensics.py
```

Require exit 0.

Required stdout terminal lines:

```text
SOURCE_IDENTITY: PASS
MODULE_PATH_IDENTITY: PASS
R6_HISTORY: PASS
P1D12_COUNTS: PASS
SELECTED_REPLAY: PASS
BEST_ROLE_REPLAY: PASS
ROLE_FACTS_12_OF_12: PASS
CLASSIFICATION_REPLAY: PASS
AGGREGATE_REPLAY: PASS
FINAL_VERIFIER: PASS
detector_or_model_calls = 0
canonical_evidence_modified = false
```

No retry after a logic/data failure.

# 17. Report update

Update only:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Append R11 section stating:
- R10 was safe but verifier-incomplete;
- R11 is a zero-call independent verifier;
- exact module paths verified;
- historical R6 loaded from Git history;
- selected IDs replayed from production ordering;
- 12/12 role facts independently matched;
- classification and aggregate replay PASS;
- canonical evidence unchanged.

# 18. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R11
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: 7d9492fa7628c4b4cc379a3c4ba976c3bd1fe228
Detector/model calls this task: 0
Canonical evidence modified: NO / YES
PY_COMPILE_GATE: PASS / FAIL
NO_INFERENCE_STATIC_GATE: PASS / FAIL
Source manifest identity: PASS / FAIL
External detector module __file__: PASS / FAIL
External imageio module __file__: PASS / FAIL
Historical R6 git-show verification: 4/4 PASS / other
P1D12 counts: 4/4 PASS / other
Selected replay right/left/above/below: 1/14/4/1 / other
Best eligible replay right/left/above/below: 1/30/4/2 / other
Best any replay right/left/above/below: 1/30/5/2 / other
Role fact comparisons: 12/12 PASS / other
Classification replay: 4/4 PASS / other
Aggregate replay: PASS / FAIL
Exact disclosure equality: PASS / FAIL
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

# 19. Commit / push

If COMPLETE:

```text
docs(rc1): independently verify reference closure
```

If STOP/FAILED:

```text
docs(rc1): record strict reference verifier stop
```

Push only current branch.
No force push.
Do not update main.

Then STOP.

# 20. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only four allowed tracked files changed;
- detector/model calls = 0;
- canonical evidence unchanged;
- source manifest identities independently verified;
- module __file__ identities independently verified;
- R6 JSON independently loaded via git show and verified;
- P1D12 counts independently verified;
- selected IDs independently replayed from production ordering;
- best eligible / best any independently replayed with full frozen tie-break;
- all 12 role records matched field-by-field;
- exact disclosure equality PASS;
- classification replay PASS;
- aggregate outcome replay PASS;
- temporary evidence files absent;
- report/handoff committed and pushed;
- NEXT not executed;
- STOP.
