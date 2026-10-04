# TO_DSH — Task 8B.3-REF01-F1-R3: Repair Forensic Harness and Complete Reference Classification

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-reference-forensics`
> Required starting HEAD: `ac1e4a9f8f452581c509ab0d73024c44c5de1be0`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> RC1 Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. ChatGPT audit decision

Task 8B.3-REF01-F1-R2 STOP is accepted.

No forensic classification exists yet.

The previous forensic script is NOT approved for rerun as-is.

ChatGPT independently verified five concrete defects in the committed script:

```text
D1 — source raster root typo
current:
C:\D\resources\Satellite dataset \Ⅱ (East Asia)\...

correct:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\...

D2 — wrong runtime source
current script inserts CANON into sys.path, so buildreasonseg.runtime is imported from:
workspace\project\BuildReasonSeg\delivery_src\...

required source:
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\...

D3 — non-frozen execution override
current script creates:
DetectorRuntime(..., device="cpu")

required:
default external RC1 DetectorRuntime()
(no device/imgsz/conf/max_det override)

D4 — missing reproduction contract
current script does not verify every rerun merged proposal against the frozen P1D12 proposals.json metadata.

D5 — missing required pre-detector gates/evidence
current script does not fully enforce:
raster SHA identity,
external module __file__ identity,
all frozen detector constants,
GT/raster identity before detector calls,
best proposal IDs/tie-break metadata,
exact scientific disclosure.
```

# 1. Detector-pass accounting correction

The R2 report said detector calls were "0–4 possible".

ChatGPT rejects that uncertainty.

The recorded traceback is:

```text
Image.open(ROOT / "1010.tif")
→ FileNotFoundError
```

and the committed script calls:

```python
raster = ...
detection = runtime.detect_global(raster)
```

in that order.

Therefore both attempted script executions failed before the first `detect_global()` call.

Frozen fact:

```text
REF01 detector passes consumed before R3 = 0
remaining authorised detector passes:
1010 = 1
1003 = 1
1008 = 1
1009 = 1
```

Do NOT treat the allowance as exhausted.

# 2. Scientific-use disclosure

The final report MUST include exactly:

> The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.

Also include a faithful Chinese disclosure.

# 3. Git gate

Require exactly:

```text
branch = fix/task8b3-ref01-reference-forensics
HEAD = ac1e4a9f8f452581c509ab0d73024c44c5de1be0
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No new branch.
No merge/rebase/reset/stash/clean/cherry-pick.

# 4. Strict prohibitions

Do NOT:
- run the current broken forensic script before repairing it;
- run `predict.py`;
- run Qwen / ProgramHead;
- run SAM2;
- run relation fields;
- run D-B1;
- run target segmentation;
- modify canonical RC1;
- modify external RC1;
- run sync in write mode;
- change detector weights/config/thresholds;
- change eligibility/selector;
- change IoU threshold 0.50;
- use CPU/device override;
- inspect source/proposal images visually;
- use subjective judgement;
- replace candidates;
- regenerate dataset records/native-vector caches;
- compute Task 7J metrics;
- implement a repair to the product;
- update main;
- force push.

# 5. Allowed tracked changes

Only:

```text
scripts/task8b3_ref01_locked_reference_forensics.py
evaluation/task8b3_ref01_locked_reference_forensics.json
docs/task8b3_ref01_locked_reference_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 6. Repair the source raster paths

Use exact paths:

```text
right:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1010.tif
SHA256 =
1688306c5edbffe4944809bd5a4db5e880d0e0fdfbec1f264eb691d24d395be2

left:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1003.tif
SHA256 =
eea4edd0db9e079e20b6cd3cc9a20bde6312c4e24049ab6e8c64273259b50c38

above:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1008.tif
SHA256 =
0efe8bc2e1d1f3f575ee7aa0670f4bf7e4a3d53350e923455dfcf5a2735095dd

below:
C:\D\resources\Satellite dataset Ⅱ (East Asia)\1. The cropped image data and raster labels\test\image\1009.tif
SHA256 =
c22134e671f2d0b70b9231c8e1fea1664b7e89f57b5e26967e1828b3f8e323d7
```

No extra slash/backslash before `Ⅱ`.

Before importing/constructing the detector, the script must verify all four:
- exist;
- SHA256 exact;
- RC1 `load_image()` reads width=512, height=512, channels=3.

If any fail:
- exit non-zero;
- detector call count remains 0.

# 7. Import product runtime from EXTERNAL RC1

The script must make external RC1 the first package root:

```python
EXTERNAL = Path(r"C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1")
REPO = Path(r"C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg")

sys.path.insert(0, str(EXTERNAL))
sys.path.insert(1, str(REPO))
```

Then import:

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

Native-vector research adapter may come from the repo:

```python
from buildreasonseg_mvp.native_vector_adapter import (
    NativeVectorDataset,
    validate_reasoning_record,
)
```

Before any detector call require:

```text
Path(detector_module.__file__).resolve()
is under
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1

Path(imageio_module.__file__).resolve()
is under
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

If not:
- exit non-zero;
- detector calls = 0.

The evidence JSON must store the actual resolved module paths.

# 8. Frozen detector constants

Before detector construction require exact:

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

No override.

Construct exactly:

```python
detector = DetectorRuntime()
```

Do NOT pass:
- checkpoint;
- device;
- imgsz;
- conf;
- max_det.

The external package path resolver will load the external RC1 detector checkpoint.

# 9. Pre-detector dataset / GT gate

Use frozen `BuildSpatialReason v0.2` TEST:

```text
test.jsonl bytes =
14415287

test.jsonl sha256 =
72525bff76ef5bdc34dedcb257a5f9c72c420fce0d78dbaa34b02d5f230301dc

records =
6219
```

Locked records exactly:

```text
right:
buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91
GT reference id = 4
GT area = 2478

left:
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3
GT reference id = 26
GT area = 3512

above:
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314
GT reference id = 4
GT area = 5013

below:
buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
GT reference id = 3
GT area = 2606
```

For each record require:
- unique match;
- `validate_reasoning_record(...) == []`;
- first reasoning operation = `argmax_area`;
- reference provenance exact;
- `label_map.shape == (512,512)`;
- GT mask non-empty;
- GT mask area exact.

If any fail:
- detector calls = 0;
- exit non-zero.

Do not regenerate caches.

# 10. Frozen P1D12 metadata gate

Load existing external `proposals.json` for:

```text
1010
1003
1008
1009
```

Require recorded counts:

```text
1010 raw=6  merged=6  eligible=4
1003 raw=66 merged=53 eligible=42
1008 raw=9  merged=9  eligible=4
1009 raw=7  merged=6  eligible=3
```

This gate is pre-detector and metadata-only.

If any disagree:
- detector calls = 0;
- exit non-zero.

# 11. Mandatory STATIC HARNESS GATE before script execution

After editing the script but BEFORE executing it, run a read-only static check.

Equivalent assertions:

```python
from pathlib import Path

p = Path("scripts/task8b3_ref01_locked_reference_forensics.py")
text = p.read_text(encoding="utf-8")

assert r"Satellite dataset Ⅱ (East Asia)" in text
assert r"Satellite dataset \Ⅱ (East Asia)" not in text

assert 'sys.path.insert(0, str(EXTERNAL))' in text
assert 'sys.path.insert(0, str(CANON))' not in text

assert "DetectorRuntime()" in text
assert 'device="cpu"' not in text
assert "device='cpu'" not in text

assert "load_image" in text
assert "__file__" in text
assert "sha256" in text.lower()
assert "proposals.json" in text
assert "RERUN_GLOBALPROPOSAL_MASK_CROP" in text
```

Record:

```text
STATIC_HARNESS_GATE = PASS
```

If fail:
- do NOT execute the script;
- STOP.

# 12. Execute forensic script exactly ONCE

Only after all pre-detector/static gates PASS, execute exactly once with the RC1 proposal environment:

```text
<RC1_PYTHON> scripts/task8b3_ref01_locked_reference_forensics.py
```

No wrapper retry.
No second run for encoding.
Set UTF-8 environment before the single invocation if needed.

Expected detector call count:

```text
4
```

Exactly one `detect_global()` per locked raster.

If the script exits non-zero:
- do not rerun;
- commit STOP evidence;
- STOP.

# 13. Mandatory detector reproduction gate

The script must compare rerun merged proposals against frozen P1D12 `proposals.json`.

Counts exact:

```text
1010 = 6 / 6 / 4
1003 = 66 / 53 / 42
1008 = 9 / 9 / 4
1009 = 7 / 6 / 3
```

Every merged proposal by `proposal_id`.

Exact fields:
- proposal_id
- source_tile_id
- mask_area
- global_bbox
- touches_image_border
- raw_index

Tolerance ≤ 1e-6:
- confidence
- centroid[0]
- centroid[1]
- border_clearance
- bbox_extent_ratio

If any mismatch:
- no classification;
- script exits non-zero;
- no retry.

Evidence JSON must store a per-candidate:

```text
reproduction_match = true/false
```

and mismatch detail when false.

# 14. Proposal mask source

Use ONLY the rerun in-memory:

```text
GlobalProposal.mask_crop
```

For every proposal:

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

`proposals.json` supplies NO mask pixels.

No transformation is permitted.

# 15. Frozen GT IoU and selection

Coverage threshold:

```text
0.50
```

Use production selector exactly:

```python
selected = select_reference(merged, family="largest")
```

Require selected not None.

Use production eligibility exactly:

```python
eligible = eligible_proposals(merged, family="largest")
```

For every merged proposal compute IoU to the canonical GT reference mask.

Rank `best_eligible` and `best_any` by:

```python
(
    -iou_to_gt,
    -confidence,
    proposal_id,
)
```

Record at minimum:

```text
selected_id
selected_iou
selected_mask_area
selected_confidence
selected_bbox

best_eligible_id
best_eligible_iou
best_eligible_confidence
best_eligible_area

best_any_id
best_any_iou
best_any_is_eligible

selected_to_best_eligible_iou_gap
```

# 16. Exclusive class

Exactly one:

```text
REFERENCE_SELECTED_CORRECT
selected_iou >= 0.50
```

```text
REFERENCE_SELECTION_WRONG_COVERED
selected_iou < 0.50
AND best_eligible_iou >= 0.50
```

```text
REFERENCE_ELIGIBILITY_BLOCKED
selected_iou < 0.50
AND best_eligible_iou < 0.50
AND best_any_iou >= 0.50
```

```text
REFERENCE_COVERAGE_MISSING
best_any_iou < 0.50
```

No fifth class.
No visual override.

# 17. Overall outcome priority

If any candidate is `REFERENCE_COVERAGE_MISSING`:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = COVERAGE
NEXT = REF01_COVERAGE_FRAGMENTATION_FORENSICS
```

Else if any is `REFERENCE_ELIGIBILITY_BLOCKED`:

```text
Outcome = REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE
Dominant next blocker = ELIGIBILITY
NEXT = REF01_ELIGIBILITY_FORENSICS
```

Else if any is `REFERENCE_SELECTION_WRONG_COVERED`:

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

# 18. Evidence JSON

Write exactly:

```text
evaluation/task8b3_ref01_locked_reference_forensics.json
```

Required top-level fields:

```text
task = 8B.3-REF01-F1-R3
starting_head
scientific_reuse_disclosure
proposal_mask_source = RERUN_GLOBALPROPOSAL_MASK_CROP
coverage_threshold = 0.50
external_detector_module_path
external_imageio_module_path
detector_config
detector_call_count = 4
test_jsonl_identity
candidates
class_counts
overall_outcome
dominant_next_blocker
next_gate
```

Each candidate must contain:
- immutable sample_id/tile/query;
- raster path/SHA;
- GT reference ID/source-feature/area;
- reproduction counts and match;
- selected proposal facts;
- best eligible facts;
- best any facts;
- exclusive class.

# 19. Report

Update existing:

```text
docs/task8b3_ref01_locked_reference_forensics.md
```

Preserve F1, R1 and R2 STOP history.

Append R3 final section with:
1. ChatGPT R2 audit findings;
2. corrected path;
3. proof prior detector passes = 0;
4. external module identity;
5. default DetectorRuntime declaration;
6. static harness gate;
7. detector calls = 4;
8. reproduction gate;
9. GT IoU table;
10. class counts;
11. Outcome/NEXT;
12. nonclaims.

Required table:

```text
relation | GT_ref | selected_id | selected_IoU | best_eligible_id | best_eligible_IoU | best_any_id | best_any_IoU | class
```

# 20. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-REF01-F1-R3
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-reference-forensics
Starting HEAD: ac1e4a9f8f452581c509ab0d73024c44c5de1be0
Prior R2 actual detector passes: 0
STATIC_HARNESS_GATE: PASS / FAIL
Source raster path typo: FIXED / NOT FIXED
Locked raster SHA identities: 4/4 PASS / other
External detector module identity: PASS / FAIL
External imageio module identity: PASS / FAIL
Detector runtime construction: DEFAULT / other
Proposal mask source: RERUN_GLOBALPROPOSAL_MASK_CROP
proposals.json role: METADATA_REPRODUCTION_ONLY
Coverage threshold: 0.50_TASK7F_FROZEN
GT reference-mask integrity: 4/4 PASS / other
Detector calls this task: 4 / other
Detector reproduction right: 6/6 eligible4 MATCH / other
Detector reproduction left: 66/53 eligible42 MATCH / other
Detector reproduction above: 9/9 eligible4 MATCH / other
Detector reproduction below: 7/6 eligible3 MATCH / other
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
right class: <exact class>
left class: <exact class>
above class: <exact class>
below class: <exact class>
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
REF-01 status: FORENSICS_COMPLETE / GATE_PASS / other
Outcome: <exact enum>
Dominant next blocker: <exact value>
Next gate: <exact enum>
Evidence: evaluation/task8b3_ref01_locked_reference_forensics.json
Report: docs/task8b3_ref01_locked_reference_forensics.md
External/canonical product files modified: NO / NO
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 21. Commit / push

If COMPLETE:

```text
docs(rc1): complete locked reference forensics
```

If STOP/FAILED:

```text
docs(rc1): record reference forensics harness stop
```

Push only:

```text
fix/task8b3-ref01-reference-forensics
```

No force push.
Do not update main.

After push:
- STOP;
- wait for ChatGPT.

# 22. COMPLETE definition

COMPLETE only if:
- exact branch/head;
- only five allowed tracked paths changed;
- corrected exact dataset path;
- prior R2 detector calls correctly recorded as 0;
- all four raster SHA checks PASS before inference;
- buildreasonseg runtime imports from external RC1;
- all frozen detector constants exact;
- `DetectorRuntime()` constructed with no overrides;
- STATIC_HARNESS_GATE PASS;
- script executed exactly once;
- exactly four detector calls;
- P1D12 proposal metadata reproduced exactly;
- IoU uses only in-memory rerun `mask_crop`;
- threshold remains exactly 0.50;
- one exclusive class per candidate;
- no Qwen/SAM2/relation/D-B1/target segmentation;
- no visual judgement/candidate replacement;
- evidence/report/handoff committed and pushed;
- NEXT not executed;
- STOP.
