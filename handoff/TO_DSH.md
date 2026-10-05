请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3A-R5C — Canonical Implementation Artifact Closure**

# TO_DSH — Task 8B.3-REF01-E3A-R5C: Canonical Implementation Artifact Closure

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-impl`
> Required starting HEAD: `1fb0e6afefbde7e1e4487cd36cfaa6e98532488c`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`

# 0. AUDIT DISPOSITION ENTERING R5C

R5 product implementation is technically present in canonical RC1, but R5 artifact/process closure is NOT approved.

Accepted R5 facts:
- canonical detector contains the approved `LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1`;
- `eligible()` remains unchanged;
- `eligible_proposals()` remains unchanged;
- largest-only exception is implemented inside reference selection;
- eight fixed tests are present;
- source_manifest remained unchanged;
- external RC1 was not synced;
- main remained unchanged;
- R5 starting HEAD has exactly one descendant commit at current HEAD.

Important correction from the user:
- `handoff/E3A_R5_apply_exact.py` was manually placed into the repository by the user.
- This is NOT classified as a DSH execution violation.
- The user has already manually deleted this file from the local working tree.
- R5C must only stage and commit that deletion correctly.

Remaining R5 closure defects:
1. tracked `handoff/E3A_R5_apply_exact.py` still exists in Git history and must be removed by this closure commit;
2. `evaluation/task8b3_ref01_eligibility_repair_impl.json` is missing;
3. R5 report/FROM_DSH do not provide auditable targeted/full pytest PASS results;
4. R5 commit message was malformed:
   `Status: COMPLETE\ncommit:\nfix(rc1): implement largest extent dominance exception`;
5. the malformed R5 message is recorded only; history must NOT be rewritten.

R5C is a ZERO-INFERENCE ARTIFACT CLOSURE task.
It does NOT re-implement the repair.
It does NOT modify detector.py or the test file.

# 1. EXECUTOR CONTRACT

DSH has NO technical discretion.

Allowed operations only:
1. verify exact branch/head;
2. stage the already-local deletion of `handoff/E3A_R5_apply_exact.py` using the exact state-dependent rule in §3;
3. verify the canonical implementation by exact static assertions;
4. py_compile the two product files;
5. run the targeted test file exactly once;
6. run canonical RC1 full tests exactly once;
7. verify external detector identity and source_manifest immutability;
8. create the exact evidence JSON schema below from observed PASS facts;
9. update report and FROM_DSH with the exact required facts;
10. make exactly one R5C commit;
11. push current branch once;
12. STOP.

Forbidden:
- no edit to detector.py;
- no edit to test_task8b_runtime.py;
- no source_manifest edit;
- no sync;
- no external RC1 write;
- no detector/model inference;
- no Qwen/SAM2/D-B1/target inference;
- no new test;
- no threshold/rule change;
- no amend;
- no force push;
- no intermediate commit/push;
- no NEXT.

Any unexpected condition or failed assertion/test => STOP.
Do not self-repair.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-impl

git rev-parse HEAD
= 1fb0e6afefbde7e1e4487cd36cfaa6e98532488c
```

Before any action inspect:

```text
git status --short
git ls-files -- handoff/E3A_R5_apply_exact.py
```

Allowed initial working tree state:
- `handoff/TO_DSH.md` may be modified because the user replaces it with this task book;
- `handoff/E3A_R5_apply_exact.py` may already appear as deleted in the local working tree because the user manually deleted it;
- no other tracked or untracked path is allowed.

Any additional path => STOP.

Do not commit or push until §11.

# 3. STAGE THE USER'S ALREADY-LOCAL PATCHER DELETION

Target tracked path:

```text
handoff/E3A_R5_apply_exact.py
```

Execute exactly one of the following two cases.

## Case A — file is already absent locally and Git reports a deletion

Require:
```text
git ls-files -- handoff/E3A_R5_apply_exact.py
```
returns exactly:
```text
handoff/E3A_R5_apply_exact.py
```

and:
```text
git status --short -- handoff/E3A_R5_apply_exact.py
```
shows a deletion state for that path.

Then run exactly:

```text
git add -u -- handoff/E3A_R5_apply_exact.py
```

Afterward require:

```text
git diff --cached --name-status -- handoff/E3A_R5_apply_exact.py
```

shows exactly a staged deletion of:

```text
handoff/E3A_R5_apply_exact.py
```

## Case B — file still exists locally

If the file exists locally, run exactly:

```text
git rm handoff/E3A_R5_apply_exact.py
```

Then require staged deletion exactly as above.

## Any other state

STOP.

Do NOT recreate the patcher.
Do NOT delete any other path.

# 4. CANONICAL IMPLEMENTATION STATIC ASSERTIONS

Read only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
```

Require detector.py contains exactly one definition each:

```text
def _reference_rank(
def _largest_reference_candidates_with_extent_exception(
```

Require exactly one occurrence each:

```text
proposal.mask_area > baseline.mask_area
proposal.confidence > baseline.confidence
```

Require the exception helper contains:

```text
proposal.bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX
```

Require `select_reference()` contains:

```text
if family == "largest":
    candidates = _largest_reference_candidates_with_extent_exception(proposals)
else:
    candidates = eligible_proposals(proposals, family=family)
```

Require the frozen base eligibility function still contains exactly these conditions, in this order:

```text
mask_area <= 0 OR mask_crop empty -> False
touches_image_border -> False
bbox_extent_ratio > MERGE_BBOX_EXTENT_RATIO_MAX -> False
family == "smallest" AND mask_area < SMALLEST_MIN_AREA_PX -> False
otherwise -> True
```

Require `eligible_proposals()` still delegates only to:

```text
eligible(proposal, family=family)
```

Require exactly eight test definitions whose names start with:

```text
test_largest_extent_
```

Any mismatch => STOP.
No repair.

# 5. PRODUCT PY_COMPILE

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -m py_compile delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

Require exit 0.

Then run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -m py_compile delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py
```

Require exit 0.

No other compile command.

# 6. TARGETED PYTEST

Run exactly once:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -m pytest delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py -q
```

Require exit 0.

Record the exact final pytest summary line verbatim as:

```text
targeted_test_summary
```

If exit != 0 => STOP.
No rerun.
No edits.

# 7. CANONICAL FULL PYTEST

Only after targeted PASS, run exactly once:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -m pytest delivery_src\BuildReasonSeg_Advisor_RC1\tests -q
```

Require exit 0.

Record the exact final pytest summary line verbatim as:

```text
canonical_full_test_summary
```

If exit != 0 => STOP.
No rerun.
No edits.

# 8. EXTERNAL / MANIFEST IMMUTABILITY

Actual external detector:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

must have SHA256 exactly:

```text
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
```

Require:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

has NO diff relative to starting HEAD.

Require:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
```

has NO diff relative to starting HEAD.

No sync helper is allowed.

# 9. EVIDENCE JSON

Only if §§3–8 all PASS, create:

```text
evaluation/task8b3_ref01_eligibility_repair_impl.json
```

Required JSON keys and fixed values:

```json
{
  "task": "8B.3-REF01-E3A-R5C",
  "starting_head": "1fb0e6afefbde7e1e4487cd36cfaa6e98532488c",
  "branch": "fix/task8b3-ref01-eligibility-repair-impl",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "closure_scope": "CANONICAL_IMPLEMENTATION_ARTIFACT_CLOSURE_NO_SYNC",
  "detector_model_calls": 0,
  "eligible_function_changed_in_r5c": false,
  "eligible_proposals_function_changed_in_r5c": false,
  "detector_source_changed_in_r5c": false,
  "test_source_changed_in_r5c": false,
  "largest_only_exception_present": true,
  "strict_area_operator": ">",
  "strict_confidence_operator": ">",
  "border_exception_allowed": false,
  "smallest_family_changed": false,
  "new_numeric_thresholds": [],
  "product_py_compile": "PASS",
  "targeted_test_exit": 0,
  "targeted_test_summary": "<EXACT OBSERVED FINAL SUMMARY LINE>",
  "canonical_full_test_exit": 0,
  "canonical_full_test_summary": "<EXACT OBSERVED FINAL SUMMARY LINE>",
  "external_detector_sha256": "82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738",
  "external_detector_unchanged": true,
  "source_manifest_updated": false,
  "manifest_status": "INTENTIONALLY_STALE_PENDING_E3B",
  "r5_user_added_handoff_patcher_removed": true,
  "r5_commit_message_defect_recorded": true,
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTATION_CLOSED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC"
}
```

Only the two pytest summary strings may vary, and only from observed stdout.

Do not add/remove/rename keys.

# 10. REPORT / FROM_DSH

Update:

```text
docs/task8b3_ref01_eligibility_repair_impl.md
```

Append authoritative section:

```text
## E3A-R5C — canonical implementation artifact closure
```

Required conclusions:

```text
Task = 8B.3-REF01-E3A-R5C
Status = COMPLETE
R5 implementation source = RETAINED
R5 product implementation redesign = NONE
R5 user-added handoff patcher = REMOVED
R5 missing evidence JSON = REPAIRED
R5 malformed commit message = RECORDED, NOT HISTORY-REWRITTEN
Detector/model calls = 0
detector.py changed in R5C = NO
test_task8b_runtime.py changed in R5C = NO
eligible() frozen base semantics = PRESERVED
eligible_proposals() frozen base semantics = PRESERVED
Largest-only extent-dominance exception = PRESENT
Strict area/confidence dominance = > / >
Border bypass = NO
Smallest family change = NO
New numeric threshold = NO
Product py_compile = PASS
Targeted pytest = PASS
Targeted summary = <observed exact summary>
Canonical full pytest = PASS
Canonical full summary = <observed exact summary>
External RC1 modified = NO
Source manifest updated = NO
Manifest status = INTENTIONALLY_STALE_PENDING_E3B
Outcome = REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTATION_CLOSED
NEXT = REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects = UNRESOLVED
final Demo inference = NOT RUN
```

Preserve ARTIFACT-FACTS exactly in:

```text
handoff/FROM_DSH.md
```

FROM_DSH active task must report the same facts.

# 11. FINAL DIFF GATE

Before commit run:

```text
git diff --name-only 1fb0e6afefbde7e1e4487cd36cfaa6e98532488c
```

Allowed ONLY:

```text
handoff/E3A_R5_apply_exact.py
evaluation/task8b3_ref01_eligibility_repair_impl.json
docs/task8b3_ref01_eligibility_repair_impl.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Important:
- `handoff/E3A_R5_apply_exact.py` must appear as DELETED.
- detector.py MUST NOT appear.
- test_task8b_runtime.py MUST NOT appear.
- source_manifest.json MUST NOT appear.
- pipeline.py MUST NOT appear.

Also require:

```text
git rev-list --count 1fb0e6afefbde7e1e4487cd36cfaa6e98532488c..HEAD
= 0
```

Any mismatch => STOP.

# 12. SINGLE COMMIT / PUSH

If every gate PASS:

```text
Status = COMPLETE
Commit message exactly:
docs(rc1): close canonical extent dominance implementation
```

Otherwise:

```text
Status = STOP
Commit message exactly:
docs(rc1): record canonical implementation closure stop
```

Commit exactly once.
NO amend.

After commit require:

```text
git rev-list --count 1fb0e6afefbde7e1e4487cd36cfaa6e98532488c..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute NEXT.

Then STOP.

# 13. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- user-added R5 handoff patcher deletion correctly staged and committed;
- canonical detector/test untouched by R5C;
- frozen implementation static assertions PASS;
- both product py_compile PASS;
- targeted pytest PASS once;
- canonical full pytest PASS once;
- external detector unchanged;
- source_manifest/pipeline unchanged;
- exact evidence JSON created;
- report/FROM_DSH complete;
- exactly one R5C commit;
- no amend/force push/intermediate commit;
- NEXT not executed;
- STOP.
