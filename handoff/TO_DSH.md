请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3A-R5D — Canonical Implementation Closure Reclassification**

# TO_DSH — Task 8B.3-REF01-E3A-R5D: Canonical Implementation Closure Reclassification

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-impl`
> Required starting HEAD: `06a031fdfb375f8ccd3689f23f4e384ebfa964ce`

# 0. CHATGPT AUDIT DISPOSITION

R5D is a PURE ARTIFACT CLOSURE task.

No product code, test code, source_manifest, external RC1, model asset, or runtime output may be changed.

ChatGPT has already decided the following:

## 0.1 R5 implementation status

The canonical implementation is technically present and correct:

```text
Design =
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

eligible() =
PRESERVED

eligible_proposals() =
PRESERVED

largest-only exception =
PRESENT

strict area dominance =
>

strict confidence dominance =
>

border bypass =
NO

smallest-family behavior =
UNCHANGED
```

## 0.2 Valid R5C targeted evidence

R5C produced:

```text
targeted test file =
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py

exit =
0

summary =
40 passed in 0.69s
```

This is accepted as valid canonical implementation regression evidence.

Do NOT rerun it in R5D.

## 0.3 R5C full-suite failure reclassification

R5C also ran a canonical delivery full suite and observed:

```text
17 failed, 101 passed, 6 errors in 39.68s
```

This is NOT classified as evidence that the extent-dominance repair is defective.

Reason:

The Git canonical source tree intentionally does not contain the full large runtime assets required for a READY delivery installation.

In particular, the Git canonical source tree does not carry the complete runtime weights/components that the delivery `test_setup_checker.py` real-project tests expect.

Therefore:

```text
pre-sync canonical full delivery suite =
NOT A VALID RELEASE GATE
```

The complete delivery test suite is deferred to E3B, AFTER:
1. canonical source_manifest is updated;
2. controlled sync to external RC1 is completed;
3. the external RC1 with full assets is checked.

Frozen disposition:

```text
canonical_full_suite_disposition =
NOT_APPLICABLE_PRE_SYNC_SOURCE_TREE_MISSING_RUNTIME_ASSETS
```

Do NOT rerun the canonical full suite in R5D.

## 0.4 R5C static 6/7 STOP reclassification

R5C recorded one static probe failure because a probe searched for a pre-repair ordering string.

That probe is NOT part of the approved post-repair contract.

Frozen disposition:

```text
r5c_static_probe_failure =
INVALID_OBSOLETE_PRE_REPAIR_PROBE
```

Do NOT recreate or rerun that probe.

## 0.5 Historical process defects

Record, but do NOT rewrite history:

```text
R5 malformed commit message =
RECORDED

R5C intermediate commit =
RECORDED

R5C final STOP commit =
RECORDED
```

No rebase/reset/amend/force-push is authorized.

# 1. EXECUTOR CONTRACT

DSH has NO technical discretion.

Allowed operations ONLY:

1. verify branch/head;
2. verify that product/test/source_manifest files have not changed since the R5 implementation commit;
3. verify the user-added patcher is absent;
4. verify Git-canonical runtime asset absence using exact paths in §4;
5. delete the obsolete R5C evidence file;
6. create the exact final E3A evidence file;
7. append the exact R5D closure section to the report;
8. update FROM_DSH;
9. create exactly one R5D commit;
10. push current branch once;
11. STOP.

Forbidden:
- NO pytest;
- NO py_compile;
- NO detector/model inference;
- NO edit to detector.py;
- NO edit to test_task8b_runtime.py;
- NO edit to any other test;
- NO edit to source_manifest;
- NO sync;
- NO external RC1 write;
- NO model asset copy;
- NO new threshold/rule;
- NO rebase/reset/amend/stash/clean;
- NO force push;
- NO intermediate commit/push;
- NO NEXT.

Any uncovered condition => STOP.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-impl

git rev-parse HEAD
= 06a031fdfb375f8ccd3689f23f4e384ebfa964ce
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No commit/push until §9.

# 3. PRODUCT IMMUTABILITY GATE

The R5 implementation commit is:

```text
1fb0e6afefbde7e1e4487cd36cfaa6e98532488c
```

Require NO diff between that commit and current HEAD for each exact path:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
```

Mechanically verify with Git.

If any of those four paths differs => STOP.

Require:

```text
handoff/E3A_R5_apply_exact.py
```

does NOT exist in current HEAD.

If it exists => STOP.

# 4. CANONICAL SOURCE-TREE ASSET GATE

Use `git ls-files` only.

Require these exact runtime asset paths are NOT tracked under canonical `delivery_src`:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/model/buildreasonseg_advisor/decoder.pt
delivery_src/BuildReasonSeg_Advisor_RC1/model/buildreasonseg_advisor/detector.pt
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/sam2/sam2.1_hiera_base_plus.pt
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/program_parser_l3_rehearsal_v1.pt
```

For each path:

```text
git ls-files -- <path>
```

must return empty.

Also inspect:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/model/buildreasonseg_advisor/
```

and require the tracked files in that directory are metadata/config artifacts, not the two runtime `.pt` weights above.

This gate establishes only:

```text
Git canonical source tree != complete external runtime delivery tree
```

Do NOT infer anything else.

# 5. REMOVE OBSOLETE R5C EVIDENCE

Delete exactly:

```text
evaluation/task8b3_ref01_e3a_r5c_canonical_closure.json
```

Use:

```text
git rm evaluation/task8b3_ref01_e3a_r5c_canonical_closure.json
```

Require success.

Do not delete any other evaluation file.

# 6. CREATE FINAL E3A EVIDENCE

Create exactly:

```text
evaluation/task8b3_ref01_eligibility_repair_impl.json
```

with exactly this JSON structure and values:

```json
{
  "task": "8B.3-REF01-E3A-R5D",
  "starting_head": "06a031fdfb375f8ccd3689f23f4e384ebfa964ce",
  "branch": "fix/task8b3-ref01-eligibility-repair-impl",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "closure_scope": "CANONICAL_IMPLEMENTATION_FINAL_CLOSURE_NO_SYNC",
  "detector_model_calls": 0,
  "product_source_changed_in_r5d": false,
  "test_source_changed_in_r5d": false,
  "source_manifest_changed_in_r5d": false,
  "largest_only_exception_present": true,
  "eligible_base_semantics_preserved": true,
  "eligible_proposals_base_semantics_preserved": true,
  "strict_area_operator": ">",
  "strict_confidence_operator": ">",
  "border_exception_allowed": false,
  "smallest_family_changed": false,
  "new_numeric_thresholds": [],
  "accepted_targeted_test": {
    "source_task": "8B.3-REF01-E3A-R5C",
    "exit": 0,
    "summary": "40 passed in 0.69s"
  },
  "canonical_full_suite_observation": {
    "source_task": "8B.3-REF01-E3A-R5C",
    "exit": 1,
    "summary": "17 failed, 101 passed, 6 errors in 39.68s",
    "disposition": "NOT_APPLICABLE_PRE_SYNC_SOURCE_TREE_MISSING_RUNTIME_ASSETS"
  },
  "r5c_static_probe_failure": {
    "disposition": "INVALID_OBSOLETE_PRE_REPAIR_PROBE"
  },
  "canonical_runtime_assets_tracked": false,
  "external_full_suite_required_after_sync": true,
  "r5_commit_message_defect_recorded": true,
  "r5c_intermediate_commit_defect_recorded": true,
  "source_manifest_status": "INTENTIONALLY_STALE_PENDING_E3B",
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTATION_CLOSED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC"
}
```

Do not add/remove/rename keys.
Do not change any fixed string.

# 7. REPORT

Update:

```text
docs/task8b3_ref01_eligibility_repair_impl.md
```

Append exactly one new authoritative section titled:

```text
## E3A-R5D — final canonical implementation closure
```

Required conclusions:

```text
Task = 8B.3-REF01-E3A-R5D
Status = COMPLETE

Canonical implementation =
CLOSED

Design =
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

Product source changed in R5D =
NO

Test source changed in R5D =
NO

eligible() base semantics =
PRESERVED

eligible_proposals() base semantics =
PRESERVED

Largest-only exception =
PRESENT

Strict area / confidence operators =
> / >

Border bypass =
NO

Smallest family changed =
NO

New numeric threshold =
NO

Accepted targeted evidence =
40 passed in 0.69s

Canonical full-suite observation =
17 failed, 101 passed, 6 errors in 39.68s

Canonical full-suite disposition =
NOT_APPLICABLE_PRE_SYNC_SOURCE_TREE_MISSING_RUNTIME_ASSETS

Reason =
Git canonical delivery_src is a source/metadata tree and does not track the complete runtime model assets required by real-delivery setup-checker tests.

R5C obsolete static probe =
INVALID_OBSOLETE_PRE_REPAIR_PROBE

Full delivery suite =
DEFERRED TO E3B EXTERNAL RC1 AFTER CONTROLLED SYNC

Source manifest status =
INTENTIONALLY_STALE_PENDING_E3B

External RC1 modified in R5D =
NO

Detector/model inference =
NONE

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

left/below reference-selection defects =
UNRESOLVED

Outcome =
REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTATION_CLOSED

NEXT =
REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
```

Also explicitly record:

```text
R5 malformed commit message and R5C intermediate-commit history are retained as historical process defects; no history rewrite was performed.
```

# 8. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Active FROM_DSH fields must include:

```text
Task: 8B.3-REF01-E3A-R5D
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-impl
Starting HEAD: 06a031fdfb375f8ccd3689f23f4e384ebfa964ce
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Product source changed in R5D: NO
Test source changed in R5D: NO
Source manifest changed in R5D: NO
User-added R5 patcher present in current HEAD: NO
Canonical runtime weights tracked in delivery_src: NO
Accepted targeted evidence: 40 passed in 0.69s
Canonical full-suite observation: 17 failed, 101 passed, 6 errors in 39.68s
Canonical full-suite disposition: NOT_APPLICABLE_PRE_SYNC_SOURCE_TREE_MISSING_RUNTIME_ASSETS
R5C static probe disposition: INVALID_OBSOLETE_PRE_REPAIR_PROBE
Canonical implementation status: CLOSED
Source manifest status: INTENTIONALLY_STALE_PENDING_E3B
External full suite required after sync: YES
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Outcome: REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTATION_CLOSED
Next gate: REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
Next action: Awaiting ChatGPT audit; do not execute NEXT.
```

# 9. FINAL DIFF GATE

Before commit:

```text
git diff --name-only 06a031fdfb375f8ccd3689f23f4e384ebfa964ce
```

Allowed ONLY:

```text
evaluation/task8b3_ref01_e3a_r5c_canonical_closure.json
evaluation/task8b3_ref01_eligibility_repair_impl.json
docs/task8b3_ref01_eligibility_repair_impl.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Require:
- old R5C evidence path appears as DELETED;
- new final evidence path appears as ADDED;
- detector.py does NOT appear;
- test_task8b_runtime.py does NOT appear;
- source_manifest.json does NOT appear;
- pipeline.py does NOT appear.

Any mismatch => STOP.

Also require:

```text
git rev-list --count 06a031fdfb375f8ccd3689f23f4e384ebfa964ce..HEAD
= 0
```

# 10. SINGLE COMMIT / PUSH

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
docs(rc1): record canonical implementation final closure stop
```

Commit exactly once.
NO amend.
NO force push.

After commit require:

```text
git rev-list --count 06a031fdfb375f8ccd3689f23f4e384ebfa964ce..HEAD
= 1
```

Push current branch exactly once.

Do not update main.
Do not execute NEXT.

Then STOP.

# 11. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- no product/test/manifest modifications;
- user-added patcher absent;
- canonical runtime large assets confirmed untracked using the exact paths;
- obsolete R5C evidence removed;
- final E3A evidence created exactly;
- report/FROM_DSH updated;
- exactly one R5D commit;
- no pytest/py_compile/inference/sync;
- NEXT not executed;
- STOP.
