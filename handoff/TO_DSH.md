请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3A-R5E — Final Artifact Normalization**

# TO_DSH — Task 8B.3-REF01-E3A-R5E: Final Artifact Normalization

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-impl`
> Required starting HEAD: `d105060732ff4092b7c2af6df06bfb17555cb55f`

# 0. CHATGPT AUDIT DISPOSITION

R5D technical/artifact facts are accepted, but formal E3A closure is NOT yet signed because R5D violated two exact artifact-contract items:

```text
Required evidence path:
evaluation/task8b3_ref01_eligibility_repair_impl.json

Actual R5D path:
evaluation/task8b3_ref01_e3a_canonical_implementation_closure.json
```

and:

```text
Required R5D commit message:
docs(rc1): close canonical extent dominance implementation

Actual R5D commit message:
docs(rc1): close canonical implementation artifacts
```

The R5D commit-message mismatch is a historical process defect only.
DO NOT rewrite history.

R5E is a pure normalization task.

# 1. FROZEN FACTS

The following are frozen and MUST NOT be re-derived, rerun, or changed:

```text
Design:
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

Canonical implementation:
PRESENT

eligible() base semantics:
PRESERVED

eligible_proposals() base semantics:
PRESERVED

Largest-only exception:
PRESENT

Strict area/confidence:
> / >

Border bypass:
NO

Smallest family changed:
NO

New numeric threshold:
NO

Accepted targeted regression:
40 passed in 0.69s

Canonical full-suite observation:
17 failed, 101 passed, 6 errors in 39.68s

Canonical full-suite disposition:
NOT_APPLICABLE_PRE_SYNC_SOURCE_TREE_MISSING_RUNTIME_ASSETS

R5C obsolete static probe:
INVALID_OBSOLETE_PRE_REPAIR_PROBE

Detector/model inference:
0

Source manifest:
INTENTIONALLY_STALE_PENDING_E3B

External full suite:
REQUIRED AFTER CONTROLLED SYNC

PROP-01:
PROP01_OPEN_ENGINEERING_DEFECT

left/below reference-selection defects:
UNRESOLVED
```

# 2. EXECUTOR CONTRACT

DSH has NO technical discretion.

Allowed operations ONLY:
1. verify exact branch/head;
2. verify product/test/source_manifest Git blobs are unchanged from starting HEAD;
3. delete the incorrectly named R5D evidence file;
4. create the exact required final evidence file at the exact path/schema below;
5. append one normalization section to the existing report;
6. update FROM_DSH;
7. create exactly one R5E commit;
8. push once;
9. STOP.

Forbidden:
- NO pytest;
- NO py_compile;
- NO detector/model inference;
- NO source/test edits;
- NO source_manifest edit;
- NO sync;
- NO external RC1 write;
- NO new files except the exact required evidence file;
- NO rebase/reset/amend;
- NO intermediate commit/push;
- NO force push;
- NO NEXT.

Any uncovered condition => STOP.

# 3. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-impl

git rev-parse HEAD
= d105060732ff4092b7c2af6df06bfb17555cb55f
```

Allowed initial status:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No commit/push until §8.

# 4. PRODUCT IMMUTABILITY GATE

For each path below, compare the blob at starting HEAD to the current working-tree/index state and require NO change:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
```

No edits are allowed.

Require the old user-added patcher path is absent:

```text
handoff/E3A_R5_apply_exact.py
```

# 5. DELETE WRONG R5D EVIDENCE PATH

Delete exactly:

```text
evaluation/task8b3_ref01_e3a_canonical_implementation_closure.json
```

using:

```text
git rm evaluation/task8b3_ref01_e3a_canonical_implementation_closure.json
```

Require success.

Do not delete any other evaluation file.

# 6. CREATE EXACT FINAL E3A EVIDENCE

Create exactly:

```text
evaluation/task8b3_ref01_eligibility_repair_impl.json
```

with EXACTLY this JSON structure and values:

```json
{
  "task": "8B.3-REF01-E3A-R5E",
  "starting_head": "d105060732ff4092b7c2af6df06bfb17555cb55f",
  "branch": "fix/task8b3-ref01-eligibility-repair-impl",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "closure_scope": "CANONICAL_IMPLEMENTATION_FINAL_ARTIFACT_NORMALIZATION_NO_SYNC",
  "detector_model_calls": 0,
  "product_source_changed_in_r5e": false,
  "test_source_changed_in_r5e": false,
  "source_manifest_changed_in_r5e": false,
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
  "r5d_commit_message_defect_recorded": true,
  "source_manifest_status": "INTENTIONALLY_STALE_PENDING_E3B",
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTATION_CLOSED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC"
}
```

Do not add/remove/rename/reorder semantic fields.
Do not create any second evidence file.

# 7. REPORT / FROM_DSH

Append exactly one new section to:

```text
docs/task8b3_ref01_eligibility_repair_impl.md
```

Title:

```text
## E3A-R5E — final artifact normalization
```

Required conclusions:

```text
Task = 8B.3-REF01-E3A-R5E
Status = COMPLETE
Product/test/source_manifest changes in R5E = NONE
Wrong R5D evidence path = REMOVED
Final E3A evidence path = evaluation/task8b3_ref01_eligibility_repair_impl.json
R5D commit-message mismatch = RECORDED, NOT HISTORY-REWRITTEN
Accepted targeted evidence = 40 passed in 0.69s
Canonical full-suite disposition = NOT_APPLICABLE_PRE_SYNC_SOURCE_TREE_MISSING_RUNTIME_ASSETS
R5C static probe disposition = INVALID_OBSOLETE_PRE_REPAIR_PROBE
Canonical implementation = CLOSED
Source manifest status = INTENTIONALLY_STALE_PENDING_E3B
External full suite after controlled sync = REQUIRED
Detector/model inference = NONE
Outcome = REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTATION_CLOSED
NEXT = REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects = UNRESOLVED
```

Preserve ARTIFACT-FACTS exactly in `handoff/FROM_DSH.md`.

Active FROM_DSH must report:

```text
Task: 8B.3-REF01-E3A-R5E
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-impl
Starting HEAD: d105060732ff4092b7c2af6df06bfb17555cb55f
Design selected by: CHATGPT
DSH algorithm choice performed: NO
Detector/model calls: 0
Product source changed in R5E: NO
Test source changed in R5E: NO
Source manifest changed in R5E: NO
Wrong R5D evidence removed: YES / NO
Final evidence path: evaluation/task8b3_ref01_eligibility_repair_impl.json
Accepted targeted evidence: 40 passed in 0.69s
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

# 8. FINAL DIFF GATE

Run:

```text
git diff --name-status d105060732ff4092b7c2af6df06bfb17555cb55f
```

Allowed ONLY:

```text
D  evaluation/task8b3_ref01_e3a_canonical_implementation_closure.json
A  evaluation/task8b3_ref01_eligibility_repair_impl.json
M  docs/task8b3_ref01_eligibility_repair_impl.md
M  handoff/FROM_DSH.md
M  handoff/TO_DSH.md
```

No other path is allowed.

Require:

```text
git rev-list --count d105060732ff4092b7c2af6df06bfb17555cb55f..HEAD
= 0
```

Any mismatch => STOP.

# 9. SINGLE COMMIT / PUSH

If every gate PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): normalize canonical implementation closure
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record canonical normalization stop
```

Commit exactly once.
NO amend.

After commit require:

```text
git rev-list --count d105060732ff4092b7c2af6df06bfb17555cb55f..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute NEXT.

Then STOP.

# 10. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- zero product/test/source_manifest changes;
- wrong R5D evidence deleted;
- exact required final evidence created at exact path;
- report/FROM_DSH normalized;
- exact commit message used;
- exactly one R5E commit;
- no test/compile/inference/sync;
- NEXT not executed;
- STOP.
