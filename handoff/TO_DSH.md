请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3B2-R1 — External Sync Validation Artifact Closure**

# TO_DSH — Task 8B.3-REF01-E3B2-R1: External Sync Validation Artifact Closure

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required starting HEAD: `5b8edd742981f1128c851c262ec576665fa0aa1f`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. CHATGPT AUDIT DISPOSITION

E3B2 technical execution is ACCEPTED.

Frozen accepted facts from E3B2:

```text
Design:
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

Starting HEAD of technical E3B2:
300cdad5629b945ffe10480351081f9c8befe7f4

Sync policy:
135 entries
runtime weights / runs / logs / inference outputs excluded = YES

External pre-sync:
exit = 1
match = 133
missing = 0
mismatch = 2

Pre-sync mismatch paths EXACTLY:
buildreasonseg/runtime/detector.py
tests/test_task8b_runtime.py

Controlled external sync:
runs = 1
exit = 0
copied = 135
verified = 135
failures = 0

External post-sync:
exit = 0
match = 135
missing = 0
mismatch = 0

External source_manifest:
copied from canonical Git object = YES
recorded sha256 =
5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c

External setup checker:
runs = 1
exit = 0
final status = BuildReasonSeg environment: READY

External targeted regression:
runs = 1
exit = 0
summary = 40 passed in 1.13s

External full delivery suite:
runs = 1
exit = 0
summary = 124 passed in 112.81s (0:01:52)

Canonical product / manifest / sync helper changed:
NO

Dependency installation:
NONE

Model inference:
NONE

PROP-01:
PROP01_OPEN_ENGINEERING_DEFECT

left/below reference-selection defects:
UNRESOLVED
```

E3B2 is NOT formally closed only because the artifact paths/schema were not the ones required by the task book.

Observed artifact-contract defects:

```text
Required evidence:
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json

Actual evidence:
evaluation/task8b3_ref01_e3b2_external_sync_validation.json
```

and:

```text
Required dedicated report:
docs/task8b3_ref01_e3b2_external_sync_full_suite.md

Actual:
MISSING

E3B2 results were instead appended historically to:
docs/task8b3_ref01_eligibility_repair_impl.md
```

R1 is artifact closure only.

# 1. EXECUTOR CONTRACT

DSH has ZERO technical discretion.

Allowed operations ONLY:
1. verify exact branch/head;
2. verify protected repo files have no working-tree/index changes;
3. run the ONE read-only source-manifest identity command in §4;
4. delete the wrongly named E3B2 evidence;
5. create the exact authoritative evidence at the exact required path;
6. create the exact dedicated report at the exact required path;
7. append one authoritative closure section to the historical implementation report;
8. replace the active FROM_DSH handoff with the exact fields below;
9. make exactly one R1 commit;
10. push once;
11. STOP.

Forbidden:
- NO sync helper execution;
- NO external pre/post `--check`;
- NO external sync/write;
- NO source_manifest copy/write;
- NO check_setup.py;
- NO pytest;
- NO py_compile;
- NO detector/model inference;
- NO canonical product/test/manifest/helper edit;
- NO dependency/environment change;
- NO rebase/reset/amend/stash/clean;
- NO force push;
- NO intermediate commit/push;
- NO NEXT/E3C.

The ONE read-only identity command in §4 is the only external-file read command newly authorized for R1.

Any uncovered condition => STOP.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-sync

git rev-parse HEAD
= 5b8edd742981f1128c851c262ec576665fa0aa1f
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No commit/push until §11.

# 3. PROTECTED REPO IMMUTABILITY GATE

Require NO working-tree/index changes for:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
```

Require the wrongly named E3B2 evidence EXISTS:

```text
evaluation/task8b3_ref01_e3b2_external_sync_validation.json
```

Require the correct final E3B2 evidence does NOT yet exist:

```text
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
```

Require the dedicated report does NOT yet exist:

```text
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
```

Any unexpected state => STOP.

# 4. ONE READ-ONLY EXTERNAL SOURCE-MANIFEST IDENTITY CHECK

Run exactly ONCE:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import hashlib,pathlib,subprocess; spec='HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json'; b=subprocess.run(['git','show',spec],check=True,stdout=subprocess.PIPE).stdout; e=pathlib.Path(r'C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json').read_bytes(); assert e==b; s=hashlib.sha256(b).hexdigest(); assert s=='5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c'; print('EXTERNAL_SOURCE_MANIFEST_IDENTITY: PASS'); print('SOURCE_MANIFEST_BYTES=',len(b)); print('SOURCE_MANIFEST_SHA256=',s)"
```

Require exit 0.

Require stdout:

```text
EXTERNAL_SOURCE_MANIFEST_IDENTITY: PASS
SOURCE_MANIFEST_SHA256= 5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c
```

Record the observed integer from:

```text
SOURCE_MANIFEST_BYTES=
```

This command is READ-ONLY.
Do NOT rerun it.
Do NOT write external RC1.

If identity fails => STOP.

# 5. DELETE WRONG E3B2 EVIDENCE

Delete exactly:

```text
evaluation/task8b3_ref01_e3b2_external_sync_validation.json
```

using:

```text
git rm evaluation/task8b3_ref01_e3b2_external_sync_validation.json
```

Require success.

Do not delete any other evaluation file.

# 6. CREATE AUTHORITATIVE E3B2 EVIDENCE

Create exactly:

```text
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
```

with EXACTLY this schema and fixed values:

```json
{
  "task": "8B.3-REF01-E3B2-R1",
  "source_task": "8B.3-REF01-E3B2",
  "starting_head": "5b8edd742981f1128c851c262ec576665fa0aa1f",
  "technical_execution_starting_head": "300cdad5629b945ffe10480351081f9c8befe7f4",
  "branch": "fix/task8b3-ref01-eligibility-repair-sync",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "scope": "EXTERNAL_SYNC_VALIDATION_ARTIFACT_CLOSURE_NO_RERUN",
  "detector_model_calls": 0,
  "canonical_source_changed_in_r1": false,
  "canonical_manifest_changed_in_r1": false,
  "sync_policy_entries": 135,
  "sync_policy_runtime_weights_included": false,
  "external_presync": {
    "exit": 1,
    "match": 133,
    "missing": 0,
    "mismatch": 2,
    "mismatch_paths": [
      "buildreasonseg/runtime/detector.py",
      "tests/test_task8b_runtime.py"
    ]
  },
  "external_sync": {
    "runs": 1,
    "exit": 0,
    "copied": 135,
    "verified": 135,
    "failures": 0
  },
  "external_postsync": {
    "exit": 0,
    "match": 135,
    "missing": 0,
    "mismatch": 0
  },
  "external_source_manifest": {
    "copied_from_git_object": true,
    "identity_rechecked_in_r1": true,
    "bytes": "<OBSERVED INTEGER FROM SECTION 4>",
    "sha256": "5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c"
  },
  "setup_checker": {
    "runs": 1,
    "exit": 0,
    "final_status": "BuildReasonSeg environment: READY"
  },
  "external_targeted_test": {
    "runs": 1,
    "exit": 0,
    "summary": "40 passed in 1.13s"
  },
  "external_full_suite": {
    "runs": 1,
    "exit": 0,
    "summary": "124 passed in 112.81s (0:01:52)",
    "expected_pass_count": 124
  },
  "external_write_performed_in_source_task": true,
  "external_write_performed_in_r1": false,
  "model_inference_performed": false,
  "wrong_e3b2_evidence_path_removed": true,
  "dedicated_report_created": true,
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY"
}
```

Only:

```text
external_source_manifest.bytes
```

may vary, and only from §4 stdout.

Do not add/remove/rename keys.

# 7. CREATE AUTHORITATIVE DEDICATED REPORT

Create exactly:

```text
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
```

with these required conclusions:

```text
# Task 8B.3-REF01-E3B2 — External Sync Validation Closure

Authoritative closure task =
8B.3-REF01-E3B2-R1

Technical source task =
8B.3-REF01-E3B2

Status =
COMPLETE

Design =
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

External pre-sync =
133 match / 0 missing / 2 mismatch

External pre-sync mismatch paths =
buildreasonseg/runtime/detector.py
tests/test_task8b_runtime.py

Controlled sync =
1 run
135 copied
135 verified
0 failures

External post-sync =
135 match / 0 missing / 0 mismatch

External source_manifest =
identical to canonical Git object
bytes = <observed §4 integer>
sha256 = 5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c

Setup checker =
BuildReasonSeg environment: READY

External targeted regression =
40 passed in 1.13s

External full delivery suite =
124 passed in 112.81s (0:01:52)

Runtime weights / runs / logs / inference outputs included by sync policy =
NO

Dependency installation =
NONE

Detector/model inference =
NONE

Canonical product / manifest / sync-helper changes in E3B2 =
NONE

R1 technical rerun =
NONE

R1 external write =
NONE

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

left/below reference-selection defects =
UNRESOLVED

Outcome =
REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED

NEXT =
REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
```

Do not add unsupported technical claims.

# 8. HISTORICAL IMPLEMENTATION REPORT

Append exactly one section to:

```text
docs/task8b3_ref01_eligibility_repair_impl.md
```

Title:

```text
## 16. E3B2-R1 — external sync validation artifact closure
```

Required content:

```text
E3B2 technical execution = ACCEPTED

Authoritative evidence =
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json

Authoritative report =
docs/task8b3_ref01_e3b2_external_sync_full_suite.md

External sync =
1 run · 135 copied · 135 verified · 0 failures

External post-sync =
135 match / 0 missing / 0 mismatch

Setup checker =
READY

Targeted regression =
40 passed in 1.13s

Full external delivery suite =
124 passed in 112.81s (0:01:52)

R1 technical command rerun =
NONE except one read-only source_manifest identity check

R1 external write =
NONE

E3B2 external sync validation =
CLOSED

NEXT =
REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
```

Do NOT rewrite historical E3B2 section.

# 9. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Replace active engineering handoff with ALL fields:

```text
Task: 8B.3-REF01-E3B2-R1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-sync
Starting HEAD: 5b8edd742981f1128c851c262ec576665fa0aa1f
Technical source task: 8B.3-REF01-E3B2
Technical execution starting HEAD: 300cdad5629b945ffe10480351081f9c8befe7f4
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Canonical source changed in R1: NO
Canonical manifest changed in R1: NO
External pre-sync: 133 match / 0 missing / 2 mismatch
External mismatch paths: buildreasonseg/runtime/detector.py; tests/test_task8b_runtime.py
External sync runs: 1
External sync: 135 copied / 135 verified / 0 failures
External post-sync: 135 match / 0 missing / 0 mismatch
External source_manifest copied from Git object: YES
External source_manifest identity rechecked in R1: YES
External source_manifest bytes: <observed §4 integer>
External source_manifest SHA256: 5c175a9ced5912106d9e3906ffedc85a1c6946b632804dc53cc5d20f15bb716c
Setup checker runs: 1
Setup checker: BuildReasonSeg environment: READY
External targeted test runs: 1
External targeted regression: 40 passed in 1.13s
External full suite runs: 1
External full suite: 124 passed in 112.81s (0:01:52)
External technical commands rerun in R1: NO
External write performed in R1: NO
Model inference performed: NO
Wrong E3B2 evidence removed: YES
Evidence: evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
Report: docs/task8b3_ref01_e3b2_external_sync_full_suite.md
Outcome: REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED
Next gate: REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects: UNRESOLVED
Next action: Awaiting ChatGPT audit; do not execute NEXT/E3C.
```

Only the manifest bytes field may vary from §4.

# 10. NO-RERUN ASSERTION

Before final diff confirm:

```text
sync helper run in R1 = NO
external --check run in R1 = NO
check_setup run in R1 = NO
pytest run in R1 = NO
py_compile run in R1 = NO
detector/model inference in R1 = NO
external write in R1 = NO
```

The §4 read-only source_manifest identity command is NOT a technical rerun.

Any violation => STOP.

# 11. FINAL DIFF GATE

Before commit:

```text
git diff --name-status 5b8edd742981f1128c851c262ec576665fa0aa1f
```

Allowed ONLY:

```text
D  evaluation/task8b3_ref01_e3b2_external_sync_validation.json
A  evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
A  docs/task8b3_ref01_e3b2_external_sync_full_suite.md
M  docs/task8b3_ref01_eligibility_repair_impl.md
M  handoff/FROM_DSH.md
M  handoff/TO_DSH.md
```

Require these do NOT appear:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

Also require:

```text
git rev-list --count 5b8edd742981f1128c851c262ec576665fa0aa1f..HEAD
= 0
```

Any mismatch => STOP.

# 12. STATUS → COMMIT MESSAGE

If §§2–11 all PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): close external extent repair validation
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record external validation closure stop
```

Commit exactly once.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count 5b8edd742981f1128c851c262ec576665fa0aa1f..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute NEXT/E3C.

Then STOP.

# 13. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- protected repo files untouched;
- one read-only external manifest identity check passes;
- wrong E3B2 evidence removed;
- exact authoritative evidence created at required path;
- exact dedicated report created at required path;
- historical implementation report receives only authoritative closure section;
- FROM_DSH contains every required field;
- no sync/setup/test/compile/inference rerun;
- no external write in R1;
- exactly one commit with exact message;
- NEXT/E3C not executed;
- STOP.
