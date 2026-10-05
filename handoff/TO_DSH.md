请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C0-R2-R2 — Correct Status Parser and Recover Audit Outputs**

# TO_DSH — Task 8B.3-REF01-E3C0-R2-R2: Correct Status Parser and Recover Audit Outputs

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `9faaf37552a39cab241623fb24b911b8706dfc5c`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required exact script: `C:\D\DeepSeekHarness\E3C0_R2_R2_audit_exact.py`
> Required script SHA256: `6edfb02cd97b7204ad9995a5af2dc4566a05d69cdb463a9cb62ca9a7213d3425`

# 0. CHATGPT AUDIT DISPOSITION

The prior R2-R1 STOP is **APPROVED AS A SAFE STOP**.

Verified STOP facts:

```text
STOP HEAD:
9faaf37552a39cab241623fb24b911b8706dfc5c

parent:
9e06b9e231be34d32983bfc12e394d629ed59366

commit:
docs(rc1): record locked replay readiness audit recovery stop

provided script SHA:
c761aa126e31756d8d4dadb64ef728df3a60047dea31b9a60c9ca65d8a0e5c0c

script runs:
1

script exit:
1

failure:
AssertionError: UNEXPECTED_WORKTREE: ['M handoff/TO_DSH.md']
```

Root cause is a ChatGPT-supplied script defect:

```python
cp.stdout.strip()
```

removed the leading space from the first porcelain-status line.

Therefore a valid:

```text
 M handoff/TO_DSH.md
```

could be transformed into:

```text
M handoff/TO_DSH.md
```

before validation.

DSH correctly STOPPED and MUST NOT be treated as having violated the task.

R2-R2 fixes only this parser defect and reruns the corrected exact audit once.

# 1. EXACT SCRIPT CORRECTION

The user supplies:

```text
E3C0_R2_R2_audit_exact.py
```

This script is derived from the prior exact R2-R1 script with only the following control corrections plus task/head identifiers:

1. starting HEAD becomes `9faaf37552a39cab241623fb24b911b8706dfc5c`;
2. task identifier becomes `8B.3-REF01-E3C0-R2-R2`;
3. stdout completion marker becomes `E3C0_R2_R2_AUDIT: COMPLETE`;
4. `run_git()` now uses:
   `rstrip("\r\n")`
   instead of `.strip()`;
5. initial `git status --short` accepts either:
   - ` M handoff/TO_DSH.md` (unstaged TO_DSH modification), or
   - `M  handoff/TO_DSH.md` (staged TO_DSH modification).

No scientific/readiness rule changed.

# 2. EXECUTOR CONTRACT

Allowed operations ONLY:

1. verify branch/head;
2. verify exact supplied script SHA;
3. inspect `git status --short`;
4. if TO_DSH is staged, leave it staged; do NOT unstage/reset it;
5. run the supplied script exactly ONCE;
6. inspect stdout and generated diff;
7. if every COMPLETE gate passes, commit exactly once;
8. push once;
9. STOP.

Forbidden:

- NO modification/reconstruction of supplied script;
- NO broad search;
- NO repo `artifacts/` or repo `inference/` discovery;
- NO external `logs/` access;
- NO detector/model inference;
- NO proposal regeneration;
- NO actual selector replay;
- NO production replay;
- NO final Demo;
- NO external write;
- NO product/test/manifest/helper edit;
- NO manual edit of generated evidence/report/FROM_DSH after successful script run;
- NO script rerun;
- NO pytest;
- NO py_compile;
- NO package/environment change;
- NO reset;
- NO checkout restore;
- NO stash;
- NO clean;
- NO amend;
- NO rebase;
- NO force push;
- NO NEXT execution.

Any uncovered state => STOP.

# 3. SCRIPT FILE / SHA GATE

Place the user-supplied script exactly at:

```text
C:\D\DeepSeekHarness\E3C0_R2_R2_audit_exact.py
```

Compute SHA256.

Require exactly:

```text
6edfb02cd97b7204ad9995a5af2dc4566a05d69cdb463a9cb62ca9a7213d3425
```

Mismatch/missing => STOP.
Do not reconstruct it.

# 4. GIT PRE-FLIGHT

Require:

```text
git branch --show-current
= audit/task8b3-ref01-locked-replay-artifacts

git rev-parse HEAD
= 9faaf37552a39cab241623fb24b911b8706dfc5c
```

Run:

```text
git status --short
```

Allowed states:

```text
<empty>
```

or exactly one of:

```text
 M handoff/TO_DSH.md
```

```text
M  handoff/TO_DSH.md
```

No other path/status.

Important:
- if TO_DSH is staged, DO NOT unstage it;
- if it is unstaged, DO NOT stage it before script execution.

# 5. EXACT SCIENTIFIC READ SCOPE

The supplied script may read ONLY:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\result.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\result.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\result.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\result.json
evaluation/task8b3_ref01_locked_reference_forensics.json
```

No discovery/search is authorized.

# 6. FROZEN READINESS RULE

For record-level replay only:

```text
persisted mask_area > 0
```

is the nonempty witness for the serialized merged proposal record.

Definitions remain:

```text
A = complete stable proposal list
B = complete selector scalars + mask_area>0
C = exact production-object/mask reconstruction material
D = complete per-proposal forensic IoU linkage
```

Allowed readiness enums ONLY:

```text
FULL_PRODUCTION_REPLAY_READY
RECORD_LEVEL_REPLAY_READY
LOCKED_REPLAY_ARTIFACTS_PARTIAL
LOCKED_REPLAY_ARTIFACTS_NOT_FOUND
```

Allowed NEXT mapping ONLY:

```text
FULL_PRODUCTION_REPLAY_READY
→ REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY

RECORD_LEVEL_REPLAY_READY
→ REF01_LOCKED_RECORD_REPLAY_DESIGN

LOCKED_REPLAY_ARTIFACTS_PARTIAL
→ REF01_LOCKED_REPLAY_INPUT_RECOVERY_DESIGN

LOCKED_REPLAY_ARTIFACTS_NOT_FOUND
→ REF01_LOCKED_REPLAY_INPUT_RECOVERY_DESIGN
```

DSH has no readiness/NEXT choice.

# 7. RUN SCRIPT EXACTLY ONCE

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3C0_R2_R2_audit_exact.py
```

Require exit code 0.

Require stdout contains:

```text
E3C0_R2_R2_AUDIT: COMPLETE
SCRIPT_SHA256= 6edfb02cd97b7204ad9995a5af2dc4566a05d69cdb463a9cb62ca9a7213d3425
CANDIDATE_ARTIFACTS= 9
```

Require:

```text
DIFF_PATHS=
```

contains exactly these four repo paths:

```text
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Accept the script-emitted `READINESS=` and `NEXT_GATE=` exactly.
Do not override them.

Any failure => do not rerun.

# 8. GENERATED EVIDENCE GATE

Require:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
```

contains:

```text
task = 8B.3-REF01-E3C0-R2-R2
base_head = 9faaf37552a39cab241623fb24b911b8706dfc5c
branch = audit/task8b3-ref01-locked-replay-artifacts
audit_script_sha256 = 6edfb02cd97b7204ad9995a5af2dc4566a05d69cdb463a9cb62ca9a7213d3425

detector_model_calls = 0
proposal_regeneration_performed = false
actual_replay_performed = false
product_source_changed = false
external_write_performed = false
broad_search_performed = false
```

Require exactly 9 candidate artifacts.

Every candidate object must contain exactly:

```text
path
bytes
sha256
tracked_by_git
format
supports_cases
fields_present
inspection_status
```

Require cases exactly:

```text
right
left
above
below
```

Every case must contain exactly:

```text
tile_id
artifact_paths
complete_proposal_list_present
proposal_scalar_fields_complete
exact_mask_or_object_material_present
forensic_correctness_linkage_present
replay_record_count
consistency_checks
```

# 9. EXACT CONSISTENCY KEYS

right:

```text
raw_count
merged_count
eligible_count
pre_selected
```

left:

```text
raw_count
merged_count
eligible_count
pre_selected
best_covered
```

above:

```text
raw_count
merged_count
eligible_count
pre_selected
best_covered
proposal5_mask_area
proposal5_confidence
proposal5_touches_image_border
proposal5_bbox_extent_ratio
proposal5_iou
proposal3_mask_area
proposal3_touches_image_border
proposal3_bbox_extent_ratio
```

below:

```text
raw_count
merged_count
eligible_count
pre_selected
best_covered
```

Every value exactly one of:

```text
MATCH
MISMATCH
NOT_AVAILABLE
```

# 10. REPORT / FROM_DSH

Report must identify:

```text
Task 8B.3-REF01-E3C0-R2-R2
starting HEAD = 9faaf37552a39cab241623fb24b911b8706dfc5c
```

FROM_DSH current handoff must identify:

```text
Task: 8B.3-REF01-E3C0-R2-R2
Status: COMPLETE
Starting HEAD: 9faaf37552a39cab241623fb24b911b8706dfc5c
Known artifacts inspected: 9
```

Preserve ARTIFACT-FACTS byte-for-byte.

# 11. NON-CLAIMS

Regardless of readiness:

```text
repaired selector replay = NOT PERFORMED
right1 / left14 / above5 / below1 = NOT YET REPLAY-VALIDATED
REF-01 = ACTIVE
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
final Demo = NOT RUN
```

# 12. FINAL DIFF GATE

Before commit require:

```text
git diff 9faaf37552a39cab241623fb24b911b8706dfc5c --name-only
```

exactly:

```text
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No protected source/test/manifest/helper diff.

Require:

```text
git rev-list --count 9faaf37552a39cab241623fb24b911b8706dfc5c..HEAD
= 0
```

before commit.

# 13. COMPLETE COMMIT

Only if all gates pass:

```text
git add docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
git add evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
git add handoff/FROM_DSH.md
git add handoff/TO_DSH.md

git commit -m "docs(rc1): complete locked replay readiness recovery"
```

Require exactly one new commit.

Push current branch once.

No force push.

Then STOP.

# 14. STOP PROTOCOL

If any gate fails:

- do not rerun script;
- do not manually repair evidence/report;
- record the exact factual failure in FROM_DSH;
- preserve any factual script-generated outputs if they exist;
- commit exactly once using:

```text
docs(rc1): record locked replay readiness recovery stop
```

Push once and STOP.

# 15. COMPLETE DEFINITION

COMPLETE requires:

- exact current head;
- corrected exact script SHA;
- one script run;
- exit 0;
- exactly 9 artifacts;
- exact evidence/candidate/case schema;
- exact consistency keys;
- readiness/NEXT from frozen mapping only;
- zero inference/regeneration/replay/external write;
- exact four-path diff;
- one COMPLETE commit;
- push once;
- NEXT not executed;
- STOP.
