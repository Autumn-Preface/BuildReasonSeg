请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C0-R2-R1 — Recover Missing Locked Replay Audit Outputs**

# TO_DSH — Task 8B.3-REF01-E3C0-R2-R1: Recover Missing Locked Replay Audit Outputs

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `9e06b9e231be34d32983bfc12e394d629ed59366`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required exact audit script local path: `C:\D\DeepSeekHarness\E3C0_R2_R1_audit_exact.py`
> Required exact audit script SHA256: `c761aa126e31756d8d4dadb64ef728df3a60047dea31b9a60c9ca65d8a0e5c0c`

# 0. CHATGPT AUDIT DISPOSITION

The prior R2 commit is **NOT APPROVED**.

Verified remote facts:

```text
R2 HEAD:
9e06b9e231be34d32983bfc12e394d629ed59366

R2 parent:
95c2a8b6ee86e4835909e82e5d58d372246f0ec3

R2 commit message:
docs(rc1): finalize locked replay readiness audit
```

But the R2 commit changed ONLY:

```text
handoff/TO_DSH.md
```

The required R2 outputs were NOT updated.

At `9e06b9e231be34d32983bfc12e394d629ed59366` the following files are still R1 content:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
handoff/FROM_DSH.md
```

Therefore the prior R2 `COMPLETE` claim is invalid.

This R2-R1 does NOT rewrite history.
It performs one fresh exact correction commit from `9e06b9e231be34d32983bfc12e394d629ed59366`.

# 1. EXECUTOR CONTRACT

DSH has ZERO technical discretion.

Allowed operations ONLY:

1. verify branch / starting HEAD;
2. verify working tree is clean except the user-replaced `handoff/TO_DSH.md`;
3. verify exact audit script exists and SHA256 matches;
4. run the exact script ONCE;
5. require script exit code 0;
6. require exact stdout gates below;
7. inspect generated repo diff only;
8. make exactly one correction commit;
9. push once;
10. STOP.

Forbidden:

- NO broad search;
- NO repo `artifacts/` access;
- NO repo `inference/` access;
- NO external `logs/` access;
- NO detector / YOLO / SAM / Qwen / MLLM inference;
- NO proposal regeneration;
- NO actual selector replay;
- NO production replay;
- NO final Demo;
- NO external write;
- NO source / test / manifest / sync-helper modification;
- NO manual edit of evidence/report/FROM_DSH after script execution;
- NO script modification;
- NO script reconstruction;
- NO script rerun;
- NO pytest;
- NO py_compile;
- NO package/environment change;
- NO amend;
- NO rebase;
- NO reset;
- NO stash;
- NO clean;
- NO force push;
- NO intermediate commit;
- NO intermediate push;
- NO NEXT execution.

If any required condition fails:
- do NOT invent a workaround;
- do NOT claim COMPLETE;
- follow §12 STOP protocol.

# 2. USER-SUPPLIED EXACT SCRIPT

The user will provide:

```text
E3C0_R2_R1_audit_exact.py
```

Place/copy it at exactly:

```text
C:\D\DeepSeekHarness\E3C0_R2_R1_audit_exact.py
```

DSH MUST NOT modify, regenerate, reformat, or patch this file.

Before execution compute SHA256.

Require exactly:

```text
c761aa126e31756d8d4dadb64ef728df3a60047dea31b9a60c9ca65d8a0e5c0c
```

If missing/mismatch => STOP.

# 3. GIT PRE-FLIGHT

From repo root require:

```text
git branch --show-current
= audit/task8b3-ref01-locked-replay-artifacts

git rev-parse HEAD
= 9e06b9e231be34d32983bfc12e394d629ed59366
```

Run:

```text
git status --short
```

Allowed result:
- no lines; or
- exactly one line:
  ` M handoff/TO_DSH.md`

Anything else => STOP.

Require:

```text
git rev-list --count 9e06b9e231be34d32983bfc12e394d629ed59366..HEAD
= 0
```

# 4. EXACT READ SCOPE

The exact script may READ ONLY these nine scientific artifacts:

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

It also reads the four repo control files needed for:
- branch/head/status/diff;
- preserving `ARTIFACT-FACTS`;
- verifying its own generated outputs.

NO discovery/search is authorized.

# 5. READINESS RULE — FROZEN

For record-level selector replay:

```text
persisted mask_area > 0
```

is accepted as the nonempty witness for the serialized merged proposal record.

This is ONLY for record-level replay.

For full production-object replay, exact `mask_crop` / `GlobalProposal` reconstruction material remains mandatory.

Definitions:

```text
A = complete stable proposal list
B = complete selector scalars + mask_area>0
C = exact production object / mask material
D = complete per-proposal forensic IoU linkage
```

Allowed readiness enum ONLY:

```text
FULL_PRODUCTION_REPLAY_READY
RECORD_LEVEL_REPLAY_READY
LOCKED_REPLAY_ARTIFACTS_PARTIAL
LOCKED_REPLAY_ARTIFACTS_NOT_FOUND
```

Mechanical mapping:

```text
A+B+C+D all four cases
→ FULL_PRODUCTION_REPLAY_READY

A+B+D all four cases and C false for >=1 case
→ RECORD_LEVEL_REPLAY_READY

known artifacts exist but A/B/D incomplete
→ LOCKED_REPLAY_ARTIFACTS_PARTIAL

no candidate artifact
→ LOCKED_REPLAY_ARTIFACTS_NOT_FOUND
```

NEXT mapping ONLY:

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

# 6. RUN EXACT SCRIPT ONCE

Run exactly once:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3C0_R2_R1_audit_exact.py
```

Do NOT redirect into a repo file.

Require exit code 0.

Require stdout contains:

```text
E3C0_R2_R1_AUDIT: COMPLETE
SCRIPT_SHA256= c761aa126e31756d8d4dadb64ef728df3a60047dea31b9a60c9ca65d8a0e5c0c
CANDIDATE_ARTIFACTS= 9
```

`READINESS=` and `NEXT_GATE=` must be accepted exactly as produced by the script.
DSH must not reinterpret them.

Require stdout `DIFF_PATHS=` contains exactly these four paths:

```text
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

If the script exits nonzero or any stdout gate fails => STOP.
DO NOT rerun.

# 7. EVIDENCE GATE

After successful script execution, inspect:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
```

Require:

```text
task = 8B.3-REF01-E3C0-R2-R1
base_head = 9e06b9e231be34d32983bfc12e394d629ed59366
branch = audit/task8b3-ref01-locked-replay-artifacts
audit_script_sha256 = c761aa126e31756d8d4dadb64ef728df3a60047dea31b9a60c9ca65d8a0e5c0c

detector_model_calls = 0
proposal_regeneration_performed = false
actual_replay_performed = false
product_source_changed = false
external_write_performed = false
broad_search_performed = false
```

Require exactly 9 `candidate_artifacts`.

Each candidate artifact object must contain exactly:

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

No `root` field.

Require four cases exactly:

```text
right
left
above
below
```

Each case must contain exactly:

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

# 8. CONSISTENCY KEY GATE

Require exact keys.

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

Every consistency value must be exactly one of:

```text
MATCH
MISMATCH
NOT_AVAILABLE
```

# 9. REPORT / FROM_DSH GATE

Require report path:

```text
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
```

It must identify:

```text
Task 8B.3-REF01-E3C0-R2-R1
starting HEAD = 9e06b9e231be34d32983bfc12e394d629ed59366
```

Require `handoff/FROM_DSH.md` current report identifies:

```text
Task: 8B.3-REF01-E3C0-R2-R1
Status: COMPLETE
Starting HEAD: 9e06b9e231be34d32983bfc12e394d629ed59366
Known artifacts inspected: 9
```

Require the original `ARTIFACT-FACTS` block is preserved.

# 10. SCIENTIFIC NON-CLAIMS

Regardless of readiness:

```text
actual repaired-selector replay = NOT PERFORMED
right1 / left14 / above5 / below1 = NOT YET REPLAY-VALIDATED
REF-01 = ACTIVE
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
final Demo = NOT RUN
```

Do not strengthen these claims.

# 11. FINAL DIFF GATE

Run:

```text
git diff --name-only 9e06b9e231be34d32983bfc12e394d629ed59366
```

Require exactly:

```text
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path.

Protected paths must have no diff:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
scripts/sync_advisor_rc1_delivery.py
docs/task8b3_ref01_eligibility_repair_impl.md
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
```

Before final commit require:

```text
git rev-list --count 9e06b9e231be34d32983bfc12e394d629ed59366..HEAD
= 0
```

# 12. COMMIT / STOP PROTOCOL

## COMPLETE

Only if §§2–11 all pass:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): recover locked replay readiness audit outputs
```

Then:

```text
git add docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
git add evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
git add handoff/FROM_DSH.md
git add handoff/TO_DSH.md
git commit -m "docs(rc1): recover locked replay readiness audit outputs"
```

Require:

```text
git rev-list --count 9e06b9e231be34d32983bfc12e394d629ed59366..HEAD
= 1
```

Push current branch exactly once.

No force push.

Then STOP.

## STOP

If anything fails before the exact script completes:
- do not fabricate evidence;
- do not manually repair generated artifacts;
- write a minimal STOP reason only into `handoff/FROM_DSH.md`;
- preserve `ARTIFACT-FACTS`;
- commit exactly once with:

```text
docs(rc1): record locked replay readiness audit recovery stop
```

If the exact script already modified the three generated outputs and then a later gate fails:
- do not rerun the script;
- keep those outputs as factual partial evidence;
- append only the STOP reason to `handoff/FROM_DSH.md`;
- commit exactly once with the same STOP message.

Push once, then STOP.

Under no circumstance may a failed gate use the COMPLETE commit message.

# 13. COMPLETE DEFINITION

COMPLETE only if:

- branch/head exactly correct;
- script SHA exactly correct;
- script runs exactly once;
- script exit = 0;
- generated evidence is R2-R1, not R1/R2;
- exact nine artifacts;
- exact candidate schema;
- exact case schema;
- exact consistency keys;
- readiness/NEXT only from frozen mapping;
- zero inference/regeneration/replay/external write;
- exact four-path diff;
- no protected-path diff;
- exactly one correction commit;
- exact COMPLETE commit message;
- push succeeds;
- NEXT not executed;
- STOP after push.
