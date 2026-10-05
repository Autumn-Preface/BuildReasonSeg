请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C0-R2 — Locked Replay Readiness Audit Finalization**

# TO_DSH — Task 8B.3-REF01-E3C0-R2: Locked Replay Readiness Audit Finalization

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `95c2a8b6ee86e4835909e82e5d58d372246f0ec3`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required exact audit script local path: `C:\D\DeepSeekHarness\E3C0_R2_audit_exact.py`
> Required audit script SHA256: `955d589643b1eec141ec3eb8ea181b2ba4b41577e4e96d0431d3698fe3ac958d`

# 0. CHATGPT AUDIT DISPOSITION

E3C0-R1 is **NOT FORMALLY APPROVED**.

R1 successfully corrected the forbidden-root problem, but two issues remain:

1. artifact/evidence schema still deviated from the frozen contract:
   - `candidate_artifacts` omitted required fields;
   - `consistency_checks` used self-invented keys rather than the exact frozen keys;
   - FROM_DSH did not use the required exact field contract.

2. R1 classified the artifacts as `LOCKED_REPLAY_ARTIFACTS_PARTIAL`, but this is not yet supported:
   - all four saved `proposals.json` files have record counts exactly equal to the frozen merged counts: 6 / 53 / 9 / 6;
   - R1 itself reported scalar fields complete;
   - the allowed Git-tracked forensics artifact stores per-proposal IoU for all proposal IDs in all four cases;
   - therefore A/B/D require one narrow deterministic re-audit before readiness can be frozen.

R2 performs **NO broad search**. It inspects only nine already-known artifacts.

# 1. CHATGPT READINESS RULE REFINEMENT

For this R2 and future **record-level** replay only:

```text
persisted mask_area > 0
```

is accepted as the nonempty witness for a serialized merged proposal record.

Reason:
- record-level replay does not instantiate `GlobalProposal`;
- `mask_area` is the persisted production scalar used for the same merged proposal;
- this rule is used only to decide whether the saved scalar record is sufficient for deterministic selector replay.

This does **NOT** satisfy exact production-object replay.

For C / `FULL_PRODUCTION_REPLAY_READY`, exact `mask_crop` and current `GlobalProposal` reconstruction material remain mandatory.

Therefore:

```text
A = complete stable proposal list
B = complete selector scalars + mask_area>0 record-level nonempty witness
C = exact GlobalProposal/mask reconstruction material
D = complete per-proposal forensic IoU linkage
```

The exact four allowed readiness enums remain unchanged.

# 2. USER-SUPPLIED EXACT SCRIPT

The user will provide the accompanying file:

```text
E3C0_R2_audit_exact.py
```

Place/copy it at exactly:

```text
C:\D\DeepSeekHarness\E3C0_R2_audit_exact.py
```

DSH MUST NOT reconstruct, edit, reformat, patch, or replace it.

Before running it, compute SHA256 and require exactly:

```text
955d589643b1eec141ec3eb8ea181b2ba4b41577e4e96d0431d3698fe3ac958d
```

If the script is missing or SHA mismatches => STOP.

# 3. EXECUTOR CONTRACT

Allowed operations ONLY:

1. verify exact branch/head;
2. verify working tree is clean except `handoff/TO_DSH.md`;
3. verify exact external audit-script SHA;
4. run that exact script ONCE;
5. inspect its stdout and generated repo diff;
6. verify the exact gates below;
7. make exactly one R2 commit;
8. push once;
9. STOP.

Forbidden:
- NO broad file search;
- NO repo `artifacts/` or repo `inference/` access;
- NO external `logs/` access;
- NO detector/model inference;
- NO proposal regeneration;
- NO production replay;
- NO final Demo;
- NO external write;
- NO product/test/manifest/helper edit;
- NO manual evidence/report/FROM_DSH editing;
- NO script modification;
- NO script rerun;
- NO pytest / py_compile;
- NO package/environment change;
- NO amend/rebase/reset/stash/clean;
- NO intermediate commit/push;
- NO NEXT.

Any script assertion failure => STOP. Do not repair it.

# 4. GIT GATE

Require:

```text
git branch --show-current
= audit/task8b3-ref01-locked-replay-artifacts

git rev-parse HEAD
= 95c2a8b6ee86e4835909e82e5d58d372246f0ec3
```

Allowed initial status:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

# 5. EXACT ARTIFACT SCOPE

The audit script is authorized to READ only:

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

No other discovery/search is authorized.

# 6. RUN EXACT SCRIPT ONCE

From repository root run exactly once:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3C0_R2_audit_exact.py
```

Require exit 0.

Capture stdout exactly.

Expected first line:

```text
E3C0_R2_AUDIT: COMPLETE
```

The observed `READINESS=` and `NEXT_GATE=` are determined mechanically by the exact script. DSH MUST NOT override them.

Do NOT rerun.

# 7. GENERATED ARTIFACT CONTRACT

The script must modify exactly:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
handoff/FROM_DSH.md
```

Together with the user-provided taskbook:

```text
handoff/TO_DSH.md
```

no other repo path may differ from starting HEAD.

## Evidence requirements

Require:
- `task = 8B.3-REF01-E3C0-R2`;
- `candidate_artifacts` contains exactly 9 records;
- every candidate record contains exactly the required semantic fields:
  - `path`
  - `bytes`
  - `sha256`
  - `tracked_by_git`
  - `format`
  - `supports_cases`
  - `fields_present`
  - `inspection_status`;
- four case objects use only the frozen case schema;
- every consistency value is exactly one of:
  - `MATCH`
  - `MISMATCH`
  - `NOT_AVAILABLE`;
- no custom readiness enum exists.

## Exact consistency keys

Require:

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

# 8. READINESS MAPPING — NO DSH CHOICE

The exact script chooses mechanically:

```text
FULL_PRODUCTION_REPLAY_READY
```
iff A+B+C+D all true for all cases.

```text
RECORD_LEVEL_REPLAY_READY
```
iff A+B+D all true for all cases and C false for at least one case.

```text
LOCKED_REPLAY_ARTIFACTS_PARTIAL
```
iff known artifacts exist but A/B/D are incomplete for at least one case.

```text
LOCKED_REPLAY_ARTIFACTS_NOT_FOUND
```
only if no candidate artifacts exist.

NEXT is mechanically mapped:

```text
FULL_PRODUCTION_REPLAY_READY
→ REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY

RECORD_LEVEL_REPLAY_READY
→ REF01_LOCKED_RECORD_REPLAY_DESIGN

PARTIAL / NOT_FOUND
→ REF01_LOCKED_REPLAY_INPUT_RECOVERY_DESIGN
```

No other enum/NEXT is allowed.

# 9. SCIENTIFIC NON-CLAIMS

Regardless of readiness:

- actual repaired selector replay has NOT occurred;
- do NOT claim `right1 / left14 / above5 / below1` validated;
- REF-01 remains ACTIVE;
- PROP-01 remains `PROP01_OPEN_ENGINEERING_DEFECT`;
- no final Demo has run;
- no detector/model call occurred.

# 10. PROTECTED FILE GATE

Require no diff for:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
scripts/sync_advisor_rc1_delivery.py
docs/task8b3_ref01_eligibility_repair_impl.md
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
```

No external file may be written.

# 11. FINAL DIFF GATE

Before commit:

```text
git diff --name-only 95c2a8b6ee86e4835909e82e5d58d372246f0ec3
```

Allowed ONLY:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Require:

```text
git rev-list --count 95c2a8b6ee86e4835909e82e5d58d372246f0ec3..HEAD
= 0
```

before commit.

# 12. COMMIT / PUSH

If the exact script exits 0 and every gate passes:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): finalize locked replay readiness audit
```

If procedure cannot be completed safely:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record locked replay readiness audit stop
```

Exactly one commit.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count 95c2a8b6ee86e4835909e82e5d58d372246f0ec3..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not execute NEXT.

Then STOP.

# 13. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- exact user-supplied script SHA verified;
- script runs exactly once;
- no broad search;
- zero inference/regeneration/replay/external writes;
- exact nine known artifacts only;
- exact candidate schema;
- exact consistency keys/status values;
- exact allowed readiness enum;
- exact NEXT mapping;
- only four allowed repo paths differ;
- exactly one commit with exact message;
- push succeeds;
- NEXT not executed;
- STOP.
