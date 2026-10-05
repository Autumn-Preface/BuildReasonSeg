请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C0-R2-R4 — Final Locked Replay Readiness Audit**

# TO_DSH — Task 8B.3-REF01-E3C0-R2-R4: Final Locked Replay Readiness Audit

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `31006384fe65bb29247c67d3b504b53d9c1ae981`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required exact script: `C:\D\DeepSeekHarness\E3C0_R2_R4_final_readiness_audit.py`
> Required script SHA256: `3865eefadb84a0de61f83198471765272bc6bc0b005ad4ff24f8ef4e58850f2c`

# 0. CHATGPT AUDIT DISPOSITION

R2-R3 is approved as a SAFE STOP with usable schema evidence.

The schema probe successfully established the real frozen `proposals.json` format for all four locked cases:

```json
{
  "count": N,
  "items": [
    {
      "proposal_id": ...,
      "source_tile_id": ...,
      "confidence": ...,
      "mask_area": ...,
      "global_bbox": ...,
      "centroid": ...,
      "touches_image_border": ...,
      "border_clearance": ...,
      "bbox_extent_ratio": ...,
      "raw_index": ...
    }
  ]
}
```

Observed counts:

```text
right / 1010 = 6
left  / 1003 = 53
above / 1008 = 9
below / 1009 = 6
```

The R2-R3 script's final STOP was caused only by another ChatGPT control-gate defect: it used `git diff --name-only` to expect newly created untracked evidence/report files before they had been added. Git does not list untracked files in that command.

The STOP commit preserved the schema evidence/report, so NO additional schema probe is needed.

R2-R4 is the final readiness audit.

# 1. FROZEN READINESS RULE

Use exactly:

```text
A = complete stable proposal list
B = complete selector scalars + persisted mask_area > 0 as record-level nonempty witness
C = exact production-object / mask reconstruction material
D = complete per-proposal forensic IoU linkage
```

Important:

```text
mask_area > 0
```

is accepted only for record-level replay.

It does NOT satisfy C.

Allowed readiness enums ONLY:

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

no candidate artifacts
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

DSH has no discretion.

# 2. EXECUTOR CONTRACT

Allowed ONLY:

1. verify branch/head;
2. verify exact supplied script SHA;
3. run exact script ONCE;
4. inspect stdout and generated tracked diff;
5. verify generated evidence/report/FROM_DSH;
6. commit exactly once;
7. push once;
8. STOP.

Forbidden:

- NO broad search;
- NO schema probe rerun;
- NO detector/model inference;
- NO proposal regeneration;
- NO actual selector replay;
- NO production replay;
- NO final Demo;
- NO external write;
- NO product/test/manifest/helper edit;
- NO manual evidence/report/FROM_DSH edit after script success;
- NO script modification/reconstruction;
- NO script rerun;
- NO pytest;
- NO py_compile;
- NO package/environment change;
- NO reset/amend/rebase/stash/clean;
- NO force push;
- NO NEXT execution.

# 3. EXACT SCIENTIFIC READ SCOPE

The exact script may read ONLY these 9 scientific artifacts:

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

It may read Git state and FROM_DSH only for control/ARTIFACT-FACTS preservation.

No discovery/search.

# 4. SCRIPT SHA GATE

Place exactly:

```text
C:\D\DeepSeekHarness\E3C0_R2_R4_final_readiness_audit.py
```

Require SHA256:

```text
3865eefadb84a0de61f83198471765272bc6bc0b005ad4ff24f8ef4e58850f2c
```

Do not modify/rebuild.

# 5. GIT PRE-FLIGHT

Require:

```text
branch = audit/task8b3-ref01-locked-replay-artifacts
HEAD = 31006384fe65bb29247c67d3b504b53d9c1ae981
```

`git status --short` may be empty or contain exactly one TO_DSH modification:

```text
 M handoff/TO_DSH.md
```

or:

```text
M  handoff/TO_DSH.md
```

No other path/status.

# 6. RUN EXACT SCRIPT ONCE

Run exactly once:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3C0_R2_R4_final_readiness_audit.py
```

Require exit 0.

Require stdout:

```text
E3C0_R2_R4_FINAL_AUDIT: COMPLETE
SCRIPT_SHA256= 3865eefadb84a0de61f83198471765272bc6bc0b005ad4ff24f8ef4e58850f2c
SCHEMA=PROPOSALS_JSON_COUNT_ITEMS_V1
CANDIDATE_ARTIFACTS=9
```

Accept the emitted:

```text
READINESS=
NEXT_GATE=
```

exactly as produced.

Do not reinterpret or rerun.

# 7. EVIDENCE CONTRACT

Require:

```text
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
```

contains:

```text
task = 8B.3-REF01-E3C0-R2-R4
base_head = 31006384fe65bb29247c67d3b504b53d9c1ae981
schema_contract = PROPOSALS_JSON_COUNT_ITEMS_V1
audit_script_sha256 = 3865eefadb84a0de61f83198471765272bc6bc0b005ad4ff24f8ef4e58850f2c

detector_model_calls = 0
proposal_regeneration_performed = false
actual_replay_performed = false
product_source_changed = false
external_write_performed = false
broad_search_performed = false
```

Require exactly 9 candidate artifacts.

Each candidate object exactly:

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

Require exactly four cases:
- right
- left
- above
- below

Each case exactly:

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

# 8. CONSISTENCY KEY CONTRACT

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

Every value must be exactly:

```text
MATCH
MISMATCH
NOT_AVAILABLE
```

# 9. NON-CLAIMS

Regardless of readiness:

```text
repaired selector replay = NOT PERFORMED
right1 / left14 / above5 / below1 = NOT YET REPLAY-VALIDATED
REF-01 = ACTIVE
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
final Demo = NOT RUN
```

# 10. FINAL DIFF GATE

Require exactly these four changed paths relative to `31006384fe65bb29247c67d3b504b53d9c1ae981`:

```text
docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No source/test/manifest/helper diff.

# 11. COMPLETE COMMIT

Only if every gate passes:

```text
git add docs/task8b3_ref01_e3c0_locked_replay_artifact_audit.md
git add evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json
git add handoff/FROM_DSH.md
git add handoff/TO_DSH.md

git commit -m "docs(rc1): establish locked replay readiness"
```

Exactly one commit.
Push current branch once.
No force push.
Then STOP.

# 12. STOP PROTOCOL

If any gate fails:
- do not rerun;
- do not manually fix generated evidence/report;
- record exact factual failure in FROM_DSH;
- commit exactly once with:
  `docs(rc1): record final locked replay readiness stop`
- push once;
- STOP.

# 13. COMPLETE DEFINITION

COMPLETE only if:
- exact head;
- exact script SHA;
- one script run / exit 0;
- real `count/items` schema used;
- exactly 9 artifacts;
- exact A/B/C/D schema;
- exact consistency keys;
- legal readiness/NEXT only;
- zero inference/regeneration/replay/external write;
- exact four-path diff;
- one commit;
- push once;
- NEXT not executed;
- STOP.
