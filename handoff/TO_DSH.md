请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C3 — Selection Repair Decision Freeze**

# TO_DSH — Task 8B.3-REF01-E3C3: Selection Repair Decision Freeze

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `6e33b88ecf9c78ef9fc62a9c79ec0a8f0c55c1bd`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required script: `C:\D\DeepSeekHarness\E3C3_selection_repair_decision.py`
> Required SHA256: `5cfcc5189afe624fe13bf8c618fc3c292fb58268e54091b66e6708204ec44c35`

# 0. CHATGPT AUDIT DISPOSITION

Task `8B.3-REF01-E3C2` is APPROVED.

Verified E3C2 facts:

```text
commit:
docs(rc1): characterize remaining reference selection blocker

predeclared rank rules:
6

globally perfect locked-four rules:
NONE

scalar rule selected:
NO

product change:
NO

inference:
NONE
```

ChatGPT now makes the technical decision.

# 1. CHATGPT TECHNICAL DECISION

Freeze exactly:

```text
REJECT_FURTHER_SCALAR_RANK_REPAIR
```

Reasons:

1. `left`:
   - current = 14
   - locked best-covered = 30
   - none of the six predeclared threshold-free rules selects 30.

2. `below`:
   - current = 1
   - locked best-covered = 2
   - proposal 1 dominates proposal 2 in `(mask_area, confidence)`;
   - proposal 1 also dominates proposal 2 in `(bbox_area, confidence)`;
   - all six predeclared scalar rules still select 1.

3. Therefore new weights, thresholds, location priors, or further scalar-rule search would be post-hoc tuning to already-consumed locked qualitative cases.

4. RC1 scientific architecture is frozen. Do NOT add a new reference-refinement model branch.

# 2. FROZEN REF-01 DISPOSITION

Automatic behavior remains:

```text
right:  1 -> 1  correct stable
left:  14 -> 14 residual selection limitation
above:  4 -> 5  eligibility blocker resolved
below:  1 -> 1  residual selection limitation
```

The only validated automatic repair retained is:

```text
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
```

REF-01 is NOT marked fully closed.

Freeze status:

```text
REF-01 =
ACTIVE_RESIDUAL_SELECTION_LIMITATION
```

# 3. ENGINEERING FALLBACK

Use the already-existing RC1 assisted reference path:

```text
ASSISTED_REFERENCE_OVERRIDE
```

Existing pipeline parameter:

```text
reference_id
```

No new product code is authorized in E3C3.

This fallback must NOT be described as automatic success.

# 4. NEXT

After this decision is recorded:

```text
NEXT =
MASK01_VALIDITY_FORENSICS_DESIGN
```

Do NOT execute MASK-01 in this task.

# 5. EXECUTOR CONTRACT

Allowed ONLY:

1. verify exact branch/head;
2. verify supplied script SHA;
3. run supplied script exactly ONCE;
4. inspect stdout;
5. verify only the four allowed paths changed;
6. stage exactly those four paths;
7. commit exactly once;
8. push once;
9. STOP.

Forbidden:

- NO detector/model inference;
- NO proposal regeneration;
- NO selector replay;
- NO selector repair implementation;
- NO source/test/manifest/helper edit;
- NO scalar rule experiment;
- NO threshold tuning;
- NO architecture change;
- NO external write;
- NO final Demo;
- NO script edit/rebuild/rerun;
- NO pytest/py_compile;
- NO package/environment change;
- NO reset/amend/rebase/stash/clean;
- NO force push;
- NO NEXT execution.

# 6. SCRIPT GATE

Place exactly:

```text
C:\D\DeepSeekHarness\E3C3_selection_repair_decision.py
```

Require SHA256:

```text
5cfcc5189afe624fe13bf8c618fc3c292fb58268e54091b66e6708204ec44c35
```

Do not modify/reconstruct.

# 7. GIT PRE-FLIGHT

Require:

```text
branch = audit/task8b3-ref01-locked-replay-artifacts
HEAD = 6e33b88ecf9c78ef9fc62a9c79ec0a8f0c55c1bd
```

Allowed `git status --short`:
- empty;
- ` M handoff/TO_DSH.md`;
- `M  handoff/TO_DSH.md`.

No other path/status.

# 8. RUN ONCE

Run exactly once:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3C3_selection_repair_decision.py
```

Require exit 0.

Require stdout:

```text
E3C3_SELECTION_REPAIR_DECISION: COMPLETE
DECISION=REJECT_FURTHER_SCALAR_RANK_REPAIR
FALLBACK=ASSISTED_REFERENCE_OVERRIDE
NEXT_GATE=MASK01_VALIDITY_FORENSICS_DESIGN
```

No rerun.

# 9. REQUIRED OUTPUTS

New:

```text
evaluation/task8b3_ref01_e3c3_selection_repair_decision.json
docs/task8b3_ref01_e3c3_selection_repair_decision.md
```

Modified:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path.

Evidence must contain exactly the frozen decision:
- `decision = REJECT_FURTHER_SCALAR_RANK_REPAIR`
- `selector_repair_implementation_performed = false`
- `engineering_fallback.mode = ASSISTED_REFERENCE_OVERRIDE`
- `engineering_fallback.existing_pipeline_parameter = reference_id`
- `ref01_status = ACTIVE_RESIDUAL_SELECTION_LIMITATION`
- `prop01_status = PROP01_OPEN_ENGINEERING_DEFECT`
- `overall_outcome = REF01_SCALAR_REPAIR_REJECTED_ASSISTED_FALLBACK_FROZEN`
- `next_gate = MASK01_VALIDITY_FORENSICS_DESIGN`

# 10. DIFF / STAGING GATE

Run:

```text
git status --porcelain=v1 --untracked-files=all
```

Only these paths may appear:

```text
docs/task8b3_ref01_e3c3_selection_repair_decision.md
evaluation/task8b3_ref01_e3c3_selection_repair_decision.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Stage exactly those four.

Require `git diff --cached --name-only` contains exactly the same four paths.

# 11. COMPLETE COMMIT

If all gates pass:

```text
git commit -m "docs(rc1): freeze residual reference selection limitation"
```

Exactly one commit.
Push current branch once.
No force push.
STOP.

# 12. STOP PROTOCOL

If any gate fails:
- do not rerun;
- do not self-repair;
- record factual failure only;
- commit exactly once:
  `docs(rc1): record reference selection decision stop`
- push once;
- STOP.

# 13. COMPLETE DEFINITION

COMPLETE requires:
- exact head/script SHA;
- one script run;
- decision exactly frozen by ChatGPT;
- no new algorithm/rank/threshold;
- assisted fallback only;
- no product changes;
- exact four paths;
- exact commit message;
- push once;
- NEXT not executed;
- STOP.
