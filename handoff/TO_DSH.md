请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C1 — Locked Record-Level Reference Selection Replay**

# TO_DSH — Task 8B.3-REF01-E3C1: Locked Record-Level Reference Selection Replay

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `6d1e8d1e69f3adefb4301948cae8918d4aec7ee1`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required script: `C:\D\DeepSeekHarness\E3C1_locked_record_replay.py`
> Required SHA256: `45134e8f26de211805bcf9a0545400aa0e00e91e702c237a4a6cd5b7266743ed`

# 0. CHATGPT AUDIT DISPOSITION

Task `8B.3-REF01-E3C0-R2-R4` is APPROVED.

Frozen readiness:

```text
right: A=true B=true C=false D=true
left:  A=true B=true C=false D=true
above: A=true B=true C=false D=true
below: A=true B=true C=false D=true

overall_readiness =
RECORD_LEVEL_REPLAY_READY

production-object replay possible =
NO

record-level replay possible =
YES

historical consistency mismatches =
0

NEXT =
REF01_LOCKED_RECORD_REPLAY_DESIGN
```

This taskbook is the ChatGPT-owned design and execution contract for that next gate.

# 1. FROZEN REPLAY DESIGN

Replay the same four persisted merged-proposal record sets:

```text
right / 1010 / 6 records
left  / 1003 / 53 records
above / 1008 / 9 records
below / 1009 / 6 records
```

The record-level adapter is frozen as follows:

```text
proposal_id             = persisted exact value
source_tile_id          = persisted exact value
confidence              = persisted exact value
mask_area               = persisted exact value
global_bbox             = persisted exact value
touches_image_border    = persisted exact value
border_clearance        = persisted exact value
centroid                = persisted exact value
raw_index               = persisted exact value

mask_crop =
1x1 True bool array,
ONLY as the frozen mask_area>0 nonempty witness

tile_index =
0 sentinel,
because current reference eligibility/ranking/exception code does not use tile_index

pad_mask_empty =
False

image_size =
None
```

This is explicitly:

```text
RECORD-LEVEL SELECTOR REPLAY
```

It is NOT:

```text
FULL PRODUCTION-OBJECT REPLAY
```

# 2. SELECTOR CODE TO EXECUTE

Do NOT reimplement the repaired selector.

The script must import and execute the current canonical:

```text
eligible_proposals()
select_reference()
GlobalProposal
MERGE_BBOX_EXTENT_RATIO_MAX
```

from:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
```

Canonical detector identity must be:

```text
Git-canonical bytes =
21257

Git-canonical SHA256 =
934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883
```

The historical baseline is replayed using current unchanged base eligibility plus rank:

```text
(-mask_area, -confidence, proposal_id)
```

without the extent exception.

The repaired selector is replayed by calling current:

```text
select_reference(..., family="largest")
```

# 3. REQUIRED LOCKED RESULT

The exact result required for COMPLETE is:

```text
right:  1 -> 1
left:  14 -> 14
above:  4 -> 5
below:  1 -> 1
```

Additionally:

```text
above proposal 5
MUST be an extent-exception candidate

above proposal 3
MUST NOT be an extent-exception candidate
because it touches the image border
```

Semantic linkage:

```text
right repaired id 1 =
locked best-IoU reference

above repaired id 5 =
locked best-IoU reference

left repaired id 14 !=
locked best-IoU id 30

below repaired id 1 !=
locked best-IoU id 2
```

Therefore if the exact replay passes:

```text
eligibility repair validation =
PASS

eligibility blocker resolved =
above

selection blockers remaining =
left, below

REF-01 =
ACTIVE

NEXT =
REF01_SELECTION_REPAIR_DESIGN
```

# 4. EXECUTOR CONTRACT

Allowed ONLY:

1. verify exact branch/head;
2. verify exact script SHA;
3. run supplied script exactly ONCE;
4. inspect stdout;
5. inspect generated evidence/report/FROM_DSH;
6. verify repo status contains only four allowed paths;
7. stage exactly four paths;
8. commit exactly once;
9. push once;
10. STOP.

Forbidden:

- NO detector/YOLO inference;
- NO SAM/SAM2 inference;
- NO Qwen/MLLM inference;
- NO proposal regeneration;
- NO image inference;
- NO detector pipeline replay;
- NO full production-object replay claim;
- NO external write;
- NO product/test/manifest/helper modification;
- NO threshold tuning;
- NO algorithm modification;
- NO manual edit of script output after successful execution;
- NO script modification/reconstruction;
- NO script rerun;
- NO pytest;
- NO py_compile;
- NO package/environment change;
- NO reset/amend/rebase/stash/clean;
- NO force push;
- NO NEXT execution.

# 5. EXACT SCIENTIFIC READ INPUTS

The script may read ONLY these scientific inputs:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\proposals.json

evaluation/task8b3_ref01_locked_reference_forensics.json

delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
```

No broad search.

# 6. INPUT HASH GATE

The supplied script verifies exactly:

```text
right proposals SHA256 =
2b09041ee86d5e7d71c05c32f2cea8669bdfbb5b7f96cda33d5e12c99a0ac45b

left proposals SHA256 =
62865e42b4d92077ce336522b97f33931b351d46ab4ec7443fb1afe2dfd798a6

above proposals SHA256 =
257297112fb2fe35f2225725e8c0a0e79338dcde10f9f4dbe20641c8fbb85e0b

below proposals SHA256 =
7c1a45690c860910b5626fc603fe942cc0497157bf6fc5d1645657029c167752

forensics SHA256 =
f7495796577cb26a14a6b76ffcb4cf9544524c25c37cdf58f6ba2b15214c6f84
```

Any mismatch => STOP.

# 7. SCRIPT GATE

Place exactly:

```text
C:\D\DeepSeekHarness\E3C1_locked_record_replay.py
```

Require SHA256:

```text
45134e8f26de211805bcf9a0545400aa0e00e91e702c237a4a6cd5b7266743ed
```

Do not edit/rebuild.

# 8. GIT PRE-FLIGHT

Require:

```text
branch =
audit/task8b3-ref01-locked-replay-artifacts

HEAD =
6d1e8d1e69f3adefb4301948cae8918d4aec7ee1
```

`git status --short` may be:
- empty;
- ` M handoff/TO_DSH.md`;
- `M  handoff/TO_DSH.md`.

No other path/status.

# 9. RUN EXACT SCRIPT ONCE

Run:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3C1_locked_record_replay.py
```

Require exit 0.

Require stdout exactly contains:

```text
E3C1_LOCKED_RECORD_REPLAY: COMPLETE
RIGHT=1->1
LEFT=14->14
ABOVE=4->5
BELOW=1->1
ELIGIBILITY_REPAIR_VALIDATION=PASS
NEXT_GATE=REF01_SELECTION_REPAIR_DESIGN
```

No rerun.

# 10. GENERATED OUTPUTS

Required new files:

```text
evaluation/task8b3_ref01_e3c1_locked_record_replay.json
docs/task8b3_ref01_e3c1_locked_record_replay.md
```

Required modified:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Evidence must contain:

```text
task = 8B.3-REF01-E3C1
design_id = LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
detector_model_calls = 0
proposal_regeneration_performed = false
actual_detector_pipeline_replay_performed = false
external_write_performed = false
product_source_changed = false
eligibility_repair_validation = PASS
eligibility_blocker_resolved_cases = ["above"]
selection_blockers_remaining = ["left","below"]
ref01_status = ACTIVE
prop01_status = PROP01_OPEN_ENGINEERING_DEFECT
overall_outcome = REF01_ELIGIBILITY_REPAIR_LOCKED_RECORD_REPLAY_PASS
next_gate = REF01_SELECTION_REPAIR_DESIGN
```

# 11. STATUS / DIFF GATE

After script success, before staging, run:

```text
git status --porcelain=v1 --untracked-files=all
```

The only path names allowed are:

```text
docs/task8b3_ref01_e3c1_locked_record_replay.md
evaluation/task8b3_ref01_e3c1_locked_record_replay.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path.

Then stage exactly:

```text
git add docs/task8b3_ref01_e3c1_locked_record_replay.md
git add evaluation/task8b3_ref01_e3c1_locked_record_replay.json
git add handoff/FROM_DSH.md
git add handoff/TO_DSH.md
```

Require:

```text
git diff --cached --name-only
```

contains exactly the same four paths.

# 12. COMPLETE COMMIT

If every gate passes:

```text
git commit -m "docs(rc1): replay locked reference selection records"
```

Require exactly one new commit over `6d1e8d1e69f3adefb4301948cae8918d4aec7ee1`.

Push current branch once.

No force push.

STOP immediately.
Do not execute `REF01_SELECTION_REPAIR_DESIGN`.

# 13. STOP PROTOCOL

If any gate fails:

- do not rerun the script;
- do not self-repair;
- preserve any factual generated output;
- update only FROM_DSH with exact STOP facts if needed;
- commit exactly once using:

```text
docs(rc1): record locked reference replay stop
```

Push once and STOP.

# 14. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/head;
- exact script SHA;
- exact frozen input hashes;
- canonical detector identity exact;
- script runs exactly once;
- baseline replay = 1/14/4/1;
- repaired replay = 1/14/5/1;
- above 5 admitted by extent exception;
- above 3 excluded by border;
- right/above repaired selection equals locked best-IoU id;
- left/below remain selection blockers;
- zero detector/model calls;
- zero proposal regeneration;
- zero external writes;
- no product changes;
- exact four repo paths committed;
- exact commit message;
- push succeeds;
- NEXT not executed;
- STOP.
