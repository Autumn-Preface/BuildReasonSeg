# TO_DSH — Task 8B.3-P1D12-R2: Close Final Inspect-Only Evidence Gaps

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `728932094a9f86f7c1607abc6a2beff12d2d2d70`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. ChatGPT audit disposition

P1D12-R1 is accepted for:

```text
locked raster SHA identities = 4/4 PASS
locked raster dimensions = 4/4 512x512
locked raster RGB readability = 4/4 PASS
external source/config integrity = 135/135 PASS
external setup = READY
existing proposal counts = reproduced
eligible counts = right 4 / left 42 / above 4 / below 3
candidate rerun = NO
candidate replacement = NO
GT / visual inspection = NO
```

Two final evidence gaps remain:

```text
GAP 1:
R1 detector-constant table omitted:
FROZEN_THRESHOLD = 0.5

GAP 2:
R1 did not directly record the nested result.json timing values:
timings.sam2
timings.relation_fields
timings.db1

The task contract requires each to equal 0.0 for all four candidates.
```

No candidate rerun is authorized.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 728932094a9f86f7c1607abc6a2beff12d2d2d70
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- execute `predict.py`;
- execute `--inspect-proposals`;
- execute detector/model inference;
- run Qwen / SAM2 / D-B1;
- run reference selection;
- run pytest;
- run sync write;
- modify canonical RC1;
- modify external RC1;
- modify/delete existing diagnostics;
- inspect preview images visually;
- access GT;
- replace candidates;
- update main;
- force push.

This task is read-only except for repository docs/handoff.

# 3. Allowed tracked changes

Only:

```text
docs/task8b3_p1d12_r1_evidence_closure.md
docs/task8b3_p1d12_r2_final_evidence_closure.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 4. Verify frozen detector threshold

Read only:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
```

Require exact effective value:

```text
FROZEN_THRESHOLD = 0.5
```

Also reconfirm, read-only:

```text
TILE_SIZE = 512
TILE_OVERLAP = 128
TILE_STRIDE = 384
IMGSZ = 640
CONF = 0.05
MAX_DET = 300
DUPLICATE_IOU = 0.50
MERGE_BBOX_EXTENT_RATIO_MAX = 0.20
```

Record:

```text
DETECTOR_CONSTANTS = MATCH
```

If `FROZEN_THRESHOLD != 0.5` or any other constant differs:
- outcome = EVIDENCE_FAILED;
- STOP.

# 5. Read exact existing result.json timing evidence

Read only the four existing files:

```text
inference/output/diagnostics/1010/result.json
inference/output/diagnostics/1003/result.json
inference/output/diagnostics/1008/result.json
inference/output/diagnostics/1009/result.json
```

Do not regenerate them.

For each candidate, record exact JSON values:

```text
status
mode
timings.language
timings.detector
timings.merge
timings.sam2
timings.relation_fields
timings.db1
timings.total
```

Require for all four:

```text
status = SUCCESS
mode = inspect
timings.sam2 = 0.0
timings.relation_fields = 0.0
timings.db1 = 0.0
```

No inference or interpretation from missing fields is allowed.
The values must be read directly from the existing JSON.

Also record exact SHA256 of each `result.json` and require they match P1D12:

```text
1010:
54c964933fca667ff9c55593439d3059c2f8ec18d30951d16a417eac880cc518

1003:
d2cef24ea3b15a528a8f2dec29faeb9fa1992bb9385633cdb2e419ea9039ad8e

1008:
bf70127106e5798b5be40a2aa317ac8921abe99b1e5015c9945a912e4657184c

1009:
ab75dd17e082c171e0b0f92eb9586329216bb32a1eafb317bcb2eab323e91507
```

If any SHA differs:
- do not rerun;
- outcome = EVIDENCE_FAILED;
- STOP.

# 6. Confirm no task execution occurred

Record explicitly:

```text
predict.py invocations = 0
--inspect-proposals invocations = 0
detector inference invocations = 0
model inference invocations = 0
diagnostics rewritten = NO
external delivery modified = NO
```

# 7. Outcome

## A — `PROP01_LOCKED_DEMO_PROPOSAL_GATE_EVIDENCE_CLOSED`

Require:

```text
FROZEN_THRESHOLD = 0.5
all frozen detector constants = MATCH
4/4 result.json SHA = unchanged
4/4 status = SUCCESS
4/4 mode = inspect
4/4 timings.sam2 = 0.0
4/4 timings.relation_fields = 0.0
4/4 timings.db1 = 0.0
no candidate/model rerun
external delivery unchanged
```

Then freeze:

```text
Task 8B.3-P1D12 = APPROVED
Proposal gate = PROP01_LOCKED_DEMO_PROPOSAL_GATE_PASS
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
NEXT = REF01_LOCKED_DEMO_REFERENCE_FORENSICS
```

Do not execute NEXT.

## B — `PROP01_LOCKED_DEMO_PROPOSAL_GATE_EVIDENCE_FAILED`

For any contradiction.

```text
NEXT = PROP01_LOCKED_DEMO_PROPOSAL_EVIDENCE_RECOVERY
```

Do not execute NEXT.

# 8. Report

Create:

```text
docs/task8b3_p1d12_r2_final_evidence_closure.md
```

Append a short R2 section to:

```text
docs/task8b3_p1d12_r1_evidence_closure.md
```

Required report table:

```text
candidate | result.json SHA match | status | mode | sam2 | relation_fields | db1
```

Also record:

```text
FROZEN_THRESHOLD
DETECTOR_CONSTANTS
no-rerun declaration
Outcome
NEXT
```

# 9. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D12-R2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 728932094a9f86f7c1607abc6a2beff12d2d2d70
Candidate rerun: NO
predict.py executed: NO
Detector/model inference executed: NO
External delivery modified: NO
FROZEN_THRESHOLD: 0.5 / other
Detector constants: MATCH / other
1010 result SHA: MATCH / other
1003 result SHA: MATCH / other
1008 result SHA: MATCH / other
1009 result SHA: MATCH / other
1010 sam2/relation/db1: 0.0/0.0/0.0 / other
1003 sam2/relation/db1: 0.0/0.0/0.0 / other
1008 sam2/relation/db1: 0.0/0.0/0.0 / other
1009 sam2/relation/db1: 0.0/0.0/0.0 / other
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Outcome: <exact enum>
Next gate: <exact enum>
Report: docs/task8b3_p1d12_r2_final_evidence_closure.md
Next action: Awaiting ChatGPT audit; do not run reference selection or full inference.
```

# 10. Commit / push

If COMPLETE:

```text
docs(rc1): finalize locked demo proposal evidence
```

If STOP/FAILED:

```text
docs(rc1): record final proposal evidence stop
```

Push normally.
No force push.

# 11. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no candidate/model rerun;
- no external modification;
- `FROZEN_THRESHOLD = 0.5`;
- all frozen detector constants match;
- all four existing `result.json` SHA values match P1D12;
- all four existing JSONs directly report `sam2/relation_fields/db1 = 0.0`;
- only allowed docs/handoff change;
- commit/push succeeds;
- STOP.
