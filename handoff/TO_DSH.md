# TO_DSH — Task 8B.3-M1B.3-R1: Post-run Integrity Evidence Only

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `ccaeb09c27744f9087557a6dace6e4262da5db9e`
> External delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Canonical RC1: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

Fill exactly one missing evidence item from M1B.3:

```text
post-run external manifest integrity
+
external/source_manifest.json byte-identity to canonical
```

Do NOT rerun B1 or B2.
Do NOT run pytest.
Do NOT run any model.

The real memory gate evidence is already frozen:

```text
B1: 5000×5000, 1 invocation, SUCCESS, tiles=169, raw=6578, merged=3066,
    old allocation signature absent, E502 absent
B2: 5000×5000, 1 invocation, SUCCESS, tiles=169, raw=7864, merged=3740,
    old allocation signature absent, E502 absent
MEM01_REAL_GATE_PASS
RC1-DEMO-MEM-01 = CLOSED subject to post-run integrity confirmation
```

# 1. Strict prohibitions

Do NOT:
- run B1 or B2 again;
- run A1–A4;
- run pytest;
- run check_setup.py;
- run predict.py;
- run any detector/Qwen/SAM2/D-B1 model;
- modify canonical product/tests/manifest;
- modify external product/tests/manifest;
- modify model assets;
- sync files;
- delete outputs/caches;
- fix PROP-01 / REF-01 / MASK-01;
- enter Task 8B.4 / 8C;
- update main;
- force push.

# 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = ccaeb09c27744f9087557a6dace6e4262da5db9e
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_m1b3_large_image_memory_gate.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No functional file may change.

# 4. Read-only external manifest verification

Read:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json
```

Require:
- schema is `BuildReasonSeg.AdvisorRC1.SourceManifest.v1`;
- exactly 135 entries;
- every listed external path exists;
- every listed path byte size matches;
- every listed path SHA256 matches.

Record exactly:

```text
Post-run external manifest: 135/135 PASS
```

If any mismatch:
- record exact path;
- do not repair;
- mark STOP;
- MEM-01 formal closure is withheld pending audit.

# 5. source_manifest byte identity

Compare bytes of:

```text
external:
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json

canonical:
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1\source_manifest.json
```

Require byte-identical.

Record:

```text
Post-run source_manifest byte-identical: YES
```

If NO:
- do not repair;
- STOP.

# 6. Normalize M1B.3 report

Append a section to:

```text
docs/task8b3_m1b3_large_image_memory_gate.md
```

Title:

```text
## 12. Post-run integrity confirmation (M1B.3-R1)
```

Record:
- no inference/tests/model execution in R1;
- external manifest = 135/135 PASS;
- source_manifest byte-identical = YES;
- no functional file modified;
- therefore original M1B.3 runtime evidence + integrity evidence jointly satisfy the closure contract.

If both checks pass, freeze:

```text
MEM01_REAL_GATE_PASS
RC1-DEMO-MEM-01 = CLOSED
M1B.3 formal gate = PASS
```

# 7. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-M1B.3-R1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: ccaeb09c27744f9087557a6dace6e4262da5db9e
Inference/tests/model execution: NONE
Post-run external manifest: 135/135 PASS / FAIL
Post-run source_manifest byte-identical: YES / NO
Functional files modified: NO
Historical B1 invocation count: 1
Historical B2 invocation count: 1
Historical B1/B2 proposal-memory signature: ABSENT / ABSENT
MEM gate verdict: MEM01_REAL_GATE_PASS / other
RC1-DEMO-MEM-01: CLOSED / OPEN
M1B.3 formal gate: PASS / FAIL
PROP-01 / REF-01 / MASK-01: UNCHANGED / UNCHANGED / UNCHANGED
Report: docs/task8b3_m1b3_large_image_memory_gate.md
Next action: Awaiting ChatGPT audit before main integration.
```

# 8. Git / commit / push

Verify only §3 paths changed.

If COMPLETE:

```text
docs(rc1): confirm post-run mem01 integrity
```

If STOP/FAILED:

```text
docs(rc1): record mem01 integrity stop
```

Push current fix branch normally, no force.

# 9. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no inference/test/model execution;
- external manifest 135/135 PASS;
- external source_manifest byte-identical to canonical;
- no functional modifications;
- M1B.3 report normalized;
- FROM_DSH complete;
- commit/push succeed;
- clean tracked tree;
- stop.

# 10. Final response

```text
TASK 8B.3-M1B.3-R1 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Inference/tests/model execution:
NONE

Post-run external manifest:
135/135 PASS / other

Post-run source_manifest byte-identical:
YES / NO

Functional files modified:
NO

Historical B1/B2 invocations:
1 / 1

MEM gate verdict:
MEM01_REAL_GATE_PASS / other

RC1-DEMO-MEM-01:
CLOSED / OPEN

M1B.3 formal gate:
PASS / FAIL

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得重跑 B1/B2、不得更新 main、不得进入 PROP-01/REF-01/MASK-01/Task 8B.4/8C。
```
