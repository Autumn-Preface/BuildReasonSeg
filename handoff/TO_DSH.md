请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3B1-R3 — Handoff Contract Normalization**

# TO_DSH — Task 8B.3-REF01-E3B1-R3: Handoff Contract Normalization

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required starting HEAD: `3004cb7e65c73a9cdc005aaa5978876cfed263db`

# 0. CHATGPT AUDIT DISPOSITION

E3B1-R2 technical and artifact results are ACCEPTED.

Frozen accepted facts:

```text
Canonical base:
f50404843f5189986f97633cd0edb6140b1d8034

Design:
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

Manifest identity basis:
GIT_CANONICAL_BLOB_BYTES

Manifest file count:
135

Detector Git-canonical identity:
21257 bytes
934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883

test_task8b_runtime.py Git-canonical identity:
28128 bytes
71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b

Manifest semantic correction:
PASS

Manifest identities:
PASS 135/135

External pre-sync:
133 match / 0 missing / 2 mismatch

Mismatch paths:
buildreasonseg/runtime/detector.py
tests/test_task8b_runtime.py

External write:
NO

External sync:
NO

Detector/model calls:
0

Authoritative evidence:
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json

Authoritative report:
docs/task8b3_ref01_e3b1_manifest_canonicalization.md

Outcome:
REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED

Next gate:
REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
```

R2 formal closure is pending ONLY because `handoff/FROM_DSH.md` did not explicitly include every required field.

R3 is handoff-only normalization.

# 1. EXECUTOR CONTRACT

Allowed operations ONLY:
1. verify branch/head;
2. verify authoritative evidence/report exist;
3. verify protected product/manifest/sync files have no working-tree/index changes;
4. replace only the active engineering handoff section of `handoff/FROM_DSH.md` with the exact field set in §5 while preserving ARTIFACT-FACTS exactly;
5. leave this taskbook as `handoff/TO_DSH.md`;
6. make exactly one commit;
7. push once;
8. STOP.

Forbidden:
- NO manifest edit;
- NO detector/test/pipeline edit;
- NO sync-helper edit;
- NO evidence edit;
- NO report edit;
- NO helper run;
- NO external `--check`;
- NO external sync/write;
- NO pytest;
- NO py_compile;
- NO detector/model inference;
- NO rebase/reset/amend/stash/clean;
- NO force push;
- NO intermediate commit/push;
- NO E3B2/NEXT.

Any uncovered condition => STOP.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-sync

git rev-parse HEAD
= 3004cb7e65c73a9cdc005aaa5978876cfed263db
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No commit/push until §7.

# 3. AUTHORITATIVE ARTIFACT PRESENCE GATE

Require both files exist:

```text
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
```

Require the obsolete path does NOT exist:

```text
evaluation/task8b3_ref01_eligibility_repair_manifest_correction.json
```

Do not modify any of these files.

# 4. PROTECTED-FILE IMMUTABILITY GATE

Require no working-tree/index changes for:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
docs/task8b3_ref01_eligibility_repair_impl.md
```

Any change => STOP.

# 5. EXACT FROM_DSH ACTIVE HANDOFF

Preserve the complete block between:

```text
<!-- ARTIFACT-FACTS:BEGIN -->
...
<!-- ARTIFACT-FACTS:END -->
```

byte-for-byte.

Replace ONLY the active engineering handoff below that block with content containing ALL of the following fields and values:

```text
Task: 8B.3-REF01-E3B1-R3
Status: COMPLETE
Branch: fix/task8b3-ref01-eligibility-repair-sync
Starting HEAD: 3004cb7e65c73a9cdc005aaa5978876cfed263db
Canonical base: f50404843f5189986f97633cd0edb6140b1d8034
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Manifest changed in R3: NO
Product/test/sync-helper changed in R3: NO
Identity basis: GIT_CANONICAL_BLOB_BYTES
Detector Git-canonical bytes: 21257
Detector Git-canonical SHA256: 934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883
Test Git-canonical bytes: 28128
Test Git-canonical SHA256: 71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b
Manifest file count: 135
Manifest semantic correction: PASS
Manifest identities: PASS 135/135
External pre-sync check: 133 match / 0 missing / 2 mismatch
External mismatch paths: buildreasonseg/runtime/detector.py; tests/test_task8b_runtime.py
External check rerun in R3: NO
External write performed: NO
External sync performed: NO
pytest run in R3: NO
py_compile run in R3: NO
Evidence: evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
Report: docs/task8b3_ref01_e3b1_manifest_canonicalization.md
Outcome: REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED
Next gate: REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects: UNRESOLVED
Next action: Awaiting ChatGPT audit; do not execute E3B2/NEXT.
```

Formatting may be Markdown table or plain text, but every field label and value above must appear verbatim.

Do not add a contradictory active status.

# 6. NO-RERUN ASSERTION

Before commit confirm:

```text
helper run in R3 = NO
external --check run in R3 = NO
external sync/write in R3 = NO
pytest run in R3 = NO
py_compile run in R3 = NO
detector/model inference in R3 = NO
```

Any non-NO => STOP.

# 7. FINAL DIFF GATE

Run:

```text
git diff --name-only 3004cb7e65c73a9cdc005aaa5978876cfed263db
```

Allowed ONLY:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path.

Also require:

```text
git rev-list --count 3004cb7e65c73a9cdc005aaa5978876cfed263db..HEAD
= 0
```

# 8. STATUS → COMMIT MESSAGE

If §§2–7 all PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): normalize manifest audit handoff
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record manifest handoff normalization stop
```

Commit exactly once.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count 3004cb7e65c73a9cdc005aaa5978876cfed263db..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute E3B2/NEXT.

Then STOP.

# 9. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- authoritative evidence/report untouched;
- all protected files untouched;
- ARTIFACT-FACTS preserved exactly;
- every required handoff field appears verbatim;
- only FROM_DSH and TO_DSH changed;
- no technical command rerun;
- exactly one commit with exact message;
- E3B2 not executed;
- STOP.
