请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3B1-R2 — Manifest Audit Artifact Closure**

# TO_DSH — Task 8B.3-REF01-E3B1-R2: Manifest Audit Artifact Closure

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required starting HEAD: `aa9da72ac840d04e5fe668a44935127dc4c1f1f0`

# 0. CHATGPT AUDIT DISPOSITION

E3B1-R1 technical manifest correction is ACCEPTED.

The following facts are frozen and MUST NOT be rerun or changed:

```text
Canonical implementation base:
f50404843f5189986f97633cd0edb6140b1d8034

Manifest identity basis:
GIT_CANONICAL_BLOB_BYTES

Manifest file count:
135

Detector Git-canonical identity:
bytes  = 21257
sha256 = 934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883

test_task8b_runtime.py Git-canonical identity:
bytes  = 28128
sha256 = 71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b

Manifest semantic correction:
PASS

Manifest Git-canonical validation:
PASS 135/135

External pre-sync --check:
exit = 1
checked = 135
match = 133
missing = 0
mismatch = 2

Mismatch paths EXACTLY:
buildreasonseg/runtime/detector.py
tests/test_task8b_runtime.py

External write:
NONE

External sync:
NONE

pytest / py_compile / inference:
NONE / NONE / NONE
```

Why the technical result is accepted:
- the sync helper validates `GIT_CANONICAL_BLOB_BYTES` using `git show HEAD:<manifest path>`;
- it checks both `len(payload)` and `sha256(payload)` against every manifest entry;
- R1 reported `MANIFEST_GIT_CANONICAL: PASS 135/135`;
- the read-only external check produced exactly 133 match / 0 missing / 2 mismatch.

E3B1-R1 is NOT formally closed only because the artifact contract was violated:

```text
Required evidence path:
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json

Actual R1 evidence path:
evaluation/task8b3_ref01_eligibility_repair_manifest_correction.json

Required dedicated report:
docs/task8b3_ref01_e3b1_manifest_canonicalization.md

Actual:
MISSING
```

R2 is artifact closure only.

# 1. EXECUTOR CONTRACT

DSH has ZERO technical discretion.

Allowed operations ONLY:
1. verify exact branch/head;
2. verify the manifest/product/sync-helper are unchanged from starting HEAD;
3. delete the wrongly named R1 evidence file;
4. create the exact required evidence at the exact path/schema below;
5. create the exact dedicated report at the exact path below;
6. append one authoritative R2 closure section to the existing implementation report;
7. update FROM_DSH;
8. make exactly one commit;
9. push once;
10. STOP.

Forbidden:
- NO manifest edit;
- NO detector.py edit;
- NO test edit;
- NO pipeline edit;
- NO sync-helper edit;
- NO external `--check` rerun;
- NO external sync/write;
- NO pytest;
- NO py_compile;
- NO detector/model inference;
- NO Qwen/SAM2/D-B1/target inference;
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
= aa9da72ac840d04e5fe668a44935127dc4c1f1f0
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No commit/push until §10.

# 3. IMMUTABILITY GATE

Require NO working-tree/index diff for:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

Require current manifest contains exactly:

```text
buildreasonseg/runtime/detector.py
bytes = 21257
sha256 = 934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883

tests/test_task8b_runtime.py
bytes = 28128
sha256 = 71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b

identity_basis = GIT_CANONICAL_BLOB_BYTES
file count = 135
```

This is READ-ONLY verification.

Any mismatch => STOP.

# 4. DELETE WRONG R1 EVIDENCE

Delete exactly:

```text
evaluation/task8b3_ref01_eligibility_repair_manifest_correction.json
```

using:

```text
git rm evaluation/task8b3_ref01_eligibility_repair_manifest_correction.json
```

Require success.

Do NOT delete any other evaluation file.

# 5. CREATE REQUIRED EVIDENCE AT EXACT PATH

Create exactly:

```text
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
```

with EXACTLY:

```json
{
  "task": "8B.3-REF01-E3B1-R2",
  "starting_head": "aa9da72ac840d04e5fe668a44935127dc4c1f1f0",
  "canonical_base": "f50404843f5189986f97633cd0edb6140b1d8034",
  "branch": "fix/task8b3-ref01-eligibility-repair-sync",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "scope": "MANIFEST_AUDIT_ARTIFACT_CLOSURE_PRE_SYNC",
  "detector_model_calls": 0,
  "manifest_schema": "BuildReasonSeg.AdvisorRC1.SourceManifest.v1",
  "identity_basis": "GIT_CANONICAL_BLOB_BYTES",
  "manifest_file_count": 135,
  "superseded_e3b1_error": "WINDOWS_WORKING_TREE_BYTE_COUNT_USED_WITH_GIT_CANONICAL_BASIS",
  "corrected_entries": {
    "buildreasonseg/runtime/detector.py": {
      "bytes": 21257,
      "sha256": "934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883"
    },
    "tests/test_task8b_runtime.py": {
      "bytes": 28128,
      "sha256": "71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b"
    }
  },
  "manifest_semantic_correction": "PASS",
  "manifest_git_canonical_validation": "PASS_135_OF_135",
  "external_presync_check_exit": 1,
  "external_presync_match": 133,
  "external_presync_missing": 0,
  "external_presync_mismatch": 2,
  "external_presync_mismatch_paths": [
    "buildreasonseg/runtime/detector.py",
    "tests/test_task8b_runtime.py"
  ],
  "external_write_performed": false,
  "source_sync_performed": false,
  "r1_wrong_evidence_path_removed": true,
  "dedicated_report_created": true,
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE"
}
```

Do not add/remove/rename keys.
Do not change fixed values.

# 6. CREATE REQUIRED DEDICATED REPORT

Create exactly:

```text
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
```

with this authoritative content:

```text
# Task 8B.3-REF01-E3B1 — Manifest Canonicalization Closure

Authoritative closure task = 8B.3-REF01-E3B1-R2
Status = COMPLETE

Canonical base =
f50404843f5189986f97633cd0edb6140b1d8034

Manifest identity basis =
GIT_CANONICAL_BLOB_BYTES

Manifest file count =
135

Original E3B1 error =
WINDOWS_WORKING_TREE_BYTE_COUNT_USED_WITH_GIT_CANONICAL_BASIS

Detector Git-canonical identity =
21257 bytes
934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883

test_task8b_runtime.py Git-canonical identity =
28128 bytes
71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b

Manifest semantic correction =
PASS

Manifest Git-canonical validation =
135/135 PASS

External pre-sync read-only comparison =
133 match / 0 missing / 2 mismatch

External mismatch paths =
buildreasonseg/runtime/detector.py
tests/test_task8b_runtime.py

External write =
NONE

External sync =
NONE

pytest / py_compile =
NONE / NONE

Detector/model inference =
NONE

E3B1-R1 technical result =
ACCEPTED

E3B1-R2 purpose =
ARTIFACT CLOSURE ONLY

Outcome =
REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED

NEXT =
REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

left/below reference-selection defects =
UNRESOLVED
```

Do not add technical claims outside these frozen facts.

# 7. EXISTING IMPLEMENTATION REPORT

Append exactly one new section to:

```text
docs/task8b3_ref01_eligibility_repair_impl.md
```

Title:

```text
## 14. E3B1-R2 — manifest audit artifact closure
```

Required content:

```text
E3B1-R1 technical correction = ACCEPTED
Authoritative evidence = evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
Authoritative dedicated report = docs/task8b3_ref01_e3b1_manifest_canonicalization.md

Git-canonical detector identity =
21257 bytes / 934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883

Git-canonical test identity =
28128 bytes / 71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b

Manifest validation =
135/135 PASS

External pre-sync =
133 match / 0 missing / 2 mismatch

External write/sync =
NONE / NONE

E3B1 manifest canonicalization =
CLOSED

NEXT =
REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
```

Do NOT rewrite historical E3B1/E3B1-R1 sections.

# 8. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Replace the active engineering handoff with:

```text
Task: 8B.3-REF01-E3B1-R2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-sync
Starting HEAD: aa9da72ac840d04e5fe668a44935127dc4c1f1f0
Canonical base: f50404843f5189986f97633cd0edb6140b1d8034
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Manifest changed in R2: NO
Product/test/sync-helper changed in R2: NO
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
External check rerun in R2: NO
External write performed: NO
External sync performed: NO
Wrong R1 evidence removed: YES
Evidence: evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
Report: docs/task8b3_ref01_e3b1_manifest_canonicalization.md
Outcome: REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED
Next gate: REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Next action: Awaiting ChatGPT audit; do not execute E3B2/NEXT.
```

# 9. NO-RERUN ASSERTION

Before final diff, confirm:

```text
pytest run in R2 = NO
py_compile run in R2 = NO
sync helper run in R2 = NO
external --check rerun in R2 = NO
detector/model inference in R2 = NO
external RC1 write in R2 = NO
```

Any one not NO => STOP.

# 10. FINAL DIFF GATE

Before commit:

```text
git diff --name-status aa9da72ac840d04e5fe668a44935127dc4c1f1f0
```

Allowed ONLY:

```text
D  evaluation/task8b3_ref01_eligibility_repair_manifest_correction.json
A  evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
A  docs/task8b3_ref01_e3b1_manifest_canonicalization.md
M  docs/task8b3_ref01_eligibility_repair_impl.md
M  handoff/FROM_DSH.md
M  handoff/TO_DSH.md
```

Require these paths do NOT appear:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

Also require:

```text
git rev-list --count aa9da72ac840d04e5fe668a44935127dc4c1f1f0..HEAD
= 0
```

Any mismatch => STOP.

# 11. STATUS → COMMIT MESSAGE

If §§2–10 all PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): close git-canonical manifest audit
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record manifest audit closure stop
```

Commit exactly once.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count aa9da72ac840d04e5fe668a44935127dc4c1f1f0..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute E3B2/NEXT.

Then STOP.

# 12. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- zero product/test/manifest/sync-helper changes;
- no helper/test/compile/inference rerun;
- wrong R1 evidence removed;
- exact evidence created at exact required path/schema;
- exact dedicated report created at exact required path;
- existing report updated only with authoritative closure section;
- FROM_DSH uses exact authoritative paths/facts;
- exactly one R2 commit with exact message;
- E3B2 not executed;
- STOP.
