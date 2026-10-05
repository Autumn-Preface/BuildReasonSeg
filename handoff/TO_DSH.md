请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3B1 — Canonical Source Manifest Canonicalization**

# TO_DSH — Task 8B.3-REF01-E3B1: Canonical Source Manifest Canonicalization

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required base branch: `fix/task8b3-ref01-eligibility-repair-impl`
> Required base HEAD: `f50404843f5189986f97633cd0edb6140b1d8034`
> New task branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. CHATGPT DECISION

E3A canonical implementation is formally CLOSED.

Approved implementation:

```text
Design:
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

Canonical implementation:
CLOSED

Accepted targeted regression:
40 passed in 0.69s

Canonical full delivery suite:
DEFERRED until external RC1 controlled sync

Current next gate:
REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
```

E3B is split into two separate tasks.

This task, E3B1, performs ONLY:

```text
canonical source_manifest identity update
+
read-only validation
+
read-only external pre-sync delta check
```

It MUST NOT sync or write external RC1.

E3B2 will perform the controlled external sync and the full external delivery suite only after ChatGPT audits E3B1.

# 1. EXECUTOR CONTRACT

DSH has NO technical design discretion.

Allowed operations only:
1. verify exact base branch/HEAD;
2. create the exact new branch;
3. update exactly two existing entries in canonical `source_manifest.json`;
4. verify all 135 manifest identities against Git canonical HEAD bytes;
5. run the sync helper in `--check` mode only against external RC1;
6. require the external pre-sync delta to be exactly two mismatches;
7. create exact evidence/report/FROM_DSH;
8. make exactly one commit;
9. push the new branch once;
10. STOP.

Forbidden:
- NO edit to detector.py;
- NO edit to test_task8b_runtime.py;
- NO edit to any other canonical source/config/test file;
- NO external write;
- NO sync without `--check`;
- NO copy to external;
- NO source_manifest copy to external;
- NO pytest;
- NO py_compile;
- NO detector/model inference;
- NO Qwen/SAM2/D-B1/target inference;
- NO rebase/reset/amend/stash/clean;
- NO force push;
- NO intermediate commit/push;
- NO NEXT/E3B2.

Any unexpected condition => STOP.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-impl

git rev-parse HEAD
= f50404843f5189986f97633cd0edb6140b1d8034
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

Create exactly:

```text
fix/task8b3-ref01-eligibility-repair-sync
```

After creation require:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-sync

git rev-parse HEAD
= f50404843f5189986f97633cd0edb6140b1d8034
```

No other branch operation.

# 3. FROZEN MANIFEST BASIS

Canonical manifest:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Current manifest contract is frozen:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
manifest file count = 135
```

Do NOT alter:
- schema;
- task;
- source_delivery;
- canonical_root;
- copy_policy;
- identity_basis;
- identity_basis_note;
- path order;
- file count.

Only two `bytes` / `sha256` identity pairs may change.

# 4. EXACT OLD → NEW IDENTITY REPLACEMENTS

## 4.1 detector.py

Path:

```text
buildreasonseg/runtime/detector.py
```

Require CURRENT manifest entry exactly:

```json
{
  "path": "buildreasonseg/runtime/detector.py",
  "bytes": 20300,
  "sha256": "82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738"
}
```

Replace ONLY the identity values with:

```json
{
  "path": "buildreasonseg/runtime/detector.py",
  "bytes": 21257,
  "sha256": "bc5aed885aa5f27b715de0ab53bf4abdcc55f5d8b930fb4076607a3071ec8ed3"
}
```

## 4.2 test_task8b_runtime.py

Path:

```text
tests/test_task8b_runtime.py
```

Require CURRENT manifest entry exactly:

```json
{
  "path": "tests/test_task8b_runtime.py",
  "bytes": 24983,
  "sha256": "6cac7e9320f2088835c51efce335b0d0391fd68884d853be63b47b4343cafecd"
}
```

Replace ONLY the identity values with:

```json
{
  "path": "tests/test_task8b_runtime.py",
  "bytes": 28128,
  "sha256": "8071fcc6a10e5f299b58f692667508d47c3291b780442b661c0060f666b5ee7d"
}
```

No other manifest entry may change.

# 5. EXACT MANIFEST UPDATE COMMAND

Use the REQUIRED_PYTHON directly.

Run exactly this single command from repository root:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import json,pathlib; p=pathlib.Path(r'delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json'); x=json.loads(p.read_text(encoding='utf-8')); assert x['schema']=='BuildReasonSeg.AdvisorRC1.SourceManifest.v1'; assert x['identity_basis']=='GIT_CANONICAL_BLOB_BYTES'; assert len(x['files'])==135; m={e['path']:e for e in x['files']}; d=m['buildreasonseg/runtime/detector.py']; t=m['tests/test_task8b_runtime.py']; assert (d['bytes'],d['sha256'])==(20300,'82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738'); assert (t['bytes'],t['sha256'])==(24983,'6cac7e9320f2088835c51efce335b0d0391fd68884d853be63b47b4343cafecd'); d['bytes']=21257; d['sha256']='bc5aed885aa5f27b715de0ab53bf4abdcc55f5d8b930fb4076607a3071ec8ed3'; t['bytes']=28128; t['sha256']='8071fcc6a10e5f299b58f692667508d47c3291b780442b661c0060f666b5ee7d'; p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')"
```

Require exit 0.

DO NOT substitute another edit method.
DO NOT edit the JSON manually.
DO NOT run the command twice.

If the command fails => STOP.

# 6. MANIFEST DIFF GATE

Run:

```text
git diff -- delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Require the semantic diff contains ONLY:

```text
detector bytes:
20300 -> 21257

detector sha256:
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
->
bc5aed885aa5f27b715de0ab53bf4abdcc55f5d8b930fb4076607a3071ec8ed3

test_task8b_runtime bytes:
24983 -> 28128

test_task8b_runtime sha256:
6cac7e9320f2088835c51efce335b0d0391fd68884d853be63b47b4343cafecd
->
8071fcc6a10e5f299b58f692667508d47c3291b780442b661c0060f666b5ee7d
```

No path/key/order/schema/policy/count change is allowed.

Any extra semantic diff => STOP.

# 7. 135/135 GIT-CANONICAL MANIFEST VALIDATION

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import scripts.sync_advisor_rc1_delivery as s; e=s.load_manifest(); assert len(e)==135; b=s.manifest_identity_basis(); assert b==s.GIT_CANONICAL_BASIS; [s.entry_source_bytes(x,b,s.CANONICAL_ROOT) for x in e]; print('MANIFEST_GIT_CANONICAL: PASS 135/135')"
```

Require:
- exit 0;
- stdout contains exactly:

```text
MANIFEST_GIT_CANONICAL: PASS 135/135
```

This validates manifest identities against the existing Git HEAD canonical blobs.

No file write is permitted by this command.

# 8. READ-ONLY EXTERNAL PRE-SYNC DELTA CHECK

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe scripts\sync_advisor_rc1_delivery.py --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 --check
```

This is read-only.

Expected exit code:

```text
1
```

Expected mismatches EXACTLY:

```text
MISMATCH buildreasonseg/runtime/detector.py
MISMATCH tests/test_task8b_runtime.py
```

No other `MISMATCH`.
No `MISSING`.

Expected final summary exactly:

```text
checked=135 match=133 missing=0 mismatch=2
```

If:
- exit is 0;
- mismatch count is not 2;
- any missing file exists;
- any third mismatch exists;

then STOP.

DO NOT run the helper without `--check`.

# 9. EVIDENCE

Create exactly:

```text
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
```

with exactly:

```json
{
  "task": "8B.3-REF01-E3B1",
  "starting_head": "f50404843f5189986f97633cd0edb6140b1d8034",
  "branch": "fix/task8b3-ref01-eligibility-repair-sync",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "scope": "CANONICAL_MANIFEST_ONLY_PRE_SYNC",
  "detector_model_calls": 0,
  "manifest_schema": "BuildReasonSeg.AdvisorRC1.SourceManifest.v1",
  "identity_basis": "GIT_CANONICAL_BLOB_BYTES",
  "manifest_file_count": 135,
  "updated_entries": {
    "buildreasonseg/runtime/detector.py": {
      "bytes": 21257,
      "sha256": "bc5aed885aa5f27b715de0ab53bf4abdcc55f5d8b930fb4076607a3071ec8ed3"
    },
    "tests/test_task8b_runtime.py": {
      "bytes": 28128,
      "sha256": "8071fcc6a10e5f299b58f692667508d47c3291b780442b661c0060f666b5ee7d"
    }
  },
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
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE"
}
```

Do not add/remove/rename keys.

# 10. REPORT

Create exactly:

```text
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
```

Required conclusions:

```text
Task = 8B.3-REF01-E3B1
Status = COMPLETE
Canonical implementation base = f50404843f5189986f97633cd0edb6140b1d8034
Manifest identity basis = GIT_CANONICAL_BLOB_BYTES
Manifest files = 135
Manifest identities validated = 135/135 PASS
Updated manifest entries = detector.py + test_task8b_runtime.py ONLY
External pre-sync comparison = 133 match / 0 missing / 2 mismatch
External mismatches = detector.py + test_task8b_runtime.py ONLY
External write = NONE
External sync = NONE
Detector/model inference = NONE
Canonical full suite = NOT RUN
Outcome = REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED
NEXT = REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects = UNRESOLVED
```

# 11. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Active fields:

```text
Task: 8B.3-REF01-E3B1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-sync
Starting HEAD: f50404843f5189986f97633cd0edb6140b1d8034
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Manifest schema: BuildReasonSeg.AdvisorRC1.SourceManifest.v1
Identity basis: GIT_CANONICAL_BLOB_BYTES
Manifest file count: 135
Manifest identities: PASS 135/135
Updated manifest entries: buildreasonseg/runtime/detector.py; tests/test_task8b_runtime.py
External pre-sync check: 133 match / 0 missing / 2 mismatch
External mismatch paths: buildreasonseg/runtime/detector.py; tests/test_task8b_runtime.py
External write performed: NO
External sync performed: NO
Outcome: REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED
Next gate: REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Next action: Awaiting ChatGPT audit; do not execute E3B2/NEXT.
```

# 12. FINAL DIFF GATE

Before commit run:

```text
git diff --name-only f50404843f5189986f97633cd0edb6140b1d8034
```

Allowed ONLY:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Require:
- detector.py does NOT appear;
- test_task8b_runtime.py does NOT appear;
- pipeline.py does NOT appear;
- sync helper does NOT appear.

Also require:

```text
git rev-list --count f50404843f5189986f97633cd0edb6140b1d8034..HEAD
= 0
```

Any mismatch => STOP.

# 13. STATUS → COMMIT MESSAGE

If every gate PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): canonicalize extent repair manifest
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record extent repair manifest stop
```

Commit exactly once.
NO amend.

After commit require:

```text
git rev-list --count f50404843f5189986f97633cd0edb6140b1d8034..HEAD
= 1
```

Push only:

```text
fix/task8b3-ref01-eligibility-repair-sync
```

exactly once.

No force push.
Do not update main.
Do not execute E3B2/NEXT.

Then STOP.

# 14. COMPLETE DEFINITION

COMPLETE only if:
- exact base branch/head;
- exact new branch;
- only two manifest identity entries updated;
- all 135 manifest identities validate against Git canonical HEAD;
- read-only external pre-sync delta is exactly 133/0/2;
- the only two mismatches are detector.py and test_task8b_runtime.py;
- no external write/sync;
- no pytest/compile/inference;
- exact evidence/report/FROM_DSH;
- exactly one commit with exact message;
- push new branch;
- E3B2 not executed;
- STOP.
