请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3B1-R1 — Correct Git-Canonical Manifest Identities**

# TO_DSH — Task 8B.3-REF01-E3B1-R1: Correct Git-Canonical Manifest Identities

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required starting HEAD: `8fdaf9c973af2b0b437d8892894e910c79cf2058`
> Git-canonical implementation base: `f50404843f5189986f97633cd0edb6140b1d8034`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. CHATGPT AUDIT OF E3B1

E3B1 is **NOT APPROVED**.

The branch/parent/commit-message discipline was correct, but the manifest canonicalization was not.

Observed E3B1 defects:

```text
manifest identity_basis =
GIT_CANONICAL_BLOB_BYTES

GitHub Git-blob sizes at canonical implementation base f5040484... =
buildreasonseg/runtime/detector.py = 21257 bytes
tests/test_task8b_runtime.py       = 28128 bytes

E3B1 manifest values =
buildreasonseg/runtime/detector.py = 21756 bytes
tests/test_task8b_runtime.py       = 28761 bytes
```

The E3B1 byte counts are consistent with Windows working-tree line-ending expansion and are NOT acceptable for a manifest whose identity basis is `GIT_CANONICAL_BLOB_BYTES`.

E3B1 also failed the artifact contract:
- required `evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json` is absent;
- required `docs/task8b3_ref01_e3b1_manifest_canonicalization.md` is absent;
- the existing E3B1 section in `docs/task8b3_ref01_eligibility_repair_impl.md` contains the superseded working-tree byte counts.

R1 corrects these issues deterministically.

# 1. EXECUTOR CONTRACT

DSH has ZERO technical discretion.

DSH MUST NOT choose byte counts or SHA256 values.

The two identities MUST be computed from exact Git object bytes using:

```text
git show f50404843f5189986f97633cd0edb6140b1d8034:delivery_src/BuildReasonSeg_Advisor_RC1/<relative path>
```

The following is forbidden:
- working-tree `Path.read_bytes()` as the canonical identity source;
- `os.path.getsize()` on the Windows working tree;
- CRLF-expanded bytes as manifest identity;
- editing detector.py or tests;
- external write/sync;
- pytest or py_compile;
- detector/model inference;
- self-selected validation logic;
- amend/rebase/reset/force push;
- intermediate commit/push;
- executing E3B2/NEXT.

Any uncovered condition => STOP.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-sync

git rev-parse HEAD
= 8fdaf9c973af2b0b437d8892894e910c79cf2058
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

Do not commit/push until §13.

# 3. PRODUCT IMMUTABILITY GATE

Require NO diff between canonical implementation base:

```text
f50404843f5189986f97633cd0edb6140b1d8034
```

and current HEAD for:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

Any difference => STOP.

# 4. CURRENT BAD-MANIFEST PRECONDITION

Open:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
len(files) = 135
```

Require current entries exactly:

```text
buildreasonseg/runtime/detector.py
bytes = 21756
sha256 = bc5aed885aa5f27b715de0ab53bf4abdcc55f5d8b930fb4076607a3071ec8ed3

tests/test_task8b_runtime.py
bytes = 28761
sha256 = 8071fcc6a10e5f299b58f692667508d47c3291b780442b661c0060f666b5ee7d
```

Any mismatch => STOP.

# 5. EXACT GIT-CANONICAL IDENTITY CORRECTION COMMAND

From repository root, run exactly ONCE:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import json,hashlib,subprocess,pathlib; B='f50404843f5189986f97633cd0edb6140b1d8034'; R='delivery_src/BuildReasonSeg_Advisor_RC1/'; g=lambda r: subprocess.run(['git','show',B+':'+R+r],check=True,stdout=subprocess.PIPE).stdout; db=g('buildreasonseg/runtime/detector.py'); tb=g('tests/test_task8b_runtime.py'); assert len(db)==21257,(len(db),'detector'); assert len(tb)==28128,(len(tb),'test'); ds=hashlib.sha256(db).hexdigest(); ts=hashlib.sha256(tb).hexdigest(); p=pathlib.Path(R+'source_manifest.json'); x=json.loads(p.read_text(encoding='utf-8')); assert x['schema']=='BuildReasonSeg.AdvisorRC1.SourceManifest.v1'; assert x['identity_basis']=='GIT_CANONICAL_BLOB_BYTES'; assert len(x['files'])==135; m={e['path']:e for e in x['files']}; d=m['buildreasonseg/runtime/detector.py']; t=m['tests/test_task8b_runtime.py']; assert (d['bytes'],d['sha256'])==(21756,'bc5aed885aa5f27b715de0ab53bf4abdcc55f5d8b930fb4076607a3071ec8ed3'); assert (t['bytes'],t['sha256'])==(28761,'8071fcc6a10e5f299b58f692667508d47c3291b780442b661c0060f666b5ee7d'); d['bytes']=len(db); d['sha256']=ds; t['bytes']=len(tb); t['sha256']=ts; p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode('utf-8')); print('DETECTOR_GIT_BYTES=',len(db)); print('DETECTOR_GIT_SHA256=',ds); print('TEST_GIT_BYTES=',len(tb)); print('TEST_GIT_SHA256=',ts)"
```

Require exit 0.

Required stdout numeric values:

```text
DETECTOR_GIT_BYTES= 21257
TEST_GIT_BYTES= 28128
```

The SHA256 strings are factual outputs derived by the fixed command.
DSH MUST copy them verbatim into evidence/report/FROM_DSH.
DSH MUST NOT substitute working-tree hashes.

Do NOT run this command twice.

# 6. SEMANTIC MANIFEST CORRECTION GATE

Run this exact validation command:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import json,hashlib,subprocess,pathlib; H='8fdaf9c973af2b0b437d8892894e910c79cf2058'; B='f50404843f5189986f97633cd0edb6140b1d8034'; R='delivery_src/BuildReasonSeg_Advisor_RC1/'; old=json.loads(subprocess.run(['git','show',H+':'+R+'source_manifest.json'],check=True,stdout=subprocess.PIPE).stdout.decode('utf-8')); cur=json.loads(pathlib.Path(R+'source_manifest.json').read_text(encoding='utf-8')); assert old['schema']==cur['schema']; assert old['identity_basis']==cur['identity_basis']=='GIT_CANONICAL_BLOB_BYTES'; assert len(old['files'])==len(cur['files'])==135; om={e['path']:e for e in old['files']}; cm={e['path']:e for e in cur['files']}; assert list(om)==list(cm); changed=[]; [(changed.append(p) if om[p]!=cm[p] else None) for p in om]; assert changed==['buildreasonseg/runtime/detector.py','tests/test_task8b_runtime.py'],changed; db=subprocess.run(['git','show',B+':'+R+'buildreasonseg/runtime/detector.py'],check=True,stdout=subprocess.PIPE).stdout; tb=subprocess.run(['git','show',B+':'+R+'tests/test_task8b_runtime.py'],check=True,stdout=subprocess.PIPE).stdout; assert cm['buildreasonseg/runtime/detector.py']=={'path':'buildreasonseg/runtime/detector.py','bytes':len(db),'sha256':hashlib.sha256(db).hexdigest()}; assert cm['tests/test_task8b_runtime.py']=={'path':'tests/test_task8b_runtime.py','bytes':len(tb),'sha256':hashlib.sha256(tb).hexdigest()}; print('MANIFEST_SEMANTIC_CORRECTION: PASS'); print('CHANGED_ENTRIES: detector.py; test_task8b_runtime.py')"
```

Require exit 0 and:

```text
MANIFEST_SEMANTIC_CORRECTION: PASS
CHANGED_ENTRIES: detector.py; test_task8b_runtime.py
```

# 7. TRUE 135/135 GIT-CANONICAL VALIDATION

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import scripts.sync_advisor_rc1_delivery as s; e=s.load_manifest(); assert len(e)==135; b=s.manifest_identity_basis(); assert b==s.GIT_CANONICAL_BASIS; [s.entry_source_bytes(x,b,s.CANONICAL_ROOT) for x in e]; print('MANIFEST_GIT_CANONICAL: PASS 135/135')"
```

Require exit 0 and exactly:

```text
MANIFEST_GIT_CANONICAL: PASS 135/135
```

If this fails:
- STOP;
- do not change manifest again;
- do not run external check.

# 8. READ-ONLY EXTERNAL PRE-SYNC CHECK

Only after §7 PASS, run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe scripts\sync_advisor_rc1_delivery.py --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 --check
```

Expected exit:

```text
1
```

Require exactly these two mismatch paths:

```text
MISMATCH buildreasonseg/runtime/detector.py
MISMATCH tests/test_task8b_runtime.py
```

Require:
- no third MISMATCH;
- no MISSING.

Final summary exactly:

```text
checked=135 match=133 missing=0 mismatch=2
```

Anything else => STOP.

This command is READ-ONLY.
Do NOT run the helper without `--check`.

# 9. EVIDENCE — REQUIRED PATH

Create exactly:

```text
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
```

Use the exact Git-canonical SHA256 strings printed by §5.

Required schema:

```json
{
  "task": "8B.3-REF01-E3B1-R1",
  "starting_head": "8fdaf9c973af2b0b437d8892894e910c79cf2058",
  "canonical_base": "f50404843f5189986f97633cd0edb6140b1d8034",
  "branch": "fix/task8b3-ref01-eligibility-repair-sync",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "scope": "CORRECT_GIT_CANONICAL_MANIFEST_IDENTITIES_PRE_SYNC",
  "detector_model_calls": 0,
  "manifest_schema": "BuildReasonSeg.AdvisorRC1.SourceManifest.v1",
  "identity_basis": "GIT_CANONICAL_BLOB_BYTES",
  "manifest_file_count": 135,
  "superseded_e3b1_identity_basis_error": "WINDOWS_WORKING_TREE_BYTE_COUNT_USED_WITH_GIT_CANONICAL_BASIS",
  "corrected_entries": {
    "buildreasonseg/runtime/detector.py": {
      "bytes": 21257,
      "sha256": "<EXACT DETECTOR_GIT_SHA256 FROM SECTION 5>"
    },
    "tests/test_task8b_runtime.py": {
      "bytes": 28128,
      "sha256": "<EXACT TEST_GIT_SHA256 FROM SECTION 5>"
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
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE"
}
```

Only the two SHA256 placeholder values may vary, and only from §5 stdout.

No other key/value may vary.

# 10. DEDICATED REPORT — REQUIRED PATH

Create exactly:

```text
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
```

Required conclusions:

```text
Task = 8B.3-REF01-E3B1-R1
Status = COMPLETE
Original E3B1 = NOT APPROVED
Error = Windows working-tree byte count was mixed with GIT_CANONICAL_BLOB_BYTES
Canonical base = f50404843f5189986f97633cd0edb6140b1d8034
Manifest identity basis = GIT_CANONICAL_BLOB_BYTES
Manifest files = 135
Detector Git-canonical bytes = 21257
Detector Git-canonical SHA256 = <exact §5 output>
Test Git-canonical bytes = 28128
Test Git-canonical SHA256 = <exact §5 output>
Changed manifest entries = detector.py + test_task8b_runtime.py ONLY
Manifest semantic correction = PASS
Manifest Git-canonical validation = 135/135 PASS
External pre-sync comparison = 133 match / 0 missing / 2 mismatch
External mismatch paths = detector.py + test_task8b_runtime.py ONLY
External write = NONE
External sync = NONE
pytest / py_compile = NONE / NONE
Detector/model inference = NONE
Outcome = REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED
NEXT = REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects = UNRESOLVED
```

# 11. SUPERSEDE THE BAD E3B1 SECTION IN EXISTING REPORT

Append exactly one section to:

```text
docs/task8b3_ref01_eligibility_repair_impl.md
```

Title:

```text
## 13. E3B1-R1 — Git-canonical identity correction
```

It must state:

```text
The E3B1 byte counts 21756 / 28761 were Windows working-tree counts and are superseded.

Authoritative identity basis:
GIT_CANONICAL_BLOB_BYTES

Authoritative byte counts:
detector.py = 21257
test_task8b_runtime.py = 28128

Authoritative SHA256:
use the exact §5 Git-object-derived values.

E3B1-R1 is authoritative for manifest canonicalization.
```

Do NOT delete or rewrite the old historical E3B1 section.

# 12. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Active fields must include:

```text
Task: 8B.3-REF01-E3B1-R1
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-sync
Starting HEAD: 8fdaf9c973af2b0b437d8892894e910c79cf2058
Canonical base: f50404843f5189986f97633cd0edb6140b1d8034
Design selected by: CHATGPT
DSH algorithm choice performed: NO
Detector/model calls: 0
Identity basis: GIT_CANONICAL_BLOB_BYTES
Detector Git-canonical bytes: 21257
Detector Git-canonical SHA256: <exact §5 output>
Test Git-canonical bytes: 28128
Test Git-canonical SHA256: <exact §5 output>
Manifest file count: 135
Manifest semantic correction: PASS
Manifest identities: PASS 135/135
External pre-sync check: 133 match / 0 missing / 2 mismatch
External mismatch paths: buildreasonseg/runtime/detector.py; tests/test_task8b_runtime.py
External write performed: NO
External sync performed: NO
Evidence: evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
Report: docs/task8b3_ref01_e3b1_manifest_canonicalization.md
Outcome: REF01_ELIGIBILITY_REPAIR_MANIFEST_CANONICALIZED
Next gate: REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_AND_FULL_SUITE
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Next action: Awaiting ChatGPT audit; do not execute E3B2/NEXT.
```

# 13. FINAL DIFF GATE

Before commit:

```text
git diff --name-only 8fdaf9c973af2b0b437d8892894e910c79cf2058
```

Allowed ONLY:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
docs/task8b3_ref01_eligibility_repair_impl.md
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
git rev-list --count 8fdaf9c973af2b0b437d8892894e910c79cf2058..HEAD
= 0
```

Anything else => STOP.

# 14. STATUS → COMMIT MESSAGE

If §§2–13 all PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): correct git-canonical extent manifest
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record git-canonical manifest correction stop
```

Commit exactly once.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count 8fdaf9c973af2b0b437d8892894e910c79cf2058..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute E3B2/NEXT.

Then STOP.

# 15. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- product/test/sync-helper unchanged;
- identities derived ONLY from Git object bytes at f5040484...;
- exact Git-canonical sizes are 21257 / 28128;
- manifest semantic diff affects only the two required entries;
- true helper-based 135/135 validation passes;
- external `--check` is exactly 133/0/2 and read-only;
- required evidence path exists;
- required dedicated report path exists;
- old E3B1 report is superseded, not rewritten;
- FROM_DSH is exact;
- no sync/pytest/compile/inference;
- exactly one correction commit with exact message;
- E3B2 not executed;
- STOP.
