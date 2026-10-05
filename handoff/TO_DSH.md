请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3B2 — Controlled External Sync and Full Delivery Suite**

# TO_DSH — Task 8B.3-REF01-E3B2: Controlled External Sync and Full Delivery Suite

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `fix/task8b3-ref01-eligibility-repair-sync`
> Required starting HEAD: `300cdad5629b945ffe10480351081f9c8befe7f4`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Canonical RC1: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\delivery_src\BuildReasonSeg_Advisor_RC1`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. CHATGPT AUDIT DISPOSITION

E3B1 is formally CLOSED.

Frozen pre-sync facts:

```text
Design:
LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

Manifest identity basis:
GIT_CANONICAL_BLOB_BYTES

Manifest file count:
135

Manifest identities:
PASS 135/135

Detector Git-canonical identity:
21257 bytes
934bbb9c3fbdbd5465fdd3e074a6ffa4721df9e77918970bfae2e88a6a91f883

test_task8b_runtime.py Git-canonical identity:
28128 bytes
71915e8abfbe96b328736c2c84a8773640ec58cd01f847f6b58aa33edf6ec46b

Last accepted external pre-sync state:
133 match / 0 missing / 2 mismatch

Mismatch paths:
buildreasonseg/runtime/detector.py
tests/test_task8b_runtime.py
```

E3B2 is the FIRST task authorized to write the external RC1 delivery for this repair.

# 1. EXECUTOR CONTRACT

DSH has ZERO technical discretion.

Allowed sequence ONLY:

1. verify exact branch/head and clean state;
2. re-validate the current external pre-sync state with `--check`;
3. verify the 135-entry sync policy contains no runtime weight/output path;
4. run the existing sync helper ONCE without `--check`;
5. run post-sync `--check`;
6. copy the canonical `source_manifest.json` Git-object bytes to external exactly once;
7. verify external `source_manifest.json` byte identity equals the canonical Git object;
8. run external `check_setup.py`;
9. run external targeted `test_task8b_runtime.py`;
10. run external full delivery test suite;
11. create exact evidence/report/FROM_DSH;
12. make exactly one repo commit;
13. push once;
14. STOP.

Forbidden:
- NO canonical product/test/manifest edit;
- NO sync-helper edit;
- NO threshold/algorithm change;
- NO detector/model inference;
- NO `predict.py`;
- NO Demo inference;
- NO Qwen/SAM2/detector forward pass initiated manually;
- NO test/source self-repair;
- NO package install/update;
- NO environment change;
- NO deletion in external RC1;
- NO external model-weight overwrite;
- NO output/runs/log cleanup;
- NO rebase/reset/amend/stash/clean;
- NO force push;
- NO intermediate commit/push;
- NO NEXT/E3C.

Any unexpected result or failed gate => STOP.
Do not self-repair.

# 2. GIT GATE

Require exactly:

```text
git branch --show-current
= fix/task8b3-ref01-eligibility-repair-sync

git rev-parse HEAD
= 300cdad5629b945ffe10480351081f9c8befe7f4
```

Allowed initial working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else => STOP.

No commit/push until §13.

# 3. CANONICAL IMMUTABILITY / ARTIFACT GATE

Require NO working-tree/index change for:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
evaluation/task8b3_ref01_e3b1_manifest_canonicalization.json
docs/task8b3_ref01_e3b1_manifest_canonicalization.md
```

Require the authoritative E3B1 evidence/report both exist.

Any mismatch => STOP.

# 4. IMMEDIATE PRE-SYNC EXTERNAL CHECK

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe scripts\sync_advisor_rc1_delivery.py --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 --check
```

Required exit:

```text
1
```

Required summary exactly:

```text
checked=135 match=133 missing=0 mismatch=2
```

Required mismatch paths EXACTLY:

```text
MISMATCH buildreasonseg/runtime/detector.py
MISMATCH tests/test_task8b_runtime.py
```

No third mismatch.
No missing.

If external state differs => STOP.
Do NOT sync.

# 5. SYNC POLICY SAFETY GATE

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import json,pathlib; p=pathlib.Path(r'delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json'); x=json.loads(p.read_text(encoding='utf-8')); ps=[e['path'] for e in x['files']]; assert len(ps)==135; assert not any(q.lower().endswith(('.pt','.pth','.ckpt','.safetensors','.bin')) for q in ps); assert not any(q.startswith(('inference/output/','runs/','logs/')) for q in ps); assert 'source_manifest.json' not in ps; print('SYNC_POLICY_SAFE: PASS 135 lightweight entries')"
```

Require exit 0 and:

```text
SYNC_POLICY_SAFE: PASS 135 lightweight entries
```

This gate establishes that the helper will not overwrite runtime weights, generated outputs, runs, or logs.

# 6. CONTROLLED EXTERNAL SYNC — EXACTLY ONCE

Run exactly ONCE:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe scripts\sync_advisor_rc1_delivery.py --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Required exit:

```text
0
```

Required final summary exactly:

```text
copied=135 verified=135 failures=0
```

If non-zero or summary differs => STOP.

Do NOT rerun sync.

# 7. POST-SYNC 135/135 CHECK

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe scripts\sync_advisor_rc1_delivery.py --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 --check
```

Require exit 0.

Required summary exactly:

```text
checked=135 match=135 missing=0 mismatch=0
```

No MISMATCH.
No MISSING.

Failure => STOP.
Do not re-sync.

# 8. COPY CANONICAL SOURCE MANIFEST TO EXTERNAL

The sync helper deliberately does not self-copy `source_manifest.json`.

After §7 PASS, run exactly ONCE:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import hashlib,pathlib,subprocess; spec='HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json'; b=subprocess.run(['git','show',spec],check=True,stdout=subprocess.PIPE).stdout; dst=pathlib.Path(r'C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\source_manifest.json'); dst.write_bytes(b); rb=dst.read_bytes(); assert rb==b; print('EXTERNAL_SOURCE_MANIFEST_COPY: PASS'); print('SOURCE_MANIFEST_BYTES=',len(b)); print('SOURCE_MANIFEST_SHA256=',hashlib.sha256(b).hexdigest())"
```

Require exit 0.

Required stdout contains:

```text
EXTERNAL_SOURCE_MANIFEST_COPY: PASS
```

Record the observed:

```text
SOURCE_MANIFEST_BYTES
SOURCE_MANIFEST_SHA256
```

verbatim for evidence/report.

Do NOT edit the manifest.
Do NOT run this command twice.

# 9. EXTERNAL SETUP CHECKER

Run exactly ONCE:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\check_setup.py
```

Require exit 0.

Required final status:

```text
BuildReasonSeg environment: READY
```

The checker is allowed to read/hash model assets and verify runtime dependencies.
This is NOT model inference.

If NOT READY => STOP.
Do not install/change anything.

Record the exact final status and any `[WARN]` lines.
Warnings are acceptable only if exit = 0 and final status is READY.

# 10. EXTERNAL TARGETED REGRESSION

Run exactly ONCE:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import os,pytest; os.chdir(r'C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1'); raise SystemExit(pytest.main(['tests/test_task8b_runtime.py','-q']))"
```

Require exit 0.

Require final pytest summary contains:

```text
40 passed
```

Record the exact final pytest summary line verbatim.

Failure => STOP.
No rerun.
No edits.

# 11. EXTERNAL FULL DELIVERY SUITE

Only after §10 PASS, run exactly ONCE:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe -c "import os,pytest; os.chdir(r'C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1'); raise SystemExit(pytest.main(['tests','-q']))"
```

Require exit 0.

Expected complete suite count:

```text
124 passed
```

Require final summary contains `124 passed` and contains none of:

```text
failed
error
ERROR
```

Record the exact final pytest summary line verbatim.

If exit non-zero or count differs => STOP.
No rerun.
No test/source/environment repair.

# 12. EVIDENCE

Only if §§4–11 all PASS, create exactly:

```text
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
```

Required structure:

```json
{
  "task": "8B.3-REF01-E3B2",
  "starting_head": "300cdad5629b945ffe10480351081f9c8befe7f4",
  "branch": "fix/task8b3-ref01-eligibility-repair-sync",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "scope": "CONTROLLED_EXTERNAL_SYNC_AND_FULL_DELIVERY_VALIDATION",
  "detector_model_calls": 0,
  "canonical_source_changed": false,
  "canonical_manifest_changed": false,
  "sync_policy_entries": 135,
  "sync_policy_runtime_weights_included": false,
  "external_presync": {
    "exit": 1,
    "match": 133,
    "missing": 0,
    "mismatch": 2,
    "mismatch_paths": [
      "buildreasonseg/runtime/detector.py",
      "tests/test_task8b_runtime.py"
    ]
  },
  "external_sync": {
    "runs": 1,
    "exit": 0,
    "copied": 135,
    "verified": 135,
    "failures": 0
  },
  "external_postsync": {
    "exit": 0,
    "match": 135,
    "missing": 0,
    "mismatch": 0
  },
  "external_source_manifest": {
    "copied_from_git_object": true,
    "bytes": "<OBSERVED INTEGER>",
    "sha256": "<OBSERVED SHA256>"
  },
  "setup_checker": {
    "runs": 1,
    "exit": 0,
    "final_status": "BuildReasonSeg environment: READY"
  },
  "external_targeted_test": {
    "runs": 1,
    "exit": 0,
    "summary": "<EXACT FINAL SUMMARY LINE>"
  },
  "external_full_suite": {
    "runs": 1,
    "exit": 0,
    "summary": "<EXACT FINAL SUMMARY LINE>",
    "expected_pass_count": 124
  },
  "external_write_performed": true,
  "model_inference_performed": false,
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY"
}
```

Only these fields may vary from observed stdout:
- external_source_manifest.bytes
- external_source_manifest.sha256
- external_targeted_test.summary
- external_full_suite.summary

No other key/value may vary.

# 13. REPORT

Create exactly:

```text
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
```

Required conclusions:

```text
Task = 8B.3-REF01-E3B2
Status = COMPLETE
Design = LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1

External pre-sync =
133 match / 0 missing / 2 mismatch

Controlled sync =
1 run
135 copied
135 verified
0 failures

External post-sync =
135 match / 0 missing / 0 mismatch

External source_manifest =
copied from Git canonical object
bytes = <observed>
sha256 = <observed>

Runtime weights / generated outputs touched by sync policy =
NO

Setup checker =
READY

External targeted regression =
40 passed

External full delivery suite =
124 passed

Detector/model inference =
NONE

Canonical source/manifest changes in E3B2 =
NONE

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

left/below reference-selection defects =
UNRESOLVED

Outcome =
REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED

NEXT =
REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
```

# 14. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Active handoff must include:

```text
Task: 8B.3-REF01-E3B2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-ref01-eligibility-repair-sync
Starting HEAD: 300cdad5629b945ffe10480351081f9c8befe7f4
Design selected by: CHATGPT
Design ID: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
DSH algorithm choice performed: NO
Detector/model calls: 0
Canonical source changed: NO
Canonical manifest changed: NO
External pre-sync: 133 match / 0 missing / 2 mismatch
External sync runs: 1
External sync: 135 copied / 135 verified / 0 failures
External post-sync: 135 match / 0 missing / 0 mismatch
External source_manifest copied from Git object: YES
External source_manifest bytes: <observed>
External source_manifest SHA256: <observed>
Setup checker runs: 1
Setup checker: READY
External targeted test runs: 1
External targeted regression: <exact summary>
External full suite runs: 1
External full suite: <exact summary>
External write performed: YES
Model inference performed: NO
Evidence: evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
Report: docs/task8b3_ref01_e3b2_external_sync_full_suite.md
Outcome: REF01_ELIGIBILITY_REPAIR_EXTERNAL_SYNC_VALIDATED
Next gate: REF01_ELIGIBILITY_REPAIR_LOCKED_PRODUCTION_REPLAY
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects: UNRESOLVED
Next action: Awaiting ChatGPT audit; do not execute NEXT/E3C.
```

# 15. FINAL REPO DIFF GATE

Before commit:

```text
git diff --name-only 300cdad5629b945ffe10480351081f9c8befe7f4
```

Allowed ONLY:

```text
evaluation/task8b3_ref01_e3b2_external_sync_full_suite.json
docs/task8b3_ref01_e3b2_external_sync_full_suite.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Require these do NOT appear:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

Also require:

```text
git rev-list --count 300cdad5629b945ffe10480351081f9c8befe7f4..HEAD
= 0
```

Any mismatch => STOP.

# 16. STATUS → COMMIT MESSAGE

If §§2–15 all PASS:

```text
Status = COMPLETE
Commit message EXACTLY:
docs(rc1): validate external extent repair sync
```

Otherwise:

```text
Status = STOP
Commit message EXACTLY:
docs(rc1): record external extent repair sync stop
```

Commit exactly once.
NO amend.
NO intermediate commit/push.

After commit require:

```text
git rev-list --count 300cdad5629b945ffe10480351081f9c8befe7f4..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute NEXT/E3C.

Then STOP.

# 17. COMPLETE DEFINITION

COMPLETE only if:
- exact branch/start HEAD;
- canonical repo product/manifest/helper remain unchanged;
- immediate pre-sync external state is exactly 133/0/2;
- sync policy is confirmed lightweight only;
- external sync runs exactly once and copies/verifies 135/135;
- post-sync check is exactly 135/0/0;
- external source_manifest equals Git canonical object bytes;
- external setup checker is READY;
- external targeted regression is 40 passed;
- external full delivery suite is 124 passed;
- no model inference occurs;
- exact evidence/report/FROM_DSH are written;
- repo diff contains only four allowed artifact/handoff paths;
- exactly one commit with exact message;
- NEXT/E3C not executed;
- STOP.
