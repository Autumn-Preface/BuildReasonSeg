请先读取 `handoff/TO_DSH.md`，并严格以该文件作为本轮唯一任务书。

本次任务名称：**Task 8B.3-REF01-E3C0-R2-R3 — Exact proposals.json Schema Probe**

# TO_DSH — Task 8B.3-REF01-E3C0-R2-R3: Exact proposals.json Schema Probe

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required branch: `audit/task8b3-ref01-locked-replay-artifacts`
> Required starting HEAD: `fa9ede22f9bee640a6211f0dfd807439d0e8bc57`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required script: `C:\D\DeepSeekHarness\E3C0_R2_R3_schema_probe.py`
> Required SHA256: `49e51459dd9bf218f42fc62021f2c15dd223991a2ede797e42c0519f44934873`

# 0. CHATGPT AUDIT DISPOSITION

E3C0-R2-R2 is approved only as a SAFE STOP.

The corrected status parser worked, but the exact audit script then failed on its own unsupported assumption about `proposals.json`:

```text
AssertionError:
AMBIGUOUS_PROPOSALS_TOP_LEVEL: keys=[]
```

This means ChatGPT does not yet know the actual JSON container schema.

DSH correctly did not modify/reconstruct/rerun the script.

R2-R3 performs ONLY a schema probe of the four exact known `proposals.json` files. It MUST NOT determine replay readiness and MUST NOT run any selector/model.

# 1. EXECUTOR CONTRACT

Allowed ONLY:

1. verify branch/head;
2. verify supplied schema-probe script SHA;
3. run supplied script exactly ONCE;
4. inspect stdout/diff;
5. commit the generated schema evidence/report/FROM_DSH/TO_DSH;
6. push once;
7. STOP.

Forbidden:

- NO broad search;
- NO `result.json` read;
- NO forensics-file read;
- NO repo artifacts/inference discovery;
- NO external logs;
- NO detector/model inference;
- NO proposal regeneration;
- NO actual replay;
- NO readiness determination;
- NO selector execution;
- NO external write;
- NO product/test/manifest/helper edit;
- NO script edit/rebuild;
- NO script rerun;
- NO pytest;
- NO py_compile;
- NO package/environment change;
- NO reset/amend/rebase/stash/clean;
- NO force push;
- NO NEXT.

# 2. EXACT FILE SCOPE

The script may read exactly these four files and no other scientific artifact:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1010\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1003\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1008\proposals.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\inference\output\diagnostics\1009\proposals.json
```

It may additionally read Git state and `handoff/FROM_DSH.md` solely to preserve ARTIFACT-FACTS and validate diff.

# 3. SCRIPT GATE

Place exactly:

```text
C:\D\DeepSeekHarness\E3C0_R2_R3_schema_probe.py
```

SHA256 must equal:

```text
49e51459dd9bf218f42fc62021f2c15dd223991a2ede797e42c0519f44934873
```

Do not modify/reconstruct.

# 4. GIT PRE-FLIGHT

Require:

```text
branch = audit/task8b3-ref01-locked-replay-artifacts
HEAD = fa9ede22f9bee640a6211f0dfd807439d0e8bc57
```

`git status --short` may be:
- empty;
- ` M handoff/TO_DSH.md`;
- `M  handoff/TO_DSH.md`.

No other entry.

# 5. RUN ONCE

Run exactly once:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3C0_R2_R3_schema_probe.py
```

Require exit 0.

Require stdout:

```text
E3C0_R2_R3_SCHEMA_PROBE: COMPLETE
FILES=4
```

No rerun.

# 6. GENERATED FILES

Require exactly these four diff paths:

```text
evaluation/task8b3_ref01_e3c0_r2r3_proposals_schema_probe.json
docs/task8b3_ref01_e3c0_r2r3_proposals_schema_probe.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Evidence must say:

```text
task = 8B.3-REF01-E3C0-R2-R3
starting_head = fa9ede22f9bee640a6211f0dfd807439d0e8bc57
scope = EXACT_FOUR_PROPOSALS_JSON_SCHEMA_PROBE
detector_model_calls = 0
proposal_regeneration_performed = false
actual_replay_performed = false
external_write_performed = false
broad_search_performed = false
overall_outcome = PROPOSALS_JSON_SCHEMA_CAPTURED
next_gate = CHATGPT_REVIEW_FOR_FINAL_READINESS_AUDIT_SCRIPT
```

Must contain exactly four file entries: right/left/above/below.

# 7. SCIENTIFIC NON-CLAIMS

This task makes NO readiness conclusion.

It MUST NOT claim:
- `RECORD_LEVEL_REPLAY_READY`;
- `FULL_PRODUCTION_REPLAY_READY`;
- repaired selector success;
- right1 / left14 / above5 / below1 validated.

Freeze:

```text
REF-01 = ACTIVE
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
```

# 8. COMMIT / PUSH

If all gates pass, commit exactly once:

```text
docs(rc1): capture locked proposal json schema
```

Push current branch once.
No force push.
Then STOP.

If any gate fails:
- do not rerun;
- record exact STOP in FROM_DSH;
- commit once:
  `docs(rc1): record proposal json schema probe stop`
- push once;
- STOP.

# 9. COMPLETE DEFINITION

COMPLETE only if:
- exact HEAD;
- exact script SHA;
- one script run;
- exactly four known `proposals.json` files read;
- zero broad search/inference/regeneration/replay/external write;
- readiness NOT determined;
- exact four output paths;
- one commit;
- push once;
- STOP.
