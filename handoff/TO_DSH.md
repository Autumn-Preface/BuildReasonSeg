# TO_DSH — Task 8B.2-R1.1: Canonical Source Policy Cleanup

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `audit/task8b2-rc1-runtime-closure`
> Required starting HEAD: `77d98223c7a91b22980be621d6e6b975a54daefd`
> External RC1 delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`
> Predecessor: Task 8B.2-R1 commit `77d98223c7a91b22980be621d6e6b975a54daefd`

---

# 0. Executor-only rule

You are the executor only. ChatGPT has already made every engineering decision required for this task.

Do not:
- choose different files to remove;
- add exclusions beyond the exact set below;
- modify RC1 runtime behaviour;
- modify `check_setup.py`, `environment.yml`, `predict.py`, detector/runtime code or model metadata;
- install/uninstall/upgrade/downgrade packages;
- run inference or training;
- access final test data;
- download anything;
- modify checkpoints;
- modify Task 6N / Task 6O research code or scientific conclusions;
- merge to `main`;
- start Task 8B.2-R2, Task 8C or external-image testing.

If a STOP condition is reached, stop and report. Do not invent a workaround.
All user-facing DSH output must be Chinese.

---

# 1. ChatGPT audit decision

Task 8B.2-R1 is not yet accepted as complete.

The R1 implementation correctly registered the RC1 lightweight source/config snapshot, but ChatGPT audit found one policy inconsistency caused by the R1 copy policy:

The canonical tree currently contains downloaded third-party Qwen/SAM2 asset files and a generated integrity-cache file, while the frozen R1 design and `CANONICAL_SOURCE.md` state that downloaded Qwen/SAM2 assets and caches remain local-only.

This task is canonical-source policy cleanup only.

No runtime defect is fixed here. `RC1-ENV-01` remains deferred.

---

# 2. Exact cleanup set

Keep the canonical root exactly:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

## 2.1 Remove this entire tracked Qwen base-model text subtree

```text
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/
```

At R1 HEAD it must contain exactly these 9 tracked files:

```text
chat_template.json
config.json
generation_config.json
merges.txt
preprocessor_config.json
tokenizer.json
tokenizer_config.json
video_preprocessor_config.json
vocab.json
```

Do not remove or modify the corresponding files in the external runnable delivery.

## 2.2 Remove this generated cache file from canonical Git source

```text
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/qwen_integrity_cache.json
```

Do not remove the external-delivery counterpart.

## 2.3 Remove this downloaded SAM2 config asset from canonical Git source

```text
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/sam2/sam2.1_hiera_b+.yaml
```

Do not remove the external-delivery counterpart.

## 2.4 Explicitly keep these project-owned metadata files

```text
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/qwen_asset_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/buildreasonseg_advisor/model.yaml
delivery_src/BuildReasonSeg_Advisor_RC1/model/buildreasonseg_advisor/metadata.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/buildreasonseg_advisor/metrics.json
```

`qwen_asset_manifest.json` is project-generated integrity metadata and remains canonical.

---

# 3. Phase A — Git safety gate

Run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git branch --show-current
git rev-parse HEAD
git status --short
git remote -v
```

Continue only if:

```text
branch = audit/task8b2-rc1-runtime-closure
HEAD   = 77d98223c7a91b22980be621d6e6b975a54daefd
```

Allowed starting working-tree state:
- only `M handoff/TO_DSH.md`, if this task book was written there; or
- completely clean.

Any other modification/untracked file -> STOP.

Do not reset, stash, clean, discard or switch branches.

STOP report:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b2_r11_stop_report.md
```

---

# 4. Phase B — Verify audit finding before modifying

Confirm all exact cleanup paths exist in the Git-tracked canonical tree.

Continue only if:
- the Qwen subtree contains exactly the 9 files listed above and no additional tracked file;
- `qwen_integrity_cache.json` exists;
- `sam2.1_hiera_b+.yaml` exists.

If any expected path is absent, or if the Qwen subtree contains any extra tracked file -> STOP.

Do not choose a new cleanup set.

---

# 5. Phase C — Remove only the frozen 11 files

Delete from the Git canonical tree only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/chat_template.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/config.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/generation_config.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/merges.txt
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/preprocessor_config.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/tokenizer.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/tokenizer_config.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/video_preprocessor_config.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/vocab.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/qwen_integrity_cache.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/sam2/sam2.1_hiera_b+.yaml
```

Do not delete any external delivery file.
Do not edit any RC1 runtime file.

---

# 6. Phase D — Update source_manifest.json

Edit only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Remove exactly the manifest entries corresponding to the 11 deleted paths.

Do not change:
- schema;
- task;
- source_delivery;
- canonical_root;
- copy_policy;
- any remaining entry path;
- any remaining entry bytes;
- any remaining entry sha256.

Expected manifest entry count:

```text
135
```

Verify:
- `files` length = 135;
- entries remain lexicographically sorted by `path`;
- no duplicates;
- every remaining manifest path exists;
- every remaining entry `bytes` equals current file size;
- every remaining entry `sha256` equals current file SHA256;
- none of the 11 removed paths appears in `files`.

If count is not exactly 135 -> STOP.

---

# 7. Phase E — Clarify CANONICAL_SOURCE.md

Edit:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/CANONICAL_SOURCE.md
```

Keep its existing meaning and add this clarification:

> Project-generated asset manifests such as `qwen_asset_manifest.json` may remain tracked because they describe integrity requirements for local assets; downloaded third-party Qwen/SAM2 asset files and generated integrity caches are not canonical Git source.

Do not make other substantive documentation changes.

The final notice must still state:
- external delivery is the runnable local package;
- canonical tree is source/config;
- downloaded model assets remain local-only;
- RC1 runtime behaviour was not changed;
- `RC1-ENV-01` remains deferred to R2.

---

# 8. Phase F — Add canonical policy regression tests

Modify only:

```text
tests/test_sync_advisor_rc1_delivery.py
```

Add coverage for all of these assertions against the real repository canonical manifest/tree:

1. No manifest path starts with:
   `model/components/program_head/Qwen3-VL-2B-Instruct/`
2. Manifest does not contain:
   `model/components/program_head/qwen_integrity_cache.json`
3. Manifest does not contain:
   `model/components/sam2/sam2.1_hiera_b+.yaml`
4. Canonical tree does not contain the Qwen3-VL-2B-Instruct directory.
5. Canonical tree does not contain `qwen_integrity_cache.json`.
6. Canonical tree does not contain `sam2.1_hiera_b+.yaml`.
7. Canonical tree still contains:
   `model/components/program_head/qwen_asset_manifest.json`
8. Real repository manifest has exactly 135 file entries.
9. Every real manifest entry exists and its bytes/SHA256 match the canonical file.

These may be one or more tests, but all nine assertions must be covered.

Do not modify unrelated tests.

---

# 9. Phase G — Sync helper source remains frozen

Do NOT modify:

```text
scripts/sync_advisor_rc1_delivery.py
```

The helper already synchronizes only manifest-listed files and does not delete unlisted destination files.

If it fails solely because the manifest now has 135 entries -> STOP rather than altering it.

---

# 10. Phase H — Verify external delivery remains untouched

Run read-only check only:

```bat
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

Expected:

```text
checked=135
match=135
missing=0
mismatch=0
exit code 0
```

The external delivery is expected to still contain the removed Qwen/SAM2/cache files. The helper must ignore them because they are no longer manifest-listed.

Do not delete them.
Do not run normal sync in this task.

Any MISSING/MISMATCH among the 135 canonical files -> STOP.

---

# 11. Phase I — Tests

Run:

```bat
python -m pytest tests/test_sync_advisor_rc1_delivery.py -q
```

Must PASS.

Then:

```bat
python -m pytest tests/ -q
```

Must preserve the full repository suite.

Do not repair unrelated failures.
Any unrelated pre-existing failure -> STOP.

Do not run predict or model inference.

---

# 12. Phase J — Update R1 report

Update:

```text
docs/task8b2_r1_rc1_canonical_source.md
```

Do not rewrite the report from scratch.

Add:

```text
## 11. ChatGPT Audit Cleanup — Task 8B.2-R1.1
```

Record:
- ChatGPT identified the original copy policy as too broad;
- DSH R1 execution itself followed the prescribed policy;
- exactly 11 canonical files were removed:
  - 9 Qwen downloaded text assets;
  - 1 generated Qwen integrity cache;
  - 1 downloaded SAM2 config asset;
- external runnable delivery was not modified;
- `qwen_asset_manifest.json` remains tracked;
- manifest entries changed from 146 to 135;
- external delivery read-only sync check result;
- dedicated test result;
- full repository test result;
- no runtime behaviour changed;
- no package install;
- no inference;
- `RC1-ENV-01` remains deferred.

Correct any earlier statement that says Qwen tokenizer/config downloaded assets remain intentionally included in the final canonical source. Preserve historical R1 facts by clearly distinguishing original R1 state from post-audit R1.1 final state.

---

# 13. Phase K — Update handoff

Update:

```text
handoff/FROM_DSH.md
```

with:

```text
Task: 8B.2-R1.1
Branch: audit/task8b2-rc1-runtime-closure
Starting HEAD: 77d98223c7a91b22980be621d6e6b975a54daefd
Canonical manifest entries: 135
Third-party downloaded assets in canonical source: NONE for the frozen cleanup set
Generated qwen integrity cache in canonical source: ABSENT
qwen_asset_manifest.json: PRESENT
External delivery check: 135/135 PASS
Dedicated tests: <actual>
Repository suite: <actual>
RC1-ENV-01: DEFERRED
Next action: Awaiting ChatGPT audit. Do not start Task 8B.2-R2.
```

Markdown formatting is allowed.

Do not modify `handoff/PROJECT_STATE.md`.

---

# 14. Phase L — Git diff gate

Run:

```bat
git status --short
git diff --check
git diff
```

The only allowed changes are:

```text
handoff/TO_DSH.md
delivery_src/BuildReasonSeg_Advisor_RC1/CANONICAL_SOURCE.md
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/<exact 9 deletions>
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/qwen_integrity_cache.json
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/sam2/sam2.1_hiera_b+.yaml
tests/test_sync_advisor_rc1_delivery.py
docs/task8b2_r1_rc1_canonical_source.md
handoff/FROM_DSH.md
```

No other path may change.

If any other path changed -> STOP.

Do not perform unrelated whitespace/format cleanup.

---

# 15. Phase M — Commit

Stage files individually. Do not use `git add .` or `git add -A`.

Verify:

```bat
git diff --cached --name-only
```

Commit exactly:

```text
fix(rc1): exclude local model assets from canonical source
```

After commit:

```bat
git status --short
git show --stat --oneline HEAD
```

Working tree must be clean. Otherwise STOP and do not push.

---

# 16. Phase N — Push

Push:

```bat
git push origin audit/task8b2-rc1-runtime-closure
```

No force push.

Then:

```bat
git rev-parse HEAD
git status --short
```

---

# 17. Completion gate

Declare `TASK 8B.2-R1.1 COMPLETE` only if all are true:

1. starting branch/head exactly matched;
2. only the frozen 11 canonical files were removed;
3. external delivery assets were not deleted or modified;
4. manifest has exactly 135 entries;
5. all 135 manifest entries match canonical bytes/SHA256;
6. Qwen downloaded text subtree is absent from canonical Git source;
7. `qwen_integrity_cache.json` is absent from canonical Git source;
8. downloaded SAM2 config is absent from canonical Git source;
9. `qwen_asset_manifest.json` remains present;
10. sync helper source was not modified;
11. external delivery `--check` is 135/135 PASS;
12. dedicated tests PASS;
13. full repository suite PASS;
14. report updated;
15. handoff updated;
16. no package installation;
17. no inference/training/final-test access;
18. exact commit created;
19. push succeeds;
20. working tree clean;
21. DSH stops.

Otherwise report `TASK 8B.2-R1.1 PARTIAL`.

Do not enter R2.

---

# 18. Final DSH reply

Output only:

```text
TASK 8B.2-R1.1 COMPLETE / PARTIAL

Branch:
audit/task8b2-rc1-runtime-closure

Commit:
<sha or NONE>

Push:
PASS / NOT DONE

Canonical manifest:
135 / other

Removed canonical-only files:
11 / other

Qwen downloaded asset subtree in canonical:
ABSENT / PRESENT

qwen_integrity_cache.json in canonical:
ABSENT / PRESENT

SAM2 downloaded config in canonical:
ABSENT / PRESENT

qwen_asset_manifest.json:
PRESENT / MISSING

External delivery --check:
135/135 PASS / FAIL / NOT RUN

Dedicated tests:
<result>

Repository suite:
<result>

RC1-ENV-01:
DEFERRED

Report:
docs/task8b2_r1_rc1_canonical_source.md

Handoff:
handoff/FROM_DSH.md

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得进入 Task 8B.2-R2、Task 8C、外部图片测试或任何新研发。
```

Then stop.
