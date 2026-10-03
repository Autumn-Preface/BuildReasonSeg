# TO_DSH — Task 8B.2-R1: Register RC1 Canonical Delivery Source

> Status: ACTIVE  
> Role boundary: ChatGPT decides; DSH executes only.  
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`  
> Existing local branch: `audit/task8b2-rc1-runtime-closure`  
> Existing external RC1 delivery: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`  
> Base repository commit at the Task 8B.2 STOP: `6c2b915dbc64acdeb099d194005d74c7180c95fa`  
> Predecessor: Task 8B.2 → PARTIAL / Phase D1 STOP because RC1 delivery had no Git-tracked canonical source.

---

# 0. Executor-only rule

You are the executor only.

Do not make technical decisions beyond the exact rules in this document.

Do not:
- choose a different canonical path;
- choose a different branch;
- redesign the RC1 package;
- refactor copied RC1 source;
- fix the Ultralytics problem in this task;
- install/uninstall/upgrade/downgrade any Python package;
- run model inference;
- modify checkpoints;
- modify Task 6N / Task 6O research code, model architecture, data, losses, metrics, thresholds, parser semantics or scientific conclusions;
- access final test data;
- download anything;
- start Task 8B.2-R2, Task 8C, external-image testing or any new development;
- resolve any ambiguity not explicitly covered here.

If any STOP condition in this document is triggered, stop immediately, write the required local stop report, and wait for ChatGPT.

All user-facing DSH output must be Chinese. File names, commands, code symbols and technical identifiers may remain English.

---

# 1. ChatGPT frozen decision

The Task 8B.2 STOP is accepted as correct.

The root structural issue is:

> `BuildReasonSeg_Advisor_RC1` currently exists as an external delivery artifact, but the Git repository does not contain a canonical tracked copy of the RC1 runtime source/configuration that produced or defines that delivery.

ChatGPT now freezes the following engineering decision:

1. The repository will gain a canonical, lightweight, Git-tracked RC1 delivery source tree at exactly:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

2. The canonical tree is a **source/config snapshot**, not a binary delivery.
3. The current external delivery is the only source snapshot allowed for this registration task.
4. All copied source/config files must be byte-for-byte identical to the current external RC1 delivery at Task 8B.2-R1 start.
5. No RC1 runtime behavior is changed in this task.
6. Model weights, downloaded model assets, images, caches, logs, generated predictions and Conda environments must remain local-only and must not enter Git.
7. A deterministic one-way sync helper will be added at exactly:

```text
scripts/sync_advisor_rc1_delivery.py
```

8. This task only establishes canonical ownership and reproducibility of the lightweight RC1 source/config layer.
9. The Ultralytics/readiness fix remains deferred to **Task 8B.2-R2**, which ChatGPT will issue only after auditing this task.

---

# 2. Expected starting state

Repository:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
```

External delivery:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Expected current local branch:

```text
audit/task8b2-rc1-runtime-closure
```

Expected repository HEAD ancestry includes:

```text
6c2b915dbc64acdeb099d194005d74c7180c95fa
```

The previous Task 8B.2 created the branch but did not commit or push it.

The only allowed pre-existing Git working-tree modification is:

```text
M handoff/TO_DSH.md
```

where that file contains the current Task 8B.2-R1 task book supplied by the user/ChatGPT.

No other pre-existing tracked modification or untracked repository file is authorized.

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

## A1. Continue only if all are true

1. Current branch is exactly:

```text
audit/task8b2-rc1-runtime-closure
```

2. `git status --short` is either:
   - only `M handoff/TO_DSH.md`, or
   - completely clean if the task book was not written into the tracked file.

3. No other tracked or untracked repository changes exist.

4. `origin` points to the existing BuildReasonSeg GitHub repository.

## A2. STOP conditions

STOP if:
- branch differs;
- branch is missing;
- there is any additional modified/untracked file;
- HEAD cannot be resolved;
- origin is missing or changed.

Do not switch branches, stash, reset, clean, discard, delete or overwrite anything.

Write local stop report:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b2_r1_stop_report.md
```

Then stop.

---

# 4. Phase B — External delivery inventory

Do not modify the delivery.

Inventory:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Generate a local-only inventory file at:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b2_r1_delivery_inventory.txt
```

The inventory must list every file as:

```text
relative_path<TAB>size_bytes
```

sorted lexicographically by normalized forward-slash relative path.

This inventory is local-only and must not be committed.

Also record whether the following paths exist:

```text
check_setup.py
environment.yml
predict.py
buildreasonseg/
model/buildreasonseg_advisor/
```

## B1. Required-path gate

Continue only if all five required paths above exist.

If any is absent:

STOP and write the stop report.

---

# 5. Phase C — Fixed canonical-copy policy

Create exactly:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

Do not create any alternative source directory.

The canonical snapshot is produced from the external delivery using the following fixed rules.

## C1. Include extensions

Copy recursively only regular files whose file-name extension, case-insensitive, is one of:

```text
.py
.pyw
.json
.yaml
.yml
.toml
.ini
.cfg
.conf
.txt
.md
.csv
```

Also copy extensionless regular files only if their exact relative path is one of:

```text
LICENSE
NOTICE
```

if those files exist.

## C2. Excluded directories

Do not copy anything under a path component equal to any of:

```text
.git
.conda
__pycache__
.pytest_cache
logs
log
cache
caches
tmp
temp
runs
outputs
output
predictions
```

Also exclude these exact subtrees:

```text
inference/input
inference/output
```

## C3. Excluded binary/model/data file types

Never copy files with these extensions, even if they appear elsewhere:

```text
.pt
.pth
.ckpt
.safetensors
.onnx
.engine
.tflite
.h5
.pb
.bin
.npy
.npz
.tif
.tiff
.jpg
.jpeg
.png
.bmp
.webp
.gif
.zip
.7z
.rar
.tar
.gz
```

## C4. Symlinks / junctions

Do not follow or copy symlinks, directory junctions or reparse-point targets.

If any candidate source/config file is reachable only through a symlink/junction/reparse point:

STOP and list it in the stop report.

## C5. Byte identity

Every copied file must be copied without text normalization or encoding conversion.

After copy, for every copied file verify:

```text
source size == canonical size
source SHA256 == canonical SHA256
```

Any mismatch → STOP.

Do not edit any copied RC1 file in this task.

---

# 6. Phase D — Canonical snapshot manifest

Create:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

UTF-8, deterministic formatting.

Required top-level fields:

```json
{
  "schema": "BuildReasonSeg.AdvisorRC1.SourceManifest.v1",
  "task": "8B.2-R1",
  "source_delivery": "C:\\D\\DeepSeekHarness\\delivery\\BuildReasonSeg_Advisor_RC1",
  "canonical_root": "delivery_src/BuildReasonSeg_Advisor_RC1",
  "copy_policy": "Task 8B.2-R1 fixed lightweight source/config policy",
  "files": []
}
```

For every copied file except `source_manifest.json` itself, `files` must contain:

```json
{
  "path": "normalized/forward/slash/path",
  "bytes": 123,
  "sha256": "lowercase hex"
}
```

Rules:
- one entry per copied file;
- lexicographic order by `path`;
- no duplicate path;
- no absolute path in individual entries;
- no model/binary files;
- no logs or outputs.

---

# 7. Phase E — Canonical source notice

Create:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/CANONICAL_SOURCE.md
```

Use exactly this substantive meaning:

```text
# BuildReasonSeg Advisor RC1 canonical source

This directory is the Git-tracked canonical source/config layer for the external
BuildReasonSeg Advisor RC1 delivery.

Registered by Task 8B.2-R1 from:
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1

This directory intentionally excludes:
- model weights/checkpoints;
- downloaded Qwen/SAM2 assets;
- user inference images;
- generated masks/overlays/diagnostics;
- logs/caches;
- Conda environments.

The external delivery remains the runnable local package. Changes to RC1
source/config must be made in this canonical tree first and synchronized to the
external delivery using scripts/sync_advisor_rc1_delivery.py.

Task 8B.2-R1 performs registration only and does not change runtime behaviour.
The Ultralytics/readiness issue remains deferred to Task 8B.2-R2 pending
ChatGPT audit.
```

Minor line wrapping is allowed. Do not add claims beyond this meaning.

---

# 8. Phase F — Deterministic sync helper

Create exactly:

```text
scripts/sync_advisor_rc1_delivery.py
```

This is a source/config sync helper only.

## F1. CLI

The script must support:

```text
python scripts/sync_advisor_rc1_delivery.py --destination <path>
python scripts/sync_advisor_rc1_delivery.py --destination <path> --check
```

Canonical source is fixed internally relative to repository root:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

Do not add a CLI option that changes canonical source.

## F2. Normal mode

Without `--check`:

1. Read `source_manifest.json`.
2. For every manifest-listed file:
   - source = canonical root / manifest path
   - destination = supplied destination / manifest path
3. Create missing parent directories.
4. Copy each file byte-for-byte.
5. Overwrite destination counterpart only for manifest-listed files.
6. Do not delete any destination file.
7. Do not touch destination weights/assets/images/logs/outputs.
8. Recompute destination SHA256 after copy and require equality with current canonical source file.
9. Print a summary:
   - copied count;
   - verified count;
   - failures.

## F3. Check mode

With `--check`:

1. Do not modify anything.
2. For every manifest-listed file:
   - verify source exists;
   - verify destination exists;
   - compare bytes and SHA256.
3. Report:
   - MATCH;
   - MISSING;
   - MISMATCH.
4. Exit code:
   - `0` only if all manifest-listed files match;
   - non-zero otherwise.

## F4. Safety

The script must refuse:
- destination equal to canonical source root;
- destination inside canonical source root;
- destination path that does not exist;
- source manifest with duplicate paths;
- manifest path containing `..`;
- absolute manifest paths.

It must not:
- run pip/conda;
- import model frameworks;
- invoke predict;
- delete destination content;
- alter model assets.

---

# 9. Phase G — Tests for sync helper

Create exactly:

```text
tests/test_sync_advisor_rc1_delivery.py
```

Use temporary directories only.

Add tests for at least:

1. manifest paths are relative and normalized;
2. duplicate manifest path rejected;
3. `..` path rejected;
4. absolute path rejected;
5. normal sync copies listed files;
6. normal sync overwrites only listed counterpart;
7. unlisted destination file is preserved;
8. check mode passes on exact match;
9. check detects missing destination file;
10. check detects mismatched destination file;
11. source==destination rejected;
12. destination-inside-source rejected.

Tests must not access:
- model weights;
- final test;
- internet;
- external RC1 delivery.

---

# 10. Phase H — Verify initial canonical snapshot against current delivery

Run from repository root:

```bat
python scripts/sync_advisor_rc1_delivery.py ^
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1 ^
  --check
```

This must be a read-only check.

Expected result:

```text
all manifest-listed files MATCH
exit code 0
```

If any MISSING/MISMATCH occurs:

STOP.

Do not run normal sync to hide the difference.

This task is registration of the current delivery snapshot; mismatch means registration is inconsistent.

---

# 11. Phase I — Repository test gate

Run the new dedicated test first:

```bat
python -m pytest tests/test_sync_advisor_rc1_delivery.py -q
```

Must PASS.

Then run the repository suite:

```bat
python -m pytest tests/ -q
```

Rules:
- Do not modify unrelated tests to make them pass.
- Do not reduce existing coverage by deleting/skipping tests.
- If failures are caused by the new files, fix only the new canonicalization/sync implementation.
- If an unrelated pre-existing test failure is encountered, STOP and report it. Do not repair unrelated code.

Do not run any training or inference command.

---

# 12. Phase J — Git safety audit before documentation

Run:

```bat
git status --short
git diff --check
```

Inspect all new/staged candidates.

The only repository changes permitted by this task are:

```text
handoff/TO_DSH.md
delivery_src/BuildReasonSeg_Advisor_RC1/**
scripts/sync_advisor_rc1_delivery.py
tests/test_sync_advisor_rc1_delivery.py
docs/task8b2_r1_rc1_canonical_source.md
handoff/FROM_DSH.md
handoff/PROJECT_STATE.md
```

`handoff/PROJECT_STATE.md` is optional and may only be changed as specified in Phase L.

If any other repository path changed:

STOP.

Also STOP if any candidate file is:
- a model/checkpoint;
- an image;
- an array/cache;
- a generated inference output;
- a log;
- larger than 10 MiB.

Do not commit local delivery logs/inventories.

---

# 13. Phase K — Required report

Create:

```text
docs/task8b2_r1_rc1_canonical_source.md
```

Required sections:

## 1. Task
`Task 8B.2-R1 — Register RC1 Canonical Delivery Source`

## 2. ChatGPT Decision
State that ChatGPT accepted the prior D1 STOP and fixed the canonical source path to:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

## 3. Starting State
Record:
- branch;
- starting HEAD;
- allowed initial working-tree state;
- external delivery path.

## 4. Registration Policy
Record:
- included extensions;
- excluded directories;
- excluded binary/model/data extensions;
- symlink policy.

## 5. Snapshot Result
Record:
- copied file count;
- total copied bytes;
- manifest path;
- all source↔canonical SHA256 verification result.

## 6. Required RC1 Files
Record whether the canonical tree now contains counterparts of:
- `check_setup.py`;
- `environment.yml`;
- `predict.py`;
- `buildreasonseg/`;
- `model/buildreasonseg_advisor/` lightweight config/metadata files.

Do not claim weights were copied.

## 7. Sync Helper
Record:
- exact script path;
- normal-mode semantics;
- check-mode semantics;
- safety restrictions.

## 8. Validation
Table:

| Check | Result |
|---|---|
| delivery inventory created locally | PASS/FAIL |
| canonical snapshot created | PASS/FAIL |
| byte identity | PASS/FAIL |
| SHA256 identity | PASS/FAIL |
| manifest deterministic | PASS/FAIL |
| external delivery `--check` | PASS/FAIL |
| dedicated sync tests | PASS/FAIL |
| repository suite | PASS/FAIL |
| no binary/model assets staged | PASS/FAIL |
| no runtime behaviour changed | PASS/FAIL |

## 9. Deferred Issue
State exactly in substance:

> `RC1-ENV-01` is intentionally not fixed in Task 8B.2-R1. Ultralytics dependency installation/pinning, readiness validation, runtime/provenance display and A1 inference remain deferred to Task 8B.2-R2 after ChatGPT audit.

## 10. Scientific Scope
State:

> No training, checkpoint modification, final-test access, architecture change, parser-semantic change, threshold tuning or research conclusion change occurred in Task 8B.2-R1.

---

# 14. Phase L — Handoff

Update:

```text
handoff/FROM_DSH.md
```

with only the current task summary:

- task = `8B.2-R1`;
- branch;
- starting HEAD;
- canonical source root;
- snapshot file count;
- snapshot verification;
- sync check result;
- dedicated test result;
- repository suite result;
- report path;
- deferred issue = `RC1-ENV-01`;
- next action = `Awaiting ChatGPT audit. Do not start Task 8B.2-R2.`

If `handoff/PROJECT_STATE.md` has a clearly existing field for the latest engineering/task status, update only that field/section to record Task 8B.2-R1 objectively.

Do not alter:
- frozen research verdicts;
- Task 7J results;
- test-consumption status;
- Task 6N/6O scientific conclusions.

If no clearly appropriate engineering-status field exists, leave `handoff/PROJECT_STATE.md` unchanged.

---

# 15. Phase M — Commit content check

Run:

```bat
git status --short
git diff --check
```

Stage files individually.

Do not use:

```bat
git add .
git add -A
```

Before commit, inspect staged paths:

```bat
git diff --cached --name-only
```

Every staged path must belong to the allowed set in Phase J.

Also verify no staged file exceeds 10 MiB.

If any violation exists:

STOP.

---

# 16. Phase N — Commit

Commit once with exactly:

```text
chore(rc1): track canonical delivery source
```

After commit:

```bat
git status --short
git show --stat --oneline HEAD
```

Working tree must be clean.

If not clean:

STOP and do not push.

---

# 17. Phase O — Push

Push exactly the current branch:

```bat
git push -u origin audit/task8b2-rc1-runtime-closure
```

No force push.

After push:

```bat
git rev-parse HEAD
git status --short
```

Record commit SHA.

---

# 18. Completion gate

Only declare `TASK 8B.2-R1 COMPLETE` if all are true:

1. correct existing branch used;
2. no unauthorized starting changes;
3. current RC1 delivery inventory captured;
4. lightweight canonical snapshot created at the exact frozen path;
5. every copied source/config file verified byte-for-byte and SHA256;
6. no model weights/assets/images/logs/caches committed;
7. source manifest created;
8. canonical-source notice created;
9. deterministic sync helper created;
10. sync-helper tests pass;
11. external RC1 `--check` passes without modifying delivery;
12. repository test suite passes;
13. report created;
14. handoff updated;
15. no runtime behavior fix performed;
16. no package install performed;
17. one commit created with exact message;
18. branch pushed successfully;
19. working tree clean;
20. DSH stops and waits for ChatGPT.

If any item is not satisfied:

declare `TASK 8B.2-R1 PARTIAL`.

Do not advance to R2.

---

# 19. Final DSH reply

Output only a concise Chinese summary in this structure:

```text
TASK 8B.2-R1 COMPLETE / PARTIAL

Branch:
audit/task8b2-rc1-runtime-closure

Commit:
<sha or NONE>

Push:
PASS / NOT DONE

Canonical source:
delivery_src/BuildReasonSeg_Advisor_RC1

Snapshot:
<file count> files / <bytes>

Source↔canonical verification:
PASS / FAIL / NOT RUN

External delivery --check:
PASS / FAIL / NOT RUN

Dedicated tests:
PASS / FAIL / NOT RUN

Repository suite:
PASS / FAIL / NOT RUN

RC1-ENV-01:
DEFERRED — no runtime fix performed

Report:
docs/task8b2_r1_rc1_canonical_source.md / NOT CREATED

Handoff:
handoff/FROM_DSH.md / NOT UPDATED

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得进入 Task 8B.2-R2、Task 8C、外部图片测试或任何新研发。
```

Then stop.
