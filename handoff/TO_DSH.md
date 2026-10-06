# TO_DSH — MASK01_D1_R1D_SOURCE_MANIFEST_CANONICALIZATION

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1c`
> Required starting HEAD: `3bf9dafac69175d36a1abb6a83d019ef118ca72c`
> New task branch: `fix/task8b3-mask01-success-semantics-r1d`

## 0. CHATGPT DECISION

R1-C is accepted and CLOSED.

Accepted remote state:

```text
branch = fix/task8b3-mask01-success-semantics-r1c
HEAD   = 3bf9dafac69175d36a1abb6a83d019ef118ca72c
parent = 60cacc8a2d4460740ad9849fef00f506ad951549
commit = test(rc1): strengthen runtime success semantics contract
```

Accepted R1-C evidence:

```text
old tautological assertion = removed
success_semantics() exact dictionary = behaviorally tested
PipelineResult.ok == (status == "SUCCESS") = behaviorally tested
targeted gate = 2/2 PASS
product source = unchanged
CLI tests = unchanged
canonical docs = unchanged
```

This task is R1-D: canonicalize `source_manifest.json` against the CURRENT GIT HEAD.

---

## 1. PURPOSE

Update only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

so that every existing manifest-listed file has the exact:

```text
bytes
sha256
```

of its canonical Git blob at the starting/current HEAD:

```text
3bf9dafac69175d36a1abb6a83d019ef118ca72c
```

The manifest already declares:

```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

Therefore the identity source is **Git canonical blob bytes**, not Windows working-tree bytes and not line-ending-expanded bytes.

Do NOT hash ordinary working-tree file reads for manifest identity.

---

## 2. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1c

HEAD =
3bf9dafac69175d36a1abb6a83d019ef118ca72c
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

Any other pre-existing mutation:
- do not delete/reset/restore/clean/stash;
- record it;
- if it overlaps this task, STOP implementation.

Create:

```text
fix/task8b3-mask01-success-semantics-r1d
```

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED PATHS

Only canonical product file allowed to change:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Task records:

```text
docs/task8b3_mask01_d1_r1d_source_manifest_canonicalization.md
evaluation/task8b3_mask01_d1_r1d_source_manifest_canonicalization.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Explicitly forbidden:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/**
delivery_src/BuildReasonSeg_Advisor_RC1/tests/**
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/**
delivery_src/BuildReasonSeg_Advisor_RC1/inference/README.md
scripts/sync_advisor_rc1_delivery.py
configs
models
weights
external RC1
```

---

## 4. MANIFEST STRUCTURE — PRESERVE

The existing top-level manifest contract is frozen.

Preserve unchanged:

```text
schema
task
source_delivery
canonical_root
copy_policy
identity_basis
identity_basis_note
```

Preserve the existing `files` path set and path order EXACTLY.

Do NOT:
- add entries;
- remove entries;
- reorder entries;
- rename paths;
- add `source_manifest.json` to its own file list.

Only each existing entry's:

```text
bytes
sha256
```

may change.

Formatting may remain normal JSON with UTF-8 text.

---

## 5. CANONICAL IDENTITY ALGORITHM

For each existing manifest entry:

```text
entry["path"] = <relative path inside delivery_src/BuildReasonSeg_Advisor_RC1>
```

read bytes from Git, conceptually:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<entry["path"]>
```

Then set:

```text
entry["bytes"]  = exact length of returned Git blob bytes
entry["sha256"] = SHA256 of those exact Git blob bytes
```

Use the CURRENT task HEAD as the identity source.

Do not derive identity from:
- `Path.read_bytes()` on the canonical working tree;
- external RC1;
- copied files;
- normalized text;
- CRLF-expanded bytes.

Reason:

```text
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

---

## 6. IMPORTANT SCOPE FACT

R1-A/B/C changed several manifest-listed canonical files, including at least:

```text
predict.py
buildreasonseg/runtime/pipeline.py
tests/test_cli_contract.py
tests/test_task8b_runtime.py
README.md
docs/model_card.md
docs/runtime_mapping.md
inference/README.md
```

Do NOT manually update only this shortlist.

Recompute `bytes` + `sha256` for **EVERY existing manifest entry** from HEAD.

This ensures no older stale entry is missed.

---

## 7. REQUIRED SELF-CHECKS BEFORE WRITE

Before replacing values, record:

```text
entry_count_before
ordered_path_digest_before
```

The ordered path digest may be SHA256 of a UTF-8 newline-joined ordered path list.

After updating values, verify:

```text
entry_count_after == entry_count_before
ordered_path_digest_after == ordered_path_digest_before
```

Required:

```text
path_set_preserved = true
path_order_preserved = true
```

If not, STOP and do not disguise as COMPLETE.

---

## 8. ALL-ENTRY GIT-CANONICAL VALIDATION

After editing `source_manifest.json`, validate every entry against the existing canonical sync logic.

From repository root, run an equivalent of:

```python
from scripts.sync_advisor_rc1_delivery import (
    CANONICAL_ROOT,
    entry_source_bytes,
    load_manifest,
    manifest_identity_basis,
)

entries = load_manifest()
basis = manifest_identity_basis()

for entry in entries:
    entry_source_bytes(entry, basis, CANONICAL_ROOT)

print("validated", len(entries), "entries", "basis", basis)
```

This validation is important because `entry_source_bytes()` compares the manifest's bytes/hash against:

```text
git show HEAD:<canonical path>
```

Required:

```text
exit 0
basis = GIT_CANONICAL_BLOB_BYTES
all entries validated
```

Any mismatch = STOP.

Do NOT weaken or bypass `entry_source_bytes()`.

---

## 9. JSON / CONTRACT VALIDATION

Also verify:

```text
source_manifest.json parses as JSON
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
canonical_root = delivery_src/BuildReasonSeg_Advisor_RC1
identity_basis = GIT_CANONICAL_BLOB_BYTES
files is a non-empty list
all paths are unique safe relative paths
every entry has path + integer bytes + 64-char lowercase SHA256
```

Required:

```text
manifest_contract_validation = PASS
```

---

## 10. NO EXTERNAL SYNC IN R1-D

Do NOT run:

```text
scripts/sync_advisor_rc1_delivery.py --destination ...
```

Do NOT write to:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

External controlled sync is D2, after canonical R1-E closure.

R1-D is canonical manifest only.

---

## 11. NO PYTEST / MODEL WORK

Do NOT run:
- pytest;
- model inference;
- training;
- inspect-proposals;
- full suite.

R1-E owns regression execution.

Known inspect issue remains:

```text
PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

Do not fix it here.

---

## 12. EVERY OUTCOME MUST BE PUSHED

Frozen project rule:

```text
ALL TASK OUTCOMES MUST BE PERSISTED TO GITHUB
```

Whether COMPLETE / STOP / FAILED:
- update `handoff/FROM_DSH.md`;
- create report/evidence;
- commit authorized state;
- push branch.

---

## 13. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1d_source_manifest_canonicalization.md
evaluation/task8b3_mask01_d1_r1d_source_manifest_canonicalization.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report at least:

```text
task_id = MASK01_D1_R1D_SOURCE_MANIFEST_CANONICALIZATION
status

starting branch/head
task branch

identity_basis = GIT_CANONICAL_BLOB_BYTES

entry_count_before
entry_count_after
ordered_path_digest_before
ordered_path_digest_after
path_set_preserved
path_order_preserved

entries_recomputed = <all existing entries>
entries_changed = <number whose bytes and/or sha256 changed>

all_entry_git_validation = PASS/FAIL
manifest_contract_validation = PASS/FAIL

product_source_changed = false
tests_changed = false
canonical_docs_changed = false
sync_script_changed = false
external_write = false

pytest = not run
model_inference = false
training = false
inspect_proposals_changed = false

github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED

next_gate = CHATGPT_R1D_REMOTE_AUDIT
```

---

## 14. DIFF GATE

Before commit:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Only these may be staged:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b3_mask01_d1_r1d_source_manifest_canonicalization.md
evaluation/task8b3_mask01_d1_r1d_source_manifest_canonicalization.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected paths:
- do not delete;
- do not stage;
- report them.

---

## 15. COMMIT / PUSH

If all-entry validation and manifest-contract validation PASS, commit exactly:

```text
git commit -m "chore(rc1): canonicalize source manifest"
```

If STOP/FAILED occurs, still commit the truthful authorized state with a truthful STOP/FAILED message.

Push:

```text
fix/task8b3-mask01-success-semantics-r1d
```

No force push.

Reports use:

```text
final_commit_sha = POST_COMMIT_EXTERNAL_FACT
```

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

Do NOT enter R1-E.

---

## 16. ABSOLUTE PROHIBITIONS

Do NOT:
- modify canonical source code;
- modify tests;
- modify canonical docs;
- modify sync script;
- add/remove/reorder manifest paths;
- hash working-tree bytes as manifest identity;
- sync external RC1;
- run pytest;
- run model inference/training;
- fix inspect-proposals;
- reset/rebase/amend/stash/clean/force-push;
- start R1-E or D2.

---

## 17. SUCCESS DEFINITION

```text
MASK01_D1_R1D_SOURCE_MANIFEST_CANONICALIZATION = COMPLETE

source_manifest.json = only canonical delivery file changed

identity basis = GIT_CANONICAL_BLOB_BYTES
all existing entries recomputed from current HEAD Git blobs

entry count = preserved
path set = preserved
path order = preserved

all-entry Git canonical validation = PASS
manifest contract validation = PASS

product source = unchanged
tests = unchanged
canonical docs = unchanged
external RC1 = unchanged

NEXT = CHATGPT_R1D_REMOTE_AUDIT
```

Then STOP.
