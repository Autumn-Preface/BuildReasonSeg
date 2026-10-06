# TO_DSH — MASK01_D1_R1D2_POST_INSPECT_MANIFEST_RECANONICALIZATION

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/rc1-inspect-proposals-preflight-order`
> Required starting HEAD: `efc48eb5ee8b0851a84e8a3d95533f1acd23cacf`
> New task branch: `fix/task8b3-mask01-success-semantics-r1d2`

## 0. CHATGPT DECISION

`RC1_INSPECT01_PREFLIGHT_ORDER_FIX` is ACCEPTED / CLOSED.

Accepted remote state:

```text
branch = fix/rc1-inspect-proposals-preflight-order
HEAD   = efc48eb5ee8b0851a84e8a3d95533f1acd23cacf
parent = 5808e64ed43fd383be9e08bf623a9bfdc27f62b4
commit = fix(rc1): validate inspect image before model setup
```

Accepted facts:

```text
inspect-only load_image() preflight = implemented before model resolution/runtime setup
normal inference ordering = unchanged
existing inspect E202/20 contract test = PASS
new inspect ordering regression = PASS
Gate A = 2/2 PASS
Gate B = 4/4 PASS
runtime pipeline/imageio/detector/docs = unchanged
```

The previous R1-D manifest is now intentionally stale because this accepted task changed two manifest-listed files:

```text
predict.py
tests/test_cli_contract.py
```

This task is a manifest-only recanonicalization before R1-E.

---

## 1. PURPOSE

Update only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

so all existing 135 entries again match the exact Git canonical blob bytes at:

```text
efc48eb5ee8b0851a84e8a3d95533f1acd23cacf
```

Frozen identity basis:

```text
GIT_CANONICAL_BLOB_BYTES
```

Do NOT use Windows working-tree bytes.

---

## 2. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/rc1-inspect-proposals-preflight-order

HEAD =
efc48eb5ee8b0851a84e8a3d95533f1acd23cacf
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

Unexpected mutations:
- do not delete/reset/restore/stash/clean;
- report them;
- STOP if they overlap authorized paths.

Create:

```text
fix/task8b3-mask01-success-semantics-r1d2
```

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED PATHS

Only canonical delivery file:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Task records:

```text
docs/task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.md
evaluation/task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Do NOT modify:
- `predict.py`;
- any runtime source;
- any tests;
- canonical docs;
- sync script;
- configs/models/weights;
- external RC1.

---

## 4. PRESERVE MANIFEST STRUCTURE

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

Preserve exactly:
- entry count;
- path set;
- path order.

Do NOT:
- add entries;
- remove entries;
- reorder entries;
- rename entries;
- add `source_manifest.json` to itself.

Only existing entries':

```text
bytes
sha256
```

may change.

Expected current entry count:

```text
135
```

---

## 5. RECOMPUTE ALL ENTRIES FROM GIT

For EVERY existing manifest entry, use canonical bytes equivalent to:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<entry path>
```

Set:

```text
bytes  = exact Git blob byte length
sha256 = SHA256 of exact Git blob bytes
```

Do not manually update only `predict.py` and `tests/test_cli_contract.py`.

Recompute all 135 entries, then report how many actually changed.

Expected changed entries are likely the two accepted INSPECT01 files, but COMPLETE status must come from all-entry validation, not this expectation.

---

## 6. PATH PRESERVATION CHECK

Record before and after:

```text
entry_count
ordered_path_digest
```

Compute `ordered_path_digest` deterministically, e.g. SHA256 of the UTF-8 newline-joined ordered path list.

Required:

```text
entry_count_before = 135
entry_count_after = 135
ordered_path_digest_before == ordered_path_digest_after
path_set_preserved = true
path_order_preserved = true
```

This time include the actual digest values in report/evidence.

---

## 7. ALL-ENTRY GIT VALIDATION

Use the existing canonical helper from:

```text
scripts/sync_advisor_rc1_delivery.py
```

Validate every entry using:

```text
manifest_identity_basis()
load_manifest()
entry_source_bytes(...)
```

Required:

```text
basis = GIT_CANONICAL_BLOB_BYTES
entries validated = 135
failures = 0
PASS
```

Do not weaken/bypass the helper.

---

## 8. MANIFEST CONTRACT VALIDATION

Require:

```text
JSON parse = PASS
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
canonical_root = delivery_src/BuildReasonSeg_Advisor_RC1
identity_basis = GIT_CANONICAL_BLOB_BYTES
files count = 135
all paths unique = true
all paths safe relative paths = true
all entries have integer bytes = true
all sha256 values are lowercase 64-char hex = true
```

Required:

```text
manifest_contract_validation = PASS
```

---

## 9. NO PYTEST / EXTERNAL SYNC

Do NOT run:
- pytest;
- model inference;
- training;
- detector;
- inspect-proposals;
- full suite.

Do NOT sync/write external RC1.

R1-E is the next canonical regression gate.

---

## 10. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.md
evaluation/task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report:

```text
task_id = MASK01_D1_R1D2_POST_INSPECT_MANIFEST_RECANONICALIZATION
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

entries_recomputed = 135
entries_changed = <n>
changed_entry_paths = [...]

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

github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_R1D2_REMOTE_AUDIT
```

---

## 11. EVERY OUTCOME MUST BE PUSHED

Whether COMPLETE / STOP / FAILED:
- update FROM_DSH;
- create report/evidence;
- commit authorized state;
- push task branch.

No result may remain local-only.

---

## 12. DIFF GATE

Only stage:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
docs/task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.md
evaluation/task8b3_mask01_d1_r1d2_post_inspect_manifest_recanonicalization.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected paths:
- do not delete;
- do not stage;
- report.

---

## 13. COMMIT / PUSH

If validations pass, commit exactly:

```text
git commit -m "chore(rc1): recanonicalize source manifest after inspect fix"
```

Push:

```text
fix/task8b3-mask01-success-semantics-r1d2
```

No force push.

If STOP/FAILED, still persist truthful authorized state.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

Do NOT enter R1-E.

---

## 14. SUCCESS DEFINITION

```text
MASK01_D1_R1D2_POST_INSPECT_MANIFEST_RECANONICALIZATION = COMPLETE

manifest entries = 135
path set/order = preserved
actual ordered path digests recorded and equal

all 135 entries recomputed from Git canonical blob bytes
all-entry Git validation = PASS
manifest contract validation = PASS

product source = unchanged
tests = unchanged
docs = unchanged
external RC1 = unchanged

canonical tree + manifest are ready for R1-E

NEXT = CHATGPT_R1D2_REMOTE_AUDIT
```

Then STOP.
