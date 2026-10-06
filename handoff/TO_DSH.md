# TO_DSH — MASK01_D2_EXTERNAL_RC1_SYNC_AND_FULL_REGRESSION

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `audit/task8b3-mask01-r1e2-r2-remaining-failure-disambiguation`
> Required starting HEAD: `8ab53f3df2664a9ad8f9f6bc8b5659a2251b0e03`
> New task branch: `delivery/task8b3-mask01-d2-external-sync-regression`
> External RC1 target: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

## 0. CHATGPT FINAL R1-E DECISION

R1-E2-R2 execution evidence is accepted, but two classifications are corrected by ChatGPT:

```text
tests/test_paths_and_package.py::test_required_structure
= A MISSING_FULL_DELIVERY_ASSET_OR_STRUCTURE
(reason: `inference/input` is an empty delivery directory and is not represented by the
135-file Git lightweight source manifest)

tests/test_setup_checker.py::test_real_project_check_reports_ready
= A MISSING_FULL_DELIVERY_ASSET_OR_STRUCTURE
(reason: its own stdout reports missing `inference/input`, model package, SAM2 assets,
and Qwen/ProgramHead assets)

tests/test_setup_checker.py::test_real_runtime_reports_ultralytics_and_ready
= A MISSING_FULL_DELIVERY_ASSET_OR_STRUCTURE
NOT C, because its own stdout shows:
Ultralytics runtime = OK
Transformers runtime = OK
while the complete-delivery structure/assets are missing
```

Independent canonical-policy evidence:

```text
source_manifest.copy_policy =
Task 8B.2-R1 fixed lightweight source/config policy
```

and `scripts/sync_advisor_rc1_delivery.py` explicitly states that it:
- copies only manifest-listed lightweight source/config files;
- never touches model weights, downloaded assets, images, logs or generated outputs;
- never deletes destination content.

The canonical source tree is therefore intentionally NOT a complete runnable delivery tree.

Corrected aggregate conclusion:

```text
GROUPS_1_2:
all 9 failures = complete-delivery asset prerequisites absent

GROUPS_3_5:
remaining failures = complete-delivery structure/assets or noncanonical frozen log fixtures
no MASK-01 code regression established

MASK01_FULL_SUITE_FAILURES_CAUSALLY_UNRELATED = true

FULL_SUITE_GATE_PLACEMENT = EXTERNAL_COMPLETE_RC1
```

Canonical closure:

```text
SUCCESS_SEMANTICS_HARDENING_V1 = IMPLEMENTED_CANONICAL
MASK01_CANONICAL_ENGINEERING_HARDENING = CLOSED

runtime SUCCESS scope = RUNTIME_STRUCTURAL_ONLY
semantic status = NOT_EVALUATED
semantic target correctness = NOT ESTABLISHED by SUCCESS

targeted canonical Gate A = 5/5 PASS
targeted canonical Gate B = 2/2 PASS
manifest Gate C = PASS
```

This task is D2: controlled canonical → external synchronization, then complete external RC1 regression.

---

## 1. GIT PREFLIGHT

Verify exactly:

```text
current branch =
audit/task8b3-mask01-r1e2-r2-remaining-failure-disambiguation

HEAD =
8ab53f3df2664a9ad8f9f6bc8b5659a2251b0e03
```

Create:

```text
delivery/task8b3-mask01-d2-external-sync-regression
```

Allowed initial working tree:
- clean, or
- only `M handoff/TO_DSH.md`.

No reset/rebase/amend/stash/clean/force-push.

---

## 2. REPOSITORY WRITE SCOPE

DO NOT modify any canonical source/test/doc/manifest file.

Only repository task records may change:

```text
docs/task8b3_mask01_d2_external_rc1_sync_and_full_regression.md
evaluation/task8b3_mask01_d2_external_rc1_sync_and_full_regression.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Canonical tree remains frozen.

The ONLY non-repository write authorized is the controlled sync to:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

through the existing sync script.

Do NOT manually copy or edit external source files.

---

## 3. EXTERNAL COMPLETE-DELIVERY PREFLIGHT

Before any sync, verify the external root exists:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

If the root does not exist:
- STOP;
- do not create it;
- persist report and push.

Before sync, inventory these complete-delivery prerequisites.

### Required files / assets

```text
VERSION
setup_env.bat

model/buildreasonseg_advisor/decoder.pt
model/buildreasonseg_advisor/detector.pt

model/components/sam2/sam2.1_hiera_base_plus.pt
model/components/sam2/sam2.1_hiera_b+.yaml

model/components/program_head/program_parser_l3_rehearsal_v1.pt
model/components/program_head/Qwen3-VL-2B-Instruct/model.safetensors

logs/task8b1_prompt_suite.json
logs/task8b_gates.json
```

### Required directories

```text
inference/input
inference/output/masks
inference/output/overlays
inference/output/diagnostics
runs/train
runs/eval
logs
```

Record for every item:

```text
exists
bytes (for files where practical)
```

For these preserved external assets, also record pre-sync SHA256:

```text
model/buildreasonseg_advisor/decoder.pt
model/buildreasonseg_advisor/detector.pt
model/components/sam2/sam2.1_hiera_base_plus.pt
model/components/program_head/program_parser_l3_rehearsal_v1.pt
logs/task8b1_prompt_suite.json
logs/task8b_gates.json
```

For the Qwen `model.safetensors`, record:

```text
exists
bytes
mtime
```

Do not hash multi-GB Qwen weights solely for this task.

If ANY required prerequisite above is missing:
- status = STOP;
- classification = EXTERNAL_DELIVERY_INCOMPLETE_PREEXISTING;
- do NOT run sync;
- do NOT create/download the missing item;
- commit/push truthful evidence;
- await ChatGPT.

---

## 4. GATE D2-A — PRE-SYNC SOURCE COMPARISON

From repository root, run:

```text
python scripts/sync_advisor_rc1_delivery.py --destination "C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1" --check
```

This is read-only.

A non-zero result BEFORE sync is allowed and expected if external source/config is stale.

Record exactly:

```text
exit
checked
match
missing
mismatch
```

Do NOT classify a pre-sync mismatch as task failure.

---

## 5. GATE D2-B — CONTROLLED SYNC

Run exactly:

```text
python scripts/sync_advisor_rc1_delivery.py --destination "C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1"
```

Required:

```text
copied = 135
verified = 135
failures = 0
exit = 0
```

The script is the only authorized mechanism for source/config writes.

If failure:
- do not manually repair;
- STOP after evidence persistence.

---

## 6. GATE D2-C — POST-SYNC SOURCE IDENTITY

Run exactly:

```text
python scripts/sync_advisor_rc1_delivery.py --destination "C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1" --check
```

Required:

```text
checked = 135
match = 135
missing = 0
mismatch = 0
exit = 0
```

This is the external source/config identity gate.

---

## 7. PRESERVED-ASSET INTEGRITY

After sync, recompute the same SHA256 values for:

```text
decoder.pt
detector.pt
sam2.1_hiera_base_plus.pt
program_parser_l3_rehearsal_v1.pt
logs/task8b1_prompt_suite.json
logs/task8b_gates.json
```

Required:

```text
pre_sha256 == post_sha256
```

For Qwen `model.safetensors`, required:

```text
exists after sync = true
bytes after == bytes before
mtime after == mtime before
```

This demonstrates that the lightweight sync did not touch preserved external assets/fixtures.

Any unexpected mutation:
- STOP;
- no manual restoration;
- report.

---

## 8. GATE D2-D — EXTERNAL SETUP READY

Working directory:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Run:

```text
python check_setup.py
```

Required:

```text
exit = 0
BuildReasonSeg environment: READY
```

Also record lines for:

```text
model package
SAM2 assets
Qwen / ProgramHead assets
Ultralytics runtime
Transformers runtime
```

They must be READY/OK.

If NOT READY:
- do not repair/install/download;
- STOP and report exact reason.

---

## 9. GATE D2-E — TARGETED MASK-01 CONTRACTS ON EXTERNAL

From external RC1 root run exactly:

```text
python -m pytest tests/test_cli_contract.py::test_r1a_single_success_semantics_block tests/test_cli_contract.py::test_r1a_single_failure_omits_success_semantics tests/test_cli_contract.py::test_r1a_batch_success_annotation_and_runtime_summary tests/test_task8b_runtime.py::test_success_semantics_contract_exact tests/test_task8b_runtime.py::test_pipeline_result_ok_is_status_compatibility -q
```

Required:

```text
5 passed
exit 0
```

Then run:

```text
python -m pytest tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt tests/test_cli_contract.py::test_inspect_unreadable_image_precedes_model_resolution -q
```

Required:

```text
2 passed
exit 0
```

No repairs if either fails.

---

## 10. GATE D2-F — COMPLETE EXTERNAL RC1 PYTEST SUITE

Still in external RC1 root, run exactly:

```text
python -m pytest -q
```

Required for COMPLETE:

```text
exit = 0
0 failed
0 errors
all collected tests pass
```

Record:
- passed;
- skipped;
- xfailed;
- failed;
- errors;
- duration;
- exact failing nodes if any.

Do NOT use:
- `-k`;
- `--lf`;
- test exclusion;
- skip/xfail edits;
- source/test repairs.

If this gate fails:
- D2 status = STOP;
- persist the complete failure list;
- do NOT modify external or canonical code;
- await ChatGPT.

---

## 11. NO MODEL INFERENCE / TRAINING

Do NOT:
- run real prediction on an image;
- run locked cases;
- train;
- tune;
- change model assets;
- change detector/reference/mask algorithms.

`check_setup.py` and pytest contract/package checks are allowed.

This D2 task is engineering synchronization/regression only.

---

## 12. EXTERNAL POST-REGRESSION INVENTORY

After all test gates, re-check:

```text
decoder.pt
detector.pt
SAM2 checkpoint
ProgramHead checkpoint
Qwen model.safetensors
logs/task8b1_prompt_suite.json
logs/task8b_gates.json
```

They must remain present.

Do not stage or copy external generated pytest cache/log output into Git.

---

## 13. REQUIRED D2 DECISION

If D2-B/C/D/E/F all PASS and preserved assets remain unchanged:

```text
MASK01_D2_EXTERNAL_RC1_SYNC_AND_FULL_REGRESSION = COMPLETE

EXTERNAL_RC1_SOURCE_IDENTITY = 135/135 MATCH
EXTERNAL_RC1_SETUP = READY
EXTERNAL_MASK01_TARGETED_CONTRACTS = PASS
EXTERNAL_RC1_FULL_SUITE = PASS

SUCCESS_SEMANTICS_HARDENING_V1 = IMPLEMENTED_EXTERNAL_RC1
MASK01_ENGINEERING_HARDENING_CHAIN = CLOSED

semantic target correctness = NOT ESTABLISHED by runtime SUCCESS
```

Do NOT claim the scientific segmentation problem is solved.

If any gate fails:

```text
MASK01_D2_EXTERNAL_RC1_SYNC_AND_FULL_REGRESSION = STOP
MASK01_ENGINEERING_HARDENING_CHAIN = CANONICAL_CLOSED_EXTERNAL_PENDING
```

and record exact blocker.

---

## 14. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d2_external_rc1_sync_and_full_regression.md
evaluation/task8b3_mask01_d2_external_rc1_sync_and_full_regression.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Required report fields:

```text
task_id = MASK01_D2_EXTERNAL_RC1_SYNC_AND_FULL_REGRESSION
status

starting branch/head
task branch
external_root

pre_sync_prerequisite_inventory
pre_sync_preserved_asset_hashes

D2-A pre-sync check command/result
D2-B sync command/result
D2-C post-sync check command/result

post_sync_preserved_asset_hashes
preserved_assets_unchanged = true/false

D2-D check_setup exit/result
D2-E targeted success-semantics result
D2-E targeted inspect result
D2-F full suite result

external_source_match = 135/135 or other
external_setup_ready = true/false
external_full_suite_pass = true/false

canonical_files_modified = false
model_assets_modified_by_sync = false
model_inference = false
training = false

semantic_target_correctness =
NOT_ESTABLISHED_BY_RUNTIME_SUCCESS

github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_D2_REMOTE_AUDIT
```

---

## 15. REPOSITORY WORKTREE AUDIT

After external work:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Only authorized report/handoff files may be staged.

No canonical delivery file may change in the repository.

---

## 16. EVERY OUTCOME MUST BE PUSHED

COMPLETE / STOP / FAILED must all be committed and pushed.

---

## 17. COMMIT / PUSH

If COMPLETE, commit exactly:

```text
git commit -m "test(rc1): verify external mask01 delivery closure"
```

If STOP/FAILED, use a truthful status commit message.

Push:

```text
delivery/task8b3-mask01-d2-external-sync-regression
```

No force push.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

---

## 18. SUCCESS DEFINITION

```text
MASK01_D2_EXTERNAL_RC1_SYNC_AND_FULL_REGRESSION = COMPLETE

external complete-delivery prerequisites = present
135 lightweight source/config files = synced and verified
preserved weights/assets/fixtures = unchanged
check_setup = READY
MASK-01 targeted contracts = 7/7 PASS
complete external pytest suite = ALL PASS

SUCCESS_SEMANTICS_HARDENING_V1 = IMPLEMENTED_EXTERNAL_RC1
MASK01_ENGINEERING_HARDENING_CHAIN = CLOSED

NEXT = CHATGPT_D2_REMOTE_AUDIT
```

Then STOP.
