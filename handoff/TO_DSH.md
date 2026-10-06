# TO_DSH — MASK01_D1_R1E2_FULL_SUITE_FAILURE_TRIAGE

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1e1`
> Required starting HEAD: `3320212a6992316196bc5b2cfacaeaf027fd070e`
> New task branch: `audit/task8b3-mask01-r1e2-full-suite-failure-triage`

## 0. FROZEN STATE

R1-E1 is ACCEPTED for its targeted canonical scope.

Verified:

```text
Gate A SUCCESS semantics = 5/5 PASS
Gate B inspect-proposals contract = 2/2 PASS
Gate C source manifest Git identity = PASS

basis = GIT_CANONICAL_BLOB_BYTES
entries = 135
ordered_path_digest =
7967127fcfadfe4f7f19a3511163c9f77476673774dcac6e8f5a7e49ea20a7bd
```

Therefore these three contracts are established:

```text
SUCCESS_SEMANTICS_HARDENING_TARGETED_CONTRACT = PASS
INSPECT_PROPOSALS_ERROR_ORDER_CONTRACT = PASS
SOURCE_MANIFEST_CANONICAL_IDENTITY = PASS
```

However, the earlier full suite on the canonical source tree produced:

```text
16 failed
108 passed
6 errors
```

The failing nodes cluster around:
- model package/assets;
- setup checker / runtime readiness;
- paths / required delivery structure;
- language Qwen availability;
- Task 8B.1 frozen log fixtures.

The canonical source manifest is explicitly a lightweight source/config snapshot, while several of those tests appear to require complete delivery assets and fixtures.

This task performs READ-ONLY FAILURE TRIAGE ONLY.

No repair is authorized.

---

## 1. PURPOSE

Determine whether each previous full-suite failure belongs to one of these categories:

```text
A = MISSING_FULL_DELIVERY_ASSET
B = MISSING_NONCANONICAL_TEST_FIXTURE
C = RUNTIME_DEPENDENCY_ENVIRONMENT
D = TRUE_CANONICAL_CODE_REGRESSION
E = OTHER
```

The goal is NOT to make tests pass.

The goal is to decide whether the complete pytest suite is a valid gate for the lightweight canonical source tree, or belongs on the external complete RC1 delivery after D2 sync.

---

## 2. GIT PREFLIGHT

Verify:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1e1

HEAD =
3320212a6992316196bc5b2cfacaeaf027fd070e
```

Create:

```text
audit/task8b3-mask01-r1e2-full-suite-failure-triage
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED CHANGES

Only:

```text
docs/task8b3_mask01_d1_r1e2_full_suite_failure_triage.md
evaluation/task8b3_mask01_d1_r1e2_full_suite_failure_triage.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

DO NOT MODIFY ANYTHING UNDER:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

Do not modify external RC1.

Generated caches/logs must not be staged.

---

## 4. DO NOT RUN THE COMPLETE SUITE AGAIN

Explicitly forbidden:

```text
python -m pytest -q
```

We already have that result.

This task runs only the previously failing groups with verbose failure output.

Do not use `-k`.
Do not repair.
Do not add skips/xfails.

---

## 5. INVENTORY CHECK — REQUIRED FIRST

From repository root, inspect the canonical source tree and its source manifest.

Run a read-only Python command/script that reports existence AND whether each path is listed in `source_manifest.json` for:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/model/buildreasonseg_advisor/decoder.pt
delivery_src/BuildReasonSeg_Advisor_RC1/model/buildreasonseg_advisor/detector.pt
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/sam2/sam2.1_hiera_base_plus.pt
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/program_parser_l3_rehearsal_v1.pt
delivery_src/BuildReasonSeg_Advisor_RC1/model/components/program_head/Qwen3-VL-2B-Instruct/model.safetensors
delivery_src/BuildReasonSeg_Advisor_RC1/logs/task8b1_prompt_suite.json
delivery_src/BuildReasonSeg_Advisor_RC1/logs/task8b_gates.json
delivery_src/BuildReasonSeg_Advisor_RC1/setup_env.bat
```

Also report existence of these directories:

```text
model/components/sam2
logs
runs/train
runs/eval
inference/output/masks
inference/output/overlays
inference/output/diagnostics
```

Do not create missing paths.

Record:

```text
exists = true/false
manifest_listed = true/false
```

for files.

---

## 6. TRIAGE GROUP 1 — LANGUAGE AVAILABILITY

Working directory:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

Run exactly:

```text
python -m pytest tests/test_language_contract.py::test_normal_path_is_qwen_first -vv
```

Record:
- exit;
- assertion/error;
- exact reason returned by `frontend.available()`;
- classification A/B/C/D/E.

No repair.

---

## 7. TRIAGE GROUP 2 — MODEL PACKAGE

Run exactly these previously failing nodes:

```text
python -m pytest tests/test_model_package.py::test_components_complete tests/test_model_package.py::test_decoder_hash_and_bytes_exact tests/test_model_package.py::test_detector_hash_and_bytes_exact tests/test_model_package.py::test_hash_mismatch_raises tests/test_model_package.py::test_metadata_hashes_match_copied_assets tests/test_model_package.py::test_model_fallback_requires_user_confirmation tests/test_model_package.py::test_model_yaml_parses tests/test_model_package.py::test_package_verification_matches -vv
```

For EACH failing node record:
- first decisive failure line;
- missing path / hash mismatch / config error / other;
- classification A/B/C/D/E.

Do not modify files or model assets.

---

## 8. TRIAGE GROUP 3 — PATHS / DELIVERY STRUCTURE

Run exactly:

```text
python -m pytest tests/test_paths_and_package.py::test_moving_root_keeps_config_and_model_paths tests/test_paths_and_package.py::test_required_structure -vv
```

For each failure record:
- exact missing file/dir or assertion;
- whether that item is expected to exist in the 135-entry source manifest;
- classification A/B/C/D/E.

---

## 9. TRIAGE GROUP 4 — SETUP CHECKER

Run exactly:

```text
python -m pytest tests/test_setup_checker.py::test_good_fixture_is_ready tests/test_setup_checker.py::test_hash_mismatch_nonzero tests/test_setup_checker.py::test_invalid_model_yaml_nonzero tests/test_setup_checker.py::test_missing_decoder_nonzero tests/test_setup_checker.py::test_missing_qwen_component_nonzero tests/test_setup_checker.py::test_missing_sam2_nonzero tests/test_setup_checker.py::test_real_project_check_reports_ready tests/test_setup_checker.py::test_real_runtime_reports_ultralytics_and_ready -vv
```

For each failing/error node record:
- whether failure occurs during fixture construction or actual test assertion;
- decisive missing file/dir/runtime dependency;
- classification A/B/C/D/E.

Do NOT install packages.
Do NOT create missing assets.

---

## 10. TRIAGE GROUP 5 — TASK 8B.1 FIXTURES

Run exactly:

```text
python -m pytest tests/test_task8b1_fallback_ux.py::test_fixture_frozen_and_complete tests/test_task8b1_fallback_ux.py::test_no_prompt_to_program_table_in_delivery_code tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim -vv
```

Record:
- exact missing fixture or other failure;
- whether that fixture is included in source_manifest.json;
- classification A/B/C/D/E.

---

## 11. REQUIRED AGGREGATE CLASSIFICATION

Build a table containing every previously failing/error node from the R1-E STOP snapshot.

Columns:

```text
node
reproduced_status
decisive_reason
required_path_or_dependency
exists_in_canonical_tree
listed_in_source_manifest
classification
mask01_change_related
```

`mask01_change_related` must be one of:

```text
YES
NO
UNRESOLVED
```

Do NOT mark `NO` merely because a file is missing.
Explain causality from the actual failure.

---

## 12. VALIDITY DECISION

At the end, choose exactly one:

### Decision 1

```text
FULL_SUITE_CANONICAL_GATE = INVALID_FOR_LIGHTWEIGHT_SOURCE_TREE
```

Use only if evidence establishes that all reproduced failures/errors are caused by intentionally absent complete-delivery assets, noncanonical fixtures, or runtime-environment requirements rather than canonical code regressions.

Then recommend:

```text
R1 canonical closure basis = targeted A/B/C contracts
full complete suite gate = move to D2 external complete RC1
```

### Decision 2

```text
FULL_SUITE_CANONICAL_GATE = VALID_AND_HAS_TRUE_REGRESSION
```

Use if at least one failure is a genuine canonical code/test regression independent of absent delivery assets/environment.

Identify those nodes.

### Decision 3

```text
FULL_SUITE_CANONICAL_GATE = UNRESOLVED
```

Use if evidence is insufficient.

Do NOT make any repairs in all three cases.

---

## 13. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1e2_full_suite_failure_triage.md
evaluation/task8b3_mask01_d1_r1e2_full_suite_failure_triage.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report:

```text
task_id = MASK01_D1_R1E2_FULL_SUITE_FAILURE_TRIAGE
status

starting branch/head
task branch

canonical_inventory = {...}

group1 result
group2 result
group3 result
group4 result
group5 result

per_node_classification = [...]

classification_counts:
A =
B =
C =
D =
E =

mask01_related:
YES =
NO =
UNRESOLVED =

full_suite_canonical_gate_decision =
INVALID_FOR_LIGHTWEIGHT_SOURCE_TREE
or VALID_AND_HAS_TRUE_REGRESSION
or UNRESOLVED

repairs_attempted = false
canonical_tracked_files_modified = false
full_suite_reexecuted = false
model_inference = false
training = false
external_write = false

next_gate = CHATGPT_R1E2_REMOTE_AUDIT
```

---

## 14. WORKTREE AUDIT

After diagnostics:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Do not stage generated files outside authorized task records.
Do not use `git clean`.

---

## 15. EVERY OUTCOME MUST BE PUSHED

Whether COMPLETE / STOP / FAILED:
- persist report/evidence/FROM_DSH;
- commit only authorized task records;
- push branch.

---

## 16. COMMIT / PUSH

Commit:

```text
git commit -m "docs(rc1): classify canonical full-suite failures"
```

Push:

```text
audit/task8b3-mask01-r1e2-full-suite-failure-triage
```

No force push.

Then STOP.

---

## 17. SUCCESS DEFINITION

```text
MASK01_D1_R1E2_FULL_SUITE_FAILURE_TRIAGE = COMPLETE

all five failing groups diagnostically rerun
all previous failing/error nodes classified
no repair attempted
canonical files unchanged

full-suite gate validity decided with evidence

NEXT = CHATGPT_R1E2_REMOTE_AUDIT
```

Then STOP.
