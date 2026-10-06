# TO_DSH — MASK01_D1_R1E2_R2_REMAINING_FAILURE_DISAMBIGUATION

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation`
> Required starting HEAD: `7251604686da5846d5bbef28048b8a7454bc4181`
> New task branch: `audit/task8b3-mask01-r1e2-r2-remaining-failure-disambiguation`

## 0. CHATGPT CORRECTION

R1-E2-R1 execution evidence is accepted, but its classification is rejected.

Independent source audit of:

```text
buildreasonseg/models/package.py
```

establishes that all 9 R1-E2-R1 nodes fail because the lightweight canonical source tree intentionally lacks complete-delivery model assets.

Corrected classification:

```text
tests/test_language_contract.py::test_normal_path_is_qwen_first
= A MISSING_FULL_DELIVERY_ASSET
(reason: missing program_parser_l3_rehearsal_v1.pt + Qwen3-VL-2B-Instruct)

tests/test_model_package.py::test_components_complete
= A

tests/test_model_package.py::test_decoder_hash_and_bytes_exact
= A (decoder.pt absent)

tests/test_model_package.py::test_detector_hash_and_bytes_exact
= A (detector.pt absent)

tests/test_model_package.py::test_hash_mismatch_raises
= A (ModelPackage.load fails first because package assets absent)

tests/test_model_package.py::test_metadata_hashes_match_copied_assets
= A (metadata target decoder.pt absent)

tests/test_model_package.py::test_model_fallback_requires_user_confirmation
= A
(reason: default_package_available() is false because default package assets are absent,
so the user-confirmation fallback branch is not reachable)

tests/test_model_package.py::test_model_yaml_parses
= A (ModelPackage.load fails first because decoder/detector absent)

tests/test_model_package.py::test_package_verification_matches
= A
```

Therefore:

```text
R1E2_R1_TRUE_REGRESSION_COUNT = 0
PREVIOUS_TRUE_REGRESSION_CLASSIFICATION = REFUTED_FOR_GROUPS_1_2
```

This task individually disambiguates the remaining REAL failing nodes from Groups 3–5.

No repair is authorized.

---

## 1. GIT PREFLIGHT

Verify:

```text
current branch =
audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation

HEAD =
7251604686da5846d5bbef28048b8a7454bc4181
```

Create:

```text
audit/task8b3-mask01-r1e2-r2-remaining-failure-disambiguation
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

No reset/rebase/amend/stash/clean/force-push.

---

## 2. ALLOWED CHANGES

Only:

```text
docs/task8b3_mask01_d1_r1e2_r2_remaining_failure_disambiguation.md
evaluation/task8b3_mask01_d1_r1e2_r2_remaining_failure_disambiguation.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

DO NOT modify anything under:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

Do not create assets/fixtures/directories.
Do not install dependencies.
Do not write external RC1.

---

## 3. CLASSIFICATION RULE

For each real node choose exactly one:

```text
A = MISSING_FULL_DELIVERY_ASSET_OR_STRUCTURE
B = MISSING_NONCANONICAL_TEST_FIXTURE
C = RUNTIME_DEPENDENCY_ENVIRONMENT
D = TRUE_CANONICAL_CODE_REGRESSION
E = OTHER
PASS = TEST_PASSES_INDIVIDUALLY
```

Use `D` only when the node can reach its intended assertion using only files/environment that the lightweight canonical source tree is actually required to contain, and current canonical code violates that contract.

Do NOT label:
- missing empty delivery directory;
- missing external model asset;
- missing logs/task fixture;
as D.

---

## 4. EXECUTION METHOD

Working directory:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

Run EXACTLY ONE node per invocation:

```text
python -m pytest <NODE_ID> -vv
```

Never batch nodes.

### Group 3 — Paths / delivery structure

1.
```text
tests/test_paths_and_package.py::test_moving_root_keeps_config_and_model_paths
```

2.
```text
tests/test_paths_and_package.py::test_required_structure
```

### Group 4 — Setup checker

3.
```text
tests/test_setup_checker.py::test_good_fixture_is_ready
```

4.
```text
tests/test_setup_checker.py::test_hash_mismatch_nonzero
```

5.
```text
tests/test_setup_checker.py::test_invalid_model_yaml_nonzero
```

6.
```text
tests/test_setup_checker.py::test_missing_decoder_nonzero
```

7.
```text
tests/test_setup_checker.py::test_missing_qwen_component_nonzero
```

8.
```text
tests/test_setup_checker.py::test_missing_sam2_nonzero
```

9.
```text
tests/test_setup_checker.py::test_real_project_check_reports_ready
```

10.
```text
tests/test_setup_checker.py::test_real_runtime_reports_ultralytics_and_ready
```

### Group 5 — Task 8B.1 fixtures

11.
```text
tests/test_task8b1_fallback_ux.py::test_fixture_frozen_and_complete
```

12.
```text
tests/test_task8b1_fallback_ux.py::test_no_prompt_to_program_table_in_delivery_code
```

13.
```text
tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim
```

Exactly 13 nodes.

---

## 5. REQUIRED PER-NODE EVIDENCE

For each node record:

```text
node_id
exit_code
summary
PASS_or_FAIL
exception/assertion
first_decisive_failure_line
required_missing_path_or_dependency
classification
classification_reason
```

Evidence must come from that node's own invocation.

No progress tokens.
No cross-node reason reuse.

---

## 6. SPECIAL INTERPRETATION RULES

### `test_required_structure`

If it fails only because a directory such as:

```text
inference/input
inference/output/masks
runs/train
runs/eval
```

is absent from the Git lightweight source snapshot, classify:

```text
A
```

because Git/source_manifest does not represent empty delivery directories.

### `test_moving_root_keeps_config_and_model_paths`

If it passes individually, classify PASS.

If it fails, record its own exact reason. Do not inherit `test_required_structure`'s reason.

### setup-checker fixture tests

For each fixture test, distinguish:
- fixture construction failure because source-side asset/fixture required for copying is absent;
- checker logic assertion failure after fixture construction.

If construction fails due omitted full-delivery asset/structure, classify A.

If a Python package/version is actually missing, classify C.

If fixture successfully constructs and checker behavior violates its intended synthetic contract, then and only then consider D.

### real-project setup tests

If they fail because the lightweight source tree is not a complete delivery package (missing model assets/components/directories), classify A.

### Task 8B.1 fixture tests

If `logs/task8b1_prompt_suite.json` or `logs/task8b_gates.json` is absent and not source-manifest-listed, classify B.

---

## 7. MASK-01 CAUSALITY CHECK

Use the frozen pre-MASK forensic base:

```text
a98ccecce20585dc37520523cd32b49b0a684248
```

For every node classified D, additionally identify the production/test dependency file whose current behavior causes the regression, then run a READ-ONLY Git diff check:

```text
git diff --name-only a98ccecce20585dc37520523cd32b49b0a684248..HEAD -- <dependency paths>
```

Record:

```text
changed_since_pre_mask = true/false
mask01_change_related = YES/NO/UNRESOLVED
```

A D node whose causal dependency is unchanged since the pre-MASK base cannot be attributed to this MASK repair chain; record:

```text
mask01_change_related = NO
```

Do not modify history.

---

## 8. NO FULL SUITE

Do NOT run:

```text
python -m pytest -q
```

Do not rerun Groups 1–2.

---

## 9. AGGREGATE RESULT

Record:

```text
A count
B count
C count
D count
E count
PASS count

true_regression_nodes = [...]
mask01_related_true_regressions = [...]
```

Then choose:

### Case 1

If:

```text
mask01_related_true_regressions = []
```

record:

```text
MASK01_FULL_SUITE_FAILURES_CAUSALLY_UNRELATED = true
```

This is true even if a pre-existing/stale canonical test regression exists elsewhere.

### Case 2

If at least one D node is causally related to files changed in the MASK chain:

```text
MASK01_FULL_SUITE_FAILURES_CAUSALLY_UNRELATED = false
```

List exact node/path.

---

## 10. FULL-SUITE GATE PLACEMENT RECOMMENDATION

Choose one:

```text
FULL_SUITE_GATE_PLACEMENT = EXTERNAL_COMPLETE_RC1
```

if all current failures require complete-delivery assets/fixtures/structure/environment absent from the lightweight source tree.

Or:

```text
FULL_SUITE_GATE_PLACEMENT = CANONICAL_SOURCE_TREE
```

if the remaining failures are genuine source-only regressions the lightweight tree is required to satisfy.

Or:

```text
FULL_SUITE_GATE_PLACEMENT = SPLIT
```

if source-only tests can be valid here but complete-delivery tests require D2.

For `SPLIT`, recommend:

```text
canonical lightweight gate = source-only + targeted contracts
external D2 gate = complete delivery full suite
```

---

## 11. WORKTREE AUDIT

After diagnostics:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Do not stage/delete generated files outside task records.

---

## 12. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1e2_r2_remaining_failure_disambiguation.md
evaluation/task8b3_mask01_d1_r1e2_r2_remaining_failure_disambiguation.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Required fields:

```text
task_id = MASK01_D1_R1E2_R2_REMAINING_FAILURE_DISAMBIGUATION
status

starting branch/head
task branch

nodes_requested = 13
nodes_executed_individually = 13

per_node = [...]

classification_counts = {...}
true_regression_nodes = [...]
mask01_related_true_regressions = [...]

MASK01_FULL_SUITE_FAILURES_CAUSALLY_UNRELATED = true/false

FULL_SUITE_GATE_PLACEMENT =
EXTERNAL_COMPLETE_RC1
or CANONICAL_SOURCE_TREE
or SPLIT

full_suite_reexecuted = false
canonical_files_modified = false
assets_created = false
fixtures_created = false
dependencies_installed = false
repairs_attempted = false
model_inference = false
external_write = false

next_gate = CHATGPT_R1E2_R2_REMOTE_AUDIT
```

---

## 13. EVERY OUTCOME MUST BE PUSHED

Persist COMPLETE / STOP / FAILED to GitHub.

---

## 14. COMMIT / PUSH

Commit exactly:

```text
git commit -m "docs(rc1): resolve remaining canonical suite failures"
```

Push:

```text
audit/task8b3-mask01-r1e2-r2-remaining-failure-disambiguation
```

No force push.

Then STOP.

---

## 15. SUCCESS DEFINITION

```text
MASK01_D1_R1E2_R2_REMAINING_FAILURE_DISAMBIGUATION = COMPLETE

13 real nodes run individually
all classifications based on own traceback
MASK causality established
full-suite gate placement decided
canonical files unchanged

NEXT = CHATGPT_R1E2_R2_REMOTE_AUDIT
```

Then STOP.
