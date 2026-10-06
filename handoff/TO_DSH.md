# TO_DSH — MASK01_D1_R1E2_R1_ASSET_REGRESSION_DISAMBIGUATION

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `audit/task8b3-mask01-r1e2-full-suite-failure-triage`
> Required starting HEAD: `7a4e5454cdaf164c8d1706412c55b869074748e0`
> New task branch: `audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation`

## 0. CHATGPT AUDIT

The previous R1-E2 remote state is accepted only as a persisted diagnostic snapshot.

Its final classification is NOT accepted.

Why:

1. The report treated non-node pytest tokens such as:
   ```text
   [
   [100%]
   at
   ```
   as if they were test nodes.

2. It classified:
   ```text
   test_normal_path_is_qwen_first
   ```
   as `D = TRUE_CANONICAL_CODE_REGRESSION`,
   even though the same audit established that the canonical lightweight tree does NOT contain:
   ```text
   model/components/program_head/Qwen3-VL-2B-Instruct
   ```
   and this test asserts `frontend.available() is True`.

3. It classified the complete model-package failure group as `D`,
   even though the same inventory established that these complete-delivery assets are absent and not source-manifest-listed:
   ```text
   model/buildreasonseg_advisor/decoder.pt
   model/buildreasonseg_advisor/detector.pt
   model/components/sam2/sam2.1_hiera_base_plus.pt
   model/components/program_head/program_parser_l3_rehearsal_v1.pt
   model/components/program_head/Qwen3-VL-2B-Instruct
   ```

Therefore:

```text
R1E2_STATE = ACCEPTED_AS_DIAGNOSTIC_SNAPSHOT
R1E2_CLASSIFICATION = REJECTED
FULL_SUITE_CANONICAL_GATE_DECISION = NOT_YET_DECIDED
```

This corrective task disambiguates only the nodes that were incorrectly labeled TRUE REGRESSION.

---

## 1. PURPOSE

Run each of the following REAL test nodes individually and capture its own output:

```text
tests/test_language_contract.py::test_normal_path_is_qwen_first

tests/test_model_package.py::test_components_complete
tests/test_model_package.py::test_decoder_hash_and_bytes_exact
tests/test_model_package.py::test_detector_hash_and_bytes_exact
tests/test_model_package.py::test_hash_mismatch_raises
tests/test_model_package.py::test_metadata_hashes_match_copied_assets
tests/test_model_package.py::test_model_fallback_requires_user_confirmation
tests/test_model_package.py::test_model_yaml_parses
tests/test_model_package.py::test_package_verification_matches
```

Exactly 9 real nodes.

Do NOT infer one node's reason from another node.
Do NOT parse progress tokens as nodes.
Do NOT run these nodes as one pytest batch.

---

## 2. GIT PREFLIGHT

Verify:

```text
current branch =
audit/task8b3-mask01-r1e2-full-suite-failure-triage

HEAD =
7a4e5454cdaf164c8d1706412c55b869074748e0
```

Create:

```text
audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED CHANGES

Only task records:

```text
docs/task8b3_mask01_d1_r1e2_r1_asset_regression_disambiguation.md
evaluation/task8b3_mask01_d1_r1e2_r1_asset_regression_disambiguation.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Do NOT modify anything under:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/
```

Do NOT create/download/copy model assets.
Do NOT install dependencies.
Do NOT write external RC1.

---

## 4. CLASSIFICATION RULE — FROZEN

For each real node choose exactly one:

```text
A = MISSING_FULL_DELIVERY_ASSET
B = MISSING_NONCANONICAL_TEST_FIXTURE
C = RUNTIME_DEPENDENCY_ENVIRONMENT
D = TRUE_CANONICAL_CODE_REGRESSION
E = OTHER
PASS = TEST_PASSES_WHEN_RUN_INDIVIDUALLY
```

Use these rules:

### A — MISSING_FULL_DELIVERY_ASSET

Use A only when the decisive failure is caused by a model/component/delivery asset or required delivery structure that:
- is absent from the lightweight canonical tree, AND
- is not part of the 135-entry lightweight source manifest.

Examples include missing:
```text
decoder.pt
detector.pt
SAM2 checkpoint/config
ProgramHead checkpoint
Qwen base directory/assets
```

### C — RUNTIME_DEPENDENCY_ENVIRONMENT

Use C when the decisive failure is a missing/wrong Python runtime dependency/version and not a code assertion.

### D — TRUE_CANONICAL_CODE_REGRESSION

Use D ONLY when:
- the test does not require an intentionally omitted full-delivery asset/fixture to reach its assertion, AND
- the actual assertion demonstrates current canonical source behavior violating the test contract.

A missing full-delivery asset MUST NOT be labeled D.

### PASS

If the node passes individually, record PASS.
A previous batched/group failure must not be assigned to a node that passes individually.

---

## 5. INVENTORY FACTS — DO NOT RE-DISCOVER AS CODE CHANGES

Previously established in the same remote state:

```text
ABSENT + NOT SOURCE-MANIFEST-LISTED:
model/buildreasonseg_advisor/decoder.pt
model/buildreasonseg_advisor/detector.pt
model/components/sam2/sam2.1_hiera_base_plus.pt
model/components/sam2/sam2.1_hiera_b+.yaml
model/components/program_head/program_parser_l3_rehearsal_v1.pt
model/components/program_head/Qwen3-VL-2B-Instruct

PRESENT:
model/buildreasonseg_advisor/model.yaml
model/buildreasonseg_advisor/metadata.json
model/buildreasonseg_advisor/metrics.json
model/components/program_head/qwen_asset_manifest.json
```

Use these facts when interpreting a node's own traceback.

---

## 6. EXACT EXECUTION METHOD

Working directory:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

Run EXACTLY ONE node per pytest invocation.

Use:

```text
python -m pytest <NODE_ID> -vv
```

for each of the 9 node IDs below.

### Node 1

```text
tests/test_language_contract.py::test_normal_path_is_qwen_first
```

### Node 2

```text
tests/test_model_package.py::test_components_complete
```

### Node 3

```text
tests/test_model_package.py::test_decoder_hash_and_bytes_exact
```

### Node 4

```text
tests/test_model_package.py::test_detector_hash_and_bytes_exact
```

### Node 5

```text
tests/test_model_package.py::test_hash_mismatch_raises
```

### Node 6

```text
tests/test_model_package.py::test_metadata_hashes_match_copied_assets
```

### Node 7

```text
tests/test_model_package.py::test_model_fallback_requires_user_confirmation
```

### Node 8

```text
tests/test_model_package.py::test_model_yaml_parses
```

### Node 9

```text
tests/test_model_package.py::test_package_verification_matches
```

Do not batch them together.

---

## 7. REQUIRED PER-NODE EVIDENCE

For EACH node record:

```text
node_id
exit_code
pytest_summary
PASS_or_FAIL
exception_type_or_assertion
first_decisive_failure_line
required_missing_path_if_any
required_runtime_dependency_if_any
classification
classification_reason
```

`first_decisive_failure_line` must come from that node's own invocation.

Do NOT use:
- another node's failure line;
- the group's final summary;
- `[100%]`;
- generic `FAILED ...` from a different node.

---

## 8. SPECIAL CHECKS

### test_normal_path_is_qwen_first

Record the exact `reason` returned by:

```python
QwenProgramHeadFrontend().available()
```

You may use a read-only Python one-liner to print it.

If the reason identifies absent Qwen assets in the lightweight tree, classification must be A, not D.

### test_model_yaml_parses

This node reads `model.yaml`, which is PRESENT and source-manifest-listed.

If it passes individually:
```text
classification = PASS
```

If it fails, capture the exact field/value mismatch before considering D.

### test_hash_mismatch_raises

Determine whether it reaches its intended synthetic hash-mismatch assertion or fails earlier because `ModelPackage.load(DEFAULT_MODEL)` / required package assets are incomplete.

Classify from the actual node output.

---

## 9. NO FULL SUITE / NO OTHER GROUPS

Do NOT run:

```text
python -m pytest -q
```

Do NOT run:
- paths group;
- setup checker group;
- Task 8B.1 group;
- targeted MASK gates again.

This task is only the 9-node D-vs-asset disambiguation.

---

## 10. AGGREGATE DECISION

After all 9 individual runs, record:

```text
true_regression_count = number classified D
missing_full_delivery_asset_count = number classified A
runtime_environment_count = number classified C
other_count = number classified E
individual_pass_count = number classified PASS
```

Then choose exactly one:

### If `true_regression_count == 0`

```text
PREVIOUS_TRUE_REGRESSION_CLASSIFICATION = REFUTED_FOR_GROUPS_1_2
```

This does NOT yet close R1-E2 globally; ChatGPT will combine this with groups 3/4/5 evidence.

### If `true_regression_count > 0`

```text
PREVIOUS_TRUE_REGRESSION_CLASSIFICATION = CONFIRMED_IN_PART
```

List exact D nodes.

---

## 11. WORKTREE AUDIT

After diagnostics:

```text
git status --porcelain=v1 --untracked-files=all
git diff --name-only
git diff --cached --name-only
```

Do not stage/delete generated files outside authorized task records.

---

## 12. REQUIRED REPORT

Create:

```text
docs/task8b3_mask01_d1_r1e2_r1_asset_regression_disambiguation.md
evaluation/task8b3_mask01_d1_r1e2_r1_asset_regression_disambiguation.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Required report fields:

```text
task_id = MASK01_D1_R1E2_R1_ASSET_REGRESSION_DISAMBIGUATION
status
starting branch/head
task branch

nodes_requested = 9
nodes_executed_individually = 9

per_node = [...]

classification_counts:
A =
B =
C =
D =
E =
PASS =

previous_true_regression_classification =
REFUTED_FOR_GROUPS_1_2
or CONFIRMED_IN_PART

true_regression_nodes = [...]

full_suite_reexecuted = false
canonical_files_modified = false
assets_created = false
dependencies_installed = false
repairs_attempted = false
model_inference = false
external_write = false

next_gate = CHATGPT_R1E2_R1_REMOTE_AUDIT
```

---

## 13. EVERY OUTCOME MUST BE PUSHED

Whether COMPLETE / STOP / FAILED:
- persist report/evidence/FROM_DSH;
- commit only authorized task records;
- push task branch.

---

## 14. COMMIT / PUSH

Commit exactly:

```text
git commit -m "docs(rc1): disambiguate asset versus code regressions"
```

Push:

```text
audit/task8b3-mask01-r1e2-r1-asset-regression-disambiguation
```

No force push.

Then STOP.

---

## 15. SUCCESS DEFINITION

```text
MASK01_D1_R1E2_R1_ASSET_REGRESSION_DISAMBIGUATION = COMPLETE

9 real nodes executed individually
no pytest progress token treated as a node
every classification based on that node's own decisive output
no repair performed
canonical files unchanged

NEXT = CHATGPT_R1E2_R1_REMOTE_AUDIT
```

Then STOP.
