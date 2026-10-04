# TO_DSH — Task 8B.3-M1A.2C-D1.1: Normalize Canonical Suite Forensics

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `6405c64c7bba40c398b16bd9f3050c3c95cc7db8`

# 0. Purpose

Correct the D1 forensic classification and normalize the report.

Frozen ChatGPT audit result:

```text
Recovered nodes = 23
A = 10
B = 13
C = 0
D = 0
E = 0
tests/test_task8b_runtime.py failing nodes = none
Gate conclusion = CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE
```

# 1. Why D1 needs correction

D1 currently contradicts itself: it records `A=7, B=10, C=0, D=6, E=0` and
`MIXED_FAILURES_REQUIRE_CODE_AUDIT`, but its rationale also says no class-D node was detected.

The six D-labelled nodes must be reclassified as follows.

# 2. Frozen six-node reclassification

A:
- `tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt`
  - `predict.py` performs model-package resolution/verification before image loading; canonical lacks decoder/detector weights.
- `tests/test_language_contract.py::test_normal_path_is_qwen_first`
  - `QwenProgramHeadFrontend.available()` requires ProgramHead checkpoint + `Qwen3-VL-2B-Instruct/`; canonical lacks both.
- `tests/test_model_package.py::test_components_complete`
  - explicitly requires a complete default model/component package.

B:
- `tests/test_task8b1_fallback_ux.py::test_fixture_frozen_and_complete`
- `tests/test_task8b1_fallback_ux.py::test_no_prompt_to_program_table_in_delivery_code`
- `tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim`
  - these depend on `ROOT/logs/task8b1_prompt_suite.json` and/or `ROOT/logs/task8b_gates.json`;
    the sanitized canonical tree has no tracked `logs/` delivery-artifact directory.

# 3. Final authoritative classification

```text
A — CANONICAL_ASSET_ABSENCE: 10
B — FIXTURE_ASSUMES_FULL_DELIVERY: 13
C — COMPACT_RUNTIME_REGRESSION_CANDIDATE: 0
D — UNRELATED_CODE_REGRESSION_CANDIDATE: 0
E — INSUFFICIENT_EVIDENCE: 0
TOTAL: 23
```

No failing node belongs to `tests/test_task8b_runtime.py`.

Historical compact gate remains:
- dedicated runtime tests: 32 passed
- 5000×5000 fake-shape guard: PASS
- manifest: 135/135 VERIFIED

Do not rerun any of these.

# 4. Correct gate conclusion

Use exactly:

```text
CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE
```

Meaning:
- sanitized canonical source is not a valid place for the complete delivery-oriented suite;
- no recovered M1A.2C failure is evidence of compact-mask runtime regression;
- the full delivery regression gate belongs after controlled canonical → external delivery sync;
- this task does NOT authorize that sync.

# 5. Strict prohibitions

Do NOT:
- run pytest, check_setup.py, predict.py, or any model;
- modify product code, tests, or `source_manifest.json`;
- modify external delivery;
- copy/download/install assets;
- sync delivery;
- fix PROP-01 / REF-01 / MASK-01;
- enter Task 8B.4 or Task 8C.

# 6. Allowed paths

Only:

```text
docs/task8b3_m1a2c_canonical_suite_forensics.md
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 7. Create dedicated forensic report

Create:

```text
docs/task8b3_m1a2c_canonical_suite_forensics.md
```

Required sections:
1. Task/scope
2. Starting HEAD
3. Observed M1A.2C result: `17 failed, 93 passed, 6 errors in 29.67s`
4. Exact 23-node list
5. Canonical asset/artifact inventory
6. Final classification table for all 23 nodes
7. Explicit corrected six-node reclassification
8. `tests/test_task8b_runtime.py failing nodes = none`
9. Historical dedicated compact gate: `32 passed`, fake-5000 guard PASS, manifest 135/135
10. Final counts: `A=10 B=13 C=0 D=0 E=0`
11. Gate conclusion: `CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE`
12. Next gate: awaiting ChatGPT audit before controlled canonical → external delivery sync.

Each row must include a concise direct dependency reason.

# 8. Normalize existing M1A report

In `docs/task8b3_m1a_compact_proposal_masks.md`, remove the active authoritative
`A=7 B=10 D=6` / `MIXED_FAILURES_REQUIRE_CODE_AUDIT` conclusion.

Replace it with a short pointer to the dedicated forensic report and:

```text
A=10 B=13 C=0 D=0 E=0
CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE
```

The original D1 misclassification may be mentioned only as audit history.

# 9. FROM_DSH

Preserve `ARTIFACT-FACTS` exactly and save UTF-8 without BOM.

Required:

```text
Task: 8B.3-M1A.2C-D1.1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: 6405c64c7bba40c398b16bd9f3050c3c95cc7db8
Pytest/check_setup/predict/model runs: NONE
Product/tests/manifest modified: NO
External delivery modified: NO
Exact node recovery: 23/23
Class A: 10
Class B: 13
Class C: 0
Class D: 0
Class E: 0
Task8b runtime failures: NONE
Dedicated compact gate: 32 PASSED (historical, not rerun)
Manifest: 135/135 VERIFIED (historical, not rerun)
Gate conclusion: CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE
Report: docs/task8b3_m1a2c_canonical_suite_forensics.md
Next action: Awaiting ChatGPT audit before controlled canonical → external delivery sync.
```

# 10. Commit / push

Verify only allowed paths changed.

If COMPLETE, exact commit:

```text
docs(rc1): normalize canonical suite forensics
```

Otherwise:

```text
docs(rc1): record canonical forensic normalization stop
```

Push current branch, no force.

# 11. COMPLETE definition

COMPLETE only if:
- no test/tool/model execution;
- no product/test/manifest/delivery modification;
- dedicated report exists;
- all 23 nodes listed;
- final counts exactly A10/B13/C0/D0/E0;
- six previous D nodes reclassified exactly as frozen;
- no active contradictory D=6/MIXED conclusion remains;
- task8b runtime failures explicitly NONE;
- gate conclusion exact;
- report/handoff committed and pushed;
- clean tree;
- stop.

# 12. Final response

```text
TASK 8B.3-M1A.2C-D1.1 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Test/model execution:
NONE

Product/tests/manifest modified:
NO

External delivery:
UNCHANGED

Exact node recovery:
23/23

Failure classes:
A=10 B=13 C=0 D=0 E=0

Task8b runtime failures:
NONE

Gate conclusion:
CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE

Report:
docs/task8b3_m1a2c_canonical_suite_forensics.md

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得同步 delivery、不得运行 pytest/真实 Demo、不得进入 Task 8B.4 或 Task 8C。
```
