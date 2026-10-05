# TO_DSH — Task 8B.3-REF01-E3A-R5: Attachment-Driven Canonical Implementation

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DSH
> Required branch: `fix/task8b3-ref01-eligibility-repair-impl`
> Required starting HEAD: `34d1a263ac91ada6ef517fc54a99c19463e9ddd3`
> Required Python: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe`
> Required attached patcher filename: `E3A_R5_apply_exact.py`
> Required attached patcher SHA256: `1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f`
> Fixed execution copy: `C:\D\DeepSeekHarness\E3A_R5_apply_exact.py`

# 0. R4 AUDIT DISPOSITION

R4 is a safe STOP but not an approved implementation.

Accepted:
- exactly one final task commit;
- no product source change;
- no source_manifest/external sync;
- no detector/model inference;
- no intermediate commit/push.

R4 failure:
- DSH did not execute the required Python `-c` extraction command directly.
- It materialised the command and attempted to launch it through a host shell, producing WinError 2.

R5 removes all Markdown/code extraction. The executable patcher is a separate ChatGPT-provided attachment.

# 1. EXECUTION CONTRACT

You receive TWO attachments:
1. this taskbook;
2. `E3A_R5_apply_exact.py`.

DSH MUST NOT recreate, transcribe, regenerate, reformat, or edit the patcher.

Allowed sequence only:
1. verify Git branch/HEAD;
2. locate the attached file `E3A_R5_apply_exact.py`;
3. calculate SHA256 of the attached bytes;
4. copy those exact bytes to `C:\D\DeepSeekHarness\E3A_R5_apply_exact.py`;
5. calculate SHA256 of the copied file;
6. execute copied patcher exactly once with REQUIRED_PYTHON;
7. delete copied patcher after PASS;
8. compile modified PRODUCT files only;
9. run targeted pytest;
10. run canonical full pytest;
11. write evidence/report/FROM_DSH;
12. make exactly one final commit;
13. push exactly once;
14. STOP.

No other execution gate or method is permitted.

Any required step impossible or failing => STOP. No self-repair.

# 2. GIT GATE

Require exactly:

```text
branch = fix/task8b3-ref01-eligibility-repair-impl
HEAD = 34d1a263ac91ada6ef517fc54a99c19463e9ddd3
```

Allowed initial repo status:
- clean; or
- only `M handoff/TO_DSH.md`.

Require:
```text
scripts/task8b3_ref01_eligibility_repair_patcher.py
```
absent.

Before final commit:
```text
NO git commit
NO git push
```

# 3. PATCHER ATTACHMENT IDENTITY GATE

Locate the attached file whose filename is exactly:

```text
E3A_R5_apply_exact.py
```

If DSH cannot access this attachment as raw bytes => STOP.

Do NOT use taskbook text as a substitute.

Attached-file SHA256 must be exactly:

```text
1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f
```

Mismatch => STOP.

Copy attached bytes exactly to:

```text
C:\D\DeepSeekHarness\E3A_R5_apply_exact.py
```

Copied-file SHA256 must again equal:

```text
1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f
```

Mismatch => STOP.

Prohibited:
- no patcher py_compile;
- no patcher edit;
- no repo copy;
- no alternate destination path.

# 4. EXECUTE PATCHER EXACTLY ONCE

Run exactly:

```text
C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg\.conda\buildreasonseg-proposal\python.exe C:\D\DeepSeekHarness\E3A_R5_apply_exact.py
```

Require exit 0 and:

```text
PATCHER_BRANCH_HEAD: PASS
EXTERNAL_DETECTOR_IDENTITY: PASS
DETECTOR_BOUNDARY_REPLACEMENT: PASS
TEST_MARKER_INSERTION: PASS
NEW_TEST_COUNT: 8
PATCHER_RESULT: PASS
```

Failure => STOP; no retry.

After PASS delete exactly:

```text
C:\D\DeepSeekHarness\E3A_R5_apply_exact.py
```

# 5. PRODUCT STATIC GATE

Run only:

```text
<REQUIRED_PYTHON> -m py_compile delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
<REQUIRED_PYTHON> -m py_compile delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py
```

Require both exit 0.

# 6. TARGETED TEST

Run exactly once:

```text
<REQUIRED_PYTHON> -m pytest delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py -q
```

Require exit 0.

Failure => STOP. No code/test edit.

# 7. CANONICAL FULL TEST

Only after targeted PASS:

```text
<REQUIRED_PYTHON> -m pytest delivery_src\BuildReasonSeg_Advisor_RC1\tests -q
```

Require exit 0.

Failure => STOP.

# 8. EXTERNAL / MANIFEST GATES

Do NOT sync external RC1.

Actual external detector SHA must remain:

```text
82531dc3b758cd8a864f77d0fa97e4132113cb46eb5a33a11f483bf51bc0a738
```

Canonical:
```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```
must remain unchanged.

Record:
```text
manifest_status = INTENTIONALLY_STALE_PENDING_E3B
detector_model_calls = 0
```

# 9. ALLOWED FINAL TRACKED DIFF

Only:
```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/detector.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_task8b_runtime.py
evaluation/task8b3_ref01_eligibility_repair_impl.json
docs/task8b3_ref01_eligibility_repair_impl.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Must remain unchanged:
```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/pipeline.py
scripts/sync_advisor_rc1_delivery.py
```

# 10. COMPLETE EVIDENCE

If all gates PASS, write exactly:

```json
{
  "task": "8B.3-REF01-E3A-R5",
  "starting_head": "34d1a263ac91ada6ef517fc54a99c19463e9ddd3",
  "branch": "fix/task8b3-ref01-eligibility-repair-impl",
  "design_id": "LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1",
  "patch_method": "ATTACHMENT_VERIFIED_BOUNDARY_PATCHER",
  "patcher_sha256": "1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f",
  "detector_model_calls": 0,
  "eligible_function_changed": false,
  "eligible_proposals_function_changed": false,
  "largest_only_exception": true,
  "strict_area_operator": ">",
  "strict_confidence_operator": ">",
  "no_baseline_behavior": "NO_EXCEPTION_SAFE_FAILURE",
  "border_exception_allowed": false,
  "smallest_family_changed": false,
  "new_numeric_thresholds": [],
  "product_py_compile": "PASS",
  "targeted_test_exit": 0,
  "canonical_full_test_exit": 0,
  "external_detector_unchanged": true,
  "source_manifest_updated": false,
  "manifest_status": "INTENTIONALLY_STALE_PENDING_E3B",
  "overall_outcome": "REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTED",
  "next_gate": "REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC"
}
```

# 11. REPORT / FROM_DSH

Append authoritative R5 section to:
```text
docs/task8b3_ref01_eligibility_repair_impl.md
```

For COMPLETE require:
```text
Task = 8B.3-REF01-E3A-R5
Status = COMPLETE
Design selected by = CHATGPT
DSH algorithm choice = NO
Patch method = ATTACHMENT_VERIFIED_BOUNDARY_PATCHER
Attached patcher SHA = 1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f
Copied patcher SHA = 1a4294aa2e3070e420dc99e6621c1c323b27e153432abff8ee4014b23d33de9f
Patcher runs = 1
Patcher compile gate = NONE
Detector/model calls = 0
eligible() changed = NO
eligible_proposals() changed = NO
Largest-only exception = YES
Strict area/confidence = > / >
Border bypass = NO
No-baseline exception = NO
Smallest family changed = NO
New numeric threshold = NO
Product py_compile = PASS
Targeted pytest = PASS
Canonical full pytest = PASS
External RC1 modified = NO
Source manifest updated = NO
Manifest status = INTENTIONALLY_STALE_PENDING_E3B
Outcome = REF01_ELIGIBILITY_REPAIR_CANONICAL_IMPLEMENTED
NEXT = REF01_ELIGIBILITY_REPAIR_CANONICALIZE_AND_SYNC
PROP-01 = PROP01_OPEN_ENGINEERING_DEFECT
left/below reference-selection defects = UNRESOLVED
final Demo inference = NOT RUN
```

Preserve ARTIFACT-FACTS exactly.

# 12. SINGLE FINAL COMMIT / PUSH GATE

Before commit require:
```text
git rev-list --count 34d1a263ac91ada6ef517fc54a99c19463e9ddd3..HEAD
= 0
```

COMPLETE iff every required gate passed, copied patcher is deleted, external detector/source_manifest unchanged, and only six allowed tracked paths differ.

If COMPLETE:
```text
Status: COMPLETE
commit:
fix(rc1): implement largest extent dominance exception
```

Otherwise:
```text
Status: STOP
commit:
fix(rc1): record extent dominance implementation stop
```

Commit exactly once.

After commit require:
```text
git rev-list --count 34d1a263ac91ada6ef517fc54a99c19463e9ddd3..HEAD
= 1
```

Push current branch exactly once.

No force push.
Do not update main.
Do not execute NEXT.

After push: STOP.
