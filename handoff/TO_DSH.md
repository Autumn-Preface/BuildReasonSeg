# TO_DSH — RC1_INSPECT01_PREFLIGHT_ORDER_FIX

> Status: ACTIVE
> Decision owner: ChatGPT
> Executor: DeepSeek Harness (DSH)
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Required starting branch: `fix/task8b3-mask01-success-semantics-r1d`
> Required starting HEAD: `5808e64ed43fd383be9e08bf623a9bfdc27f62b4`
> New task branch: `fix/rc1-inspect-proposals-preflight-order`

## 0. CHATGPT AUDIT / DECISION

R1-D is accepted and CLOSED.

Accepted remote state:

```text
branch = fix/task8b3-mask01-success-semantics-r1d
HEAD   = 5808e64ed43fd383be9e08bf623a9bfdc27f62b4
parent = 3bf9dafac69175d36a1abb6a83d019ef118ca72c
commit = chore(rc1): canonicalize source manifest
```

R1-D facts:

```text
manifest entries = 135
entries changed = 8
entries already canonical = 127
path count preserved = true
path order preserved = true
all-entry Git canonical validation = PASS
manifest contract validation = PASS
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

The report omitted the requested literal ordered-path digest values, but the remote diff independently shows no path additions/removals/reordering. This evidence omission does not block acceptance.

Before R1-E full canonical regression, one known pre-existing canonical contract failure must be resolved:

```text
test_predict_inspect_proposals_does_not_require_prompt
expected = E202 / exit 20
historical observed = E3xx / exit 30
classification = PREEXISTING_CANONICAL_CONTRACT_FAILURE_CANDIDATE
```

ChatGPT root-cause audit:

```text
handler()
  -> _validate_arguments()
  -> resolve_config()
  -> _resolve_and_report()          # model/package verification
  -> PredictRuntime()               # runtime initialization
  -> if args.inspect_proposals:
       inspect_proposals()
         -> load_image()
```

Therefore an unreadable inspect image can fail on model/package setup before reaching the image loader.

Frozen engineering correction:

```text
INSPECT_IMAGE_PREFLIGHT_BEFORE_MODEL_V1
```

For `--inspect-proposals` only, validate/read the image with the existing `load_image()` after argument/config validation but BEFORE `_resolve_and_report()` and `PredictRuntime()`.

Normal inference order remains unchanged.

---

## 1. PURPOSE

Make the inspect-proposals error contract deterministic:

```text
existing file + supported extension + unreadable bytes
+ --inspect-proposals
=> E202 IMAGE_UNREADABLE
=> exit 20
```

This is an engineering ordering fix only.

It does NOT:
- bypass model verification for valid images;
- remove detector/runtime requirements;
- change detector behavior;
- change proposal merge behavior;
- change normal inference ordering;
- change algorithms or thresholds.

For a valid readable inspect image, execution still proceeds through the existing model verification and detector runtime.

---

## 2. GIT PREFLIGHT

Verify exactly:

```text
current branch =
fix/task8b3-mask01-success-semantics-r1d

HEAD =
5808e64ed43fd383be9e08bf623a9bfdc27f62b4
```

Allowed initial worktree:
- clean, or
- only `M handoff/TO_DSH.md`.

Unexpected mutations:
- do not delete/reset/restore/stash/clean;
- record them;
- STOP if they overlap authorized paths.

Create:

```text
fix/rc1-inspect-proposals-preflight-order
```

No reset/rebase/amend/stash/clean/force-push.

---

## 3. ALLOWED PATHS

Only product/test paths:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
```

Task records:

```text
docs/rc1_inspect01_preflight_order_fix.md
evaluation/rc1_inspect01_preflight_order_fix.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other path may change.

Forbidden:

```text
buildreasonseg/runtime/pipeline.py
buildreasonseg/runtime/imageio.py
buildreasonseg/runtime/detector.py
tests/test_task8b_runtime.py
source_manifest.json
canonical docs
scripts/sync_advisor_rc1_delivery.py
configs
models
weights
external RC1
```

---

## 4. EXACT PRODUCT CHANGE

In:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
```

import the existing image loader:

```python
from buildreasonseg.runtime.imageio import load_image
```

In `handler(args)`, preserve:

```python
_validate_arguments(args)
config = resolve_config(args.config)
if not config.is_file():
    raise BuildReasonSegError("E101", detail=f"配置文件不存在: {config}")
```

Immediately AFTER that config check and BEFORE:

```python
package_name = _resolve_and_report(args)
```

insert exactly this inspect-only preflight:

```python
if args.inspect_proposals:
    # Establish the image-input error contract before model/package initialization.
    load_image(args.image)
```

Then keep the existing:

```python
package_name = _resolve_and_report(args)
...
runtime = PredictRuntime(...)
...
if args.inspect_proposals:
    ...
    result = inspect_proposals(runtime, request)
```

unchanged.

The valid inspect path may decode the image again inside `inspect_proposals()`. That duplicate read is accepted for this narrow RC1 engineering fix; do NOT widen scope by changing pipeline signatures or caching `LoadedImage`.

---

## 5. NORMAL INFERENCE MUST NOT MOVE

Do NOT move `_resolve_and_report()` for normal inference.

Frozen normal ordering remains:

```text
argument validation
config validation
model package resolution / verification
PredictRuntime initialization
language/runtime inference path
```

The new preflight is guarded only by:

```python
if args.inspect_proposals:
```

---

## 6. EXISTING INTEGRATION TEST — PRESERVE

Do NOT weaken or rewrite away:

```text
test_predict_inspect_proposals_does_not_require_prompt
```

Its expected behavior remains:

```text
unreadable .png
--inspect-proposals
no E101 prompt error
E202
exit 20
```

This existing test must PASS after the product fix.

Do NOT change expected E202 to E3xx.

---

## 7. ADD DIRECT ORDERING TEST

Add exactly:

```text
test_inspect_unreadable_image_precedes_model_resolution
```

to:

```text
tests/test_cli_contract.py
```

Test design:

1. Create:
   ```python
   image = tmp_path / "tile.png"
   image.write_bytes(b"not-an-image")
   ```

2. Build real args:
   ```python
   import predict as predict_module
   args = predict_module.build_parser().parse_args(
       ["--image", str(image), "--inspect-proposals"]
   )
   ```

3. Monkeypatch:
   ```python
   predict_module._resolve_and_report
   ```
   to fail immediately if called, e.g. a function that invokes:
   ```python
   pytest.fail("_resolve_and_report must not run before unreadable inspect image validation")
   ```

4. Call real:
   ```python
   predict_module.handler(args)
   ```

5. Assert:
   ```python
   with pytest.raises(BuildReasonSegError) as error:
       ...
   assert error.value.code == "E202"
   ```

Import the real:

```python
from buildreasonseg.errors import BuildReasonSegError
```

No exception swallowing.
No source-text scanning.

This proves model resolution is not reached before unreadable inspect-image validation.

---

## 8. TEST GATES — EXACT

From:

```text
delivery_src/BuildReasonSeg_Advisor_RC1
```

### Gate A — inspect ordering contract

Run exactly:

```text
python -m pytest \
  tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt \
  tests/test_cli_contract.py::test_inspect_unreadable_image_precedes_model_resolution \
  -q
```

Required:

```text
2 passed
exit 0
```

### Gate B — unaffected CLI validation

Run exactly:

```text
python -m pytest \
  tests/test_cli_contract.py::test_predict_help_lists_frozen_arguments \
  tests/test_cli_contract.py::test_predict_prompt_required_for_normal_inference \
  tests/test_cli_contract.py::test_predict_missing_image_reports_e201 \
  tests/test_cli_contract.py::test_predict_unsupported_image_type_reports_e203 \
  -q
```

Required:

```text
4 passed
exit 0
```

Do NOT replace with `-k`.

Do NOT run full `test_cli_contract.py`.
Do NOT run full canonical suite.

---

## 9. MANIFEST — INTENTIONALLY DEFERRED ONE MORE TIME

This task changes manifest-listed files:

```text
predict.py
tests/test_cli_contract.py
```

Therefore the R1-D manifest will become stale after this task.

Do NOT update `source_manifest.json` in this same task.

Record:

```text
manifest_state_after_task = STALE_BY_EXPECTED_INSPECT_FIX
next_manifest_gate = R1D2_RECANONICALIZATION
```

A narrow manifest recanonicalization task will follow this fix before R1-E.

---

## 10. PRODUCT FREEZE OUTSIDE ORDERING

Do NOT change:
- `_resolve_and_report()` implementation;
- `PredictRuntime`;
- `inspect_proposals()` implementation;
- image loader implementation;
- detector;
- proposal merge;
- CLI output semantics;
- SUCCESS semantics;
- batch behavior;
- normal inference behavior;
- any threshold / model / architecture.

---

## 11. NO MODEL WORK / EXTERNAL WRITE

Do NOT:
- run model inference beyond the test process behavior above;
- run detector on a valid image;
- run training;
- sync external RC1;
- modify external delivery.

The unreadable integration test should fail at image loading before model resolution.

---

## 12. EVERY OUTCOME MUST BE PUSHED

Frozen rule:

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
docs/rc1_inspect01_preflight_order_fix.md
evaluation/rc1_inspect01_preflight_order_fix.json
```

Update:

```text
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Report:

```text
task_id = RC1_INSPECT01_PREFLIGHT_ORDER_FIX
status

starting branch/head
task branch

root_cause =
model/package resolution preceded image readability validation in inspect mode

frozen_fix =
INSPECT_IMAGE_PREFLIGHT_BEFORE_MODEL_V1

inspect_preflight_uses_existing_load_image = true
normal_inference_order_changed = false
valid_inspect_model_verification_bypassed = false

Gate A command/result
Gate B command/result

predict_py_changed = true
cli_test_changed = true
pipeline_changed = false
imageio_changed = false
detector_changed = false
canonical_docs_changed = false
manifest_changed = false

manifest_state_after_task = STALE_BY_EXPECTED_INSPECT_FIX
next_manifest_gate = R1D2_RECANONICALIZATION

model_inference = false
training = false
external_write = false

github_persistence_policy = ALL_TASK_OUTCOMES_PUSHED
next_gate = CHATGPT_INSPECT01_REMOTE_AUDIT
```

---

## 14. DIFF GATE

Only stage:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/predict.py
delivery_src/BuildReasonSeg_Advisor_RC1/tests/test_cli_contract.py
docs/rc1_inspect01_preflight_order_fix.md
evaluation/rc1_inspect01_preflight_order_fix.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Unexpected paths:
- do not delete;
- do not stage;
- report them.

---

## 15. COMMIT / PUSH

If Gate A and Gate B pass, commit exactly:

```text
git commit -m "fix(rc1): validate inspect image before model setup"
```

If STOP/FAILED occurs, commit truthful authorized state with a truthful status message.

Push:

```text
fix/rc1-inspect-proposals-preflight-order
```

No force push.

After push print:

```text
LOCAL_FINAL_HEAD=<sha>
REMOTE_FINAL_HEAD=<sha>
FINAL_PARENT=<sha>
```

Then STOP.

Do NOT enter R1-D2 or R1-E.

---

## 16. SUCCESS DEFINITION

```text
RC1_INSPECT01_PREFLIGHT_ORDER_FIX = COMPLETE

unreadable inspect image:
E202
exit 20
model resolution not reached first

existing inspect contract test = PASS
direct ordering test = PASS
unaffected CLI gate = 4/4 PASS

normal inference order = unchanged
valid inspect still requires model verification + detector
runtime pipeline = unchanged
detector = unchanged
algorithms = unchanged

manifest = intentionally stale
NEXT = R1D2_RECANONICALIZATION after ChatGPT audit
```

Then STOP.
