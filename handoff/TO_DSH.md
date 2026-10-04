# TO_DSH — Task 8B.3-M1A.2A-R1: Fix Reference Crop Intersection + Handoff BOM

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-mem01-compact-proposals`
> Required starting HEAD: `10d36117993f128c9a3d6bd2bea114dc63120121`

# 0. Why this correction exists

ChatGPT re-read GitHub after propagation and found M1A.2A commit:
`10d36117993f128c9a3d6bd2bea114dc63120121` (`fix(rc1): store compact proposal masks in canonical runtime`).

The compact detector implementation is present, but `core.reference_mask_from_proposal()`
computes crop bounds against the image/context only, not against `proposal.global_bbox`.
When the 512 context begins before the proposal bbox, this yields negative offsets into
`proposal.mask_crop` and can return an empty/misaligned reference mask.

Also `handoff/FROM_DSH.md` again has a UTF-8 BOM before
`<!-- ARTIFACT-FACTS:BEGIN -->`.

This task fixes only those two issues.

# 1. Strict prohibitions

Do NOT modify detector.py, outputs.py, tests, source_manifest.json, external delivery,
model/config/threshold/tiling/merge/reference policy, PROP-01, REF-01, MASK-01,
Task 8B.4, or Task 8C.

Do NOT run pytest, real predict, six-image Demo, YOLO, Qwen, SAM2, or D-B1.

Allowed changed paths only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/buildreasonseg/runtime/core.py
docs/task8b3_m1a_compact_proposal_masks.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 2. Git gate

Require:

```text
branch = fix/task8b3-mem01-compact-proposals
HEAD = 10d36117993f128c9a3d6bd2bea114dc63120121
```

Allowed initial tree: clean or only `M handoff/TO_DSH.md`.

# 3. Fix `reference_mask_from_proposal()`

The valid global rectangle must be:

```text
original image ∩ 512 reasoning context ∩ proposal.global_bbox
```

With inclusive proposal bbox `(top,left,bottom,right)`, use:

```python
valid_top = max(0, context.top, top)
valid_left = max(0, context.left, left)
valid_bottom = min(image_height, context.top + CONTEXT_SIZE, bottom + 1)
valid_right = min(image_width, context.left + CONTEXT_SIZE, right + 1)
```

If empty, return the all-false 512 context mask.

Then slice `proposal.mask_crop` using offsets from proposal `top,left` and paste using
offsets from `context.top,context.left`.

Do not allocate a full-image proposal mask.

# 4. Minimal smoke

Do NOT edit tests.

Run `py_compile` for core.py.

Then run one ephemeral Python smoke with exactly:
1. context begins above/left of proposal bbox;
2. context clips proposal at top/left;
3. context clips proposal at bottom/right.

For each, compare compact `reference_mask_from_proposal()` with a test-only legacy
full-image reconstruction sliced into the same context. Require bit-identical output.

No model loading.

# 5. Fix BOM

Rewrite `handoff/FROM_DSH.md` as UTF-8 without BOM.

First bytes must not be `EF BB BF`.

`ARTIFACT-FACTS` contents, ordering and values must remain unchanged.

# 6. Report / handoff

Append report section:

```text
## Task 8B.3-M1A.2A-R1 — Reference Crop Intersection Correction
```

Record starting HEAD, bug, corrected 3-way intersection, 3-case smoke result,
BOM removal, tests/manifest not changed, external delivery unchanged, real inference not run.

Update FROM_DSH fields:

```text
Task: 8B.3-M1A.2A-R1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-mem01-compact-proposals
Starting HEAD: 10d36117993f128c9a3d6bd2bea114dc63120121
Reference crop 3-way intersection: PASS / FAIL
Legacy-equivalence smoke: PASS / FAIL
FROM_DSH encoding: UTF-8 WITHOUT BOM / FAIL
ARTIFACT-FACTS: PRESERVED / FAIL
Tests: NOT RUN BY DESIGN
Manifest: NOT UPDATED BY DESIGN
External delivery modified: NO
Real inference: NO
Next action: Awaiting ChatGPT audit.
```

# 7. Commit / push

If COMPLETE:

```text
fix(rc1): correct compact reference crop
```

Otherwise:

```text
docs(rc1): record reference crop correction stop
```

Push current branch, no force.

# 8. Final response

```text
TASK 8B.3-M1A.2A-R1 COMPLETE / PARTIAL / STOP / FAILED

Commit:
<sha or NONE>

Push:
PASS / FAIL

Reference crop intersection:
PASS / FAIL

Legacy-equivalence smoke:
PASS / FAIL / NOT RUN

FROM_DSH encoding:
UTF-8 WITHOUT BOM / FAIL

ARTIFACT-FACTS:
PRESERVED / FAIL

Tests:
NOT RUN BY DESIGN

Manifest:
NOT UPDATED BY DESIGN

External delivery:
UNCHANGED

Real inference:
NOT RUN

STOP reason:
<none or exact>

等待 ChatGPT 审核；不得编辑 tests/manifest、不得同步 delivery、不得运行真实 Demo。
```
