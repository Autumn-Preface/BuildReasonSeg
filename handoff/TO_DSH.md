# TO_DSH — Task 8B.3-D1: Demo Failure Forensics Only

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `eval/task8b3-six-image-demo-suite`
> Required starting HEAD: `648dc9d48886f6da478f8e27b54116d48ca1d725`
> External RC1: `C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1`

# 0. Purpose

Perform a forensic/root-cause audit of the already completed Task 8B.3-R4B six-image Demo baseline.

This task is **read-only with respect to all model/runtime/input/output artifacts**.

No inference is authorized.

The goal is to convert the six observed Demo failures into precise engineering defect facts before any fix is designed.

# 1. Permanent reporting rule

For COMPLETE / PARTIAL / STOP / FAILED, when Git remains safe:
1. create/update the task report;
2. update `handoff/FROM_DSH.md`;
3. commit;
4. push current branch;
5. stop and wait for ChatGPT.

# 2. Strict prohibitions

Do not:
- run `predict.py`;
- run `scripts/task8b3_interactive_suite.py`;
- run detector/SAM2/D-B1/Qwen inference;
- rerun any A1–B2 sample;
- modify any file under external delivery;
- modify any existing inference input/output/log/diagnostic artifact;
- modify any product/canonical/harness/test source;
- modify threshold/config/checkpoint/model assets;
- install packages;
- train/download;
- access final test;
- use Assisted Mode / `--reference-id` / `--inspect-proposals`;
- implement Task 8B.4;
- enter Task 8C.

Only repository documentation/handoff may change:

```text
docs/task8b3_d1_demo_failure_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 3. Git safety

Run:

```bat
cd /d C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
git branch --show-current
git rev-parse HEAD
git status --short
```

Continue only if:

```text
branch = eval/task8b3-six-image-demo-suite
HEAD = 648dc9d48886f6da478f8e27b54116d48ca1d725
```

Allowed working tree:
- clean; or
- only `M handoff/TO_DSH.md`.

Anything else -> STOP and persist/push handoff if safe.

# 4. Frozen baseline facts

Do not reinterpret the formal R4B run as a new evaluation.

Frozen result:

```text
A1 language DIRECT_CORRECT   runtime SUCCESS
A2 language FALLBACK_CORRECT runtime FAILED E401
A3 language DIRECT_CORRECT   runtime SUCCESS
A4 language FALLBACK_CORRECT runtime SUCCESS
B1 language FALLBACK_CORRECT runtime FAILED E502
B2 language DIRECT_CORRECT   runtime FAILED E502
```

Human visual audit by ChatGPT after R4B:

```text
A1: semantic FAIL — selected Reference is a fragment/proposal of a larger blue-roof complex;
    target is a tiny fragment and not a convincing separate right-nearest building.

A3: semantic FAIL — target mask is the best-looking of the three successes, but selected Reference
    is a small roadside proposal rather than the visually largest building; therefore the full
    Reference→Relation→Target instruction is not satisfied.

A4: semantic FAIL — selected Reference is not a convincing global largest complete building;
    final target is a small/incomplete fragment.

Therefore the six-image Demo baseline currently has:
language resolution 6/6,
runtime success 3/6,
manual end-to-end semantic success 0/6.
This 0/6 is a qualitative six-sample Demo audit, NOT a research benchmark metric.
```

Do not change these human-review labels in DSH.

# 5. Evidence paths to inspect read-only

Use existing files only.

Formal suite summary/logs:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_suite_results.json
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_transcripts\A1.txt
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_transcripts\A2.txt
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_transcripts\A3.txt
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_transcripts\A4.txt
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_transcripts\B1.txt
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1\logs\task8b3_transcripts\B2.txt
```

Success diagnostics:

```text
inference\output\diagnostics\A1_001\
inference\output\diagnostics\A3\
inference\output\diagnostics\A4\
```

Failed diagnostics:

```text
inference\output\diagnostics\A2\
inference\output\diagnostics\B1\
inference\output\diagnostics\B2\
```

Relevant canonical source read-only:

```text
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\detector.py
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\pipeline.py
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\outputs.py
delivery_src\BuildReasonSeg_Advisor_RC1\buildreasonseg\runtime\context.py
```

# 6. Forensics A — Reference selection on A1/A3/A4

For each A1, A3, A4, read `proposals.json` and `result.json`.

Produce a table with:

```text
sample
merged_proposal_count
selected_reference_id
selected_reference_area
selected_reference_bbox
selected_reference_bbox_extent_ratio
selected_reference_touches_border
selected_reference_confidence
largest_proposal_id_by_mask_area
largest_proposal_area
largest_proposal_eligible_under_frozen_rule?
largest_eligible_proposal_id
largest_eligible_proposal_area
selected_is_largest_eligible?
number_of_eligible_proposals
```

Use the actual frozen eligibility rule from canonical `detector.py`.

Do not change or reinterpret the rule.

Then state separately for each sample:

1. whether implementation selected the largest **eligible detected proposal** correctly;
2. whether that selected proposal visually/structurally represents the user's intended “largest building” according to the frozen ChatGPT visual audit;
3. whether discrepancy is:
   - proposal coverage/fragmentation,
   - eligibility filtering,
   - or both.

Do not claim a ground-truth building identity.

# 7. Forensics B — Semantic contract of `largest`

Read `detector.py` and document exactly:

- eligibility criteria;
- area/confidence/id tie-break;
- whether “largest” means largest among all proposals or largest among eligible proposals.

Explicitly compare the actual implementation semantics to the user-facing prompt semantics:

```text
user-facing: largest building in the image
implementation: <exact factual semantics>
```

If these differ, label this as a semantic-contract mismatch; do not fix it.

# 8. Forensics C — Why A1/A4 tiny fragments can return SUCCESS

Read `pipeline.py`.

List every post-inference hard validity gate applied before `status=SUCCESS`.

Answer factually:
- Is there a minimum target area gate?
- Is there a target connected-component/instance-integrity gate?
- Is there a requirement that target correspond to an existing detector proposal?
- Is there only non-empty / non-padding / directional-centroid validation?

Use the actual source only.

For A1 and A4, record mask area from existing `result.json`.

Do not propose a threshold in this task.

# 9. Forensics D — A2 E401

From existing A2 transcript/result/proposals only:

Record:
- Qwen initial program;
- suggested program;
- final language status;
- tile count;
- raw proposal count;
- merged proposal count;
- detector-related warning text;
- final E401 detail;
- detector timing if available.

State only supported conclusions.

Important:
- do not claim the NMS warning definitely caused zero proposals unless evidence proves causality;
- classify as detector/proposal-stage failure.

# 10. Forensics E — B1/B2 E502 memory failure

From existing transcripts and source only.

Record:
- image dimensions;
- exact allocation error;
- requested allocation shape/dtype/size;
- all relevant full-frame boolean allocation sites in the execution path before detector return.

In particular inspect `DetectorRuntime.detect_global()` for allocations semantically equivalent to:

```python
np.zeros((height_px, width_px), dtype=bool)
```

Calculate:

```text
5000 * 5000 * 1 byte
```

in bytes and MiB.

Compare it to the exact exception request (`23.8 MiB`, bool, `(5000, 5000)`).

State one of:

```text
EXACT_SIGNATURE_MATCH
STRONG_MATCH
NOT_CONFIRMED
```

based only on static code + exception signature.

Also document whether the full-frame bool allocation is made:
- once per image,
- once per tile,
- or once per raw detection/proposal.

Do not implement a memory fix.

# 11. Defect taxonomy

Create exactly these candidate defect IDs, but mark CONFIRMED / PARTIAL / NOT CONFIRMED according to evidence:

```text
RC1-DEMO-REF-01
Reference semantics/proposal quality: user-facing “largest building” may resolve to a fragmented or
eligibility-filtered proposal rather than the visually largest complete building.

RC1-DEMO-MASK-01
SUCCESS validity is too weak to reject tiny/incomplete target fragments.

RC1-DEMO-PROP-01
A2 detector/proposal path can return zero proposals on a building-containing Demo scene.

RC1-DEMO-MEM-01
Large-image path uses full-frame boolean proposal allocations and fails on 5000×5000 inputs.
```

For each:
- evidence;
- affected samples;
- layer;
- severity: blocker / major / minor;
- whether fix requires product code change;
- whether scientific model/checkpoint change is required (`yes/no/not yet known`).

Do not design the fix yet.

# 12. Prioritization recommendation

Produce a factual fix-order recommendation without implementing:

Recommended ordering must explicitly consider dependencies between defects.

Use this decision policy:

1. bugs preventing the pipeline from running at all;
2. proposal/reference correctness;
3. target validity/quality guard;
4. packaging/output layout;
5. free manual Demo.

Task 8B.4 remains deferred until ChatGPT reviews this forensics report.

# 13. Report

Create:

```text
docs/task8b3_d1_demo_failure_forensics.md
```

Required sections:

1. Task and scope
2. Frozen R4B baseline
3. A1/A3/A4 Reference-selection table
4. `largest` semantic-contract audit
5. A1/A4 SUCCESS-validity audit
6. A2 E401 forensics
7. B1/B2 E502 memory forensics
8. Defect taxonomy
9. Dependency/prioritization recommendation
10. No-change statement
11. Next gate = Awaiting ChatGPT audit

# 14. FROM_DSH

Update `handoff/FROM_DSH.md`, preserving `ARTIFACT-FACTS` verbatim.

Required:

```text
Task: 8B.3-D1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: eval/task8b3-six-image-demo-suite
Starting HEAD: 648dc9d4...
Inference executed: NO
Delivery modified: NO
Product/harness/tests modified: NO
Reference forensics: <summary>
A2 forensics: <summary>
B1/B2 memory signature: <status>
Defects: <IDs + status>
Report: docs/task8b3_d1_demo_failure_forensics.md
Output-layout proposal: ACCEPTED / STILL DEFERRED TO TASK 8B.4
STOP reason: <none/exact>
Next action: Awaiting ChatGPT audit.
```

# 15. Git gate / commit / push

Allowed repository changes only:

```text
docs/task8b3_d1_demo_failure_forensics.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

Run:

```bat
git status --short
git diff --check
```

Stage individually.

If COMPLETE, commit exactly:

```text
docs(demo): record six-image failure forensics
```

If PARTIAL/STOP/FAILED:

```text
docs(demo): record demo forensics stop
```

Push:

```bat
git push origin eval/task8b3-six-image-demo-suite
```

No force.

# 16. COMPLETE definition

COMPLETE only if:
- no inference ran;
- no delivery artifact changed;
- no source/test/harness changed;
- A1/A3/A4 proposal/reference facts quantified;
- implementation `largest` semantics documented;
- SUCCESS validity gates documented;
- A2 forensics documented without unsupported causality;
- B1/B2 memory signature statically audited;
- defect statuses assigned;
- report/FROM_DSH created;
- commit/push succeed;
- clean tree;
- DSH stops.

# 17. Final response

```text
TASK 8B.3-D1 COMPLETE / PARTIAL / STOP / FAILED

Branch:
eval/task8b3-six-image-demo-suite

Commit:
<sha or NONE>

Push:
PASS / FAIL / NOT POSSIBLE

Inference executed:
NO

Delivery modified:
NO

Product/harness/tests modified:
NO

Reference forensics:
<one-line summary>

A2:
<one-line summary>

B1/B2 memory:
<EXACT_SIGNATURE_MATCH / STRONG_MATCH / NOT_CONFIRMED>

Defects:
RC1-DEMO-REF-01=<status>
RC1-DEMO-MASK-01=<status>
RC1-DEMO-PROP-01=<status>
RC1-DEMO-MEM-01=<status>

Report:
docs/task8b3_d1_demo_failure_forensics.md

Handoff:
handoff/FROM_DSH.md

Output layout:
ACCEPTED / STILL DEFERRED TO TASK 8B.4

STOP reason:
<none or exact reason>

等待 ChatGPT 审核；不得修复缺陷、不得运行 Demo、不得进入 Task 8B.4 或 Task 8C。
```
