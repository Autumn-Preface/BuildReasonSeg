# TO_DSH — Task 8B.3-P1D11A: Implement PROP-01 Supported-Domain Policy in Canonical RC1 Docs

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `d414c33762968ac4e6ea441f334082fc43adc0d0`

# 0. Frozen decision

P1D10-R3 is approved by ChatGPT.

Freeze:

```text
raster outcome =
PROP01_LOCKED_DEMO_RASTERS_RESOLVED

replacement selection policy =
READY

locked candidate status =
FINAL_METADATA_LOCK

Demo policy feasibility =
DEMO_POLICY_PATH_READY

primary resolution =
PROP01_RESOLUTION_DEMO_POLICY

PROP-01 status =
PROP01_OPEN_ENGINEERING_DEFECT

A2 provenance/domain =
NOT ESTABLISHED
```

The four immutable locked candidates are:

```text
right =
buildsr_test_1010_3_largest_to_right_of_to_nearest_df818125cf91

left =
buildsr_test_1003_3_largest_to_left_of_to_nearest_f3fcb14e14c3

above =
buildsr_test_1008_3_largest_to_above_to_nearest_5e191d7ac314

below =
buildsr_test_1009_3_largest_to_below_to_nearest_bd900ccef450
```

No locked candidate has yet been run as a new RC1 Demo acceptance case.

This task implements policy/documentation only. It does NOT close PROP-01.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = d414c33762968ac4e6ea441f334082fc43adc0d0
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any detector/model/inference;
- run `predict.py`, `--inspect-proposals`, Qwen, SAM2, D-B1;
- run pytest/check_setup;
- train/fine-tune/export/download;
- copy any locked raster into RC1;
- run or inspect locked candidates;
- replace any locked candidate;
- modify A1/A2/A3/A4/B1/B2;
- modify runtime code, tests, configs, checkpoints or model metadata;
- modify external delivery;
- run `sync_advisor_rc1_delivery.py` without `--check`;
- mark PROP-01 CLOSED;
- claim A2 is proven out-of-domain;
- claim the four locked candidates succeed;
- claim arbitrary aerial-image, cross-city or cross-sensor robustness;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

# 3. Allowed tracked changes

Only:

```text
docs/task8b3_p1d11_supported_domain_policy.md
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other tracked file may change.

# 4. Canonical source-manifest preflight

Before editing canonical RC1 files, read:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
manifest entry count = 135
README.md entry exists exactly once
docs/model_card.md entry exists exactly once
```

Perform a read-only full canonical manifest self-check:

For every one of the 135 manifest entries:

```text
canonical file exists
actual bytes == manifest bytes
actual SHA256 == manifest SHA256
```

Record:

```text
pre-edit canonical manifest check = 135/135 PASS
```

If not exactly 135/135:
- do NOT edit README/model_card/source_manifest;
- update report/handoffs only with STOP evidence;
- commit using STOP message;
- push;
- STOP.

Do not “repair” unrelated stale manifest entries in this task.

# 5. Authoritative supported-domain policy

Create:

```text
docs/task8b3_p1d11_supported_domain_policy.md
```

The document must state the following as authoritative policy.

## 5.1 Positively verified scope

```text
Research/evaluation domain:
BuildSpatialReason v0.2
over WHU-EA-NativeVector v1.0
source: WHU Building Dataset — Satellite dataset II (East Asia)
split view: scene_disjoint_v1

image modality:
RGB optical overhead/aerial imagery

instance concept:
building instances represented by WHU native-vector annotations

reasoning scope:
tile-relative spatial reasoning

formal RC1 semantics:
largest -> left_of  -> nearest
largest -> right_of -> nearest
largest -> above    -> nearest
largest -> below    -> nearest
```

Clarify:

```text
software input-format/size support != demonstrated generalization domain
```

The runtime may read PNG/JPEG/TIFF and tile large images, but this does not establish broad geographic/sensor robustness.

## 5.2 Explicit non-claims

State all:

```text
cross-city generalization = NOT ESTABLISHED
broad geographic generalization = NOT ESTABLISHED
cross-sensor generalization = NOT ESTABLISHED
arbitrary aerial-image robustness = NOT ESTABLISHED
SAR / infrared / raw multispectral = NOT SUPPORTED
unrestricted natural language = NOT ESTABLISHED
```

Also preserve the v0.2 limitation:

```text
scene-disjoint separation is scene separation, not proof of cross-city generalization
```

## 5.3 A2 policy

State exactly in meaning:

```text
A2 provenance/domain = NOT ESTABLISHED.
A2 is a fixed, documented persistent non-detection / stress case.
It remains part of the historical Task 8B.3 six-case diagnostic evidence.
It was not fixed by P1D1-P1D9.
It must not be described as proven out-of-domain.
It must not be silently removed or rewritten as a success.
```

Summarize bounded evidence without overclaiming:

```text
active YOLO26 tiled/full-frame -> zero
active YOLO26 at diagnostic conf=0.001 -> zero
same-lineage epoch-18 YOLO26 -> zero
independent WHU YOLOv8m baseline -> zero
one predeclared validation-moment photometric rescue -> zero
```

Do NOT infer cause beyond that evidence.

## 5.4 Two distinct Demo sets

Define them separately.

### Historical diagnostic suite

```text
A1/A2/A3/A4/B1/B2
```

Purpose:
- historical fixed defect/audit evidence;
- preserve failures and manual findings;
- not retroactively relabeled as a 6/6 success suite.

### Locked supported-domain qualitative candidates

Source:

```text
BuildSpatialReason v0.2 TEST
```

Selection:
- deterministic;
- metadata-only;
- selected after Task 7J final metrics had already been consumed;
- no detector/parser/runtime/manual visual result was consulted;
- reuse is disclosed;
- no research metric is changed.

List the exact four immutable sample IDs from §0.

State:

```text
These are locked candidates, NOT yet demonstrated successes.
Future failure does not permit replacement without a new ChatGPT decision explicitly recording that failed lock.
```

## 5.5 Test-reuse disclosure

Include verbatim in English:

```text
The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.
```

And a faithful Chinese equivalent.

# 6. Update canonical RC1 README

Modify only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
```

Required changes:

1. Under `## 输入`, replace the ambiguous phrase:

```text
RGB 光学遥感影像（正式输入域）
```

with wording that distinguishes:
- accepted software input modality;
- verified research/evaluation domain.

2. Add a compact subsection:

```text
### 已验证数据域与 Demo 边界
```

It must state:
- WHU East Asia / BuildSpatialReason v0.2 / WHU-EA-NativeVector v1.0 / scene_disjoint_v1;
- scene-disjoint does not establish cross-city generalization;
- arbitrary aerial images are not guaranteed;
- A2 is a documented persistent non-detection case with unknown provenance;
- A2 is not claimed out-of-domain;
- historical six-case diagnostic suite is not rewritten;
- four locked v0.2 TEST cases are qualitative candidates only and have not yet been run as a new acceptance Demo;
- deterministic test reuse is disclosed.

3. Do not alter runtime commands, algorithms, thresholds, model names or metrics.

# 7. Update canonical RC1 model card

Modify only:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Required changes:

1. Expand `## 2. 数据` so the verified domain is explicit:
   - WHU East Asia;
   - native-vector;
   - `scene_disjoint_v1`;
   - tile-relative;
   - four formal L3 programs.

2. In known limitations add:
   - scene split != cross-city evidence;
   - arbitrary aerial/cross-sensor robustness not established;
   - A2 is a persistent non-detection case with provenance NOT ESTABLISHED;
   - do not classify A2 as proven out-of-domain.

3. Add a short `Demo policy` subsection:
   - historical A1–B2 diagnostic suite retained;
   - locked candidates come from v0.2 TEST by metadata-only deterministic rule;
   - qualitative reuse disclosed;
   - no Task7J metric/model/threshold/seed/architecture changed;
   - locked candidates are not yet claimed successful.

4. Do not change any frozen metric value.

# 8. Update source_manifest.json narrowly

After README/model_card edits:

Update exactly these two manifest entries:

```text
README.md
docs/model_card.md
```

Set their:

```text
bytes = actual post-edit file bytes
sha256 = actual post-edit SHA256
```

Do not:
- add/remove manifest paths;
- change entry count;
- rewrite unrelated entries;
- change schema/task/copy_policy metadata.

Then perform the full canonical self-check again.

Require:

```text
post-edit canonical manifest check = 135/135 PASS
manifest entry count = 135
```

If any non-target manifest entry changes or fails:
- STOP;
- do not sync external.

# 9. External delivery boundary

External delivery must remain untouched:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Do NOT run write sync.

A read-only comparison after canonical edits is allowed:

```text
python scripts/sync_advisor_rc1_delivery.py
  --destination C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
  --check
```

If run, expected result after canonical doc edits is exactly two content mismatches:

```text
README.md
docs/model_card.md
```

because external sync is intentionally deferred.

If result differs from exactly those two mismatches:
- record it;
- STOP before any external change.

Do not treat expected two-doc mismatch as failure of this canonical-only task.

# 10. Policy status after implementation

If COMPLETE, record:

```text
supported-domain policy = IMPLEMENTED_IN_CANONICAL_RC1_DOCS
A2 status = DOCUMENTED_PERSISTENT_NON_DETECTION
A2 domain = NOT ESTABLISHED
locked candidate status = FINAL_METADATA_LOCK
locked candidate runtime status = NOT YET RUN
PROP-01 status = PROP01_OPEN_ENGINEERING_DEFECT
scientific freeze preserved = YES
```

Do NOT close PROP-01.

# 11. Next gate

If COMPLETE:

```text
NEXT = PROP01_LOCKED_DEMO_PROPOSAL_GATE
```

Purpose of the future gate:
- copy/use the already locked four rasters under a controlled diagnostic location;
- run the frozen active detector only;
- test whether each locked supported-domain candidate produces usable proposals;
- candidate replacement remains forbidden.

Do NOT execute that gate now.

If STOP:

```text
NEXT = PROP01_POLICY_IMPLEMENTATION_RECOVERY
```

# 12. Report

Create:

```text
docs/task8b3_p1d11_supported_domain_policy.md
```

Required audit appendix:

```text
Task: 8B.3-P1D11A
Starting HEAD
pre-edit manifest check
files changed
post-edit manifest check
external write sync = NOT RUN
external read-only check result if run
policy status
A2 status/domain
locked candidate status/runtime status
PROP-01 status
scientific freeze
next gate
```

# 13. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11A
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: d414c33762968ac4e6ea441f334082fc43adc0d0
Model/test execution: NONE
Functional runtime files modified: NO
External delivery modified: NO
Pre-edit canonical manifest: 135/135 PASS / FAIL
Post-edit canonical manifest: 135/135 PASS / FAIL / NOT RUN
Manifest entry count: 135 / other
Canonical policy doc: docs/task8b3_p1d11_supported_domain_policy.md
Canonical README policy: UPDATED / NOT UPDATED
Canonical model card policy: UPDATED / NOT UPDATED
A2 domain classification: NOT ESTABLISHED
A2 policy status: DOCUMENTED_PERSISTENT_NON_DETECTION
Locked candidate status: FINAL_METADATA_LOCK
Locked candidate runtime status: NOT YET RUN
Supported-domain policy: IMPLEMENTED_IN_CANONICAL_RC1_DOCS / NOT IMPLEMENTED
PROP-01 status: PROP01_OPEN_ENGINEERING_DEFECT
Scientific freeze preserved: YES / NO
External read-only sync check: <result / NOT RUN>
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Next action: Awaiting ChatGPT audit; do not run locked candidates or sync external.
```

# 14. Commit / push

If COMPLETE:

```text
docs(rc1): implement prop01 supported-domain policy
```

If STOP/FAILED:

```text
docs(rc1): record prop01 policy implementation stop
```

Push current branch normally.
No force push.

# 15. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model/test execution;
- no functional runtime modification;
- no external write;
- pre-edit manifest self-check = 135/135;
- authoritative policy doc created;
- README and model_card updated with evidence-bounded wording;
- A2 remains NOT ESTABLISHED and is not called proven out-of-domain;
- historical six-case audit preserved;
- four exact locks documented as not-yet-run candidates;
- test reuse disclosure present;
- only two manifest entries updated;
- post-edit manifest self-check = 135/135;
- PROP-01 remains OPEN;
- next gate recorded but not executed;
- commit/push succeeds;
- STOP.
