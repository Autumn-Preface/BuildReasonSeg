# TO_DSH — Task 8B.3-P1D11A-R2: Implement Supported-Domain Policy After Manifest Normalization

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `581f91de8145a5b680d510b185b38b6e9878aa1d`

# 0. Frozen prerequisite

P1D11M1 is approved by ChatGPT.

Freeze:

```text
source manifest identity basis =
GIT_CANONICAL_BLOB_BYTES

source manifest entries =
135

post-normalization Git-canonical check =
135/135 PASS

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

primary resolution =
PROP01_RESOLUTION_DEMO_POLICY

A2 provenance/domain =
NOT ESTABLISHED

A2 status =
documented persistent non-detection / stress case

locked candidate status =
FINAL_METADATA_LOCK

locked candidate runtime status =
NOT YET RUN
```

This task implements policy/documentation only.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 581f91de8145a5b680d510b185b38b6e9878aa1d
```

Allowed initial tracked tree:

```text
clean
```

or only:

```text
M handoff/TO_DSH.md
```

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:

- run any detector/model/inference;
- run `predict.py`, `--inspect-proposals`, Qwen, SAM2, D-B1;
- run pytest/check_setup;
- train/fine-tune/export/download;
- modify runtime source;
- modify tests;
- modify configs;
- modify checkpoints/model assets;
- modify `.gitattributes`;
- change Git config;
- run/copy/inspect locked Demo candidates;
- replace any locked candidate;
- modify A1/A2/A3/A4/B1/B2;
- modify external delivery;
- run external write sync;
- mark PROP-01 CLOSED;
- call A2 proven out-of-domain;
- claim arbitrary aerial/cross-city/cross-sensor robustness;
- change Task7J metrics/model/threshold/seed/architecture;
- enter REF-01/MASK-01/Task 8B.4/8C;
- update main;
- force push.

# 3. Allowed tracked changes

Only:

```text
docs/task8b3_p1d11_supported_domain_policy.md
docs/task8b3_p1d11a_policy_implementation.md
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

No other tracked file may change.

# 4. Pre-edit manifest gate

Read:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

Require:

```text
schema = BuildReasonSeg.AdvisorRC1.SourceManifest.v1
identity_basis = GIT_CANONICAL_BLOB_BYTES
manifest entry count = 135
README.md entry count = 1
docs/model_card.md entry count = 1
```

For all 135 entries, compare manifest identity against exact Git canonical bytes from:

```text
git show HEAD:delivery_src/BuildReasonSeg_Advisor_RC1/<path>
```

Binary capture only.

Require:

```text
pre-edit Git-canonical manifest = 135/135 PASS
```

If not:
- do not edit policy/README/model_card;
- record STOP;
- commit/push STOP evidence;
- STOP.

# 5. Create authoritative supported-domain policy

Create:

```text
docs/task8b3_p1d11_supported_domain_policy.md
```

Required content:

## 5.1 Verified research/evaluation domain

State positively:

```text
BuildSpatialReason v0.2
over WHU-EA-NativeVector v1.0
source = WHU Building Dataset — Satellite dataset II (East Asia)
split view = scene_disjoint_v1
modality = RGB optical overhead/aerial imagery
instance concept = WHU native-vector building instances
reasoning = tile-relative spatial reasoning
formal RC1 programs =
  largest -> left_of  -> nearest
  largest -> right_of -> nearest
  largest -> above    -> nearest
  largest -> below    -> nearest
```

State explicitly:

```text
software format/size acceptance != demonstrated generalization domain
scene-disjoint split != proof of cross-city generalization
```

## 5.2 Explicit non-claims

Include:

```text
cross-city generalization = NOT ESTABLISHED
broad geographic generalization = NOT ESTABLISHED
cross-sensor generalization = NOT ESTABLISHED
arbitrary aerial-image robustness = NOT ESTABLISHED
unrestricted natural language = NOT ESTABLISHED
SAR / infrared / raw multispectral = NOT SUPPORTED
```

## 5.3 A2 policy

State:

```text
A2 provenance/domain = NOT ESTABLISHED.
A2 is a fixed historical persistent non-detection / stress case.
A2 remains part of the historical Task 8B.3 six-case diagnostic evidence.
P1D1-P1D9 did not fix it.
A2 must not be called proven out-of-domain.
A2 must not be silently removed or rewritten as a success.
```

Bounded evidence summary only:

```text
active YOLO26 tiled/full-frame -> zero
active YOLO26 diagnostic conf=0.001 -> zero
same-lineage epoch-18 YOLO26 -> zero
independent WHU YOLOv8m -> zero
one predeclared validation-moment affine rescue -> zero
```

Do not assert root cause.

## 5.4 Separate Demo sets

Historical diagnostic suite:

```text
A1/A2/A3/A4/B1/B2
```

Purpose:
- preserve historical defect/audit evidence;
- do not retroactively relabel it 6/6 success.

Locked supported-domain qualitative candidates:

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

State:

```text
source = BuildSpatialReason v0.2 TEST
selection = deterministic metadata-only
selected after Task7J final metrics had already been consumed
no detector/parser/runtime/manual visual outcome used
candidate IDs immutable without a new ChatGPT decision
runtime status = NOT YET RUN
success status = NOT ESTABLISHED
```

## 5.5 Required reuse disclosure

Include verbatim English:

```text
The qualitative Demo candidates are deterministically selected from the frozen BuildSpatialReason v0.2 test split after the Task 7J final frozen-architecture test metrics were already consumed. Their qualitative reuse does not alter, replace, or re-select any reported Task 7J metric, model, threshold, seed, or architecture.
```

Also include faithful Chinese equivalent.

# 6. Update canonical README

Modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
```

Required:

- replace ambiguous wording equivalent to `RGB 光学遥感影像（正式输入域）`;
- distinguish:
  - software-readable input modality/format;
  - verified research/evaluation domain;
- add subsection:

```text
### 已验证数据域与 Demo 边界
```

It must state:
- WHU East Asia;
- BuildSpatialReason v0.2;
- WHU-EA-NativeVector v1.0;
- `scene_disjoint_v1`;
- tile-relative four-relation reasoning;
- scene split is not proof of cross-city generalization;
- arbitrary aerial image success is not guaranteed;
- A2 provenance/domain NOT ESTABLISHED;
- A2 is documented persistent non-detection/stress case;
- A2 is not proven out-of-domain;
- historical six-case suite is preserved;
- four locked v0.2 TEST cases are NOT YET RUN and not claimed successful;
- qualitative test reuse is disclosed.

Do not change:
- commands;
- thresholds;
- algorithms;
- metrics;
- model names.

# 7. Update canonical model card

Modify:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Required:

- clarify verified WHU East Asia / native-vector / scene_disjoint_v1 / tile-relative domain;
- preserve four formal L3 programs;
- add limitations:
  - scene split != cross-city evidence;
  - arbitrary aerial robustness not established;
  - cross-sensor robustness not established;
  - A2 persistent non-detection, provenance NOT ESTABLISHED;
  - A2 not proven out-of-domain;
- add Demo-policy subsection:
  - historical A1–B2 audit retained;
  - four locked TEST candidates metadata-only/deterministic;
  - NOT YET RUN;
  - qualitative reuse disclosed;
  - no Task7J metric/model/threshold/seed/architecture changed.

No frozen metric value may change.

# 8. Stage README/model_card and update manifest using Git-index bytes

Stage ONLY:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/README.md
delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Read exact staged canonical bytes via:

```text
git show :delivery_src/BuildReasonSeg_Advisor_RC1/README.md
git show :delivery_src/BuildReasonSeg_Advisor_RC1/docs/model_card.md
```

Compute:

```text
bytes
sha256
```

from those binary streams.

Update exactly these two manifest entries:

```text
README.md
docs/model_card.md
```

Do not modify any other entry identity.

Do not modify:

```text
identity_basis
identity_basis_note
schema
task
source_delivery
canonical_root
copy_policy
```

Stage:

```text
delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

# 9. Post-edit staged/index manifest gate

Parse the staged manifest from:

```text
git show :delivery_src/BuildReasonSeg_Advisor_RC1/source_manifest.json
```

For every entry:
- if source file is staged, read `git show :<path>`;
- otherwise read `git show HEAD:<path>`.

Require:

```text
post-edit Git-canonical/index manifest = 135/135 PASS
entry count = 135
missing paths = 0
duplicate paths = 0
identity_basis = GIT_CANONICAL_BLOB_BYTES
```

Require exactly:

```text
manifest entry identities changed = 2
changed manifest paths =
README.md
docs/model_card.md
```

Any other entry identity change -> STOP.

# 10. External boundary

External delivery remains untouched:

```text
C:\D\DeepSeekHarness\delivery\BuildReasonSeg_Advisor_RC1
```

Do NOT run write sync.

A read-only comparison is optional, not required.

If performed, record result only.
Do not interpret CRLF byte differences as canonical source drift.

# 11. Update implementation report

Append/update:

```text
docs/task8b3_p1d11a_policy_implementation.md
```

Required sections:

1. prior STOP history
2. P1D11M1 manifest normalization prerequisite
3. starting HEAD
4. pre-edit 135/135 gate
5. policy files changed
6. README summary
7. model card summary
8. A2 wording audit
9. locked-candidate wording audit
10. manifest two-entry update
11. post-edit 135/135 gate
12. external write sync = NOT RUN
13. PROP-01 remains OPEN
14. scientific freeze preserved
15. exact next gate.

# 12. Final status

If COMPLETE:

```text
supported-domain policy =
IMPLEMENTED_IN_CANONICAL_RC1_DOCS

A2 domain =
NOT ESTABLISHED

A2 policy =
DOCUMENTED_PERSISTENT_NON_DETECTION

locked candidate status =
FINAL_METADATA_LOCK

locked candidate runtime status =
NOT YET RUN

PROP-01 =
PROP01_OPEN_ENGINEERING_DEFECT

scientific freeze =
PRESERVED

NEXT =
PROP01_LOCKED_DEMO_PROPOSAL_GATE
```

Do not execute next gate.

# 13. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D11A-R2
Status: COMPLETE / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 581f91de8145a5b680d510b185b38b6e9878aa1d
Model/test execution: NONE
Functional runtime files modified: NO
Tests modified: NO
External delivery modified: NO
Pre-edit Git-canonical manifest: 135/135 PASS / FAIL
Post-edit staged/index manifest: 135/135 PASS / FAIL / NOT RUN
Manifest entry count: 135 / other
Manifest identity basis: GIT_CANONICAL_BLOB_BYTES / other
Manifest entry identities changed: 2 / other
Changed manifest entries: README.md; docs/model_card.md / other
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
External write sync: NOT RUN
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Next action: Awaiting ChatGPT audit; do not run candidates or sync external.
```

# 14. Commit / push

If COMPLETE:

```text
docs(rc1): implement prop01 supported-domain policy
```

If STOP/FAILED:

```text
docs(rc1): record prop01 policy retry stop
```

Push current branch normally.
No force push.

# 15. COMPLETE definition

COMPLETE only if:

- exact starting HEAD;
- no model/test execution;
- no runtime/test/config/checkpoint changes;
- pre-edit manifest 135/135 under Git-canonical basis;
- authoritative policy doc created;
- README/model_card updated with evidence-bounded wording;
- A2 remains NOT ESTABLISHED and not called proven out-of-domain;
- historical six-case audit retained;
- exact four locked candidates retained as NOT YET RUN;
- required test-reuse disclosure present;
- only README/model_card manifest identities changed;
- post-edit staged/index manifest = 135/135;
- external delivery untouched;
- PROP-01 remains OPEN;
- scientific freeze preserved;
- next gate not executed;
- only authorized files committed;
- commit/push succeeds;
- STOP.
