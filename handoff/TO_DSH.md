# TO_DSH — Task 8B.3-P1D10: PROP-01 Resolution Decision — Detector Adaptation vs Demo Policy

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `70e2e2361e9e16cca891138de5d2712318282b2b`

# 0. Purpose

P1D1–P1D9 have exhausted the bounded diagnostic tree for A2 without modifying the product.

Frozen evidence:

```text
active continued YOLO26m-seg:
  tiled conf=0.05     -> zero
  full-frame conf=0.05 -> zero
  tiled conf=0.001    -> zero

same-lineage epoch-18 YOLO26m-seg:
  tiled conf=0.05 -> zero

independent validated WHU YOLOv8m-seg:
  tiled conf=0.05 -> zero

A2 encoding:
  no anomaly found

A2 photometric audit:
  original combined rule -> not a strong photometric outlier
  split robustness -> sensitive to split composition
  low contrast relative to val/test and successful controls

single predeclared validation-moment affine rescue:
  transformed Y_mean ≈ 91.59
  transformed Y_std  ≈ 26.86
  active detector still -> zero
```

Therefore no further ad-hoc threshold, checkpoint, or image-enhancement experiments are authorized.

This task decides the scientifically honest resolution path for `RC1-DEMO-PROP-01`.

It is a READ-ONLY / DOCS-ONLY decision audit.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 70e2e2361e9e16cca891138de5d2712318282b2b
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any model, detector, `predict.py`, `--inspect-proposals`, Qwen, SAM2, D-B1, pytest or setup check;
- modify A2 or any Demo input;
- modify detector/runtime/config/tests/manifests/checkpoints;
- train/fine-tune/download/export;
- implement a fallback proposal source;
- replace A2;
- remove A2;
- adopt the P1D9 transform;
- lower thresholds;
- enter REF-01 or MASK-01;
- enter Task 8B.4 or 8C;
- modify main;
- force push.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d10_prop01_resolution_decision.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 4. Recover Demo-suite policy and provenance

Search the available repository/history for Task 8B.3 / R4B / RC1 Demo documentation.

Establish:

```text
why A1/A2/A3/A4/B1/B2 were selected
whether they were pre-registered as a fixed acceptance suite
whether each case was intended as:
  success case
  stress case
  negative/failure case
whether replacing/removing A2 is currently allowed by an existing rule
whether Demo success was ever defined as 6/6 runtime success
whether manual semantic correctness was an explicit release gate
```

For unsupported facts write:

```text
NOT ESTABLISHED
```

Do not infer intent from filenames alone.

# 5. Separate scientific freeze from RC1 engineering

Using Task 7J and RC1 metadata/docs, establish separately:

```text
A. frozen research architecture/evaluation
B. post-freeze RC1 engineering/demo packaging
```

For each candidate intervention below, classify its effect:

```text
1. retrain/fine-tune current YOLO26 detector
2. replace YOLO26 with another trained detector
3. add a second proposal-source fallback
4. add image preprocessing before detector
5. keep detector unchanged but define an explicit supported-input/domain policy
6. replace A2 in a future Demo success suite while retaining A2 as a documented stress/failure case
```

Use exactly one impact enum per intervention:

```text
TOUCHES_FROZEN_RESEARCH_ARCHITECTURE
ENGINEERING_ONLY_IF_SEPARATELY_VERSIONED_AND_REVALIDATED
POLICY_ONLY_NO_ARCHITECTURE_CHANGE
IMPACT_NOT_ESTABLISHED
```

If an intervention can belong to two categories depending on use, state the conditions explicitly.

# 6. Adaptation feasibility audit

Without training or inference, assess whether detector adaptation is presently executable from available assets.

Record:

```text
A2 ground-truth building masks available? YES / NO / NOT ESTABLISHED
A2-like labeled images available? YES / NO / NOT ESTABLISHED
A2 source/domain provenance available? YES / NO
existing adaptation dataset specification? YES / NO
existing held-out adaptation validation set? YES / NO
existing acceptance metric/threshold for adapted detector? YES / NO
```

A detector adaptation path is `READY` only if there is enough labeled data and a held-out validation protocol to
avoid tuning to A2 alone.

Choose exactly one:

```text
DETECTOR_ADAPTATION_READY
DETECTOR_ADAPTATION_NOT_READY
DETECTOR_ADAPTATION_STATUS_INCOMPLETE
```

# 7. Demo-policy feasibility audit

Determine whether a supported-domain Demo policy can be stated transparently without falsifying claims.

Audit whether existing project scope already limits inputs to:

```text
overhead/aerial building imagery
WHU-like building instance domain
specific image sizes/formats
```

Do not create a new limitation unless supported by existing training/product facts.

Assess whether A2 can honestly be retained as:

```text
documented unsupported/stress failure case
```

without being counted as a successful Demo case.

Also assess whether a future replacement success case could be selected from a predeclared supported-domain pool
without cherry-picking.

Choose exactly one:

```text
DEMO_POLICY_PATH_READY
DEMO_POLICY_PATH_NOT_READY
DEMO_POLICY_STATUS_INCOMPLETE
```

# 8. Anti-cherry-picking requirement

If a future Demo success-case replacement is considered permissible, define BEFORE any replacement is selected:

```text
eligible source pool
selection rule
required relation type
required image constraints
required no-leakage condition
required detector precheck policy
whether detector outcome may be consulted during selection
```

Critical rule:

```text
The replacement case MUST NOT be selected by trying images until one succeeds.
```

A valid future selection rule must be independent of detector outcome.

If no such rule can be grounded in existing assets, record:

```text
REPLACEMENT_SELECTION_POLICY_NOT_READY
```

# 9. PROP-01 resolution options

Evaluate exactly these options:

## Option A — Immediate detector adaptation

Description:
retrain/fine-tune or replace proposal detector now.

## Option B — Engineering fallback proposal source

Description:
add a fallback proposal source for zero-proposal cases while preserving the frozen research implementation as a
separate baseline/version.

## Option C — Supported-domain Demo policy

Description:
keep the frozen detector unchanged, document the supported input domain, retain A2 as a stress/failure case, and
construct any future success suite using a predeclared outcome-independent selection rule.

## Option D — Keep A2 as mandatory 6/6 success and block release

Description:
do not change architecture or policy until the current frozen detector succeeds on A2.

For each option record:

```text
scientific honesty
engineering feasibility now
required new data/code
revalidation burden
risk of overfitting/cherry-picking
effect on Challenge Cup Demo credibility
```

# 10. Decision rule

Choose exactly ONE primary resolution.

## `PROP01_RESOLUTION_DEMO_POLICY`

Use only if:
- detector adaptation is NOT READY;
- current bounded diagnostics give strong evidence A2 is a persistent unsupported/blind-spot case;
- a transparent supported-domain/stress-case policy is feasible;
- no claim requires arbitrary-image robustness.

## `PROP01_RESOLUTION_DETECTOR_ADAPTATION`

Use only if:
- adaptation is READY with labeled data + held-out validation;
- the project requirement genuinely requires A2-like inputs to succeed;
- scientific/versioning impact is explicitly manageable.

## `PROP01_RESOLUTION_ENGINEERING_FALLBACK_DESIGN`

Use only if:
- a fallback proposal source is justified by existing project scope;
- adaptation is not the preferred path;
- a separately versioned engineering fork can be validated without silently rewriting frozen research results.

## `PROP01_RESOLUTION_BLOCKED`

Use if evidence is insufficient to select an honest resolution.

# 11. PROP-01 defect status decision

Choose exactly ONE:

```text
PROP01_OPEN_ENGINEERING_DEFECT
PROP01_RECLASSIFIED_SUPPORTED_DOMAIN_FAILURE
PROP01_BLOCKING_RELEASE_DEFECT
```

Important:
- do NOT mark PROP-01 CLOSED in this task;
- reclassification is not closure;
- implementation/policy documentation still requires a later task.

# 12. Required release-language draft

Write exact proposed user-facing wording for the chosen resolution, max 120 Chinese characters, suitable for README /
Demo documentation.

It must not claim:
- arbitrary aerial-image robustness;
- that A2 was fixed if it was not;
- that frozen scientific results used a new detector if they did not.

# 13. Required next gate

Choose exactly ONE based on the primary resolution:

If `PROP01_RESOLUTION_DEMO_POLICY`:

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION
```

If `PROP01_RESOLUTION_DETECTOR_ADAPTATION`:

```text
NEXT = PROP01_ADAPTATION_PROTOCOL_DESIGN
```

If `PROP01_RESOLUTION_ENGINEERING_FALLBACK_DESIGN`:

```text
NEXT = PROP01_FALLBACK_PROPOSAL_DESIGN
```

If blocked:

```text
NEXT = PROP01_DECISION_EVIDENCE_RECOVERY
```

Do not execute it.

# 14. Report

Create:

```text
docs/task8b3_p1d10_prop01_resolution_decision.md
```

Required sections:

1. scope / starting HEAD
2. frozen P1D1–P1D9 evidence table
3. Demo-suite provenance/policy
4. scientific freeze vs RC1 engineering separation
5. six intervention impact classifications
6. adaptation feasibility
7. Demo-policy feasibility
8. anti-cherry-picking replacement rule readiness
9. four-option comparison
10. exact primary resolution
11. exact PROP-01 defect status
12. rationale
13. release-language draft
14. exact next gate
15. no model execution
16. no functional modification
17. MEM-01 CLOSED
18. REF-01/MASK-01 OPEN untouched.

# 15. FROM_DSH

Preserve ARTIFACT-FACTS exactly.
UTF-8 without BOM.

Required:

```text
Task: 8B.3-P1D10
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 70e2e2361e9e16cca891138de5d2712318282b2b
Model/test execution: NONE
Functional files modified: NO
Demo suite policy provenance: ESTABLISHED / PARTIAL / NOT ESTABLISHED
Detector adaptation feasibility: <enum>
Demo policy feasibility: <enum>
Replacement selection policy: READY / NOT READY / NOT APPLICABLE
Primary resolution: <exact enum>
PROP-01 status: <exact enum>
Scientific freeze preserved: YES / NO / NOT ESTABLISHED
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN / RECLASSIFIED (not closed)
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d10_prop01_resolution_decision.md
Next action: Awaiting ChatGPT audit; do not execute next gate.
```

# 16. Commit / push

If COMPLETE:

```text
docs(rc1): decide prop01 resolution path
```

If STOP/FAILED:

```text
docs(rc1): record prop01 resolution decision stop
```

Push current branch normally.
No force push.

# 17. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model/test execution;
- no functional modification;
- Demo policy/provenance audited;
- scientific vs engineering effects separated;
- adaptation readiness assessed;
- Demo-policy readiness assessed;
- anti-cherry-picking rule readiness assessed;
- four options compared;
- one exact primary resolution chosen;
- one exact PROP-01 status chosen;
- one exact next gate chosen but not executed;
- report/handoff committed and pushed;
- STOP.
