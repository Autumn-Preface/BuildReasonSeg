# TO_DSH — Task 8B.3-P1D10-R1: Correct PROP-01 Resolution Logic and Demo-Pool Readiness

> Status: ACTIVE
> Role boundary: ChatGPT decides; DSH executes only.
> Repository: `C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg`
> Branch: `fix/task8b3-prop01-a2-zero-proposals`
> Required starting HEAD: `2edee1af99d89740cf5e75c99b08fe872b0bcef1`

# 0. Audit disposition

P1D10 is NOT YET APPROVED.

Accepted facts:
- no model/test execution;
- no functional modifications;
- `DETECTOR_ADAPTATION_NOT_READY` is supported;
- detector replacement/fine-tuning touches frozen research claims;
- a policy-only path can in principle preserve the frozen architecture;
- bounded P1D1–P1D9 diagnostics did not find a simple repair for A2.

Two conclusions require correction:

1. `A2 source/domain = NOT ESTABLISHED` cannot support the statement
   `A2 is outside the detector's effective domain`.
   Detector failure alone must not define the domain boundary.

2. `artifacts/task6m_yolo_native/images/val` was declared a READY replacement-success pool without establishing:
   - executable Reference→Relation→Target metadata for candidate selection;
   - whether using the detector validation split creates avoidable Demo/model-selection leakage;
   - whether a better pre-existing frozen relation-annotated pool exists.

This R1 repairs only the decision logic and pool-readiness evidence.

# 1. Git gate

Require exactly:

```text
branch = fix/task8b3-prop01-a2-zero-proposals
HEAD = 2edee1af99d89740cf5e75c99b08fe872b0bcef1
```

Allowed initial tracked tree:
- clean; or
- only `M handoff/TO_DSH.md`.

No reset/rebase/stash/clean/merge.

# 2. Strict prohibitions

Do NOT:
- run any detector/model/inference;
- run `predict.py`, `--inspect-proposals`, Qwen, SAM2, D-B1, pytest, check_setup;
- train/fine-tune/export/download;
- modify detector/runtime/config/tests/manifests/checkpoints;
- modify or replace Demo inputs;
- select an actual replacement image;
- inspect detector outputs for any replacement candidate;
- enter REF-01/MASK-01/Task 8B.4/8C;
- modify main;
- force push.

# 3. Allowed repository changes

Only:

```text
docs/task8b3_p1d10_prop01_resolution_decision.md
handoff/FROM_DSH.md
handoff/TO_DSH.md
```

# 4. Correct the A2 domain statement

Retain:

```text
A2 source/domain provenance = NOT ESTABLISHED
A2 exact membership in active train/val/test = NOT ESTABLISHED
A2 repeatedly yields zero proposals under P1D2–P1D9
```

Forbidden conclusion:

```text
A2 is proven outside the detector's effective domain
```

Replace with an evidence-bounded statement:

```text
A2 is a persistent documented failure/stress case whose provenance is unknown;
the current evidence does not establish whether it is in-domain or out-of-domain.
```

The supported-input policy, if chosen, must be defined by positive provenance/asset criteria,
not retrospectively by whether the detector succeeds.

# 5. Audit candidate Demo pools

Read local frozen assets and documentation only.

Audit at least these possible pools if present:

```text
A. artifacts/task6m_yolo_native/images/val
B. artifacts/task6m_yolo_native/images/test
C. frozen BuildSpatialReason validation split
D. frozen BuildSpatialReason test split
E. any pre-existing RC1/Demo candidate pool documented before PROP-01 diagnostics
```

For each record:

```text
exists
frozen before P1D10? YES / NO / NOT ESTABLISHED
image identity available? YES / NO
Reference→Relation→Target metadata available? YES / NO
program labels available? YES / NO
reference/target instance identity available? YES / NO
scene/split identity available? YES / NO
used for detector training? YES / NO
used for detector validation/model selection? YES / NO
used for final research test? YES / NO
candidate selection can be performed without detector output? YES / NO
```

Do not run models.

# 6. Leakage / credibility classification

For each candidate pool choose exactly one:

```text
CLEAN_DEMO_SELECTION_POOL
USABLE_WITH_DISCLOSURE
MODEL_SELECTION_LEAKAGE_RISK
MISSING_RELATION_METADATA
POOL_NOT_AVAILABLE
POOL_STATUS_INCOMPLETE
```

Rules:

- A detector validation split must NOT be called clean merely because it is not training data.
- A final research test split may be `USABLE_WITH_DISCLOSURE` for demonstration examples only if:
  - selection is deterministic and outcome-independent;
  - no new metric/model selection is performed;
  - reuse is explicitly disclosed.
- A pool without executable relation/reference/target metadata cannot be READY for the intended relation Demo.

# 7. Define a valid replacement-selection contract

Only if at least one pool is `CLEAN_DEMO_SELECTION_POOL` or `USABLE_WITH_DISCLOSURE`, define BEFORE selection:

```text
exact pool
exact split/version
eligible program/relation set
required reference/target validity
image-format/size rule
scene-disjoint/no-train rule
deterministic ordering
number of cases k
tie-breaking
whether model/detector outputs may be consulted = NO
whether manual visual quality may be consulted = NO before selection
```

The rule must be executable from metadata alone.

If no such pool exists:

```text
REPLACEMENT_SELECTION_POLICY_NOT_READY
```

Do not select any image in this task.

# 8. Re-evaluate Demo-policy feasibility

Choose exactly one:

```text
DEMO_POLICY_PATH_READY
DEMO_POLICY_PATH_NOT_READY
DEMO_POLICY_STATUS_INCOMPLETE
```

`READY` requires both:
- a support scope that can be defined by positive, pre-existing provenance/asset criteria;
- a replacement selection policy that is executable without detector/manual outcome consultation.

# 9. Re-evaluate primary resolution

Choose exactly one:

```text
PROP01_RESOLUTION_DEMO_POLICY
PROP01_RESOLUTION_DETECTOR_ADAPTATION
PROP01_RESOLUTION_ENGINEERING_FALLBACK_DESIGN
PROP01_RESOLUTION_BLOCKED
```

Guidance:

- Do not choose DEMO_POLICY merely because A2 fails.
- DEMO_POLICY is allowed if a transparent positive support scope and non-cherry-picked Demo selection contract are ready.
- DETECTOR_ADAPTATION remains disallowed unless readiness facts changed without new execution.
- If no honest pool/policy can be made ready, choose BLOCKED.

# 10. Re-evaluate PROP-01 status

Choose exactly one:

```text
PROP01_OPEN_ENGINEERING_DEFECT
PROP01_RECLASSIFIED_SUPPORTED_DOMAIN_FAILURE
PROP01_BLOCKING_RELEASE_DEFECT
```

Rules:

- `PROP01_RECLASSIFIED_SUPPORTED_DOMAIN_FAILURE` is NOT allowed unless evidence establishes that A2 violates an independently defined supported-domain contract.
- Unknown provenance is not proof of domain violation.
- PROP-01 must not be marked CLOSED.

# 11. Correct release wording

Write ≤120 Chinese characters.

It must:
- state the actual verified support scope positively;
- call A2 a documented failure/stress case if appropriate;
- NOT say A2 is "域外" unless independently established;
- NOT imply arbitrary aerial-image robustness;
- NOT imply A2 was fixed.

# 12. Scientific freeze

Reconfirm:

```text
scientific freeze preserved = YES / NO / NOT ESTABLISHED
```

Policy documentation and deterministic Demo-case selection must remain separate from frozen research metrics and architecture.

# 13. Next gate

If resolution = DEMO_POLICY:

```text
NEXT = PROP01_SUPPORTED_DOMAIN_POLICY_IMPLEMENTATION
```

If BLOCKED:

```text
NEXT = PROP01_DECISION_EVIDENCE_RECOVERY
```

If ADAPTATION:

```text
NEXT = PROP01_ADAPTATION_PROTOCOL_DESIGN
```

If FALLBACK:

```text
NEXT = PROP01_FALLBACK_PROPOSAL_DESIGN
```

Do NOT execute it.

# 14. Report repair

Update:

```text
docs/task8b3_p1d10_prop01_resolution_decision.md
```

Append an R1 audit section containing:

1. ChatGPT audit correction
2. corrected A2 domain statement
3. candidate-pool audit table
4. leakage/credibility enum for each pool
5. executable metadata-only selection contract OR NOT READY
6. corrected Demo-policy feasibility enum
7. corrected primary resolution
8. corrected PROP-01 status
9. corrected release-language draft
10. scientific-freeze statement
11. exact next gate
12. no model/test execution
13. no functional modification.

Do not delete historical P1D10 text; mark superseded statements explicitly.

# 15. FROM_DSH

Preserve ARTIFACT-FACTS exactly.

Required:

```text
Task: 8B.3-P1D10-R1
Status: COMPLETE / PARTIAL / STOP / FAILED
Branch: fix/task8b3-prop01-a2-zero-proposals
Starting HEAD: 2edee1af99d89740cf5e75c99b08fe872b0bcef1
Model/test execution: NONE
Functional files modified: NO
A2 domain provenance: NOT ESTABLISHED
A2 domain classification: IN_DOMAIN / OUT_OF_DOMAIN / NOT ESTABLISHED
Best Demo candidate pool: <path/version / NONE>
Best pool classification: <exact enum>
Replacement selection policy: READY / NOT READY
Detector adaptation feasibility: DETECTOR_ADAPTATION_NOT_READY
Demo policy feasibility: <exact enum>
Primary resolution: <exact enum>
PROP-01 status: <exact enum>
Scientific freeze preserved: YES / NO / NOT ESTABLISHED
Next gate: <exact enum>
RC1-DEMO-MEM-01: CLOSED
RC1-DEMO-PROP-01: OPEN / RECLASSIFIED / BLOCKING (not closed)
RC1-DEMO-REF-01: OPEN
RC1-DEMO-MASK-01: OPEN
Report: docs/task8b3_p1d10_prop01_resolution_decision.md
Next action: Awaiting ChatGPT audit; do not execute next gate.
```

# 16. Commit / push

If COMPLETE:

```text
docs(rc1): correct prop01 resolution decision
```

If STOP/FAILED:

```text
docs(rc1): record prop01 decision correction stop
```

Push normally. No force push.

# 17. COMPLETE definition

COMPLETE only if:
- exact starting HEAD;
- no model/test execution;
- no functional modification;
- A2 is not circularly classified as out-of-domain;
- candidate Demo pools are audited for relation metadata and leakage;
- replacement rule is metadata-only and outcome-independent, or explicitly NOT READY;
- primary resolution/status are re-evaluated;
- superseded P1D10 claims are marked;
- report/handoff committed and pushed;
- STOP.
