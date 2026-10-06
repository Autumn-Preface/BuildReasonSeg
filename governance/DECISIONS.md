# DECISIONS.md — Frozen Decision Ledger

A compact ledger of frozen decisions. Each entry records: Decision ID, Status, Decision, Rationale / evidence summary,
What is prohibited, Reopen condition.

## GOV-D001

```text
Decision ID: GOV-D001
Status: FROZEN
Decision: SCIENTIFIC_ARCHITECTURE_FROZEN
Rationale / evidence: the RC1 architecture (Qwen 2B ProgramHead, YOLO26m proposals, deterministic spatial executor,
  SAM/SAM2 visual path, GRF, target-aware segmentation, Reference -> Relation -> Target -> Segmentation) is the frozen
  scientific baseline of the project.
Prohibited: architecture substitution, new model branch, architecture rescue.
Reopen condition: explicit Supervisor decision only.
```

## GOV-D002

```text
Decision ID: GOV-D002
Status: FROZEN
Decision: REJECT_FURTHER_SCALAR_RANK_REPAIR
Rationale / evidence: REF-01 forensics showed the residual left/below selection limitation is not resolvable by further
  scalar rank repair; the design fix was applied only where justified (above).
Prohibited: scalar fitting, new ranking heuristic, locked-case rescue heuristics.
Reopen condition: new evidence approved by the Supervisor.
```

## GOV-D003

```text
Decision ID: GOV-D003
Status: IMPLEMENTED
Decision: LARGEST_EXTENT_DOMINANCE_EXCEPTION_V1
Rationale / evidence: the above-case reference selection defect was repaired by the extent-dominance exception with the
  production ranking unchanged elsewhere.
Prohibited: extending the exception to other cases without Supervisor approval.
Reopen condition: Supervisor decision, or new forensics on the remaining residual cases.
```

## GOV-D004

```text
Decision ID: GOV-D004
Status: FROZEN
Decision: REJECT_THRESHOLD_BASED_MASK_VALIDITY_REPAIR
Rationale / evidence: MASK-01 forensics established the padding gate is ineffective and no threshold for semantic mask
  quality could be justified from the locked artifacts.
Prohibited: post-hoc semantic-quality threshold repair, mask-quality threshold tuning.
Reopen condition: Supervisor decision with new independent evidence.
```

## GOV-D005

```text
Decision ID: GOV-D005
Status: IMPLEMENTED_EXTERNAL_RC1
Decision: SUCCESS_SEMANTICS_HARDENING_V1
Rationale / evidence: SUCCESS now means runtime structural success only; validity_scope = RUNTIME_STRUCTURAL_ONLY,
  semantic_status = NOT_EVALUATED, semantic target correctness is not established. Verified on the external complete
  RC1 (targeted contracts 7/7 PASS, external full suite 130/130 PASS).
Prohibited: presenting SUCCESS as semantic target correctness.
Reopen condition: Supervisor decision.
```

## GOV-D006

```text
Decision ID: GOV-D006
Status: IMPLEMENTED
Decision: INSPECT_IMAGE_PREFLIGHT_BEFORE_MODEL_V1
Rationale / evidence: an inspect-only load_image() preflight before model resolution makes a corrupt-but-valid-extension
  image return E202 / exit 20 deterministically.
Prohibited: changing the inspect error contract, or reordering normal inference.
Reopen condition: Supervisor decision.
```

## GOV-D007

```text
Decision ID: GOV-D007
Status: FROZEN
Decision: LIGHTWEIGHT_CANONICAL_VS_EXTERNAL_COMPLETE_DELIVERY
Rationale / evidence: the canonical source tree is a lightweight source/config snapshot; the complete-delivery full suite
  belongs on the external complete RC1 (canonical tree full-suite failures are missing-asset/fixture/environment causes,
  not MASK-01 regressions).
Decisions: canonical tree carries source/config only; the complete suite gate belongs to external RC1; sync is one-way
  manifest-listed source/config through scripts/sync_advisor_rc1_delivery.py; preserved external weights, assets and log
  fixtures must not be overwritten by sync.
Prohibited: running the complete-suite gate as a canonical-tree acceptance rule; manual edits of external source.
Reopen condition: Supervisor decision.
```

## GOV-D008

```text
Decision ID: GOV-D008
Status: ACTIVE
Decision: EXECUTOR_NEUTRAL_HANDOFF_V1
Rationale / evidence: repository state is the handoff source of truth so Codex or DSH can take over mid-task without the
  previous executor's chat history.
Decisions: Chat remains Supervisor; Codex is the preferred executor; DSH is the fallback executor; repo state is the
  handoff source of truth.
Prohibited: relying on chat history as the only state; pre-authorizing tasks.
Reopen condition: Supervisor decision.
```
