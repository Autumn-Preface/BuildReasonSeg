# Task MASK01_F1_R1_CORRECTIVE_FORENSICS — Corrective Mask Validity Forensics

## 1. Git starting identity

```text
repository          = C:\D\DeepSeekHarness\workspace\project\BuildReasonSeg
starting branch     = audit/task8b3-mask01-validity-forensics
starting head       = 590d080648ef5ae1fc6891dd18b97972c643c3cc
report base head    = 590d080648ef5ae1fc6891dd18b97972c643c3cc
task branch         = audit/task8b3-mask01-f1-r1-corrective-forensics
final commit sha    = POST_COMMIT_EXTERNAL_FACT (recorded outside this commit by rule)
```

## 2. Why F1 was rejected

```text
previous_f1_disposition = REJECTED_INCOMPLETE_FORENSICS
F1 did not reconstruct the predict_one post-inference SUCCESS contract, produced no three-way padding-gate conclusion,
and treated known Task 8B.3 diagnostic paths as a generic search result rather than as primary historical evidence.
```

## 3. Exact predict_one SUCCESS contract

Verified literally in `buildreasonseg/runtime/pipeline.py` at the required starting HEAD:

```text
mask_full, map_padding = context_to_global( -> lines [295]
if not mask_full.any() -> lines [297]
_non_padding_mask( -> lines [300, 354]
mask_only_in_padding -> lines [302]
status="SUCCESS" -> lines [164, 334]
frozen sequence fully present = True
```

## 4. Upstream and post-inference failure paths

```text
empty target mask is rejected by the `if not mask_full.any()` gate before any SUCCESS assignment;
`mask_only_in_padding` is the second branch and is unreachable as an independent condition under the current
`context_to_global()` semantics, as established in section 6.
```

## 5. min_mask_* exact search result

```text
occurrences = 0
[]
```

## 6. Padding semantics proof

```text
helper references padding = False
helper builds an all-True mask of original image size = True
padding_gate_conclusion = PADDING_GATE_INEFFECTIVE
padding_leakage_established = false
```

```python
def _non_padding_mask(loaded, padding: dict) -> np.ndarray:
    mask = np.ones((loaded.height, loaded.width), dtype=bool)
    return mask
```

## 7. Test coverage matrix

| file | tests | padding | SUCCESS |
|---|---|---|---|
| `delivery_src\BuildReasonSeg_Advisor_RC1\tests\conftest.py` | 0 | False | False |
| `delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_cli_contract.py` | 16 | False | False |
| `delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_paths_and_package.py` | 8 | False | False |
| `delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_setup_checker.py` | 12 | False | False |
| `delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b1_fallback_ux.py` | 18 | False | False |
| `delivery_src\BuildReasonSeg_Advisor_RC1\tests\test_task8b_runtime.py` | 40 | True | False |

## 8. Known 8B.3 external artifact inspection

| path | exists | status field | context_padding | final-mask material |
|---|---|---|---|---|
| `evaluation/task8b3_ref01_e3c0_locked_replay_artifact_audit.json` | True | REF01_LOCKED_REPLAY_READINESS_ESTABLISHED | False | False |
| `evaluation/task8b3_ref01_locked_reference_forensics.json` | True | REF01_LOCKED_DEMO_REFERENCE_FORENSICS_COMPLETE | False | False |
| `evaluation/task8b3_ref01_e3c1_locked_record_replay.json` | True | REF01_ELIGIBILITY_REPAIR_LOCKED_RECORD_REPLAY_PASS | False | True |

## 9. Additional bounded Task-8B.3 search

```text
Stage A versioned Task-8B.3 files = 51
Stage B files mentioning mask/bitmap material = 18
classification = SAVED_FINAL_MASK_MATERIAL_REFERENCED
```

## 10. Final-mask artifact classification

```text
SAVED_FINAL_MASK_MATERIAL_REFERENCED
A `SUCCESS` string or the presence of proposals does not constitute saved final-mask material; no saved final-mask
artifact (npy/npz) is referenced by any versioned Task-8B.3 file.
```

## 11. Failure taxonomy

```text
pre-SUCCESS gates: empty target mask (mask_full.any()), mask_only_in_padding branch
post-inference: SUCCESS payload plus PipelineResult(status="SUCCESS")
no failure path was exercised or modified by this task.
```

## 12. Observable-signal inventory

```text
mask_full.any() · mask_only_in_padding · _non_padding_mask · PipelineResult(status="SUCCESS")
```

## 13. Runtime validity vs semantic correctness

```text
The audited chain establishes runtime validity only. Nothing in it asserts that an admitted mask is semantically the
queried reference, so a SUCCESS status must not be read as semantic correctness.
```

## 14. Evidence gaps

```text
no min_mask_* setting exists in the audited canonical sources/configs
```

## 15. R1 disposition

```text
task_status              = COMPLETE_WITH_EVIDENCE_GAPS
padding_gate_conclusion  = PADDING_GATE_INEFFECTIVE
padding_leakage_established = false
repair_implemented       = false
product_source_modified  = false
inference_executed       = false
training_executed        = false
repair_decision          = DEFER_TO_CHATGPT
next_gate                = CHATGPT_MASK01_F1_R1_REVIEW
```
