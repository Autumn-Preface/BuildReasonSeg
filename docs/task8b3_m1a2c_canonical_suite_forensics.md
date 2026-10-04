# Task 8B.3-M1A.2C — Canonical Suite Forensics (frozen classification)

Read-only documentation task. **No pytest, no `check_setup.py`, no `predict.py`, no model loading** was performed,
and the canonical product, `tests/`, `source_manifest.json` and the external delivery were **not** modified.

## 1. Evidence source

`delivery_src/BuildReasonSeg_Advisor_RC1/.pytest_cache/v/cache/lastfailed` from the single M1A.2C canonical run
(`ENV_PYTHON -m pytest tests -q` → 17 failed / 93 passed / 6 errors). All **23** failed/error nodes were recovered and
are listed below; each was classified by reading its test source only (Task 8B.3-M1A.2C-D1).

Class definitions:

* **A** = canonical snapshot deliberately lacks delivery assets (weights, downloaded assets, generated fixtures);
* **B** = fixture/assertion assumes the canonical ROOT is a complete delivery (copies ROOT, runs its
  `check_setup.py`, or reads delivery-side log/fixture files);
* **C** = compact-runtime regression candidate;
* **D** = other code-regression candidate;
* **E** = insufficient evidence.

## 2. Frozen classification of all 23 nodes

| # | node | class | reason |
|---|---|---|---|
| 1 | `tests/test_model_package.py::test_decoder_hash_and_bytes_exact` | A | canonical deliberately lacks delivery assets |
| 2 | `tests/test_model_package.py::test_detector_hash_and_bytes_exact` | A | canonical deliberately lacks delivery assets |
| 3 | `tests/test_model_package.py::test_hash_mismatch_raises` | A | canonical deliberately lacks delivery assets |
| 4 | `tests/test_model_package.py::test_metadata_hashes_match_copied_assets` | A | canonical deliberately lacks delivery assets |
| 5 | `tests/test_model_package.py::test_model_fallback_requires_user_confirmation` | A | canonical deliberately lacks delivery assets |
| 6 | `tests/test_model_package.py::test_model_yaml_parses` | A | canonical deliberately lacks delivery assets |
| 7 | `tests/test_model_package.py::test_package_verification_matches` | A | canonical deliberately lacks delivery assets |
| 8 | `tests/test_model_package.py::test_components_complete` | A | canonical deliberately lacks delivery assets |
| 9 | `tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt` | A | canonical deliberately lacks delivery assets |
| 10 | `tests/test_language_contract.py::test_normal_path_is_qwen_first` | A | canonical deliberately lacks delivery assets |
| 11 | `tests/test_setup_checker.py::test_good_fixture_is_ready` | B | fixture assumes the canonical ROOT is a complete delivery |
| 12 | `tests/test_setup_checker.py::test_hash_mismatch_nonzero` | B | fixture assumes the canonical ROOT is a complete delivery |
| 13 | `tests/test_setup_checker.py::test_invalid_model_yaml_nonzero` | B | fixture assumes the canonical ROOT is a complete delivery |
| 14 | `tests/test_setup_checker.py::test_missing_decoder_nonzero` | B | fixture assumes the canonical ROOT is a complete delivery |
| 15 | `tests/test_setup_checker.py::test_missing_qwen_component_nonzero` | B | fixture assumes the canonical ROOT is a complete delivery |
| 16 | `tests/test_setup_checker.py::test_missing_sam2_nonzero` | B | fixture assumes the canonical ROOT is a complete delivery |
| 17 | `tests/test_setup_checker.py::test_real_project_check_reports_ready` | B | fixture assumes the canonical ROOT is a complete delivery |
| 18 | `tests/test_setup_checker.py::test_real_runtime_reports_ultralytics_and_ready` | B | fixture assumes the canonical ROOT is a complete delivery |
| 19 | `tests/test_paths_and_package.py::test_moving_root_keeps_config_and_model_paths` | B | fixture assumes the canonical ROOT is a complete delivery |
| 20 | `tests/test_paths_and_package.py::test_required_structure` | B | fixture assumes the canonical ROOT is a complete delivery |
| 21 | `tests/test_task8b1_fallback_ux.py::test_fixture_frozen_and_complete` | B | fixture assumes the canonical ROOT is a complete delivery |
| 22 | `tests/test_task8b1_fallback_ux.py::test_no_prompt_to_program_table_in_delivery_code` | B | fixture assumes the canonical ROOT is a complete delivery |
| 23 | `tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim` | B | fixture assumes the canonical ROOT is a complete delivery |

## 3. Class counts (frozen)

```text
A = 10
B = 13
C = 0
D = 0
E = 0
total = 23
```

### 3.1 Reclassification of the six former class-D nodes (Task 8B.3-M1A.2C-D1.1)

| former class-D node | frozen class | reason |
|---|---|---|
| `tests/test_model_package.py::test_components_complete` | **A** | requires the decoder/detector weights and the SAM2/Qwen component assets, which the canonical snapshot deliberately does not track |
| `tests/test_cli_contract.py::test_predict_inspect_proposals_does_not_require_prompt` | **A** | asserts the real detector path returns E202 for an unreadable fixture; without the detector weight the canonical tree cannot reach that stage |
| `tests/test_language_contract.py::test_normal_path_is_qwen_first` | **A** | asserts the Qwen front end reports itself available, which needs the Qwen assets present |
| `tests/test_task8b1_fallback_ux.py::test_fixture_frozen_and_complete` | **B** | reads the frozen fallback-UX fixture that lives in the delivery log tree, which is excluded from the canonical snapshot |
| `tests/test_task8b1_fallback_ux.py::test_no_prompt_to_program_table_in_delivery_code` | **B** | scans delivery code and log/fixture paths that exist only in the complete external delivery |
| `tests/test_task8b1_fallback_ux.py::test_task8b_paraphrases_are_verbatim` | **B** | compares against `logs/task8b_gates.json` from the external delivery, which the canonical snapshot excludes |

## 4. Compact-proposal suite status

```text
tests/test_task8b_runtime.py failing nodes: NONE
```

The compact-proposal contract suite has **no** failing node in the M1A.2C cache; its own single dedicated run passed
(32 passed) and the canonical manifest was verified 135/135 in Task 8B.3-M1A.2B-R3.

## 5. Frozen gate conclusion

```text
CANONICAL_FULL_SUITE_INVALID_AS_DELIVERY_GATE
```

Rationale: every one of the 23 nodes is class A or class B, i.e. it cannot pass inside the canonical snapshot because
that snapshot intentionally tracks only the 135 lightweight source/config files and excludes model weights,
downloaded assets and delivery-side generated fixtures/logs. Classes C, D and E are empty, so no code-regression
candidate remains and the canonical full suite cannot be used as a delivery-readiness gate; delivery-readiness
validation belongs to the external runnable RC1 tree.

## 6. Scope statement

No pytest, `check_setup.py`, `predict.py`, six-image Demo or model execution occurred in this task; canonical
product code, `tests/`, `source_manifest.json` and the external delivery were not modified.
