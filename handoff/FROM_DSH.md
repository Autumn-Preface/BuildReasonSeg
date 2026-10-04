<!-- ARTIFACT-FACTS:BEGIN -->
dataset_version: v0.1.1
total_samples: 25229
split_train: 15592
split_val: 3884
split_test: 5753
level_1: 17275
level_2: 5036
level_3: 2918
level2_type_a: 2275
level2_type_b: 2761
level3_trivial: 1256
level3_nontrivial: 1662
semantic_policy_version: "1.0"
generator_version: v0.1.1
quality_json_path: evaluation/build_spatial_reason_v0.1.1_quality.json
sample_pack_path: evaluation/build_spatial_reason_v0.1.1_samples
<!-- ARTIFACT-FACTS:END -->

# FROM_DSH — Task 8B.3-P1D11B Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D11B` |
| Status | **STOP** (pre-sync external gate differs from expectation) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `f9bf6be13035445f45f956c2b6a5a64ccf7e6032` |
| Canonical manifest gate | 135/135 PASS against Git canonical bytes |
| Pre-sync external check | checked 135 · match 129 · missing 0 · mismatch 6 (expected 133/0/2) |
| Expected mismatches | `README.md`, `docs/model_card.md` |
| Unexpected mismatches | `buildreasonseg/runtime/core.py`, `buildreasonseg/runtime/detector.py`, `buildreasonseg/runtime/outputs.py`, `tests/test_task8b_runtime.py` |
| Root cause of the four extra mismatches | external copies hold Git canonical (LF) content while the canonical working tree holds CRLF-expanded content; the helper compares raw working-tree bytes |
| Controlled 135-file sync | NOT RUN (0 invocations) |
| External source_manifest copy | NOT RUN |
| External setup checker | NOT RUN |
| External delivery modified | NO (stopped before any write) |
| pytest / model inference / locked-candidate runs | NONE / NONE / NONE |
| Frozen metrics / architecture / locked candidates | UNCHANGED |
| PROP-01 status | OPEN (not closed) |
| Report | `docs/task8b3_p1d11b_supported_domain_policy_external_sync.md` |
| STOP reason | mandated pre-sync expectation is 133 match / 2 mismatch, measured 129 / 6; the task book forbids syncing when the pre-sync result differs |
| Next action | Awaiting ChatGPT decision on the comparison semantics (CRLF-insensitive helper vs working-tree normalization) before the policy sync is retried |

Watt was not needed for Task 8B.3-P1D11B (no downloads, no transfers).

No external delivery file was written and no canonical file was changed except this report and handoff.
