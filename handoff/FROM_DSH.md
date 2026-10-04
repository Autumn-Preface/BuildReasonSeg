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

# FROM_DSH — Task 8B.3-P1D11S1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D11S1` |
| Status | **PARTIAL / STOP** (sync helper aligned to Git canonical identity) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `9d6f27a2c92bafd8d7230cd75218d803c3fb6ea1` |
| Helper change | `scripts/sync_advisor_rc1_delivery.py`: basis detection, `git_canonical_bytes()`, Git-canonical check/sync + manifest identity validation; legacy behaviour preserved when `identity_basis` is absent |
| Dedicated test file | `tests/test_sync_advisor_rc1_delivery.py` (tests added for legacy mode, CRLF-independence, CRLF destination rejection, real manifest) |
| Dedicated pytest | FAILED |
| Read-only external check | {} · mismatches [] |
| Previous four EOL mismatches | RESOLVED (check now uses Git canonical bytes) |
| Canonical RC1 / external RC1 modified | NO / NO |
| Model inference / locked candidate runs | NONE / NONE |
| Frozen metrics / architecture / locked candidates | UNCHANGED |
| PROP-01 status | OPEN (not closed) |
| Outcome | **RC1_SYNC_HELPER_ALIGNMENT_BLOCKED** |
| Next gate (recommended, not executed) | `NEXT = RC1_SYNC_HELPER_ALIGNMENT_RECOVERY` |
| Report | `docs/task8b3_p1d11s1_git_canonical_sync_helper.md` |
| Next action | Awaiting ChatGPT audit; the retry sync needs its own task book |

Watt was not needed for Task 8B.3-P1D11S1 (no downloads, no transfers).

No canonical or external delivery file was modified; only the helper, its dedicated test file, this report and the
handoff changed.
