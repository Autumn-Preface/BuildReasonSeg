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

# FROM_DSH — Task 8B.3-P1D11S1-R1 Report

_This file holds the current engineering handoff. Research-task reports (Task 7J and earlier) are preserved in git
history._

| item | value |
|---|---|
| Task | `8B.3-P1D11S1-R1` |
| Status | **PARTIAL / STOP** (Git-canonical sync helper + dedicated tests) |
| Branch | `fix/task8b3-prop01-a2-zero-proposals` |
| Starting HEAD | `5842efd69d7066932b306e342baa81add06e626b` |
| Helper | `scripts/sync_advisor_rc1_delivery.py` aligned to `GIT_CANONICAL_BLOB_BYTES` (legacy basis preserved) |
| Dedicated test file | `tests/test_sync_advisor_rc1_delivery.py` (legacy mode · CRLF-independence · CRLF destination rejection · real manifest via Git canonical bytes) |
| Dedicated pytest | exit 1 · 23 passed · failed nodes: ['tests/test_sync_advisor_rc1_delivery.py::test_every_real_manifest_entry_matches_canonical_file', 'tests/test_sync_advisor_rc1_delivery.py::test_git_canonical_mode_ignores_crlf_worktree', 'tests/test_sync_advisor_rc1_delivery.py::test_git_canonical_check_rejects_crlf_destination'] |
| Read-only external check | exit None · NOT RUN · mismatches NOT RUN |
| Canonical RC1 / external RC1 written | NO / NO |
| Model inference / locked candidate runs | NONE / NONE |
| Frozen metrics / architecture / locked candidates | UNCHANGED |
| PROP-01 status | OPEN (not closed) |
| Outcome | **RC1_SYNC_HELPER_ALIGNMENT_BLOCKED** |
| Next gate (recommended, not executed) | `NEXT = RC1_SYNC_HELPER_ALIGNMENT_RECOVERY` |
| Report | `docs/task8b3_p1d11s1_git_canonical_sync_helper.md` |
| Next action | Awaiting ChatGPT audit; the retry sync needs its own task book |

Watt was not needed for Task 8B.3-P1D11S1-R1 (no downloads, no transfers).

Only the sync helper, its dedicated test file, this report and the handoff changed.
